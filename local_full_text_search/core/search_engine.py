from __future__ import annotations

import copy
import hashlib
import logging
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import (
    CancelledError,
    IndexNotReadyError,
    OperationCancelled,
    SearchTimeout,
)
from local_full_text_search.core.foreground_activity import search_active
from local_full_text_search.core.search_index_coordinator import (
    get_search_index_coordinator,
)
from local_full_text_search.core.normalizer import (
    contains_cjk,
    count_hits,
    make_context,
    normalize_text,
    parse_terms,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.services.logging_service import log_event, log_exception
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.models.search_result import SearchHit, SearchResult

logger = logging.getLogger(__name__)

# Context is presentation data, not the searchable source of truth.  Keep a
# hard per-block budget so one pathological log/worksheet cannot monopolise a
# search worker or allocate an offsets array proportional to the whole block.
MAX_CONTEXT_CHARS = 256 * 1024
MAX_SEARCH_SECONDS = 120.0


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@dataclass(slots=True)
class SearchPage:
    results: list[SearchResult]
    total_candidates: int
    total_confirmed: int
    elapsed_ms: int
    page: int
    page_size: int
    available_results: int
    truncated: bool
    partial: bool = False


class SearchEngine:
    def __init__(
        self,
        db: DatabaseManager,
        *,
        generation_isolated: bool = False,
        snapshot_generation: int | None = None,
    ) -> None:
        self.db = db
        self.generation_isolated = bool(generation_isolated)
        self.snapshot_generation = snapshot_generation
        self.compact_snapshot = self._detect_compact_snapshot()
        self._trigram_tables: dict[str, bool] = {}
        self._snapshot_connection: sqlite3.Connection | None = None

    def search(
        self,
        query: SearchQuery,
        cancel_token: CancelToken | None = None,
        *,
        request_id: str | None = None,
        status_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        partial_callback: Callable[[SearchPage], None] | None = None,
    ) -> SearchPage:
        """Search one committed read snapshot.

        Keeping one read-only connection for the complete query prevents the
        metadata and content phases from observing different index commits.
        In legacy mode the foreground search lease prevents a shared writer
        from publishing during the read.  In generation-isolated mode the
        selected immutable snapshot has no writer to pause.
        """

        # SearchEngine 不创建线程。调用它的 SearchWorker 已经运行在搜索 QThread；
        # 这里专注于协调读取快照、执行算法并返回 SearchPage。
        lease = None
        request_id = str(request_id or f"search-{uuid.uuid4().hex}")
        query_hash = hashlib.sha256(query.text.encode("utf-8")).hexdigest()
        token = cancel_token or CancelToken()
        timeout_timer: threading.Timer | None = None
        log_event(
            "search.requested",
            component="search_engine",
            request_id=request_id,
            query_hash=query_hash,
            query_length=len(query.text),
            query_mode=query.mode,
        )
        try:
            if self.generation_isolated:
                # The active generation is immutable and the indexer writes a
                # different candidate file. There is no shared SQLite writer
                # to pause and no search lease should throttle indexing.
                # 中文理解：搜索读 gen-N.db，索引写 gen-(N+1).building.db，两个
                # SQLite 文件彼此隔离，所以搜索不需要暂停后台索引。
                log_event(
                    "search.snapshot.active",
                    component="search_engine",
                    request_id=request_id,
                    query_hash=query_hash,
                    generation=self.snapshot_generation,
                    database_path=str(self.db.db_path),
                    generation_isolated=True,
                )
                if token.cancelled:
                    raise CancelledError("搜索已取消")
            else:
                # 兼容旧模式：搜索和索引可能访问同一数据库，必须向协调器申请
                # 搜索租约，让写入在搜索期间安全暂停。
                log_event(
                    "search.waiting_for_index" if not search_active() else "search.waiting_for_lease",
                    component="search_engine",
                    request_id=request_id,
                    query_hash=query_hash,
                )
                if status_callback is not None:
                    status_callback(
                        "waiting_for_index" if not search_active() else "waiting"
                    )
                lease = get_search_index_coordinator().request_search(
                    request_id,
                    cancel_event=None,
                    readiness_checker=self.db.search_projection_readiness,
                    cancel_checker=lambda: token.cancelled,
                    max_wait_seconds=10.0,
                )
            if lease is not None:
                log_event(
                    "search.lease.acquired",
                    component="search_engine",
                    request_id=request_id,
                    query_hash=query_hash,
                    database_path=str(self.db.db_path),
                    generation_isolated=False,
                )
                if status_callback is not None:
                    status_callback("active")
            elif status_callback is not None:
                # An immutable published generation never pauses the writer;
                # it is a read-only snapshot alongside the candidate file.
                status_callback("snapshot_active")
            # 单独的计时器只负责在超时时设置 CancelToken；实际搜索代码会在
            # 检查 token 时退出。这不是执行搜索的线程。
            timeout_timer = threading.Timer(
                MAX_SEARCH_SECONDS,
                lambda: token.cancel(reason="SEARCH_TIMEOUT"),
            )
            timeout_timer.daemon = True
            timeout_timer.start()
            # 整次搜索只使用同一个只读连接和同一个事务快照，避免“查文件名时
            # 看到版本A、查正文时却看到版本B”的不一致。
            with self.db.connect_read_only(timeout_seconds=2.0) as connection:
                connection.execute("BEGIN")
                previous = self._snapshot_connection
                self._snapshot_connection = connection
                try:
                    # _search_impl 完成：解析关键词 -> 找候选 -> 精确确认 ->
                    # 计算分数 -> 分页封装。
                    result = self._search_impl(
                        query,
                        token,
                        progress_callback=progress_callback,
                        partial_callback=partial_callback,
                    )
                    log_event(
                        "search.completed",
                        component="search_engine",
                        request_id=request_id,
                        query_hash=query_hash,
                        generation=self.snapshot_generation,
                        database_path=str(self.db.db_path),
                        generation_isolated=self.generation_isolated,
                        elapsed_ms=result.elapsed_ms,
                        checked_candidates=result.total_candidates,
                        confirmed_files=result.total_confirmed,
                    )
                    return result
                finally:
                    self._snapshot_connection = previous
        except SearchTimeout as exc:
            log_event(
                "search.timeout",
                level=logging.WARNING,
                component="search_engine",
                request_id=request_id,
                query_hash=query_hash,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        except CancelledError as exc:
            log_event(
                "search.cancelled",
                component="search_engine",
                request_id=request_id,
                query_hash=query_hash,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
        except OperationCancelled as exc:
            if token.cancelled:
                cancelled = CancelledError("搜索已取消")
                log_event(
                    "search.cancelled",
                    component="search_engine",
                    request_id=request_id,
                    query_hash=query_hash,
                    error_type=type(cancelled).__name__,
                    error_message=str(cancelled),
                )
                raise cancelled from exc
            raise
        except Exception as exc:
            log_exception(
                "search.failed",
                exc,
                component="search_engine",
                request_id=request_id,
                query_hash=query_hash,
            )
            raise
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()
            if lease is not None:
                lease.release()
                log_event(
                    "search.lease.released",
                    component="search_engine",
                    request_id=request_id,
                    query_hash=query_hash,
                )

    def _search_impl(
        self,
        query: SearchQuery,
        cancel_token: CancelToken | None = None,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        partial_callback: Callable[[SearchPage], None] | None = None,
    ) -> SearchPage:
        # 第一步不是立刻查表，而是确认当前已提交投影确实可以用于搜索。
        # Full readiness diagnostics scan large tables. Keep the foreground
        # search on the bounded committed-projection gate instead.
        readiness = (
            self._compact_readiness()
            if self.compact_snapshot
            else self.db.search_projection_readiness()
        )
        if not bool(readiness.get("search_available", readiness["ready"])):
            raise IndexNotReadyError(
                "尚无已经完成并提交的可搜索文件："
                f"{readiness.get('complete_files', 0)}/{readiness.get('eligible_files', 0)} 个范围内文件完成，"
                f"仍有 {readiness.get('blocking_files', 0)} 个文件需要处理"
            )
        if (
            query.mode != "regex"
            and contains_cjk(query.text)
            and any(character.isspace() for character in query.text)
            and not query.ignore_spaces
        ):
            # 中文关键词中空格经常只是输入分隔符。这里复制一份新 SearchQuery
            # 并开启 ignore_spaces，不直接修改调用方传入的请求对象。
            query = replace(query, ignore_spaces=True)
        # 从这里开始统计真正的检索耗时。
        started = time.perf_counter()
        token = cancel_token or CancelToken()
        # regex 模式会预编译正则；普通模式返回 None。
        regex_pattern = self._compile_regex(query)
        # 先按搜索模式切分原始关键词，再用与索引一致的规则进行归一化。
        raw_terms = parse_terms(query.text, query.mode)
        normalized_terms = [
            normalize_text(
                term,
                case_sensitive=query.case_sensitive,
                ignore_spaces=query.ignore_spaces,
                ignore_hyphens=query.ignore_hyphens,
            )
            for term in raw_terms
            if term.strip()
        ]
        normalized_terms = [term for term in normalized_terms if term]
        if not normalized_terms:
            # 归一化后没有有效关键词，不访问大表，直接返回空页。
            return SearchPage([], 0, 0, 0, query.page, query.page_size, 0, False)
        # 根据 SearchQuery 的三个来源开关，决定查询文件名/路径还是正文。
        search_filename, search_path, search_content = self._source_flags(query)
        # 同一个文件可能由文件名和多个正文块重复命中；字典以 file_id 合并结果。
        confirmed_by_file: dict[int, SearchResult] = {}
        page_start = max(0, (max(1, int(query.page)) - 1) * max(1, int(query.page_size)))
        page_end = page_start + max(1, min(100, int(query.page_size)))
        display_file_limit = max(1, min(100, int(query.page_size)))
        max_blocks_per_file = 20
        pending_content_rows: dict[int, list[sqlite3.Row]] = {}
        content_candidate_file_ids: set[int] = set()
        content_candidate_file_order: list[int] = []
        content_candidate_file_indexes: dict[int, int] = {}
        total_candidates = 0
        full_scan = self._requires_full_scan(query)
        total_to_scan = 0
        metadata_strategy = ""
        content_strategy = ""
        if search_filename or search_path:
            _sql, _params, metadata_strategy = self._metadata_query(
                query,
                normalized_terms,
                search_filename,
                search_path,
            )
        if search_content:
            _sql, _params, content_strategy = self._content_query(
                query,
                normalized_terms,
            )

        def slow_reason(strategy: str) -> str:
            if strategy == "like":
                return "关键词较短，正在使用兼容搜索"
            if strategy == "scan":
                return "正在执行正则或兼容全文扫描"
            return ""

        if full_scan:
            self._report_progress(
                progress_callback,
                stage="preparing_scan",
                phase_label="正在计算需要扫描的范围...",
                started=started,
                progress_kind="busy",
                can_cancel=True,
            )
            if search_filename or search_path:
                sql, params, _strategy = self._metadata_query(
                    query,
                    normalized_terms,
                    search_filename,
                    search_path,
                )
                total_to_scan += self._count_query_rows(sql, params, token)
            if search_content:
                sql, params, _strategy = self._content_query(
                    query,
                    normalized_terms,
                )
                total_to_scan += self._count_query_rows(sql, params, token)

        def consume(
            rows: Iterator[sqlite3.Row],
            *,
            stage: str,
            phase_label: str,
            reason: str,
            defer_content_hydration: bool = False,
        ) -> None:
            nonlocal total_candidates
            last_reported = total_candidates
            for row in rows:
                token.throw_if_cancelled()
                total_candidates += 1
                if defer_content_hydration:
                    file_id = int(row["file_id"])
                    if file_id not in content_candidate_file_ids:
                        content_candidate_file_ids.add(file_id)
                        content_candidate_file_indexes[file_id] = len(content_candidate_file_order)
                        content_candidate_file_order.append(file_id)
                    candidate_index = content_candidate_file_indexes[file_id]
                    if file_id in pending_content_rows or page_start <= candidate_index < page_end:
                        pending_content_rows.setdefault(file_id, []).append(row)
                    if total_candidates - last_reported >= 256:
                        self._report_progress(
                            progress_callback,
                            stage=stage,
                            phase_label=phase_label,
                            started=started,
                            progress_kind="determinate" if full_scan else "busy",
                            checked_candidates=total_candidates,
                            total_candidates=total_to_scan if full_scan else 0,
                            confirmed_files=len(confirmed_by_file),
                            can_cancel=True,
                            slow_reason=reason,
                        )
                        last_reported = total_candidates
                    continue
                file_id = int(row["file_id"])
                existing = confirmed_by_file.get(file_id)
                if (
                    not full_scan
                    and existing is None
                    and len(confirmed_by_file) >= display_file_limit
                ):
                    continue
                if existing is not None and len(existing.matches) >= max_blocks_per_file:
                    continue
                if int(row["candidate_only"] or 0):
                    hydrated = self._hydrate_content_row(row, token)
                    if hydrated is None:
                        continue
                    row = hydrated
                if self._matches(row, normalized_terms, query, regex_pattern, token):
                    result = self._row_to_result(
                        row,
                        raw_terms,
                        normalized_terms,
                        query,
                        regex_pattern,
                        token,
                    )
                    existing = confirmed_by_file.get(result.file_id)
                    if existing is None:
                        confirmed_by_file[result.file_id] = result
                    else:
                        self._merge_result(existing, result)
                if total_candidates - last_reported >= 256:
                    self._report_progress(
                        progress_callback,
                        stage=stage,
                        phase_label=phase_label,
                        started=started,
                        progress_kind="determinate" if full_scan else "busy",
                        checked_candidates=total_candidates,
                        total_candidates=total_to_scan if full_scan else 0,
                        confirmed_files=len(confirmed_by_file),
                        can_cancel=True,
                        slow_reason=reason,
                    )
                    last_reported = total_candidates

        if search_filename or search_path:
            self._report_progress(
                progress_callback,
                stage="searching_metadata",
                phase_label="正在搜索文件名和路径...",
                started=started,
                progress_kind="determinate" if full_scan else "busy",
                checked_candidates=total_candidates,
                total_candidates=total_to_scan if full_scan else 0,
                confirmed_files=len(confirmed_by_file),
                can_cancel=True,
                slow_reason=slow_reason(metadata_strategy),
            )
            consume(
                self._metadata_candidates(
                    query,
                    normalized_terms,
                    token,
                    search_filename,
                    search_path,
                ),
                stage="searching_metadata",
                phase_label="正在搜索文件名和路径...",
                reason=slow_reason(metadata_strategy),
            )
            if partial_callback is not None and confirmed_by_file:
                partial_callback(
                    self._build_page(
                        confirmed_by_file,
                        query,
                        total_candidates,
                        started,
                        partial=True,
                    )
                )

        if search_content:
            self._report_progress(
                progress_callback,
                stage="searching_content",
                phase_label="正在搜索正文...",
                started=started,
                progress_kind="determinate" if full_scan else "busy",
                checked_candidates=total_candidates,
                total_candidates=total_to_scan if full_scan else 0,
                confirmed_files=len(confirmed_by_file),
                can_cancel=True,
                slow_reason=(
                    slow_reason(content_strategy)
                ),
            )
            consume(
                self._content_candidates(query, normalized_terms, token),
                stage="scanning" if full_scan else "searching_content",
                phase_label=(
                    "正在扫描正文..." if full_scan else "正在搜索正文..."
                ),
                reason=slow_reason(content_strategy),
                defer_content_hydration=(
                    not full_scan
                    and content_strategy != "scan"
                ),
            )

            # Confirm only the selected result window. The candidate query
            # above still visits every lightweight block so counts remain
            # honest, but raw_text/context are fetched for at most 100 files
            # and 20 blocks per file.
            for file_id, candidates in pending_content_rows.items():
                token.throw_if_cancelled()
                for candidate in candidates[:max_blocks_per_file]:
                    hydrated = self._hydrate_content_row(candidate, token)
                    if hydrated is None:
                        continue
                    if self._matches(hydrated, normalized_terms, query, regex_pattern, token):
                        result = self._row_to_result(
                            hydrated,
                            raw_terms,
                            normalized_terms,
                            query,
                            regex_pattern,
                            token,
                        )
                        existing = confirmed_by_file.get(result.file_id)
                        if existing is None:
                            confirmed_by_file[result.file_id] = result
                        else:
                            self._merge_result(existing, result)

        self._report_progress(
            progress_callback,
            stage="sorting",
            phase_label="正在排序和整理结果...",
            started=started,
            progress_kind="busy",
            checked_candidates=total_candidates,
            total_candidates=total_to_scan if full_scan else 0,
            confirmed_files=len(confirmed_by_file),
            can_cancel=True,
        )
        page = self._build_page(
            confirmed_by_file,
            query,
            total_candidates,
            started,
            partial=False,
            total_confirmed_override=(
                len(set(confirmed_by_file).union(content_candidate_file_ids))
                if content_candidate_file_ids
                else None
            ),
            page_already_selected=bool(content_candidate_file_ids),
        )
        self._report_progress(
            progress_callback,
            stage="complete",
            phase_label="搜索完成",
            started=started,
            progress_kind="determinate",
            checked_candidates=total_candidates,
            total_candidates=max(total_candidates, total_to_scan),
            confirmed_files=page.total_confirmed,
            can_cancel=False,
        )
        if page.elapsed_ms >= 2_000:
            logger.warning(
                "Slow search completed: mode=%s query_length=%s candidates=%s confirmed=%s elapsed_ms=%s full_scan=%s",
                query.mode,
                len(query.text),
                total_candidates,
                page.total_confirmed,
                page.elapsed_ms,
                full_scan,
            )
        return page

    def _build_page(
        self,
        confirmed_by_file: dict[int, SearchResult],
        query: SearchQuery,
        total_candidates: int,
        started: float,
        *,
        partial: bool,
        total_confirmed_override: int | None = None,
        page_already_selected: bool = False,
    ) -> SearchPage:
        confirmed = copy.deepcopy(list(confirmed_by_file.values()))
        for result in confirmed:
            self._finalize_result(result)
        confirmed.sort(key=lambda item: self._result_sort_key(item, query.sort_by))
        total_confirmed = max(
            len(confirmed),
            int(total_confirmed_override or 0),
        )
        max_results = max(1, int(query.max_results or total_confirmed or 1))
        available_results = min(total_confirmed, max_results)
        confirmed = confirmed[:available_results]
        page = max(1, query.page)
        start = 0 if page_already_selected else (page - 1) * query.page_size
        end = len(confirmed) if page_already_selected else start + query.page_size
        return SearchPage(
            confirmed[start:end],
            total_candidates,
            total_confirmed,
            int((time.perf_counter() - started) * 1000),
            page,
            query.page_size,
            available_results,
            total_confirmed > available_results,
            partial,
        )

    def _result_sort_key(
        self,
        result: SearchResult,
        sort_by: str,
    ) -> tuple[object, ...]:
        filename = result.filename.casefold()
        path = result.file_path.casefold()
        if sort_by == "modified_desc":
            return (-result.modified_time, -result.score, filename, path)
        if sort_by == "modified_asc":
            return (result.modified_time, -result.score, filename, path)
        if sort_by == "size_desc":
            return (-result.size_bytes, -result.score, filename, path)
        if sort_by == "size_asc":
            return (result.size_bytes, -result.score, filename, path)
        if sort_by == "filename_asc":
            return (filename, -result.score, -result.modified_time, path)
        if sort_by == "filename_desc":
            return (
                self._reverse_text_key(filename),
                -result.score,
                -result.modified_time,
                self._reverse_text_key(path),
            )
        return (-result.score, -result.modified_time, filename, path)

    @staticmethod
    def _reverse_text_key(value: str) -> tuple[int, ...]:
        return tuple(-ord(character) for character in value) + (1,)

    def _compile_regex(self, query: SearchQuery) -> re.Pattern[str] | None:
        if query.mode != "regex":
            return None
        flags = 0 if query.case_sensitive else re.IGNORECASE
        try:
            return re.compile(query.text, flags)
        except re.error as exc:
            raise ValueError(f"正则表达式无效：{exc}") from exc

    def _candidate_rows(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
        *,
        search_filename: bool,
        search_path: bool,
        search_content: bool,
    ) -> Iterator[sqlite3.Row]:
        if search_filename or search_path:
            yield from self._metadata_candidates(query, normalized_terms, token, search_filename, search_path)
        if search_content:
            yield from self._content_candidates(query, normalized_terms, token)

    def _source_flags(self, query: SearchQuery) -> tuple[bool, bool, bool]:
        search_filename = query.search_filename
        search_path = query.search_path
        search_content = query.search_content
        if query.mode == "filename":
            return True, query.search_path, False
        if not (search_filename or search_path or search_content):
            search_content = True
        return search_filename, search_path, search_content

    def _content_candidates(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
    ) -> Iterator[sqlite3.Row]:
        sql, params, _strategy = self._content_query(
            query,
            normalized_terms,
            include_text=self._requires_full_scan(query),
        )
        yield from self._stream_rows(sql, params, token)

    def _content_query(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        *,
        include_text: bool = True,
    ) -> tuple[str, list[Any], str]:
        where: list[str] = (
            ["1 = 1"]
            if self.compact_snapshot
            else [
                "f.is_deleted = 0",
                "f.parse_status IN ('success', 'metadata_only')",
                """NOT EXISTS (
                    SELECT 1 FROM index_scope_exclusions excluded
                    WHERE excluded.file_id = f.id
                      AND excluded.revoked_at IS NULL
                      AND excluded.invalidated_at IS NULL
                )""",
            ]
        )
        params: list[Any] = []
        source_params: list[Any] = []
        source_where: list[str] = []
        strategy = "scan"
        source_grouping = ""
        source_table = "content_fts"
        source_id_expression = self._fts_rowid_expression("content_fts")
        if self._can_use_fts_match("content_fts", query, normalized_terms):
            source_where.append("content_fts MATCH ?")
            source_params.append(
                self._fts_match_expression(
                    normalized_terms,
                    query.mode,
                    column=None if self.compact_snapshot else "normalized_text",
                    phrase=not self.compact_snapshot,
                )
            )
            strategy = "fts_match"
        elif not self._requires_full_scan(query):
            if self.compact_snapshot:
                # The compact snapshot intentionally does not persist a
                # second normalized-text column. A two-character query scans
                # compressed blocks in bounded batches and verifies matches
                # in Python.
                source_table = "search_blocks"
                source_id_expression = "CAST(id AS INTEGER)"
                # Trigram FTS5 cannot return a two-character CJK term.  Use
                # the decoded compact view for a coarse candidate filter, then
                # keep the Python matcher as the exact verifier.  Returning
                # every block here made the total counter include unverified
                # files while the hydration window could contain none of them.
                like_predicates = ["raw_text LIKE ?" for _ in normalized_terms]
                source_where.append(
                    "(" + (" OR " if query.mode == "any" else " AND ").join(like_predicates) + ")"
                )
                source_params.extend(f"%{term}%" for term in normalized_terms)
                strategy = "compatibility_scan"
            else:
                text_predicates = self._like_predicates(
                    ["normalized_text"], query, normalized_terms, source_params
                )
                source_where.append("(" + text_predicates + ")")
                strategy = "like"
        where.append("(cb.source_type IS NULL OR cb.source_type != 'metadata')")
        self._append_filters(where, params, query)
        if not query.include_ocr:
            where.append("(cb.source_type IS NULL OR cb.source_type != 'ocr')")
        elif not query.include_ocr_fuzzy:
            where.append(
                "(cb.source_type IS NULL OR cb.source_type != 'ocr' OR COALESCE(cb.ocr_confidence, 0) >= ?)"
            )
            params.append(float(query.ocr_min_confidence))
        text_columns = (
            "cb.raw_text, cb.normalized_text, 0 AS candidate_only"
            if include_text
            else "NULL AS raw_text, NULL AS normalized_text, 1 AS candidate_only"
        )
        outer_columns = """
            block_id, file_id, path, filename, extension, size_bytes,
            modified_time, parse_status, location_text, page_number,
            slide_number, sheet_name, cell_start, cell_end, line_start,
            line_end, raw_text, normalized_text, source_type, ocr_confidence,
            candidate_only
        """
        columns = f"""
            cb.id AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
            f.size_bytes, f.modified_time, f.parse_status, cb.location_text,
            cb.page_number, cb.slide_number, cb.sheet_name,
            cb.cell_start, cb.cell_end, cb.line_start, cb.line_end,
            {text_columns}, cb.source_type, cb.ocr_confidence
        """
        predicates = " AND ".join(where)
        source_predicates = " AND ".join(source_where) or "1 = 1"
        block_source = "search_blocks" if self.compact_snapshot else "content_blocks"
        # Materialize the expensive FTS/LIKE candidate set once. The old query
        # scanned content_fts independently for document-owned and file-owned
        # blocks, which made short CJK searches pay the same full scan twice.
        sql = f"""
            WITH matched_blocks(block_id) AS MATERIALIZED (
                SELECT DISTINCT {source_id_expression}
                FROM {source_table}
                WHERE {source_predicates}
                {source_grouping}
            )
            SELECT {columns}
            FROM matched_blocks mb
            JOIN {block_source} cb ON cb.id = mb.block_id
            JOIN files f ON f.document_id = cb.document_id
            WHERE cb.document_id IS NOT NULL AND {predicates}
            UNION ALL
            SELECT {columns}
            FROM matched_blocks mb
            JOIN {block_source} cb ON cb.id = mb.block_id
            JOIN files f ON f.id = cb.file_id
            WHERE cb.document_id IS NULL AND {predicates}
        """
        return sql, [*source_params, *params, *params], strategy

    def _hydrate_content_row(
        self,
        candidate: sqlite3.Row,
        token: CancelToken,
    ) -> sqlite3.Row | None:
        token.throw_if_cancelled()
        connection = self._snapshot_connection
        owns_connection = connection is None
        if owns_connection:
            context = self.db.connect_read_only(timeout_seconds=0.75)
            connection = context.__enter__()
        block_source = "search_blocks" if self.compact_snapshot else "content_blocks"
        try:
            assert connection is not None
            row = connection.execute(
                f"""
                SELECT cb.id AS block_id, f.id AS file_id, f.path, f.filename,
                       f.extension, f.size_bytes, f.modified_time,
                       f.parse_status, cb.location_text, cb.page_number,
                       cb.slide_number, cb.sheet_name, cb.cell_start, cb.cell_end,
                       cb.line_start, cb.line_end, cb.raw_text, cb.normalized_text,
                       cb.source_type, cb.ocr_confidence, 0 AS candidate_only
                FROM {block_source} cb
                JOIN files f ON f.id = ?
                WHERE cb.id = ?
                  AND {"1 = 1" if self.compact_snapshot else "f.is_deleted = 0"}
                  AND f.parse_status IN ('success', 'metadata_only')
                  AND (
                      cb.document_id IS NULL
                      OR cb.document_id = f.document_id
                      OR (cb.document_id = -f.id AND f.document_id IS NULL)
                  )
                """,
                (int(candidate["file_id"]), int(candidate["block_id"])),
            ).fetchone()
            return row
        finally:
            if owns_connection:
                context.__exit__(None, None, None)

    def _detect_compact_snapshot(self) -> bool:
        try:
            with self.db.connect_read_only(timeout_seconds=0.5) as con:
                return (
                    con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot_meta'"
                    ).fetchone()
                    is not None
                    and con.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='view' AND name='search_blocks'"
                    ).fetchone()
                    is not None
                )
        except sqlite3.Error:
            return False

    def _compact_readiness(self) -> dict[str, object]:
        connection = self._snapshot_connection
        owns_connection = connection is None
        context = None
        if owns_connection:
            context = self.db.connect_read_only(timeout_seconds=1.0)
            connection = context.__enter__()
        try:
            assert connection is not None
            row = connection.execute(
                "SELECT * FROM snapshot_meta WHERE id = 1"
            ).fetchone()
            if row is None or int(row["state"]) != 1:
                return {"search_available": False, "ready": False}
            return {
                "search_available": True,
                "ready": True,
                "index_complete": True,
                "search_is_partial": False,
                "active_generation": int(row["generation"]),
                "complete_files": int(row["file_count"]),
                "eligible_files": int(row["file_count"]),
                "blocking_files": 0,
                "document_count": int(row["document_count"]),
                "block_count": int(row["block_count"]),
            }
        finally:
            if owns_connection and context is not None:
                context.__exit__(None, None, None)

    def _metadata_candidates(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
        search_filename: bool,
        search_path: bool,
    ) -> Iterator[sqlite3.Row]:
        sql, params, _strategy = self._metadata_query(
            query,
            normalized_terms,
            search_filename,
            search_path,
        )
        yield from self._stream_rows(sql, params, token)

    def _metadata_query(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        search_filename: bool,
        search_path: bool,
    ) -> tuple[str, list[Any], str]:
        where: list[str] = (
            ["1 = 1"]
            if self.compact_snapshot
            else [
                "f.is_deleted = 0",
                "f.parse_status IN ('success', 'metadata_only')",
                """NOT EXISTS (
                    SELECT 1 FROM index_scope_exclusions excluded
                    WHERE excluded.file_id = f.id
                      AND excluded.revoked_at IS NULL
                      AND excluded.invalidated_at IS NULL
                )""",
            ]
        )
        params: list[Any] = []
        fields: list[str] = []
        if search_filename:
            fields.append("LOWER(f.filename)")
        if search_path:
            fields.append("LOWER(f.path)")
        if not fields:
            fields.append("LOWER(f.filename)")
        use_fts = self._can_use_fts_match("files_fts", query, normalized_terms)
        strategy = "scan"
        if use_fts:
            if self.compact_snapshot:
                column = None
            elif search_filename and not search_path:
                column = "filename"
            elif search_path and not search_filename:
                column = "path"
            else:
                column = None
            where.append("files_fts MATCH ?")
            params.append(
                self._fts_match_expression(
                    normalized_terms,
                    query.mode,
                    column=column,
                    phrase=not self.compact_snapshot,
                )
            )
            strategy = "fts_match"
        elif not self._requires_full_scan(query):
            where.append("(" + self._like_predicates(fields, query, normalized_terms, params) + ")")
            strategy = "like"
        self._append_filters(where, params, query)
        if search_filename and search_path:
            raw_expression = "f.filename || char(10) || f.path"
        elif search_path:
            raw_expression = "f.path"
        else:
            raw_expression = "f.filename"
        from_clause = (
            "FROM files_fts JOIN files f "
            f"ON f.id = {self._fts_rowid_expression('files_fts')}"
            if use_fts
            else "FROM files f"
        )
        sql = f"""
            SELECT
                NULL AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, '文件名/路径' AS location_text,
                NULL AS page_number, NULL AS slide_number, NULL AS sheet_name,
                NULL AS cell_start, NULL AS cell_end, NULL AS line_start, NULL AS line_end,
                {raw_expression} AS raw_text,
                {raw_expression} AS normalized_text,
                'metadata' AS source_type, NULL AS ocr_confidence,
                0 AS candidate_only
            {from_clause}
            WHERE {' AND '.join(where)}
        """
        return sql, params, strategy

    def _can_use_fts_match(
        self,
        table: str,
        query: SearchQuery,
        normalized_terms: list[str],
    ) -> bool:
        return (
            not self._requires_full_scan(query)
            and bool(normalized_terms)
            and all(len(term) >= 3 for term in normalized_terms)
            and self._fts_table_uses_trigram(table)
        )

    def _fts_table_uses_trigram(self, table: str) -> bool:
        cached = self._trigram_tables.get(table)
        if cached is not None:
            return cached
        connection = self._snapshot_connection
        if connection is not None:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
        else:
            with self.db.connect_read_only() as con:
                row = con.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
        definition = str(row["sql"] or "") if row is not None else ""
        available = "trigram" in definition.lower()
        self._trigram_tables[table] = available
        return available

    def _fts_rowid_expression(self, table: str) -> str:
        """Return the row identity for legacy or compact contentless FTS."""

        connection = self._snapshot_connection
        if connection is not None:
            row = connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
        else:
            with self.db.connect_read_only(timeout_seconds=0.5) as con:
                row = con.execute(
                    "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                    (table,),
                ).fetchone()
        definition = str(row["sql"] or "") if row is not None else ""
        return f"CAST({table}.rowid AS INTEGER)" if "content=''" in definition.lower() else (
            "CAST(block_id AS INTEGER)" if table == "content_fts" else "CAST(file_id AS INTEGER)"
        )

    @staticmethod
    def _fts_match_expression(
        normalized_terms: list[str],
        mode: str,
        *,
        column: str | None,
        phrase: bool = True,
    ) -> str:
        glue = " OR " if mode == "any" else " AND "
        expressions: list[str] = []
        for term in normalized_terms:
            if phrase:
                terms = ['"' + term.replace('"', '""') + '"']
            else:
                # detail=none cannot execute a multi-token phrase. Expand a
                # long term into its trigram tokens and let the raw-text
                # verification phase enforce order and exact semantics.
                terms = [
                    '"' + term[index : index + 3].replace('"', '""') + '"'
                    for index in range(max(1, len(term) - 2))
                ]
            term_expression = " AND ".join(terms)
            if len(terms) > 1:
                term_expression = f"({term_expression})"
            expressions.append(
                f"{column} : {term_expression}" if column else term_expression
            )
        return glue.join(expressions)

    def _requires_full_scan(self, query: SearchQuery) -> bool:
        return query.mode == "regex" or query.ignore_spaces or query.ignore_hyphens

    def _count_query_rows(
        self,
        sql: str,
        params: list[Any],
        token: CancelToken,
    ) -> int:
        token.throw_if_cancelled()
        for attempt in range(3):
            try:
                connection = self._snapshot_connection
                if connection is not None:
                    connection.set_progress_handler(
                        lambda: 1 if token.cancelled else 0,
                        2_000,
                    )
                    row = connection.execute(
                        f"SELECT COUNT(*) AS n FROM ({sql}) candidate_rows",
                        params,
                    ).fetchone()
                    return int(row["n"] or 0)
                with self.db.connect_read_only(timeout_seconds=0.75) as con:
                    con.set_progress_handler(lambda: 1 if token.cancelled else 0, 2_000)
                    row = con.execute(
                        f"SELECT COUNT(*) AS n FROM ({sql}) candidate_rows",
                        params,
                    ).fetchone()
                    return int(row["n"] or 0)
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if token.cancelled and "interrupt" in message:
                    if token.cancel_reason == "SEARCH_TIMEOUT":
                        raise SearchTimeout("搜索达到时间上限") from exc
                    raise CancelledError("搜索已取消") from exc
                if "database is locked" in message or "database is busy" in message:
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
                        continue
                raise
        raise sqlite3.OperationalError("database is locked")

    @staticmethod
    def _report_progress(
        callback: Callable[[dict[str, object]], None] | None,
        *,
        stage: str,
        phase_label: str,
        started: float,
        progress_kind: str,
        checked_candidates: int = 0,
        total_candidates: int = 0,
        confirmed_files: int = 0,
        can_cancel: bool,
        slow_reason: str = "",
    ) -> None:
        if callback is None:
            return
        if progress_kind == "determinate" and total_candidates > 0:
            percent = min(
                100,
                int(max(0, checked_candidates) * 100 / total_candidates),
            )
        else:
            percent = None
        callback(
            {
                "stage": stage,
                "phase_label": phase_label,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
                "metadata_candidates": (
                    checked_candidates if stage == "searching_metadata" else 0
                ),
                "content_candidates": (
                    checked_candidates
                    if stage in {"searching_content", "scanning"}
                    else 0
                ),
                "checked_candidates": checked_candidates,
                "total_candidates": total_candidates,
                "confirmed_files": confirmed_files,
                "percent": percent,
                "progress_kind": progress_kind,
                "can_cancel": can_cancel,
                "slow_reason": slow_reason,
            }
        )

    def _stream_rows(
        self,
        sql: str,
        params: list[Any],
        token: CancelToken,
        *,
        batch_size: int = 512,
    ) -> Iterator[sqlite3.Row]:
        for attempt in range(3):
            try:
                connection = self._snapshot_connection
                if connection is not None:
                    connection.set_progress_handler(
                        lambda: 1 if token.cancelled else 0,
                        2_000,
                    )
                    cursor = connection.execute(sql, params)
                    while True:
                        token.throw_if_cancelled()
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            return
                        for row in rows:
                            token.throw_if_cancelled()
                            yield row
                    return
                with self.db.connect_read_only(timeout_seconds=0.75) as con:
                    con.set_progress_handler(lambda: 1 if token.cancelled else 0, 2_000)
                    cursor = con.execute(sql, params)
                    while True:
                        token.throw_if_cancelled()
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            return
                        for row in rows:
                            token.throw_if_cancelled()
                            yield row
                return
            except sqlite3.OperationalError as exc:
                message = str(exc).lower()
                if token.cancelled and "interrupt" in message:
                    if token.cancel_reason == "SEARCH_TIMEOUT":
                        raise SearchTimeout("搜索达到时间上限") from exc
                    raise CancelledError("搜索已取消") from exc
                if "database is locked" in message or "database is busy" in message:
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
                        continue
                raise
        raise sqlite3.OperationalError("database is locked")

    def _like_predicates(
        self,
        fields: list[str],
        query: SearchQuery,
        normalized_terms: list[str],
        params: list[Any],
    ) -> str:
        glue = " OR " if query.mode == "any" else " AND "
        term_predicates: list[str] = []
        for term in normalized_terms:
            field_predicates = []
            for field in fields:
                field_predicates.append(f"{field} LIKE ?")
                params.append(f"%{term}%")
            term_predicates.append("(" + " OR ".join(field_predicates) + ")")
        return glue.join(term_predicates)

    def _append_filters(self, where: list[str], params: list[Any], query: SearchQuery) -> None:
        if query.root_ids:
            where.append(f"f.root_id IN ({','.join('?' for _ in query.root_ids)})")
            params.extend(query.root_ids)
        if query.extensions:
            extensions = [ext if ext.startswith(".") else f".{ext}" for ext in query.extensions]
            where.append(f"f.extension IN ({','.join('?' for _ in extensions)})")
            params.extend(ext.lower() for ext in extensions)
        if query.date_from:
            where.append("f.modified_time >= ?")
            params.append(query.date_from.timestamp())
        if query.date_to:
            where.append("f.modified_time <= ?")
            params.append(query.date_to.timestamp())
        if query.min_size is not None:
            where.append("f.size_bytes >= ?")
            params.append(query.min_size)
        if query.max_size is not None:
            where.append("f.size_bytes <= ?")
            params.append(query.max_size)

    def _matches(
        self,
        row: sqlite3.Row,
        terms: list[str],
        query: SearchQuery,
        regex_pattern: re.Pattern[str] | None,
        token: CancelToken | None = None,
    ) -> bool:
        # 候选召回只是“可能命中”；_matches 会读取真实文本做最终确认，
        # 防止 FTS/LIKE 候选因为分词或归一化规则产生假阳性。
        if token is not None:
            token.throw_if_cancelled()
        if regex_pattern is not None:
            text = str(row["raw_text"] or "")
            if query.ignore_spaces or query.ignore_hyphens:
                text = normalize_text(
                    text,
                    case_sensitive=True,
                    ignore_spaces=query.ignore_spaces,
                    ignore_hyphens=query.ignore_hyphens,
                )
            return regex_pattern.search(text) is not None
        stored_normalized = str(row["normalized_text"] or "")
        if (
            stored_normalized
            and not query.case_sensitive
            and not query.ignore_spaces
            and not query.ignore_hyphens
        ):
            # Parsers persist this exact normalization for the normal query
            # path.  Re-normalizing it here was the dominant cost for huge
            # blocks and provided no additional correctness.
            text = stored_normalized
        else:
            text = normalize_text(
                str(row["raw_text"] or ""),
                case_sensitive=query.case_sensitive,
                ignore_spaces=query.ignore_spaces,
                ignore_hyphens=query.ignore_hyphens,
            )
        mode = query.mode
        if mode in {"exact", "phrase", "filename"}:
            # 精确、短语和文件名模式都要求完整目标串出现。
            return terms[0] in text
        if mode == "all":
            # all 模式要求每个关键词都出现。
            return all(term in text for term in terms)
        if mode == "any":
            # any 模式只要求至少一个关键词出现。
            return any(term in text for term in terms)
        return terms[0] in text

    def _row_to_result(
        self,
        row: sqlite3.Row,
        raw_terms: list[str],
        normalized_terms: list[str],
        query: SearchQuery,
        regex_pattern: re.Pattern[str] | None,
        token: CancelToken | None = None,
    ) -> SearchResult:
        # 将一条确认命中的数据库行转换成 UI 可以直接使用的 SearchResult，
        # 同时生成上下文片段、命中次数和相关度分数。
        if token is not None:
            token.throw_if_cancelled()
        raw_text = str(row["raw_text"] or "")
        stored_normalized = str(row["normalized_text"] or "")
        if (
            stored_normalized
            and not query.case_sensitive
            and not query.ignore_spaces
            and not query.ignore_hyphens
        ):
            normalized_text = stored_normalized
        else:
            normalized_text = normalize_text(
                raw_text,
                case_sensitive=query.case_sensitive,
                ignore_spaces=query.ignore_spaces,
                ignore_hyphens=query.ignore_hyphens,
            )
        if regex_pattern is None:
            hits = count_hits(normalized_text, normalized_terms)
            if token is not None:
                token.throw_if_cancelled()
            context = make_context(
                raw_text[:MAX_CONTEXT_CHARS],
                raw_terms,
                case_sensitive=query.case_sensitive,
                ignore_spaces=query.ignore_spaces,
                ignore_hyphens=query.ignore_hyphens,
            )
        else:
            first_match: re.Match[str] | None = None
            hits = 0
            for regex_match in regex_pattern.finditer(raw_text):
                if token is not None and hits % 256 == 0:
                    token.throw_if_cancelled()
                if first_match is None:
                    first_match = regex_match
                hits += 1
            context = make_regex_context(
                raw_text,
                first_match,
            )
        filename_norm = normalize_text(
            str(row["filename"]),
            case_sensitive=query.case_sensitive,
            ignore_spaces=query.ignore_spaces,
            ignore_hyphens=query.ignore_hyphens,
        )
        # 基础分是正文中的命中次数。
        score = float(hits)
        first = normalized_terms[0]
        if filename_norm == first:
            # 文件名完全等于关键词通常最相关，因此给予最高加分。
            score += 1000
        elif filename_norm.startswith(first):
            # 文件名以关键词开头，相关度次之。
            score += 500
        elif first in filename_norm:
            # 文件名只要包含关键词，也比纯正文命中更靠前。
            score += 250
        if str(row["source_type"]) == "metadata":
            # metadata 表示文件名/路径等元数据命中。
            score += 50
        if str(row["source_type"]) == "ocr":
            # OCR 可能存在识别误差，所以相同条件下稍微降权。
            score -= 25
        confidence = float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None
        is_fuzzy = (
            str(row["source_type"]) == "ocr"
            and (confidence is None or confidence < float(query.ocr_min_confidence))
        )
        file_path = str(row["path"])
        location_text = str(row["location_text"] or "")
        if " > " in file_path and str(row["source_type"] or "") != "metadata":
            archive_path, internal_path = file_path.split(" > ", 1)
            archive_name = archive_path.replace("\\", "/").rsplit("/", 1)[-1]
            location_text = f"{archive_name} > {internal_path} > {location_text}"
        hit = SearchHit(
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            location_text=location_text,
            context=context,
            hit_count=max(hits, 1),
            source_type=str(row["source_type"] or "native_text"),
            page_number=_optional_int(row["page_number"]),
            slide_number=_optional_int(row["slide_number"]),
            sheet_name=_optional_text(row["sheet_name"]),
            cell_start=_optional_text(row["cell_start"]),
            cell_end=_optional_text(row["cell_end"]),
            line_start=_optional_int(row["line_start"]),
            line_end=_optional_int(row["line_end"]),
            ocr_confidence=confidence,
            is_fuzzy=is_fuzzy,
        )
        return SearchResult(
            file_id=int(row["file_id"]),
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            file_path=file_path,
            filename=str(row["filename"]),
            extension=str(row["extension"] or ""),
            size_bytes=int(row["size_bytes"] or 0),
            modified_time=float(row["modified_time"] or 0),
            location_text=hit.location_text,
            context=hit.context,
            hit_count=hit.hit_count,
            source_type=hit.source_type,
            parse_status=str(row["parse_status"] or ""),
            score=score,
            page_number=hit.page_number,
            slide_number=hit.slide_number,
            sheet_name=hit.sheet_name,
            cell_start=hit.cell_start,
            cell_end=hit.cell_end,
            line_start=hit.line_start,
            line_end=hit.line_end,
            ocr_confidence=hit.ocr_confidence,
            has_fuzzy_match=hit.is_fuzzy,
            matches=[hit],
        )

    def _merge_result(self, target: SearchResult, source: SearchResult) -> None:
        seen = {(hit.block_id, hit.source_type, hit.location_text, hit.context) for hit in target.matches}
        for hit in source.matches:
            key = (hit.block_id, hit.source_type, hit.location_text, hit.context)
            if key not in seen:
                target.matches.append(hit)
                seen.add(key)
        target.score += source.score
        target.hit_count += source.hit_count
        if source.ocr_confidence is not None:
            target.ocr_confidence = max(target.ocr_confidence or 0.0, source.ocr_confidence)
        target.has_fuzzy_match = target.has_fuzzy_match or source.has_fuzzy_match

    def _finalize_result(self, result: SearchResult) -> None:
        result.matches.sort(key=self._hit_sort_key)
        result.hit_count = sum(hit.hit_count for hit in result.matches)
        result.block_id = next((hit.block_id for hit in result.matches if hit.block_id is not None), None)
        result.source_type = self._dominant_source_type(result.matches)
        result.ocr_confidence = self._best_ocr_confidence(result.matches)
        result.has_fuzzy_match = any(hit.is_fuzzy for hit in result.matches)
        primary_hit = next(
            (hit for hit in result.matches if hit.block_id is not None),
            result.matches[0] if result.matches else None,
        )
        if primary_hit is not None:
            result.page_number = primary_hit.page_number
            result.slide_number = primary_hit.slide_number
            result.sheet_name = primary_hit.sheet_name
            result.cell_start = primary_hit.cell_start
            result.cell_end = primary_hit.cell_end
            result.line_start = primary_hit.line_start
            result.line_end = primary_hit.line_end
        result.location_text = self._summarize_locations(result.matches)
        result.context = self._combine_contexts(result.matches)

    def _hit_sort_key(self, hit: SearchHit) -> tuple[int, str]:
        priority = {"metadata": 0, "native_text": 1, "ocr": 2}
        return priority.get(hit.source_type, 1), hit.location_text

    def _dominant_source_type(self, hits: list[SearchHit]) -> str:
        priority = {"native_text": 3, "ocr": 2, "metadata": 1}
        return max((hit.source_type for hit in hits), key=lambda source: priority.get(source, 2), default="native_text")

    def _best_ocr_confidence(self, hits: list[SearchHit]) -> float | None:
        values = [hit.ocr_confidence for hit in hits if hit.ocr_confidence is not None]
        return max(values) if values else None

    def _summarize_locations(self, hits: list[SearchHit]) -> str:
        locations: list[str] = []
        for hit in hits:
            location = hit.location_text or self._source_label(hit.source_type)
            if location not in locations:
                locations.append(location)
        if len(locations) <= 3:
            return "、".join(locations)
        return "、".join(locations[:3]) + f" 等 {len(locations)} 处"

    def _combine_contexts(self, hits: list[SearchHit]) -> str:
        parts = []
        for hit in hits:
            label = hit.location_text or self._source_label(hit.source_type)
            context = hit.context or "文件名/路径命中"
            parts.append(f"{label}\n{context}")
        return "\n\n".join(parts)

    def _source_label(self, source_type: str) -> str:
        if source_type == "metadata":
            return "文件名/路径"
        if source_type == "ocr":
            return "OCR"
        return "正文"


def make_regex_context(raw_text: str, match: re.Match[str] | None, before: int = 80, after: int = 120) -> str:
    if not raw_text:
        return ""
    if match is None:
        snippet = raw_text[: before + after]
        return snippet + ("..." if len(raw_text) > len(snippet) else "")
    start = max(0, match.start() - before)
    end = min(len(raw_text), match.end() + after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(raw_text) else ""
    return prefix + raw_text[start:end].replace("\n", " ") + suffix
