from __future__ import annotations

import copy
import logging
import re
import sqlite3
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import Any

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import CancelledError, IndexNotReadyError
from local_full_text_search.core.normalizer import (
    contains_cjk,
    count_hits,
    make_context,
    normalize_text,
    parse_terms,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.models.search_result import SearchHit, SearchResult

logger = logging.getLogger(__name__)


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
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db
        self._trigram_tables: dict[str, bool] = {}

    def search(
        self,
        query: SearchQuery,
        cancel_token: CancelToken | None = None,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        partial_callback: Callable[[SearchPage], None] | None = None,
    ) -> SearchPage:
        readiness = self.db.index_readiness()
        if not bool(readiness["ready"]):
            raise IndexNotReadyError(
                "完整索引尚未完成："
                f"{readiness['complete_files']}/{readiness['eligible_files']} 个范围内文件完成，"
                f"仍有 {readiness['blocking_files']} 个文件需要处理"
            )
        if (
            query.mode != "regex"
            and contains_cjk(query.text)
            and any(character.isspace() for character in query.text)
            and not query.ignore_spaces
        ):
            query = replace(query, ignore_spaces=True)
        started = time.perf_counter()
        token = cancel_token or CancelToken()
        regex_pattern = self._compile_regex(query)
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
            return SearchPage([], 0, 0, 0, query.page, query.page_size, 0, False)
        search_filename, search_path, search_content = self._source_flags(query)
        confirmed_by_file: dict[int, SearchResult] = {}
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
        ) -> None:
            nonlocal total_candidates
            last_reported = total_candidates
            for row in rows:
                token.throw_if_cancelled()
                total_candidates += 1
                if self._matches(row, normalized_terms, query, regex_pattern):
                    result = self._row_to_result(
                        row,
                        raw_terms,
                        normalized_terms,
                        query,
                        regex_pattern,
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
            )

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
    ) -> SearchPage:
        confirmed = copy.deepcopy(list(confirmed_by_file.values()))
        for result in confirmed:
            self._finalize_result(result)
        confirmed.sort(
            key=lambda item: (-item.score, -item.modified_time, item.filename.lower())
        )
        total_confirmed = len(confirmed)
        max_results = max(1, int(query.max_results or total_confirmed or 1))
        available_results = min(total_confirmed, max_results)
        confirmed = confirmed[:available_results]
        page = max(1, query.page)
        start = (page - 1) * query.page_size
        end = start + query.page_size
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
        sql, params, _strategy = self._content_query(query, normalized_terms)
        yield from self._stream_rows(sql, params, token)

    def _content_query(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
    ) -> tuple[str, list[Any], str]:
        where: list[str] = [
            "f.is_deleted = 0",
            """NOT EXISTS (
                SELECT 1 FROM index_scope_exclusions excluded
                WHERE excluded.file_id = f.id
                  AND excluded.revoked_at IS NULL
                  AND excluded.invalidated_at IS NULL
            )""",
        ]
        params: list[Any] = []
        strategy = "scan"
        if self._can_use_fts_match("content_fts", query, normalized_terms):
            where.append("content_fts MATCH ?")
            params.append(
                self._fts_match_expression(
                    normalized_terms,
                    query.mode,
                    column="normalized_text",
                )
            )
            strategy = "fts_match"
        elif not self._requires_full_scan(query):
            text_predicates = self._like_predicates(["ft.normalized_text"], query, normalized_terms, params)
            where.append("(" + text_predicates + ")")
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
        sql = f"""
            SELECT
                cb.id AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, cb.location_text,
                cb.raw_text, cb.normalized_text, cb.source_type, cb.ocr_confidence
            FROM content_fts ft
            JOIN content_blocks cb ON cb.id = CAST(ft.block_id AS INTEGER)
            JOIN files f ON (
                (cb.document_id IS NOT NULL AND f.document_id = cb.document_id)
                OR (cb.document_id IS NULL AND f.id = cb.file_id)
            )
            WHERE {' AND '.join(where)}
        """
        return sql, params, strategy

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
        where: list[str] = [
            "f.is_deleted = 0",
            """NOT EXISTS (
                SELECT 1 FROM index_scope_exclusions excluded
                WHERE excluded.file_id = f.id
                  AND excluded.revoked_at IS NULL
                  AND excluded.invalidated_at IS NULL
            )""",
        ]
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
            if search_filename and not search_path:
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
            "ON f.id = CAST(files_fts.file_id AS INTEGER)"
            if use_fts
            else "FROM files f"
        )
        sql = f"""
            SELECT
                NULL AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, '文件名/路径' AS location_text,
                {raw_expression} AS raw_text,
                {raw_expression} AS normalized_text,
                'metadata' AS source_type, NULL AS ocr_confidence
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
        with self.db.connect() as con:
            row = con.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?",
                (table,),
            ).fetchone()
        definition = str(row["sql"] or "") if row is not None else ""
        available = "trigram" in definition.lower()
        self._trigram_tables[table] = available
        return available

    @staticmethod
    def _fts_match_expression(
        normalized_terms: list[str],
        mode: str,
        *,
        column: str | None,
    ) -> str:
        glue = " OR " if mode == "any" else " AND "
        expressions: list[str] = []
        for term in normalized_terms:
            quoted = '"' + term.replace('"', '""') + '"'
            expressions.append(f"{column} : {quoted}" if column else quoted)
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
        try:
            with self.db.connect() as con:
                con.set_progress_handler(lambda: 1 if token.cancelled else 0, 2_000)
                row = con.execute(
                    f"SELECT COUNT(*) AS n FROM ({sql}) candidate_rows",
                    params,
                ).fetchone()
                return int(row["n"] or 0)
        except sqlite3.OperationalError as exc:
            if token.cancelled and "interrupt" in str(exc).lower():
                raise CancelledError("搜索已取消") from exc
            raise

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
        try:
            with self.db.connect() as con:
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
        except sqlite3.OperationalError as exc:
            if token.cancelled and "interrupt" in str(exc).lower():
                raise CancelledError("搜索已取消") from exc
            raise

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
    ) -> bool:
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
        text = normalize_text(
            str(row["normalized_text"] or ""),
            case_sensitive=query.case_sensitive,
            ignore_spaces=query.ignore_spaces,
            ignore_hyphens=query.ignore_hyphens,
        )
        mode = query.mode
        if mode in {"exact", "phrase", "filename"}:
            return terms[0] in text
        if mode == "all":
            return all(term in text for term in terms)
        if mode == "any":
            return any(term in text for term in terms)
        return terms[0] in text

    def _row_to_result(
        self,
        row: sqlite3.Row,
        raw_terms: list[str],
        normalized_terms: list[str],
        query: SearchQuery,
        regex_pattern: re.Pattern[str] | None,
    ) -> SearchResult:
        raw_text = str(row["raw_text"] or "")
        normalized_text = normalize_text(
            raw_text,
            case_sensitive=query.case_sensitive,
            ignore_spaces=query.ignore_spaces,
            ignore_hyphens=query.ignore_hyphens,
        )
        if regex_pattern is None:
            hits = count_hits(normalized_text, normalized_terms)
            context = make_context(
                raw_text,
                raw_terms,
                case_sensitive=query.case_sensitive,
                ignore_spaces=query.ignore_spaces,
                ignore_hyphens=query.ignore_hyphens,
            )
        else:
            first_match: re.Match[str] | None = None
            hits = 0
            for regex_match in regex_pattern.finditer(raw_text):
                if first_match is None:
                    first_match = regex_match
                hits += 1
            context = make_regex_context(raw_text, first_match)
        filename_norm = normalize_text(
            str(row["filename"]),
            case_sensitive=query.case_sensitive,
            ignore_spaces=query.ignore_spaces,
            ignore_hyphens=query.ignore_hyphens,
        )
        score = float(hits)
        first = normalized_terms[0]
        if filename_norm == first:
            score += 1000
        elif filename_norm.startswith(first):
            score += 500
        elif first in filename_norm:
            score += 250
        if str(row["source_type"]) == "metadata":
            score += 50
        if str(row["source_type"]) == "ocr":
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
