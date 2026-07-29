from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Any

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import IndexNotReadyError
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


class SearchEngine:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def search(self, query: SearchQuery, cancel_token: CancelToken | None = None) -> SearchPage:
        readiness = self.db.index_readiness()
        if not bool(readiness["ready"]):
            raise IndexNotReadyError(
                "完整索引尚未完成："
                f"{readiness['complete_files']}/{readiness['eligible_files']} 个文件完整成功，"
                f"仍有 {readiness['blocking_files']} 个文件需要处理"
            )
        if query.mode != "regex" and contains_cjk(query.text) and not query.ignore_spaces:
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
        rows = self._candidate_rows(
            query,
            normalized_terms,
            token,
            search_filename=search_filename,
            search_path=search_path,
            search_content=search_content,
        )
        for row in rows:
            token.throw_if_cancelled()
            total_candidates += 1
            if self._matches(row, normalized_terms, query, regex_pattern):
                result = self._row_to_result(row, raw_terms, normalized_terms, query, regex_pattern)
                existing = confirmed_by_file.get(result.file_id)
                if existing is None:
                    confirmed_by_file[result.file_id] = result
                else:
                    self._merge_result(existing, result)
        confirmed = list(confirmed_by_file.values())
        for result in confirmed:
            self._finalize_result(result)
        confirmed.sort(key=lambda item: (-item.score, -item.modified_time, item.filename.lower()))
        total_confirmed = len(confirmed)
        max_results = max(1, int(query.max_results or total_confirmed or 1))
        available_results = min(total_confirmed, max_results)
        confirmed = confirmed[:available_results]
        page = max(1, query.page)
        start = (page - 1) * query.page_size
        end = start + query.page_size
        elapsed = int((time.perf_counter() - started) * 1000)
        return SearchPage(
            confirmed[start:end],
            total_candidates,
            total_confirmed,
            elapsed,
            page,
            query.page_size,
            available_results,
            total_confirmed > available_results,
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
        where: list[str] = ["f.is_deleted = 0"]
        params: list[Any] = []
        if not self._requires_full_scan(query):
            text_predicates = self._like_predicates(["ft.normalized_text"], query, normalized_terms, params)
            where.append("(" + text_predicates + ")")
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
        yield from self._stream_rows(sql, params, token)

    def _metadata_candidates(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
        search_filename: bool,
        search_path: bool,
    ) -> Iterator[sqlite3.Row]:
        where: list[str] = ["f.is_deleted = 0"]
        params: list[Any] = []
        fields: list[str] = []
        if search_filename:
            fields.append("LOWER(f.filename)")
        if search_path:
            fields.append("LOWER(f.path)")
        if not fields:
            fields.append("LOWER(f.filename)")
        if not self._requires_full_scan(query):
            where.append("(" + self._like_predicates(fields, query, normalized_terms, params) + ")")
        self._append_filters(where, params, query)
        sql = f"""
            SELECT
                NULL AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, '文件名/路径' AS location_text,
                f.filename || char(10) || f.path AS raw_text,
                f.filename || char(10) || f.path AS normalized_text,
                'metadata' AS source_type, NULL AS ocr_confidence
            FROM files f
            WHERE {' AND '.join(where)}
        """
        yield from self._stream_rows(sql, params, token)

    def _requires_full_scan(self, query: SearchQuery) -> bool:
        return query.mode == "regex" or query.ignore_spaces or query.ignore_hyphens

    def _stream_rows(
        self,
        sql: str,
        params: list[Any],
        token: CancelToken,
        *,
        batch_size: int = 512,
    ) -> Iterator[sqlite3.Row]:
        with self.db.connect() as con:
            cursor = con.execute(sql, params)
            while True:
                token.throw_if_cancelled()
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    return
                for row in rows:
                    token.throw_if_cancelled()
                    yield row

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
            context = make_context(raw_text, raw_terms)
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
        hit = SearchHit(
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            location_text=str(row["location_text"] or ""),
            context=context,
            hit_count=max(hits, 1),
            source_type=str(row["source_type"] or "native_text"),
            ocr_confidence=confidence,
            is_fuzzy=is_fuzzy,
        )
        return SearchResult(
            file_id=int(row["file_id"]),
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            file_path=str(row["path"]),
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
