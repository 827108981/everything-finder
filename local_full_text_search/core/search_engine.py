from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Any

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.normalizer import count_hits, make_context, normalize_text, parse_terms
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.models.search_result import SearchResult


@dataclass(slots=True)
class SearchPage:
    results: list[SearchResult]
    total_candidates: int
    total_confirmed: int
    elapsed_ms: int
    page: int
    page_size: int


class SearchEngine:
    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def search(self, query: SearchQuery, cancel_token: CancelToken | None = None) -> SearchPage:
        started = time.perf_counter()
        token = cancel_token or CancelToken()
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
            return SearchPage([], 0, 0, 0, query.page, query.page_size)
        if query.mode == "filename":
            rows = self._filename_candidates(query, normalized_terms, token)
        else:
            rows = self._content_candidates(query, normalized_terms, token)
        confirmed: list[SearchResult] = []
        for row in rows:
            token.throw_if_cancelled()
            if self._matches(row, normalized_terms, query.mode):
                result = self._row_to_result(row, raw_terms, normalized_terms, query)
                confirmed.append(result)
        confirmed.sort(key=lambda item: (-item.score, -item.modified_time, item.filename.lower()))
        total_confirmed = len(confirmed)
        page = max(1, query.page)
        start = (page - 1) * query.page_size
        end = start + query.page_size
        elapsed = int((time.perf_counter() - started) * 1000)
        return SearchPage(confirmed[start:end], len(rows), total_confirmed, elapsed, page, query.page_size)

    def _content_candidates(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
    ) -> list[sqlite3.Row]:
        where: list[str] = ["f.is_deleted = 0"]
        params: list[Any] = []
        text_predicates = self._like_predicates(query, normalized_terms, params)
        where.append("(" + text_predicates + ")")
        self._append_filters(where, params, query)
        if not query.include_ocr:
            where.append("(cb.source_type IS NULL OR cb.source_type != 'ocr')")
        sql = f"""
            SELECT
                cb.id AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, cb.location_text,
                cb.raw_text, cb.normalized_text, cb.source_type, cb.ocr_confidence
            FROM content_fts ft
            JOIN content_blocks cb ON cb.id = CAST(ft.block_id AS INTEGER)
            JOIN files f ON f.id = cb.file_id
            WHERE {' AND '.join(where)}
            LIMIT ?
        """
        params.append(max(query.page_size * 50, 5000))
        with self.db.connect() as con:
            token.throw_if_cancelled()
            return list(con.execute(sql, params).fetchall())

    def _filename_candidates(
        self,
        query: SearchQuery,
        normalized_terms: list[str],
        token: CancelToken,
    ) -> list[sqlite3.Row]:
        where: list[str] = ["f.is_deleted = 0"]
        params: list[Any] = []
        pieces: list[str] = []
        for term in normalized_terms:
            if query.mode == "any":
                op = "OR"
            else:
                op = "AND"
            field_parts: list[str] = []
            field_parts.append("LOWER(f.filename) LIKE ?")
            params.append(f"%{term}%")
            if query.search_path:
                field_parts.append("LOWER(f.path) LIKE ?")
                params.append(f"%{term}%")
            pieces.append("(" + " OR ".join(field_parts) + ")")
        where.append((" OR " if query.mode == "any" else " AND ").join(pieces))
        self._append_filters(where, params, query)
        sql = f"""
            SELECT
                NULL AS block_id, f.id AS file_id, f.path, f.filename, f.extension,
                f.size_bytes, f.modified_time, f.parse_status, '文件名/路径' AS location_text,
                f.filename || ' ' || f.path AS raw_text,
                LOWER(f.filename || ' ' || f.path) AS normalized_text,
                'metadata' AS source_type, NULL AS ocr_confidence
            FROM files f
            WHERE {' AND '.join(where)}
            LIMIT ?
        """
        params.append(max(query.page_size * 50, 5000))
        with self.db.connect() as con:
            token.throw_if_cancelled()
            return list(con.execute(sql, params).fetchall())

    def _like_predicates(self, query: SearchQuery, normalized_terms: list[str], params: list[Any]) -> str:
        glue = " OR " if query.mode == "any" else " AND "
        term_predicates: list[str] = []
        for term in normalized_terms:
            fields: list[str] = []
            if query.search_content:
                fields.append("ft.normalized_text LIKE ?")
                params.append(f"%{term}%")
            if query.search_filename:
                fields.append("ft.filename LIKE ?")
                params.append(f"%{term}%")
            if query.search_path:
                fields.append("ft.path LIKE ?")
                params.append(f"%{term}%")
            if not fields:
                fields.append("ft.normalized_text LIKE ?")
                params.append(f"%{term}%")
            term_predicates.append("(" + " OR ".join(fields) + ")")
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

    def _matches(self, row: sqlite3.Row, terms: list[str], mode: str) -> bool:
        text = str(row["normalized_text"] or "")
        if mode in {"exact", "phrase", "filename"}:
            return terms[0] in text
        if mode == "all":
            return all(term in text for term in terms)
        if mode == "any":
            return any(term in text for term in terms)
        if mode == "regex":
            try:
                return re.search(terms[0], text) is not None
            except re.error:
                return False
        return terms[0] in text

    def _row_to_result(
        self,
        row: sqlite3.Row,
        raw_terms: list[str],
        normalized_terms: list[str],
        query: SearchQuery,
    ) -> SearchResult:
        normalized_text = str(row["normalized_text"] or "")
        hits = count_hits(normalized_text, normalized_terms)
        filename_norm = normalize_text(str(row["filename"]), case_sensitive=query.case_sensitive)
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
        return SearchResult(
            file_id=int(row["file_id"]),
            block_id=int(row["block_id"]) if row["block_id"] is not None else None,
            file_path=str(row["path"]),
            filename=str(row["filename"]),
            extension=str(row["extension"] or ""),
            size_bytes=int(row["size_bytes"] or 0),
            modified_time=float(row["modified_time"] or 0),
            location_text=str(row["location_text"] or ""),
            context=make_context(str(row["raw_text"] or ""), raw_terms),
            hit_count=max(hits, 1),
            source_type=str(row["source_type"] or "native_text"),
            parse_status=str(row["parse_status"] or ""),
            score=score,
            ocr_confidence=float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None,
        )
