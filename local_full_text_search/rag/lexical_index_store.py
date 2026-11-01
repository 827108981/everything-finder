from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from local_full_text_search.rag.errors import LexicalIndexNotReadyError
from local_full_text_search.rag.index_store import RagBundle, RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, RagQuery
from local_full_text_search.rag.retrieval.lexical_analyzer import (
    LexicalAnalyzer, normalize_lexical_text,
)
from local_full_text_search.rag.retrieval.lexical_models import LexicalTermContribution
from local_full_text_search.rag.retrieval.lexical_scoring import (
    SCORE_KIND, bm25_term_score, inverse_document_frequency,
)


LEXICAL_SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class LexicalIndexBundle:
    source_bundle_id: str
    source_content_digest: str
    analyzer_fingerprint: str
    index_fingerprint: str
    path: Path
    database_path: Path
    chunk_count: int
    created_at: str


class LexicalIndexStore:
    def __init__(self, rag_root: Path) -> None:
        self.root = Path(rag_root).resolve() / "lexical"

    def bundle_root(self, source_bundle_id: str) -> Path:
        return self.root / source_bundle_id

    def active(self, source_bundle_id: str, analyzer_fingerprint: str) -> LexicalIndexBundle:
        pointer = self.bundle_root(source_bundle_id) / "active.json"
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            if int(payload.get("schema_version", -1)) != LEXICAL_SCHEMA_VERSION:
                raise ValueError("lexical schema version mismatch")
            if payload.get("source_bundle_id") != source_bundle_id:
                raise ValueError("source bundle mismatch")
            if payload.get("analyzer_fingerprint") != analyzer_fingerprint:
                raise ValueError("analyzer fingerprint mismatch")
            path = (pointer.parent / str(payload["directory"])).resolve()
            if path.parent != pointer.parent.resolve():
                raise ValueError("invalid lexical index path")
            database_path = path / "lexical.db"
            connection = self.connect_read_only(database_path)
            try:
                meta = connection.execute("SELECT * FROM lexical_meta WHERE id=1").fetchone()
                if meta is None or meta["state"] != "published":
                    raise ValueError("lexical index is not published")
                if int(meta["schema_version"]) != LEXICAL_SCHEMA_VERSION:
                    raise ValueError("lexical database schema version mismatch")
                checks = {
                    "source_bundle_id": source_bundle_id,
                    "analyzer_fingerprint": analyzer_fingerprint,
                    "index_fingerprint": str(payload["index_fingerprint"]),
                }
                for key, expected in checks.items():
                    if str(meta[key]) != str(expected):
                        raise ValueError(f"{key} mismatch")
                quick = connection.execute("PRAGMA quick_check").fetchone()[0]
                if str(quick).lower() != "ok":
                    raise ValueError(f"quick_check={quick}")
            finally:
                connection.close()
            return LexicalIndexBundle(
                source_bundle_id=source_bundle_id,
                source_content_digest=str(payload["source_content_digest"]),
                analyzer_fingerprint=analyzer_fingerprint,
                index_fingerprint=str(payload["index_fingerprint"]),
                path=path,
                database_path=database_path,
                chunk_count=int(payload["chunk_count"]),
                created_at=str(payload.get("created_at") or ""),
            )
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise LexicalIndexNotReadyError(
                "关键词 V2 索引未就绪或与当前 AI 索引不一致；请先构建对应的关键词索引"
            ) from exc

    @staticmethod
    def connect_read_only(path: Path) -> sqlite3.Connection:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(
            f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True, timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def publish_pointer(bundle: LexicalIndexBundle) -> None:
        pointer = bundle.path.parent / "active.json"
        payload = {
            "schema_version": LEXICAL_SCHEMA_VERSION,
            "source_bundle_id": bundle.source_bundle_id,
            "source_content_digest": bundle.source_content_digest,
            "analyzer_fingerprint": bundle.analyzer_fingerprint,
            "index_fingerprint": bundle.index_fingerprint,
            "directory": bundle.path.name,
            "chunk_count": bundle.chunk_count,
            "created_at": bundle.created_at,
        }
        temporary = pointer.with_suffix(".json.tmp")
        with temporary.open("wb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)

    def search(
        self, bundle: LexicalIndexBundle, rag_store: RagIndexStore, rag_bundle: RagBundle,
        analyzer: LexicalAnalyzer, query: RagQuery, *, title_weight: float = 0.2,
        complete_match_enabled: bool = True, k1: float = 1.2, b: float = 0.75,
        diagnostics: dict[str, object] | None = None,
    ) -> list[EvidenceChunk]:
        started = time.monotonic()
        analysis = analyzer.analyze_query(query.text)
        if not analysis.tokens:
            if diagnostics is not None:
                diagnostics.update({"backend": "word_v2", "actual_terms": [], "candidate_count": 0})
            return []
        lexical = self.connect_read_only(bundle.database_path)
        metadata = rag_store.connect_read_only(rag_bundle.metadata_path)
        try:
            totals = lexical.execute(
                "SELECT COUNT(*) AS n, AVG(body_length) AS body_avg, AVG(title_length) AS title_avg "
                "FROM lexical_documents"
            ).fetchone()
            document_count = int(totals["n"] or 0)
            body_average = float(totals["body_avg"] or 1.0)
            title_average = float(totals["title_avg"] or 1.0)
            term_rows: dict[str, sqlite3.Row] = {}
            for token in analysis.tokens:
                row = lexical.execute(
                    "SELECT term_id,document_frequency FROM lexical_terms "
                    "WHERE term_kind='token' AND term_text=?", (token.normalized,),
                ).fetchone()
                if row is not None:
                    term_rows[token.normalized] = row

            active_phrases = []
            phrase_documents: dict[str, dict[str, set[str]]] = {}
            if complete_match_enabled:
                for phrase in analysis.locator_phrases:
                    located = self._locate_phrase_documents(metadata, phrase.normalized)
                    verified: dict[str, set[str]] = {}
                    for chunk_id, fields in located.items():
                        chunk = rag_store.load_chunk(metadata, chunk_id)
                        if chunk is None:
                            continue
                        values = {
                            "body": normalize_lexical_text(chunk.text),
                            "title": normalize_lexical_text(Path(chunk.source.filename).stem),
                        }
                        matched = {field for field in fields if phrase.normalized in values[field]}
                        if matched:
                            verified[chunk_id] = matched
                    if verified:
                        active_phrases.append(phrase)
                        phrase_documents[phrase.normalized] = verified

            candidate_ids: set[str] = set()
            if term_rows:
                placeholders = ",".join("?" for _ in term_rows)
                rows = lexical.execute(
                    f"SELECT DISTINCT d.stable_chunk_id FROM lexical_occurrences o "
                    "JOIN lexical_terms t ON t.term_id=o.term_id "
                    "JOIN lexical_documents d ON d.document_id=o.document_id "
                    f"WHERE t.term_text IN ({placeholders}) AND t.term_kind='token'",
                    tuple(term_rows),
                ).fetchall()
                candidate_ids.update(str(row[0]) for row in rows)
            for verified in phrase_documents.values():
                candidate_ids.update(verified)
            located_count = len(candidate_ids)
            documents = self._load_filtered_documents(lexical, candidate_ids, query)
            filtered_count = len(documents)

            occurrence_map: dict[str, dict[str, dict[str, int]]] = {}
            if documents and term_rows:
                doc_placeholders = ",".join("?" for _ in documents)
                term_placeholders = ",".join("?" for _ in term_rows)
                document_ids = [int(row["document_id"]) for row in documents.values()]
                rows = lexical.execute(
                    f"SELECT t.term_text,o.document_id,o.field,SUM(o.term_frequency) AS tf "
                    f"FROM lexical_occurrences o JOIN lexical_terms t ON t.term_id=o.term_id "
                    f"WHERE t.term_kind='token' AND t.term_text IN ({term_placeholders}) "
                    f"AND o.document_id IN ({doc_placeholders}) "
                    "GROUP BY t.term_text,o.document_id,o.field",
                    (*term_rows, *document_ids),
                ).fetchall()
                stable_by_id = {
                    int(row["document_id"]): stable_id for stable_id, row in documents.items()
                }
                for row in rows:
                    occurrence_map.setdefault(stable_by_id[int(row["document_id"])], {}).setdefault(
                        str(row["term_text"]), {}
                    )[str(row["field"])] = int(row["tf"])

            scored_rows: list[tuple[float, str, tuple[dict[str, object], ...]]] = []
            for chunk_id, document in documents.items():
                matched_phrases = [
                    phrase for phrase in active_phrases
                    if chunk_id in phrase_documents.get(phrase.normalized, {})
                    and any(
                        not occurrence_map.get(chunk_id, {}).get(component)
                        for component in phrase.component_terms
                    )
                ]
                body_norm = ""
                title_norm = ""
                if matched_phrases:
                    phrase_chunk = rag_store.load_chunk(metadata, chunk_id)
                    if phrase_chunk is None:
                        continue
                    body_norm = normalize_lexical_text(phrase_chunk.text)
                    title_norm = normalize_lexical_text(Path(phrase_chunk.source.filename).stem)
                suppressed = {
                    component for phrase in matched_phrases for component in phrase.component_terms
                }
                contributions: list[LexicalTermContribution] = []
                body_total = 0.0
                title_total = 0.0
                title_cap = 0.0
                for token in analysis.tokens:
                    term = token.normalized
                    if term in suppressed or term not in term_rows:
                        continue
                    frequencies = occurrence_map.get(chunk_id, {}).get(term, {})
                    body_tf, title_tf = frequencies.get("body", 0), frequencies.get("title", 0)
                    if not body_tf and not title_tf:
                        continue
                    df = int(term_rows[term]["document_frequency"])
                    body_score = bm25_term_score(
                        document_count=document_count, document_frequency=df,
                        term_frequency=body_tf, field_length=int(document["body_length"]),
                        average_field_length=body_average, k1=k1, b=b,
                    )
                    title_score = bm25_term_score(
                        document_count=document_count, document_frequency=df,
                        term_frequency=title_tf, field_length=int(document["title_length"]),
                        average_field_length=title_average, k1=k1, b=b,
                    )
                    body_total += body_score
                    title_total += title_score
                    title_cap += inverse_document_frequency(document_count, df)
                    contributions.append(LexicalTermContribution(
                        term, df, body_tf, title_tf, body_score,
                        min(title_score, inverse_document_frequency(document_count, df)),
                        ("token",),
                    ))
                for phrase in matched_phrases:
                    fields = phrase_documents[phrase.normalized][chunk_id]
                    body_tf = body_norm.count(phrase.normalized) if "body" in fields else 0
                    title_tf = title_norm.count(phrase.normalized) if "title" in fields else 0
                    df = len(phrase_documents[phrase.normalized])
                    body_score = bm25_term_score(
                        document_count=document_count, document_frequency=df,
                        term_frequency=body_tf, field_length=int(document["body_length"]),
                        average_field_length=body_average, k1=k1, b=b,
                    )
                    title_score = bm25_term_score(
                        document_count=document_count, document_frequency=df,
                        term_frequency=title_tf, field_length=int(document["title_length"]),
                        average_field_length=title_average, k1=k1, b=b,
                    )
                    body_total += body_score
                    title_total += title_score
                    title_cap += inverse_document_frequency(document_count, df)
                    contributions.append(LexicalTermContribution(
                        phrase.normalized, df, body_tf, title_tf, body_score,
                        min(title_score, inverse_document_frequency(document_count, df)),
                        ("complete_text_locator",),
                    ))
                score = body_total + float(title_weight) * min(title_total, title_cap)
                if score <= 0:
                    continue
                scored_rows.append((
                    score, chunk_id, tuple(item.to_dict() for item in contributions),
                ))
            scored_rows.sort(key=lambda item: (-item[0], item[1]))
            results: list[EvidenceChunk] = []
            for score, chunk_id, contributions in scored_rows[:max(1, int(query.lexical_top_k))]:
                chunk = rag_store.load_chunk(metadata, chunk_id)
                if chunk is None:
                    continue
                results.append(EvidenceChunk(
                    chunk=chunk, lexical_rank=len(results) + 1, lexical_score=score,
                    lexical_score_kind=SCORE_KIND, lexical_contributions=contributions,
                ))
            if diagnostics is not None:
                diagnostics.update({
                    "backend": "word_v2",
                    "score_kind": SCORE_KIND,
                    "score_direction": "higher_is_better",
                    "source_bundle_id": bundle.source_bundle_id,
                    "lexical_index_fingerprint": bundle.index_fingerprint,
                    "analyzer_fingerprint": analysis.analyzer_fingerprint,
                    "actual_terms": [token.normalized for token in analysis.tokens],
                    "duplicate_terms_removed": list(analysis.duplicate_terms),
                    "complete_match_terms": [phrase.normalized for phrase in active_phrases],
                    "located_candidate_count": located_count,
                    "filtered_candidate_count": filtered_count,
                    "scored_candidate_count": len(scored_rows),
                    "candidate_count": len(results),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "title_weight": float(title_weight),
                    "k1": float(k1), "b": float(b),
                })
            return results
        finally:
            metadata.close()
            lexical.close()

    @staticmethod
    def _load_filtered_documents(
        connection: sqlite3.Connection, candidate_ids: set[str], query: RagQuery,
    ) -> dict[str, sqlite3.Row]:
        if not candidate_ids:
            return {}
        output: dict[str, sqlite3.Row] = {}
        ids = sorted(candidate_ids)
        for offset in range(0, len(ids), 500):
            batch = ids[offset:offset + 500]
            placeholders = ",".join("?" for _ in batch)
            clauses = [f"stable_chunk_id IN ({placeholders})"]
            parameters: list[object] = list(batch)
            if query.root_ids:
                root_marks = ",".join("?" for _ in query.root_ids)
                clauses.append(f"root_id IN ({root_marks})")
                parameters.extend(query.root_ids)
            if query.extensions:
                extension_marks = ",".join("?" for _ in query.extensions)
                clauses.append(f"LOWER(extension) IN ({extension_marks})")
                parameters.extend(item.lower() for item in query.extensions)
            if not query.include_ocr:
                clauses.append("source_type<>'ocr'")
            else:
                clauses.append("(source_type<>'ocr' OR ocr_confidence IS NULL OR ocr_confidence>=?)")
                parameters.append(float(query.min_ocr_confidence))
            rows = connection.execute(
                "SELECT * FROM lexical_documents WHERE " + " AND ".join(clauses), parameters,
            ).fetchall()
            output.update({str(row["stable_chunk_id"]): row for row in rows})
        return output

    @staticmethod
    def _locate_phrase_documents(
        connection: sqlite3.Connection, phrase: str,
    ) -> dict[str, set[str]]:
        # 现有 AI metadata.db 已有正文 trigram FTS。这里只把它当候选定位器，
        # 随后仍回到原文核验完整连续字符串，不采用旧 BM25 分数。
        escaped = phrase.replace('"', '""')
        rows = connection.execute(
            "SELECT stable_chunk_id FROM rag_chunks_fts "
            "WHERE rag_chunks_fts MATCH ? ORDER BY stable_chunk_id",
            (f'normalized_text : "{escaped}"',),
        ).fetchall()
        return {str(row[0]): {"body"} for row in rows}
