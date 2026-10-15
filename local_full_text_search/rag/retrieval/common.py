from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.rag.models import (
    EvidenceChunk,
    RagChunk,
    RagQuery,
    SourceRef,
    SourceSpan,
)


def source_filter_sql(query: RagQuery, source_alias: str = "s") -> tuple[str, tuple[object, ...]]:
    clauses = ["1=1"]
    params: list[object] = []
    if query.root_ids:
        clauses.append(f"{source_alias}.root_id IN ({','.join('?' for _ in query.root_ids)})")
        params.extend(query.root_ids)
    extensions = tuple(
        sorted({ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in query.extensions})
    )
    if extensions:
        clauses.append(f"LOWER({source_alias}.extension) IN ({','.join('?' for _ in extensions)})")
        params.extend(extensions)
    else:
        clauses.append("0")
    if not query.include_ocr:
        clauses.append(f"{source_alias}.source_type != 'ocr'")
    else:
        clauses.append(
            f"({source_alias}.source_type != 'ocr' OR "
            f"COALESCE({source_alias}.ocr_confidence, 0) >= ?)"
        )
        params.append(float(query.min_ocr_confidence))
    return " AND ".join(clauses), tuple(params)


def allowed_source_exists(query: RagQuery, chunk_alias: str = "c") -> tuple[str, tuple[object, ...]]:
    predicates, params = source_filter_sql(query, "s")
    return (
        f"EXISTS (SELECT 1 FROM rag_sources s WHERE s.chunk_rowid={chunk_alias}.chunk_rowid AND {predicates})",
        params,
    )


def load_evidence(connection, chunk_rowid: int, query: RagQuery) -> EvidenceChunk | None:
    chunk_row = connection.execute(
        "SELECT chunk_rowid,stable_chunk_id,text,normalized_text,text_hash,chunk_policy_version "
        "FROM rag_chunks WHERE chunk_rowid=?",
        (int(chunk_rowid),),
    ).fetchone()
    if chunk_row is None:
        return None
    predicates, params = source_filter_sql(query, "s")
    rows = connection.execute(
        "SELECT s.* FROM rag_sources s WHERE s.chunk_rowid=? AND "
        f"{predicates} ORDER BY s.root_id,s.file_id",
        (int(chunk_rowid), *params),
    ).fetchall()
    sources: list[SourceRef] = []
    for row in rows:
        source_path = str(row["file_path"] or "")
        try:
            if not Path(source_path).is_file():
                continue
        except OSError:
            continue
        source = SourceRef(
            generation=_source_generation(connection),
            stable_chunk_id=str(chunk_row["stable_chunk_id"]),
            file_id=int(row["file_id"]),
            root_id=int(row["root_id"]),
            document_id=int(row["document_id"]),
            file_path=source_path,
            filename=str(row["filename"] or ""),
            extension=str(row["extension"] or "").lower(),
            spans=tuple(SourceSpan(**item) for item in json.loads(str(row["spans_json"]))),
            block_index_start=min(int(item["block_index"]) for item in json.loads(str(row["spans_json"]))),
            block_index_end=max(int(item["block_index"]) for item in json.loads(str(row["spans_json"]))),
            location_text=str(row["location_text"] or ""),
            page_number=_optional_int(row["page_number"]),
            line_start=_optional_int(row["line_start"]),
            line_end=_optional_int(row["line_end"]),
            source_type=str(row["source_type"] or "native_text"),
            ocr_confidence=(float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None),
        )
        sources.append(source)
    if not sources:
        return None
    chunk = RagChunk(
        stable_chunk_id=str(chunk_row["stable_chunk_id"]),
        text=str(chunk_row["text"]),
        normalized_text=str(chunk_row["normalized_text"]),
        text_hash=str(chunk_row["text_hash"]),
        chunk_policy_version=str(chunk_row["chunk_policy_version"]),
        sources=tuple(sources),
    )
    return EvidenceChunk(chunk=chunk, selected_source=sources[0])


def _source_generation(connection) -> int:
    row = connection.execute("SELECT source_generation FROM rag_meta WHERE id=1").fetchone()
    if row is None:
        raise RuntimeError("RAG sidecar metadata missing")
    return int(row[0])


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None
