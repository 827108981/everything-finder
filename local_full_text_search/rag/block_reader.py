from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

from local_full_text_search.core.text_blob_codec import decode_text_blob
from local_full_text_search.rag.models import SourceBlock


class BlockReader:
    """Reads published, already-parsed blocks without touching the control DB."""

    def iter_blocks(
        self,
        database_path: Path,
        generation: int,
        *,
        include_ocr: bool = True,
        min_ocr_confidence: float = 0.60,
    ) -> Iterator[SourceBlock]:
        path = Path(database_path).resolve()
        uri = f"file:{path.as_posix()}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.create_function("decode_text", 2, decode_text_blob)
        try:
            if not self._has_search_blocks(connection):
                raise RuntimeError("当前正式索引不包含可供 RAG 使用的 search_blocks 视图")
            sql = """
                SELECT
                    sb.id AS block_id,
                    sb.file_id,
                    sb.document_id,
                    sb.block_index,
                    sb.raw_text,
                    sb.location_text,
                    sb.page_number,
                    sb.slide_number,
                    sb.sheet_name,
                    sb.cell_start,
                    sb.cell_end,
                    sb.line_start,
                    sb.line_end,
                    sb.source_type,
                    sb.ocr_confidence,
                    f.root_id,
                    f.path AS file_path,
                    f.filename,
                    f.extension,
                    f.content_hash
                FROM search_blocks sb
                JOIN files f ON f.id = sb.file_id
                WHERE sb.raw_text IS NOT NULL AND length(trim(sb.raw_text)) > 0
                ORDER BY COALESCE(sb.document_id, -sb.file_id),
                         COALESCE(sb.page_number, sb.slide_number, 0),
                         COALESCE(sb.sheet_name, ''), sb.block_index
            """
            for row in connection.execute(sql):
                source_type = str(row["source_type"] or "native_text")
                confidence = row["ocr_confidence"]
                if source_type == "ocr":
                    # Compact snapshots store OCR confidence as an integer
                    # scaled by 10,000; older operational DBs store 0.0..1.0.
                    if confidence is not None and float(confidence) > 1.0:
                        confidence = float(confidence) / 10_000.0
                    if not include_ocr:
                        continue
                    if confidence is not None and float(confidence) < min_ocr_confidence:
                        continue
                file_path = str(row["file_path"] or "")
                document_id = row["document_id"]
                content_hash = row["content_hash"]
                document_key = str(content_hash or document_id or file_path)
                yield SourceBlock(
                    source_generation=int(generation),
                    block_id=int(row["block_id"]),
                    document_id=int(document_id) if document_id is not None else None,
                    file_id=int(row["file_id"]),
                    root_id=int(row["root_id"]),
                    file_path=file_path,
                    filename=str(row["filename"] or Path(file_path).name),
                    extension=str(row["extension"] or Path(file_path).suffix).lower(),
                    document_key=document_key,
                    block_index=int(row["block_index"] or 0),
                    text=str(row["raw_text"] or ""),
                    location_text=str(row["location_text"] or ""),
                    page_number=self._int_or_none(row["page_number"]),
                    slide_number=self._int_or_none(row["slide_number"]),
                    sheet_name=str(row["sheet_name"]) if row["sheet_name"] else None,
                    cell_start=str(row["cell_start"]) if row["cell_start"] else None,
                    cell_end=str(row["cell_end"]) if row["cell_end"] else None,
                    line_start=self._int_or_none(row["line_start"]),
                    line_end=self._int_or_none(row["line_end"]),
                    source_type=source_type,
                    ocr_confidence=float(confidence) if confidence is not None else None,
                )
        finally:
            connection.close()

    @staticmethod
    def _has_search_blocks(connection: sqlite3.Connection) -> bool:
        return connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='view' AND name='search_blocks'"
        ).fetchone() is not None

    @staticmethod
    def _int_or_none(value: object) -> int | None:
        return int(value) if value is not None else None
