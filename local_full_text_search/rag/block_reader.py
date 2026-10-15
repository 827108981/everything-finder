from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_generation_store import (
    IndexGenerationStore,
    PublishedGeneration,
)
from local_full_text_search.rag.models import (
    FileSource,
    NoPublishedGenerationError,
    SourceBlock,
    UnsupportedSnapshotError,
)

_DEFAULT_EXTENSIONS = frozenset({".pdf", ".docx", ".txt"})
_KNOWN_BLOCK_KINDS = (
    "pdf_page",
    "pdf_page_ocr",
    "docx_paragraph",
    "docx_table_row",
    "docx_header_footer",
    "text",
)


def _compact_kind_code(value: str) -> int:
    digest = hashlib.blake2s(value.encode("utf-8"), digest_size=2).digest()
    return max(1, int.from_bytes(digest, "big"))


def _compact_kind_map() -> dict[int, str]:
    result: dict[int, str] = {}
    for value in _KNOWN_BLOCK_KINDS:
        code = _compact_kind_code(value)
        if code in result:
            raise RuntimeError(f"compact block kind code collision: {value}")
        result[code] = value
    return result


_COMPACT_KIND_NAMES = _compact_kind_map()


@dataclass(frozen=True, slots=True)
class DocumentBlocks:
    generation: int
    document_id: int
    files: tuple[FileSource, ...]
    blocks: tuple[SourceBlock, ...]

    def __post_init__(self) -> None:
        if self.generation <= 0 or self.document_id <= 0:
            raise ValueError("generation and document_id must be positive")
        if not self.files or not self.blocks:
            raise ValueError("document must contain at least one file and one block")
        if any(block.document_id != self.document_id for block in self.blocks):
            raise ValueError("all blocks must belong to document_id")
        if any(block.generation != self.generation for block in self.blocks):
            raise ValueError("all blocks must belong to generation")


class BlockReader:
    """Read source blocks from the active immutable search generation."""

    def __init__(self, generation_base_path: Path) -> None:
        self.store = IndexGenerationStore(Path(generation_base_path))

    @contextmanager
    def open_active(self) -> Iterator["ActiveBlockReader"]:
        try:
            with self.store.acquire_active() as (active, _lease):
                if not active.path.is_file():
                    raise NoPublishedGenerationError("published generation file disappeared")
                database = DatabaseManager(active.path, read_only=True)
                with database.connect_read_only(timeout_seconds=2.0, operation="rag_block_reader") as con:
                    con.execute("BEGIN")
                    yield ActiveBlockReader(active, con)
        except FileNotFoundError as exc:
            raise NoPublishedGenerationError(str(exc) or "no published generation") from exc
        except sqlite3.Error as exc:
            raise UnsupportedSnapshotError(f"cannot read published generation: {exc}") from exc


class ActiveBlockReader:
    def __init__(self, active: PublishedGeneration, connection: sqlite3.Connection) -> None:
        self.active = active
        self.connection = connection
        self.generation = int(active.generation)
        self.compact = self._has_object("view", "search_blocks")
        self._assert_generation()
        if self.compact:
            self.block_source = "search_blocks"
        elif self._has_object("table", "content_blocks"):
            self.block_source = "content_blocks"
        else:
            raise UnsupportedSnapshotError("published database has no readable content block table")

    def iter_documents(
        self,
        *,
        file_ids: set[int] | None = None,
        extensions: frozenset[str] = _DEFAULT_EXTENSIONS,
        limit: int | None = None,
        batch_size: int = 128,
    ) -> Iterator[DocumentBlocks]:
        normalized_extensions = frozenset(
            ext.lower() if ext.startswith(".") else f".{ext.lower()}"
            for ext in extensions
            if ext
        )
        selected_file_ids = sorted({int(value) for value in file_ids or ()})
        if file_ids is not None and not selected_file_ids:
            return
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        remaining = None if limit is None else max(0, int(limit))
        if remaining == 0:
            return
        last_document_id = -1

        while remaining is None or remaining > 0:
            page_size = batch_size if remaining is None else min(batch_size, remaining)
            doc_sql, doc_params = self._document_page_sql(
                selected_file_ids, normalized_extensions, page_size
            )
            rows = self.connection.execute(
                doc_sql, (last_document_id, *doc_params, page_size)
            ).fetchall()
            if not rows:
                return
            document_ids = [int(row[0]) for row in rows]
            last_document_id = document_ids[-1]
            files_by_document = self._read_files(document_ids, selected_file_ids, normalized_extensions)
            for document_id, blocks in self._iter_block_groups(document_ids):
                files = files_by_document.get(document_id, ())
                if not files or not blocks:
                    continue
                yield DocumentBlocks(
                    generation=self.generation,
                    document_id=document_id,
                    files=files,
                    blocks=tuple(blocks),
                )
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return

    def _document_page_sql(
        self,
        file_ids: list[int],
        extensions: frozenset[str],
        page_size: int,
    ) -> tuple[str, tuple[object, ...]]:
        extension_marks = ",".join("?" for _ in extensions)
        clauses = [
            "f.document_id > ?",
            "f.document_id IS NOT NULL",
            "r.enabled = 1",
            "f.parse_status IN ('success', 'metadata_only')",
            "f.container_file_id IS NULL",
            f"LOWER(f.extension) IN ({extension_marks})" if extensions else "0",
        ]
        params: list[object] = list(sorted(extensions))
        if file_ids:
            clauses.append(f"f.id IN ({','.join('?' for _ in file_ids)})")
            params.extend(file_ids)
        return (
            "SELECT DISTINCT f.document_id FROM files AS f "
            "JOIN roots AS r ON r.id = f.root_id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY f.document_id LIMIT ?",
            tuple(params),
        )

    def _read_files(
        self,
        document_ids: list[int],
        file_ids: list[int],
        extensions: frozenset[str],
    ) -> dict[int, tuple[FileSource, ...]]:
        clauses = [
            f"f.document_id IN ({','.join('?' for _ in document_ids)})",
            "r.enabled = 1",
            "f.parse_status IN ('success', 'metadata_only')",
            "f.container_file_id IS NULL",
            f"LOWER(f.extension) IN ({','.join('?' for _ in extensions)})" if extensions else "0",
        ]
        params: list[object] = [*document_ids, *sorted(extensions)]
        if file_ids:
            clauses.append(f"f.id IN ({','.join('?' for _ in file_ids)})")
            params.extend(file_ids)
        rows = self.connection.execute(
            "SELECT f.id, f.root_id, f.document_id, f.path, f.filename, f.extension, "
            "f.parser_name, f.parser_version FROM files AS f "
            "JOIN roots AS r ON r.id = f.root_id WHERE "
            + " AND ".join(clauses)
            + " ORDER BY f.document_id, f.id",
            tuple(params),
        ).fetchall()
        mutable: dict[int, list[FileSource]] = {}
        for row in rows:
            document_id = int(row["document_id"])
            mutable.setdefault(document_id, []).append(
                FileSource(
                    file_id=int(row["id"]),
                    root_id=int(row["root_id"]),
                    file_path=str(row["path"] or ""),
                    filename=str(row["filename"] or ""),
                    extension=str(row["extension"] or "").lower(),
                    parser_name=str(row["parser_name"] or ""),
                    parser_version=str(row["parser_version"] or ""),
                )
            )
        return {key: tuple(value) for key, value in mutable.items()}

    def _iter_block_groups(
        self,
        document_ids: list[int],
    ) -> Iterator[tuple[int, list[SourceBlock]]]:
        if not document_ids:
            return
        if self.compact:
            select = (
                "id, document_id, block_index, block_type, raw_text, location_text, "
                "page_number, slide_number, sheet_name, cell_start, cell_end, "
                "line_start, line_end, source_type, ocr_confidence"
            )
        else:
            select = (
                "id, document_id, block_index, block_type, raw_text, location_text, "
                "page_number, slide_number, sheet_name, cell_start, cell_end, "
                "line_start, line_end, source_type, ocr_confidence"
            )
        rows = self.connection.execute(
            f"SELECT {select} FROM {self.block_source} "
            f"WHERE document_id IN ({','.join('?' for _ in document_ids)}) "
            "AND (source_type IS NULL OR source_type != 'metadata') "
            "ORDER BY document_id, block_index, id",
            tuple(document_ids),
        )
        current_document_id: int | None = None
        current_blocks: list[SourceBlock] = []
        for row in rows:
            document_id = int(row["document_id"])
            if current_document_id is not None and document_id != current_document_id:
                yield current_document_id, current_blocks
                current_blocks = []
            current_document_id = document_id
            block = self._source_block_from_row(row)
            if block is not None:
                current_blocks.append(block)
        if current_document_id is not None:
            yield current_document_id, current_blocks

    def _source_block_from_row(self, row: sqlite3.Row) -> SourceBlock | None:
        raw_text = str(row["raw_text"] or "")
        if not raw_text.strip():
            return None
        kind_value = row["block_type"]
        if self.compact:
            try:
                block_kind = _COMPACT_KIND_NAMES.get(int(kind_value), "unknown")
            except (TypeError, ValueError):
                block_kind = "unknown"
        else:
            block_kind = str(kind_value or "unknown")
        confidence = row["ocr_confidence"]
        confidence_value = float(confidence) if confidence is not None else None
        if confidence_value is not None and self.compact:
            confidence_value /= 10000.0
        if confidence_value is not None:
            confidence_value = min(1.0, max(0.0, confidence_value))
        return SourceBlock(
            generation=self.generation,
            document_id=int(row["document_id"]),
            block_id=int(row["id"]),
            block_index=int(row["block_index"] or 0),
            block_kind=block_kind,
            raw_text=raw_text,
            location_text=str(row["location_text"] or ""),
            page_number=_optional_int(row["page_number"]),
            slide_number=_optional_int(row["slide_number"]),
            sheet_name=_optional_str(row["sheet_name"]),
            cell_start=_optional_str(row["cell_start"]),
            cell_end=_optional_str(row["cell_end"]),
            line_start=_optional_int(row["line_start"]),
            line_end=_optional_int(row["line_end"]),
            source_type=str(row["source_type"] or "native_text"),
            ocr_confidence=confidence_value,
        )

    def _has_object(self, object_type: str, name: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, name),
        ).fetchone()
        return row is not None

    def _assert_generation(self) -> None:
        if not self._has_object("table", "snapshot_meta"):
            # IndexGenerationStore validates published files against this marker;
            # this guard provides a direct, readable error for an unsupported schema.
            raise UnsupportedSnapshotError("published database has no snapshot_meta table")
        row = self.connection.execute(
            "SELECT generation, state FROM snapshot_meta WHERE id = 1"
        ).fetchone()
        if row is None or int(row["generation"]) != self.generation or int(row["state"]) != 1:
            raise UnsupportedSnapshotError("manifest and database generation do not match")


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _optional_str(value: object) -> str | None:
    return str(value) if value is not None else None
