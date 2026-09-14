"""The intentionally small schema used by published search snapshots."""

from __future__ import annotations

import sqlite3
from pathlib import Path


COMPACT_SNAPSHOT_SCHEMA_VERSION = 2


def is_compact_snapshot(path: Path) -> bool:
    if not Path(path).is_file():
        return False
    try:
        con = sqlite3.connect(
            f"file:{Path(path).resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=1,
        )
        try:
            return con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='view' AND name='search_blocks'"
            ).fetchone() is not None
        finally:
            con.close()
    except sqlite3.Error:
        return False


def configure_new_database(connection: sqlite3.Connection) -> None:
    """Apply space-conscious settings before creating any table."""

    connection.executescript(
        """
        PRAGMA page_size = 4096;
        PRAGMA auto_vacuum = INCREMENTAL;
        PRAGMA encoding = 'UTF-8';
        PRAGMA foreign_keys = ON;
        """
    )


def create_compact_snapshot_schema(connection: sqlite3.Connection) -> None:
    """Create a search-only schema with no task or diagnostic history."""

    connection.executescript(
        """
        CREATE TABLE snapshot_meta (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            generation INTEGER NOT NULL,
            schema_version INTEGER NOT NULL,
            published_at INTEGER NOT NULL,
            root_count INTEGER NOT NULL,
            file_count INTEGER NOT NULL,
            document_count INTEGER NOT NULL,
            block_count INTEGER NOT NULL,
            parser_version TEXT NOT NULL,
            state INTEGER NOT NULL
        );

        CREATE TABLE roots (
            id INTEGER PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            include_subfolders INTEGER NOT NULL,
            enabled INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'ready'
        );

        CREATE TABLE extensions (
            id INTEGER PRIMARY KEY,
            value TEXT NOT NULL UNIQUE
        );

        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            content_hash BLOB NOT NULL,
            parser_kind INTEGER NOT NULL,
            parser_version INTEGER NOT NULL,
            block_count INTEGER NOT NULL,
            text_chars INTEGER NOT NULL
        );

        CREATE TABLE files (
            id INTEGER PRIMARY KEY,
            root_id INTEGER NOT NULL,
            document_id INTEGER,
            path TEXT NOT NULL UNIQUE,
            filename TEXT NOT NULL,
            extension TEXT,
            extension_id INTEGER,
            size_bytes INTEGER NOT NULL,
            modified_time REAL NOT NULL,
            modified_ns INTEGER NOT NULL,
            quick_fingerprint TEXT NOT NULL DEFAULT '',
            parser_name TEXT NOT NULL DEFAULT '',
            parser_version TEXT NOT NULL DEFAULT '',
            content_hash TEXT,
            parse_status TEXT NOT NULL,
            source_kind INTEGER NOT NULL DEFAULT 0,
            container_file_id INTEGER,
            member_order INTEGER,
            file_reference INTEGER,
            FOREIGN KEY(root_id) REFERENCES roots(id),
            FOREIGN KEY(document_id) REFERENCES documents(id),
            FOREIGN KEY(extension_id) REFERENCES extensions(id)
        );

        CREATE TABLE locations (
            id INTEGER PRIMARY KEY,
            document_id INTEGER NOT NULL,
            kind INTEGER NOT NULL,
            ordinal INTEGER,
            label TEXT,
            page_number INTEGER,
            slide_number INTEGER,
            sheet_name TEXT,
            cell_start TEXT,
            cell_end TEXT,
            line_start INTEGER,
            line_end INTEGER,
            range_start INTEGER,
            range_end INTEGER,
            UNIQUE(
                document_id, kind, ordinal, label, page_number, slide_number,
                sheet_name, cell_start, cell_end, line_start, line_end,
                range_start, range_end
            ),
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
        );

        CREATE TABLE content_blocks (
            id INTEGER PRIMARY KEY,
            file_id INTEGER NOT NULL,
            document_id INTEGER NOT NULL,
            location_id INTEGER,
            block_order INTEGER NOT NULL,
            block_kind INTEGER NOT NULL,
            source_kind INTEGER NOT NULL,
            codec INTEGER NOT NULL,
            raw_chars INTEGER NOT NULL,
            text_blob BLOB NOT NULL,
            ocr_confidence INTEGER,
            FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
            FOREIGN KEY(location_id) REFERENCES locations(id)
        );

        CREATE VIEW search_blocks AS
        SELECT
            cb.id,
            cb.file_id,
            cb.document_id,
            cb.block_order AS block_index,
            cb.block_kind AS block_type,
            decode_text(cb.text_blob, cb.codec) AS raw_text,
            '' AS normalized_text,
            CASE cb.source_kind
                WHEN 2 THEN 'ocr'
                WHEN 3 THEN 'metadata'
                ELSE 'native_text'
            END AS source_type,
            cb.ocr_confidence,
            loc.label AS location_text,
            loc.page_number,
            loc.slide_number,
            loc.sheet_name,
            loc.cell_start,
            loc.cell_end,
            loc.line_start,
            loc.line_end
        FROM content_blocks cb
        LEFT JOIN locations loc ON loc.id = cb.location_id;

        CREATE VIRTUAL TABLE content_fts USING fts5(
            normalized_text,
            content='',
            contentless_delete=1,
            detail=none,
            tokenize='trigram'
        );

        CREATE VIRTUAL TABLE files_fts USING fts5(
            normalized_path,
            content='',
            contentless_delete=1,
            detail=none,
            tokenize='trigram'
        );

        CREATE INDEX idx_files_root_ext ON files(root_id, extension_id, id);
        CREATE INDEX idx_files_document ON files(document_id, id);
        CREATE INDEX idx_blocks_document_order
            ON content_blocks(document_id, block_order);
        CREATE INDEX idx_locations_document
            ON locations(document_id, kind, ordinal);
        """
    )


def validate_compact_snapshot_schema(path: Path) -> dict[str, object]:
    """Read-only structural validation used by tests and diagnostics."""

    connection = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'shadow')"
            )
        }
        required = {
            "snapshot_meta",
            "roots",
            "extensions",
            "documents",
            "files",
            "locations",
            "content_blocks",
            "content_fts",
            "files_fts",
        }
        forbidden = {
            "short_tokens",
            "short_tokens_candidate",
            "content_fts_candidate",
            "parse_tasks",
            "index_runs",
            "search_history",
        }
        missing = sorted(required - tables)
        present_forbidden = sorted(forbidden & tables)
        return {
            "valid": not missing and not present_forbidden,
            "missing": missing,
            "forbidden": present_forbidden,
            "page_count": int(connection.execute("PRAGMA page_count").fetchone()[0]),
            "freelist_count": int(connection.execute("PRAGMA freelist_count").fetchone()[0]),
        }
    finally:
        connection.close()
