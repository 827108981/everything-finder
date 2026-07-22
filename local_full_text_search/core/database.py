from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

from local_full_text_search.config.constants import DB_PATH, PARSER_VERSION
from local_full_text_search.models.content_block import ContentBlock


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """Small SQLite gateway. Each call opens its own connection for thread safety."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        con.execute("PRAGMA synchronous = NORMAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def initialize(self) -> None:
        with self.connect() as con:
            self._create_schema(con)

    def _create_schema(self, con: sqlite3.Connection) -> None:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS roots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL UNIQUE,
                enabled INTEGER NOT NULL DEFAULT 1,
                include_subfolders INTEGER NOT NULL DEFAULT 1,
                exclude_rules_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_scan_at TEXT,
                status TEXT
            );

            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_id INTEGER NOT NULL,
                path TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                extension TEXT,
                size_bytes INTEGER,
                modified_time REAL,
                created_time REAL,
                quick_fingerprint TEXT,
                content_hash TEXT,
                parse_status TEXT NOT NULL,
                parse_error_code TEXT,
                parse_error_message TEXT,
                parser_name TEXT,
                parser_version TEXT,
                indexed_at TEXT,
                last_seen_at TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(root_id) REFERENCES roots(id)
            );

            CREATE TABLE IF NOT EXISTS content_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                block_index INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                location_text TEXT,
                page_number INTEGER,
                slide_number INTEGER,
                sheet_name TEXT,
                cell_start TEXT,
                cell_end TEXT,
                line_start INTEGER,
                line_end INTEGER,
                raw_text TEXT,
                normalized_text TEXT,
                source_type TEXT,
                ocr_confidence REAL,
                extra_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS short_tokens (
                token TEXT NOT NULL,
                block_id INTEGER NOT NULL,
                position_count INTEGER DEFAULT 1,
                PRIMARY KEY(token, block_id),
                FOREIGN KEY(block_id) REFERENCES content_blocks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS parse_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                error_message TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_files_root ON files(root_id);
            CREATE INDEX IF NOT EXISTS idx_files_ext ON files(extension);
            CREATE INDEX IF NOT EXISTS idx_files_status ON files(parse_status);
            CREATE INDEX IF NOT EXISTS idx_files_deleted ON files(is_deleted);
            CREATE INDEX IF NOT EXISTS idx_blocks_file ON content_blocks(file_id);
            """
        )
        self._ensure_fts(con)

    def _ensure_fts(self, con: sqlite3.Connection) -> None:
        try:
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    block_id UNINDEXED,
                    file_id UNINDEXED,
                    filename,
                    path,
                    location_text,
                    normalized_text,
                    tokenize='trigram'
                )
                """
            )
        except sqlite3.OperationalError:
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    block_id UNINDEXED,
                    file_id UNINDEXED,
                    filename,
                    path,
                    location_text,
                    normalized_text
                )
                """
            )

    def add_root(self, path: Path, include_subfolders: bool = True, enabled: bool = True) -> int:
        now = utc_now()
        with self.connect() as con:
            con.execute(
                """
                INSERT INTO roots(path, enabled, include_subfolders, exclude_rules_json, created_at, updated_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    enabled=excluded.enabled,
                    include_subfolders=excluded.include_subfolders,
                    updated_at=excluded.updated_at,
                    status='ready'
                """,
                (str(path), int(enabled), int(include_subfolders), "{}", now, now, "ready"),
            )
            row = con.execute("SELECT id FROM roots WHERE path = ?", (str(path),)).fetchone()
            return int(row["id"])

    def remove_root(self, root_id: int) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM content_fts WHERE file_id IN (SELECT id FROM files WHERE root_id = ?)", (root_id,))
            con.execute("DELETE FROM files WHERE root_id = ?", (root_id,))
            con.execute("DELETE FROM roots WHERE id = ?", (root_id,))

    def set_root_enabled(self, root_id: int, enabled: bool) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE roots SET enabled = ?, updated_at = ? WHERE id = ?",
                (int(enabled), utc_now(), root_id),
            )

    def list_roots(self, enabled_only: bool = False) -> list[sqlite3.Row]:
        sql = "SELECT * FROM roots"
        params: tuple[object, ...] = ()
        if enabled_only:
            sql += " WHERE enabled = 1"
        sql += " ORDER BY id"
        with self.connect() as con:
            return list(con.execute(sql, params).fetchall())

    def upsert_file_metadata(self, root_id: int, file_path: Path) -> tuple[int, bool]:
        stat = file_path.stat()
        fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
        now = utc_now()
        path_text = str(file_path)
        with self.connect() as con:
            existing = con.execute(
                "SELECT id, quick_fingerprint, parse_status, parser_version FROM files WHERE path = ?",
                (path_text,),
            ).fetchone()
            if existing is None:
                cur = con.execute(
                    """
                    INSERT INTO files(
                        root_id, path, filename, extension, size_bytes, modified_time, created_time,
                        quick_fingerprint, parse_status, parser_version, last_seen_at, is_deleted
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0)
                    """,
                    (
                        root_id,
                        path_text,
                        file_path.name,
                        file_path.suffix.lower(),
                        stat.st_size,
                        stat.st_mtime,
                        stat.st_ctime,
                        fingerprint,
                        PARSER_VERSION,
                        now,
                    ),
                )
                return int(cur.lastrowid), True
            changed = (
                existing["quick_fingerprint"] != fingerprint
                or existing["parser_version"] != PARSER_VERSION
                or existing["parse_status"]
                in {
                    "pending",
                    "processing",
                    "failed_retryable",
                    "cancelled",
                    "ocr_disabled",
                    "converter_missing",
                    "metadata_only",
                    "partial_success",
                }
            )
            con.execute(
                """
                UPDATE files SET
                    root_id = ?, filename = ?, extension = ?, size_bytes = ?, modified_time = ?,
                    created_time = ?, quick_fingerprint = ?, last_seen_at = ?, is_deleted = 0,
                    parse_status = CASE WHEN ? THEN 'pending' ELSE parse_status END
                WHERE id = ?
                """,
                (
                    root_id,
                    file_path.name,
                    file_path.suffix.lower(),
                    stat.st_size,
                    stat.st_mtime,
                    stat.st_ctime,
                    fingerprint,
                    now,
                    int(changed),
                    int(existing["id"]),
                ),
            )
            return int(existing["id"]), bool(changed)

    def mark_processing(self, file_id: int) -> None:
        with self.connect() as con:
            con.execute("UPDATE files SET parse_status = 'processing' WHERE id = ?", (file_id,))

    def replace_file_blocks(
        self,
        file_id: int,
        filename: str,
        path: str,
        blocks: Sequence[ContentBlock],
        *,
        parser_name: str,
        status: str = "success",
    ) -> None:
        now = utc_now()
        with self.connect() as con:
            con.execute("DELETE FROM content_fts WHERE file_id = ?", (file_id,))
            con.execute("DELETE FROM content_blocks WHERE file_id = ?", (file_id,))
            for block in blocks:
                cur = con.execute(
                    """
                    INSERT INTO content_blocks(
                        file_id, block_index, block_type, location_text, page_number, slide_number,
                        sheet_name, cell_start, cell_end, line_start, line_end, raw_text,
                        normalized_text, source_type, ocr_confidence, extra_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        block.block_index,
                        block.block_type,
                        block.location_text,
                        block.page_number,
                        block.slide_number,
                        block.sheet_name,
                        block.cell_start,
                        block.cell_end,
                        block.line_start,
                        block.line_end,
                        block.raw_text,
                        block.normalized_text,
                        block.source_type,
                        block.ocr_confidence,
                        json.dumps(block.extra or {}, ensure_ascii=False),
                        now,
                    ),
                )
                block_id = int(cur.lastrowid)
                con.execute(
                    """
                    INSERT INTO content_fts(rowid, block_id, file_id, filename, path, location_text, normalized_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (block_id, block_id, file_id, filename, path, block.location_text, block.normalized_text),
                )
                self._insert_short_tokens(con, block_id, block.normalized_text)
            con.execute(
                """
                UPDATE files SET
                    parse_status = ?, parse_error_code = NULL, parse_error_message = NULL,
                    parser_name = ?, parser_version = ?, indexed_at = ?, is_deleted = 0
                WHERE id = ?
                """,
                (status, parser_name, PARSER_VERSION, now, file_id),
            )

    def replace_file_blocks_many(
        self,
        items: Sequence[tuple[int, str, str, Sequence[ContentBlock], str, str, str | None, str | None]],
    ) -> None:
        """Replace blocks for multiple files in one SQLite transaction.

        Full-folder indexing creates thousands of tiny writes. Batching keeps
        SQLite in WAL mode but avoids paying connection/commit overhead for
        every single file.
        """

        if not items:
            return
        now = utc_now()
        with self.connect() as con:
            for file_id, filename, path, blocks, parser_name, status, error_code, error_message in items:
                con.execute("DELETE FROM content_fts WHERE file_id = ?", (file_id,))
                con.execute("DELETE FROM content_blocks WHERE file_id = ?", (file_id,))
                for block in blocks:
                    cur = con.execute(
                        """
                        INSERT INTO content_blocks(
                            file_id, block_index, block_type, location_text, page_number, slide_number,
                            sheet_name, cell_start, cell_end, line_start, line_end, raw_text,
                            normalized_text, source_type, ocr_confidence, extra_json, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            block.block_index,
                            block.block_type,
                            block.location_text,
                            block.page_number,
                            block.slide_number,
                            block.sheet_name,
                            block.cell_start,
                            block.cell_end,
                            block.line_start,
                            block.line_end,
                            block.raw_text,
                            block.normalized_text,
                            block.source_type,
                            block.ocr_confidence,
                            json.dumps(block.extra or {}, ensure_ascii=False),
                            now,
                        ),
                    )
                    block_id = int(cur.lastrowid)
                    con.execute(
                        """
                        INSERT INTO content_fts(rowid, block_id, file_id, filename, path, location_text, normalized_text)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (block_id, block_id, file_id, filename, path, block.location_text, block.normalized_text),
                    )
                    self._insert_short_tokens(con, block_id, block.normalized_text)
                con.execute(
                    """
                    UPDATE files SET
                        parse_status = ?, parse_error_code = ?, parse_error_message = ?,
                        parser_name = ?, parser_version = ?, indexed_at = ?, is_deleted = 0
                    WHERE id = ?
                    """,
                    (
                        status,
                        error_code,
                        (error_message[:1000] if error_message else None),
                        parser_name,
                        PARSER_VERSION,
                        now,
                        file_id,
                    ),
                )

    def _insert_short_tokens(self, con: sqlite3.Connection, block_id: int, normalized_text: str) -> None:
        tokens: dict[str, int] = {}
        compact = normalized_text.replace(" ", "")
        limit = min(len(compact), 5000)
        for i in range(limit):
            for width in (1, 2):
                token = compact[i : i + width]
                if len(token) == width and (any("\u4e00" <= ch <= "\u9fff" for ch in token) or token.isalnum()):
                    tokens[token] = tokens.get(token, 0) + 1
        con.executemany(
            "INSERT OR REPLACE INTO short_tokens(token, block_id, position_count) VALUES (?, ?, ?)",
            ((token, block_id, count) for token, count in tokens.items()),
        )

    def record_failure(
        self,
        file_id: int,
        error_code: str,
        message: str,
        *,
        parser_name: str | None = None,
        retryable: bool = False,
    ) -> None:
        status = "failed_retryable" if retryable else "failed"
        with self.connect() as con:
            con.execute(
                """
                UPDATE files SET parse_status = ?, parse_error_code = ?, parse_error_message = ?,
                    parser_name = ?, indexed_at = ?
                WHERE id = ?
                """,
                (status, error_code, message[:1000], parser_name, utc_now(), file_id),
            )

    def set_file_error_status(
        self,
        file_id: int,
        status: str,
        error_code: str,
        message: str,
        *,
        parser_name: str | None = None,
    ) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE files SET parse_status = ?, parse_error_code = ?, parse_error_message = ?,
                    parser_name = ?, indexed_at = ?
                WHERE id = ?
                """,
                (status, error_code, message[:1000], parser_name, utc_now(), file_id),
            )

    def mark_unsupported_with_metadata(self, file_id: int, filename: str, path: str, block: ContentBlock) -> None:
        self.replace_file_blocks(
            file_id,
            filename,
            path,
            [block],
            parser_name="metadata",
            status="unsupported",
        )

    def active_paths_for_root(self, root_id: int) -> set[str]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT path FROM files WHERE root_id = ? AND is_deleted = 0",
                (root_id,),
            ).fetchall()
            return {str(row["path"]) for row in rows}

    def mark_deleted_paths(self, paths: set[str]) -> int:
        if not paths:
            return 0
        with self.connect() as con:
            rows = con.execute(
                f"SELECT id FROM files WHERE path IN ({','.join('?' for _ in paths)})",
                tuple(paths),
            ).fetchall()
            file_ids = [int(row["id"]) for row in rows]
            if not file_ids:
                return 0
            con.execute(
                f"DELETE FROM content_fts WHERE file_id IN ({','.join('?' for _ in file_ids)})",
                tuple(file_ids),
            )
            con.execute(
                f"UPDATE files SET is_deleted = 1, parse_status = 'deleted' WHERE id IN ({','.join('?' for _ in file_ids)})",
                tuple(file_ids),
            )
            return len(file_ids)

    def update_root_scan_time(self, root_id: int, status: str = "ready") -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE roots SET last_scan_at = ?, status = ?, updated_at = ? WHERE id = ?",
                (utc_now(), status, utc_now(), root_id),
            )

    def failed_files(self, limit: int = 500) -> list[sqlite3.Row]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT path, extension, parse_status, parse_error_code, parse_error_message, parser_name, indexed_at
                    FROM files
                    WHERE parse_status IN (
                        'failed',
                        'failed_retryable',
                        'unsupported',
                        'skipped',
                        'metadata_only',
                        'ocr_disabled',
                        'ocr_failed',
                        'converter_missing',
                        'partial_success',
                        'password_protected'
                    )
                    ORDER BY indexed_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            )

    def stats(self) -> dict[str, int]:
        with self.connect() as con:
            total = con.execute("SELECT COUNT(*) AS n FROM files WHERE is_deleted = 0").fetchone()["n"]
            blocks = con.execute("SELECT COUNT(*) AS n FROM content_blocks").fetchone()["n"]
            failed = con.execute(
                "SELECT COUNT(*) AS n FROM files WHERE parse_status IN ('failed', 'failed_retryable') AND is_deleted = 0"
            ).fetchone()["n"]
            unsupported = con.execute(
                "SELECT COUNT(*) AS n FROM files WHERE parse_status = 'unsupported' AND is_deleted = 0"
            ).fetchone()["n"]
            metadata_only = con.execute(
                "SELECT COUNT(*) AS n FROM files WHERE parse_status IN ('metadata_only', 'skipped', 'ocr_disabled', 'converter_missing') AND is_deleted = 0"
            ).fetchone()["n"]
            return {
                "files": int(total),
                "blocks": int(blocks),
                "failed": int(failed),
                "unsupported": int(unsupported),
                "metadata_only": int(metadata_only),
            }
