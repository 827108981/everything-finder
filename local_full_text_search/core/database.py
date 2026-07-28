from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_full_text_search.config.constants import DB_PATH, PARSER_VERSION
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import FileTiming, IndexRunMetrics

SCHEMA_VERSION = 4
SUCCESSFUL_DOCUMENT_STATUSES = {
    "success",
    "partial_success",
    "metadata_only",
    "ocr_disabled",
    "converter_missing",
    "unsupported",
    "skipped",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """SQLite gateway with backward-compatible schema migrations."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connections_lock = threading.Lock()
        self._active_connections: set[sqlite3.Connection] = set()

    @contextmanager
    def connect(self, *, timeout_seconds: float = 30.0) -> Iterator[sqlite3.Connection]:
        timeout = max(0.0, float(timeout_seconds))
        busy_timeout_ms = max(0, int(timeout * 1000))
        con = sqlite3.connect(self.db_path, timeout=timeout)
        with self._connections_lock:
            self._active_connections.add(con)
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            con.execute("PRAGMA synchronous = NORMAL")
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            with self._connections_lock:
                self._active_connections.discard(con)
            con.close()

    def interrupt_active_connections(self) -> int:
        """Interrupt in-flight SQLite statements, primarily during forced cancellation."""

        with self._connections_lock:
            connections = tuple(self._active_connections)
        interrupted = 0
        for con in connections:
            try:
                con.interrupt()
                interrupted += 1
            except sqlite3.Error:
                # The connection may have completed and closed after the snapshot.
                continue
        return interrupted

    def initialize(self) -> None:
        previous_version = self._database_user_version()
        if previous_version is not None and previous_version < SCHEMA_VERSION:
            self._backup_legacy_database(previous_version)
        with self.connect() as con:
            con.execute("PRAGMA journal_mode = WAL")
            self._create_schema(con)
            self._migrate_schema_v3(con, previous_version)
            self._migrate_schema_v4(con, previous_version)
            self._ensure_fts(con)
            self._recover_interrupted_tasks(con)
            con.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        if previous_version is not None and previous_version < SCHEMA_VERSION:
            with self.connect() as con:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        if self._fts_is_dirty() and not self.has_incomplete_full_batch():
            self.rebuild_content_fts()

    def _database_user_version(self) -> int | None:
        if not self.db_path.is_file() or self.db_path.stat().st_size == 0:
            return None
        con = sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True, timeout=30)
        try:
            has_tables = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' LIMIT 1"
            ).fetchone()
            if has_tables is None:
                return None
            return int(con.execute("PRAGMA user_version").fetchone()[0])
        finally:
            con.close()

    def _backup_legacy_database(self, previous_version: int) -> Path:
        backup_path = self.db_path.with_name(
            f"{self.db_path.stem}.schema-v{previous_version}.backup{self.db_path.suffix}"
        )
        if backup_path.is_file():
            return backup_path
        temporary_path = backup_path.with_suffix(backup_path.suffix + ".tmp")
        temporary_path.unlink(missing_ok=True)
        source = sqlite3.connect(self.db_path, timeout=30)
        target = sqlite3.connect(temporary_path, timeout=30)
        try:
            source.backup(target)
            target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            target.commit()
        finally:
            target.close()
            source.close()
        os.replace(temporary_path, backup_path)
        return backup_path

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

            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_key TEXT NOT NULL,
                parser_name TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                parse_error_code TEXT,
                parse_error_message TEXT,
                block_count INTEGER NOT NULL DEFAULT 0,
                text_chars INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(content_key, parser_name, parser_version)
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
                content_key TEXT,
                document_id INTEGER,
                parse_status TEXT NOT NULL,
                parse_error_code TEXT,
                parse_error_message TEXT,
                parser_name TEXT,
                parser_version TEXT,
                indexed_at TEXT,
                last_seen_at TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY(root_id) REFERENCES roots(id),
                FOREIGN KEY(document_id) REFERENCES documents(id)
            );

            CREATE TABLE IF NOT EXISTS content_blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                document_id INTEGER,
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
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
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
                run_id TEXT,
                task_type TEXT NOT NULL,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                created_at TEXT NOT NULL,
                queued_at TEXT,
                started_at TEXT,
                spooled_at TEXT,
                written_at TEXT,
                finished_at TEXT,
                spool_path TEXT,
                spool_checksum TEXT,
                error_code TEXT,
                error_message TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS index_runs (
                id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                mode TEXT NOT NULL,
                status TEXT NOT NULL,
                discovered_files INTEGER NOT NULL DEFAULT 0,
                discovered_bytes INTEGER NOT NULL DEFAULT 0,
                scan_ms INTEGER NOT NULL DEFAULT 0,
                parse_ms INTEGER NOT NULL DEFAULT 0,
                write_ms INTEGER NOT NULL DEFAULT 0,
                fts_ms INTEGER NOT NULL DEFAULT 0,
                total_ms INTEGER NOT NULL DEFAULT 0,
                peak_rss_bytes INTEGER NOT NULL DEFAULT 0,
                summary_json TEXT
            );

            CREATE TABLE IF NOT EXISTS index_file_metrics (
                run_id TEXT NOT NULL,
                file_id INTEGER NOT NULL,
                extension TEXT,
                size_bytes INTEGER,
                queue_name TEXT,
                queue_wait_ms INTEGER,
                parse_ms INTEGER,
                block_count INTEGER,
                text_chars INTEGER,
                spool_bytes INTEGER,
                worker_pid INTEGER,
                PRIMARY KEY(run_id, file_id),
                FOREIGN KEY(run_id) REFERENCES index_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS index_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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

    def _migrate_schema_v3(
        self,
        con: sqlite3.Connection,
        previous_version: int | None,
    ) -> None:
        self._ensure_column(con, "files", "content_key", "TEXT")
        self._ensure_column(con, "files", "document_id", "INTEGER REFERENCES documents(id)")
        self._ensure_column(con, "content_blocks", "document_id", "INTEGER REFERENCES documents(id)")
        for name, declaration in (
            ("run_id", "TEXT"),
            ("queued_at", "TEXT"),
            ("spooled_at", "TEXT"),
            ("written_at", "TEXT"),
            ("spool_path", "TEXT"),
            ("spool_checksum", "TEXT"),
            ("error_code", "TEXT"),
        ):
            self._ensure_column(con, "parse_tasks", name, declaration)
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_files_document ON files(document_id);
            CREATE INDEX IF NOT EXISTS idx_files_content_key ON files(content_key);
            CREATE INDEX IF NOT EXISTS idx_blocks_document ON content_blocks(document_id);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_blocks_document_index
                ON content_blocks(document_id, block_index)
                WHERE document_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_parse_tasks_status ON parse_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_parse_tasks_run ON parse_tasks(run_id);
            """
        )
        if previous_version is not None and previous_version < 3:
            active_files = int(
                con.execute("SELECT COUNT(*) FROM files WHERE is_deleted = 0").fetchone()[0]
            )
            if active_files:
                self._set_index_state(con, "schema_v3_rebuild_required", "1")
                con.execute(
                    """
                    UPDATE files
                    SET parser_version = NULL, parse_status = 'pending'
                    WHERE is_deleted = 0
                    """
                )

    @staticmethod
    def _migrate_schema_v4(
        con: sqlite3.Connection,
        previous_version: int | None,
    ) -> None:
        if previous_version is None or previous_version >= 4:
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_short_tokens_block ON short_tokens(block_id)"
            )
            return
        # short_tokens was populated by schema-v2 but is not read or written by
        # current search code. Keeping millions of legacy rows makes each
        # content-block cascade scan the whole table during a rebuild.
        con.execute("DROP TABLE IF EXISTS short_tokens")
        con.executescript(
            """
            CREATE TABLE short_tokens (
                token TEXT NOT NULL,
                block_id INTEGER NOT NULL,
                position_count INTEGER DEFAULT 1,
                PRIMARY KEY(token, block_id),
                FOREIGN KEY(block_id) REFERENCES content_blocks(id) ON DELETE CASCADE
            );
            CREATE INDEX idx_short_tokens_block ON short_tokens(block_id);
            """
        )

    @staticmethod
    def _ensure_column(
        con: sqlite3.Connection,
        table: str,
        column: str,
        declaration: str,
    ) -> None:
        columns = {str(row["name"]) for row in con.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def _ensure_fts(self, con: sqlite3.Connection) -> None:
        tokenizer = "tokenize='trigram'"
        try:
            con.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS content_fts USING fts5(
                    block_id UNINDEXED,
                    file_id UNINDEXED,
                    filename,
                    path,
                    location_text,
                    normalized_text,
                    {tokenizer}
                )
                """
            )
            con.execute(
                f"""
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    file_id UNINDEXED,
                    filename,
                    path,
                    {tokenizer}
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
            con.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS files_fts USING fts5(
                    file_id UNINDEXED,
                    filename,
                    path
                )
                """
            )
        con.execute(
            """
            INSERT INTO files_fts(rowid, file_id, filename, path)
            SELECT f.id, f.id, f.filename, f.path
            FROM files f
            WHERE NOT EXISTS (SELECT 1 FROM files_fts ft WHERE ft.rowid = f.id)
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
            document_ids = self._document_ids_for_root(con, root_id)
            self._reassign_canonical_blocks_before_delete(con, root_id, document_ids)
            con.execute(
                "DELETE FROM content_fts WHERE file_id IN (SELECT id FROM files WHERE root_id = ?) AND block_id IN (SELECT id FROM content_blocks WHERE document_id IS NULL)",
                (root_id,),
            )
            con.execute("DELETE FROM files_fts WHERE file_id IN (SELECT id FROM files WHERE root_id = ?)", (root_id,))
            con.execute("DELETE FROM files WHERE root_id = ?", (root_id,))
            con.execute("DELETE FROM roots WHERE id = ?", (root_id,))
            self._garbage_collect_documents(con, document_ids)

    def _document_ids_for_root(self, con: sqlite3.Connection, root_id: int) -> set[int]:
        return {
            int(row["document_id"])
            for row in con.execute(
                "SELECT DISTINCT document_id FROM files WHERE root_id = ? AND document_id IS NOT NULL",
                (root_id,),
            )
        }

    def _reassign_canonical_blocks_before_delete(
        self,
        con: sqlite3.Connection,
        root_id: int,
        document_ids: set[int],
    ) -> None:
        for document_id in document_ids:
            replacement = con.execute(
                "SELECT id FROM files WHERE document_id = ? AND root_id != ? LIMIT 1",
                (document_id, root_id),
            ).fetchone()
            if replacement is not None:
                con.execute(
                    "UPDATE content_blocks SET file_id = ? WHERE document_id = ? AND file_id IN (SELECT id FROM files WHERE root_id = ?)",
                    (int(replacement["id"]), document_id, root_id),
                )

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

    def upsert_file_metadata(
        self,
        root_id: int,
        file_path: Path,
        *,
        retry_failed_files: bool = False,
        compute_full_hash: bool = False,
        mark_processing: bool = False,
        parser_version: str | None = None,
    ) -> tuple[int, bool]:
        versions = {str(file_path): parser_version} if parser_version else None
        results, errors = self.upsert_file_metadata_many(
            root_id,
            [file_path],
            retry_failed_files=retry_failed_files,
            compute_full_hash=compute_full_hash,
            mark_processing=mark_processing,
            parser_versions=versions,
        )
        if errors:
            raise errors[0][1]
        return results[0][1], results[0][2]

    def upsert_file_metadata_many(
        self,
        root_id: int,
        file_paths: Sequence[Path],
        *,
        retry_failed_files: bool = False,
        compute_full_hash: bool = False,
        mark_processing: bool = False,
        parser_versions: dict[str, str | None] | None = None,
    ) -> tuple[list[tuple[Path, int, bool]], list[tuple[Path, Exception]]]:
        prepared: list[tuple[Path, int, float, float, str, str | None, str | None]] = []
        errors: list[tuple[Path, Exception]] = []
        versions = parser_versions or {}
        for file_path in file_paths:
            try:
                stat = file_path.stat()
                fingerprint = f"{stat.st_size}:{stat.st_mtime_ns}"
                content_hash = _sha256_file(file_path) if compute_full_hash else None
                prepared.append(
                    (
                        file_path,
                        stat.st_size,
                        stat.st_mtime,
                        stat.st_ctime,
                        fingerprint,
                        content_hash,
                        versions.get(str(file_path)),
                    )
                )
            except Exception as exc:
                errors.append((file_path, exc))

        results: list[tuple[Path, int, bool]] = []
        if not prepared:
            return results, errors
        now = utc_now()
        with self.connect() as con:
            for file_path, size, modified_time, created_time, fingerprint, content_hash, expected_version in prepared:
                file_id, changed = self._upsert_file_metadata_in_connection(
                    con,
                    root_id,
                    file_path,
                    size=size,
                    modified_time=modified_time,
                    created_time=created_time,
                    fingerprint=fingerprint,
                    content_hash=content_hash,
                    now=now,
                    retry_failed_files=retry_failed_files,
                    compute_full_hash=compute_full_hash,
                    mark_processing=mark_processing,
                    expected_parser_version=expected_version,
                )
                results.append((file_path, file_id, changed))
        return results, errors

    def _upsert_file_metadata_in_connection(
        self,
        con: sqlite3.Connection,
        root_id: int,
        file_path: Path,
        *,
        size: int,
        modified_time: float,
        created_time: float,
        fingerprint: str,
        content_hash: str | None,
        now: str,
        retry_failed_files: bool,
        compute_full_hash: bool,
        mark_processing: bool,
        expected_parser_version: str | None,
    ) -> tuple[int, bool]:
        path_text = str(file_path)
        changed_status = "processing" if mark_processing else "pending"
        existing = con.execute(
            "SELECT id, quick_fingerprint, content_hash, parse_status, parser_version, parse_error_code FROM files WHERE path = ?",
            (path_text,),
        ).fetchone()
        if existing is None:
            cur = con.execute(
                """
                INSERT INTO files(
                    root_id, path, filename, extension, size_bytes, modified_time, created_time,
                    quick_fingerprint, content_hash, parse_status, parser_version, last_seen_at, is_deleted
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    root_id,
                    path_text,
                    file_path.name,
                    file_path.suffix.lower(),
                    size,
                    modified_time,
                    created_time,
                    fingerprint,
                    content_hash,
                    changed_status,
                    expected_parser_version or PARSER_VERSION,
                    now,
                ),
            )
            file_id = int(cur.lastrowid)
            self._upsert_file_fts(con, file_id, file_path.name, path_text)
            return file_id, True
        retry_statuses: set[str] = set()
        if retry_failed_files and existing["parse_error_code"] in {
            "FILE_IN_USE",
            "OCR_UNAVAILABLE",
            "PROCESS_WORKER_CRASH",
            "PARSE_TIMEOUT",
        }:
            retry_statuses.update({"failed", "failed_retryable", "ocr_failed"})
        changed = (
            existing["quick_fingerprint"] != fingerprint
            or (expected_parser_version is not None and existing["parser_version"] != expected_parser_version)
            or (compute_full_hash and existing["content_hash"] != content_hash)
            or existing["parse_status"] in {"pending", "processing", "cancelled", "deleted"}
            or existing["parse_status"] in retry_statuses
        )
        next_status = changed_status if changed else str(existing["parse_status"])
        con.execute(
            """
            UPDATE files SET
                root_id = ?, filename = ?, extension = ?, size_bytes = ?, modified_time = ?,
                created_time = ?, quick_fingerprint = ?, content_hash = COALESCE(?, content_hash),
                last_seen_at = ?, is_deleted = 0, parse_status = ?,
                parser_version = COALESCE(?, parser_version)
            WHERE id = ?
            """,
            (
                root_id,
                file_path.name,
                file_path.suffix.lower(),
                size,
                modified_time,
                created_time,
                fingerprint,
                content_hash,
                now,
                next_status,
                expected_parser_version,
                int(existing["id"]),
            ),
        )
        file_id = int(existing["id"])
        self._upsert_file_fts(con, file_id, file_path.name, path_text)
        return file_id, bool(changed)

    @staticmethod
    def _upsert_file_fts(
        con: sqlite3.Connection,
        file_id: int,
        filename: str,
        path: str,
    ) -> None:
        con.execute("DELETE FROM files_fts WHERE rowid = ?", (file_id,))
        con.execute(
            "INSERT INTO files_fts(rowid, file_id, filename, path) VALUES (?, ?, ?, ?)",
            (file_id, file_id, filename, path),
        )

    def mark_processing(self, file_id: int) -> None:
        with self.connect() as con:
            con.execute("UPDATE files SET parse_status = 'processing' WHERE id = ?", (file_id,))

    def invalidate_file(self, path: str | Path) -> bool:
        with self.connect() as con:
            cursor = con.execute(
                "UPDATE files SET parse_status = 'pending' WHERE path = ? AND is_deleted = 0",
                (str(path),),
            )
            return cursor.rowcount > 0

    def reuse_cached_document(
        self,
        file_ids: Sequence[int],
        content_key: str,
        parser_name: str,
        parser_version: str,
    ) -> tuple[bool, str | None]:
        if not file_ids:
            return False, None
        with self.connect() as con:
            document = con.execute(
                """
                SELECT id, parse_status FROM documents
                WHERE content_key = ? AND parser_name = ? AND parser_version = ?
                  AND parse_status NOT IN ('running', 'failed_retryable')
                """,
                (content_key, parser_name, parser_version),
            ).fetchone()
            if document is None:
                return False, None
            now = utc_now()
            placeholders = ",".join("?" for _ in file_ids)
            old_ids = self._document_ids_for_files(con, file_ids)
            con.execute(
                f"""
                UPDATE files SET document_id = ?, content_key = ?, parser_name = ?, parser_version = ?,
                    parse_status = ?, parse_error_code = NULL, parse_error_message = NULL,
                    indexed_at = ?, is_deleted = 0
                WHERE id IN ({placeholders})
                """,
                (
                    int(document["id"]),
                    content_key,
                    parser_name,
                    parser_version,
                    str(document["parse_status"]),
                    now,
                    *file_ids,
                ),
            )
            self._garbage_collect_documents(con, old_ids - {int(document["id"])})
            return True, str(document["parse_status"])

    def find_cached_documents(
        self,
        identities: Sequence[tuple[str, str, str]],
    ) -> dict[tuple[str, str, str], tuple[int, str]]:
        result: dict[tuple[str, str, str], tuple[int, str]] = {}
        unique = list(dict.fromkeys(identities))
        with self.connect() as con:
            for offset in range(0, len(unique), 200):
                chunk = unique[offset : offset + 200]
                if not chunk:
                    continue
                predicates = " OR ".join(
                    "(content_key = ? AND parser_name = ? AND parser_version = ?)"
                    for _ in chunk
                )
                params = [value for identity in chunk for value in identity]
                rows = con.execute(
                    f"""
                    SELECT id, content_key, parser_name, parser_version, parse_status
                    FROM documents
                    WHERE parse_status NOT IN ('running', 'failed_retryable')
                      AND ({predicates})
                    """,
                    params,
                ).fetchall()
                for row in rows:
                    key = (
                        str(row["content_key"]),
                        str(row["parser_name"]),
                        str(row["parser_version"]),
                    )
                    result[key] = (int(row["id"]), str(row["parse_status"]))
        return result

    def link_cached_document(
        self,
        file_ids: Sequence[int],
        document_id: int,
        content_key: str,
        parser_name: str,
        parser_version: str,
        status: str,
    ) -> None:
        if not file_ids:
            return
        with self.connect() as con:
            placeholders = ",".join("?" for _ in file_ids)
            old_ids = self._document_ids_for_files(con, file_ids)
            con.execute(
                f"""
                UPDATE files SET document_id = ?, content_key = ?, parser_name = ?, parser_version = ?,
                    parse_status = ?, parse_error_code = NULL, parse_error_message = NULL,
                    indexed_at = ?, is_deleted = 0
                WHERE id IN ({placeholders})
                """,
                (
                    int(document_id),
                    content_key,
                    parser_name,
                    parser_version,
                    status,
                    utc_now(),
                    *file_ids,
                ),
            )
            self._garbage_collect_documents(con, old_ids - {int(document_id)})

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
        self.replace_document_blocks_many(
            [
                {
                    "file_id": file_id,
                    "file_ids": [file_id],
                    "filename": filename,
                    "path": path,
                    "blocks": blocks,
                    "parser_name": parser_name,
                    "parser_version": PARSER_VERSION,
                    "status": status,
                    "error_code": None,
                    "error_message": None,
                    "content_key": f"legacy-file:{file_id}:{parser_name}",
                    "task_id": None,
                }
            ]
        )

    def replace_file_blocks_many(
        self,
        items: Sequence[tuple[int, str, str, Sequence[ContentBlock], str, str, str | None, str | None]],
    ) -> None:
        converted = []
        for file_id, filename, path, blocks, parser_name, status, error_code, error_message in items:
            converted.append(
                {
                    "file_id": file_id,
                    "file_ids": [file_id],
                    "filename": filename,
                    "path": path,
                    "blocks": blocks,
                    "parser_name": parser_name,
                    "parser_version": PARSER_VERSION,
                    "status": status,
                    "error_code": error_code,
                    "error_message": error_message,
                    "content_key": f"legacy-file:{file_id}:{parser_name}",
                    "task_id": None,
                }
            )
        self.replace_document_blocks_many(converted)

    def replace_document_blocks_many(
        self,
        items: Sequence[dict[str, Any]],
        *,
        update_fts: bool = True,
    ) -> None:
        if not items:
            return
        now = utc_now()
        with self.connect() as con:
            for item in items:
                self._replace_document_item(con, item, now, update_fts=update_fts)
            if not update_fts:
                self._set_index_state(con, "content_fts_dirty", "1")

    def _replace_document_item(
        self,
        con: sqlite3.Connection,
        item: dict[str, Any],
        now: str,
        *,
        update_fts: bool,
    ) -> None:
        primary_file_id = int(item["file_id"])
        file_ids = [int(value) for value in item.get("file_ids") or [primary_file_id]]
        blocks = list(item.get("blocks") or [])
        parser_name = str(item.get("parser_name") or "unknown")
        parser_version = str(item.get("parser_version") or PARSER_VERSION)
        status = str(item.get("status") or "success")
        error_code = item.get("error_code")
        error_message = item.get("error_message")
        content_key = str(item.get("content_key") or f"file:{primary_file_id}:{parser_name}")
        old_document_ids = self._document_ids_for_files(con, file_ids)
        document_id: int | None = None
        placeholders = ",".join("?" for _ in file_ids)

        # Schema-v2 rows are owned directly by a file and have no document_id.
        # Remove those rows before attaching the path to its schema-v3 document,
        # otherwise an upgrade would keep both the old and new searchable text.
        if update_fts:
            con.execute(
                f"DELETE FROM content_fts WHERE rowid IN ("
                f"SELECT id FROM content_blocks WHERE file_id IN ({placeholders}) "
                "AND document_id IS NULL)",
                tuple(file_ids),
            )
        con.execute(
            f"DELETE FROM content_blocks WHERE file_id IN ({placeholders}) AND document_id IS NULL",
            tuple(file_ids),
        )

        if status in SUCCESSFUL_DOCUMENT_STATUSES:
            con.execute(
                """
                INSERT INTO documents(
                    content_key, parser_name, parser_version, parse_status,
                    parse_error_code, parse_error_message, block_count, text_chars,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, 0, ?, ?)
                ON CONFLICT(content_key, parser_name, parser_version) DO UPDATE SET
                    parse_status=excluded.parse_status,
                    parse_error_code=excluded.parse_error_code,
                    parse_error_message=excluded.parse_error_message,
                    updated_at=excluded.updated_at
                """,
                (
                    content_key,
                    parser_name,
                    parser_version,
                    status,
                    error_code,
                    _trim_message(error_message),
                    now,
                    now,
                ),
            )
            document = con.execute(
                "SELECT id FROM documents WHERE content_key = ? AND parser_name = ? AND parser_version = ?",
                (content_key, parser_name, parser_version),
            ).fetchone()
            document_id = int(document["id"])
            existing_count = int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM content_blocks WHERE document_id = ?",
                    (document_id,),
                ).fetchone()["n"]
            )
            if blocks or existing_count == 0:
                self._delete_document_blocks(con, document_id, update_fts=update_fts)
                for block in blocks:
                    block_id = self._insert_content_block(
                        con,
                        primary_file_id,
                        document_id,
                        block,
                        now,
                    )
                    if update_fts:
                        self._insert_content_fts(
                            con,
                            block_id,
                            primary_file_id,
                            str(item.get("filename") or Path(str(item.get("path") or "")).name),
                            str(item.get("path") or ""),
                            block,
                        )
                con.execute(
                    "UPDATE documents SET block_count = ?, text_chars = ?, updated_at = ? WHERE id = ?",
                    (len(blocks), sum(len(block.raw_text) for block in blocks), now, document_id),
                )
        else:
            if update_fts:
                con.execute(
                    "DELETE FROM content_fts WHERE rowid IN ("
                    "SELECT id FROM content_blocks WHERE file_id = ? AND document_id IS NULL)",
                    (primary_file_id,),
                )
            con.execute("DELETE FROM content_blocks WHERE file_id = ? AND document_id IS NULL", (primary_file_id,))

        con.execute(
            f"""
            UPDATE files SET
                document_id = ?, content_key = ?, parse_status = ?, parse_error_code = ?,
                parse_error_message = ?, parser_name = ?, parser_version = ?, indexed_at = ?,
                is_deleted = 0
            WHERE id IN ({placeholders})
            """,
            (
                document_id,
                content_key,
                status,
                error_code,
                _trim_message(error_message),
                parser_name,
                parser_version,
                now,
                *file_ids,
            ),
        )
        task_id = item.get("task_id")
        if task_id is not None:
            con.execute(
                """
                UPDATE parse_tasks SET status = ?, written_at = ?, finished_at = ?,
                    error_code = ?, error_message = ? WHERE id = ?
                """,
                (
                    "complete" if status in SUCCESSFUL_DOCUMENT_STATUSES else "failed",
                    now,
                    now,
                    error_code,
                    _trim_message(error_message),
                    int(task_id),
                ),
            )
        if document_id is not None:
            old_document_ids.discard(document_id)
        self._garbage_collect_documents(con, old_document_ids, update_fts=update_fts)

    @staticmethod
    def _document_ids_for_files(con: sqlite3.Connection, file_ids: Sequence[int]) -> set[int]:
        if not file_ids:
            return set()
        placeholders = ",".join("?" for _ in file_ids)
        return {
            int(row["document_id"])
            for row in con.execute(
                f"SELECT DISTINCT document_id FROM files WHERE id IN ({placeholders}) AND document_id IS NOT NULL",
                tuple(file_ids),
            )
        }

    @staticmethod
    def _insert_content_block(
        con: sqlite3.Connection,
        file_id: int,
        document_id: int,
        block: ContentBlock,
        now: str,
    ) -> int:
        cur = con.execute(
            """
            INSERT INTO content_blocks(
                file_id, document_id, block_index, block_type, location_text, page_number,
                slide_number, sheet_name, cell_start, cell_end, line_start, line_end,
                raw_text, normalized_text, source_type, ocr_confidence, extra_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                document_id,
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
        return int(cur.lastrowid)

    @staticmethod
    def _insert_content_fts(
        con: sqlite3.Connection,
        block_id: int,
        file_id: int,
        filename: str,
        path: str,
        block: ContentBlock,
    ) -> None:
        con.execute(
            """
            INSERT INTO content_fts(rowid, block_id, file_id, filename, path, location_text, normalized_text)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (block_id, block_id, file_id, filename, path, block.location_text, block.normalized_text),
        )

    @staticmethod
    def _delete_document_blocks(
        con: sqlite3.Connection,
        document_id: int,
        *,
        update_fts: bool = True,
    ) -> None:
        if update_fts:
            # content_fts.rowid is the content block id. Filtering on the
            # unindexed block_id column forces a full FTS5 scan for every
            # document and can hold SQLite's write lock for minutes.
            con.execute(
                "DELETE FROM content_fts WHERE rowid IN ("
                "SELECT id FROM content_blocks WHERE document_id = ?)",
                (document_id,),
            )
        con.execute("DELETE FROM content_blocks WHERE document_id = ?", (document_id,))

    def _garbage_collect_documents(
        self,
        con: sqlite3.Connection,
        document_ids: set[int],
        *,
        update_fts: bool = True,
    ) -> None:
        for document_id in document_ids:
            used = con.execute(
                "SELECT 1 FROM files WHERE document_id = ? LIMIT 1",
                (document_id,),
            ).fetchone()
            if used is None:
                self._delete_document_blocks(con, document_id, update_fts=update_fts)
                con.execute("DELETE FROM documents WHERE id = ?", (document_id,))

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
                    parser_name = ?, indexed_at = ? WHERE id = ?
                """,
                (status, error_code, _trim_message(message), parser_name, utc_now(), file_id),
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
                    parser_name = ?, indexed_at = ? WHERE id = ?
                """,
                (status, error_code, _trim_message(message), parser_name, utc_now(), file_id),
            )

    def mark_unsupported_with_metadata(
        self,
        file_id: int,
        filename: str,
        path: str,
        block: ContentBlock,
    ) -> None:
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
            placeholders = ",".join("?" for _ in paths)
            rows = con.execute(
                f"SELECT id, document_id FROM files WHERE path IN ({placeholders})",
                tuple(paths),
            ).fetchall()
            file_ids = [int(row["id"]) for row in rows]
            document_ids = {
                int(row["document_id"])
                for row in rows
                if row["document_id"] is not None
            }
            if not file_ids:
                return 0
            id_placeholders = ",".join("?" for _ in file_ids)
            for document_id in document_ids:
                replacement = con.execute(
                    f"""
                    SELECT id FROM files
                    WHERE document_id = ? AND id NOT IN ({id_placeholders}) AND is_deleted = 0
                    LIMIT 1
                    """,
                    (document_id, *file_ids),
                ).fetchone()
                if replacement is not None:
                    con.execute(
                        f"""
                        UPDATE content_blocks SET file_id = ?
                        WHERE document_id = ? AND file_id IN ({id_placeholders})
                        """,
                        (int(replacement["id"]), document_id, *file_ids),
                    )
            con.execute(
                f"DELETE FROM content_fts WHERE file_id IN ({id_placeholders}) AND block_id IN (SELECT id FROM content_blocks WHERE document_id IS NULL)",
                tuple(file_ids),
            )
            con.execute(
                f"UPDATE files SET document_id = NULL, is_deleted = 1, parse_status = 'deleted' WHERE id IN ({id_placeholders})",
                tuple(file_ids),
            )
            self._garbage_collect_documents(con, document_ids)
            return len(file_ids)

    def update_root_scan_time(self, root_id: int, status: str = "ready") -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE roots SET last_scan_at = ?, status = ?, updated_at = ? WHERE id = ?",
                (utc_now(), status, utc_now(), root_id),
            )

    def create_parse_task(
        self,
        file_id: int,
        run_id: str,
        task_type: str,
        priority: int = 100,
    ) -> int:
        return self.create_parse_tasks([(file_id, run_id, task_type, priority)])[0]

    def create_parse_tasks(
        self,
        tasks: Sequence[tuple[int, str, str, int]],
    ) -> list[int]:
        if not tasks:
            return []
        now = utc_now()
        result: list[int] = []
        with self.connect() as con:
            for file_id, run_id, task_type, priority in tasks:
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET status = 'superseded', finished_at = ?
                    WHERE file_id = ? AND status = 'queued'
                    """,
                    (now, file_id),
                )
                cur = con.execute(
                    """
                    INSERT INTO parse_tasks(
                        file_id, run_id, task_type, status, priority, created_at, queued_at
                    ) VALUES (?, ?, ?, 'queued', ?, ?, ?)
                    """,
                    (file_id, run_id, task_type, priority, now, now),
                )
                result.append(int(cur.lastrowid))
        return result

    def mark_task_running(self, task_id: int) -> None:
        self.mark_tasks_running([task_id])

    def mark_tasks_running(self, task_ids: Sequence[int]) -> None:
        self._mark_tasks_running(task_ids, timeout_seconds=30.0)

    def try_mark_tasks_running(
        self,
        task_ids: Sequence[int],
        *,
        timeout_seconds: float = 0.0,
    ) -> bool:
        """Best-effort diagnostic state update that never waits on a busy database."""

        try:
            self._mark_tasks_running(task_ids, timeout_seconds=timeout_seconds)
            return True
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                return False
            raise

    def _mark_tasks_running(
        self,
        task_ids: Sequence[int],
        *,
        timeout_seconds: float,
    ) -> None:
        if not task_ids:
            return
        with self.connect(timeout_seconds=timeout_seconds) as con:
            now = utc_now()
            con.executemany(
                "UPDATE parse_tasks SET status = 'running', started_at = ? WHERE id = ?",
                [(now, int(task_id)) for task_id in task_ids],
            )
            placeholders = ",".join("?" for _ in task_ids)
            con.execute(
                f"UPDATE files SET parse_status = 'processing' WHERE id IN (SELECT file_id FROM parse_tasks WHERE id IN ({placeholders}))",
                tuple(int(task_id) for task_id in task_ids),
            )

    def mark_task_spooled(self, task_id: int, spool_path: Path, checksum: str) -> None:
        self._mark_task_spooled(task_id, spool_path, checksum, timeout_seconds=30.0)

    def try_mark_task_spooled(
        self,
        task_id: int,
        spool_path: Path,
        checksum: str,
        *,
        timeout_seconds: float = 0.0,
    ) -> bool:
        """Best-effort spool registration; the result file remains the source of truth."""

        try:
            self._mark_task_spooled(task_id, spool_path, checksum, timeout_seconds=timeout_seconds)
            return True
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                return False
            raise

    def _mark_task_spooled(
        self,
        task_id: int,
        spool_path: Path,
        checksum: str,
        *,
        timeout_seconds: float,
    ) -> None:
        with self.connect(timeout_seconds=timeout_seconds) as con:
            con.execute(
                """
                UPDATE parse_tasks SET status = 'spooled', spooled_at = ?,
                    spool_path = ?, spool_checksum = ? WHERE id = ?
                """,
                (utc_now(), str(spool_path), checksum, task_id),
            )

    def mark_task_failed(self, task_id: int, error_code: str, message: str) -> None:
        with self.connect() as con:
            con.execute(
                """
                UPDATE parse_tasks SET status = 'failed', finished_at = ?, error_code = ?,
                    error_message = ? WHERE id = ?
                """,
                (utc_now(), error_code, _trim_message(message), task_id),
            )

    def recoverable_spooled_tasks(self, root_id: int) -> list[sqlite3.Row]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT pt.*, f.path
                    FROM parse_tasks pt
                    JOIN files f ON f.id = pt.file_id
                    WHERE pt.status = 'spooled' AND f.root_id = ?
                    ORDER BY pt.id
                    """,
                    (root_id,),
                ).fetchall()
            )

    @staticmethod
    def _recover_interrupted_tasks(con: sqlite3.Connection) -> None:
        con.execute(
            """
            UPDATE index_runs SET status = 'interrupted', finished_at = ?
            WHERE status = 'running'
            """,
            (utc_now(),),
        )
        con.execute(
            """
            UPDATE parse_tasks SET status = 'queued', started_at = NULL
            WHERE status = 'running'
            """
        )
        con.execute(
            """
            UPDATE files SET parse_status = 'pending'
            WHERE id IN (SELECT file_id FROM parse_tasks WHERE status = 'queued')
              AND parse_status = 'processing'
            """
        )

    def start_index_run(self, metrics: IndexRunMetrics) -> None:
        with self.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO index_runs(id, started_at, mode, status)
                VALUES (?, ?, ?, 'running')
                """,
                (metrics.run_id, utc_now(), metrics.mode),
            )

    def finish_index_run(
        self,
        metrics: IndexRunMetrics,
        status: str,
        summary: dict[str, object],
    ) -> None:
        parse_ms = sum(metrics.parse_ms_by_lane.values())
        with self.connect() as con:
            con.execute(
                """
                UPDATE index_runs SET
                    finished_at = ?, status = ?, discovered_files = ?, discovered_bytes = ?,
                    scan_ms = ?, parse_ms = ?, write_ms = ?, fts_ms = ?, total_ms = ?,
                    peak_rss_bytes = ?, summary_json = ?
                WHERE id = ?
                """,
                (
                    utc_now(),
                    status,
                    metrics.discovered_files,
                    metrics.discovered_bytes,
                    metrics.scan_ms,
                    parse_ms,
                    metrics.database_write_ms,
                    metrics.fts_build_ms,
                    metrics.total_ms,
                    metrics.peak_rss_bytes,
                    json.dumps({"metrics": metrics.to_dict(), "summary": summary}, ensure_ascii=False),
                    metrics.run_id,
                ),
            )
            con.execute(
                """
                DELETE FROM index_file_metrics
                WHERE run_id IN (
                    SELECT id FROM index_runs ORDER BY started_at DESC LIMIT -1 OFFSET 3
                )
                """
            )
            con.execute(
                """
                DELETE FROM index_runs
                WHERE id IN (
                    SELECT id FROM index_runs ORDER BY started_at DESC LIMIT -1 OFFSET 3
                )
                """
            )

    def record_file_metrics(self, run_id: str, timings: Sequence[FileTiming]) -> None:
        if not timings:
            return
        with self.connect() as con:
            con.executemany(
                """
                INSERT OR REPLACE INTO index_file_metrics(
                    run_id, file_id, extension, size_bytes, queue_name, queue_wait_ms,
                    parse_ms, block_count, text_chars, spool_bytes, worker_pid
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        item.file_id,
                        item.extension,
                        item.size_bytes,
                        item.queue_name,
                        item.queue_wait_ms,
                        item.parse_ms,
                        item.block_count,
                        item.text_chars,
                        item.spool_bytes,
                        item.worker_pid,
                    )
                    for item in timings
                ],
            )

    def active_file_count(self) -> int:
        with self.connect() as con:
            return int(con.execute("SELECT COUNT(*) AS n FROM files WHERE is_deleted = 0").fetchone()["n"])

    def begin_deferred_fts(self) -> None:
        with self.connect() as con:
            self._set_index_state(con, "content_fts_dirty", "1")
            self._set_index_state(con, "full_batch_incomplete", "1")

    def has_incomplete_full_batch(self) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM index_state WHERE key = 'full_batch_incomplete'"
            ).fetchone()
            return row is not None and str(row["value"]) == "1"

    def mark_full_batch_complete(self) -> None:
        with self.connect() as con:
            self._set_index_state(con, "full_batch_incomplete", "0")

    def requires_full_rebuild(self) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM index_state WHERE key = 'schema_v3_rebuild_required'"
            ).fetchone()
            return row is not None and str(row["value"]) == "1"

    def mark_full_rebuild_complete(self) -> None:
        with self.connect() as con:
            self._set_index_state(con, "schema_v3_rebuild_required", "0")

    def rebuild_content_fts(self) -> int:
        started = datetime.now(timezone.utc)
        with self.connect() as con:
            con.execute("DELETE FROM content_fts")
            con.execute(
                """
                INSERT INTO content_fts(rowid, block_id, file_id, filename, path, location_text, normalized_text)
                SELECT cb.id, cb.id, cb.file_id, COALESCE(f.filename, ''), COALESCE(f.path, ''),
                       cb.location_text, cb.normalized_text
                FROM content_blocks cb
                LEFT JOIN files f ON f.id = cb.file_id
                """
            )
            self._set_index_state(con, "content_fts_dirty", "0")
        return int((datetime.now(timezone.utc) - started).total_seconds() * 1000)

    def rebuild_files_fts(self) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM files_fts")
            con.execute(
                """
                INSERT INTO files_fts(rowid, file_id, filename, path)
                SELECT id, id, filename, path FROM files
                """
            )

    def _fts_is_dirty(self) -> bool:
        with self.connect() as con:
            row = con.execute("SELECT value FROM index_state WHERE key = 'content_fts_dirty'").fetchone()
            return row is not None and str(row["value"]) == "1"

    @staticmethod
    def _set_index_state(con: sqlite3.Connection, key: str, value: str) -> None:
        con.execute(
            "INSERT INTO index_state(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def integrity_report(self) -> dict[str, object]:
        with self.connect() as con:
            integrity = [str(row[0]) for row in con.execute("PRAGMA integrity_check")]
            foreign_keys = [tuple(row) for row in con.execute("PRAGMA foreign_key_check")]
            orphan_documents = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM documents d
                    WHERE NOT EXISTS (SELECT 1 FROM files f WHERE f.document_id = d.id)
                    """
                ).fetchone()[0]
            )
            return {
                "integrity": integrity,
                "foreign_key_errors": foreign_keys,
                "orphan_documents": orphan_documents,
            }

    def failed_files(self, limit: int = 500) -> list[sqlite3.Row]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT path, extension, parse_status, parse_error_code, parse_error_message, parser_name, indexed_at
                    FROM files
                    WHERE parse_status IN (
                        'failed', 'failed_retryable', 'unsupported', 'skipped', 'metadata_only',
                        'ocr_disabled', 'ocr_failed', 'converter_missing', 'partial_success',
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
            documents = con.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            return {
                "files": int(total),
                "blocks": int(blocks),
                "documents": int(documents),
                "failed": int(failed),
                "unsupported": int(unsupported),
                "metadata_only": int(metadata_only),
            }

    def add_search_history(self, query_text: str, max_entries: int = 50) -> None:
        text = query_text.strip()
        if not text:
            return
        with self.connect() as con:
            con.execute("DELETE FROM search_history WHERE query_text = ?", (text,))
            con.execute(
                "INSERT INTO search_history(query_text, created_at) VALUES (?, ?)",
                (text, utc_now()),
            )
            con.execute(
                """
                DELETE FROM search_history
                WHERE id NOT IN (
                    SELECT id FROM search_history ORDER BY id DESC LIMIT ?
                )
                """,
                (max(1, max_entries),),
            )

    def search_history(self, limit: int = 20) -> list[str]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT query_text FROM search_history ORDER BY id DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()
            return [str(row["query_text"]) for row in rows]

    def clear_search_history(self) -> None:
        with self.connect() as con:
            con.execute("DELETE FROM search_history")


def _trim_message(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text[:1000] if text else None


def _is_database_busy(exc: sqlite3.OperationalError) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
