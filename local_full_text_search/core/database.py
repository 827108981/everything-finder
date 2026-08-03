from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_full_text_search.config.constants import DB_PATH, PARSER_VERSION, VIDEO_EXTENSIONS
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import FileTiming, IndexRunMetrics

SCHEMA_VERSION = 8
SUCCESSFUL_DOCUMENT_STATUSES = {
    "success",
    "metadata_only",
}
NON_BLOCKING_PARSE_STATUSES = {
    "success",
    "metadata_only",
}
TERMINAL_PARSE_TASK_STATUSES = {
    "complete",
    "failed",
    "superseded",
    "invalidated",
    "cancelled",
}
MANUALLY_EXCLUDABLE_STATUSES = {
    "failed",
    "failed_retryable",
    "unsupported",
    "skipped",
    "ocr_disabled",
    "ocr_failed",
    "converter_missing",
    "partial_success",
    "password_protected",
}
INDEX_RUN_RETENTION = 50


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _replace_file_with_retry(
    source: Path,
    target: Path,
    *,
    timeout_seconds: float = 3.0,
) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


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
        requires_structural_repair = bool(
            previous_version == SCHEMA_VERSION
            and not self._schema_layout_complete()
        )
        backup_path: Path | None = None
        if previous_version is not None and (
            previous_version < SCHEMA_VERSION
            or requires_structural_repair
        ):
            backup_path = self._backup_legacy_database(
                previous_version
            )
        try:
            with self.connect() as con:
                con.execute("PRAGMA journal_mode = WAL")
                self._create_schema(con)
                self._migrate_schema_v3(con, previous_version)
                self._migrate_schema_v4(con, previous_version)
                self._migrate_schema_v5(con)
                self._migrate_schema_v6(con)
                self._migrate_schema_v7(con)
                self._migrate_schema_v8(con)
                self._ensure_fts(con)
                self._recover_interrupted_tasks(con)
                con.execute(
                    f"PRAGMA user_version = {SCHEMA_VERSION}"
                )
        except Exception:
            if backup_path is not None:
                self._restore_migration_backup(backup_path)
            raise
        if backup_path is not None:
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

    def _schema_layout_complete(self) -> bool:
        if not self.db_path.is_file():
            return False
        con = sqlite3.connect(
            f"file:{self.db_path.as_posix()}?mode=ro",
            uri=True,
            timeout=30,
        )
        try:
            tables = {
                str(row[0])
                for row in con.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            if not {
                "pdf_page_identities",
                "ocr_requests",
                "ocr_exact_cache",
                "index_versions",
                "resource_events",
                "backend_benchmarks",
                "index_scope_exclusions",
            }.issubset(tables):
                return False
            parse_task_columns = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(parse_tasks)")
            }
            if not {
                "lease_owner",
                "lease_expires_at",
                "confirmed_at",
                "source_digest",
                "task_version",
                "result_digest",
            }.issubset(parse_task_columns):
                return False
            content_block_columns = {
                str(row[1])
                for row in con.execute(
                    "PRAGMA table_info(content_blocks)"
                )
            }
            if "index_version_id" not in content_block_columns:
                return False
            file_columns = {
                str(row[1])
                for row in con.execute("PRAGMA table_info(files)")
            }
            if "parse_diagnostics_json" not in file_columns:
                return False
            indexes = {
                str(row[0])
                for row in con.execute(
                    """
                    SELECT name FROM sqlite_master
                    WHERE type = 'index'
                    """
                )
            }
            return {
                "idx_content_blocks_index_version",
                "idx_scope_exclusions_active_file",
            }.issubset(indexes)
        except sqlite3.Error:
            return False
        finally:
            con.close()

    def _backup_legacy_database(self, previous_version: int) -> Path:
        backup_path = self.db_path.with_name(
            f"{self.db_path.stem}.schema-v{previous_version}.backup{self.db_path.suffix}"
        )
        if backup_path.is_file():
            stamp = datetime.now(timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            backup_path = self.db_path.with_name(
                f"{self.db_path.stem}.schema-v{previous_version}-{stamp}"
                f".backup{self.db_path.suffix}"
            )
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

    def _restore_migration_backup(self, backup_path: Path) -> Path:
        """Restore the pre-migration snapshot and preserve the failed copy."""

        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )
        failed_path = self.db_path.with_name(
            f"{self.db_path.stem}.migration-failed-{stamp}"
            f"{self.db_path.suffix}"
        )
        try:
            current = sqlite3.connect(self.db_path, timeout=30)
            try:
                current.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            finally:
                current.close()
        except sqlite3.Error:
            pass
        if self.db_path.exists():
            _replace_file_with_retry(self.db_path, failed_path)
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(self.db_path) + suffix)
            if sidecar.exists():
                sidecar.unlink()
        temporary = self.db_path.with_suffix(
            self.db_path.suffix + ".restore.tmp"
        )
        temporary.unlink(missing_ok=True)
        source = sqlite3.connect(backup_path, timeout=30)
        target = sqlite3.connect(temporary, timeout=30)
        try:
            source.backup(target)
            target.commit()
        except Exception:
            target.close()
            source.close()
            temporary.unlink(missing_ok=True)
            if (
                failed_path.exists()
                and not self.db_path.exists()
            ):
                _replace_file_with_retry(failed_path, self.db_path)
            raise
        else:
            target.close()
            source.close()
        _replace_file_with_retry(temporary, self.db_path)
        return failed_path

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
                parse_diagnostics_json TEXT,
                parser_name TEXT,
                parser_version TEXT,
                indexed_at TEXT,
                last_seen_at TEXT,
                is_deleted INTEGER NOT NULL DEFAULT 0,
                source_kind TEXT NOT NULL DEFAULT 'file',
                container_file_id INTEGER,
                internal_path TEXT,
                member_order INTEGER,
                member_crc32 INTEGER,
                member_uncompressed_size INTEGER,
                content_hash_full TEXT,
                FOREIGN KEY(root_id) REFERENCES roots(id),
                FOREIGN KEY(document_id) REFERENCES documents(id),
                FOREIGN KEY(container_file_id) REFERENCES files(id)
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
                index_version_id INTEGER,
                created_at TEXT NOT NULL,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE,
                FOREIGN KEY(index_version_id) REFERENCES index_versions(id)
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
                parent_task_id INTEGER,
                run_id TEXT,
                task_type TEXT NOT NULL,
                unit_key TEXT,
                payload_json TEXT,
                status TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                retry_count INTEGER NOT NULL DEFAULT 0,
                max_retries INTEGER NOT NULL DEFAULT 3,
                progress_phase TEXT,
                progress_completed INTEGER NOT NULL DEFAULT 0,
                progress_total INTEGER NOT NULL DEFAULT 0,
                progress_unit_type TEXT,
                progress_cursor TEXT,
                progress_bytes_read INTEGER NOT NULL DEFAULT 0,
                progress_output_blocks INTEGER NOT NULL DEFAULT 0,
                checkpoint_version INTEGER NOT NULL DEFAULT 0,
                last_semantic_progress_at TEXT,
                worker_pid INTEGER,
                stall_signature TEXT,
                checkpoint_path TEXT,
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
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_task_id) REFERENCES parse_tasks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS parse_task_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                attempt_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_progress_at TEXT,
                finished_at TEXT,
                worker_pid INTEGER,
                error_code TEXT,
                error_message TEXT,
                UNIQUE(task_id, attempt_no),
                FOREIGN KEY(task_id) REFERENCES parse_tasks(id) ON DELETE CASCADE
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

    def _migrate_schema_v5(self, con: sqlite3.Connection) -> None:
        for name, declaration in (
            ("source_kind", "TEXT NOT NULL DEFAULT 'file'"),
            ("container_file_id", "INTEGER REFERENCES files(id)"),
            ("internal_path", "TEXT"),
            ("member_order", "INTEGER"),
            ("member_crc32", "INTEGER"),
            ("member_uncompressed_size", "INTEGER"),
            ("content_hash_full", "TEXT"),
        ):
            self._ensure_column(con, "files", name, declaration)
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_files_container ON files(container_file_id);
            DROP INDEX IF EXISTS idx_files_source_kind;
            CREATE INDEX IF NOT EXISTS idx_files_source_kind
                ON files(source_kind, container_file_id, is_deleted);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_files_zip_member_source
                ON files(container_file_id, internal_path)
                WHERE source_kind = 'zip_member' AND is_deleted = 0;
            CREATE INDEX IF NOT EXISTS idx_files_content_hash_full
                ON files(content_hash_full);
            CREATE INDEX IF NOT EXISTS idx_files_exact_content
                ON files(extension, member_uncompressed_size, content_hash_full);
            """
        )

    def _migrate_schema_v6(self, con: sqlite3.Connection) -> None:
        for name, declaration in (
            ("parent_task_id", "INTEGER REFERENCES parse_tasks(id)"),
            ("unit_key", "TEXT"),
            ("payload_json", "TEXT"),
            ("progress_phase", "TEXT"),
            ("progress_completed", "INTEGER NOT NULL DEFAULT 0"),
            ("progress_total", "INTEGER NOT NULL DEFAULT 0"),
            ("progress_unit_type", "TEXT"),
            ("progress_cursor", "TEXT"),
            ("progress_bytes_read", "INTEGER NOT NULL DEFAULT 0"),
            ("progress_output_blocks", "INTEGER NOT NULL DEFAULT 0"),
            ("checkpoint_version", "INTEGER NOT NULL DEFAULT 0"),
            ("last_semantic_progress_at", "TEXT"),
            ("worker_pid", "INTEGER"),
            ("stall_signature", "TEXT"),
            ("checkpoint_path", "TEXT"),
        ):
            self._ensure_column(con, "parse_tasks", name, declaration)
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS parse_task_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER NOT NULL,
                attempt_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                last_progress_at TEXT,
                finished_at TEXT,
                worker_pid INTEGER,
                error_code TEXT,
                error_message TEXT,
                UNIQUE(task_id, attempt_no),
                FOREIGN KEY(task_id) REFERENCES parse_tasks(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_parse_tasks_parent
                ON parse_tasks(parent_task_id, status);
            CREATE INDEX IF NOT EXISTS idx_parse_tasks_unit
                ON parse_tasks(file_id, task_type, unit_key);
            DROP INDEX IF EXISTS idx_parse_tasks_parent_unit;
            CREATE UNIQUE INDEX idx_parse_tasks_parent_unit
                ON parse_tasks(parent_task_id, task_type, unit_key);
            CREATE INDEX IF NOT EXISTS idx_parse_task_attempts_task
                ON parse_task_attempts(task_id, attempt_no);
            """
        )

    def _migrate_schema_v7(self, con: sqlite3.Connection) -> None:
        for name, declaration in (
            ("lease_owner", "TEXT"),
            ("lease_expires_at", "TEXT"),
            ("confirmed_at", "TEXT"),
            ("source_digest", "TEXT"),
            ("task_version", "TEXT"),
            ("result_digest", "TEXT"),
        ):
            self._ensure_column(con, "parse_tasks", name, declaration)
        self._ensure_column(
            con,
            "content_blocks",
            "index_version_id",
            "INTEGER REFERENCES index_versions(id)",
        )

        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS pdf_page_identities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                source_digest TEXT NOT NULL,
                parser_version TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                page_identity TEXT NOT NULL,
                width_points REAL,
                height_points REAL,
                classification TEXT NOT NULL,
                native_task_id INTEGER,
                ocr_task_id INTEGER,
                created_at TEXT NOT NULL,
                invalidated_at TEXT,
                UNIQUE(file_id, source_digest, parser_version, page_number),
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(native_task_id) REFERENCES parse_tasks(id),
                FOREIGN KEY(ocr_task_id) REFERENCES parse_tasks(id)
            );

            CREATE TABLE IF NOT EXISTS ocr_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                parent_task_id INTEGER,
                source_kind TEXT NOT NULL,
                source_unit TEXT NOT NULL,
                image_spool_path TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                config_fingerprint TEXT NOT NULL,
                priority INTEGER NOT NULL DEFAULT 100,
                pixel_cost INTEGER NOT NULL,
                status TEXT NOT NULL,
                lease_owner TEXT,
                lease_expires_at TEXT,
                checkpoint_cursor TEXT,
                result_spool_path TEXT,
                result_digest TEXT,
                created_at TEXT NOT NULL,
                confirmed_at TEXT,
                error_code TEXT,
                error_message TEXT,
                UNIQUE(parent_task_id, source_kind, source_unit, config_fingerprint),
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(parent_task_id) REFERENCES parse_tasks(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS ocr_exact_cache (
                cache_key TEXT PRIMARY KEY,
                content_sha256 TEXT NOT NULL,
                config_fingerprint TEXT NOT NULL,
                source_width INTEGER NOT NULL,
                source_height INTEGER NOT NULL,
                result_json TEXT NOT NULL,
                result_digest TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_used_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS index_versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                root_id INTEGER,
                run_id TEXT NOT NULL,
                version_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                document_count INTEGER NOT NULL DEFAULT 0,
                block_count INTEGER NOT NULL DEFAULT 0,
                content_digest TEXT,
                created_at TEXT NOT NULL,
                activated_at TEXT,
                failed_at TEXT,
                error_message TEXT,
                FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS resource_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                event_type TEXT NOT NULL,
                lane TEXT,
                value REAL,
                unit TEXT,
                detail_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS backend_benchmarks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backend TEXT NOT NULL,
                model_fingerprint TEXT NOT NULL,
                corpus_fingerprint TEXT NOT NULL,
                settings_fingerprint TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                peak_rss_bytes INTEGER NOT NULL,
                accuracy_digest TEXT NOT NULL,
                detail_json TEXT,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_parse_tasks_lease
                ON parse_tasks(status, lease_expires_at, priority);
            CREATE INDEX IF NOT EXISTS idx_pdf_page_identity_file
                ON pdf_page_identities(file_id, source_digest, page_number);
            CREATE INDEX IF NOT EXISTS idx_ocr_requests_claim
                ON ocr_requests(status, lease_expires_at, priority);
            CREATE INDEX IF NOT EXISTS idx_index_versions_root
                ON index_versions(root_id, status);
            CREATE INDEX IF NOT EXISTS idx_content_blocks_index_version
                ON content_blocks(index_version_id);
            CREATE INDEX IF NOT EXISTS idx_resource_events_run
                ON resource_events(run_id, event_type);
            CREATE INDEX IF NOT EXISTS idx_backend_benchmarks_lookup
                ON backend_benchmarks(
                    backend, model_fingerprint, corpus_fingerprint,
                    settings_fingerprint
                );
            """
        )

    @staticmethod
    def _migrate_schema_v8(con: sqlite3.Connection) -> None:
        DatabaseManager._ensure_column(
            con,
            "files",
            "parse_diagnostics_json",
            "TEXT",
        )
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_scope_exclusions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id INTEGER NOT NULL,
                root_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                source_size_bytes INTEGER,
                source_modified_time REAL,
                source_quick_fingerprint TEXT,
                source_content_hash TEXT,
                parse_status TEXT NOT NULL,
                parse_error_code TEXT,
                parse_error_message_digest TEXT,
                parser_name TEXT,
                parser_version TEXT,
                reason TEXT NOT NULL,
                operation_source TEXT NOT NULL,
                candidate_index_version_id INTEGER,
                published_index_version_id INTEGER,
                created_at TEXT NOT NULL,
                revoked_at TEXT,
                revocation_reason TEXT,
                revoked_by TEXT,
                invalidated_at TEXT,
                invalidation_reason TEXT,
                FOREIGN KEY(file_id) REFERENCES files(id) ON DELETE CASCADE,
                FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE,
                FOREIGN KEY(candidate_index_version_id) REFERENCES index_versions(id),
                FOREIGN KEY(published_index_version_id) REFERENCES index_versions(id)
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_scope_exclusions_active_file
                ON index_scope_exclusions(file_id)
                WHERE revoked_at IS NULL AND invalidated_at IS NULL;
            CREATE INDEX IF NOT EXISTS idx_scope_exclusions_root
                ON index_scope_exclusions(root_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_scope_exclusions_history
                ON index_scope_exclusions(file_id, created_at);
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
            WHERE f.is_deleted = 0
              AND NOT EXISTS (
                  SELECT 1 FROM index_scope_exclusions e
                  WHERE e.file_id = f.id
                    AND e.revoked_at IS NULL
                    AND e.invalidated_at IS NULL
              )
              AND NOT EXISTS (SELECT 1 FROM files_fts ft WHERE ft.rowid = f.id)
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
                    status='pending'
                """,
                (str(path), int(enabled), int(include_subfolders), "{}", now, now, "pending"),
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

    def upsert_precomputed_file_metadata_many(
        self,
        root_id: int,
        metadata_rows: Sequence[object],
        *,
        retry_failed_files: bool = False,
        compute_full_hash: bool = False,
        mark_processing: bool = False,
        parser_versions: dict[str, str | None] | None = None,
    ) -> tuple[list[tuple[Path, int, bool]], list[tuple[Path, Exception]]]:
        """Persist metadata already collected in a recoverable planning process."""

        versions = parser_versions or {}
        results: list[tuple[Path, int, bool]] = []
        errors: list[tuple[Path, Exception]] = []
        if not metadata_rows:
            return results, errors
        now = utc_now()
        with self.connect() as con:
            for row in metadata_rows:
                file_path = Path(str(getattr(row, "path")))
                try:
                    content_hash = getattr(row, "content_hash", None)
                    file_id, changed = self._upsert_file_metadata_in_connection(
                        con,
                        root_id,
                        file_path,
                        size=int(getattr(row, "size_bytes")),
                        modified_time=float(getattr(row, "modified_time")),
                        created_time=float(getattr(row, "created_time")),
                        fingerprint=(
                            f"{int(getattr(row, 'size_bytes'))}:"
                            f"{int(getattr(row, 'modified_time_ns'))}"
                        ),
                        content_hash=str(content_hash) if content_hash else None,
                        now=now,
                        retry_failed_files=retry_failed_files,
                        compute_full_hash=compute_full_hash,
                        mark_processing=mark_processing,
                        expected_parser_version=versions.get(str(file_path)),
                    )
                    results.append((file_path, file_id, changed))
                except Exception as exc:
                    errors.append((file_path, exc))
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
            "SELECT id, extension, quick_fingerprint, content_hash, parse_status, parser_version, parse_error_code FROM files WHERE path = ?",
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
        source_identity_changed = bool(
            existing["quick_fingerprint"] != fingerprint
            or (
                compute_full_hash
                and content_hash is not None
                and existing["content_hash"] != content_hash
            )
        )
        if source_identity_changed:
            self._invalidate_active_scope_exclusion(
                con,
                int(existing["id"]),
                reason="source_identity_changed",
                invalidated_at=now,
            )
        retry_incomplete = (
            retry_failed_files
            and str(existing["parse_status"]) != "success"
            and not (
                str(existing["extension"] or "") in VIDEO_EXTENSIONS
                and str(existing["parse_status"]) == "metadata_only"
            )
        )
        changed = (
            existing["quick_fingerprint"] != fingerprint
            or (expected_parser_version is not None and existing["parser_version"] != expected_parser_version)
            or (compute_full_hash and existing["content_hash"] != content_hash)
            or existing["parse_status"] in {"pending", "processing", "cancelled", "deleted"}
            or retry_incomplete
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

    def sync_zip_members(
        self,
        root_id: int,
        container_file_id: int,
        archive_path: Path,
        members: Sequence[Any],
        parser_versions: dict[str, str],
        *,
        retry_failed_files: bool = False,
    ) -> list[tuple[Any, int, bool, str]]:
        """Upsert independently searchable sources for a validated ZIP manifest."""

        now = utc_now()
        archive_stat = archive_path.stat()
        results: list[tuple[Any, int, bool, str]] = []
        active_paths: set[str] = set()
        with self.connect() as con:
            for member in members:
                internal_path = str(member.internal_path)
                display_path = f"{archive_path} > {internal_path}"
                active_paths.add(display_path)
                filename = Path(internal_path).name
                extension = Path(internal_path).suffix.lower()
                expected_version = parser_versions[display_path]
                member_sha = str(member.sha256) if member.sha256 else None
                fingerprint = f"zip:{int(member.size_bytes)}:{int(member.crc32)}"
                existing = con.execute(
                    """
                    SELECT id, quick_fingerprint, parser_version, parse_status,
                        content_hash, content_hash_full
                    FROM files WHERE path = ?
                    """,
                    (display_path,),
                ).fetchone()
                if existing is None:
                    cursor = con.execute(
                        """
                        INSERT INTO files(
                            root_id, path, filename, extension, size_bytes, modified_time,
                            created_time, quick_fingerprint, content_hash, content_hash_full,
                            parse_status, parser_version, last_seen_at, is_deleted, source_kind,
                            container_file_id, internal_path, member_order, member_crc32,
                            member_uncompressed_size
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, 0,
                                  'zip_member', ?, ?, ?, ?, ?)
                        """,
                        (
                            root_id,
                            display_path,
                            filename,
                            extension,
                            int(member.size_bytes),
                            archive_stat.st_mtime,
                            archive_stat.st_ctime,
                            fingerprint,
                            member_sha,
                            member_sha,
                            expected_version,
                            now,
                            container_file_id,
                            internal_path,
                            int(member.member_index),
                            int(member.crc32),
                            int(member.size_bytes),
                        ),
                    )
                    file_id = int(cursor.lastrowid)
                    changed = True
                else:
                    retry_incomplete = retry_failed_files and str(existing["parse_status"]) != "success"
                    stored_hash = member_sha or str(existing["content_hash_full"] or existing["content_hash"] or "") or None
                    changed = (
                        str(existing["quick_fingerprint"] or "") != fingerprint
                        or str(existing["parser_version"] or "") != expected_version
                        or str(existing["parse_status"] or "")
                        in {"pending", "processing", "cancelled", "deleted"}
                        or retry_incomplete
                    )
                    file_id = int(existing["id"])
                    con.execute(
                        """
                        UPDATE files SET root_id = ?, filename = ?, extension = ?, size_bytes = ?,
                            modified_time = ?, created_time = ?, quick_fingerprint = ?,
                            content_hash = ?, content_hash_full = ?, parser_version = ?,
                            last_seen_at = ?, is_deleted = 0, source_kind = 'zip_member',
                            container_file_id = ?, internal_path = ?, member_order = ?,
                            member_crc32 = ?, member_uncompressed_size = ?,
                            parse_status = CASE WHEN ? THEN 'pending' ELSE parse_status END
                        WHERE id = ?
                        """,
                        (
                            root_id,
                            filename,
                            extension,
                            int(member.size_bytes),
                            archive_stat.st_mtime,
                            archive_stat.st_ctime,
                            fingerprint,
                            stored_hash,
                            stored_hash,
                            expected_version,
                            now,
                            container_file_id,
                            internal_path,
                            int(member.member_index),
                            int(member.crc32),
                            int(member.size_bytes),
                            int(changed),
                            file_id,
                        ),
                    )
                self._upsert_file_fts(con, file_id, filename, display_path)
                results.append((member, file_id, bool(changed), display_path))

            stale_rows = con.execute(
                "SELECT id, path FROM files WHERE container_file_id = ? AND is_deleted = 0",
                (container_file_id,),
            ).fetchall()
            stale_ids = [
                int(row["id"])
                for row in stale_rows
                if str(row["path"]) not in active_paths
            ]
            self._mark_deleted_file_ids(con, stale_ids)

            old_document_ids = self._document_ids_for_files(con, [container_file_id])
            con.execute(
                """
                UPDATE files SET document_id = NULL, content_key = NULL,
                    parse_status = 'success', parse_error_code = NULL,
                    parse_error_message = NULL, parser_name = 'zip_manifest', indexed_at = ?
                WHERE id = ?
                """,
                (now, container_file_id),
            )
            self._garbage_collect_documents(con, old_document_ids)
        return results

    def zip_container_requires_sync(self, container_file_id: int) -> bool:
        """Return whether interrupted or failed virtual members need manifest recovery."""

        with self.connect() as con:
            row = con.execute(
                """
                SELECT 1 FROM files
                WHERE container_file_id = ? AND is_deleted = 0
                  AND parse_status NOT IN ('success', 'metadata_only')
                LIMIT 1
                """,
                (container_file_id,),
            ).fetchone()
            return row is not None

    def active_zip_member_candidate_keys(self) -> set[tuple[str, int]]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT DISTINCT extension, member_uncompressed_size
                FROM files
                WHERE source_kind = 'zip_member' AND is_deleted = 0
                  AND content_hash_full IS NOT NULL
                """
            ).fetchall()
        return {
            (str(row["extension"] or ""), int(row["member_uncompressed_size"] or 0))
            for row in rows
        }

    def file_content_states(self, file_ids: Sequence[int]) -> dict[int, tuple[str | None, str | None]]:
        ids = list(dict.fromkeys(int(file_id) for file_id in file_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as con:
            rows = con.execute(
                f"SELECT id, content_hash_full, content_key FROM files WHERE id IN ({placeholders})",
                tuple(ids),
            ).fetchall()
        return {
            int(row["id"]): (
                str(row["content_hash_full"]) if row["content_hash_full"] else None,
                str(row["content_key"]) if row["content_key"] else None,
            )
            for row in rows
        }

    def set_content_hash_full(self, file_id: int, content_hash_full: str) -> None:
        with self.connect() as con:
            con.execute(
                "UPDATE files SET content_hash = ?, content_hash_full = ? WHERE id = ?",
                (content_hash_full, content_hash_full, int(file_id)),
            )

    @staticmethod
    def _upsert_file_fts(
        con: sqlite3.Connection,
        file_id: int,
        filename: str,
        path: str,
    ) -> None:
        con.execute("DELETE FROM files_fts WHERE rowid = ?", (file_id,))
        excluded = con.execute(
            """
            SELECT 1 FROM index_scope_exclusions
            WHERE file_id = ?
              AND revoked_at IS NULL
              AND invalidated_at IS NULL
            """,
            (int(file_id),),
        ).fetchone()
        if excluded is not None:
            return
        con.execute(
            "INSERT INTO files_fts(rowid, file_id, filename, path) VALUES (?, ?, ?, ?)",
            (file_id, file_id, filename, path),
        )

    @staticmethod
    def _invalidate_active_scope_exclusion(
        con: sqlite3.Connection,
        file_id: int,
        *,
        reason: str,
        invalidated_at: str | None = None,
    ) -> int:
        cursor = con.execute(
            """
            UPDATE index_scope_exclusions
            SET invalidated_at = ?, invalidation_reason = ?
            WHERE file_id = ?
              AND revoked_at IS NULL
              AND invalidated_at IS NULL
            """,
            (invalidated_at or utc_now(), str(reason), int(file_id)),
        )
        return int(cursor.rowcount)

    def mark_processing(self, file_id: int) -> None:
        with self.connect() as con:
            con.execute("UPDATE files SET parse_status = 'processing' WHERE id = ?", (file_id,))

    def mark_video_excluded(self, file_ids: Sequence[int]) -> None:
        """Keep video filename/path metadata without creating searchable content blocks."""
        ids = [int(value) for value in file_ids]
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        now = utc_now()
        with self.connect() as con:
            old_document_ids = self._document_ids_for_files(con, ids)
            con.execute(
                f"DELETE FROM content_fts WHERE rowid IN ("
                f"SELECT id FROM content_blocks WHERE file_id IN ({placeholders}) "
                "AND document_id IS NULL)",
                tuple(ids),
            )
            con.execute(
                f"DELETE FROM content_blocks WHERE file_id IN ({placeholders}) "
                "AND document_id IS NULL",
                tuple(ids),
            )
            con.execute(
                f"""
                UPDATE files SET
                    document_id = NULL, content_key = NULL, parse_status = 'metadata_only',
                    parse_error_code = NULL, parse_error_message = NULL,
                    parser_name = 'video_metadata', parser_version = ?, indexed_at = ?
                WHERE id IN ({placeholders})
                """,
                (PARSER_VERSION, now, *ids),
            )
            self._garbage_collect_documents(con, old_document_ids)

    def invalidate_file(self, path: str | Path) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT id, container_file_id FROM files WHERE path = ? AND is_deleted = 0",
                (str(path),),
            ).fetchone()
            if row is None:
                return False
            target_id = int(row["container_file_id"] or row["id"])
            cursor = con.execute(
                "UPDATE files SET parse_status = 'pending' WHERE id = ? AND is_deleted = 0",
                (target_id,),
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
                  AND parse_status IN ('success', 'metadata_only')
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
                    WHERE parse_status IN ('success', 'metadata_only')
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
        diagnostics = list(item.get("diagnostics") or [])
        diagnostics_json = (
            json.dumps(
                diagnostics,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if diagnostics
            else None
        )
        content_key = str(item.get("content_key") or f"file:{primary_file_id}:{parser_name}")
        content_hash_full_value = item.get("content_hash_full")
        content_hash_full = str(content_hash_full_value) if content_hash_full_value else None
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
                parse_error_message = ?, parse_diagnostics_json = ?,
                parser_name = ?, parser_version = ?, indexed_at = ?,
                content_hash = COALESCE(?, content_hash),
                content_hash_full = COALESCE(?, content_hash_full),
                is_deleted = 0
            WHERE id IN ({placeholders})
            """,
            (
                document_id,
                content_key,
                status,
                error_code,
                _trim_message(error_message),
                diagnostics_json,
                parser_name,
                parser_version,
                now,
                content_hash_full,
                content_hash_full,
                *file_ids,
            ),
        )
        task_id = item.get("task_id")
        if task_id is not None:
            # A policy skip (for example a ZIP safety limit) is a completed
            # parser decision. The file remains visible as a blocker, but the
            # task must not remain failed and independently block publication.
            task_status = (
                "complete"
                if status in SUCCESSFUL_DOCUMENT_STATUSES or status == "skipped"
                else "failed"
            )
            con.execute(
                """
                UPDATE parse_tasks SET status = ?, written_at = ?, finished_at = ?,
                    error_code = ?, error_message = ? WHERE id = ?
                """,
                (
                    task_status,
                    now,
                    now,
                    error_code,
                    _trim_message(error_message),
                    int(task_id),
                ),
            )
            con.execute(
                """
                UPDATE parse_task_attempts
                SET status = ?, finished_at = ?, error_code = ?, error_message = ?
                WHERE id = (
                    SELECT id FROM parse_task_attempts
                    WHERE task_id = ?
                    ORDER BY attempt_no DESC
                    LIMIT 1
                )
                """,
                (
                    task_status,
                    now,
                    error_code,
                    _trim_message(error_message),
                    int(task_id),
                ),
            )
            con.execute(
                """
                UPDATE parse_tasks
                SET status = ?, finished_at = COALESCE(finished_at, ?),
                    error_code = CASE
                        WHEN ? = 'failed' THEN COALESCE(error_code, ?)
                        ELSE error_code
                    END
                WHERE parent_task_id = ?
                  AND status NOT IN (
                      'complete', 'failed', 'superseded', 'invalidated', 'cancelled'
                  )
                """,
                (
                    task_status,
                    now,
                    task_status,
                    error_code,
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
                    parse_diagnostics_json = NULL,
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
                    parse_diagnostics_json = NULL,
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
                """
                SELECT path FROM files
                WHERE root_id = ? AND is_deleted = 0 AND source_kind = 'file'
                """,
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
            container_ids = [int(row["id"]) for row in rows]
            file_ids = list(container_ids)
            if container_ids:
                container_placeholders = ",".join("?" for _ in container_ids)
                child_rows = con.execute(
                    f"SELECT id FROM files WHERE container_file_id IN ({container_placeholders})",
                    tuple(container_ids),
                ).fetchall()
                file_ids.extend(int(row["id"]) for row in child_rows)
            if not file_ids:
                return 0
            self._mark_deleted_file_ids(con, file_ids)
            return len(file_ids)

    def _mark_deleted_file_ids(
        self,
        con: sqlite3.Connection,
        file_ids: Sequence[int],
    ) -> None:
        ids = list(dict.fromkeys(int(value) for value in file_ids))
        if not ids:
            return
        placeholders = ",".join("?" for _ in ids)
        document_ids = self._document_ids_for_files(con, ids)
        for document_id in document_ids:
            replacement = con.execute(
                f"""
                SELECT id FROM files
                WHERE document_id = ? AND id NOT IN ({placeholders}) AND is_deleted = 0
                LIMIT 1
                """,
                (document_id, *ids),
            ).fetchone()
            if replacement is not None:
                con.execute(
                    f"""
                    UPDATE content_blocks SET file_id = ?
                    WHERE document_id = ? AND file_id IN ({placeholders})
                    """,
                    (int(replacement["id"]), document_id, *ids),
                )
        con.execute(
            f"""
            DELETE FROM content_fts
            WHERE file_id IN ({placeholders})
              AND block_id IN (SELECT id FROM content_blocks WHERE document_id IS NULL)
            """,
            tuple(ids),
        )
        con.execute(
            f"""
            UPDATE files SET document_id = NULL, is_deleted = 1, parse_status = 'deleted'
            WHERE id IN ({placeholders})
            """,
            tuple(ids),
        )
        self._garbage_collect_documents(con, document_ids)

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
            for task_id in task_ids:
                con.execute(
                    """
                    INSERT INTO parse_task_attempts(
                        task_id, attempt_no, status, started_at
                    )
                    SELECT ?, COALESCE(MAX(attempt_no), 0) + 1, 'running', ?
                    FROM parse_task_attempts
                    WHERE task_id = ?
                    """,
                    (int(task_id), now, int(task_id)),
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
            con.execute(
                """
                UPDATE parse_task_attempts SET status = 'spooled'
                WHERE id = (
                    SELECT id FROM parse_task_attempts
                    WHERE task_id = ?
                    ORDER BY attempt_no DESC
                    LIMIT 1
                )
                """,
                (task_id,),
            )

    def try_update_task_progress(
        self,
        task_id: int,
        *,
        phase: str,
        completed: int,
        total: int,
        unit_type: str,
        cursor: str,
        bytes_read: int,
        output_blocks: int,
        checkpoint_version: int,
        worker_pid: int | None,
        checkpoint_path: str | None,
        timeout_seconds: float = 0.0,
    ) -> bool:
        """Persist semantic progress without ever blocking the parser watchdog."""

        try:
            now = utc_now()
            with self.connect(timeout_seconds=timeout_seconds) as con:
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET progress_phase = ?, progress_completed = ?,
                        progress_total = ?, progress_unit_type = ?,
                        progress_cursor = ?, progress_bytes_read = ?,
                        progress_output_blocks = ?, checkpoint_version = ?,
                        last_semantic_progress_at = ?, worker_pid = ?,
                        checkpoint_path = ?
                    WHERE id = ?
                    """,
                    (
                        phase,
                        max(0, int(completed)),
                        max(0, int(total)),
                        unit_type,
                        cursor,
                        max(0, int(bytes_read)),
                        max(0, int(output_blocks)),
                        max(0, int(checkpoint_version)),
                        now,
                        worker_pid,
                        checkpoint_path,
                        int(task_id),
                    ),
                )
                con.execute(
                    """
                    UPDATE parse_task_attempts
                    SET last_progress_at = ?, worker_pid = COALESCE(?, worker_pid)
                    WHERE id = (
                        SELECT id FROM parse_task_attempts
                        WHERE task_id = ?
                        ORDER BY attempt_no DESC
                        LIMIT 1
                    )
                    """,
                    (now, worker_pid, int(task_id)),
                )
            return True
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                return False
            raise

    def try_record_child_task_progress(
        self,
        parent_task_id: int,
        *,
        task_type: str,
        unit_key: str,
        status: str,
        phase: str,
        completed: int,
        total: int,
        worker_pid: int | None,
        timeout_seconds: float = 0.0,
    ) -> bool:
        """Upsert a durable page/region task discovered by a parser worker."""

        try:
            now = utc_now()
            with self.connect(timeout_seconds=timeout_seconds) as con:
                con.execute(
                    """
                    INSERT INTO parse_tasks(
                        file_id, parent_task_id, run_id, task_type, unit_key,
                        status, priority, created_at, queued_at, started_at,
                        finished_at, progress_phase, progress_completed,
                        progress_total, last_semantic_progress_at, worker_pid
                    )
                    SELECT file_id, id, run_id, ?, ?, ?, priority, ?, ?, ?,
                           CASE WHEN ? = 'complete' THEN ? ELSE NULL END,
                           ?, ?, ?, ?, ?
                    FROM parse_tasks
                    WHERE id = ?
                    ON CONFLICT(parent_task_id, task_type, unit_key) DO UPDATE SET
                        status = excluded.status,
                        started_at = COALESCE(parse_tasks.started_at, excluded.started_at),
                        finished_at = excluded.finished_at,
                        progress_phase = excluded.progress_phase,
                        progress_completed = excluded.progress_completed,
                        progress_total = excluded.progress_total,
                        last_semantic_progress_at = excluded.last_semantic_progress_at,
                        worker_pid = excluded.worker_pid
                    """,
                    (
                        task_type,
                        unit_key,
                        status,
                        now,
                        now,
                        now,
                        status,
                        now,
                        phase,
                        max(0, int(completed)),
                        max(0, int(total)),
                        now,
                        worker_pid,
                        int(parent_task_id),
                    ),
                )
            return True
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                return False
            raise

    def mark_task_failed(self, task_id: int, error_code: str, message: str) -> None:
        with self.connect() as con:
            now = utc_now()
            con.execute(
                """
                UPDATE parse_tasks SET status = 'failed', finished_at = ?, error_code = ?,
                    error_message = ? WHERE id = ?
                """,
                (now, error_code, _trim_message(message), task_id),
            )
            con.execute(
                """
                UPDATE parse_task_attempts
                SET status = 'failed', finished_at = ?, error_code = ?,
                    error_message = ?
                WHERE id = (
                    SELECT id FROM parse_task_attempts
                    WHERE task_id = ?
                    ORDER BY attempt_no DESC
                    LIMIT 1
                )
                """,
                (now, error_code, _trim_message(message), task_id),
            )

    def mark_task_attempt_interrupted(
        self,
        task_id: int,
        error_code: str,
        message: str,
    ) -> None:
        """Close the current attempt while leaving its task retryable."""

        with self.connect() as con:
            con.execute(
                """
                UPDATE parse_task_attempts
                SET status = 'interrupted', finished_at = ?,
                    error_code = ?, error_message = ?
                WHERE id = (
                    SELECT id FROM parse_task_attempts
                    WHERE task_id = ? AND status = 'running'
                    ORDER BY attempt_no DESC
                    LIMIT 1
                )
                """,
                (
                    utc_now(),
                    str(error_code),
                    _trim_message(message),
                    int(task_id),
                ),
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
        now = utc_now()
        con.execute(
            """
            UPDATE index_runs SET status = 'interrupted', finished_at = ?
            WHERE status = 'running'
            """,
            (now,),
        )
        con.execute(
            """
            UPDATE parse_task_attempts
            SET status = 'interrupted', finished_at = ?
            WHERE status IN ('running', 'spooled')
            """,
            (now,),
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
        con.execute(
            """
            UPDATE ocr_requests
            SET status = 'queued', lease_owner = NULL,
                lease_expires_at = NULL
            WHERE status IN ('running', 'spooled', 'paused')
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
                f"""
                DELETE FROM index_file_metrics
                WHERE run_id IN (
                    SELECT id FROM index_runs ORDER BY started_at DESC LIMIT -1 OFFSET {INDEX_RUN_RETENTION}
                )
                """
            )
            con.execute(
                f"""
                DELETE FROM index_runs
                WHERE id IN (
                    SELECT id FROM index_runs ORDER BY started_at DESC LIMIT -1 OFFSET {INDEX_RUN_RETENTION}
                )
                """
            )

    def recent_index_runs_since(self, started_at: str) -> list[dict[str, object]]:
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT id, started_at, finished_at, mode, status, discovered_files,
                    discovered_bytes, scan_ms, parse_ms, write_ms, fts_ms, total_ms,
                    peak_rss_bytes, summary_json
                FROM index_runs
                WHERE started_at >= ?
                ORDER BY started_at ASC
                """,
                (started_at,),
            ).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            summary_json = str(row["summary_json"] or "")
            try:
                parsed_summary = json.loads(summary_json) if summary_json else {}
            except json.JSONDecodeError:
                parsed_summary = {}
            results.append(
                {
                    "id": str(row["id"]),
                    "started_at": str(row["started_at"]),
                    "finished_at": str(row["finished_at"] or ""),
                    "mode": str(row["mode"]),
                    "status": str(row["status"]),
                    "discovered_files": int(row["discovered_files"] or 0),
                    "discovered_bytes": int(row["discovered_bytes"] or 0),
                    "scan_ms": int(row["scan_ms"] or 0),
                    "parse_ms": int(row["parse_ms"] or 0),
                    "write_ms": int(row["write_ms"] or 0),
                    "fts_ms": int(row["fts_ms"] or 0),
                    "total_ms": int(row["total_ms"] or 0),
                    "peak_rss_bytes": int(row["peak_rss_bytes"] or 0),
                    "summary": parsed_summary,
                }
            )
        return results

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

    def root_completion(self, root_id: int) -> dict[str, int]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        with self.connect() as con:
            row = con.execute(
                f"""
                SELECT
                    COUNT(*) AS discovered,
                    SUM(CASE WHEN f.extension IN ({video_placeholders}) THEN 1 ELSE 0 END) AS video_excluded,
                    SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                  AND e.id IS NULL THEN 1 ELSE 0 END) AS eligible,
                    SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                  AND e.id IS NULL
                                  AND f.parse_status IN ('success', 'metadata_only')
                             THEN 1 ELSE 0 END) AS complete,
                    SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END) AS manual_excluded,
                    SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                  AND e.id IS NULL
                                  AND f.parse_status = 'metadata_only'
                             THEN 1 ELSE 0 END) AS metadata_only_complete
                FROM files f
                LEFT JOIN index_scope_exclusions e
                  ON e.file_id = f.id
                 AND e.revoked_at IS NULL
                 AND e.invalidated_at IS NULL
                WHERE f.root_id = ? AND f.is_deleted = 0
                """,
                (
                    *video_extensions,
                    *video_extensions,
                    *video_extensions,
                    *video_extensions,
                    root_id,
                ),
            ).fetchone()
        eligible = int(row["eligible"] or 0)
        complete = int(row["complete"] or 0)
        return {
            "discovered": int(row["discovered"] or 0),
            "video_excluded": int(row["video_excluded"] or 0),
            "eligible": eligible,
            "complete": complete,
            "blocking": max(0, eligible - complete),
            "manual_excluded": int(row["manual_excluded"] or 0),
            "metadata_only_complete": int(
                row["metadata_only_complete"] or 0
            ),
        }

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
                WHERE EXISTS (
                    SELECT 1
                    FROM files visible
                    WHERE visible.is_deleted = 0
                      AND (
                          (cb.document_id IS NOT NULL AND visible.document_id = cb.document_id)
                          OR (cb.document_id IS NULL AND visible.id = cb.file_id)
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = visible.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                )
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
                SELECT f.id, f.id, f.filename, f.path
                FROM files f
                WHERE f.is_deleted = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
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

    def exclude_files_from_index(
        self,
        file_ids: Sequence[int],
        *,
        reason: str,
        operation_source: str = "ui",
        candidate_index_version_id: int | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> int:
        ids = list(dict.fromkeys(int(value) for value in file_ids))
        reason_text = str(reason).strip()
        source_text = str(operation_source).strip() or "ui"
        if not ids:
            return 0
        if not reason_text:
            raise ValueError("人工排除原因不能为空")

        def report(
            stage: str,
            phase_label: str,
            *,
            processed_files: int = 0,
            total_files: int = len(ids),
            large_fts_operation: bool = False,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": stage,
                        "phase_label": phase_label,
                        "processed_files": processed_files,
                        "total_files": total_files,
                        "large_fts_operation": large_fts_operation,
                        "can_cancel": True,
                    }
                )

        def throw_if_cancelled() -> None:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError("人工排除任务已取消")

        placeholders = ",".join("?" for _ in ids)
        now = utc_now()
        report("validating", "正在校验选中文件")
        throw_if_cancelled()
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT f.*, e.id AS active_exclusion_id
                FROM files f
                LEFT JOIN index_scope_exclusions e
                  ON e.file_id = f.id
                 AND e.revoked_at IS NULL
                 AND e.invalidated_at IS NULL
                WHERE f.id IN ({placeholders}) AND f.is_deleted = 0
                ORDER BY f.id
                """,
                tuple(ids),
            ).fetchall()
            found_ids = {int(row["id"]) for row in rows}
            missing = [file_id for file_id in ids if file_id not in found_ids]
            if missing:
                raise ValueError(
                    "文件不存在或已删除：" + ", ".join(str(value) for value in missing)
                )
            terminal_placeholders = ",".join(
                "?" for _ in TERMINAL_PARSE_TASK_STATUSES
            )
            active_tasks = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM parse_tasks
                    WHERE file_id IN ({placeholders})
                      AND status NOT IN ({terminal_placeholders})
                    """,
                    (*ids, *sorted(TERMINAL_PARSE_TASK_STATUSES)),
                ).fetchone()[0]
            )
            if active_tasks:
                raise RuntimeError("所选文件仍有活动解析任务，请先暂停或等待安全点")
            inserted_ids: list[int] = []
            report("recording_exclusions", "正在记录排除范围")
            for row in rows:
                throw_if_cancelled()
                if row["active_exclusion_id"] is not None:
                    continue
                parse_status = str(row["parse_status"] or "")
                if parse_status not in MANUALLY_EXCLUDABLE_STATUSES:
                    raise ValueError(
                        f"文件状态不允许人工排除：{row['path']} ({parse_status})"
                    )
                error_message = str(row["parse_error_message"] or "")
                error_digest = (
                    hashlib.sha256(error_message.encode("utf-8")).hexdigest()
                    if error_message
                    else None
                )
                con.execute(
                    """
                    INSERT INTO index_scope_exclusions(
                        file_id, root_id, path, source_size_bytes,
                        source_modified_time, source_quick_fingerprint,
                        source_content_hash, parse_status, parse_error_code,
                        parse_error_message_digest, parser_name, parser_version,
                        reason, operation_source, candidate_index_version_id,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(row["id"]),
                        int(row["root_id"]),
                        str(row["path"]),
                        row["size_bytes"],
                        row["modified_time"],
                        row["quick_fingerprint"],
                        row["content_hash_full"] or row["content_hash"],
                        parse_status,
                        row["parse_error_code"],
                        error_digest,
                        row["parser_name"],
                        row["parser_version"],
                        reason_text,
                        source_text,
                        candidate_index_version_id,
                        now,
                    ),
                )
                inserted_ids.append(int(row["id"]))
                report(
                    "recording_exclusions",
                    "正在记录排除范围",
                    processed_files=len(inserted_ids),
                )
            if not inserted_ids:
                return 0
            throw_if_cancelled()
            self._after_scope_exclusion_audit(con, inserted_ids)
            inserted_placeholders = ",".join("?" for _ in inserted_ids)
            report(
                "cleaning_content_fts",
                "正在清理正文索引",
                processed_files=len(inserted_ids),
                large_fts_operation=True,
            )
            throw_if_cancelled()
            con.execute(
                f"DELETE FROM files_fts WHERE file_id IN ({inserted_placeholders})",
                tuple(inserted_ids),
            )
            con.execute(
                """
                DELETE FROM content_fts
                WHERE rowid IN (
                    SELECT cb.id
                    FROM content_blocks cb
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM files visible
                        WHERE visible.is_deleted = 0
                          AND (
                              (cb.document_id IS NOT NULL AND visible.document_id = cb.document_id)
                              OR (cb.document_id IS NULL AND visible.id = cb.file_id)
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM index_scope_exclusions active
                              WHERE active.file_id = visible.id
                                AND active.revoked_at IS NULL
                                AND active.invalidated_at IS NULL
                          )
                    )
                )
                """
            )
            throw_if_cancelled()
            published_roots = self._publish_current_scope_if_unblocked(
                con,
                now,
                progress_callback=progress_callback,
                cancel_requested=cancel_requested,
            )
            if published_roots <= 0:
                report(
                    "refreshing_index_state",
                    "正在刷新索引状态",
                    processed_files=len(inserted_ids),
                )
            throw_if_cancelled()
            return len(inserted_ids)

    def _after_scope_exclusion_audit(
        self,
        connection: sqlite3.Connection,
        file_ids: list[int],
    ) -> None:
        """Test seam after audit writes but before searchable rows change."""
        del connection, file_ids

    def _publish_current_scope_if_unblocked(
        self,
        con: sqlite3.Connection,
        now: str,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> int:
        """Publish the current searchable scope after its last blocker is removed."""

        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        video_placeholders = ",".join("?" for _ in video_extensions)
        enabled_roots = int(
            con.execute(
                "SELECT COUNT(*) FROM roots WHERE enabled = 1"
            ).fetchone()[0]
        )
        if enabled_roots <= 0:
            return 0
        blockers = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM files f
                WHERE f.is_deleted = 0
                  AND f.extension NOT IN ({video_placeholders})
                  AND f.parse_status NOT IN ('success', 'metadata_only')
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """,
                video_extensions,
            ).fetchone()[0]
        )
        if blockers:
            return 0
        self._reconcile_residual_parse_tasks(con, now)
        terminal_placeholders = ",".join(
            "?" for _ in TERMINAL_PARSE_TASK_STATUSES
        )
        active_tasks = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM parse_tasks pt
                JOIN files f ON f.id = pt.file_id
                WHERE f.is_deleted = 0
                  AND pt.status NOT IN ({terminal_placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """,
                tuple(sorted(TERMINAL_PARSE_TASK_STATUSES)),
            ).fetchone()[0]
        )
        if active_tasks:
            return 0

        def report(stage: str, phase_label: str) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": stage,
                        "phase_label": phase_label,
                        "processed_files": 0,
                        "total_files": 0,
                        "large_fts_operation": stage
                        in {"rebuilding_content_fts", "updating_filename_fts"},
                        "can_cancel": True,
                    }
                )

        def throw_if_cancelled() -> None:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError("人工排除任务已取消")

        report("rebuilding_content_fts", "正在重建全文索引")
        throw_if_cancelled()
        con.execute("DELETE FROM content_fts")
        con.execute(
            """
            INSERT INTO content_fts(
                rowid, block_id, file_id, filename, path,
                location_text, normalized_text
            )
            SELECT cb.id, cb.id, cb.file_id, COALESCE(f.filename, ''),
                   COALESCE(f.path, ''), cb.location_text,
                   cb.normalized_text
            FROM content_blocks cb
            LEFT JOIN files f ON f.id = cb.file_id
            WHERE EXISTS (
                SELECT 1 FROM files visible
                WHERE visible.is_deleted = 0
                  AND (
                      (cb.document_id IS NOT NULL
                       AND visible.document_id = cb.document_id)
                      OR (cb.document_id IS NULL AND visible.id = cb.file_id)
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = visible.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
            )
            """
        )
        report("updating_filename_fts", "正在更新文件名索引")
        throw_if_cancelled()
        con.execute("DELETE FROM files_fts")
        con.execute(
            """
            INSERT INTO files_fts(rowid, file_id, filename, path)
            SELECT f.id, f.id, f.filename, f.path
            FROM files f
            WHERE f.is_deleted = 0
              AND NOT EXISTS (
                  SELECT 1 FROM index_scope_exclusions e
                  WHERE e.file_id = f.id
                    AND e.revoked_at IS NULL
                    AND e.invalidated_at IS NULL
              )
            """
        )
        report("refreshing_index_state", "正在刷新索引状态")
        throw_if_cancelled()
        con.execute(
            """
            UPDATE index_runs
            SET status = 'incomplete', finished_at = COALESCE(finished_at, ?)
            WHERE status = 'running'
            """,
            (now,),
        )
        con.execute(
            """
            UPDATE roots
            SET status = 'ready', last_scan_at = ?, updated_at = ?
            WHERE enabled = 1
            """,
            (now, now),
        )
        self._set_index_state(con, "content_fts_dirty", "0")
        self._set_index_state(con, "full_batch_incomplete", "0")
        self._discard_staging_index_versions(
            con,
            now,
            "Current scope was rebuilt after manual exclusion",
        )
        return enabled_roots

    def _reconcile_residual_parse_tasks(
        self,
        con: sqlite3.Connection,
        now: str,
    ) -> dict[str, int]:
        terminal_placeholders = ",".join(
            "?" for _ in TERMINAL_PARSE_TASK_STATUSES
        )
        terminal_values = tuple(sorted(TERMINAL_PARSE_TASK_STATUSES))
        excluded_ids = [
            int(row["id"])
            for row in con.execute(
                f"""
                SELECT pt.id
                FROM parse_tasks pt
                JOIN files f ON f.id = pt.file_id
                WHERE f.is_deleted = 0
                  AND pt.status NOT IN ({terminal_placeholders})
                  AND EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """,
                terminal_values,
            ).fetchall()
        ]
        successful_ids = [
            int(row["id"])
            for row in con.execute(
                f"""
                SELECT pt.id
                FROM parse_tasks pt
                JOIN files f ON f.id = pt.file_id
                WHERE f.is_deleted = 0
                  AND f.parse_status IN ('success', 'metadata_only')
                  AND pt.status NOT IN ({terminal_placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """,
                terminal_values,
            ).fetchall()
        ]

        def finish_tasks(
            task_ids: list[int],
            *,
            status: str,
            error_code: str,
            message: str,
        ) -> None:
            if not task_ids:
                return
            placeholders = ",".join("?" for _ in task_ids)
            con.execute(
                f"""
                UPDATE parse_tasks
                SET status = ?, finished_at = COALESCE(finished_at, ?),
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_code = COALESCE(error_code, ?),
                    error_message = COALESCE(error_message, ?)
                WHERE id IN ({placeholders})
                """,
                (status, now, error_code, message, *task_ids),
            )
            con.execute(
                f"""
                UPDATE parse_task_attempts
                SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                    error_code = COALESCE(error_code, ?),
                    error_message = COALESCE(error_message, ?)
                WHERE task_id IN ({placeholders}) AND status = 'running'
                """,
                (now, error_code, message, *task_ids),
            )

        finish_tasks(
            excluded_ids,
            status="cancelled",
            error_code="SCOPE_EXCLUDED",
            message="文件已从当前索引范围排除",
        )
        finish_tasks(
            successful_ids,
            status="invalidated",
            error_code="STALE_TASK_AFTER_SUCCESS",
            message="文件已有成功终态，残留解析任务已失效",
        )
        return {
            "cancelled_tasks": len(excluded_ids),
            "invalidated_tasks": len(successful_ids),
        }

    @staticmethod
    def _discard_staging_index_versions(
        con: sqlite3.Connection,
        now: str,
        message: str,
    ) -> int:
        cursor = con.execute(
            """
            UPDATE index_versions
            SET status = 'failed', failed_at = COALESCE(failed_at, ?),
                error_message = COALESCE(error_message, ?)
            WHERE status = 'staging'
            """,
            (now, message),
        )
        return max(0, int(cursor.rowcount or 0))

    def restore_files_to_index(
        self,
        file_ids: Sequence[int],
        *,
        reason: str,
        operation_source: str = "ui",
    ) -> int:
        ids = list(dict.fromkeys(int(value) for value in file_ids))
        if not ids:
            return 0
        reason_text = str(reason).strip()
        if not reason_text:
            raise ValueError("恢复纳入原因不能为空")
        placeholders = ",".join("?" for _ in ids)
        now = utc_now()
        with self.connect() as con:
            cursor = con.execute(
                f"""
                UPDATE index_scope_exclusions
                SET revoked_at = ?, revocation_reason = ?, revoked_by = ?
                WHERE file_id IN ({placeholders})
                  AND revoked_at IS NULL
                  AND invalidated_at IS NULL
                """,
                (now, reason_text, str(operation_source).strip() or "ui", *ids),
            )
            restored = int(cursor.rowcount)
            if restored:
                rows = con.execute(
                    f"""
                    SELECT id, filename, path FROM files
                    WHERE id IN ({placeholders}) AND is_deleted = 0
                    """,
                    tuple(ids),
                ).fetchall()
                for row in rows:
                    self._upsert_file_fts(
                        con,
                        int(row["id"]),
                        str(row["filename"]),
                        str(row["path"]),
                    )
            return restored

    def excluded_files(
        self,
        limit: int = 500,
        *,
        include_history: bool = False,
    ) -> list[sqlite3.Row]:
        history_filter = (
            ""
            if include_history
            else "AND e.revoked_at IS NULL AND e.invalidated_at IS NULL"
        )
        with self.connect() as con:
            return list(
                con.execute(
                    f"""
                    SELECT e.*, 'manual_excluded' AS scope_state,
                           f.filename, f.extension, f.indexed_at,
                           f.parse_status AS current_parse_status,
                           f.parse_error_code AS current_error_code,
                           f.parse_error_message AS current_error_message,
                           f.parse_diagnostics_json
                    FROM index_scope_exclusions e
                    JOIN files f ON f.id = e.file_id
                    WHERE 1 = 1 {history_filter}
                    ORDER BY e.created_at DESC, e.id DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
            )

    def metadata_only_files(self, limit: int = 500) -> list[sqlite3.Row]:
        with self.connect() as con:
            return list(
                con.execute(
                    """
                    SELECT f.*, 'included' AS scope_state
                    FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status = 'metadata_only'
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = f.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                    ORDER BY f.indexed_at DESC
                    LIMIT ?
                    """,
                    (max(1, int(limit)),),
                ).fetchall()
            )

    def failed_files(self, limit: int = 500) -> list[sqlite3.Row]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        with self.connect() as con:
            return list(
                con.execute(
                    f"""
                    SELECT f.id, f.path, f.filename, f.extension, f.parse_status,
                           f.parse_error_code, f.parse_error_message,
                           f.parse_diagnostics_json,
                           f.parser_name, f.indexed_at,
                           'included' AS scope_state,
                           COALESCE(pt.progress_phase, '') AS progress_phase,
                           COALESCE(pt.progress_cursor, '') AS progress_cursor,
                           CASE
                               WHEN f.parse_error_code = 'PARSE_NO_PROGRESS'
                                   THEN '检查文件是否损坏或位于不稳定存储；修复后重新尝试'
                               WHEN f.parse_error_code = 'PROCESS_WORKER_CRASH'
                                   THEN '关闭占用该文件的程序后重试；若重复发生请保留诊断日志'
                               WHEN f.parse_error_code LIKE 'LEGACY_%'
                                    OR f.parse_error_code LIKE '%CONVERTER%'
                                   THEN '确认旧版 Office 文件可正常打开且本机转换器可用，然后重新尝试'
                               WHEN f.parse_status = 'password_protected'
                                   THEN '移除文件密码或提供未加密副本后重新尝试'
                               WHEN f.parse_status = 'ocr_failed'
                                   THEN '确认图片可读取；可单独打开文件检查后重新尝试'
                               ELSE '根据错误原因修复源文件或依赖后重新尝试'
                           END AS recovery_advice
                    FROM files AS f
                    LEFT JOIN parse_tasks AS pt
                      ON pt.id = (
                          SELECT latest.id
                          FROM parse_tasks AS latest
                          WHERE latest.file_id = f.id
                          ORDER BY latest.id DESC
                          LIMIT 1
                      )
                    WHERE f.parse_status IN (
                        'pending', 'processing', 'cancelled',
                        'failed', 'failed_retryable', 'unsupported', 'skipped',
                        'ocr_disabled', 'ocr_failed', 'converter_missing', 'partial_success',
                        'password_protected'
                    )
                      AND NOT (
                          f.extension IN ({video_placeholders})
                          AND f.parse_status = 'metadata_only'
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = f.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                    ORDER BY f.indexed_at DESC
                    LIMIT ?
                    """,
                    (*sorted(VIDEO_EXTENSIONS), limit),
                ).fetchall()
            )

    def force_complete_current_scope(
        self,
        *,
        reason: str,
        operation_source: str = "ui_force_complete",
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        cancel_requested: Callable[[], bool] | None = None,
    ) -> dict[str, int]:
        """Audit-exclude every remaining blocker and atomically open search."""

        reason_text = str(reason).strip()
        source_text = str(operation_source).strip() or "ui_force_complete"
        if not reason_text:
            raise ValueError("强力完成原因不能为空")
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        now = utc_now()

        def report(
            stage: str,
            phase_label: str,
            *,
            processed_files: int = 0,
            total_files: int = 0,
            large_fts_operation: bool = False,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": stage,
                        "phase_label": phase_label,
                        "processed_files": processed_files,
                        "total_files": total_files,
                        "large_fts_operation": large_fts_operation,
                        "can_cancel": True,
                    }
                )

        def throw_if_cancelled() -> None:
            if cancel_requested is not None and cancel_requested():
                raise CancelledError("索引状态修复已取消")

        report("validating", "正在诊断索引状态")
        throw_if_cancelled()
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT f.*
                FROM files f
                WHERE f.is_deleted = 0
                  AND f.extension NOT IN ({video_placeholders})
                  AND f.parse_status NOT IN ('success', 'metadata_only')
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                ORDER BY f.id
                """,
                video_extensions,
            ).fetchall()
            file_ids = [int(row["id"]) for row in rows]
            report(
                "recording_exclusions",
                "正在处理剩余阻断项",
                total_files=len(file_ids),
            )
            cancelled_tasks = 0
            if file_ids:
                placeholders = ",".join("?" for _ in file_ids)
                terminal_placeholders = ",".join(
                    "?" for _ in TERMINAL_PARSE_TASK_STATUSES
                )
                active_task_rows = con.execute(
                    f"""
                    SELECT id FROM parse_tasks
                    WHERE file_id IN ({placeholders})
                      AND status NOT IN ({terminal_placeholders})
                    """,
                    (
                        *file_ids,
                        *sorted(TERMINAL_PARSE_TASK_STATUSES),
                    ),
                ).fetchall()
                active_task_ids = [int(row["id"]) for row in active_task_rows]
                if active_task_ids:
                    task_placeholders = ",".join("?" for _ in active_task_ids)
                    con.execute(
                        f"""
                        UPDATE parse_tasks
                        SET status = 'cancelled', finished_at = ?,
                            lease_owner = NULL, lease_expires_at = NULL,
                            error_code = 'FORCE_COMPLETED_EXCLUDED',
                            error_message = ?
                        WHERE id IN ({task_placeholders})
                        """,
                        (
                            now,
                            "用户强力完成本次索引，任务已停止并排除源文件",
                            *active_task_ids,
                        ),
                    )
                    con.execute(
                        f"""
                        UPDATE parse_task_attempts
                        SET status = 'interrupted', finished_at = ?,
                            error_code = 'FORCE_COMPLETED_EXCLUDED',
                            error_message = ?
                        WHERE task_id IN ({task_placeholders})
                          AND status = 'running'
                        """,
                        (
                            now,
                            "用户强力完成本次索引",
                            *active_task_ids,
                        ),
                    )
                    cancelled_tasks = len(active_task_ids)
                con.execute(
                    f"""
                    UPDATE files
                    SET parse_status = 'failed',
                        parse_error_code = 'FORCE_COMPLETED_EXCLUDED',
                        parse_error_message = ?, indexed_at = ?
                    WHERE id IN ({placeholders})
                      AND parse_status IN ('pending', 'processing', 'cancelled')
                    """,
                    (
                        "多次恢复后仍未完成，已由用户强力完成并排除",
                        now,
                        *file_ids,
                    ),
                )
                refreshed = {
                    int(row["id"]): row
                    for row in con.execute(
                        f"SELECT * FROM files WHERE id IN ({placeholders})",
                        tuple(file_ids),
                    ).fetchall()
                }
                inserted_ids: list[int] = []
                for file_id in file_ids:
                    throw_if_cancelled()
                    row = refreshed[file_id]
                    error_message = str(row["parse_error_message"] or "")
                    error_digest = (
                        hashlib.sha256(error_message.encode("utf-8")).hexdigest()
                        if error_message
                        else None
                    )
                    con.execute(
                        """
                        INSERT INTO index_scope_exclusions(
                            file_id, root_id, path, source_size_bytes,
                            source_modified_time, source_quick_fingerprint,
                            source_content_hash, parse_status, parse_error_code,
                            parse_error_message_digest, parser_name, parser_version,
                            reason, operation_source, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            file_id,
                            int(row["root_id"]),
                            str(row["path"]),
                            row["size_bytes"],
                            row["modified_time"],
                            row["quick_fingerprint"],
                            row["content_hash_full"] or row["content_hash"],
                            str(row["parse_status"] or ""),
                            row["parse_error_code"],
                            error_digest,
                            row["parser_name"],
                            row["parser_version"],
                            reason_text,
                            source_text,
                            now,
                        ),
                    )
                    inserted_ids.append(file_id)
                    report(
                        "recording_exclusions",
                        "正在处理剩余阻断项",
                        processed_files=len(inserted_ids),
                        total_files=len(file_ids),
                    )
                self._after_scope_exclusion_audit(con, inserted_ids)

            # FTS replacement and readiness state are committed with the audit.
            reconciled = self._reconcile_residual_parse_tasks(con, now)
            cancelled_tasks += int(reconciled["cancelled_tasks"])
            report(
                "rebuilding_content_fts",
                "正在重建全文索引",
                large_fts_operation=True,
            )
            throw_if_cancelled()
            con.execute("DELETE FROM content_fts")
            con.execute(
                """
                INSERT INTO content_fts(
                    rowid, block_id, file_id, filename, path,
                    location_text, normalized_text
                )
                SELECT cb.id, cb.id, cb.file_id, COALESCE(f.filename, ''),
                       COALESCE(f.path, ''), cb.location_text,
                       cb.normalized_text
                FROM content_blocks cb
                LEFT JOIN files f ON f.id = cb.file_id
                WHERE EXISTS (
                    SELECT 1 FROM files visible
                    WHERE visible.is_deleted = 0
                      AND (
                          (cb.document_id IS NOT NULL
                           AND visible.document_id = cb.document_id)
                          OR (cb.document_id IS NULL AND visible.id = cb.file_id)
                      )
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = visible.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                )
                """
            )
            report(
                "updating_filename_fts",
                "正在更新文件名索引",
                large_fts_operation=True,
            )
            throw_if_cancelled()
            con.execute("DELETE FROM files_fts")
            con.execute(
                """
                INSERT INTO files_fts(rowid, file_id, filename, path)
                SELECT f.id, f.id, f.filename, f.path
                FROM files f
                WHERE f.is_deleted = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """
            )
            report("refreshing_index_state", "正在刷新索引状态")
            throw_if_cancelled()
            con.execute(
                """
                UPDATE index_runs
                SET status = 'incomplete', finished_at = COALESCE(finished_at, ?)
                WHERE status = 'running'
                """,
                (now,),
            )
            con.execute(
                f"""
                UPDATE roots
                SET status = 'ready', last_scan_at = ?, updated_at = ?
                WHERE enabled = 1
                  AND NOT EXISTS (
                      SELECT 1 FROM files f
                      WHERE f.root_id = roots.id AND f.is_deleted = 0
                        AND f.extension NOT IN ({video_placeholders})
                        AND f.parse_status NOT IN ('success', 'metadata_only')
                        AND NOT EXISTS (
                            SELECT 1 FROM index_scope_exclusions e
                            WHERE e.file_id = f.id
                              AND e.revoked_at IS NULL
                              AND e.invalidated_at IS NULL
                        )
                  )
                """,
                (now, now, *video_extensions),
            )
            self._set_index_state(con, "content_fts_dirty", "0")
            self._set_index_state(con, "full_batch_incomplete", "0")
            discarded_candidates = self._discard_staging_index_versions(
                con,
                now,
                "Current scope was rebuilt by force complete",
            )
            ready_roots = int(
                con.execute(
                    "SELECT COUNT(*) FROM roots WHERE enabled = 1 AND status = 'ready'"
                ).fetchone()[0]
            )
        return {
            "excluded_files": len(file_ids),
            "cancelled_tasks": cancelled_tasks,
            "invalidated_tasks": int(reconciled["invalidated_tasks"]),
            "discarded_candidates": discarded_candidates,
            "ready_roots": ready_roots,
        }

    def stats(self) -> dict[str, int]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        with self.connect() as con:
            total = con.execute("SELECT COUNT(*) AS n FROM files WHERE is_deleted = 0").fetchone()["n"]
            blocks = con.execute("SELECT COUNT(*) AS n FROM content_blocks").fetchone()["n"]
            failed = con.execute(
                """
                SELECT COUNT(*) AS n FROM files f
                WHERE f.parse_status IN ('failed', 'failed_retryable')
                  AND f.is_deleted = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """
            ).fetchone()["n"]
            unsupported = con.execute(
                "SELECT COUNT(*) AS n FROM files WHERE parse_status = 'unsupported' AND is_deleted = 0"
            ).fetchone()["n"]
            metadata_only = con.execute(
                "SELECT COUNT(*) AS n FROM files WHERE parse_status IN ('metadata_only', 'skipped', 'ocr_disabled', 'converter_missing') AND is_deleted = 0"
            ).fetchone()["n"]
            documents = con.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            video_excluded = con.execute(
                f"SELECT COUNT(*) AS n FROM files WHERE is_deleted = 0 AND extension IN ({video_placeholders})",
                video_extensions,
            ).fetchone()["n"]
            manual_excluded = int(
                con.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM files f
                    JOIN index_scope_exclusions e ON e.file_id = f.id
                    WHERE f.is_deleted = 0
                      AND e.revoked_at IS NULL
                      AND e.invalidated_at IS NULL
                    """
                ).fetchone()["n"]
            )
            eligible = int(total) - int(video_excluded) - manual_excluded
            completion = con.execute(
                f"""
                SELECT
                    SUM(CASE WHEN f.parse_status = 'success' THEN 1 ELSE 0 END)
                        AS successful,
                    SUM(CASE WHEN f.parse_status = 'metadata_only' THEN 1 ELSE 0 END)
                        AS metadata_only_complete
                FROM files f
                WHERE f.is_deleted = 0
                  AND f.extension NOT IN ({video_placeholders})
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                """,
                video_extensions,
            ).fetchone()
            successful = int(completion["successful"] or 0)
            metadata_only_complete = int(
                completion["metadata_only_complete"] or 0
            )
            complete = successful + metadata_only_complete
            return {
                "files": int(total),
                "discovered_files": int(total),
                "blocks": int(blocks),
                "documents": int(documents),
                "failed": int(failed),
                "unsupported": int(unsupported),
                "metadata_only": int(metadata_only),
                "eligible_files": int(eligible),
                "complete_files": int(complete),
                "successful_files": successful,
                "metadata_only_complete_files": metadata_only_complete,
                "blocking_files": max(0, int(eligible) - int(complete)),
                "video_excluded": int(video_excluded),
                "manual_excluded_files": manual_excluded,
            }

    def index_readiness(self) -> dict[str, object]:
        stats = self.stats()
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        terminal_placeholders = ",".join(
            "?" for _ in TERMINAL_PARSE_TASK_STATUSES
        )
        with self.connect() as con:
            enabled_roots = int(
                con.execute("SELECT COUNT(*) AS n FROM roots WHERE enabled = 1").fetchone()["n"]
            )
            active_runs = int(
                con.execute("SELECT COUNT(*) AS n FROM index_runs WHERE status = 'running'").fetchone()["n"]
            )
            unready_roots = int(
                con.execute(
                    """
                    SELECT COUNT(*) AS n FROM roots
                    WHERE enabled = 1 AND COALESCE(status, '') != 'ready'
                    """
                ).fetchone()["n"]
            )
            unfinished_tasks = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) AS n
                    FROM parse_tasks pt
                    JOIN files f ON f.id = pt.file_id
                    JOIN roots r ON r.id = f.root_id
                    WHERE r.enabled = 1
                      AND f.is_deleted = 0
                      AND f.extension NOT IN ({video_placeholders})
                      AND pt.status NOT IN ({terminal_placeholders})
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = f.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                    """,
                    (
                        *video_extensions,
                        *sorted(TERMINAL_PARSE_TASK_STATUSES),
                    ),
                ).fetchone()["n"]
            )
            unpublished_candidates = int(
                con.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM index_versions iv
                    LEFT JOIN roots r ON r.id = iv.root_id
                    WHERE iv.status = 'staging'
                      AND (iv.root_id IS NULL OR r.enabled = 1)
                    """
                ).fetchone()["n"]
            )
            state_rows = {
                str(row["key"]): str(row["value"])
                for row in con.execute(
                    """
                    SELECT key, value FROM index_state
                    WHERE key IN ('content_fts_dirty', 'full_batch_incomplete')
                    """
                )
            }
        content_fts_dirty = state_rows.get("content_fts_dirty") == "1"
        full_batch_incomplete = state_rows.get("full_batch_incomplete") == "1"
        not_ready_reasons: list[str] = []
        if enabled_roots <= 0:
            not_ready_reasons.append("no_enabled_roots")
        if int(stats["blocking_files"]) > 0:
            not_ready_reasons.append("blocking_files")
        if unfinished_tasks > 0:
            not_ready_reasons.append("unfinished_tasks")
        if active_runs > 0:
            not_ready_reasons.append("active_run")
        if content_fts_dirty:
            not_ready_reasons.append("content_fts_dirty")
        if full_batch_incomplete:
            not_ready_reasons.append("full_batch_incomplete")
        if unready_roots > 0:
            not_ready_reasons.append("unready_root")
        if unpublished_candidates > 0:
            not_ready_reasons.append("unpublished_candidate")
        ready = (
            enabled_roots > 0
            and int(stats["blocking_files"]) == 0
            and unfinished_tasks == 0
            and active_runs == 0
            and unready_roots == 0
            and not full_batch_incomplete
            and not content_fts_dirty
            and unpublished_candidates == 0
        )
        return {
            **stats,
            "enabled_roots": enabled_roots,
            "active_runs": active_runs,
            "unready_roots": unready_roots,
            "unfinished_tasks": unfinished_tasks,
            "content_fts_dirty": content_fts_dirty,
            "full_batch_incomplete": full_batch_incomplete,
            "unpublished_candidates": unpublished_candidates,
            "not_ready_reasons": not_ready_reasons,
            "repairable": bool(
                enabled_roots > 0 and active_runs == 0 and not ready
            ),
            "ready": ready,
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
