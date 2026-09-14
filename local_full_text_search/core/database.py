from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from local_full_text_search.config.constants import DB_PATH, PARSER_VERSION, VIDEO_EXTENSIONS
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.normalizer import is_cjk_bigram
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import FileTiming, IndexRunMetrics
from local_full_text_search.services.logging_service import log_event

SCHEMA_VERSION = 9
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
    "failed_retryable",
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
FINAL_FAILURE_STATUSES = {
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
STALE_PARSE_TASK_TIMEOUT_SECONDS = 120
NO_WORKER_TASK_TIMEOUT_SECONDS = 60
PDF_PAGE_STALE_TASK_TIMEOUT_SECONDS = 900
PDF_PAGE_NO_WORKER_TIMEOUT_SECONDS = 600
STALE_INDEX_RUN_TIMEOUT_SECONDS = 600
ROOT_RETRY_DELAYS_SECONDS = (30, 120, 600)
SQLITE_INT64_MIN = -(1 << 63)
SQLITE_INT64_MAX = (1 << 63) - 1

# Finalization must never wait forever on the single SQLite writer slot.  The
# limits are deliberately conservative: a normal bounded write gets several
# short opportunities to recover, while a persistent lock is surfaced as a
# controlled run failure instead of an endless heartbeat-only loop.
SQLITE_BUSY_CIRCUIT_MAX_ATTEMPTS = 8
SQLITE_BUSY_CIRCUIT_MAX_SECONDS = 15.0
SQLITE_WRITER_BUSY_CIRCUIT_MAX_ATTEMPTS = 12
SQLITE_WRITER_BUSY_CIRCUIT_MAX_SECONDS = 30.0
SQLITE_CACHE_LINK_BATCH_SIZE = 400

# SQLite permits one writer at a time per database file.  Control and
# candidate databases are separate files and must not contend on one global
# process lock.  The registry is process-local; an external owner remains
# explicitly unknown in diagnostics.
_DATABASE_WRITE_LOCKS_LOCK = threading.Lock()
_DATABASE_WRITE_LOCKS: dict[str, threading.RLock] = {}
_DATABASE_LOCK_STATE_LOCK = threading.Lock()
_DATABASE_LOCK_STATES: dict[str, dict[str, object]] = {}


def _normalized_database_path(path: Path) -> str:
    try:
        return str(Path(path).resolve()).casefold()
    except OSError:
        return str(Path(path).absolute()).casefold()


def _database_write_lock(path: Path) -> tuple[str, threading.RLock]:
    key = _normalized_database_path(path)
    with _DATABASE_WRITE_LOCKS_LOCK:
        return key, _DATABASE_WRITE_LOCKS.setdefault(key, threading.RLock())
logger = logging.getLogger(__name__)


def _writer_lock_acquired(database_key: str, operation: str) -> None:
    thread_id = threading.get_ident()
    with _DATABASE_LOCK_STATE_LOCK:
        state = _DATABASE_LOCK_STATES.setdefault(database_key, {})
        if state.get("owner_thread_id") == thread_id:
            state["owner_depth"] = int(state.get("owner_depth") or 0) + 1
            return
        state.update(
            {
                "owner_thread_id": thread_id,
                "owner_operation": str(operation or "connect"),
                "operation_id": f"dbtx:{uuid.uuid4().hex}",
                "acquired_monotonic": time.monotonic(),
                "owner_depth": 1,
                "last_commit_at": state.get("last_commit_at", ""),
            }
        )


def _writer_lock_released(database_key: str, *, committed: bool = False) -> None:
    thread_id = threading.get_ident()
    with _DATABASE_LOCK_STATE_LOCK:
        state = _DATABASE_LOCK_STATES.get(database_key)
        if not state or state.get("owner_thread_id") != thread_id:
            return
        state["owner_depth"] = max(0, int(state.get("owner_depth") or 0) - 1)
        if committed:
            state["last_commit_at"] = utc_now()
        if int(state["owner_depth"]) == 0:
            hold_ms = int(
                max(0.0, time.monotonic() - float(state.get("acquired_monotonic") or 0.0))
                * 1000
            )
            if hold_ms >= 1000:
                log_event(
                    "database.sqlite.lock_hold",
                    level=logging.WARNING,
                    component="database",
                    database_path=database_key,
                    operation=str(state.get("owner_operation") or "connect"),
                    operation_id=str(state.get("operation_id") or ""),
                    thread_id=thread_id,
                    hold_ms=hold_ms,
                    last_commit_at=str(state.get("last_commit_at") or ""),
                )
            state.update(
                {
                    "owner_thread_id": None,
                    "owner_operation": "",
                    "operation_id": "",
                    "acquired_monotonic": 0.0,
                    "owner_depth": 0,
                }
            )


def _writer_lock_snapshot(database_key: str) -> dict[str, object]:
    with _DATABASE_LOCK_STATE_LOCK:
        state = dict(_DATABASE_LOCK_STATES.get(database_key) or {})
        owner = state.get("owner_thread_id")
        acquired = float(state.get("acquired_monotonic") or 0.0)
        return {
            "owner_thread_id": owner,
            "owner_operation": str(state.get("owner_operation") or ""),
            "operation_id": str(state.get("operation_id") or ""),
            "owner_hold_ms": (
                int(max(0.0, time.monotonic() - acquired) * 1000)
                if owner is not None and acquired > 0
                else 0
            ),
            "owner_depth": int(state.get("owner_depth") or 0),
            "last_commit_at": str(state.get("last_commit_at") or ""),
            "owner_confidence": "process_local" if owner is not None else "unknown",
        }


class SQLiteBusyCircuitOpen(sqlite3.OperationalError):
    """Raised after a bounded operation remains blocked by SQLite."""


class SQLiteBusyCircuit:
    """Bound repeated SQLite busy errors and emit actionable diagnostics."""

    def __init__(
        self,
        operation: str,
        *,
        max_attempts: int = SQLITE_BUSY_CIRCUIT_MAX_ATTEMPTS,
        max_seconds: float = SQLITE_BUSY_CIRCUIT_MAX_SECONDS,
        database_path: Path | None = None,
        run_id: str | None = None,
    ) -> None:
        self.operation = str(operation or "sqlite_operation")
        self.max_attempts = max(1, int(max_attempts))
        self.max_seconds = max(0.1, float(max_seconds))
        self.database_path = Path(database_path).resolve() if database_path else None
        self.run_id = str(run_id or "")
        self.attempts = 0
        self.total_busy_count = 0
        self.busy_started_monotonic = 0.0
        self.last_busy_monotonic = 0.0
        self.last_error = ""
        self._open = False

    @property
    def is_open(self) -> bool:
        return bool(self._open)

    @property
    def elapsed_seconds(self) -> float:
        if self.busy_started_monotonic <= 0:
            return 0.0
        return max(0.0, time.monotonic() - self.busy_started_monotonic)

    def reset_after_success(self, *, stage: str = "") -> None:
        if self.attempts <= 0:
            return
        log_event(
            "database.sqlite.busy_recovered",
            component="database",
            run_id=self.run_id or None,
            database_path=str(self.database_path) if self.database_path else None,
            operation=self.operation,
            stage=str(stage or ""),
            busy_attempts=int(self.attempts),
            busy_elapsed_ms=int(self.elapsed_seconds * 1000),
            total_busy_count=int(self.total_busy_count),
        )
        self.attempts = 0
        self.busy_started_monotonic = 0.0
        self.last_busy_monotonic = 0.0
        self.last_error = ""

    def note_busy(
        self,
        exc: BaseException,
        *,
        stage: str = "",
        batch_size: int | None = None,
        completed: int | None = None,
        total: int | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> None:
        now = time.monotonic()
        if self.busy_started_monotonic <= 0:
            self.busy_started_monotonic = now
        self.last_busy_monotonic = now
        self.attempts += 1
        self.total_busy_count += 1
        self.last_error = str(exc)[:1000]
        elapsed_ms = int(self.elapsed_seconds * 1000)
        fields: dict[str, object] = {
            "operation": self.operation,
            "stage": str(stage or ""),
            "attempt": int(self.attempts),
            "max_attempts": int(self.max_attempts),
            "busy_elapsed_ms": elapsed_ms,
            "max_busy_ms": int(self.max_seconds * 1000),
            "error_type": type(exc).__name__,
            "error_message": self.last_error,
        }
        if self.database_path is not None:
            fields["database_path"] = str(self.database_path)
        if self.run_id:
            fields["run_id"] = self.run_id
        if batch_size is not None:
            fields["batch_size"] = max(0, int(batch_size))
        if completed is not None:
            fields["completed"] = max(0, int(completed))
        if total is not None:
            fields["total"] = max(0, int(total))
        if wait_timeout_seconds is not None:
            fields["wait_timeout_ms"] = max(
                0,
                int(float(wait_timeout_seconds) * 1000),
            )
        log_event(
            "database.sqlite.busy",
            level=logging.WARNING,
            component="database",
            **fields,
        )
        if self.attempts >= self.max_attempts or self.elapsed_seconds >= self.max_seconds:
            self._open = True
            log_event(
                "database.sqlite.busy_circuit_open",
                level=logging.ERROR,
                component="database",
                run_id=self.run_id or None,
                database_path=str(self.database_path) if self.database_path else None,
                operation=self.operation,
                stage=str(stage or ""),
                busy_attempts=int(self.attempts),
                busy_elapsed_ms=elapsed_ms,
                max_attempts=int(self.max_attempts),
                max_busy_ms=int(self.max_seconds * 1000),
                error_type=type(exc).__name__,
                error_message=self.last_error,
            )
            raise SQLiteBusyCircuitOpen(
                "database is locked: "
                f"SQLite 忙熔断已打开 operation={self.operation} "
                f"attempts={self.attempts} elapsed_ms={elapsed_ms}"
            ) from exc


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sqlite_int64(value: object, *, default: int = 0) -> int:
    """Normalize filesystem integers before binding them to SQLite INTEGER."""

    try:
        normalized = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if normalized < SQLITE_INT64_MIN or normalized > SQLITE_INT64_MAX:
        return default
    return normalized


def _usn_state_root_id(key: object) -> int | None:
    parts = str(key or "").split(":", 2)
    if len(parts) < 2 or parts[0] != "usn_checkpoint":
        return None
    try:
        return int(parts[1])
    except (TypeError, ValueError):
        return None


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

    def __init__(self, db_path: Path = DB_PATH, *, read_only: bool = False) -> None:
        self.db_path = db_path
        self.read_only = bool(read_only)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connections_lock = threading.Lock()
        self._active_connections: set[sqlite3.Connection] = set()
        self._busy_log_lock = threading.Lock()
        self._busy_log_last: dict[str, float] = {}
        self._busy_log_suppressed: dict[str, int] = {}
        self._quick_readiness_cache: dict[str, object] | None = None
        self._short_tokens_deferred = False
        self._startup_migration_deferred = False

    @contextmanager
    def connect(
        self,
        *,
        timeout_seconds: float = 30.0,
        operation: str = "",
    ) -> Iterator[sqlite3.Connection]:
        """Open a serialized SQLite writer connection.

        Application-level search/index pausing is owned by
        ``SearchIndexCoordinator`` and ``IndexWriter``.  This gateway only
        handles SQLite locking and transaction lifecycle, so a planned search
        pause can no longer be misreported as ``database is locked``.
        """
        if self.read_only:
            raise sqlite3.OperationalError(
                "活动搜索快照是只读数据库，禁止写入"
            )
        timeout = max(0.0, float(timeout_seconds))
        busy_timeout_ms = max(0, int(timeout * 1000))
        lock_wait_started = time.monotonic()
        database_key, writer_lock = _database_write_lock(self.db_path)
        acquired = writer_lock.acquire(timeout=max(0.0, timeout))
        if not acquired:
            wait_ms = int((time.monotonic() - lock_wait_started) * 1000)
            owner_snapshot = _writer_lock_snapshot(database_key)
            suppressed = self._busy_log_slot(
                f"{operation or 'connect'}:process_writer_lock"
            )
            if suppressed is not None:
                log_event(
                    "database.sqlite.busy",
                    level=logging.WARNING,
                    component="database",
                    database_path=str(self.db_path),
                    operation=str(operation or "connect"),
                    stage="process_writer_lock",
                    attempt=1,
                    max_attempts=1,
                    busy_elapsed_ms=wait_ms,
                    wait_timeout_ms=busy_timeout_ms,
                    active_connections=self.active_connection_count(),
                    suppressed_since_last=suppressed,
                    **owner_snapshot,
                    error_type="OperationalError",
                    error_message="database is locked (process writer lock timeout)",
                )
            raise sqlite3.OperationalError("database is locked")
        _writer_lock_acquired(database_key, operation)
        committed = False
        try:
            con = sqlite3.connect(self.db_path, timeout=timeout)
        except Exception:
            _writer_lock_released(database_key)
            writer_lock.release()
            raise
        with self._connections_lock:
            self._active_connections.add(con)
        transaction_started = time.monotonic()
        try:
            con.row_factory = sqlite3.Row
            con.execute("PRAGMA foreign_keys = ON")
            con.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
            con.execute("PRAGMA synchronous = NORMAL")
            yield con
            # Durable monotonic commit sequence.  Older/temporary databases
            # may not have index_state yet; those transactions still commit
            # normally and retain the process-local lock diagnostics.
            try:
                con.execute(
                    """
                    INSERT INTO index_state(key, value)
                    VALUES ('data_commit_sequence', '1')
                    ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1
                    """
                )
            except sqlite3.OperationalError:
                pass
            con.commit()
            committed = True
            transaction_ms = int((time.monotonic() - transaction_started) * 1000)
            if transaction_ms >= 1000:
                log_event(
                    "database.sqlite.transaction_slow",
                    level=logging.WARNING,
                    component="database",
                    database_path=str(self.db_path),
                    operation=str(operation or "transaction"),
                    transaction_ms=transaction_ms,
                    active_connections=self.active_connection_count(),
                )
        except Exception as exc:
            if isinstance(exc, sqlite3.OperationalError) and _is_database_busy(exc):
                suppressed = self._busy_log_slot(
                    f"{operation or 'transaction'}:sqlite_transaction"
                )
                if suppressed is not None:
                    log_event(
                        "database.sqlite.busy",
                        level=logging.WARNING,
                        component="database",
                        database_path=str(self.db_path),
                        operation=str(operation or "transaction"),
                        stage="sqlite_transaction",
                        busy_elapsed_ms=int(
                            (time.monotonic() - transaction_started) * 1000
                        ),
                        wait_timeout_ms=busy_timeout_ms,
                        active_connections=self.active_connection_count(),
                        suppressed_since_last=suppressed,
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:1000],
                    )
            try:
                con.rollback()
            except sqlite3.Error:
                logger.debug(
                    "SQLite rollback failed after operation error: path=%s operation=%s",
                    self.db_path,
                    operation or "transaction",
                    exc_info=True,
                )
            raise
        finally:
            with self._connections_lock:
                self._active_connections.discard(con)
            con.close()
            _writer_lock_released(database_key, committed=committed)
            writer_lock.release()

    def active_connection_count(self) -> int:
        """Return the number of currently tracked connections for diagnostics."""

        with self._connections_lock:
            return len(self._active_connections)

    def _busy_log_slot(self, key: str, *, interval_seconds: float = 1.0) -> int | None:
        now = time.monotonic()
        normalized = str(key or "sqlite")
        with self._busy_log_lock:
            previous = self._busy_log_last.get(normalized, 0.0)
            if now - previous < max(0.1, float(interval_seconds)):
                self._busy_log_suppressed[normalized] = (
                    int(self._busy_log_suppressed.get(normalized, 0)) + 1
                )
                return None
            self._busy_log_last[normalized] = now
            return int(self._busy_log_suppressed.pop(normalized, 0))

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

    def _run_write_with_retry(
        self,
        operation: Callable[[], Any],
        *,
        timeout_seconds: float = 5.0,
        operation_name: str = "write_with_retry",
    ) -> Any:
        """Run a short write operation with bounded SQLite busy backoff."""

        deadline = time.monotonic() + max(0.0, float(timeout_seconds))
        delay = 0.05
        attempts = 0
        while True:
            try:
                return operation()
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc) or time.monotonic() >= deadline:
                    raise
                attempts += 1
                log_event(
                    "database.sqlite.busy.retry",
                    level=logging.WARNING,
                    component="database",
                    database_path=str(self.db_path),
                    operation=str(operation_name or "write_with_retry"),
                    attempt=attempts,
                    elapsed_ms=int(
                        max(0.0, float(timeout_seconds) - max(0.0, deadline - time.monotonic()))
                        * 1000
                    ),
                    retry_deadline_ms=int(max(0.0, deadline - time.monotonic()) * 1000),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(0.75, delay * 1.7)

    def initialize(
        self,
        *,
        startup_fast: bool = False,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        def report_migration(
            stage: str,
            phase_label: str,
            **payload: object,
        ) -> None:
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": stage,
                        "phase_label": phase_label,
                        **payload,
                    }
                )

        self._startup_migration_deferred = False
        if startup_fast:
            if self._looks_like_existing_sqlite_file():
                # Never open an existing database on the GUI thread. Even a
                # read-only sqlite_master probe can replay a multi-gigabyte WAL
                # before returning. The startup scan worker performs the full
                # schema/projection check after QApplication has shown the UI.
                self._startup_migration_deferred = True
                logger.info(
                    "Deferring database startup check until background startup scan: %s",
                    self.db_path,
                )
                return
        previous_version = self._database_user_version()
        backup_path: Path | None = None
        # Versioned migrations that can rewrite existing user data keep a full
        # rollback copy.  Same-version layout repairs are additive/index-only
        # work and run in SQLite transactions; copying a multi-gigabyte corpus
        # before them is needless peak disk usage and caused repeated backups
        # on every interrupted short-token upgrade.
        if previous_version is not None and previous_version < SCHEMA_VERSION:
            report_migration(
                "database_migration_backup",
                "正在安全备份索引数据库",
            )
            backup_path = self._backup_legacy_database(
                previous_version,
                progress_callback=progress_callback,
            )
        try:
            with self.connect(timeout_seconds=0.5) as con:
                con.execute("PRAGMA journal_mode = WAL")
                report_migration(
                    "database_migration_schema",
                    "备份完成，正在升级数据库结构",
                )
                self._create_schema(con)
                self._migrate_schema_v3(con, previous_version)
                self._migrate_schema_v4(con, previous_version)
                self._migrate_schema_v5(con)
                self._migrate_schema_v6(con)
                self._migrate_schema_v7(con)
                self._migrate_schema_v8(con)
                self._migrate_schema_v9(con)
                report_migration(
                    "database_migration_short_token_index",
                    "正在创建短词倒排索引（首次升级可能需要较长时间）",
                )
                self._migrate_schema_v10(con)
                report_migration(
                    "database_migration_fts",
                    "正在核对全文索引结构",
                )
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
            report_migration(
                "database_migration_checkpoint",
                "正在提交数据库升级结果",
            )
            with self.connect(timeout_seconds=0.5) as con:
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        else:
            # Same-version additive cleanup can remove a large legacy derived
            # table.  Truncate that migration WAL once the transaction is
            # committed so the discarded table does not remain as a second
            # multi-gigabyte disk artifact.
            try:
                report_migration(
                    "database_migration_checkpoint",
                    "正在回收旧索引占用的临时空间",
                )
                with self.connect(timeout_seconds=0.5) as con:
                    con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc):
                    raise
                logger.warning(
                    "Same-version migration WAL checkpoint deferred because SQLite is busy"
                )
        # A forced shutdown can interrupt a chunked large-document write
        # before its final file switch. Such documents are intentionally
        # unlinked; reclaim them in bounded transactions on the next normal
        # initialization instead of allowing them to accumulate forever.
        self.cleanup_orphan_staging_documents()
        if self._fts_is_dirty() and not self.has_incomplete_full_batch():
            try:
                pending_fts = self.pending_content_fts_file_ids()
                if pending_fts:
                    self.sync_content_fts_for_files(pending_fts)
                else:
                    self.rebuild_content_fts()
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc):
                    raise
                logger.warning(
                    "Startup FTS repair deferred because SQLite is busy; pending markers retained"
                )
        self._initialize_derived_projection_state(
            progress_callback=progress_callback,
        )
        report_migration(
            "database_migration_complete",
            "数据库升级完成，正在准备扫描",
        )

    def _initialize_derived_projection_state(
        self,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        """Backfill v10 projection metadata once for an existing database."""

        try:
            with self.connect_read_only(timeout_seconds=1.0) as con:
                rows = {
                    str(row["key"]): str(row["value"])
                    for row in con.execute(
                        """
                        SELECT key, value FROM index_state
                        WHERE key IN (
                            'content_fts_dirty', 'content_fts_active_generation',
                            'short_tokens_dirty', 'short_tokens_active_generation',
                            'readiness_counters_generation'
                        )
                        """
                    )
                }
                blocks = int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0])
        except sqlite3.Error:
            return
        if blocks and rows.get("content_fts_dirty") != "1" and not rows.get(
            "content_fts_active_generation"
        ):
            with self.connect(timeout_seconds=1.0) as con:
                self._set_index_state(con, "search_projection_generation", "1")
                self._set_index_state(con, "content_fts_active_generation", "1")
                self._set_index_state(con, "search_projection_updated_at", utc_now())
        if blocks and not rows.get("readiness_counters_generation"):
            try:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "database_migration_counters",
                            "phase_label": "正在重建索引统计",
                        }
                    )
                self.rebuild_index_counters()
            except sqlite3.Error as exc:
                if not _is_database_busy(exc):
                    logger.warning("Derived projection backfill failed", exc_info=True)

    def reconcile_stale_index_runs(
        self,
        *,
        stale_after_seconds: int = STALE_INDEX_RUN_TIMEOUT_SECONDS,
    ) -> int:
        """Close abandoned index runs that have no live parse activity."""

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - max(1, int(stale_after_seconds))
        recovered = 0
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT ir.id, ir.started_at, ir.heartbeat_at, ir.owner_pid
                FROM index_runs ir
                WHERE ir.status = 'running'
                """
            ).fetchall()
            for row in rows:
                try:
                    heartbeat = row["heartbeat_at"] or row["started_at"]
                    started = datetime.fromisoformat(str(heartbeat)).timestamp()
                except (TypeError, ValueError, OSError):
                    started = 0.0
                if started > cutoff:
                    continue
                if row["owner_pid"] is not None and self._pid_is_alive(row["owner_pid"]):
                    continue
                run_id = str(row["id"])
                active = con.execute(
                    """
                    SELECT pt.*
                    FROM parse_tasks pt
                    WHERE pt.run_id = ?
                      AND pt.status IN ('queued', 'running', 'spooled', 'blocked', 'paused')
                    """,
                    (run_id,),
                ).fetchall()
                if any(
                    self._pid_is_alive(task["worker_pid"])
                    or self._task_is_recent(task, now)
                    for task in active
                ):
                    continue
                self._converge_index_run_tasks(con, run_id, "interrupted")
                con.execute(
                    """
                    UPDATE ocr_requests
                    SET status = 'queued', lease_owner = NULL,
                        lease_expires_at = NULL
                    WHERE parent_task_id IN (
                        SELECT id FROM parse_tasks WHERE run_id = ?
                    )
                      AND status IN ('running', 'spooled', 'paused')
                    """,
                    (run_id,),
                )
                con.execute(
                    """
                    UPDATE index_runs
                    SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                        summary_json = COALESCE(summary_json, ?),
                        ended_reason = COALESCE(ended_reason, 'STALE_INDEX_RUN_RECOVERED')
                    WHERE id = ? AND status = 'running'
                    """,
                    (
                        utc_now(),
                        json.dumps(
                            {
                                "recovery_reason": "STALE_INDEX_RUN_RECOVERED",
                            },
                            ensure_ascii=False,
                        ),
                        run_id,
                    ),
                )
                con.execute(
                    """
                    UPDATE files
                    SET parse_status = 'failed_retryable',
                        parse_error_code = COALESCE(parse_error_code, 'STALE_INDEX_RUN_RECOVERED'),
                        parse_error_message = COALESCE(parse_error_message, '索引批次已失联，文件可重新尝试')
                    WHERE id IN (
                        SELECT file_id FROM parse_tasks WHERE run_id = ?
                    )
                      AND parse_status IN ('pending', 'processing', 'cancelled')
                      AND NOT EXISTS (
                          SELECT 1 FROM parse_tasks durable
                          WHERE durable.run_id = ?
                            AND durable.file_id = files.id
                            AND durable.status = 'spooled'
                            AND COALESCE(durable.spool_path, '') <> ''
                            AND COALESCE(durable.spool_checksum, '') <> ''
                      )
                    """,
                    (run_id, run_id),
                )
                recovered += 1
        return recovered

    def _startup_fast_path_ready(self) -> bool:
        """Return true when an existing database can be opened without repair."""

        try:
            return (
                self.db_path.is_file()
                and self._database_user_version() == SCHEMA_VERSION
                and self._schema_layout_complete()
            )
        except (OSError, sqlite3.Error):
            return False

    def _looks_like_existing_sqlite_file(self) -> bool:
        """Check only the fixed SQLite header; never trigger WAL recovery."""

        try:
            if not self.db_path.is_file() or self.db_path.stat().st_size <= 0:
                return False
            with self.db_path.open("rb") as handle:
                return handle.read(16) == b"SQLite format 3\x00"
        except OSError:
            return False

    @property
    def startup_migration_deferred(self) -> bool:
        """Whether GUI startup intentionally deferred an additive migration."""

        return self._startup_migration_deferred

    def _startup_legacy_layout_usable(self) -> bool:
        """Return true for a readable v9 database awaiting additive v10 work."""

        if self._database_user_version() != SCHEMA_VERSION:
            return False
        try:
            con = sqlite3.connect(
                f"file:{self.db_path.as_posix()}?mode=ro",
                uri=True,
                timeout=1.0,
            )
            try:
                tables = {
                    str(row[0])
                    for row in con.execute(
                        "SELECT name FROM sqlite_master WHERE type = 'table'"
                    )
                }
                return {
                    "roots",
                    "files",
                    "content_blocks",
                    "content_fts",
                    "files_fts",
                    "index_state",
                    "index_runs",
                    "index_versions",
                    "parse_tasks",
                    "search_history",
                }.issubset(tables)
            finally:
                con.close()
        except sqlite3.Error:
            return False

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

    @contextmanager
    def connect_read_only(
        self,
        *,
        timeout_seconds: float = 2.0,
        operation: str = "",
    ) -> Iterator[sqlite3.Connection]:
        """Open a non-writing WAL reader for foreground search work."""

        timeout = max(0.0, float(timeout_seconds))
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        con = sqlite3.connect(uri, uri=True, timeout=timeout)
        with self._connections_lock:
            self._active_connections.add(con)
        started = time.monotonic()
        try:
            con.row_factory = sqlite3.Row
            try:
                from local_full_text_search.core.text_blob_codec import decode_text_blob

                con.create_function("decode_text", 2, decode_text_blob)
            except ImportError:
                # Operational databases do not use the compact snapshot view.
                pass
            con.execute("PRAGMA query_only = ON")
            con.execute(f"PRAGMA busy_timeout = {max(0, int(timeout * 1000))}")
            # Do not set PRAGMA synchronous on a read-only connection. SQLite
            # may need a checkpoint/schema lock for that pragma even though
            # the caller only intends to read, which turns ordinary writer
            # contention into ``database is locked`` at connection setup.
            # These cache pragmas are connection-local and safe for readers.
            con.execute("PRAGMA cache_size = -65536")
            con.execute("PRAGMA mmap_size = 268435456")
            yield con
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                log_event(
                    "database.sqlite.read_busy",
                    level=logging.WARNING,
                    component="database",
                    database_path=str(self.db_path),
                    operation=str(operation or "read_only"),
                    elapsed_ms=int((time.monotonic() - started) * 1000),
                    wait_timeout_ms=max(0, int(timeout * 1000)),
                    error_type=type(exc).__name__,
                    error_message=str(exc)[:1000],
                )
            raise
        finally:
            with self._connections_lock:
                self._active_connections.discard(con)
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
                "index_counters",
            }.issubset(tables):
                return False
            if {"short_tokens", "short_tokens_candidate"}.intersection(tables):
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
            if "file_reference" not in file_columns:
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

    def _backup_legacy_database(
        self,
        previous_version: int,
        *,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> Path:
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
            source_size = max(1, int(self.db_path.stat().st_size))
            page_size = int(source.execute("PRAGMA page_size").fetchone()[0])
            last_reported = 0.0

            def report(_status: int, remaining: int, total: int) -> None:
                nonlocal last_reported
                now = time.monotonic()
                if progress_callback is None or (
                    remaining > 0 and now - last_reported < 0.25
                ):
                    return
                last_reported = now
                total_pages = max(1, int(total))
                completed_pages = max(0, total_pages - max(0, int(remaining)))
                copied_bytes = min(source_size, completed_pages * page_size)
                progress_callback(
                    {
                        "stage": "database_migration_backup",
                        "phase_label": "正在安全备份索引数据库",
                        "completed_bytes": copied_bytes,
                        "total_bytes": source_size,
                        "progress_percent": int(100 * completed_pages / total_pages),
                    }
                )

            source.backup(target, pages=1024, progress=report, sleep=0.02)
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
                file_reference INTEGER,
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
        # The greenfield schema deliberately has no Chinese two-character
        # inverted table.  Legacy databases are not migrated by the new
        # release; users start from a clean index directory.
        return

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
    def _migrate_schema_v9(con: sqlite3.Connection) -> None:
        """Persist NTFS file references used by the restart fast path."""

        DatabaseManager._ensure_column(con, "files", "file_reference", "INTEGER")
        con.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_files_root_reference
                ON files(root_id, file_reference, is_deleted);
            CREATE TABLE IF NOT EXISTS usn_directories (
                root_id INTEGER NOT NULL,
                file_reference INTEGER NOT NULL,
                parent_reference INTEGER,
                path TEXT NOT NULL,
                PRIMARY KEY(root_id, file_reference),
                FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_usn_directories_root_parent
                ON usn_directories(root_id, parent_reference);
            """
        )

    @staticmethod
    def _migrate_schema_v10(con: sqlite3.Connection) -> None:
        """Add coordination metadata without changing the user_version value.

        The application historically exposed schema version 9 to migration
        tooling. Keep that compatibility contract while making the v10 layout
        additive and idempotent.
        """

        for name, declaration in (
            ("owner_pid", "INTEGER"),
            ("owner_instance_id", "TEXT"),
            ("heartbeat_at", "TEXT"),
            ("cancel_requested_at", "TEXT"),
            ("ended_reason", "TEXT"),
        ):
            DatabaseManager._ensure_column(con, "index_runs", name, declaration)
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS index_counters (
                root_id INTEGER PRIMARY KEY,
                file_count INTEGER NOT NULL DEFAULT 0,
                eligible_files INTEGER NOT NULL DEFAULT 0,
                complete_files INTEGER NOT NULL DEFAULT 0,
                blocking_files INTEGER NOT NULL DEFAULT 0,
                metadata_only_files INTEGER NOT NULL DEFAULT 0,
                video_excluded INTEGER NOT NULL DEFAULT 0,
                manual_excluded INTEGER NOT NULL DEFAULT 0,
                block_count INTEGER NOT NULL DEFAULT 0,
                document_count INTEGER NOT NULL DEFAULT 0,
                generation INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(root_id) REFERENCES roots(id) ON DELETE CASCADE
            );
            -- short_tokens(token, block_id) already has an implicit UNIQUE
            -- index from its primary key. Do not add a duplicate composite
            -- index here: the v2 sidecar owns short-term lookup instead.
            DROP INDEX IF EXISTS idx_short_tokens_token_block;
            DROP TABLE IF EXISTS short_tokens_candidate;
            DROP TABLE IF EXISTS short_tokens;
            CREATE INDEX IF NOT EXISTS idx_index_runs_status_heartbeat
                ON index_runs(status, heartbeat_at);
            """
        )
        for key, value in (
            ("content_fts_active_generation", "0"),
            ("short_tokens_generation", "0"),
            ("short_tokens_active_generation", "0"),
            ("short_tokens_coverage_block_count", "0"),
            ("search_projection_block_count", "0"),
            ("short_tokens_dirty", "1"),
            ("readiness_counters_generation", "0"),
        ):
            con.execute(
                "INSERT OR IGNORE INTO index_state(key, value) VALUES (?, ?)",
                (key, value),
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
        with self.connect_read_only() as con:
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
                        _sqlite_int64(stat.st_size),
                        float(stat.st_mtime) if math.isfinite(float(stat.st_mtime)) else 0.0,
                        float(stat.st_ctime) if math.isfinite(float(stat.st_ctime)) else 0.0,
                        f"{_sqlite_int64(stat.st_size)}:{_sqlite_int64(stat.st_mtime_ns)}",
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
        return self._run_write_with_retry(
            lambda: self._upsert_precomputed_file_metadata_many(
                root_id,
                metadata_rows,
                retry_failed_files=retry_failed_files,
                compute_full_hash=compute_full_hash,
                mark_processing=mark_processing,
                parser_versions=parser_versions,
            ),
            timeout_seconds=5.0,
            operation_name="upsert_precomputed_file_metadata",
        )

    def _upsert_precomputed_file_metadata_many(
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
                    size = _sqlite_int64(getattr(row, "size_bytes"))
                    modified_ns = _sqlite_int64(getattr(row, "modified_time_ns"))
                    modified_time = float(getattr(row, "modified_time"))
                    created_time = float(getattr(row, "created_time"))
                    if not math.isfinite(modified_time):
                        modified_time = 0.0
                    if not math.isfinite(created_time):
                        created_time = 0.0
                    kwargs = dict(
                        size=size,
                        modified_time=modified_time,
                        created_time=created_time,
                        fingerprint=f"{size}:{modified_ns}",
                        content_hash=str(content_hash) if content_hash else None,
                        now=now,
                        retry_failed_files=retry_failed_files,
                        compute_full_hash=compute_full_hash,
                        mark_processing=mark_processing,
                        expected_parser_version=versions.get(str(file_path)),
                        file_reference=_sqlite_int64(
                            getattr(row, "file_reference", 0) or 0
                        ),
                    )
                    try:
                        file_id, changed = self._upsert_file_metadata_in_connection(
                            con, root_id, file_path, **kwargs
                        )
                    except (OverflowError, ValueError, TypeError) as exc:
                        # A malformed Windows metadata value must not discard
                        # the file. Keep the path and parser work schedulable;
                        # the degraded identity is deliberately conservative.
                        logger.warning(
                            "Metadata degraded for %s: %s",
                            file_path,
                            exc,
                        )
                        kwargs["file_reference"] = 0
                        kwargs["size"] = max(0, min(int(size), SQLITE_INT64_MAX))
                        kwargs["fingerprint"] = f"{kwargs['size']}:0"
                        file_id, changed = self._upsert_file_metadata_in_connection(
                            con, root_id, file_path, **kwargs
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
        file_reference: int = 0,
    ) -> tuple[int, bool]:
        # Windows filesystem metadata may expose unsigned values wider than
        # SQLite's signed INTEGER range (notably file IDs on some volumes).
        # Normalize at the database boundary so one unusual file cannot abort
        # the whole planning batch.
        size = _sqlite_int64(size)
        file_reference = _sqlite_int64(file_reference)
        path_text = str(file_path)
        changed_status = "processing" if mark_processing else "pending"
        existing = con.execute(
            """
            SELECT f.id, f.extension, f.quick_fingerprint, f.content_hash,
                   f.parse_status, f.parser_version, f.parse_error_code,
                   EXISTS (
                       SELECT 1 FROM index_scope_exclusions e
                       WHERE e.file_id = f.id
                         AND e.revoked_at IS NULL
                         AND e.invalidated_at IS NULL
                   ) AS manually_excluded
            FROM files f
            WHERE f.path = ?
            """,
            (path_text,),
        ).fetchone()
        if existing is None:
            cur = con.execute(
                """
                INSERT INTO files(
                    root_id, path, filename, extension, size_bytes, modified_time, created_time,
                    quick_fingerprint, content_hash, parse_status, parser_version, last_seen_at, is_deleted,
                    file_reference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
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
                    file_reference,
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
        elif bool(existing["manually_excluded"]):
            con.execute(
                """
                UPDATE files SET
                    root_id = ?, filename = ?, extension = ?, size_bytes = ?,
                    modified_time = ?, created_time = ?, quick_fingerprint = ?,
                    content_hash = COALESCE(?, content_hash), last_seen_at = ?,
                    is_deleted = 0,
                    file_reference = COALESCE(NULLIF(?, 0), file_reference)
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
                    file_reference,
                    int(existing["id"]),
                ),
            )
            file_id = int(existing["id"])
            self._upsert_file_fts(con, file_id, file_path.name, path_text)
            return file_id, False
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
            or (
                str(existing["parse_status"] or "") == "failed_retryable"
                and str(existing["parse_error_code"] or "") in {
                    "STALE_PARSE_TASK_RECOVERED",
                    "STALE_INDEX_RUN_RECOVERED",
                    "INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT",
                }
            )
            or retry_incomplete
        )
        next_status = changed_status if changed else str(existing["parse_status"])
        con.execute(
            """
            UPDATE files SET
                root_id = ?, filename = ?, extension = ?, size_bytes = ?, modified_time = ?,
                created_time = ?, quick_fingerprint = ?, content_hash = COALESCE(?, content_hash),
                last_seen_at = ?, is_deleted = 0, parse_status = ?,
                parser_version = COALESCE(?, parser_version),
                file_reference = COALESCE(NULLIF(?, 0), file_reference)
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
                file_reference,
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
        archive_mtime = float(archive_stat.st_mtime)
        archive_ctime = float(archive_stat.st_ctime)
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
                member_size = _sqlite_int64(getattr(member, "size_bytes", 0))
                member_crc32 = _sqlite_int64(getattr(member, "crc32", 0))
                member_index = _sqlite_int64(getattr(member, "member_index", 0))
                fingerprint = f"zip:{member_size}:{member_crc32}"
                existing = con.execute(
                    """
                    SELECT id, quick_fingerprint, parser_version, parse_status,
                        parse_error_code,
                        content_hash, content_hash_full,
                        EXISTS (
                            SELECT 1 FROM index_scope_exclusions e
                            WHERE e.file_id = files.id
                              AND e.revoked_at IS NULL
                              AND e.invalidated_at IS NULL
                        ) AS manually_excluded
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
                            member_size,
                            archive_mtime,
                            archive_ctime,
                            fingerprint,
                            member_sha,
                            member_sha,
                            expected_version,
                            now,
                            container_file_id,
                            internal_path,
                            member_index,
                            member_crc32,
                            member_size,
                        ),
                    )
                    file_id = int(cursor.lastrowid)
                    changed = True
                else:
                    member_identity_changed = bool(
                        str(existing["quick_fingerprint"] or "") != fingerprint
                        or (
                            member_sha is not None
                            and str(
                                existing["content_hash_full"]
                                or existing["content_hash"]
                                or ""
                            )
                            != member_sha
                        )
                    )
                    if member_identity_changed:
                        self._invalidate_active_scope_exclusion(
                            con,
                            int(existing["id"]),
                            reason="source_identity_changed",
                            invalidated_at=now,
                        )
                    retry_incomplete = (
                        retry_failed_files and str(existing["parse_status"]) != "success"
                    ) or (
                        str(existing["parse_status"] or "") == "failed_retryable"
                        and str(existing["parse_error_code"] or "") in {
                            "STALE_PARSE_TASK_RECOVERED",
                            "STALE_INDEX_RUN_RECOVERED",
                            "INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT",
                        }
                    )
                    stored_hash = member_sha or str(existing["content_hash_full"] or existing["content_hash"] or "") or None
                    changed = (
                        member_identity_changed
                        or str(existing["parser_version"] or "") != expected_version
                        or str(existing["parse_status"] or "")
                        in {"pending", "processing", "cancelled", "deleted"}
                        or retry_incomplete
                    )
                    if bool(existing["manually_excluded"]) and not member_identity_changed:
                        changed = False
                    file_id = int(existing["id"])
                    con.execute(
                        """
                        UPDATE files SET root_id = ?, filename = ?, extension = ?, size_bytes = ?,
                            modified_time = ?, created_time = ?, quick_fingerprint = ?,
                            content_hash = ?, content_hash_full = ?,
                            parser_version = CASE WHEN ? THEN ? ELSE parser_version END,
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
                            member_size,
                            archive_mtime,
                            archive_ctime,
                            fingerprint,
                            stored_hash,
                            stored_hash,
                            int(changed),
                            expected_version,
                            now,
                            container_file_id,
                            internal_path,
                            member_index,
                            member_crc32,
                            member_size,
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

    def set_content_hash_full_many(
        self,
        values: Sequence[tuple[int, str]],
    ) -> int:
        """Persist content hashes in one transaction.

        Planning can discover hundreds or thousands of hashes in one batch.
        Opening and committing a connection for every file makes that phase
        disproportionately expensive, especially on network-backed folders.
        The last value for a file id wins so callers can safely accumulate
        results from multiple planning lanes.
        """

        deduplicated: dict[int, str] = {}
        for file_id, content_hash_full in values:
            normalized = str(content_hash_full or "").strip()
            if normalized:
                deduplicated[int(file_id)] = normalized
        if not deduplicated:
            return 0
        with self.connect() as con:
            con.executemany(
                "UPDATE files SET content_hash = ?, content_hash_full = ? WHERE id = ?",
                [
                    (content_hash, content_hash, file_id)
                    for file_id, content_hash in deduplicated.items()
                ],
            )
        return len(deduplicated)

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
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[tuple[str, str, str], tuple[int, str]]:
        result: dict[tuple[str, str, str], tuple[int, str]] = {}
        unique = list(dict.fromkeys(identities))
        total = len(unique)
        started = time.perf_counter()
        log_event(
            "database.cache_discovery.started",
            component="database",
            database_path=str(self.db_path),
            total_identities=total,
            chunk_size=200,
        )
        with self.connect_read_only(
            timeout_seconds=0.75,
            operation="cache_document_discovery",
        ) as con:
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
                completed = min(total, offset + len(chunk))
                if progress_callback is not None:
                    progress_callback(completed, total)
        log_event(
            "database.cache_discovery.completed",
            component="database",
            database_path=str(self.db_path),
            total_identities=total,
            cache_hits=len(result),
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return result

    def link_cached_documents_batch(
        self,
        links: Sequence[
            tuple[Sequence[int], int, str, str, str, str]
        ],
        *,
        batch_file_ids: int = SQLITE_CACHE_LINK_BATCH_SIZE,
        start_offset: int = 0,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Associate many files with cached documents in bounded transactions.

        The old finalization path opened one writer transaction per cached
        file.  On a large incremental run that turned a transient SQLite lock
        into thousands of lock retries.  Normalize all aliases to one mapping
        per file, then update at most ``batch_file_ids`` rows per transaction.
        The durable document garbage-collection step still runs for each
        batch, so the operation remains safe after interruption.
        """

        normalized: dict[int, tuple[int, str, str, str, str]] = {}
        for file_ids, document_id, content_key, parser_name, parser_version, status in links:
            for raw_file_id in file_ids:
                file_id = int(raw_file_id)
                if file_id <= 0:
                    continue
                normalized[file_id] = (
                    int(document_id),
                    str(content_key),
                    str(parser_name),
                    str(parser_version),
                    str(status),
                )
        if not normalized:
            return 0

        size = max(1, min(SQLITE_CACHE_LINK_BATCH_SIZE, int(batch_file_ids)))
        items = sorted(normalized.items())
        total = len(items)
        offset_start = max(0, min(total, int(start_offset)))
        linked = offset_start
        batch_count = 0
        started = time.perf_counter()
        log_event(
            "database.cache_link.started",
            component="database",
            database_path=str(self.db_path),
            total_files=total,
            start_offset=offset_start,
            requested_batches=(
                max(0, total - offset_start) + size - 1
            ) // size,
            batch_size=size,
        )
        for offset in range(offset_start, total, size):
            batch = items[offset : offset + size]
            file_ids = [file_id for file_id, _ in batch]
            batch_started = time.perf_counter()
            now = utc_now()
            new_document_ids = {values[0] for _, values in batch}
            with self.connect(
                timeout_seconds=0.75,
                operation="cache_document_link_batch",
            ) as con:
                old_ids = self._document_ids_for_files(con, file_ids)
                con.executemany(
                    """
                    UPDATE files SET document_id = ?, content_key = ?,
                        parser_name = ?, parser_version = ?, parse_status = ?,
                        parse_error_code = NULL, parse_error_message = NULL,
                        indexed_at = ?, is_deleted = 0
                    WHERE id = ?
                    """,
                    [
                        (
                            values[0],
                            values[1],
                            values[2],
                            values[3],
                            values[4],
                            now,
                            file_id,
                        )
                        for file_id, values in batch
                    ],
                )
                self._garbage_collect_documents(
                    con,
                    old_ids - new_document_ids,
                )
            linked += len(batch)
            batch_count += 1
            elapsed_ms = int((time.perf_counter() - batch_started) * 1000)
            log_event(
                "database.cache_link.batch_completed",
                component="database",
                database_path=str(self.db_path),
                batch_index=batch_count,
                batch_files=len(batch),
                completed_files=linked,
                total_files=total,
                elapsed_ms=elapsed_ms,
            )
            if progress_callback is not None:
                progress_callback(linked, total)
        log_event(
            "database.cache_link.completed",
            component="database",
            database_path=str(self.db_path),
            total_files=total,
            completed_files=linked,
            batch_count=batch_count,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return linked

    def link_cached_document(
        self,
        file_ids: Sequence[int],
        document_id: int,
        content_key: str,
        parser_name: str,
        parser_version: str,
        status: str,
    ) -> None:
        """Backward-compatible wrapper for one cached-document association."""

        self.link_cached_documents_batch(
            [
                (
                    file_ids,
                    int(document_id),
                    str(content_key),
                    str(parser_name),
                    str(parser_version),
                    str(status),
                )
            ]
        )

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
        remove_old_fts: bool = False,
    ) -> None:
        if not items:
            return
        now = utc_now()
        # This path replaces content blocks and updates projection state, so it
        # must use the serialized writer connection.  Read-only connections are
        # reserved for snapshot/query helpers; using one here causes a delayed
        # "attempt to write a readonly database" failure in the writer thread.
        with self.connect(timeout_seconds=5.0) as con:
            for item in items:
                self._replace_document_item(
                    con,
                    item,
                    now,
                    update_fts=update_fts,
                    remove_old_fts=remove_old_fts,
                )
            if update_fts:
                self._advance_search_projection(con, updated_at=now)
            else:
                self._set_index_state(con, "content_fts_dirty", "1")
            self._set_index_state(con, "short_tokens_dirty", "1")
        # Derived projections are rebuilt once at the end of an index run.
        # Keeping this transaction cheap is what lets parsing and metadata
        # writes make progress on large collections.
        if update_fts:
            try:
                from local_full_text_search.core.search_index_coordinator import (
                    INDEX_RUNNING,
                    get_search_index_coordinator,
                )

                coordinator_state = get_search_index_coordinator().snapshot().state
            except Exception:
                coordinator_state = ""
            if coordinator_state != INDEX_RUNNING:
                self.rebuild_short_tokens()
                self.rebuild_index_counters()

    def replace_document_blocks_chunked(
        self,
        item: dict[str, Any],
        *,
        chunk_blocks: int = 256,
        chunk_bytes: int = 512 * 1024,
        update_fts: bool = False,
        remove_old_fts: bool = True,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Write one large document without monopolising SQLite.

        A parser result can contain tens of thousands of blocks (legacy XLS
        conversion is the common case).  The old path inserted every block,
        updated the file and closed the task in one transaction.  This method
        writes to an unlinked staging document in short transactions, then
        switches the file rows only after every block is durable.  A cancelled
        or failed write therefore leaves the previously published document
        untouched and can be retried on the next run.
        """

        if not item:
            return
        blocks = list(item.get("blocks") or [])
        status = str(item.get("status") or "success")
        block_limit = max(1, min(512, int(chunk_blocks)))
        byte_limit = max(64 * 1024, int(chunk_bytes))
        total = len(blocks)
        total_bytes = sum(
            len(str(block.raw_text or "").encode("utf-8")) for block in blocks
        )
        if (
            status not in SUCCESSFUL_DOCUMENT_STATUSES
            or (
                total <= block_limit
                and total_bytes <= byte_limit
            )
        ):
            self.replace_document_blocks_many(
                [item],
                update_fts=update_fts,
                remove_old_fts=remove_old_fts,
            )
            return

        primary_file_id = int(item["file_id"])
        file_ids = [
            int(value)
            for value in item.get("file_ids") or [primary_file_id]
        ]
        parser_name = str(item.get("parser_name") or "unknown")
        parser_version = str(item.get("parser_version") or PARSER_VERSION)
        error_code = item.get("error_code")
        error_message = item.get("error_message")
        diagnostics = list(item.get("diagnostics") or [])
        diagnostics_json = (
            json.dumps(diagnostics, ensure_ascii=False, separators=(",", ":"))
            if diagnostics
            else None
        )
        content_key = str(
            item.get("content_key") or f"file:{primary_file_id}:{parser_name}"
        )
        content_hash_full_value = item.get("content_hash_full")
        content_hash_full = (
            str(content_hash_full_value) if content_hash_full_value else None
        )
        now = utc_now()
        staging_key = f"{content_key}#staging:{uuid.uuid4().hex}"
        staging_document_id: int | None = None
        old_document_ids: set[int] = set()

        def excluded(con: sqlite3.Connection) -> bool:
            placeholders = ",".join("?" for _ in file_ids)
            return (
                con.execute(
                    f"""
                    SELECT 1 FROM index_scope_exclusions
                    WHERE file_id IN ({placeholders})
                      AND revoked_at IS NULL AND invalidated_at IS NULL
                    LIMIT 1
                    """,
                    tuple(file_ids),
                ).fetchone()
                is not None
            )

        try:
            with self.connect(timeout_seconds=0.5) as con:
                if excluded(con):
                    task_id = item.get("task_id")
                    if task_id is not None:
                        con.execute(
                            """
                            UPDATE parse_tasks
                            SET status = 'cancelled',
                                finished_at = COALESCE(finished_at, ?),
                                error_code = COALESCE(error_code, 'EXCLUSION_REQUESTED'),
                                error_message = COALESCE(error_message, '文件已人工排除')
                            WHERE id = ?
                            """,
                            (now, int(task_id)),
                        )
                    return
                old_document_ids = self._document_ids_for_files(con, file_ids)
                placeholders = ",".join("?" for _ in file_ids)
                # Remove legacy file-owned blocks before the staged document
                # is linked.  These rows are normally tiny migration leftovers.
                if update_fts or remove_old_fts:
                    con.execute(
                        f"DELETE FROM content_fts WHERE rowid IN ("
                        f"SELECT id FROM content_blocks WHERE file_id IN ({placeholders}) "
                        "AND document_id IS NULL)",
                        tuple(file_ids),
                    )
                con.execute(
                    f"DELETE FROM content_blocks WHERE file_id IN ({placeholders}) "
                    "AND document_id IS NULL",
                    tuple(file_ids),
                )
                con.execute(
                    """
                    INSERT INTO documents(
                        content_key, parser_name, parser_version, parse_status,
                        parse_error_code, parse_error_message, block_count, text_chars,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, 'processing', ?, ?, 0, 0, ?, ?)
                    """,
                    (
                        staging_key,
                        parser_name,
                        parser_version,
                        error_code,
                        _trim_message(error_message),
                        now,
                        now,
                    ),
                )
                row = con.execute(
                    "SELECT id FROM documents WHERE content_key = ? AND parser_name = ? AND parser_version = ?",
                    (staging_key, parser_name, parser_version),
                ).fetchone()
                if row is None:
                    raise RuntimeError("无法创建分块索引暂存文档")
                staging_document_id = int(row["id"])

            inserted = 0
            offset = 0
            while offset < total:
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledError("索引写入已取消")
                selected: list[ContentBlock] = []
                selected_bytes = 0
                while offset + len(selected) < total:
                    block = blocks[offset + len(selected)]
                    block_bytes = len(str(block.raw_text or "").encode("utf-8"))
                    if selected and (
                        len(selected) >= block_limit
                        or selected_bytes + block_bytes > byte_limit
                    ):
                        break
                    selected.append(block)
                    selected_bytes += block_bytes
                with self.connect(timeout_seconds=0.5) as con:
                    if excluded(con):
                        raise CancelledError("文件已人工排除")
                    for block in selected:
                        self._insert_content_block(
                            con,
                            primary_file_id,
                            int(staging_document_id),
                            block,
                            now,
                        )
                    con.execute(
                        """
                        UPDATE documents
                        SET block_count = block_count + ?,
                            text_chars = text_chars + ?,
                            updated_at = ?
                        WHERE id = ?
                        """,
                        (
                            len(selected),
                            sum(len(str(block.raw_text or "")) for block in selected),
                            now,
                            int(staging_document_id),
                        ),
                    )
                offset += len(selected)
                inserted = offset
                if progress_callback is not None:
                    progress_callback(inserted, total)

            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError("索引写入已取消")

            with self.connect(timeout_seconds=0.5) as con:
                if excluded(con):
                    raise CancelledError("文件已人工排除")
                # Keep the canonical document identity unique while switching
                # only after the staged body is complete.
                target = con.execute(
                    """
                    SELECT id FROM documents
                    WHERE content_key = ? AND parser_name = ? AND parser_version = ?
                    LIMIT 1
                    """,
                    (content_key, parser_name, parser_version),
                ).fetchone()
                final_content_key = staging_key
                final_document_id = int(staging_document_id)
                if target is None:
                    final_content_key = content_key
                    con.execute(
                        "UPDATE documents SET content_key = ?, parse_status = ?, "
                        "parse_error_code = ?, parse_error_message = ?, updated_at = ? "
                        "WHERE id = ?",
                        (
                            content_key,
                            status,
                            error_code,
                            _trim_message(error_message),
                            now,
                            int(staging_document_id),
                        ),
                    )
                else:
                    # Swap the canonical identity and all aliases to the
                    # fully written staging document. No content row is moved
                    # or deleted in this final transaction, so publication
                    # stays atomic and bounded even for a 15,000-block XLS.
                    target_id = int(target["id"])
                    target_is_current = target_id in old_document_ids
                    if target_is_current:
                        con.execute(
                            "UPDATE documents SET content_key = ? WHERE id = ?",
                            (f"{content_key}#old:{uuid.uuid4().hex}", target_id),
                        )
                        con.execute(
                            """
                            UPDATE documents SET content_key = ?, parse_status = ?,
                                parse_error_code = ?, parse_error_message = ?, updated_at = ?
                            WHERE id = ?
                            """,
                            (
                                content_key,
                                status,
                                error_code,
                                _trim_message(error_message),
                                now,
                                int(staging_document_id),
                            ),
                        )
                        con.execute(
                            "UPDATE files SET document_id = ? WHERE document_id = ?",
                            (int(staging_document_id), target_id),
                        )
                        old_document_ids.add(target_id)
                        final_document_id = int(staging_document_id)
                        final_content_key = content_key
                    else:
                        # Another visible file still owns the canonical
                        # document. Reuse it and discard the redundant staged
                        # parse in bounded batches after the file switch.
                        final_document_id = target_id
                        final_content_key = content_key
                        old_document_ids.add(int(staging_document_id))
                if final_content_key == staging_key:
                    con.execute(
                        "UPDATE documents SET parse_status = ?, parse_error_code = ?, "
                        "parse_error_message = ?, updated_at = ? WHERE id = ?",
                        (
                            status,
                            error_code,
                            _trim_message(error_message),
                            now,
                            int(staging_document_id),
                        ),
                    )
                placeholders = ",".join("?" for _ in file_ids)
                con.execute(
                    f"""
                    UPDATE files SET
                        document_id = ?, content_key = ?, parse_status = ?,
                        parse_error_code = ?, parse_error_message = ?,
                        parse_diagnostics_json = ?, parser_name = ?, parser_version = ?,
                        indexed_at = ?, content_hash = COALESCE(?, content_hash),
                        content_hash_full = COALESCE(?, content_hash_full), is_deleted = 0
                    WHERE id IN ({placeholders})
                    """,
                    (
                        int(final_document_id),
                        final_content_key,
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
                            WHERE task_id = ? ORDER BY attempt_no DESC LIMIT 1
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
                        UPDATE parse_tasks SET status = ?, finished_at = COALESCE(finished_at, ?),
                            error_code = CASE WHEN ? = 'failed' THEN COALESCE(error_code, ?) ELSE error_code END
                        WHERE parent_task_id = ?
                          AND status NOT IN ('complete', 'failed', 'superseded', 'invalidated', 'cancelled')
                        """,
                        (
                            task_status,
                            now,
                            task_status,
                            error_code,
                            int(task_id),
                        ),
                    )
                # Chunked publication is always deferred.  Even when a caller
                # requested update_fts=True, inserting FTS rows in this loop
                # would bring back the long writer transaction we are avoiding.
                self._set_index_state(con, "content_fts_dirty", "1")
                self._set_index_state(con, "short_tokens_dirty", "1")
                if status in SUCCESSFUL_DOCUMENT_STATUSES:
                    con.executemany(
                        """
                        INSERT INTO index_state(key, value) VALUES (?, ?)
                        ON CONFLICT(key) DO UPDATE SET value = excluded.value
                        """,
                        [(f"content_fts_pending:{file_id}", now) for file_id in file_ids],
                    )

            # Old content is no longer visible after the file switch.  Delete
            # it in bounded transactions so cleanup cannot recreate the lock
            # starvation this method is designed to avoid.
            for document_id in old_document_ids:
                if document_id == final_document_id:
                    continue
                self._garbage_collect_document_bounded(document_id)
        except BaseException:
            if staging_document_id is not None:
                self._discard_document_bounded(staging_document_id)
            raise

    def _discard_document_bounded(self, document_id: int, *, batch_size: int = 256) -> None:
        """Remove an unlinked staging document in short writer transactions."""

        while True:
            with self.connect(timeout_seconds=0.5) as con:
                if con.execute(
                    "SELECT 1 FROM files WHERE document_id = ? LIMIT 1",
                    (int(document_id),),
                ).fetchone() is not None:
                    return
                rows = con.execute(
                    "SELECT id FROM content_blocks WHERE document_id = ? LIMIT ?",
                    (int(document_id), max(1, int(batch_size))),
                ).fetchall()
                if not rows:
                    con.execute("DELETE FROM documents WHERE id = ?", (int(document_id),))
                    return
                ids = [int(row["id"]) for row in rows]
                placeholders = ",".join("?" for _ in ids)
                con.execute(
                    f"DELETE FROM content_fts WHERE rowid IN ({placeholders})",
                    tuple(ids),
                )
                con.execute(
                    f"DELETE FROM content_blocks WHERE id IN ({placeholders})",
                    tuple(ids),
                )

    def _garbage_collect_document_bounded(self, document_id: int) -> None:
        """Garbage-collect an unreferenced document without a giant DELETE."""

        self._discard_document_bounded(int(document_id))

    def cleanup_orphan_staging_documents(self, *, limit: int = 100) -> int:
        """Recover temporary documents left by an interrupted chunked write."""

        with self.connect_read_only(
            timeout_seconds=5.0,
            operation="cleanup_orphan_staging_documents",
        ) as con:
            rows = con.execute(
                """
                SELECT d.id FROM documents d
                WHERE (
                        instr(d.content_key, '#staging:') > 0
                        OR instr(d.content_key, '#old:') > 0
                      )
                  AND NOT EXISTS (
                      SELECT 1 FROM files f WHERE f.document_id = d.id
                  )
                ORDER BY d.id
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        for row in rows:
            self._discard_document_bounded(int(row["id"]))
        return len(rows)

    def _advance_search_projection(
        self,
        con: sqlite3.Connection,
        *,
        updated_at: str,
    ) -> None:
        con.execute(
            """
            INSERT INTO index_state(key, value)
            VALUES ('search_projection_generation', '1')
            ON CONFLICT(key) DO UPDATE SET
                value = CAST(COALESCE(index_state.value, '0') AS INTEGER) + 1
            """
        )
        self._set_index_state(
            con,
            "search_projection_updated_at",
            str(updated_at),
        )
        row = con.execute(
            "SELECT value FROM index_state WHERE key = 'search_projection_generation'"
        ).fetchone()
        self._set_index_state(
            con,
            "content_fts_active_generation",
            str(int(row["value"]) if row is not None else 1),
        )

    def _rebuild_short_tokens_legacy(
        self,
        *,
        cancel_event: threading.Event | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> int:
        """Build and atomically publish the complete two-character index."""

        from collections import Counter

        from local_full_text_search.core.search_index_coordinator import (
            get_search_index_coordinator,
        )

        coordinator = get_search_index_coordinator()
        if not coordinator.try_begin_index_write():
            return 0
        try:
            with self.connect_read_only(timeout_seconds=1.0) as con:
                state = con.execute(
                    "SELECT value FROM index_state WHERE key = 'short_tokens_generation'"
                ).fetchone()
            try:
                generation = int(state["value"]) + 1 if state is not None else 1
            except (TypeError, ValueError):
                generation = 1
            with self.connect(timeout_seconds=5.0) as con:
                con.execute("DELETE FROM short_tokens_candidate")
            if cancel_event is not None and cancel_event.is_set():
                return 0
            insert_sql = """
                INSERT INTO short_tokens_candidate(
                    token, block_id, position_count, generation
                ) VALUES (?, ?, ?, ?)
            """
            token_count = 0
            row_count = 0
            last_progress_report = 0.0
            token_batch: list[tuple[str, int, int, int]] = []

            def clear_candidate() -> None:
                with self.connect(timeout_seconds=5.0) as cleanup_con:
                    cleanup_con.execute("DELETE FROM short_tokens_candidate")

            def flush_tokens() -> None:
                nonlocal token_count
                if not token_batch:
                    return
                with self.connect(timeout_seconds=5.0) as write_con:
                    write_con.executemany(insert_sql, token_batch)
                token_count += len(token_batch)
                token_batch.clear()

            try:
                with self.connect_read_only(timeout_seconds=5.0) as read_con:
                    cursor = read_con.execute(
                        """
                        SELECT cb.id, cb.normalized_text
                        FROM content_blocks cb
                        WHERE EXISTS (
                            SELECT 1 FROM files visible
                            WHERE visible.is_deleted = 0
                              AND (
                                  (cb.document_id IS NOT NULL
                                   AND visible.document_id = cb.document_id)
                                  OR (cb.document_id IS NULL
                                      AND visible.id = cb.file_id)
                              )
                              AND NOT EXISTS (
                                  SELECT 1 FROM index_scope_exclusions excluded
                                  WHERE excluded.file_id = visible.id
                                    AND excluded.revoked_at IS NULL
                                    AND excluded.invalidated_at IS NULL
                              )
                        )
                        ORDER BY cb.id
                        """
                    )
                    while True:
                        rows = cursor.fetchmany(128)
                        if not rows:
                            break
                        for row in rows:
                            if cancel_event is not None and cancel_event.is_set():
                                clear_candidate()
                                return 0
                            row_count += 1
                            now = time.monotonic()
                            if (
                                progress_callback is not None
                                and now - last_progress_report >= 0.5
                            ):
                                progress_callback(
                                    {
                                        "stage": "database_migration_short_tokens",
                                        "phase_label": (
                                            "正在重建两字词快速索引"
                                            f"（已处理 {row_count:,} 个内容块）"
                                        ),
                                        "processed_blocks": row_count,
                                    }
                                )
                                last_progress_report = now
                            text = str(row["normalized_text"] or "")
                            counts = Counter(
                                text[index : index + 2]
                                for index in range(max(0, len(text) - 1))
                                if not text[index].isspace()
                                and not text[index + 1].isspace()
                            )
                            token_batch.extend(
                                (token, int(row["id"]), int(count), generation)
                                for token, count in counts.items()
                                if len(token) == 2
                            )
                            if len(token_batch) >= 10_000:
                                flush_tokens()
                flush_tokens()
                if cancel_event is not None and cancel_event.is_set():
                    clear_candidate()
                    return 0
            except BaseException:
                try:
                    clear_candidate()
                except sqlite3.Error:
                    logger.warning(
                        "Failed to clean short-token candidate generation",
                        exc_info=True,
                    )
                raise
            with self.connect(timeout_seconds=5.0) as con:
                if progress_callback is not None:
                    progress_callback(
                        {
                            "stage": "database_migration_short_tokens",
                            "phase_label": "正在发布两字词快速索引",
                            "processed_blocks": row_count,
                        }
                    )
                current_blocks = int(
                    con.execute(
                        """
                        SELECT COUNT(*) FROM content_blocks cb
                        WHERE EXISTS (
                            SELECT 1 FROM files visible
                            WHERE visible.is_deleted = 0
                              AND (
                                  (cb.document_id IS NOT NULL
                                   AND visible.document_id = cb.document_id)
                                  OR (cb.document_id IS NULL
                                      AND visible.id = cb.file_id)
                              )
                              AND NOT EXISTS (
                                  SELECT 1 FROM index_scope_exclusions excluded
                                  WHERE excluded.file_id = visible.id
                                    AND excluded.revoked_at IS NULL
                                    AND excluded.invalidated_at IS NULL
                              )
                        )
                        """
                    ).fetchone()[0]
                )
                if current_blocks != row_count:
                    self._set_index_state(con, "short_tokens_dirty", "1")
                    con.execute("DELETE FROM short_tokens_candidate")
                    return 0
                con.execute("DELETE FROM short_tokens")
                con.execute(
                    """
                    INSERT INTO short_tokens(token, block_id, position_count)
                    SELECT token, block_id, position_count
                    FROM short_tokens_candidate
                    WHERE generation = ?
                    """,
                    (generation,),
                )
                self._set_index_state(con, "short_tokens_generation", str(generation))
                self._set_index_state(
                    con, "short_tokens_active_generation", str(generation)
                )
                self._set_index_state(
                    con, "short_tokens_coverage_block_count", str(row_count)
                )
                self._set_index_state(
                    con, "search_projection_block_count", str(row_count)
                )
                self._set_index_state(con, "short_tokens_dirty", "0")
                con.execute("DELETE FROM short_tokens_candidate")
            return token_count
        finally:
            coordinator.end_index_write()

    def rebuild_short_tokens(
        self,
        *,
        cancel_event: threading.Event | None = None,
        cancel_check: Callable[[], bool] | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> int:
        """Retained as a compatibility no-op during the short2 removal.

        The old candidate/active tables duplicated millions of postings in the
        primary database and made its WAL grow with the whole corpus.  The new
        sidecar is derived data: a failed or cancelled build simply leaves the
        prior sidecar untouched.
        """

        return 0

        from collections import Counter

        from local_full_text_search.core.search_index_coordinator import (
            get_search_index_coordinator,
        )

        source_sql = """
            SELECT cb.id, cb.normalized_text
            FROM content_blocks cb
            WHERE EXISTS (
                SELECT 1 FROM files visible
                WHERE visible.is_deleted = 0
                  AND (
                      (cb.document_id IS NOT NULL
                       AND visible.document_id = cb.document_id)
                      OR (cb.document_id IS NULL
                          AND visible.id = cb.file_id)
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions excluded
                      WHERE excluded.file_id = visible.id
                        AND excluded.revoked_at IS NULL
                        AND excluded.invalidated_at IS NULL
                  )
            )
            ORDER BY cb.id
        """
        # COUNT(*) over the correlated visibility predicate is disproportionately
        # expensive on a multi-gigabyte database.  The stream below still
        # applies the authoritative visibility predicate; this value is only a
        # progress denominator and is intentionally allowed to be approximate.
        count_sql = "SELECT COUNT(*) FROM content_blocks"
        active_path = self.short_token_index_path
        candidate_path = short_token_candidate_path(self.db_path)
        coordinator = get_search_index_coordinator()
        candidate: sqlite3.Connection | None = None
        token_count = 0
        source_block_count = 0
        published = False
        reserved_write = False

        def cancelled() -> bool:
            return bool(
                (cancel_event is not None and cancel_event.is_set())
                or (cancel_check is not None and cancel_check())
            )

        try:
            candidate_path.unlink(missing_ok=True)
            with self.connect_read_only(timeout_seconds=5.0) as read_con:
                expected_blocks = int(read_con.execute(count_sql).fetchone()[0])
                state = read_con.execute(
                    "SELECT value FROM index_state WHERE key = 'short_tokens_dirty'"
                ).fetchone()
                has_active_exclusions = read_con.execute(
                    """
                    SELECT 1 FROM index_scope_exclusions
                    WHERE revoked_at IS NULL AND invalidated_at IS NULL
                    LIMIT 1
                    """
                ).fetchone() is not None
            if (
                (state is None or str(state["value"]) != "1")
                and (
                    self.short_token_index_available()
                    if has_active_exclusions
                    else self.short_token_index_available(
                        expected_blocks=expected_blocks
                    )
                )
            ):
                return 0
            with self.connect_read_only(timeout_seconds=5.0) as read_con:
                sample_rows = read_con.execute(
                    f"SELECT normalized_text FROM ({source_sql}) sample_blocks LIMIT 256"
                ).fetchall()
            sample_postings = sum(
                len(
                    {
                        encode_cjk_bigram(text[index : index + 2])
                        for index in range(max(0, len(text) - 1))
                        if is_cjk_bigram(text[index : index + 2])
                    }
                )
                for row in sample_rows
                for text in (str(row["normalized_text"] or ""),)
            )
            estimated_postings = int(
                (sample_postings * expected_blocks) / max(1, len(sample_rows))
            )
            # The sidecar's WITHOUT ROWID primary key plus its block-id
            # cleanup index need roughly 96 bytes/posting in a pessimistic
            # build. Keep a 50% safety margin and a small absolute floor.
            required_bytes = max(
                2 * 1024 * 1024 * 1024,
                int(estimated_postings * 96 * 1.5),
            )
            available_bytes = shutil.disk_usage(candidate_path.parent).free
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "short_token_sidecar_preflight",
                        "phase_label": (
                            "正在评估中文两字索引空间"
                            f"（预计需要 {required_bytes / (1024 ** 3):.1f} GB）"
                        ),
                        "estimated_postings": estimated_postings,
                        "required_bytes": required_bytes,
                        "available_bytes": available_bytes,
                    }
                )
            if available_bytes < required_bytes:
                raise RuntimeError(
                    "中文两字索引磁盘空间不足："
                    f"预计需要 {required_bytes / (1024 ** 3):.1f} GB，"
                    f"当前可用 {available_bytes / (1024 ** 3):.1f} GB"
                )

            candidate = sqlite3.connect(candidate_path)
            candidate.execute("PRAGMA journal_mode = DELETE")
            candidate.execute("PRAGMA synchronous = NORMAL")
            candidate.execute("PRAGMA temp_store = MEMORY")
            candidate.executescript(
                """
                CREATE TABLE short_tokens (
                    token_code INTEGER NOT NULL,
                    block_id INTEGER NOT NULL,
                    position_count INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(token_code, block_id)
                ) WITHOUT ROWID;
                CREATE TABLE metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            pending: list[tuple[int, int, int]] = []
            last_reported = 0.0

            def flush() -> None:
                nonlocal token_count
                if not pending:
                    return
                candidate.executemany(
                    """
                    INSERT INTO short_tokens(token_code, block_id, position_count)
                    VALUES (?, ?, ?)
                    """,
                    pending,
                )
                candidate.commit()
                token_count += len(pending)
                pending.clear()

            with self.connect_read_only(timeout_seconds=5.0) as read_con:
                cursor = read_con.execute(source_sql)
                while True:
                    rows = cursor.fetchmany(128)
                    if not rows:
                        break
                    for row in rows:
                        if cancelled():
                            return 0
                        source_block_count += 1
                        text = str(row["normalized_text"] or "")
                        counts = Counter(
                            encode_cjk_bigram(pair)
                            for index in range(max(0, len(text) - 1))
                            for pair in (text[index : index + 2],)
                            if is_cjk_bigram(pair)
                        )
                        pending.extend(
                            (code, int(row["id"]), int(count))
                            for code, count in counts.items()
                        )
                        if len(pending) >= 20_000:
                            flush()
                        now = time.monotonic()
                        if progress_callback is not None and now - last_reported >= 0.5:
                            progress_callback(
                                {
                                    "stage": "short_token_sidecar_build",
                                    "phase_label": (
                                        "正在构建中文两字快速索引"
                                        f"（{source_block_count:,}/{expected_blocks:,} 内容块）"
                                    ),
                                    "processed_blocks": source_block_count,
                                    "total_blocks": expected_blocks,
                                    "token_count": token_count + len(pending),
                                }
                            )
                            last_reported = now
            flush()
            if cancelled():
                return 0
            if source_block_count != expected_blocks:
                return 0
            candidate.execute(
                "CREATE INDEX idx_short_tokens_block ON short_tokens(block_id)"
            )
            candidate.executemany(
                "INSERT INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("version", str(SHORT_TOKEN_INDEX_VERSION)),
                    ("source_block_count", str(source_block_count)),
                    ("token_count", str(token_count)),
                    ("built_at", utc_now()),
                ),
            )
            check = candidate.execute("PRAGMA quick_check").fetchone()
            if check is None or str(check[0]).lower() != "ok":
                raise sqlite3.DatabaseError("short-token sidecar quick_check failed")
            candidate.commit()
            candidate.close()
            candidate = None

            # Candidate construction never writes the primary database.  Only
            # replacement plus its small generation flip needs a search-safe
            # coordinator reservation.
            while not coordinator.try_begin_index_write():
                if cancelled():
                    return 0
                time.sleep(0.05)
            reserved_write = True
            if progress_callback is not None:
                progress_callback(
                    {
                        "stage": "short_token_sidecar_publish",
                        "phase_label": "正在安全发布中文两字快速索引",
                        "processed_blocks": source_block_count,
                        "total_blocks": expected_blocks,
                    }
                )
            _replace_file_with_retry(candidate_path, active_path, timeout_seconds=5.0)
            with self.connect(timeout_seconds=5.0) as con:
                state = con.execute(
                    "SELECT value FROM index_state WHERE key = 'short_tokens_generation'"
                ).fetchone()
                try:
                    generation = int(state["value"]) + 1 if state is not None else 1
                except (TypeError, ValueError):
                    generation = 1
                self._set_index_state(con, "short_tokens_generation", str(generation))
                self._set_index_state(con, "short_tokens_active_generation", str(generation))
                self._set_index_state(con, "short_tokens_coverage_block_count", str(source_block_count))
                self._set_index_state(con, "search_projection_block_count", str(source_block_count))
                self._set_index_state(con, "short_tokens_dirty", "0")
                self._set_index_state(
                    con,
                    "short_token_index_version",
                    str(SHORT_TOKEN_INDEX_VERSION),
                )
            published = True
            return token_count
        finally:
            if candidate is not None:
                candidate.close()
            if not published:
                candidate_path.unlink(missing_ok=True)
            if reserved_write:
                coordinator.end_index_write()

    def rebuild_index_counters(
        self,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
    ) -> int:
        """Rebuild readiness counts without holding the SQLite write slot.

        The expensive aggregates run on one read snapshot.  Only replacing the
        small counter table and its generation markers uses a writer
        transaction.  The SQLite progress handler reports real VM execution
        while a large aggregate is running, so the finalization watchdog can
        distinguish a working query from a dead one.
        """

        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        video_placeholders = ",".join("?" for _ in video_extensions)
        generation = 1
        file_counters: list[tuple[int, int, int, int, int, int, int, int]] = []
        root_block_counts: dict[int, int] = {}
        root_document_counts: dict[int, int] = {}
        global_block_count = 0
        phase = "readiness_counters"
        phase_total = 4
        phase_completed = 0
        last_progress_at = 0.0
        progress_operations = 0

        def report(detail: str, *, completed: int | None = None) -> None:
            nonlocal phase_completed
            if progress_callback is None:
                return
            if completed is not None:
                phase_completed = max(phase_completed, int(completed))
            progress_callback(
                phase,
                phase_completed,
                phase_total,
                detail,
            )

        def sqlite_progress() -> int:
            nonlocal last_progress_at, progress_operations
            progress_operations += 100_000
            now = time.monotonic()
            if progress_callback is not None and now - last_progress_at >= 0.25:
                last_progress_at = now
                report(
                    f"SQLite 统计执行中，约 {progress_operations:,} 条虚拟机指令"
                )
            return 0

        with self.connect_read_only(
            timeout_seconds=30.0,
            operation="readiness_counters_read",
        ) as con:
            con.set_progress_handler(sqlite_progress, 100_000)
            row = con.execute(
                "SELECT value FROM index_state WHERE key = 'search_projection_generation'"
            ).fetchone()
            try:
                generation = max(1, int(row["value"]) if row is not None else 0)
            except (TypeError, ValueError):
                generation = 1
            root_ids = [
                int(root_row["id"])
                for root_row in con.execute("SELECT id FROM roots ORDER BY id")
            ]

            report("读取文件状态统计")
            file_rows = con.execute(
                f"""
                SELECT f.root_id,
                       COUNT(f.id) AS file_count,
                       COALESCE(SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                         AND e.id IS NULL THEN 1 ELSE 0 END), 0)
                           AS eligible_files,
                       COALESCE(SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                         AND e.id IS NULL
                                         AND f.parse_status IN ('success', 'metadata_only')
                                        THEN 1 ELSE 0 END), 0)
                           AS complete_files,
                       COALESCE(SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                         AND e.id IS NULL
                                         AND f.parse_status NOT IN ('success', 'metadata_only')
                                        THEN 1 ELSE 0 END), 0)
                           AS blocking_files,
                       COALESCE(SUM(CASE WHEN f.parse_status = 'metadata_only'
                                         AND e.id IS NULL THEN 1 ELSE 0 END), 0)
                           AS metadata_only_files,
                       COALESCE(SUM(CASE WHEN f.extension IN ({video_placeholders})
                                        THEN 1 ELSE 0 END), 0)
                           AS video_excluded,
                       COALESCE(SUM(CASE WHEN e.id IS NOT NULL THEN 1 ELSE 0 END), 0)
                           AS manual_excluded
                FROM files f
                LEFT JOIN index_scope_exclusions e
                  ON e.file_id = f.id
                 AND e.revoked_at IS NULL
                 AND e.invalidated_at IS NULL
                WHERE f.is_deleted = 0
                GROUP BY f.root_id
                """,
                (
                    *video_extensions,
                    *video_extensions,
                    *video_extensions,
                    *video_extensions,
                ),
            ).fetchall()
            file_counter_by_root = {
                int(row["root_id"]): (
                    int(row["file_count"] or 0),
                    int(row["eligible_files"] or 0),
                    int(row["complete_files"] or 0),
                    int(row["blocking_files"] or 0),
                    int(row["metadata_only_files"] or 0),
                    int(row["video_excluded"] or 0),
                    int(row["manual_excluded"] or 0),
                )
                for row in file_rows
            }
            file_counters = [
                (
                    root_id,
                    *file_counter_by_root.get(root_id, (0, 0, 0, 0, 0, 0, 0)),
                )
                for root_id in root_ids
            ]
            report("文件状态统计完成", completed=1)

            report("统计每个范围的正文块")
            root_block_rows = con.execute(
                """
                SELECT visible.root_id, COUNT(DISTINCT cb.id) AS block_count
                FROM files visible
                JOIN content_blocks cb
                  ON (cb.document_id IS NOT NULL
                      AND cb.document_id = visible.document_id)
                  OR (cb.document_id IS NULL
                      AND cb.file_id = visible.id)
                WHERE visible.is_deleted = 0
                GROUP BY visible.root_id
                """
            ).fetchall()
            root_block_counts = {
                int(row["root_id"]): int(row["block_count"] or 0)
                for row in root_block_rows
            }
            report("范围正文块统计完成", completed=2)

            report("统计每个范围的文档")
            root_document_rows = con.execute(
                """
                SELECT root_id, COUNT(DISTINCT document_id) AS document_count
                FROM files
                WHERE is_deleted = 0 AND document_id IS NOT NULL
                GROUP BY root_id
                """
            ).fetchall()
            root_document_counts = {
                int(row["root_id"]): int(row["document_count"] or 0)
                for row in root_document_rows
            }
            report("范围文档统计完成", completed=3)

            global_block_count = int(
                con.execute(
                    """
                    SELECT COUNT(DISTINCT cb.id)
                    FROM content_blocks cb
                    JOIN files visible
                      ON (cb.document_id IS NOT NULL
                          AND cb.document_id = visible.document_id)
                      OR (cb.document_id IS NULL
                          AND cb.file_id = visible.id)
                    WHERE visible.is_deleted = 0
                      AND NOT EXISTS (
                          SELECT 1
                          FROM index_scope_exclusions excluded
                          WHERE excluded.file_id = visible.id
                            AND excluded.revoked_at IS NULL
                            AND excluded.invalidated_at IS NULL
                      )
                    """
                ).fetchone()[0]
                or 0
            )
            report("全局正文块统计完成", completed=4)
            con.set_progress_handler(None, 0)

        now = utc_now()
        with self.connect(
            timeout_seconds=5.0,
            operation="readiness_counters_commit",
        ) as con:
            con.execute("DELETE FROM index_counters")
            con.executemany(
                """
                INSERT INTO index_counters(
                    root_id, file_count, eligible_files, complete_files,
                    blocking_files, metadata_only_files, video_excluded,
                    manual_excluded, block_count, document_count, generation,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        root_id,
                        file_count,
                        eligible_files,
                        complete_files,
                        blocking_files,
                        metadata_only_files,
                        video_excluded,
                        manual_excluded,
                        root_block_counts.get(root_id, 0),
                        root_document_counts.get(root_id, 0),
                        generation,
                        now,
                    )
                    for (
                        root_id,
                        file_count,
                        eligible_files,
                        complete_files,
                        blocking_files,
                        metadata_only_files,
                        video_excluded,
                        manual_excluded,
                    ) in file_counters
                ],
            )
            self._set_index_state(
                con,
                "readiness_counters_generation",
                str(generation),
            )
            self._set_index_state(
                con,
                "search_projection_block_count",
                str(global_block_count),
            )
            return int(
                con.execute("SELECT COUNT(*) FROM index_counters").fetchone()[0]
            )

    def _replace_document_item(
        self,
        con: sqlite3.Connection,
        item: dict[str, Any],
        now: str,
        *,
        update_fts: bool,
        remove_old_fts: bool,
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
        # A parser can cross the exclusion transaction boundary after its
        # safe-stop request. The exclusion audit wins: discard that late
        # artifact instead of reintroducing searchable rows.
        excluded_placeholders = ",".join("?" for _ in file_ids)
        excluded = con.execute(
            f"""
            SELECT 1 FROM index_scope_exclusions
            WHERE file_id IN ({excluded_placeholders})
              AND revoked_at IS NULL AND invalidated_at IS NULL
            LIMIT 1
            """,
            tuple(file_ids),
        ).fetchone()
        if excluded is not None:
            task_id = item.get("task_id")
            if task_id is not None:
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET status = 'cancelled', finished_at = COALESCE(finished_at, ?),
                        error_code = COALESCE(error_code, 'EXCLUSION_REQUESTED'),
                        error_message = COALESCE(error_message, '文件已人工排除')
                    WHERE id = ? AND status NOT IN ('complete', 'failed', 'superseded', 'invalidated', 'cancelled')
                    """,
                    (now, int(task_id)),
                )
            return
        old_document_ids = self._document_ids_for_files(con, file_ids)
        document_id: int | None = None
        placeholders = ",".join("?" for _ in file_ids)

        # Schema-v2 rows are owned directly by a file and have no document_id.
        # Remove those rows before attaching the path to its schema-v3 document,
        # otherwise an upgrade would keep both the old and new searchable text.
        if update_fts or remove_old_fts:
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
                self._delete_document_blocks(
                    con,
                    document_id,
                    update_fts=update_fts or remove_old_fts,
                )
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
            if update_fts or remove_old_fts:
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
        self._garbage_collect_documents(
            con,
            old_document_ids,
            update_fts=update_fts or remove_old_fts,
        )
        if (
            not update_fts
            and remove_old_fts
            and status in SUCCESSFUL_DOCUMENT_STATUSES
        ):
            # No schema migration is required: one index_state row per file
            # is a durable recovery queue for an interrupted incremental FTS
            # publication.  Content/task state and this marker commit in the
            # same transaction, so a crash cannot leave a successful file
            # permanently absent from FTS.
            con.executemany(
                """
                INSERT INTO index_state(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [
                    (f"content_fts_pending:{file_id}", now)
                    for file_id in file_ids
                ],
            )

    def pending_content_fts_file_ids(
        self,
        root_id: int | None = None,
    ) -> tuple[int, ...]:
        """Return durable incremental FTS work left by this or an older run."""

        root_clause = ""
        params: tuple[object, ...] = ()
        if root_id is not None:
            root_clause = " AND f.root_id = ?"
            params = (int(root_id),)
        with self.connect_read_only(
            timeout_seconds=5.0,
            operation="pending_content_fts_file_ids",
        ) as con:
            rows = con.execute(
                f"""
                SELECT CAST(substr(s.key, length('content_fts_pending:') + 1)
                            AS INTEGER) AS file_id
                FROM index_state s
                JOIN files f
                  ON f.id = CAST(substr(
                      s.key, length('content_fts_pending:') + 1
                  ) AS INTEGER)
                WHERE s.key LIKE 'content_fts_pending:%'{root_clause}
                ORDER BY file_id
                """,
                params,
            ).fetchall()
        return tuple(int(row["file_id"]) for row in rows)

    def sync_content_fts_for_files(
        self,
        file_ids: Sequence[int],
        *,
        batch_blocks: int = 64,
        batch_bytes: int = 512 * 1024,
        start_block_id: int = 0,
        cursor_callback: Callable[[int], None] | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int:
        """Synchronize selected documents into FTS using short transactions.

        Large documents are first committed without FTS maintenance so parser
        task state can keep moving. Their trigram rows are then inserted in
        small transactions here, releasing SQLite's single writer slot after
        every batch.
        """

        ids = sorted({int(value) for value in file_ids})
        if not ids:
            return 0
        started = time.perf_counter()
        log_event(
            "database.fts_sync.started",
            component="database",
            database_path=str(self.db_path),
            file_count=len(ids),
            batch_blocks=max(1, min(256, int(batch_blocks))),
            batch_bytes=max(1024, int(batch_bytes)),
        )
        # Keep every IN-list comfortably below SQLite's variable limit.  The
        # pending queue is global across roots, so a recovery pass can easily
        # contain more than one thousand files even though each root batch is
        # small.
        max_ids_per_call = 400
        if len(ids) > max_ids_per_call:
            if start_block_id or cursor_callback is not None:
                raise ValueError(
                    "FTS cursor resume requires a single file-id chunk"
                )
            synced_total = 0
            for offset in range(0, len(ids), max_ids_per_call):
                synced_total += self.sync_content_fts_for_files(
                    ids[offset : offset + max_ids_per_call],
                    batch_blocks=batch_blocks,
                    batch_bytes=batch_bytes,
                    progress_callback=progress_callback,
                )
            log_event(
                "database.fts_sync.completed",
                component="database",
                database_path=str(self.db_path),
                file_count=len(ids),
                synced_blocks=synced_total,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                split_batches=(len(ids) + max_ids_per_call - 1) // max_ids_per_call,
            )
            return synced_total
        placeholders = ",".join("?" for _ in ids)
        with self.connect_read_only(
            timeout_seconds=5.0,
            operation="fts_sync_document_lookup",
        ) as con:
            document_ids = [
                int(row["document_id"])
                for row in con.execute(
                    f"""
                    SELECT DISTINCT document_id FROM files
                    WHERE id IN ({placeholders})
                      AND document_id IS NOT NULL
                      AND is_deleted = 0
                    """,
                    tuple(ids),
                )
            ]
        if not document_ids:
            with self.connect(
                timeout_seconds=0.5,
                operation="fts_sync_pending_cleanup",
            ) as con:
                pending_keys = [
                    f"content_fts_pending:{file_id}" for file_id in ids
                ]
                pending_placeholders = ",".join("?" for _ in pending_keys)
                con.execute(
                    f"DELETE FROM index_state WHERE key IN ({pending_placeholders})",
                    tuple(pending_keys),
                )
                remaining = con.execute(
                    "SELECT 1 FROM index_state "
                    "WHERE key LIKE 'content_fts_pending:%' LIMIT 1"
                ).fetchone()
                self._set_index_state(
                    con,
                    "content_fts_dirty",
                    "0" if remaining is None else "1",
                )
            log_event(
                "database.fts_sync.completed",
                component="database",
                database_path=str(self.db_path),
                file_count=len(ids),
                synced_blocks=0,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                empty_document_ids=True,
            )
            return 0

        document_placeholders = ",".join("?" for _ in document_ids)
        with self.connect_read_only(
            timeout_seconds=5.0,
            operation="fts_sync_block_count",
        ) as con:
            total_blocks = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM content_blocks
                    WHERE document_id IN ({document_placeholders})
                    """,
                    tuple(document_ids),
                ).fetchone()[0]
            )

        inserted = 0
        cursor = max(0, int(start_block_id))
        limit = max(1, min(256, int(batch_blocks)))
        byte_limit = max(1024, int(batch_bytes))
        while True:
            with self.connect_read_only(
                timeout_seconds=5.0,
                operation="fts_sync_block_batch_read",
            ) as con:
                rows = con.execute(
                    f"""
                    SELECT cb.id, cb.file_id, COALESCE(f.filename, '') AS filename,
                           COALESCE(f.path, '') AS path, cb.location_text,
                           cb.normalized_text
                    FROM content_blocks cb
                    LEFT JOIN files f ON f.id = cb.file_id
                    WHERE cb.document_id IN ({document_placeholders})
                      AND cb.id > ?
                    ORDER BY cb.id
                    LIMIT ?
                    """,
                    (*document_ids, cursor, limit),
                ).fetchall()
            if not rows:
                break
            selected: list[sqlite3.Row] = []
            selected_bytes = 0
            for row in rows:
                row_bytes = len(
                    str(row["normalized_text"] or "").encode("utf-8")
                )
                if selected and selected_bytes + row_bytes > byte_limit:
                    break
                selected.append(row)
                selected_bytes += row_bytes
            row_ids = [int(row["id"]) for row in selected]
            row_placeholders = ",".join("?" for _ in row_ids)
            batch_started = time.perf_counter()
            with self.connect(
                timeout_seconds=0.5,
                operation="fts_sync_batch",
            ) as con:
                # Delete and replay only this bounded row batch. This makes
                # publication idempotent and prevents one multi-thousand-row
                # Excel/PDF from monopolising SQLite's single writer slot.
                con.execute(
                    f"DELETE FROM content_fts WHERE rowid IN ({row_placeholders})",
                    tuple(row_ids),
                )
                con.executemany(
                    """
                    INSERT INTO content_fts(
                        rowid, block_id, file_id, filename, path,
                        location_text, normalized_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            int(row["id"]),
                            int(row["id"]),
                            int(row["file_id"]),
                            str(row["filename"] or ""),
                            str(row["path"] or ""),
                            row["location_text"],
                            row["normalized_text"],
                        )
                        for row in selected
                    ],
                )
            inserted += len(selected)
            cursor = int(selected[-1]["id"])
            if cursor_callback is not None:
                cursor_callback(cursor)
            log_event(
                "database.fts_sync.batch_completed",
                component="database",
                database_path=str(self.db_path),
                batch_rows=len(selected),
                completed_rows=inserted,
                total_rows=total_blocks,
                last_block_id=cursor,
                batch_bytes=selected_bytes,
                elapsed_ms=int((time.perf_counter() - batch_started) * 1000),
            )
            if progress_callback is not None:
                progress_callback(inserted, total_blocks)

        with self.connect(
            timeout_seconds=0.5,
            operation="fts_sync_pending_cleanup",
        ) as con:
            pending_keys = [
                f"content_fts_pending:{file_id}" for file_id in ids
            ]
            pending_placeholders = ",".join("?" for _ in pending_keys)
            con.execute(
                f"DELETE FROM index_state WHERE key IN ({pending_placeholders})",
                tuple(pending_keys),
            )
            # Do not compare the whole content_blocks table with FTS here.
            # That query is quadratic on a large trigram index and was the
            # direct cause of the 9:00 freeze. Durable per-file markers are
            # the authoritative completion signal; a later run will replay
            # any marker left by a concurrent writer or interruption.
            remaining = con.execute(
                "SELECT 1 FROM index_state "
                "WHERE key LIKE 'content_fts_pending:%' LIMIT 1"
            ).fetchone()
            if remaining is None:
                self._set_index_state(con, "content_fts_dirty", "0")
            else:
                self._set_index_state(con, "content_fts_dirty", "1")
            self._advance_search_projection(con, updated_at=utc_now())
        log_event(
            "database.fts_sync.completed",
            component="database",
            database_path=str(self.db_path),
            file_count=len(ids),
            synced_blocks=inserted,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return inserted

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

    def queue_files_for_retry(self, file_ids: Sequence[int]) -> int:
        ids = sorted({int(value) for value in file_ids})
        if not ids:
            return 0
        placeholders = ",".join("?" for _ in ids)
        with self.connect(timeout_seconds=0.5) as con:
            cursor = con.execute(
                f"""
                UPDATE files
                SET parse_status = 'pending'
                WHERE id IN ({placeholders})
                  AND is_deleted = 0
                  AND parse_status IN (
                      'pending', 'processing', 'failed', 'failed_retryable',
                      'unsupported', 'skipped',
                      'ocr_disabled', 'ocr_failed', 'converter_missing',
                      'partial_success', 'password_protected', 'cancelled'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions excluded
                      WHERE excluded.file_id = files.id
                        AND excluded.revoked_at IS NULL
                        AND excluded.invalidated_at IS NULL
                  )
                """,
                tuple(ids),
            )
            return int(cursor.rowcount)

    def root_ids_for_files(self, file_ids: Sequence[int]) -> tuple[int, ...]:
        """Return enabled search roots owning the selected files."""

        ids = sorted({int(value) for value in file_ids})
        if not ids:
            return ()
        placeholders = ",".join("?" for _ in ids)
        with self.connect(timeout_seconds=0.5) as con:
            rows = con.execute(
                f"""
                SELECT DISTINCT f.root_id
                FROM files f
                JOIN roots r ON r.id = f.root_id
                WHERE f.id IN ({placeholders})
                  AND f.is_deleted = 0
                  AND r.enabled = 1
                ORDER BY f.root_id
                """,
                tuple(ids),
            ).fetchall()
        return tuple(int(row["root_id"]) for row in rows)

    def pending_paths_for_root(self, root_id: int) -> set[Path]:
        """Return files that still need planning, including orphan rows."""

        with self.connect_read_only() as con:
            rows = con.execute(
                """
                SELECT path
                FROM files
                WHERE root_id = ?
                  AND is_deleted = 0
                  AND (
                      parse_status IN ('pending', 'processing', 'cancelled')
                      OR (
                          parse_status = 'failed_retryable'
                          AND parse_error_code IN (
                              'STALE_PARSE_TASK_RECOVERED',
                              'STALE_INDEX_RUN_RECOVERED',
                              'INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT'
                          )
                      )
                  )
                """,
                (int(root_id),),
            ).fetchall()
        return {Path(str(row["path"])) for row in rows if row["path"]}

    def pending_source_paths_for_root(self, root_id: int) -> set[Path]:
        """Resolve orphan physical and ZIP-member rows back to scan inputs."""

        with self.connect_read_only() as con:
            rows = con.execute(
                """
                SELECT f.path, f.source_kind, container.path AS container_path
                FROM files f
                LEFT JOIN files container ON container.id = f.container_file_id
                WHERE f.root_id = ?
                  AND f.is_deleted = 0
                  AND f.parse_status IN ('pending', 'processing', 'cancelled')
                """,
                (int(root_id),),
            ).fetchall()
        result: set[Path] = set()
        for row in rows:
            source = row["container_path"] if row["source_kind"] == "zip_member" else row["path"]
            if source and " > " not in str(source):
                result.add(Path(str(source)))
        return result

    def paths_for_files(
        self,
        file_ids: Sequence[int],
        *,
        root_id: int | None = None,
    ) -> set[Path]:
        ids = sorted({int(value) for value in file_ids})
        if not ids:
            return set()
        placeholders = ",".join("?" for _ in ids)
        params: list[object] = [*ids]
        root_clause = ""
        if root_id is not None:
            root_clause = " AND root_id = ?"
            params.append(int(root_id))
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT path FROM files
                WHERE id IN ({placeholders})
                  AND is_deleted = 0{root_clause}
                """,
                tuple(params),
            ).fetchall()
        paths: set[Path] = set()
        for row in rows:
            text = str(row["path"])
            # ZIP members are displayed as ``archive > member`` but the
            # scheduler must rediscover their physical archive container.
            paths.add(Path(text.split(" > ", 1)[0]))
        return paths

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
        with self.connect_read_only(timeout_seconds=0.75) as con:
            rows = con.execute(
                """
                SELECT path FROM files
                WHERE root_id = ? AND is_deleted = 0 AND source_kind = 'file'
                """,
                (root_id,),
            ).fetchall()
            return {str(row["path"]) for row in rows}

    def active_file_references_for_root(self, root_id: int) -> dict[int, str]:
        """Return persisted NTFS file references for an incremental root."""

        # This is a read-only gate.  It must never wait behind the application
        # writer lock: foreground search can safely use the last committed
        # projection while an index batch is being committed.
        try:
            with self.connect_read_only(timeout_seconds=0.75) as con:
                rows = con.execute(
                    """
                    SELECT file_reference, path
                    FROM files
                    WHERE root_id = ? AND is_deleted = 0 AND source_kind = 'file'
                      AND file_reference IS NOT NULL AND file_reference != 0
                    """,
                    (int(root_id),),
                ).fetchall()
                return {
                    int(row["file_reference"]): str(row["path"])
                    for row in rows
                }
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                logger.warning(
                    "Incremental file-reference snapshot deferred: root_id=%s",
                    root_id,
                )
                return {}
            raise

    def active_file_count_for_root(self, root_id: int) -> int:
        with self.connect_read_only(timeout_seconds=0.75) as con:
            return int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM files
                    WHERE root_id = ? AND is_deleted = 0 AND source_kind = 'file'
                    """,
                    (int(root_id),),
                ).fetchone()[0]
            )

    def retryable_paths_for_root(self, root_id: int) -> list[Path]:
        """Return current non-success files that an explicit retry must revisit."""

        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        with self.connect_read_only(timeout_seconds=0.75) as con:
            rows = con.execute(
                f"""
                SELECT f.path
                FROM files AS f
                WHERE f.root_id = ?
                  AND f.is_deleted = 0
                  AND COALESCE(f.parse_status, '') != 'success'
                  AND NOT (
                      f.extension IN ({video_placeholders})
                      AND f.parse_status = 'metadata_only'
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions AS e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                ORDER BY f.path
                """,
                (int(root_id), *sorted(VIDEO_EXTENSIONS)),
            ).fetchall()
        return [Path(str(row["path"])) for row in rows]

    def load_usn_checkpoint(self, root_id: int) -> dict[str, str] | None:
        prefix = f"usn_checkpoint:{int(root_id)}:"
        with self.connect_read_only(timeout_seconds=0.75) as con:
            rows = con.execute(
                "SELECT key, value FROM index_state WHERE key LIKE ?",
                (prefix + "%",),
            ).fetchall()
        values = {
            str(row["key"])[len(prefix):]: str(row["value"])
            for row in rows
        }
        return values or None

    def save_usn_checkpoint(
        self,
        root_id: int,
        *,
        journal_id: int,
        next_usn: int,
        volume_serial: int,
    ) -> None:
        prefix = f"usn_checkpoint:{int(root_id)}:"
        values = {
            "journal_id": str(int(journal_id)),
            "next_usn": str(int(next_usn)),
            "volume_serial": str(int(volume_serial)),
        }
        with self.connect() as con:
            con.executemany(
                """
                INSERT INTO index_state(key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                [(prefix + key, value) for key, value in values.items()],
            )

    def usn_directories_for_root(self, root_id: int) -> dict[int, str]:
        with self.connect_read_only(timeout_seconds=0.75) as con:
            rows = con.execute(
                "SELECT file_reference, path FROM usn_directories WHERE root_id = ?",
                (int(root_id),),
            ).fetchall()
        return {int(row["file_reference"]): str(row["path"]) for row in rows}

    def apply_usn_directory_changes(
        self,
        root_id: int,
        upserts: Sequence[tuple[int, int | None, str]],
        removed_paths: Sequence[str] = (),
    ) -> int:
        """Apply directory moves/creates/deletes observed in a USN pass."""

        normalized_upserts = list(
            dict.fromkeys(
                (
                    int(reference),
                    None if parent is None else int(parent),
                    str(path),
                )
                for reference, parent, path in upserts
                if int(reference) != 0 and str(path)
            )
        )
        normalized_removed = {
            os.path.normcase(os.path.abspath(str(path))).rstrip("\\/")
            for path in removed_paths
            if str(path)
        }
        if not normalized_upserts and not normalized_removed:
            return 0

        with self.connect() as con:
            removed_references: list[int] = []
            if normalized_removed:
                rows = con.execute(
                    "SELECT file_reference, path FROM usn_directories WHERE root_id = ?",
                    (int(root_id),),
                ).fetchall()
                for row in rows:
                    stored = os.path.normcase(
                        os.path.abspath(str(row["path"]))
                    ).rstrip("\\/")
                    if any(
                        stored == removed
                        or stored.startswith(removed + os.sep)
                        for removed in normalized_removed
                    ):
                        removed_references.append(int(row["file_reference"]))
            for offset in range(0, len(removed_references), 500):
                batch = removed_references[offset : offset + 500]
                placeholders = ",".join("?" for _ in batch)
                con.execute(
                    f"DELETE FROM usn_directories WHERE root_id = ? "
                    f"AND file_reference IN ({placeholders})",
                    (int(root_id), *batch),
                )
            if normalized_upserts:
                con.executemany(
                    """
                    INSERT INTO usn_directories(
                        root_id, file_reference, parent_reference, path
                    ) VALUES (?, ?, ?, ?)
                    ON CONFLICT(root_id, file_reference) DO UPDATE SET
                        parent_reference = excluded.parent_reference,
                        path = excluded.path
                    """,
                    [
                        (int(root_id), reference, parent, path)
                        for reference, parent, path in normalized_upserts
                    ],
                )
        return len(normalized_upserts) + len(removed_references)

    def refresh_usn_directories(
        self,
        root_id: int,
        root_path: Path,
        *,
        include_subfolders: bool,
    ) -> int:
        """Record directory file references after a trusted full traversal."""

        root_path = Path(root_path)
        if not root_path.is_dir():
            return 0
        entries: list[tuple[int, int | None, str]] = []
        iterator = os.walk(root_path) if include_subfolders else [(str(root_path), [], [])]
        for current, _dirs, _files in iterator:
            directory = Path(current)
            try:
                reference = _sqlite_int64(getattr(directory.stat(), "st_ino", 0) or 0)
                parent = directory.parent
                parent_reference = (
                    _sqlite_int64(getattr(parent.stat(), "st_ino", 0) or 0)
                    if directory != root_path
                    else None
                )
            except OSError:
                continue
            if reference:
                entries.append((reference, parent_reference, str(directory)))
        with self.connect() as con:
            con.execute("DELETE FROM usn_directories WHERE root_id = ?", (int(root_id),))
            con.executemany(
                """
                INSERT INTO usn_directories(root_id, file_reference, parent_reference, path)
                VALUES (?, ?, ?, ?)
                """,
                [(int(root_id), reference, parent_reference, path) for reference, parent_reference, path in entries],
            )
        return len(entries)

    def export_usn_state(
        self,
        root_ids: Sequence[int] | None = None,
    ) -> dict[str, list[dict[str, object]]]:
        """Export restart-critical USN metadata without exporting search data."""

        ids = sorted({int(value) for value in (root_ids or ())})
        with self.connect_read_only(timeout_seconds=2.0) as con:
            if ids:
                checkpoint_predicate = " OR ".join(
                    "key LIKE ?" for _ in ids
                )
                checkpoint_rows = con.execute(
                    "SELECT key, value FROM index_state "
                    f"WHERE {checkpoint_predicate} ORDER BY key",
                    tuple(
                        f"usn_checkpoint:{root_id}:%"
                        for root_id in ids
                    ),
                ).fetchall()
                placeholders = ",".join("?" for _ in ids)
                directory_rows = con.execute(
                    "SELECT root_id, file_reference, parent_reference, path "
                    f"FROM usn_directories WHERE root_id IN ({placeholders}) "
                    "ORDER BY root_id, file_reference",
                    tuple(ids),
                ).fetchall()
            else:
                checkpoint_rows = con.execute(
                    "SELECT key, value FROM index_state "
                    "WHERE key LIKE 'usn_checkpoint:%' ORDER BY key"
                ).fetchall()
                directory_rows = con.execute(
                    "SELECT root_id, file_reference, parent_reference, path "
                    "FROM usn_directories ORDER BY root_id, file_reference"
                ).fetchall()
        return {
            "checkpoints": [dict(row) for row in checkpoint_rows],
            "directories": [dict(row) for row in directory_rows],
        }

    def import_usn_state(
        self,
        state: dict[str, object],
        *,
        root_ids: Sequence[int] | None = None,
    ) -> None:
        """Import USN metadata for candidate construction or control storage."""

        checkpoints = [
            dict(row)
            for row in (state.get("checkpoints") or [])
            if isinstance(row, dict)
            and row.get("key") is not None
            and str(row["key"]).startswith("usn_checkpoint:")
        ]
        directories = [
            dict(row)
            for row in (state.get("directories") or [])
            if isinstance(row, dict) and row.get("root_id") is not None
        ]
        ids = sorted({int(value) for value in (root_ids or ())})
        if not ids:
            ids = sorted(
                {
                    int(row["root_id"])
                    for row in directories
                }
                | {
                    root_id
                    for row in checkpoints
                    if (root_id := _usn_state_root_id(row.get("key")))
                    is not None
                }
            )
        if not ids:
            return
        with self.connect() as con:
            for root_id in ids:
                con.execute(
                    "DELETE FROM index_state WHERE key LIKE ?",
                    (f"usn_checkpoint:{root_id}:%",),
                )
                con.execute(
                    "DELETE FROM usn_directories WHERE root_id = ?",
                    (int(root_id),),
                )
            selected = set(ids)
            checkpoint_values = [
                (
                    str(row["key"]),
                    str(row.get("value") or ""),
                )
                for row in checkpoints
                if _usn_state_root_id(row.get("key")) in selected
            ]
            if checkpoint_values:
                con.executemany(
                    """
                    INSERT INTO index_state(key, value) VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    checkpoint_values,
                )
            directory_values = [
                (
                    int(row["root_id"]),
                    int(row["file_reference"]),
                    (
                        int(row["parent_reference"])
                        if row.get("parent_reference") is not None
                        else None
                    ),
                    str(row["path"]),
                )
                for row in directories
                if int(row["root_id"]) in selected
                and row.get("file_reference") is not None
                and row.get("path") is not None
            ]
            if directory_values:
                con.executemany(
                    """
                    INSERT INTO usn_directories(
                        root_id, file_reference, parent_reference, path
                    ) VALUES (?, ?, ?, ?)
                    """,
                    directory_values,
                )

    def relocate_file_reference(
        self,
        root_id: int,
        file_reference: int,
        destination: Path,
    ) -> bool:
        """Move an indexed physical file without invalidating its content."""

        destination = Path(destination)
        with self.connect() as con:
            row = con.execute(
                """
                SELECT id FROM files
                WHERE root_id = ? AND file_reference = ? AND is_deleted = 0
                  AND source_kind = 'file'
                LIMIT 1
                """,
                (int(root_id), int(file_reference)),
            ).fetchone()
            if row is None:
                return False
            file_id = int(row["id"])
            con.execute(
                """
                UPDATE files SET path = ?, filename = ?, extension = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (str(destination), destination.name, destination.suffix.lower(), utc_now(), file_id),
            )
            self._upsert_file_fts(con, file_id, destination.name, str(destination))
        return True

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

    def update_root_scan_time(
        self,
        root_id: int,
        status: str = "ready",
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        self._run_write_with_retry(
            lambda: self._update_root_scan_time_once(root_id, status),
            timeout_seconds=timeout_seconds,
            operation_name="update_root_scan_time",
        )

    def _update_root_scan_time_once(self, root_id: int, status: str) -> None:
        with self.connect(timeout_seconds=0.75) as con:
            con.execute(
                "UPDATE roots SET last_scan_at = ?, status = ?, updated_at = ? WHERE id = ?",
                (utc_now(), status, utc_now(), root_id),
            )
            if status == "ready":
                con.execute(
                    "DELETE FROM index_state WHERE key = ?",
                    (f"root_retry:{int(root_id)}",),
                )

    def root_retry_due(self, root_id: int) -> bool:
        """Return whether an automatic retry may run for one failed root."""

        with self.connect_read_only() as con:
            root = con.execute(
                "SELECT status FROM roots WHERE id = ?",
                (int(root_id),),
            ).fetchone()
            if root is None or str(root["status"] or "") != "retry_pending":
                return True
            row = con.execute(
                "SELECT value FROM index_state WHERE key = ?",
                (f"root_retry:{int(root_id)}",),
            ).fetchone()
        if row is None:
            return True
        try:
            return float(json.loads(str(row["value"])).get("next_retry_epoch", 0)) <= time.time()
        except (TypeError, ValueError, json.JSONDecodeError):
            return True

    def mark_root_retry_pending(
        self,
        root_id: int,
        error: BaseException,
        *,
        timeout_seconds: float = 0.25,
    ) -> None:
        """Record an isolated root failure with bounded exponential backoff."""

        self._run_write_with_retry(
            lambda: self._mark_root_retry_pending_once(root_id, error),
            timeout_seconds=max(1.0, timeout_seconds),
            operation_name="mark_root_retry_pending",
        )

    def _mark_root_retry_pending_once(
        self,
        root_id: int,
        error: BaseException,
    ) -> None:
        """Persist retry metadata in one short transaction."""

        key = f"root_retry:{int(root_id)}"
        now = time.time()
        with self.connect(timeout_seconds=0.75) as con:
            row = con.execute(
                "SELECT value FROM index_state WHERE key = ?",
                (key,),
            ).fetchone()
            try:
                previous_attempt = int(json.loads(str(row["value"])).get("attempt", 0)) if row else 0
            except (TypeError, ValueError, json.JSONDecodeError):
                previous_attempt = 0
            attempt = min(previous_attempt + 1, len(ROOT_RETRY_DELAYS_SECONDS))
            delay = ROOT_RETRY_DELAYS_SECONDS[attempt - 1]
            payload = json.dumps(
                {
                    "attempt": attempt,
                    "next_retry_epoch": now + delay,
                    "last_error": f"{type(error).__name__}: {error}"[:1000],
                },
                ensure_ascii=False,
            )
            self._set_index_state(con, key, payload)
            con.execute(
                "UPDATE roots SET status = 'retry_pending', updated_at = ? WHERE id = ?",
                (utc_now(), int(root_id)),
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
        """Persist task state without allowing a transient writer lock to abort a root."""

        if not task_ids:
            return
        # A large OCR/PDF result can legitimately hold the single SQLite
        # writer for longer than the old 20-second retry window.  Task state
        # is non-critical telemetry, but it must not make the root fail just
        # because the writer is finishing a durable result.
        deadline = time.monotonic() + 5.0
        delay = 0.05
        while True:
            try:
                self._mark_tasks_running(
                    task_ids,
                    timeout_seconds=min(2.0, max(0.0, deadline - time.monotonic())),
                )
                return
            except sqlite3.OperationalError as exc:
                if not _is_database_busy(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(delay)
                delay = min(0.75, delay * 1.7)

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
        with self.connect_read_only(timeout_seconds=5.0) as con:
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

    def _recover_interrupted_tasks(self, con: sqlite3.Connection) -> None:
        """Recover only dead, heartbeat-stale runs with no live task."""

        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - STALE_INDEX_RUN_TIMEOUT_SECONDS
        stale_runs: list[str] = []
        for row in con.execute(
            "SELECT id, started_at, heartbeat_at, owner_pid FROM index_runs WHERE status = 'running'"
        ).fetchall():
            heartbeat = row["heartbeat_at"] or row["started_at"]
            try:
                heartbeat_at = datetime.fromisoformat(str(heartbeat)).timestamp()
            except (TypeError, ValueError, OSError):
                heartbeat_at = 0.0
            if heartbeat_at > cutoff:
                continue
            if row["owner_pid"] is not None and self._pid_is_alive(row["owner_pid"]):
                continue
            tasks = con.execute(
                """
                SELECT * FROM parse_tasks
                WHERE run_id = ?
                  AND status IN ('queued', 'running', 'spooled', 'blocked', 'paused')
                """,
                (str(row["id"]),),
            ).fetchall()
            if any(self._row_is_real_active_task(task, now) for task in tasks):
                continue
            stale_runs.append(str(row["id"]))
        # Legacy/orphan OCR leases have no running owner batch to heartbeat.
        # They are safe to reclaim, while requests attached to a live index
        # run remain untouched even when another process opens the database.
        con.execute(
            """
            UPDATE ocr_requests
            SET status = 'queued', lease_owner = NULL,
                lease_expires_at = NULL
            WHERE status IN ('running', 'spooled', 'paused')
              AND NOT EXISTS (
                  SELECT 1
                  FROM parse_tasks parent
                  JOIN index_runs owner ON owner.id = parent.run_id
                  WHERE parent.id = ocr_requests.parent_task_id
                    AND owner.status = 'running'
              )
            """
        )
        if not stale_runs:
            return
        placeholders = ",".join("?" for _ in stale_runs)
        timestamp = utc_now()
        con.execute(
            f"""
            UPDATE index_runs
            SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                ended_reason = COALESCE(ended_reason, 'STALE_INDEX_RUN_RECOVERED')
            WHERE id IN ({placeholders}) AND status = 'running'
            """,
            (timestamp, *stale_runs),
        )
        con.execute(
            f"""
            UPDATE parse_task_attempts
            SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                error_code = COALESCE(error_code, 'STALE_PARSE_TASK_RECOVERED'),
                error_message = COALESCE(error_message, '程序启动时恢复旧批次残留任务')
            WHERE task_id IN (SELECT id FROM parse_tasks WHERE run_id IN ({placeholders}))
              AND status IN ('running', 'spooled')
            """,
            (timestamp, *stale_runs),
        )
        con.execute(
            f"""
            UPDATE parse_tasks
            SET status = 'failed_retryable', finished_at = COALESCE(finished_at, ?),
                lease_owner = NULL, lease_expires_at = NULL,
                error_code = COALESCE(error_code, 'STALE_PARSE_TASK_RECOVERED'),
                error_message = COALESCE(error_message, '程序启动时恢复旧批次残留任务')
            WHERE run_id IN ({placeholders})
              AND status NOT IN ('complete', 'failed', 'failed_retryable',
                                 'superseded', 'invalidated', 'cancelled')
              AND NOT (
                  status = 'spooled'
                  AND COALESCE(spool_path, '') <> ''
                  AND COALESCE(spool_checksum, '') <> ''
              )
            """,
            (timestamp, *stale_runs),
        )
        con.execute(
            f"""
            UPDATE files SET parse_status = 'failed_retryable',
                parse_error_code = COALESCE(parse_error_code, 'STALE_PARSE_TASK_RECOVERED'),
                parse_error_message = COALESCE(parse_error_message, '程序启动时恢复旧批次残留任务')
            WHERE id IN (
                SELECT file_id FROM parse_tasks
                WHERE run_id IN ({placeholders}) AND error_code = 'STALE_PARSE_TASK_RECOVERED'
            )
              AND parse_status IN ('pending', 'processing', 'cancelled')
            """,
            stale_runs,
        )
        con.execute(
            f"""
            UPDATE ocr_requests
            SET status = 'queued', lease_owner = NULL,
                lease_expires_at = NULL
            WHERE parent_task_id IN (
                SELECT id FROM parse_tasks WHERE run_id IN ({placeholders})
            )
              AND status IN ('running', 'spooled', 'paused')
            """,
            stale_runs,
        )

    def start_index_run(
        self,
        metrics: IndexRunMetrics,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        with self.connect(timeout_seconds=timeout_seconds) as con:
            con.execute(
                """
                INSERT OR REPLACE INTO index_runs(
                    id, started_at, mode, status, owner_pid, owner_instance_id,
                    heartbeat_at, cancel_requested_at, ended_reason
                ) VALUES (?, ?, ?, 'running', ?, ?, ?, NULL, NULL)
                """,
                (
                    metrics.run_id,
                    utc_now(),
                    metrics.mode,
                    os.getpid(),
                    f"pid-{os.getpid()}-{uuid.uuid4().hex[:12]}",
                    utc_now(),
                ),
            )

    def heartbeat_index_run(self, run_id: str) -> None:
        with self.connect(timeout_seconds=0.25) as con:
            con.execute(
                "UPDATE index_runs SET heartbeat_at = ? WHERE id = ? AND status = 'running'",
                (utc_now(), str(run_id)),
            )

    def finish_index_run(
        self,
        metrics: IndexRunMetrics,
        status: str,
        summary: dict[str, object],
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        parse_ms = sum(metrics.parse_ms_by_lane.values())
        with self.connect(timeout_seconds=timeout_seconds) as con:
            self._converge_index_run_tasks(con, metrics.run_id, status)
            con.execute(
                """
                UPDATE index_runs SET
                    finished_at = ?, status = ?, discovered_files = ?, discovered_bytes = ?,
                    scan_ms = ?, parse_ms = ?, write_ms = ?, fts_ms = ?, total_ms = ?,
                    peak_rss_bytes = ?, summary_json = ?, ended_reason = ?
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
                    status,
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
        with self.connect(
            timeout_seconds=0.75,
            operation="record_file_metrics",
        ) as con:
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
        with self.connect_read_only(
            timeout_seconds=2.0,
            operation="active_file_count",
        ) as con:
            return int(con.execute("SELECT COUNT(*) AS n FROM files WHERE is_deleted = 0").fetchone()["n"])

    def data_commit_sequence(self) -> int:
        """Return the durable data-commit sequence used by stall diagnostics."""

        try:
            with self.connect_read_only(
                timeout_seconds=0.75,
                operation="data_commit_sequence",
            ) as con:
                row = con.execute(
                    "SELECT value FROM index_state WHERE key = 'data_commit_sequence'"
                ).fetchone()
            return int(row["value"]) if row is not None else 0
        except (sqlite3.Error, TypeError, ValueError):
            return 0

    def root_completion(self, root_id: int) -> dict[str, int]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        failure_placeholders = ",".join("?" for _ in FINAL_FAILURE_STATUSES)
        with self.connect_read_only(
            timeout_seconds=2.0,
            operation="root_completion",
        ) as con:
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
                    ,SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                  AND e.id IS NULL
                                  AND f.parse_status IN ('pending', 'processing', 'cancelled')
                             THEN 1 ELSE 0 END) AS in_progress
                    ,SUM(CASE WHEN f.extension NOT IN ({video_placeholders})
                                  AND e.id IS NULL
                                  AND f.parse_status IN ({failure_placeholders})
                             THEN 1 ELSE 0 END) AS failed
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
                    *video_extensions,
                    *video_extensions,
                    *sorted(FINAL_FAILURE_STATUSES),
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
            "in_progress": int(row["in_progress"] or 0),
            "failed": int(row["failed"] or 0),
        }

    def begin_deferred_fts(self) -> None:
        with self.connect() as con:
            self._set_index_state(con, "content_fts_dirty", "1")
            self._set_index_state(con, "full_batch_incomplete", "1")
            self._set_index_state(con, "short_tokens_dirty", "1")

    def set_partial_search_projection_active(self, active: bool) -> None:
        with self.connect() as con:
            self._set_index_state(
                con,
                "partial_search_projection_active",
                "1" if active else "0",
            )
            if active:
                row = con.execute(
                    "SELECT value FROM index_state WHERE key = 'search_projection_generation'"
                ).fetchone()
                try:
                    generation = int(row["value"]) if row is not None else 0
                except (TypeError, ValueError):
                    generation = 0
                if generation <= 0:
                    self._set_index_state(con, "search_projection_generation", "1")
                    self._set_index_state(
                        con,
                        "search_projection_updated_at",
                        utc_now(),
                    )

    def converge_index_run_tasks(self, run_id: str, status: str) -> int:
        """Close non-terminal tasks left by an index run.

        This is also called when run telemetry itself could not be persisted,
        so task lifecycle state never depends on the optional metrics row.
        """

        return int(
            self._run_write_with_retry(
                lambda: self._converge_index_run_tasks_once(run_id, status),
                timeout_seconds=5.0,
                operation_name="converge_index_run_tasks",
            )
        )

    def _converge_index_run_tasks_once(self, run_id: str, status: str) -> int:
        with self.connect(timeout_seconds=0.75) as con:
            converged = self._converge_index_run_tasks(con, run_id, status)
            con.execute(
                """
                UPDATE index_runs
                SET status = ?, finished_at = COALESCE(finished_at, ?)
                WHERE id = ? AND status = 'running'
                """,
                (str(status), utc_now(), str(run_id)),
            )
            return converged

    @staticmethod
    def _converge_index_run_tasks(
        con: sqlite3.Connection,
        run_id: str,
        status: str,
    ) -> int:
        now = utc_now()
        terminal = tuple(sorted(TERMINAL_PARSE_TASK_STATUSES))
        placeholders = ",".join("?" for _ in terminal)
        rows = con.execute(
            f"""
            SELECT id, file_id FROM parse_tasks
            WHERE run_id = ? AND status NOT IN ({placeholders})
              AND NOT (
                  status = 'spooled'
                  AND COALESCE(spool_path, '') <> ''
                  AND COALESCE(spool_checksum, '') <> ''
              )
            """,
            (str(run_id), *terminal),
        ).fetchall()
        if not rows:
            return 0
        ids = [int(row["id"]) for row in rows]
        task_placeholders = ",".join("?" for _ in ids)
        error_code = "INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT"
        message = "索引批次结束时任务没有终态结果"
        # Never expand a whole run's task IDs into an SQL IN-list. Large
        # drives can exceed SQLite's variable limit exactly while the error
        # handler is trying to recover an interrupted run, masking the real
        # failure with "too many SQL variables".
        con.execute(
            "CREATE TEMP TABLE IF NOT EXISTS _converge_task_ids "
            "(id INTEGER PRIMARY KEY)"
        )
        con.execute("DELETE FROM _converge_task_ids")
        try:
            con.executemany(
                "INSERT INTO _converge_task_ids(id) VALUES (?)",
                ((task_id,) for task_id in ids),
            )
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'failed_retryable', finished_at = COALESCE(finished_at, ?),
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_code = COALESCE(error_code, ?),
                    error_message = COALESCE(error_message, ?)
                WHERE id IN (SELECT id FROM _converge_task_ids)
                """,
                (now, error_code, message),
            )
            con.execute(
                """
                UPDATE parse_task_attempts
                SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                    error_code = COALESCE(error_code, ?),
                    error_message = COALESCE(error_message, ?)
                WHERE task_id IN (SELECT id FROM _converge_task_ids)
                  AND status IN ('running', 'spooled')
                """,
                (now, error_code, message),
            )
            con.execute(
                """
                UPDATE files
                SET parse_status = CASE WHEN parse_status IN ('pending', 'processing', 'cancelled') THEN 'failed_retryable' ELSE parse_status END,
                    parse_error_code = CASE WHEN parse_status IN ('pending', 'processing', 'cancelled') THEN COALESCE(parse_error_code, ?) ELSE parse_error_code END,
                    parse_error_message = CASE WHEN parse_status IN ('pending', 'processing', 'cancelled') THEN COALESCE(parse_error_message, ?) ELSE parse_error_message END
                WHERE id IN (
                    SELECT file_id FROM parse_tasks
                    WHERE id IN (SELECT id FROM _converge_task_ids)
                )
                """,
                (error_code, message),
            )
            return len(ids)
        finally:
            con.execute("DELETE FROM _converge_task_ids")

    @staticmethod
    def _pid_is_alive(worker_pid: object) -> bool:
        try:
            pid = int(worker_pid or 0)
        except (TypeError, ValueError):
            return False
        if pid <= 0:
            return False
        if pid == os.getpid():
            return True
        try:
            import psutil

            process = psutil.Process(pid)
            return bool(
                process.is_running()
                and process.status() != psutil.STATUS_ZOMBIE
            )
        except (psutil.NoSuchProcess, psutil.ZombieProcess, OSError, SystemError):
            return False
        except (ImportError, AttributeError):
            # POSIX fallback for minimal environments. Never use os.kill(...,
            # 0) on Windows: CPython maps unsupported signals to
            # TerminateProcess there, which can kill the process being probed.
            if os.name == "nt":
                return False
            try:
                os.kill(pid, 0)
            except (OSError, PermissionError, SystemError):
                return False
            return True

    @classmethod
    def _task_is_recent(
        cls,
        row: sqlite3.Row,
        now: datetime,
        stale_after_seconds: int = STALE_PARSE_TASK_TIMEOUT_SECONDS,
    ) -> bool:
        for key in (
            "last_semantic_progress_at",
            "last_progress_at",
            "started_at",
            "created_at",
        ):
            value = row[key] if key in row.keys() else None
            if not value:
                continue
            try:
                stamp = datetime.fromisoformat(str(value))
            except ValueError:
                continue
            if (now - stamp).total_seconds() <= max(0, int(stale_after_seconds)):
                return True
        return False

    @classmethod
    def _row_is_real_active_task(
        cls,
        row: sqlite3.Row,
        now: datetime,
        stale_after_seconds: int = STALE_PARSE_TASK_TIMEOUT_SECONDS,
        no_worker_after_seconds: int = NO_WORKER_TASK_TIMEOUT_SECONDS,
    ) -> bool:
        status = str(row["status"] or "")
        if status not in {"queued", "running", "spooled", "blocked", "paused"}:
            return False
        run_status_value = (
            row["run_status"] if "run_status" in row.keys() else "running"
        )
        run_status = str(run_status_value or "")
        if run_status not in {"", "running"}:
            return False
        try:
            worker_pid = int(row["worker_pid"] or 0)
        except (OSError, PermissionError, SystemError, ValueError, TypeError):
            worker_pid = 0
        if worker_pid > 0:
            return cls._pid_is_alive(worker_pid) or cls._task_is_recent(
                row,
                now,
                stale_after_seconds,
            )
        # A running row without a worker or any progress is a bookkeeping
        # orphan, not an active parser. Give process startup a short grace
        # period, then allow recovery while the foreground run is still open.
        has_progress = any(
            bool(row[key] if key in row.keys() else None)
            for key in ("last_semantic_progress_at", "last_progress_at")
        )
        return cls._task_is_recent(
            row,
            now,
            stale_after_seconds
            if has_progress or status != "running"
            else min(stale_after_seconds, max(1, int(no_worker_after_seconds))),
        )

    def reconcile_stale_parse_tasks(
        self,
        *,
        stale_after_seconds: int = STALE_PARSE_TASK_TIMEOUT_SECONDS,
        no_worker_after_seconds: int = NO_WORKER_TASK_TIMEOUT_SECONDS,
        active_run_id: str | None = None,
    ) -> int:
        """Mark old-batch, dead-worker and no-heartbeat tasks retryable."""

        now_dt = datetime.now(timezone.utc)
        now = utc_now()
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT pt.*, ir.status AS run_status,
                       pta.last_progress_at
                FROM parse_tasks pt
                LEFT JOIN index_runs ir ON ir.id = pt.run_id
                LEFT JOIN parse_task_attempts pta ON pta.id = (
                    SELECT latest.id FROM parse_task_attempts latest
                    WHERE latest.task_id = pt.id
                    ORDER BY latest.attempt_no DESC LIMIT 1
                )
                WHERE pt.status IN ('queued', 'running', 'spooled', 'blocked', 'paused')
                """
            ).fetchall()
            protected_run_id = str(active_run_id or "")
            if protected_run_id:
                # A live scheduler may have submitted a task before SQLite
                # accepted the small bookkeeping UPDATE.  Such a row can
                # still look queued/running with no worker PID.  The caller
                # owns this run and its in-memory executor/watchdog is the
                # source of truth until the run reaches its final convergence
                # phase; do not recycle these rows mid-run.
                rows = [
                    row
                    for row in rows
                    if not (
                        str(row["run_id"] or "") == protected_run_id
                        and str(row["run_status"] or "") in {"", "running"}
                    )
                ]
            stale_ids = [
                int(row["id"])
                for row in rows
                if str(row["task_type"] or "") not in {
                    "pdf_native_page",
                    "pdf_ocr_page",
                }
                and not (
                    str(row["status"] or "") == "spooled"
                    and str(row["spool_path"] or "")
                    and str(row["spool_checksum"] or "")
                )
                and not self._row_is_real_active_task(
                    row,
                    now_dt,
                    stale_after_seconds,
                    no_worker_after_seconds,
                )
                or (
                    str(row["task_type"] or "") in {
                        "pdf_native_page",
                        "pdf_ocr_page",
                    }
                    and not (
                        str(row["status"] or "") == "spooled"
                        and str(row["spool_path"] or "")
                        and str(row["spool_checksum"] or "")
                    )
                    and not self._row_is_real_active_task(
                        row,
                        now_dt,
                        max(
                            int(stale_after_seconds),
                            PDF_PAGE_STALE_TASK_TIMEOUT_SECONDS,
                        ),
                        max(
                            int(no_worker_after_seconds),
                            PDF_PAGE_NO_WORKER_TIMEOUT_SECONDS,
                        ),
                    )
                )
            ]
            if not stale_ids:
                return 0
            stale_id_set = set(stale_ids)
            for row in rows:
                if int(row["id"]) not in stale_id_set:
                    continue
                is_pdf_page = str(row["task_type"] or "") in {
                    "pdf_native_page",
                    "pdf_ocr_page",
                }
                log_event(
                    "parse.task.stale_recovered",
                    level=logging.WARNING,
                    component="database",
                    task_id=int(row["id"]),
                    file_id=int(row["file_id"]),
                    parent_task_id=(
                        int(row["parent_task_id"])
                        if row["parent_task_id"] is not None
                        else None
                    ),
                    run_id=str(row["run_id"] or ""),
                    task_type=str(row["task_type"] or ""),
                    previous_status=str(row["status"] or ""),
                    worker_pid=(
                        int(row["worker_pid"])
                        if row["worker_pid"] is not None
                        else None
                    ),
                    run_status=str(row["run_status"] or ""),
                    started_at=str(row["started_at"] or ""),
                    last_progress_at=str(row["last_progress_at"] or ""),
                    last_semantic_progress_at=str(
                        row["last_semantic_progress_at"] or ""
                    ),
                    stale_after_seconds=(
                        max(int(stale_after_seconds), PDF_PAGE_STALE_TASK_TIMEOUT_SECONDS)
                        if is_pdf_page
                        else int(stale_after_seconds)
                    ),
                    no_worker_after_seconds=(
                        max(int(no_worker_after_seconds), PDF_PAGE_NO_WORKER_TIMEOUT_SECONDS)
                        if is_pdf_page
                        else int(no_worker_after_seconds)
                    ),
                    recovery_reason="no_live_worker_or_recent_progress",
                )
            con.execute(
                "CREATE TEMP TABLE IF NOT EXISTS _stale_parse_task_ids "
                "(id INTEGER PRIMARY KEY)"
            )
            con.execute("DELETE FROM _stale_parse_task_ids")
            try:
                con.executemany(
                    "INSERT INTO _stale_parse_task_ids(id) VALUES (?)",
                    ((task_id,) for task_id in stale_ids),
                )
                con.execute(
                    """
                UPDATE parse_tasks
                SET status = CASE
                        WHEN file_id IN (
                            SELECT id FROM files
                            WHERE parse_status IN ('success', 'metadata_only')
                        ) THEN 'invalidated'
                        ELSE 'failed_retryable'
                    END,
                    finished_at = COALESCE(finished_at, ?),
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_code = CASE
                        WHEN file_id IN (
                            SELECT id FROM files
                            WHERE parse_status IN ('success', 'metadata_only')
                        ) THEN COALESCE(error_code, 'STALE_TASK_AFTER_SUCCESS')
                        ELSE 'STALE_PARSE_TASK_RECOVERED'
                    END,
                    error_message = CASE
                        WHEN file_id IN (
                            SELECT id FROM files
                            WHERE parse_status IN ('success', 'metadata_only')
                        ) THEN COALESCE(error_message, '文件已有成功终态，残留解析任务已失效')
                        ELSE '无活动运行、worker 或心跳的残留任务已恢复'
                    END
                WHERE id IN (SELECT id FROM _stale_parse_task_ids)
                """,
                    (now,),
                )
                con.execute(
                    """
                UPDATE parse_task_attempts
                SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                    error_code = COALESCE(error_code, 'STALE_PARSE_TASK_RECOVERED'),
                    error_message = COALESCE(error_message, 'worker 已退出或任务批次已结束')
                WHERE task_id IN (SELECT id FROM _stale_parse_task_ids)
                  AND status IN ('running', 'spooled')
                """,
                    (now,),
                )
                con.execute(
                    """
                UPDATE files
                SET parse_status = 'failed_retryable',
                    parse_error_code = COALESCE(parse_error_code, 'STALE_PARSE_TASK_RECOVERED'),
                    parse_error_message = COALESCE(parse_error_message, '残留解析任务已恢复，可重新尝试')
                WHERE id IN (
                    SELECT file_id FROM parse_tasks
                    WHERE id IN (SELECT id FROM _stale_parse_task_ids)
                )
                  AND (
                      parse_status IN ('pending', 'processing', 'cancelled')
                      OR (
                          parse_status = 'failed_retryable'
                          AND parse_error_code IN (
                              'STALE_PARSE_TASK_RECOVERED',
                              'STALE_INDEX_RUN_RECOVERED',
                              'INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT'
                          )
                      )
                  )
                """,
                    (),
                )
                return len(stale_ids)
            finally:
                con.execute("DELETE FROM _stale_parse_task_ids")

    def reconcile_orphan_pending_files(self) -> int:
        """Close genuinely stale pending files after an index run stops.

        A planner can persist file metadata before it has submitted the
        corresponding parse task. Do not immediately convert those fresh
        rows to failures when a run ends; give the next retry/recovery pass a
        bounded grace period first.
        """

        with self.connect() as con:
            if int(con.execute("SELECT COUNT(*) FROM index_runs WHERE status = 'running'").fetchone()[0]) > 0:
                return 0
            grace_seconds = max(0, int(STALE_INDEX_RUN_TIMEOUT_SECONDS))
            recent_run = con.execute(
                """
                SELECT 1
                FROM index_runs
                WHERE finished_at IS NOT NULL
                  AND julianday(finished_at) >= julianday('now', ?)
                LIMIT 1
                """,
                (f"-{grace_seconds} seconds",),
            ).fetchone()
            if recent_run is not None:
                return 0
            # A file without a parse task is a planning backlog, not a parse
            # failure. Leave its pending state intact so the next directory
            # or USN scan can recreate the task. This method is retained as a
            # compatibility hook for callers that expect a reconciliation
            # count, but it must never manufacture a user-visible failure.
            rows = con.execute(
                """
                SELECT COUNT(*) AS n
                FROM files
                WHERE is_deleted = 0
                  AND parse_status IN ('pending', 'processing', 'cancelled')
                  AND last_seen_at IS NOT NULL
                  AND julianday(last_seen_at) < julianday('now', ?)
                  AND NOT EXISTS (
                      SELECT 1 FROM parse_tasks pt
                      WHERE pt.file_id = files.id
                        AND pt.status IN ('queued', 'running', 'spooled', 'blocked', 'paused')
                  )
                """,
                (f"-{grace_seconds} seconds",),
            ).fetchone()
            return max(0, int(rows["n"] or 0))

    def active_parse_task_ids(self, file_ids: Sequence[int]) -> list[int]:
        ids = list(dict.fromkeys(int(value) for value in file_ids))
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT pt.*, ir.status AS run_status,
                       pta.last_progress_at
                FROM parse_tasks pt
                LEFT JOIN index_runs ir ON ir.id = pt.run_id
                LEFT JOIN parse_task_attempts pta ON pta.id = (
                    SELECT latest.id FROM parse_task_attempts latest
                    WHERE latest.task_id = pt.id
                    ORDER BY latest.attempt_no DESC LIMIT 1
                )
                WHERE pt.file_id IN ({placeholders})
                  AND pt.status IN ('queued', 'running', 'spooled', 'blocked', 'paused')
                """,
                tuple(ids),
            ).fetchall()
        now = datetime.now(timezone.utc)
        return [int(row["id"]) for row in rows if self._row_is_real_active_task(row, now)]

    def exclusion_task_ids(self, file_ids: Sequence[int]) -> list[int]:
        """Return tasks that an exclusion request should settle immediately."""

        ids = list(dict.fromkeys(int(value) for value in file_ids))
        if not ids:
            return []
        self.reconcile_stale_parse_tasks()
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT pt.*, ir.status AS run_status,
                       pta.last_progress_at
                FROM parse_tasks pt
                LEFT JOIN index_runs ir ON ir.id = pt.run_id
                LEFT JOIN parse_task_attempts pta ON pta.id = (
                    SELECT latest.id FROM parse_task_attempts latest
                    WHERE latest.task_id = pt.id
                    ORDER BY latest.attempt_no DESC LIMIT 1
                )
                WHERE pt.file_id IN ({placeholders})
                  AND pt.status IN ('queued', 'blocked', 'paused', 'running', 'spooled')
                """,
                tuple(ids),
            ).fetchall()
        now = datetime.now(timezone.utc)
        return [
            int(row["id"])
            for row in rows
            if self._row_is_real_active_task(row, now)
        ]

    def has_incomplete_full_batch(self) -> bool:
        with self.connect() as con:
            row = con.execute(
                "SELECT value FROM index_state WHERE key = 'full_batch_incomplete'"
            ).fetchone()
            return row is not None and str(row["value"]) == "1"

    def mark_full_batch_complete(self) -> None:
        with self.connect() as con:
            self._set_index_state(con, "full_batch_incomplete", "0")
            self._set_index_state(con, "partial_search_projection_active", "0")

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
        real_active_task_ids = set(self.exclusion_task_ids(ids))
        if real_active_task_ids:
            raise RuntimeError("选中文件存在活动解析任务，请等待任务安全结束后再排除")
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
            if real_active_task_ids:
                active_placeholders = ",".join("?" for _ in real_active_task_ids)
                con.execute(
                    f"""
                    UPDATE parse_tasks
                    SET status = 'cancelled', finished_at = COALESCE(finished_at, ?),
                        lease_owner = NULL, lease_expires_at = NULL,
                        error_code = 'EXCLUSION_REQUESTED',
                        error_message = '选中文件请求人工排除，任务已在安全点停止'
                    WHERE id IN ({active_placeholders})
                    """,
                    (now, *sorted(real_active_task_ids)),
                )
                con.execute(
                    f"""
                    UPDATE parse_task_attempts
                    SET status = 'interrupted', finished_at = COALESCE(finished_at, ?),
                        error_code = 'EXCLUSION_REQUESTED',
                        error_message = '选中文件请求人工排除'
                    WHERE task_id IN ({active_placeholders}) AND status = 'running'
                    """,
                    (now, *sorted(real_active_task_ids)),
                )
                con.execute(
                    f"""
                    UPDATE files
                    SET parse_status = 'failed_retryable',
                        parse_error_code = 'EXCLUSION_REQUESTED',
                        parse_error_message = '选中文件请求人工排除'
                    WHERE id IN (SELECT file_id FROM parse_tasks WHERE id IN ({active_placeholders}))
                      AND parse_status = 'processing'
                    """,
                    tuple(sorted(real_active_task_ids)),
                )
            active_file_ids = {
                int(row["file_id"])
                for row in con.execute(
                    f"SELECT DISTINCT file_id FROM parse_tasks WHERE id IN ({','.join('?' for _ in real_active_task_ids)})",
                    tuple(sorted(real_active_task_ids)),
                ).fetchall()
            } if real_active_task_ids else set()
            inserted_ids: list[int] = []
            report("recording_exclusions", "正在记录排除范围")
            for row in rows:
                throw_if_cancelled()
                if row["active_exclusion_id"] is not None:
                    continue
                parse_status = str(row["parse_status"] or "")
                if int(row["id"]) in active_file_ids and parse_status == "processing":
                    parse_status = "failed_retryable"
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
            result_count = len(inserted_ids)
        try:
            self.rebuild_short_tokens()
            self.rebuild_index_counters()
        except sqlite3.Error as exc:
            if not _is_database_busy(exc):
                logger.warning("派生索引在人工排除后未能立即发布", exc_info=True)
        return result_count

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
                LEFT JOIN index_runs ir ON ir.id = pt.run_id
                WHERE f.is_deleted = 0
                  AND pt.status NOT IN ({terminal_placeholders})
                  AND ir.status = 'running'
                  AND (
                      pt.worker_pid IS NOT NULL
                      OR julianday(COALESCE(
                          pt.last_semantic_progress_at,
                          pt.started_at,
                          pt.created_at
                      )) >= julianday('now', '-120 seconds')
                  )
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

        # Manual exclusion already removes the affected FTS rows above.  A
        # full rebuild is only needed when parsing explicitly deferred FTS
        # writes; otherwise rebuilding a multi-gigabyte projection for a
        # single file (or zero files during state repair) blocks search for a
        # long time.
        fts_dirty_row = con.execute(
            "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
        ).fetchone()
        fts_rebuild_required = (
            fts_dirty_row is not None and str(fts_dirty_row["value"]) == "1"
        )
        if fts_rebuild_required:
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
        else:
            logger.info(
                "Skipping full FTS rebuild during scope publication: projection is clean"
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
        self._set_index_state(con, "short_tokens_dirty", "1")
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

    def cached_full_hashes(
        self,
        file_ids: Sequence[int],
    ) -> dict[int, tuple[str | None, str | None]]:
        """Return persisted full hashes together with their source fingerprint."""

        ids = list(dict.fromkeys(int(file_id) for file_id in file_ids))
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self.connect() as con:
            rows = con.execute(
                f"""
                SELECT id, quick_fingerprint, content_hash_full
                FROM files
                WHERE id IN ({placeholders}) AND content_hash_full IS NOT NULL
                """,
                tuple(ids),
            ).fetchall()
        return {
            int(row["id"]): (
                str(row["quick_fingerprint"] or ""),
                str(row["content_hash_full"] or ""),
            )
            for row in rows
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
        limit: int | None = 500,
        *,
        include_history: bool = False,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        history_filter = (
            ""
            if include_history
            else "AND e.revoked_at IS NULL AND e.invalidated_at IS NULL"
        )
        limit_clause = "LIMIT -1" if limit is None else "LIMIT ?"
        parameters: tuple[object, ...] = () if limit is None else (
            max(1, int(limit)),
        )
        with self.connect_read_only() as con:
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
                    {limit_clause} OFFSET ?
                    """,
                    (*parameters, max(0, int(offset))),
                ).fetchall()
            )

    def metadata_only_files(
        self,
        limit: int | None = 500,
        *,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        limit_clause = "LIMIT -1" if limit is None else "LIMIT ?"
        parameters: tuple[object, ...] = () if limit is None else (
            max(1, int(limit)),
        )
        with self.connect_read_only() as con:
            return list(
                con.execute(
                    f"""
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
                    {limit_clause} OFFSET ?
                    """,
                    (*parameters, max(0, int(offset))),
                ).fetchall()
            )

    def failed_files(
        self,
        limit: int | None = 500,
        *,
        offset: int = 0,
    ) -> list[sqlite3.Row]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        display_statuses = tuple(
            sorted(FINAL_FAILURE_STATUSES | {"pending", "processing", "cancelled"})
        )
        failure_placeholders = ",".join("?" for _ in display_statuses)
        limit_clause = "LIMIT -1" if limit is None else "LIMIT ?"
        parameters: tuple[object, ...] = () if limit is None else (
            max(1, int(limit)),
        )
        with self.connect_read_only() as con:
            return list(
                con.execute(
                    f"""
                    SELECT f.id, f.path, f.filename, f.extension, f.parse_status,
                           f.parse_error_code, f.parse_error_message,
                           f.parse_diagnostics_json,
                           f.parser_name, f.indexed_at,
                           'included' AS scope_state,
                           CASE
                               WHEN f.parse_status IN ('pending', 'processing', 'cancelled')
                                   THEN 'in_progress'
                               ELSE 'failed'
                           END AS failure_state,
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
                    WHERE f.parse_status IN ({failure_placeholders})
                      AND f.is_deleted = 0
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
                    {limit_clause} OFFSET ?
                    """,
                    (
                        *display_statuses,
                        *sorted(VIDEO_EXTENSIONS),
                        *parameters,
                        max(0, int(offset)),
                    ),
                ).fetchall()
            )

    def failed_file_counts(self) -> dict[str, int]:
        """Return true totals for failed and in-progress index scopes.

        The UI deliberately fetches only one page of rows; these counts keep
        the tabs and progress text accurate for folders containing thousands of
        failures.
        """

        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        statuses = tuple(sorted(FINAL_FAILURE_STATUSES))
        in_progress_statuses = ("pending", "processing", "cancelled")
        status_placeholders = ",".join("?" for _ in statuses)
        in_progress_placeholders = ",".join("?" for _ in in_progress_statuses)
        with self.connect_read_only() as con:
            blocking = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
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
                    """,
                    (*statuses, *sorted(VIDEO_EXTENSIONS)),
                ).fetchone()["n"]
            )
            in_progress = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) AS n FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({in_progress_placeholders})
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
                    """,
                    (*in_progress_statuses, *sorted(VIDEO_EXTENSIONS)),
                ).fetchone()["n"]
            )
            excluded = int(
                con.execute(
                    """
                    SELECT COUNT(*) AS n
                    FROM index_scope_exclusions e
                    JOIN files f ON f.id = e.file_id
                    WHERE f.is_deleted = 0
                      AND e.revoked_at IS NULL
                      AND e.invalidated_at IS NULL
                    """
                ).fetchone()["n"]
            )
            metadata = int(
                con.execute(
                    """
                    SELECT COUNT(*) AS n FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status = 'metadata_only'
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions e
                          WHERE e.file_id = f.id
                            AND e.revoked_at IS NULL
                            AND e.invalidated_at IS NULL
                      )
                    """
                ).fetchone()["n"]
            )
        return {
            "blocking": blocking,
            "in_progress": in_progress,
            "excluded": excluded,
            "metadata": metadata,
        }

    def failed_file_ids(
        self,
        *,
        status_filter: str = "",
        extension_filter: str = "",
    ) -> list[int]:
        """Return every current blocker ID without loading row payloads."""

        statuses = tuple(
            sorted(FINAL_FAILURE_STATUSES | {"pending", "processing", "cancelled"})
        )
        status_placeholders = ",".join("?" for _ in statuses)
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        predicates = [
            f"f.parse_status IN ({status_placeholders})",
            "f.is_deleted = 0",
            f"NOT (f.extension IN ({video_placeholders}) AND f.parse_status = 'metadata_only')",
            "NOT EXISTS (SELECT 1 FROM index_scope_exclusions e WHERE e.file_id = f.id AND e.revoked_at IS NULL AND e.invalidated_at IS NULL)",
        ]
        params: list[object] = [*statuses, *sorted(VIDEO_EXTENSIONS)]
        if status_filter:
            predicates.append("f.parse_status = ?")
            params.append(str(status_filter))
        if extension_filter:
            predicates.append("COALESCE(f.extension, '') = ?")
            params.append(str(extension_filter))
        with self.connect_read_only() as con:
            rows = con.execute(
                f"""
                SELECT f.id
                FROM files f
                WHERE {' AND '.join(predicates)}
                ORDER BY f.id
                """,
                tuple(params),
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def failed_snapshot(
        self,
        *,
        page_size: int = 100,
        offset: int = 0,
    ) -> dict[str, object]:
        """Read failure totals, one page and all IDs in one read transaction."""

        statuses = tuple(sorted(FINAL_FAILURE_STATUSES))
        in_progress_statuses = ("pending", "processing", "cancelled")
        status_placeholders = ",".join("?" for _ in statuses)
        in_progress_placeholders = ",".join("?" for _ in in_progress_statuses)
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        status_args = (*statuses, *sorted(VIDEO_EXTENSIONS))
        in_progress_args = (*in_progress_statuses, *sorted(VIDEO_EXTENSIONS))
        limit = max(1, int(page_size))
        page_offset = max(0, int(offset))
        scope_exclusion = """
            NOT EXISTS (
                SELECT 1 FROM index_scope_exclusions e
                WHERE e.file_id = f.id
                  AND e.revoked_at IS NULL
                  AND e.invalidated_at IS NULL
            )
        """
        with self.connect_read_only(
            timeout_seconds=2.0,
            operation="failed_snapshot",
        ) as con:
            con.execute("BEGIN")
            blocking = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (f.extension IN ({video_placeholders})
                               AND f.parse_status = 'metadata_only')
                      AND {scope_exclusion}
                    """,
                    status_args,
                ).fetchone()[0]
            )
            in_progress = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({in_progress_placeholders})
                      AND NOT (f.extension IN ({video_placeholders})
                               AND f.parse_status = 'metadata_only')
                      AND {scope_exclusion}
                    """,
                    in_progress_args,
                ).fetchone()[0]
            )
            excluded = int(
                con.execute(
                    """
                    SELECT COUNT(*) FROM index_scope_exclusions e
                    JOIN files f ON f.id = e.file_id
                    WHERE f.is_deleted = 0
                      AND e.revoked_at IS NULL
                      AND e.invalidated_at IS NULL
                    """
                ).fetchone()[0]
            )
            metadata = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status = 'metadata_only'
                      AND {scope_exclusion}
                    """
                ).fetchone()[0]
            )
            rows = [
                dict(row)
                for row in con.execute(
                    f"""
                    SELECT f.id, f.path, f.filename, f.extension, f.parse_status,
                           f.parse_error_code, f.parse_error_message,
                           f.parse_diagnostics_json, f.parser_name, f.indexed_at,
                           'included' AS scope_state,
                           COALESCE(pt.progress_phase, '') AS progress_phase,
                           COALESCE(pt.progress_cursor, '') AS progress_cursor,
                           CASE
                               WHEN f.parse_error_code = 'PARSE_NO_PROGRESS'
                                   THEN '检查文件是否损坏或位于不稳定存储；修复后重新尝试'
                               WHEN f.parse_error_code = 'PROCESS_WORKER_CRASH'
                                   THEN '关闭占用该文件的程序后重试；若重复发生请保留诊断日志'
                               ELSE '根据错误原因修复源文件或依赖后重新尝试'
                           END AS recovery_advice
                    FROM files f
                    LEFT JOIN parse_tasks pt ON pt.id = (
                        SELECT latest.id FROM parse_tasks latest
                        WHERE latest.file_id = f.id
                        ORDER BY latest.id DESC LIMIT 1
                    )
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (f.extension IN ({video_placeholders})
                               AND f.parse_status = 'metadata_only')
                      AND {scope_exclusion}
                    ORDER BY f.indexed_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (*status_args, limit, page_offset),
                ).fetchall()
            ]
            all_ids = [
                int(row[0])
                for row in con.execute(
                    f"""
                    SELECT f.id FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (f.extension IN ({video_placeholders})
                               AND f.parse_status = 'metadata_only')
                      AND {scope_exclusion}
                    ORDER BY f.id
                    """,
                    status_args,
                ).fetchall()
            ]
        return {
            "failed_counts": {
                "blocking": blocking,
                "in_progress": in_progress,
                "excluded": excluded,
                "metadata": metadata,
            },
            "failed_rows": rows,
            "all_blocking_ids": all_ids,
        }

    def force_complete_current_scope(
        self,
        file_ids: Sequence[int] | None = None,
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
            selected_ids = sorted({int(value) for value in (file_ids or ())})
            selection_clause = ""
            selection_params: tuple[object, ...] = ()
            if selected_ids:
                selection_clause = "AND f.id IN (" + ",".join("?" for _ in selected_ids) + ")"
                selection_params = tuple(selected_ids)
            rows = con.execute(
                f"""
                SELECT f.*
                FROM files f
                WHERE f.is_deleted = 0
                  AND f.extension NOT IN ({video_placeholders})
                  AND f.parse_status NOT IN ('success', 'metadata_only')
                  {selection_clause}
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                  )
                ORDER BY f.id
                """,
                (*video_extensions, *selection_params),
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
            fts_dirty_row = con.execute(
                "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
            ).fetchone()
            fts_rebuild_required = (
                fts_dirty_row is not None and str(fts_dirty_row["value"]) == "1"
            )
            if fts_rebuild_required:
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
            else:
                logger.info(
                    "Skipping full FTS rebuild during force completion: projection is clean"
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
            self._set_index_state(con, "short_tokens_dirty", "1")
        result = {
            "excluded_files": len(file_ids),
            "cancelled_tasks": cancelled_tasks,
            "invalidated_tasks": int(reconciled["invalidated_tasks"]),
            "discarded_candidates": discarded_candidates,
            "ready_roots": ready_roots,
        }
        try:
            self.rebuild_short_tokens()
            self.rebuild_index_counters()
        except sqlite3.Error as exc:
            if not _is_database_busy(exc):
                logger.warning("派生索引在强力完成后未能立即发布", exc_info=True)
        return result

    def blocking_file_status_counts(self) -> dict[str, int]:
        """Return the exact parse-status breakdown behind blocking_files."""

        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        with self.connect_read_only() as con:
            rows = con.execute(
                f"""
                SELECT COALESCE(f.parse_status, '') AS parse_status, COUNT(*) AS n
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
                GROUP BY COALESCE(f.parse_status, '')
                ORDER BY n DESC, parse_status
                """,
                video_extensions,
            ).fetchall()
        return {str(row["parse_status"]): int(row["n"]) for row in rows}

    def blocking_file_diagnostics(
        self,
        *,
        limit: int | None = 25,
    ) -> list[dict[str, object]]:
        """Return actionable details for files preventing publication.

        ``limit=None`` is reserved for the candidate failure report, which
        must retain every blocker so the UI can select and repair all rows.
        Normal readiness probes continue to use a small sample.
        """

        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        limit_clause = "LIMIT -1" if limit is None else "LIMIT ?"
        limit_parameters: tuple[object, ...] = (
            () if limit is None else (max(1, int(limit)),)
        )
        with self.connect_read_only() as con:
            rows = con.execute(
                f"""
                SELECT f.id, f.root_id, COALESCE(r.path, '') AS root_path,
                       f.path, f.filename, f.extension, f.parse_status,
                       f.parse_error_code, f.parse_error_message,
                       f.parser_name, f.parser_version, f.indexed_at,
                       pt.id AS task_id, pt.task_type, pt.status AS task_status,
                       COALESCE(pt.progress_phase, '') AS progress_phase,
                       COALESCE(pt.progress_cursor, '') AS progress_cursor,
                       COALESCE(pt.progress_completed, 0) AS progress_completed,
                       COALESCE(pt.progress_total, 0) AS progress_total
                FROM files f
                LEFT JOIN roots r ON r.id = f.root_id
                LEFT JOIN parse_tasks pt ON pt.id = (
                    SELECT latest.id FROM parse_tasks latest
                    WHERE latest.file_id = f.id
                    ORDER BY latest.id DESC LIMIT 1
                )
                WHERE f.is_deleted = 0
                  AND f.extension NOT IN ({video_placeholders})
                  AND f.parse_status NOT IN ('success', 'metadata_only')
                  AND NOT EXISTS (
                      SELECT 1 FROM index_scope_exclusions e
                      WHERE e.file_id = f.id
                        AND e.revoked_at IS NULL
                        AND e.invalidated_at IS NULL
                   )
                 ORDER BY f.root_id, f.id
                 {limit_clause}
                 """,
                 (*video_extensions, *limit_parameters),
             ).fetchall()
        return [dict(row) for row in rows]

    def unready_root_diagnostics(self, *, limit: int = 25) -> list[dict[str, object]]:
        """Return enabled roots whose candidate status is not ready."""

        with self.connect_read_only() as con:
            rows = con.execute(
                """
                SELECT id, path, enabled, status, last_scan_at, updated_at
                FROM roots
                WHERE enabled = 1 AND COALESCE(status, '') != 'ready'
                ORDER BY id
                LIMIT ?
                """,
                (max(1, min(100, int(limit))),),
            ).fetchall()
        return [dict(row) for row in rows]

    def stats(self) -> dict[str, int]:
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        with self.connect_read_only() as con:
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

    def index_readiness(self, *, reconcile: bool = True) -> dict[str, object]:
        # Collapse old-batch and dead-worker leftovers before calculating the
        # user-facing readiness gate. A stale row must not keep exclusion or
        # search state blocked forever.
        if reconcile:
            try:
                self.reconcile_stale_index_runs()
                self.reconcile_stale_parse_tasks()
                self.reconcile_orphan_pending_files()
            except sqlite3.Error:
                logger.debug("Unable to reconcile stale parse tasks", exc_info=True)
        stats = self.stats()
        blocking_status_counts = (
            self.blocking_file_status_counts()
            if int(stats.get("blocking_files") or 0) > 0
            else {}
        )
        blocking_file_samples = (
            self.blocking_file_diagnostics(limit=25)
            if int(stats.get("blocking_files") or 0) > 0
            else []
        )
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        terminal_placeholders = ",".join(
            "?" for _ in TERMINAL_PARSE_TASK_STATUSES
        )
        with self.connect_read_only() as con:
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
            orphan_pending_files = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) AS n
                    FROM files f
                    JOIN roots r ON r.id = f.root_id
                    WHERE r.enabled = 1
                      AND f.is_deleted = 0
                      AND f.extension NOT IN ({video_placeholders})
                      AND (
                          f.parse_status IN ('pending', 'processing', 'cancelled')
                          OR (
                              f.parse_status = 'failed_retryable'
                              AND f.parse_error_code IN (
                                  'STALE_PARSE_TASK_RECOVERED',
                                  'STALE_INDEX_RUN_RECOVERED',
                                  'INDEX_RUN_ENDED_WITHOUT_TERMINAL_RESULT'
                              )
                          )
                      )
                      AND NOT EXISTS (
                          SELECT 1
                          FROM parse_tasks pt
                          WHERE pt.file_id = f.id
                            AND pt.status NOT IN ({terminal_placeholders})
                      )
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
            active_index_versions = int(
                con.execute(
                    "SELECT COUNT(*) AS n FROM index_versions WHERE status = 'active'"
                ).fetchone()["n"]
            )
            state_rows = {
                str(row["key"]): str(row["value"])
                for row in con.execute(
                    """
                    SELECT key, value FROM index_state
                    WHERE key IN (
                        'content_fts_dirty', 'full_batch_incomplete',
                        'search_projection_generation',
                        'search_projection_updated_at',
                        'partial_search_projection_active',
                        'content_fts_active_generation',
                        'short_tokens_dirty',
                        'short_tokens_active_generation',
                        'short_tokens_coverage_block_count',
                        'search_projection_block_count',
                        'readiness_counters_generation'
                    )
                    """
                )
            }
        content_fts_dirty = state_rows.get("content_fts_dirty") == "1"
        full_batch_incomplete = state_rows.get("full_batch_incomplete") == "1"
        partial_search_projection_active = (
            state_rows.get("partial_search_projection_active") == "1"
        )
        content_fts_dirty = content_fts_dirty or (
            state_rows.get("content_fts_active_generation", "0") == "0"
            and int(stats["blocks"]) > 0
        )
        # The dedicated Chinese two-character index was removed. Two-character
        # queries use the bounded compatibility scan, so it is not a readiness
        # gate for the published full-text projection.
        short_tokens_dirty = False
        try:
            short_tokens_active_generation = int(
                state_rows.get("short_tokens_active_generation", "0")
            )
            short_tokens_coverage = int(
                state_rows.get("short_tokens_coverage_block_count", "0")
            )
            projection_block_count = int(
                state_rows.get("search_projection_block_count", "0")
            )
            content_fts_active_generation = int(
                state_rows.get("content_fts_active_generation", "0")
            )
        except (TypeError, ValueError):
            short_tokens_active_generation = 0
            short_tokens_coverage = 0
            content_fts_active_generation = 0
            projection_block_count = 0
        short_token_sidecar_ready = False
        projection_ready = bool(
            projection_block_count == 0
            or (
                not content_fts_dirty
                and content_fts_active_generation > 0
            )
        )
        try:
            search_projection_generation = max(
                0,
                int(state_rows.get("search_projection_generation", "0")),
            )
        except (TypeError, ValueError):
            search_projection_generation = 0
        not_ready_reasons: list[str] = []
        if enabled_roots <= 0:
            not_ready_reasons.append("no_enabled_roots")
        if int(stats["blocking_files"]) > 0:
            not_ready_reasons.append("blocking_files")
        if unfinished_tasks > 0:
            not_ready_reasons.append("unfinished_tasks")
        if orphan_pending_files > 0:
            not_ready_reasons.append("pending_without_task")
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
        if not projection_ready:
            not_ready_reasons.append("projection_incomplete")
        ready = (
            enabled_roots > 0
            and int(stats["blocking_files"]) == 0
            and unfinished_tasks == 0
            and orphan_pending_files == 0
            and active_runs == 0
            and unready_roots == 0
            and not full_batch_incomplete
            and not content_fts_dirty
            and unpublished_candidates == 0
            and projection_ready
        )
        search_available = ready
        background_task_state = (
            "running"
            if (int(unfinished_tasks) > 0 or int(orphan_pending_files) > 0) and int(active_runs) > 0
            else "residual"
            if int(unfinished_tasks) > 0 or int(orphan_pending_files) > 0
            else "finishing"
            if int(active_runs) > 0
            else "converged"
        )
        return {
            **stats,
            "enabled_roots": enabled_roots,
            "active_runs": active_runs,
            "unready_roots": unready_roots,
            "unfinished_tasks": unfinished_tasks,
            "orphan_pending_files": orphan_pending_files,
            "background_task_state": background_task_state,
            "content_fts_dirty": content_fts_dirty,
            "full_batch_incomplete": full_batch_incomplete,
            "unpublished_candidates": unpublished_candidates,
            "active_index_versions": active_index_versions,
            "not_ready_reasons": not_ready_reasons,
            "blocking_status_counts": blocking_status_counts,
            "blocking_file_samples": blocking_file_samples,
            "unready_root_samples": self.unready_root_diagnostics(limit=25),
            "repairable": bool(
                enabled_roots > 0 and active_runs == 0 and not ready
            ),
            "search_projection_generation": search_projection_generation,
            "search_projection_updated_at": state_rows.get(
                "search_projection_updated_at",
                "",
            ),
            "content_fts_active_generation": content_fts_active_generation,
            "short_tokens_dirty": short_tokens_dirty,
            "short_tokens_active_generation": short_tokens_active_generation,
            "short_tokens_coverage_block_count": short_tokens_coverage,
            "short_token_sidecar_ready": False,
            "search_projection_block_count": projection_block_count,
            "projection_ready": projection_ready,
            "search_available": search_available,
            "search_is_partial": False,
            "failure_alert_ready": True,
            "partial_search_projection_active": (
                partial_search_projection_active
            ),
            "index_complete": ready,
            "ready": ready,
        }

    def quick_index_readiness(self) -> dict[str, object]:
        """Return a bounded startup search gate without scanning large tables.

        The full counters in :meth:`index_readiness` require several aggregate
        scans and can take noticeable time on a cold, multi-gigabyte database.
        Startup only needs to know whether a committed projection already
        exists so the search controls can become usable while full statistics
        are collected in a background thread.
        """

        try:
            # This is a read-only gate. It must never wait behind the
            # application writer lock: search can use the last committed
            # projection while an index batch is being committed.
            with self.connect_read_only(timeout_seconds=0.75) as con:
                enabled_roots = int(
                    con.execute(
                        "SELECT COUNT(*) FROM roots WHERE enabled = 1"
                    ).fetchone()[0]
                )
                active_runs = int(
                    con.execute(
                        "SELECT COUNT(*) FROM index_runs WHERE status = 'running'"
                    ).fetchone()[0]
                )
                active_index_versions = int(
                    con.execute(
                        "SELECT COUNT(*) FROM index_versions WHERE status = 'active'"
                    ).fetchone()[0]
                )
                state_rows = {
                    str(row["key"]): str(row["value"])
                    for row in con.execute(
                        """
                        SELECT key, value FROM index_state
                        WHERE key IN (
                            'content_fts_dirty', 'full_batch_incomplete',
                            'search_projection_generation',
                            'search_projection_updated_at',
                            'partial_search_projection_active',
                            'content_fts_active_generation',
                            'short_tokens_dirty',
                            'short_tokens_active_generation',
                            'short_tokens_coverage_block_count',
                            'readiness_counters_generation'
                        )
                        """
                    )
                }
                has_blocks = bool(
                    con.execute(
                        "SELECT 1 FROM content_blocks LIMIT 1"
                    ).fetchone()
                )
        except sqlite3.OperationalError as exc:
            if _is_database_busy(exc):
                if self._quick_readiness_cache is not None:
                    cached = dict(self._quick_readiness_cache)
                    cached["readiness_stale"] = True
                    cached["readiness_degraded"] = True
                    return cached
                # A rollback-journal database can briefly reject even a
                # read-only probe while a writer is committing.  Do not turn
                # that short window into a foreground search failure.  The
                # actual search uses its own read-only connection and will
                # either see the last committed projection or report a real
                # query error through the worker without a modal dialog.
                if self.db_path.is_file():
                    return {
                        "enabled_roots": 0,
                        "active_runs": 1,
                        "active_index_versions": 0,
                        "content_fts_dirty": False,
                        "search_projection_generation": 0,
                        "search_projection_updated_at": "",
                        "partial_search_projection_active": True,
                        "search_available": False,
                        "search_is_partial": False,
                        "index_complete": False,
                        "ready": False,
                        "startup_snapshot_pending": True,
                        "readiness_degraded": True,
                        "quality": "quick",
                        "stats_complete": False,
                        "complete_files": None,
                        "eligible_files": None,
                        "blocking_files": None,
                        "video_excluded": None,
                        "manual_excluded_files": None,
                    }
            raise
        try:
            generation = max(
                0,
                int(state_rows.get("search_projection_generation", "0")),
            )
        except (TypeError, ValueError):
            generation = 0
        content_fts_dirty = state_rows.get("content_fts_dirty") == "1"
        partial_active = (
            state_rows.get("partial_search_projection_active") == "1"
        )
        # Quick readiness never asserts that counters or all projections are
        # complete. The detail snapshot/search gate is the only source of a
        # positive ordinary-search decision.
        search_available = False
        result = {
            "quality": "quick",
            "stats_complete": False,
            "enabled_roots": enabled_roots,
            "active_runs": active_runs,
            "active_index_versions": active_index_versions,
            "content_fts_dirty": content_fts_dirty,
            "full_batch_incomplete": state_rows.get("full_batch_incomplete") == "1",
            "search_projection_generation": generation,
            "search_projection_updated_at": state_rows.get(
                "search_projection_updated_at",
                "",
            ),
            "partial_search_projection_active": partial_active,
            "search_available": search_available,
            "search_is_partial": False,
            "index_complete": False,
            "ready": False,
            "startup_snapshot_pending": True,
            "complete_files": None,
            "eligible_files": None,
            "blocking_files": None,
            "video_excluded": None,
            "manual_excluded_files": None,
        }
        self._quick_readiness_cache = dict(result)
        return result

    def search_projection_readiness(self) -> dict[str, object]:
        """Return the bounded, strict gate used before an ordinary search."""

        terminal = tuple(sorted(TERMINAL_PARSE_TASK_STATUSES))
        placeholders = ",".join("?" for _ in terminal)
        with self.connect_read_only(timeout_seconds=1.0) as con:
            roots = con.execute(
                "SELECT id, status FROM roots WHERE enabled = 1"
            ).fetchall()
            root_ids = {int(row["id"]) for row in roots}
            counters = {
                int(row["root_id"]): dict(row)
                for row in con.execute(
                    "SELECT * FROM index_counters WHERE root_id IN "
                    f"({','.join('?' for _ in root_ids)})"
                    if root_ids
                    else "SELECT * FROM index_counters WHERE 0",
                    tuple(sorted(root_ids)),
                ).fetchall()
            }
            active_runs = int(
                con.execute(
                    "SELECT COUNT(*) FROM index_runs WHERE status = 'running'"
                ).fetchone()[0]
            )
            unfinished = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM parse_tasks pt
                    JOIN files f ON f.id = pt.file_id
                    JOIN roots r ON r.id = f.root_id
                    WHERE r.enabled = 1 AND f.is_deleted = 0
                      AND pt.status NOT IN ({placeholders})
                    """,
                    terminal,
                ).fetchone()[0]
            )
            staging = int(
                con.execute(
                    "SELECT COUNT(*) FROM index_versions WHERE status = 'staging'"
                ).fetchone()[0]
            )
            state = {
                str(row["key"]): str(row["value"])
                for row in con.execute(
                    """
                    SELECT key, value FROM index_state
                    WHERE key IN (
                        'content_fts_dirty', 'full_batch_incomplete',
                        'content_fts_active_generation',
                        'short_tokens_dirty', 'short_tokens_active_generation',
                        'short_tokens_coverage_block_count',
                        'search_projection_block_count',
                        'search_projection_generation',
                        'readiness_counters_generation'
                    )
                    """
                ).fetchall()
            }
        sums = {
            key: sum(int(row.get(key) or 0) for row in counters.values())
            for key in (
                "file_count",
                "eligible_files",
                "complete_files",
                "blocking_files",
                "metadata_only_files",
                "video_excluded",
                "manual_excluded",
                "block_count",
                "document_count",
            )
        }
        try:
            content_generation = int(state.get("content_fts_active_generation", "0"))
            projection_block_count = int(state.get("search_projection_block_count", "0"))
            counter_generation = int(state.get("readiness_counters_generation", "0"))
        except (TypeError, ValueError):
            content_generation = counter_generation = 0
            projection_block_count = 0
        counters_ready = bool(
            root_ids
            and root_ids == set(counters)
            and counter_generation > 0
            and all(int(row.get("generation") or 0) > 0 for row in counters.values())
        )
        short_token_sidecar_ready = False
        projection_ready = bool(
            projection_block_count == 0
            or (
                state.get("content_fts_dirty") != "1"
                and content_generation > 0
            )
        )
        ready = bool(
            root_ids
            and counters_ready
            and all(str(row["status"] or "") == "ready" for row in roots)
            and active_runs == 0
            and unfinished == 0
            and staging == 0
            and state.get("content_fts_dirty") != "1"
            and state.get("full_batch_incomplete") != "1"
            and projection_ready
        )
        return {
            "quality": "complete" if counters_ready else "degraded",
            "stats_complete": counters_ready,
            **sums,
            "enabled_roots": len(root_ids),
            "active_runs": active_runs,
            "unfinished_tasks": unfinished,
            "unpublished_candidates": staging,
            "content_fts_dirty": state.get("content_fts_dirty") == "1",
            "full_batch_incomplete": state.get("full_batch_incomplete") == "1",
            "content_fts_active_generation": content_generation,
            "short_tokens_dirty": False,
            "short_tokens_active_generation": 0,
            "short_tokens_coverage_block_count": 0,
            "short_token_sidecar_ready": short_token_sidecar_ready,
            "search_projection_block_count": projection_block_count,
            "projection_ready": projection_ready,
            "search_available": ready,
            "search_is_partial": False,
            "index_complete": ready,
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
        with self.connect_read_only() as con:
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
    if isinstance(exc, SQLiteBusyCircuitOpen):
        return True
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
