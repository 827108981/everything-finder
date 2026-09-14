from __future__ import annotations

import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
import unittest
from pathlib import Path

import pytest

from local_full_text_search.core.database import (
    SCHEMA_VERSION,
    DatabaseManager,
    SQLiteBusyCircuit,
    SQLiteBusyCircuitOpen,
)
from local_full_text_search.core.planning_tasks import PreparedFileMetadata
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.models.search_query import SearchQuery


class DatabaseIndexSearchTests(unittest.TestCase):
    def test_no_worker_task_becomes_stale_after_short_grace_period(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "stuck.txt"
            source.write_text("stuck", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _changed = db.upsert_file_metadata(root_id, source)
            old = (datetime.now(timezone.utc) - timedelta(seconds=61)).isoformat()
            with db.connect() as con:
                con.execute(
                    """
                    INSERT INTO index_runs(id, started_at, mode, status)
                    VALUES ('stuck-run', ?, 'incremental', 'running')
                    """,
                    (old,),
                )
                con.execute(
                    """
                    INSERT INTO parse_tasks(
                        file_id, run_id, task_type, status, created_at,
                        started_at, progress_phase, progress_completed, progress_total
                    ) VALUES (?, 'stuck-run', 'normal', 'running', ?, ?, '', 0, 0)
                    """,
                    (file_id, old, old),
                )

            recovered = db.reconcile_stale_parse_tasks(stale_after_seconds=120)

            self.assertEqual(recovered, 1)
            with db.connect_read_only() as con:
                task = con.execute(
                    "SELECT status, error_code FROM parse_tasks WHERE file_id = ?",
                    (file_id,),
                ).fetchone()
                file_row = con.execute(
                    "SELECT parse_status, parse_error_code FROM files WHERE id = ?",
                    (file_id,),
                ).fetchone()
            self.assertEqual(task["status"], "failed_retryable")
            self.assertEqual(task["error_code"], "STALE_PARSE_TASK_RECOVERED")
            self.assertEqual(file_row["parse_status"], "failed_retryable")

    def test_active_run_tasks_are_not_recovered_by_mid_run_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "queued.txt"
            source.write_text("queued", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _changed = db.upsert_file_metadata(root_id, source)
            old = (datetime.now(timezone.utc) - timedelta(seconds=600)).isoformat()
            with db.connect() as con:
                con.execute(
                    """
                    INSERT INTO index_runs(id, started_at, heartbeat_at, mode, status)
                    VALUES ('active-run', ?, ?, 'incremental', 'running')
                    """,
                    (old, old),
                )
                con.execute(
                    """
                    INSERT INTO parse_tasks(
                        file_id, run_id, task_type, status, created_at, queued_at
                    ) VALUES (?, 'active-run', 'normal', 'queued', ?, ?)
                    """,
                    (file_id, old, old),
                )

            assert db.reconcile_stale_parse_tasks(
                stale_after_seconds=120,
                no_worker_after_seconds=60,
                active_run_id="active-run",
            ) == 0
            with db.connect_read_only() as con:
                task = con.execute(
                    "SELECT status FROM parse_tasks WHERE file_id = ?",
                    (file_id,),
                ).fetchone()
            assert task["status"] == "queued"

            assert db.reconcile_stale_parse_tasks(
                stale_after_seconds=120,
                no_worker_after_seconds=60,
                active_run_id="another-run",
            ) == 1

    def test_readiness_exposes_blocking_file_status_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "blocked.txt"
            source.write_text("blocked", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _changed = db.upsert_file_metadata(root_id, source)
            with db.connect() as con:
                con.execute(
                    "UPDATE files SET parse_status = 'partial_success', "
                    "parse_error_code = 'PARTIAL_TEST' WHERE id = ?",
                    (file_id,),
                )
                con.execute(
                    "UPDATE roots SET status = 'incomplete' WHERE id = ?",
                    (root_id,),
                )

            readiness = db.index_readiness(reconcile=False)

            assert readiness["blocking_files"] == 1
            assert readiness["blocking_status_counts"] == {"partial_success": 1}
            assert readiness["blocking_file_samples"][0]["path"] == str(source)
            assert readiness["blocking_file_samples"][0]["parse_error_code"] == "PARTIAL_TEST"
            assert readiness["unready_root_samples"][0]["path"] == str(root)

    def test_pid_probe_treats_windows_system_error_as_dead_worker(self) -> None:
        original_kill = os.kill

        def raise_system_error(_pid: int, _signal: int) -> None:
            raise SystemError("invalid process handle")

        try:
            os.kill = raise_system_error  # type: ignore[assignment]
            self.assertFalse(DatabaseManager._pid_is_alive(12345))
        finally:
            os.kill = original_kill  # type: ignore[assignment]

    def test_stale_index_run_requires_dead_owner_and_no_active_task(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "stale.txt"
            source.write_text("stale", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _changed = db.upsert_file_metadata(root_id, source)
            old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            with db.connect() as con:
                con.execute(
                    """
                    INSERT INTO index_runs(
                        id, started_at, mode, status, owner_pid, heartbeat_at
                    ) VALUES ('live-owner-run', ?, 'incremental', 'running', ?, ?)
                    """,
                    (old, os.getpid(), old),
                )
                con.execute(
                    """
                    INSERT INTO index_runs(
                        id, started_at, mode, status, owner_pid, heartbeat_at
                    ) VALUES ('dead-owner-run', ?, 'incremental', 'running', 999999, ?)
                    """,
                    (old, old),
                )
                con.execute(
                    """
                    INSERT INTO parse_tasks(
                        file_id, run_id, task_type, status, created_at, started_at
                    ) VALUES (?, 'dead-owner-run', 'text', 'queued', ?, ?)
                    """,
                    (file_id, old, old),
                )

            assert db.reconcile_stale_index_runs(stale_after_seconds=1) == 1
            with db.connect_read_only() as con:
                runs = {
                    str(row["id"]): str(row["status"])
                    for row in con.execute(
                        "SELECT id, status FROM index_runs ORDER BY id"
                    )
                }
                task = con.execute(
                    "SELECT status FROM parse_tasks WHERE run_id = 'dead-owner-run'"
                ).fetchone()
            self.assertEqual(runs["live-owner-run"], "running")
            self.assertEqual(runs["dead-owner-run"], "interrupted")
            self.assertEqual(task["status"], "failed_retryable")

    def test_startup_requeues_ocr_owned_by_a_stale_index_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "stale-ocr.png"
            source.write_bytes(b"image")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _changed = db.upsert_file_metadata(root_id, source)
            old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            with db.connect() as con:
                con.execute(
                    """
                    INSERT INTO index_runs(
                        id, started_at, mode, status, owner_pid, heartbeat_at
                    ) VALUES ('stale-ocr-run', ?, 'incremental', 'running', 999999, ?)
                    """,
                    (old, old),
                )
                task_id = int(
                    con.execute(
                        """
                        INSERT INTO parse_tasks(
                            file_id, run_id, task_type, status, created_at, started_at
                        ) VALUES (?, 'stale-ocr-run', 'ocr', 'running', ?, ?)
                        RETURNING id
                        """,
                        (file_id, old, old),
                    ).fetchone()[0]
                )
                con.execute(
                    """
                    INSERT INTO ocr_requests(
                        file_id, parent_task_id, source_kind, source_unit,
                        image_spool_path, content_sha256, width, height,
                        config_fingerprint, priority, pixel_cost, status,
                        lease_owner, lease_expires_at, created_at
                    ) VALUES (?, ?, 'image', 'image', ?, ?, 1, 1,
                              'cfg', 100, 1, 'running', 'dead-worker', ?, ?)
                    """,
                    (file_id, task_id, str(source), "a" * 64, old, old),
                )

            DatabaseManager(base / "index.db").initialize()

            with db.connect_read_only() as con:
                request = con.execute(
                    "SELECT status, lease_owner, lease_expires_at FROM ocr_requests"
                ).fetchone()
                run = con.execute(
                    "SELECT status FROM index_runs WHERE id = 'stale-ocr-run'"
                ).fetchone()
            self.assertEqual(request["status"], "queued")
            self.assertIsNone(request["lease_owner"])
            self.assertIsNone(request["lease_expires_at"])
            self.assertEqual(run["status"], "interrupted")

    def test_orphan_pending_files_wait_for_a_stale_and_run_grace_period(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "pending.txt"
            source.write_text("pending", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, changed = db.upsert_file_metadata(root_id, source)
            self.assertTrue(changed)

            # Fresh metadata is not an orphan failure just because no parser
            # task has been submitted yet.
            self.assertEqual(db.reconcile_orphan_pending_files(), 0)
            with db.connect() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT parse_status FROM files WHERE id = ?", (file_id,)
                    ).fetchone()["parse_status"],
                    "pending",
                )

            old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            with db.connect() as connection:
                connection.execute(
                    "UPDATE files SET last_seen_at = ? WHERE id = ?", (old, file_id)
                )
                connection.execute(
                    """
                    INSERT INTO index_runs(id, started_at, finished_at, mode, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        "recent-run",
                        old,
                        datetime.now(timezone.utc).isoformat(),
                        "normal",
                        "failed",
                    ),
                )

            # A just-finished run still gets a recovery window.
            self.assertEqual(db.reconcile_orphan_pending_files(), 0)
            with db.connect() as connection:
                connection.execute(
                    "UPDATE index_runs SET finished_at = ? WHERE id = ?",
                    (old, "recent-run"),
                )

            self.assertEqual(db.reconcile_orphan_pending_files(), 1)
            with db.connect() as connection:
                row = connection.execute(
                    "SELECT parse_status, parse_error_code FROM files WHERE id = ?",
                    (file_id,),
                ).fetchone()
            self.assertEqual(row["parse_status"], "pending")
            self.assertIsNone(row["parse_error_code"])

    def test_read_only_snapshot_does_not_wait_for_writer_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            db = DatabaseManager(base / "index.db")
            db.initialize()
            db.add_root(root)
            with db._connections_lock:
                pass
            original = db.list_roots
            locker = sqlite3.connect(db.db_path)
            try:
                locker.execute("BEGIN IMMEDIATE")
                started = time.perf_counter()
                # The read-only connection must be independent of the
                # in-process writer reservation.
                rows = db.connect_read_only(timeout_seconds=0.2)
                with rows as connection:
                    self.assertEqual(connection.execute("SELECT COUNT(*) FROM roots").fetchone()[0], 1)
                self.assertLess(time.perf_counter() - started, 0.5)
                del original
                locker.rollback()
            finally:
                locker.close()

    def test_orphan_pending_file_is_replanned_on_next_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "orphan.txt"
            target.write_text("ORPHAN_REPLAN", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, target)
            manager = IndexManager(db, AppSettings(enable_ocr=False))
            summary = manager.index_root(root_id)
            self.assertEqual(summary.indexed, 1)
            with db.connect_read_only() as connection:
                row = connection.execute(
                    "SELECT parse_status FROM files WHERE id = ?", (file_id,)
                ).fetchone()
            self.assertEqual(row["parse_status"], "success")

    def test_schema_v4_database_migrates_before_v5_indexes_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "existing.txt"
            source.write_text("V4_TO_V5_MIGRATION_PRESERVES_CONTENT", encoding="utf-8")
            db_path = base / "index.db"
            db = DatabaseManager(db_path)
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            con = sqlite3.connect(db_path)
            try:
                con.execute("PRAGMA foreign_keys = OFF")
                con.execute("PRAGMA legacy_alter_table = ON")
                for index_name in (
                    "idx_files_container",
                    "idx_files_source_kind",
                    "idx_files_zip_member_source",
                    "idx_files_content_hash_full",
                    "idx_files_exact_content",
                ):
                    con.execute(f"DROP INDEX IF EXISTS {index_name}")
                con.execute("ALTER TABLE files RENAME TO files_v5")
                con.execute(
                    """
                    CREATE TABLE files (
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
                    )
                    """
                )
                legacy_columns = (
                    "id, root_id, path, filename, extension, size_bytes, "
                    "modified_time, created_time, quick_fingerprint, content_hash, "
                    "content_key, document_id, parse_status, parse_error_code, "
                    "parse_error_message, parser_name, parser_version, indexed_at, "
                    "last_seen_at, is_deleted"
                )
                con.execute(
                    f"INSERT INTO files({legacy_columns}) "
                    f"SELECT {legacy_columns} FROM files_v5"
                )
                con.execute("DROP TABLE files_v5")
                con.execute("PRAGMA user_version = 4")
                con.commit()
            finally:
                con.close()

            migrated = DatabaseManager(db_path)
            migrated.initialize()

            backup = base / "index.schema-v4.backup.db"
            self.assertTrue(backup.is_file())
            with migrated.connect() as con:
                columns = {
                    str(row["name"]) for row in con.execute("PRAGMA table_info(files)")
                }
                indexes = {
                    str(row["name"]) for row in con.execute("PRAGMA index_list(files)")
                }
                row = con.execute(
                    "SELECT parse_status FROM files WHERE path = ?",
                    (str(source),),
                ).fetchone()
                self.assertEqual(
                    con.execute("PRAGMA user_version").fetchone()[0],
                    SCHEMA_VERSION,
                )
            self.assertIn("container_file_id", columns)
            self.assertIn("source_kind", columns)
            self.assertIn("idx_files_container", indexes)
            self.assertIn("idx_files_zip_member_source", indexes)
            self.assertEqual(row["parse_status"], "success")
            self.assertEqual(
                SearchEngine(migrated)
                .search(
                    SearchQuery(
                        text="V4_TO_V5_MIGRATION_PRESERVES_CONTENT",
                        mode="exact",
                    )
                )
                .total_confirmed,
                1,
            )

    def test_text_file_index_and_search_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "产品资料.txt").write_text("BS-2800M2 全自动生化分析仪\n校准 吸光度", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.indexed, 1)

            engine = SearchEngine(db)
            exact = engine.search(SearchQuery(text="BS-2800M2", mode="exact"))
            self.assertEqual(exact.total_confirmed, 1)
            all_terms = engine.search(SearchQuery(text="生化 校准 吸光度", mode="all"))
            self.assertEqual(all_terms.total_confirmed, 1)
            any_terms = engine.search(SearchQuery(text="不存在 吸光度", mode="any"))
            self.assertEqual(any_terms.total_confirmed, 1)
            filename = engine.search(SearchQuery(text="产品资料", mode="filename"))
            self.assertEqual(filename.total_confirmed, 1)

    def test_incremental_skip_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "a.txt"
            target.write_text("alpha beta", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings())
            first = manager.index_root(root_id)
            second = manager.index_root(root_id)
            self.assertEqual(first.indexed, 1)
            self.assertGreaterEqual(second.skipped, 1)
            target.unlink()
            deleted = manager.index_root(root_id)
            self.assertEqual(deleted.deleted, 1)
            result = SearchEngine(db).search(SearchQuery(text="alpha", mode="exact"))
            self.assertEqual(result.total_confirmed, 0)

    def test_unsupported_format_is_outside_index_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "raw.bin").write_bytes(b"\x00\x01")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.scanned, 0)
            self.assertEqual(summary.unsupported, 0)
            self.assertEqual(db.failed_files(), [])
            self.assertTrue(db.index_readiness()["ready"])

    def test_mp4_is_excluded_from_parse_completion_and_failures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "培训视频.mp4").write_bytes(b"not a real video")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.excluded_video, 1)
            self.assertEqual(db.failed_files(), [])
            readiness = db.index_readiness()
            self.assertTrue(readiness["ready"])
            self.assertEqual(readiness["eligible_files"], 0)
            self.assertEqual(readiness["video_excluded"], 1)

    def test_search_history_is_deduplicated_and_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "index.db")
            db.initialize()
            db.add_search_history("alpha", max_entries=2)
            db.add_search_history("beta", max_entries=2)
            db.add_search_history("alpha", max_entries=2)

            self.assertEqual(db.search_history(), ["alpha", "beta"])
            db.clear_search_history()
            self.assertEqual(db.search_history(), [])

    def test_failed_file_scopes_are_paginated_with_true_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            paths = []
            for index in range(5):
                path = root / f"failed-{index}.txt"
                path.write_text(f"failure {index}", encoding="utf-8")
                paths.append(path)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_ids = [db.upsert_file_metadata(root_id, path)[0] for path in paths]
            with db.connect() as con:
                con.executemany(
                    "UPDATE files SET parse_status = 'failed' WHERE id = ?",
                    [(file_id,) for file_id in file_ids],
                )

            db.exclude_files_from_index(
                [file_ids[0]],
                reason="pagination test",
            )
            totals = db.failed_file_counts()
            self.assertEqual(totals["blocking"], 4)
            self.assertEqual(totals["excluded"], 1)
            self.assertEqual(
                len(db.failed_files(limit=2, offset=2)),
                2,
            )
            self.assertEqual(
                len(db.excluded_files(limit=1, offset=0)),
                1,
            )

    def test_diagnostic_task_updates_skip_a_busy_database_quickly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "sample.txt"
            target.write_text("database lock regression", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, target)
            task_id = db.create_parse_task(file_id, "lock-test", "normal")

            locker = sqlite3.connect(db.db_path, timeout=0.1)
            try:
                locker.execute("BEGIN IMMEDIATE")
                started = time.perf_counter()
                self.assertFalse(
                    db.try_mark_tasks_running([task_id], timeout_seconds=0.0)
                )
                self.assertFalse(
                    db.try_mark_task_spooled(
                        task_id,
                        base / "result.spool",
                        "checksum",
                        timeout_seconds=0.0,
                    )
                )
                self.assertLess(time.perf_counter() - started, 0.2)
            finally:
                locker.rollback()
                locker.close()

            db.mark_task_running(task_id)
            db.mark_task_spooled(task_id, base / "result.spool", "checksum")
            with db.connect() as con:
                task = con.execute(
                    "SELECT status, spool_checksum FROM parse_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
            self.assertEqual(task["status"], "spooled")
            self.assertEqual(task["spool_checksum"], "checksum")

    def test_valid_spooled_task_survives_run_convergence_and_restart(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "large.xls"
            source.write_bytes(b"legacy excel placeholder")
            database_path = base / "index.db"
            db = DatabaseManager(database_path)
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, source)
            run_id = "run-with-durable-spool"
            with db.connect() as connection:
                connection.execute(
                    """
                    INSERT INTO index_runs(id, started_at, mode, status)
                    VALUES (?, ?, 'normal', 'running')
                    """,
                    (run_id, datetime.now(timezone.utc).isoformat()),
                )
            task_id = db.create_parse_task(file_id, run_id, "legacy_excel")
            spool = base / "large-result.pickle"
            spool.write_bytes(b"durable parse result")
            db.mark_task_running(task_id)
            db.mark_task_spooled(task_id, spool, "valid-checksum")

            with db.connect() as connection:
                converged = db._converge_index_run_tasks(
                    connection,
                    run_id,
                    "interrupted",
                )
                status = connection.execute(
                    "SELECT status FROM parse_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()["status"]
            self.assertEqual(converged, 0)
            self.assertEqual(status, "spooled")

            restarted = DatabaseManager(database_path)
            restarted.initialize()
            recoverable = restarted.recoverable_spooled_tasks(root_id)
            self.assertEqual([int(row["id"]) for row in recoverable], [task_id])

    def test_parse_task_attempt_and_semantic_progress_are_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "manual.pdf"
            target.write_bytes(b"%PDF-test")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, target)
            task_id = db.create_parse_task(file_id, "progress-run", "pdf")

            db.mark_task_running(task_id)
            self.assertTrue(
                db.try_update_task_progress(
                    task_id,
                    phase="pdf_ocr_page",
                    completed=3,
                    total=10,
                    unit_type="page",
                    cursor="3",
                    bytes_read=4096,
                    output_blocks=3,
                    checkpoint_version=3,
                    worker_pid=1234,
                    checkpoint_path=str(base / "checkpoint.pickle"),
                )
            )

            with db.connect() as con:
                task = con.execute(
                    """
                    SELECT progress_phase, progress_completed, progress_total,
                           progress_cursor, worker_pid, checkpoint_path
                    FROM parse_tasks WHERE id = ?
                    """,
                    (task_id,),
                ).fetchone()
                attempt = con.execute(
                    """
                    SELECT attempt_no, status, last_progress_at, worker_pid
                    FROM parse_task_attempts WHERE task_id = ?
                    """,
                    (task_id,),
                ).fetchone()
            self.assertEqual(task["progress_phase"], "pdf_ocr_page")
            self.assertEqual(task["progress_completed"], 3)
            self.assertEqual(task["progress_total"], 10)
            self.assertEqual(task["progress_cursor"], "3")
            self.assertEqual(task["worker_pid"], 1234)
            self.assertTrue(str(task["checkpoint_path"]).endswith("checkpoint.pickle"))
            self.assertEqual(attempt["attempt_no"], 1)
            self.assertEqual(attempt["status"], "running")
            self.assertIsNotNone(attempt["last_progress_at"])
            self.assertEqual(attempt["worker_pid"], 1234)

            self.assertTrue(
                db.try_record_child_task_progress(
                    task_id,
                    task_type="pdf_ocr_page",
                    unit_key="page:3",
                    status="running",
                    phase="pdf_ocr_preview",
                    completed=0,
                    total=1,
                    worker_pid=1234,
                )
            )
            self.assertTrue(
                db.try_record_child_task_progress(
                    task_id,
                    task_type="pdf_ocr_page",
                    unit_key="page:3",
                    status="complete",
                    phase="pdf_ocr_page",
                    completed=1,
                    total=1,
                    worker_pid=1234,
                )
            )
            with db.connect() as con:
                children = con.execute(
                    """
                    SELECT parent_task_id, task_type, unit_key, status
                    FROM parse_tasks WHERE parent_task_id = ?
                    """,
                    (task_id,),
                ).fetchall()
            self.assertEqual(len(children), 1)
            self.assertEqual(children[0]["task_type"], "pdf_ocr_page")
            self.assertEqual(children[0]["unit_key"], "page:3")
            self.assertEqual(children[0]["status"], "complete")

    def test_worker_crash_closes_only_the_active_parse_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "scan.png"
            target.write_bytes(b"image")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, target)
            task_id = db.create_parse_task(file_id, "crash-run", "ocr")

            db.mark_task_running(task_id)
            db.mark_task_attempt_interrupted(
                task_id,
                "PROCESS_WORKER_CRASH",
                "OCR worker exited after models were ready",
            )
            db.mark_task_running(task_id)

            with db.connect() as con:
                attempts = con.execute(
                    """
                    SELECT attempt_no, status, finished_at,
                           error_code, error_message
                    FROM parse_task_attempts
                    WHERE task_id = ? ORDER BY attempt_no
                    """,
                    (task_id,),
                ).fetchall()
            self.assertEqual(len(attempts), 2)
            self.assertEqual(attempts[0]["status"], "interrupted")
            self.assertIsNotNone(attempts[0]["finished_at"])
            self.assertEqual(
                attempts[0]["error_code"],
                "PROCESS_WORKER_CRASH",
            )
            self.assertIn("models were ready", attempts[0]["error_message"])
            self.assertEqual(attempts[1]["status"], "running")
            self.assertIsNone(attempts[1]["finished_at"])

    def test_interrupt_active_connections_stops_a_long_statement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = DatabaseManager(Path(tmp) / "index.db")
            db.initialize()
            query_started = threading.Event()
            errors: list[BaseException] = []

            def run_query() -> None:
                try:
                    with db.connect() as con:
                        query_started.set()
                        con.execute(
                            """
                            WITH RECURSIVE counter(value) AS (
                                VALUES(0)
                                UNION ALL
                                SELECT value + 1 FROM counter WHERE value < 100000000
                            )
                            SELECT SUM(value) FROM counter
                            """
                        ).fetchone()
                except BaseException as exc:
                    errors.append(exc)

            worker = threading.Thread(target=run_query, daemon=True)
            worker.start()
            self.assertTrue(query_started.wait(timeout=1.0))
            time.sleep(0.05)
            started = time.perf_counter()
            self.assertGreaterEqual(db.interrupt_active_connections(), 1)
            worker.join(timeout=1.0)

            self.assertFalse(worker.is_alive())
            self.assertLess(time.perf_counter() - started, 0.5)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], sqlite3.OperationalError)
            self.assertIn("interrupted", str(errors[0]).lower())


if __name__ == "__main__":
    unittest.main()
def test_upsert_precomputed_metadata_does_not_stat_in_database_layer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    file_path = root / "sample.txt"
    file_path.write_text("sample", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    metadata = PreparedFileMetadata(
        path=str(file_path),
        size_bytes=6,
        modified_time=123.0,
        created_time=122.0,
        modified_time_ns=123_000_000_000,
        worker_pid=42,
    )

    prepared, errors = database.upsert_precomputed_file_metadata_many(
        root_id,
        [metadata],
        retry_failed_files=False,
        compute_full_hash=False,
        mark_processing=False,
        parser_versions={str(file_path): "test-v1"},
    )

    assert errors == []
    assert prepared[0][0] == file_path
    assert prepared[0][2] is True


def test_latest_schema_has_task_leases_pdf_ocr_versions_and_scope_tables(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(tmp_path / "schema-latest.db")
    database.initialize()

    with database.connect() as con:
        assert con.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        task_columns = {
            str(row["name"])
            for row in con.execute("PRAGMA table_info(parse_tasks)").fetchall()
        }
        tables = {
            str(row["name"])
            for row in con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {
        "lease_owner",
        "lease_expires_at",
        "confirmed_at",
        "source_digest",
        "task_version",
        "result_digest",
    } <= task_columns
    assert {
        "pdf_page_identities",
        "ocr_requests",
        "ocr_exact_cache",
        "index_versions",
        "resource_events",
        "backend_benchmarks",
        "index_scope_exclusions",
    } <= tables


def test_p0_content_hashes_can_be_persisted_as_one_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")
    database = DatabaseManager(tmp_path / "batch-hash.db")
    database.initialize()
    root_id = database.add_root(root)
    metadata = [
        PreparedFileMetadata(
            path=str(path),
            size_bytes=path.stat().st_size,
            modified_time=path.stat().st_mtime,
            created_time=path.stat().st_ctime,
            modified_time_ns=path.stat().st_mtime_ns,
        )
        for path in (first, second)
    ]
    prepared, errors = database.upsert_precomputed_file_metadata_many(
        root_id,
        metadata,
        retry_failed_files=False,
        compute_full_hash=False,
        mark_processing=False,
        parser_versions={str(first): "test-v1", str(second): "test-v1"},
    )
    assert errors == []
    ids = {str(path): file_id for path, file_id, _changed in prepared}

    written = database.set_content_hash_full_many(
        [(ids[str(first)], "a" * 64), (ids[str(second)], "b" * 64)]
    )

    assert written == 2
    assert database.file_content_states([ids[str(first)], ids[str(second)]]) == {
        ids[str(first)]: ("a" * 64, None),
        ids[str(second)]: ("b" * 64, None),
    }


def test_cached_document_links_are_written_in_bounded_batches(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    source = root / "source.txt"
    duplicate = root / "duplicate.txt"
    source.write_text("same cached content", encoding="utf-8")
    duplicate.write_text("same cached content", encoding="utf-8")
    database = DatabaseManager(tmp_path / "cache-link.db")
    database.initialize()
    root_id = database.add_root(root)
    source_id, _ = database.upsert_file_metadata(root_id, source)
    duplicate_id, _ = database.upsert_file_metadata(root_id, duplicate)
    database.replace_file_blocks(
        source_id,
        source.name,
        source,
        [],
        parser_name="text",
    )
    with database.connect_read_only() as connection:
        source_row = connection.execute(
            "SELECT document_id, content_key, parser_name, parser_version "
            "FROM files WHERE id = ?",
            (source_id,),
        ).fetchone()
    assert source_row is not None
    linked = database.link_cached_documents_batch(
        [
            (
                [duplicate_id],
                int(source_row["document_id"]),
                str(source_row["content_key"]),
                str(source_row["parser_name"]),
                str(source_row["parser_version"]),
                "success",
            )
        ],
        batch_file_ids=1,
    )
    assert linked == 1
    with database.connect_read_only() as connection:
        duplicate_row = connection.execute(
            "SELECT document_id, parse_status FROM files WHERE id = ?",
            (duplicate_id,),
        ).fetchone()
    assert duplicate_row is not None
    assert int(duplicate_row["document_id"]) == int(source_row["document_id"])
    assert str(duplicate_row["parse_status"]) == "success"


def test_cached_document_links_resume_after_a_committed_batch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    source = root / "source.txt"
    first = root / "first.txt"
    second = root / "second.txt"
    source.write_text("same cached content", encoding="utf-8")
    first.write_text("same cached content", encoding="utf-8")
    second.write_text("same cached content", encoding="utf-8")
    database = DatabaseManager(tmp_path / "cache-link-resume.db")
    database.initialize()
    root_id = database.add_root(root)
    source_id, _ = database.upsert_file_metadata(root_id, source)
    first_id, _ = database.upsert_file_metadata(root_id, first)
    second_id, _ = database.upsert_file_metadata(root_id, second)
    database.replace_file_blocks(source_id, source.name, source, [], parser_name="text")
    with database.connect_read_only() as connection:
        source_row = connection.execute(
            "SELECT document_id, content_key, parser_name, parser_version "
            "FROM files WHERE id = ?",
            (source_id,),
        ).fetchone()
    assert source_row is not None
    link = (
        [first_id],
        int(source_row["document_id"]),
        str(source_row["content_key"]),
        str(source_row["parser_name"]),
        str(source_row["parser_version"]),
        "success",
    )
    second_link = (*link[:1], int(link[1]), *link[2:])
    second_link = ([second_id], *second_link[1:])
    progress: list[tuple[int, int]] = []
    linked = database.link_cached_documents_batch(
        [link, second_link],
        batch_file_ids=1,
        start_offset=1,
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert linked == 2
    assert progress == [(2, 2)]
    with database.connect_read_only() as connection:
        rows = connection.execute(
            "SELECT id, parse_status FROM files WHERE id IN (?, ?) ORDER BY id",
            (first_id, second_id),
        ).fetchall()
    assert rows[0]["parse_status"] != "success"
    assert str(rows[1]["parse_status"]) == "success"


def test_rebuild_index_counters_reads_before_short_counter_commit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    empty_root = tmp_path / "empty"
    root.mkdir()
    empty_root.mkdir()
    source = root / "pending.txt"
    source.write_text("counter progress", encoding="utf-8")

    database = DatabaseManager(tmp_path / "counter-progress.db")
    database.initialize()
    root_id = database.add_root(root)
    empty_root_id = database.add_root(empty_root)
    database.upsert_file_metadata(root_id, source)
    progress: list[tuple[str, int, int, str]] = []

    assert database.rebuild_index_counters(
        progress_callback=lambda phase, completed, total, detail: progress.append(
            (phase, completed, total, detail)
        )
    ) == 2

    with database.connect_read_only() as connection:
        rows = connection.execute(
            "SELECT root_id, file_count, blocking_files FROM index_counters "
            "WHERE root_id IN (?, ?) ORDER BY root_id",
            (root_id, empty_root_id),
        ).fetchall()

    assert [(int(row["root_id"]), int(row["file_count"]), int(row["blocking_files"])) for row in rows] == [
        (root_id, 1, 1),
        (empty_root_id, 0, 0),
    ]
    assert progress
    assert progress[-1][1:3] == (4, 4)


def test_sqlite_busy_circuit_opens_after_bounded_attempts() -> None:
    circuit = SQLiteBusyCircuit(
        "test_finalization",
        max_attempts=2,
        max_seconds=60,
    )
    busy = sqlite3.OperationalError("database is locked")
    circuit.note_busy(busy, stage="test")
    with pytest.raises(SQLiteBusyCircuitOpen):
        circuit.note_busy(busy, stage="test")
