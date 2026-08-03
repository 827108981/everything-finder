from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_full_text_search.core.database import SCHEMA_VERSION, DatabaseManager
from local_full_text_search.core.planning_tasks import PreparedFileMetadata
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.models.search_query import SearchQuery


class DatabaseIndexSearchTests(unittest.TestCase):
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
