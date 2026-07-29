from __future__ import annotations

import sqlite3
import tempfile
import threading
import time
import unittest
from pathlib import Path

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.models.search_query import SearchQuery


class DatabaseIndexSearchTests(unittest.TestCase):
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
