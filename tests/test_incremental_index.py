from __future__ import annotations

import os
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.index_manager import IndexSummary
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


class IncrementalIndexTests(unittest.TestCase):
    def test_modified_file_replaces_old_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "data.txt"
            target.write_text("old value", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings())
            manager.index_root(root_id)
            time.sleep(0.01)
            target.write_text("new value", encoding="utf-8")
            manager.index_root(root_id)
            engine = SearchEngine(db)
            self.assertEqual(engine.search(SearchQuery(text="old", mode="exact")).total_confirmed, 0)
            self.assertEqual(engine.search(SearchQuery(text="new", mode="exact")).total_confirmed, 1)

    def test_restored_file_with_same_fingerprint_rebuilds_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "restore.txt"
            target.write_text("RESTORE_BODY_HIT", encoding="utf-8")
            original = target.stat()
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings())
            manager.index_root(root_id)
            target.unlink()
            manager.index_root(root_id)

            target.write_text("RESTORE_BODY_HIT", encoding="utf-8")
            os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
            restored = manager.index_root(root_id)
            page = SearchEngine(db).search(
                SearchQuery(
                    text="RESTORE_BODY_HIT",
                    mode="exact",
                    search_filename=False,
                    search_path=False,
                )
            )

            self.assertEqual(restored.indexed, 1)
            self.assertEqual(page.total_confirmed, 1)

    def test_reused_files_emit_immediate_progress_after_cache_planning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            target = root / "unchanged.txt"
            target.write_text("UNCHANGED_CACHE_HIT", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=False))
            manager.index_root(root_id)

            progress: list[dict[str, object]] = []
            reused = manager.index_root(root_id, progress_callback=progress.append)

            self.assertEqual(reused.indexed, 0)
            self.assertEqual(reused.skipped, 1)
            self.assertTrue(
                any(
                    int(payload.get("reused_files") or 0) >= 1
                    for payload in progress
                )
            )

    def test_incomplete_file_is_retried_on_each_full_index_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "broken.zip").write_bytes(b"not a zip")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(retry_failed_files=True))

            first = manager.index_root(root_id)
            second = manager.index_root(root_id)

            self.assertEqual(first.failed, 1)
            self.assertEqual(second.failed, 1)
            self.assertEqual(second.skipped, 0)
            self.assertFalse(db.index_readiness()["ready"])

    def test_incomplete_projection_blocks_search_until_failed_file_is_resolved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "good.txt").write_text(
                "PIPELINE_PARTIAL_HIT",
                encoding="utf-8",
            )
            (root / "broken.zip").write_bytes(b"not a zip")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(
                db,
                AppSettings(
                    defer_fts_during_full_scan=True,
                    partial_search_during_index=True,
                    retry_failed_files=False,
                ),
            )

            first = manager.index_root(root_id)
            readiness = db.index_readiness()
            with self.assertRaises(Exception):
                SearchEngine(db).search(
                    SearchQuery(
                        text="PIPELINE_PARTIAL_HIT",
                        mode="exact",
                        search_filename=False,
                        search_path=False,
                        search_content=True,
                    )
                )
            second = manager.index_root(root_id)

            self.assertEqual(first.indexed, 1)
            self.assertEqual(first.failed, 1)
            self.assertFalse(readiness["ready"])
            self.assertFalse(readiness["search_available"])
            self.assertEqual(second.failed, 0)
            self.assertEqual(second.indexed, 0)
            self.assertEqual(second.skipped, 2)

    def test_incremental_update_blocks_existing_search_until_scope_is_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            good = root / "good.txt"
            good.write_text("INCREMENTAL_SEARCH_HIT", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(
                db,
                AppSettings(
                    partial_search_during_index=True,
                    retry_failed_files=False,
                ),
            )
            manager.index_root(root_id)

            # Simulate a database created before partial-search state was set.
            db.set_partial_search_projection_active(False)
            (root / "broken.zip").write_bytes(b"not a zip")
            second = manager.index_root(root_id)

            readiness = db.index_readiness()
            self.assertEqual(second.failed, 1)
            self.assertFalse(readiness["ready"])
            self.assertFalse(readiness["search_available"])
            with self.assertRaises(Exception):
                SearchEngine(db).search(
                    SearchQuery(
                        text="INCREMENTAL_SEARCH_HIT",
                        mode="exact",
                        search_filename=False,
                        search_path=False,
                        search_content=True,
                    )
                )

    def test_one_root_failure_does_not_prevent_later_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_root = base / "first"
            second_root = base / "second"
            first_root.mkdir()
            second_root.mkdir()
            db = DatabaseManager(base / "index.db")
            db.initialize()
            first_id = db.add_root(first_root)
            second_id = db.add_root(second_root)
            manager = IndexManager(db, AppSettings())
            calls: list[int] = []

            def fake_index_root(root_id, _token=None, _progress_callback=None):
                calls.append(int(root_id))
                if int(root_id) == first_id:
                    raise PermissionError(5, "access denied", str(first_root))
                return IndexSummary(scanned=1, indexed=1)

            manager.index_root = fake_index_root  # type: ignore[method-assign]
            summary = manager.index_enabled_roots()

            self.assertIn(second_id, calls)
            self.assertEqual(summary.indexed, 1)
            self.assertGreaterEqual(summary.failed, 1)

    def test_database_lock_stops_root_loop_instead_of_repeating_every_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first_root = base / "first"
            second_root = base / "second"
            first_root.mkdir()
            second_root.mkdir()
            db = DatabaseManager(base / "index.db")
            db.initialize()
            first_id = db.add_root(first_root)
            second_id = db.add_root(second_root)
            manager = IndexManager(db, AppSettings())
            calls: list[int] = []

            def fake_index_root(root_id, _token=None, _progress_callback=None):
                calls.append(int(root_id))
                raise sqlite3.OperationalError("database is locked")

            manager.index_root = fake_index_root  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "数据库被其他进程"):
                manager.index_enabled_roots()

            self.assertEqual(calls, [first_id])
            self.assertNotIn(second_id, calls)


if __name__ == "__main__":
    unittest.main()
