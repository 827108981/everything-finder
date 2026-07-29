from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
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


if __name__ == "__main__":
    unittest.main()
