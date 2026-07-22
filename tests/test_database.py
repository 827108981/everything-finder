from __future__ import annotations

import tempfile
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

    def test_unsupported_format_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "raw.bin").write_bytes(b"\x00\x01")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.unsupported, 1)
            failed = db.failed_files()
            self.assertEqual(failed[0]["parse_status"], "unsupported")

    def test_mp4_is_metadata_only_not_failed(self) -> None:
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
            diagnostics = db.failed_files()
            self.assertEqual(diagnostics[0]["parse_status"], "metadata_only")


if __name__ == "__main__":
    unittest.main()
