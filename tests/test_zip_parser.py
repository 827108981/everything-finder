from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import IndexNotReadyError
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.zip_parser import ZipParser


class ZipParserTests(unittest.TestCase):
    def test_zip_inner_text_is_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("docs/readme.txt", "ZIP_INNER_HIT")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.failed, 0)
            page = SearchEngine(db).search(SearchQuery(text="ZIP_INNER_HIT", mode="exact"))
            self.assertEqual(page.total_confirmed, 1)
            self.assertIn("archive.zip > docs/readme.txt", page.results[0].location_text)

    def test_zip_path_traversal_blocks_search_until_fully_successful(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("../evil.txt", "bad")
                handle.writestr("good.txt", "ZIP_SAFE_HIT")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.partial_success, 1)
            self.assertFalse(db.index_readiness()["ready"])
            with self.assertRaises(IndexNotReadyError):
                SearchEngine(db).search(SearchQuery(text="ZIP_SAFE_HIT", mode="exact"))
            self.assertEqual(db.stats()["blocks"], 0)

    def test_zip_temporary_directory_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            temp_root = base / "temp"
            archive = base / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("readme.txt", "ZIP_TEMP_CLEANUP")

            with patch("local_full_text_search.parsers.zip_parser.TEMP_DIR", temp_root):
                blocks = list(ZipParser(AppSettings()).parse(archive, CancelToken()))

            self.assertTrue(blocks)
            self.assertFalse(list(temp_root.glob("zip_index_*")))


if __name__ == "__main__":
    unittest.main()
