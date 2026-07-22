from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


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

    def test_zip_path_traversal_is_partial_success_or_metadata(self) -> None:
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
            page = SearchEngine(db).search(SearchQuery(text="ZIP_SAFE_HIT", mode="exact"))
            self.assertEqual(page.total_confirmed, 1)


if __name__ == "__main__":
    unittest.main()
