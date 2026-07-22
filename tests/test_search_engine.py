from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


class SearchEngineTests(unittest.TestCase):
    def test_extension_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "hit.txt").write_text("needle", encoding="utf-8")
            (root / "hit.md").write_text("needle", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)
            page = SearchEngine(db).search(SearchQuery(text="needle", mode="exact", extensions=[".md"]))
            self.assertEqual(page.total_confirmed, 1)
            self.assertTrue(page.results[0].filename.endswith(".md"))


if __name__ == "__main__":
    unittest.main()
