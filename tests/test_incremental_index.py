from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
