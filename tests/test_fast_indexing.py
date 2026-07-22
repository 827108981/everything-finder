from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


class FastIndexingTests(unittest.TestCase):
    def test_parallel_batch_indexing_covers_all_changed_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            for index in range(30):
                (root / f"doc_{index:02d}.txt").write_text(f"PARALLEL_HIT_{index:02d}", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(parser_workers=4, index_write_batch_size=7, max_pending_parse_tasks=10)
            summary = IndexManager(db, settings).index_root(root_id)
            self.assertEqual(summary.indexed, 30)
            page = SearchEngine(db).search(SearchQuery(text="PARALLEL_HIT", mode="exact", page_size=100))
            self.assertEqual(page.total_confirmed, 30)

    def test_tiny_image_is_metadata_only_without_loading_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            Image.new("RGB", (20, 20), "white").save(root / "icon.png")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(enable_ocr=True, ocr_images=True, min_ocr_image_pixels=12_000)
            summary = IndexManager(db, settings).index_root(root_id)
            self.assertEqual(summary.failed, 0)
            self.assertEqual(summary.metadata_only, 1)
            diagnostics = db.failed_files()
            self.assertEqual(diagnostics[0]["parse_error_code"], "IMAGE_TOO_SMALL_FOR_OCR")


if __name__ == "__main__":
    unittest.main()
