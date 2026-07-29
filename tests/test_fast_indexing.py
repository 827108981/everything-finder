from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager, ParseJob, ParseOutcome
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
            self.assertEqual(db.stats()["blocks"], 30)
            with db.connect() as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM short_tokens").fetchone()[0], 0)
            page = SearchEngine(db).search(SearchQuery(text="PARALLEL_HIT", mode="exact", page_size=100))
            self.assertEqual(page.total_confirmed, 30)

    def test_tiny_image_is_complete_without_loading_ocr(self) -> None:
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
            self.assertEqual(summary.indexed, 1)
            self.assertEqual(db.failed_files(), [])
            self.assertTrue(db.index_readiness()["ready"])

    def test_saturated_ocr_lane_does_not_block_normal_documents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            for index in range(12):
                Image.new("RGB", (20, 20), "white").save(root / f"image_{index:02d}.png")
            normal_path = root / "normal.txt"
            normal_path.write_text("NORMAL_QUEUE_HIT", encoding="utf-8")

            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            completions: list[tuple[str, str]] = []

            def fake_parse(job: ParseJob, settings: AppSettings, token: object) -> ParseOutcome:
                if job.file_path.suffix.lower() == ".png":
                    time.sleep(0.03)
                return ParseOutcome(job.file_id, job.file_path, [], "fake", "success")

            def capture_progress(payload: dict[str, object]) -> None:
                if payload.get("stage") == "indexing" and payload.get("completed_queue"):
                    completions.append(
                        (
                            Path(str(payload["completed_file"])).name,
                            str(payload["completed_queue"]),
                        )
                    )

            settings = AppSettings(
                normal_pending_tasks=1,
                ocr_pending_tasks=1,
                ocr_workers=1,
                index_write_batch_size=1,
            )
            with patch("local_full_text_search.core.index_manager.parse_file_worker", side_effect=fake_parse):
                summary = IndexManager(db, settings).index_root(root_id, progress_callback=capture_progress)

            normal_position = next(index for index, item in enumerate(completions) if item[0] == "normal.txt")
            self.assertLess(normal_position, 3)
            self.assertEqual(completions[normal_position][1], "normal")
            self.assertEqual(summary.indexed, 13)


if __name__ == "__main__":
    unittest.main()
