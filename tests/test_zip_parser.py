from __future__ import annotations

import tempfile
import unittest
import zipfile
import uuid
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import IndexNotReadyError
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.index_manager import IndexSummary
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.zip_parser import ZipParser
from local_full_text_search.parsers.zip_parser import decoded_zip_member_name


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

    def test_zip_member_indexes_are_stable_when_directory_entries_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("docs/", "")
                handle.writestr("docs/readme.txt", "ZIP_DIRECTORY_ENTRY_SHIFT_HIT")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)

            summary = IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            self.assertEqual(summary.failed, 0)
            page = SearchEngine(db).search(
                SearchQuery(text="ZIP_DIRECTORY_ENTRY_SHIFT_HIT", mode="exact")
            )
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

    def test_zip_member_and_directory_file_are_exactly_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            payload = "ZIP_DIRECTORY_SHARED_HIT"
            plain = root / "plain.txt"
            plain.write_text(payload, encoding="utf-8")
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("docs/copy.txt", payload)
                handle.writestr("docs/other.txt", "ZIP_UNIQUE_HIT_VALUE")

            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            shared = SearchEngine(db).search(SearchQuery(text=payload, mode="exact"))
            self.assertEqual(shared.total_confirmed, 2)
            self.assertEqual(
                {result.filename for result in shared.results},
                {"plain.txt", "copy.txt"},
            )
            with db.connect() as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM documents").fetchone()[0], 2)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0], 2)
                document_ids = {
                    int(row[0])
                    for row in con.execute(
                        "SELECT document_id FROM files WHERE filename IN ('plain.txt', 'copy.txt')"
                    )
                }
                run_summary = con.execute(
                    "SELECT summary_json FROM index_runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()[0]
            self.assertEqual(len(document_ids), 1)
            self.assertIn('"dedup_parse_avoided_count": 1', run_summary)

            archive.unlink()
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text=payload, mode="exact")).total_confirmed,
                1,
            )
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text="ZIP_UNIQUE_HIT_VALUE", mode="exact")).total_confirmed,
                0,
            )

    def test_same_size_zip_member_with_different_bytes_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "plain.txt").write_text("AAAA", encoding="utf-8")
            with zipfile.ZipFile(root / "archive.zip", "w") as handle:
                handle.writestr("copy.txt", "BBBB")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)

            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            self.assertEqual(db.stats()["documents"], 2)
            self.assertEqual(SearchEngine(db).search(SearchQuery(text="AAAA")).total_confirmed, 1)
            self.assertEqual(SearchEngine(db).search(SearchQuery(text="BBBB")).total_confirmed, 1)

    def test_new_directory_file_reuses_an_already_indexed_zip_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            payload = "CROSS_RUN_ZIP_DEDUP_HIT"
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("copy.txt", payload)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=False))
            manager.index_root(root_id)

            (root / "later.txt").write_text(payload, encoding="utf-8")
            second = manager.index_root(root_id)

            self.assertEqual(second.indexed, 0)
            self.assertEqual(second.failed, 0)
            self.assertEqual(db.stats()["documents"], 1)
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text=payload, mode="exact")).total_confirmed,
                2,
            )

    def test_unchanged_zip_recovers_pending_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("resume.txt", "ZIP_MEMBER_RESUME_HIT")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=False))
            manager.index_root(root_id)
            with db.connect() as con:
                con.execute(
                    "UPDATE files SET parse_status = 'pending' WHERE source_kind = 'zip_member'"
                )

            recovery = manager.index_root(root_id)

            self.assertEqual(recovery.indexed, 0)
            self.assertGreaterEqual(recovery.skipped, 1)
            self.assertEqual(recovery.failed, 0)
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text="ZIP_MEMBER_RESUME_HIT", mode="exact")).total_confirmed,
                1,
            )

    def test_large_zip_is_planned_as_independent_member_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                for index in range(6):
                    handle.writestr(f"docs/{index}.txt", f"UNIQUE_MEMBER_{index}")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=False, slow_file_workers=3))
            run_id = uuid.uuid4().hex
            metrics = IndexRunMetrics(run_id=run_id)
            db.start_index_run(metrics)

            jobs = manager._prepare_jobs(
                root_id,
                [archive],
                run_id,
                IndexSummary(scanned=1),
                metrics,
                CancelToken(),
            )

            self.assertEqual(len(jobs), 6)
            self.assertEqual({job.lane for job in jobs}, {"zip"})
            self.assertTrue(all(job.archive_path == archive for job in jobs))
            self.assertEqual({job.archive_member_index for job in jobs}, set(range(6)))

    def test_legacy_gb18030_member_name_is_recovered(self) -> None:
        expected = "故障处理指引/余量检测光耦组件.pdf"
        info = zipfile.ZipInfo()
        info.filename = expected.encode("gb18030").decode("cp437")
        info.flag_bits = 0

        self.assertEqual(decoded_zip_member_name(info), expected)


if __name__ == "__main__":
    unittest.main()
