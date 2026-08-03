from __future__ import annotations

import tempfile
import unittest
import json
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
    def test_partial_failure_records_exact_member_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "field-failure.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("nested/broken.xml", b"broken")
            parser = ZipParser(AppSettings(enable_ocr=False))

            with patch.object(
                parser,
                "_parse_text_member",
                side_effect=UnicodeError(
                    "UTF-16 stream does not start with BOM"
                ),
            ):
                list(parser.parse(archive, CancelToken()))

            self.assertEqual(parser.last_error_code, "ZIP_PARTIAL_FAILURE")
            self.assertEqual(len(parser.last_diagnostics), 1)
            diagnostic = parser.last_diagnostics[0]
            self.assertEqual(diagnostic["member_path"], "nested/broken.xml")
            self.assertEqual(diagnostic["parser"], "text")
            self.assertEqual(diagnostic["error_code"], "UnicodeError")
            self.assertIn("does not start with BOM", diagnostic["error_message"])

    def test_member_diagnostics_round_trip_through_failed_file_query(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "partial.zip"
            archive.write_bytes(b"placeholder")
            database = DatabaseManager(base / "index.db")
            database.initialize()
            root_id = database.add_root(root)
            file_id, _ = database.upsert_file_metadata(root_id, archive)
            diagnostics = [
                {
                    "member_path": "docs/broken.xml",
                    "parser": "text",
                    "error_code": "UnicodeError",
                    "error_message": "invalid UTF-16 stream",
                }
            ]

            database.replace_document_blocks_many(
                [
                    {
                        "file_id": file_id,
                        "file_ids": [file_id],
                        "filename": archive.name,
                        "path": str(archive),
                        "blocks": [],
                        "parser_name": "zip",
                        "parser_version": "test-v1",
                        "status": "failed",
                        "error_code": "ZIP_PARTIAL_FAILURE",
                        "error_message": "one member failed",
                        "diagnostics": diagnostics,
                        "content_key": "zip-partial-test",
                        "task_id": None,
                    }
                ]
            )

            row = database.failed_files()[0]
            self.assertEqual(
                json.loads(row["parse_diagnostics_json"]),
                diagnostics,
            )

    def test_zip_safety_skip_completes_task_but_keeps_file_blocking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "large.zip"
            archive.write_bytes(b"placeholder")
            database = DatabaseManager(base / "index.db")
            database.initialize()
            root_id = database.add_root(root)
            file_id, _ = database.upsert_file_metadata(root_id, archive)
            task_id = database.create_parse_task(file_id, "zip-limit", "zip")

            database.replace_document_blocks_many(
                [
                    {
                        "file_id": file_id,
                        "file_ids": [file_id],
                        "filename": archive.name,
                        "path": str(archive),
                        "blocks": [],
                        "parser_name": "zip",
                        "parser_version": "test-v1",
                        "status": "skipped",
                        "error_code": "ZIP_SIZE_LIMIT",
                        "error_message": "archive exceeds the safety limit",
                        "diagnostics": [],
                        "content_key": "zip-limit-test",
                        "task_id": task_id,
                    }
                ]
            )

            with database.connect() as connection:
                task_status = connection.execute(
                    "SELECT status FROM parse_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()[0]
            readiness = database.index_readiness()

            self.assertEqual(task_status, "complete")
            self.assertEqual(readiness["unfinished_tasks"], 0)
            self.assertEqual(readiness["blocking_files"], 1)
            self.assertFalse(readiness["ready"])

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

    def test_large_zip_is_planned_as_independent_format_lane_jobs(self) -> None:
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
            self.assertEqual({job.lane for job in jobs}, {"normal"})
            self.assertTrue(all(job.archive_path == archive for job in jobs))
            self.assertEqual({job.archive_member_index for job in jobs}, set(range(6)))

    def test_zip_members_are_dispatched_to_their_real_format_lanes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            archive = root / "mixed.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("readme.txt", "text")
                handle.writestr("manual.pdf", b"%PDF-placeholder")
                handle.writestr("photo.png", b"png-placeholder")
                handle.writestr("legacy.doc", b"legacy-placeholder")
                handle.writestr("nested.zip", b"nested-placeholder")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=True))
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

            lanes = {job.file_path.suffix.lower(): job.lane for job in jobs}
            self.assertEqual(lanes[".txt"], "normal")
            self.assertEqual(lanes[".pdf"], "pdf")
            self.assertEqual(lanes[".png"], "ocr")
            self.assertEqual(lanes[".doc"], "legacy_word")
            self.assertEqual(lanes[".zip"], "zip")

    def test_dedup_candidate_member_reuses_planning_spool_during_parse(self) -> None:
        from local_full_text_search.core.index_manager import parse_file_with_registry
        from local_full_text_search.parsers.parser_registry import ParserRegistry

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            payload = "ZIP_SINGLE_EXTRACTION_PAYLOAD_B"
            plain_payload = "ZIP_SINGLE_EXTRACTION_PAYLOAD_A"
            (root / "plain.txt").write_text(plain_payload, encoding="utf-8")
            archive = root / "archive.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("copy.txt", payload)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(enable_ocr=False)
            manager = IndexManager(db, settings)
            run_id = uuid.uuid4().hex
            metrics = IndexRunMetrics(run_id=run_id)
            db.start_index_run(metrics)
            jobs = manager._prepare_jobs(
                root_id,
                [root / "plain.txt", archive],
                run_id,
                IndexSummary(scanned=2),
                metrics,
                CancelToken(),
            )
            member_job = next(job for job in jobs if job.archive_path is not None)

            self.assertIsNotNone(member_job.source_spool_path)
            with patch(
                "local_full_text_search.core.index_manager.materialize_zip_member",
                side_effect=AssertionError("ZIP member extracted twice"),
            ):
                outcome = parse_file_with_registry(
                    member_job,
                    ParserRegistry(settings),
                    CancelToken(),
                    settings,
                )

            self.assertEqual(outcome.status, "success")
            self.assertIn(payload, outcome.blocks[0].raw_text)

    def test_legacy_gb18030_member_name_is_recovered(self) -> None:
        expected = "故障处理指引/余量检测光耦组件.pdf"
        info = zipfile.ZipInfo()
        info.filename = expected.encode("gb18030").decode("cp437")
        info.flag_bits = 0

        self.assertEqual(decoded_zip_member_name(info), expected)


if __name__ == "__main__":
    unittest.main()
