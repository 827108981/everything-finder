from __future__ import annotations

import pickle
import tempfile
import unittest
import errno
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.content_fingerprint import fingerprint_file
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import (
    IndexManager,
    ParseJob,
    ParseOutcome,
    load_partial_parse_checkpoint,
    parse_file_process_worker,
    parse_file_with_registry,
    parser_identity_for_path,
    partial_parse_path,
    error_code_for_exception,
    failed_parse_outcome,
    user_message_for_exception,
    sha256_path,
)
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.parsers.parser_registry import ParserRegistry


class SpoolRecoveryTests(unittest.TestCase):
    def test_process_result_publish_retries_transient_windows_file_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "locked-result.txt"
            source.write_text("LOCKED_RESULT_TEXT", encoding="utf-8")
            settings = AppSettings(enable_ocr=False)
            job = ParseJob(
                file_id=6,
                file_path=source,
                parser_name="text",
                lane="normal",
            )
            original_replace = Path.replace
            result_attempts = 0

            def transient_lock(path: Path, target: Path) -> Path:
                nonlocal result_attempts
                if Path(target).suffix != ".pickle":
                    return original_replace(path, target)
                result_attempts += 1
                if result_attempts < 3:
                    exc = PermissionError(errno.EACCES, "sharing violation")
                    exc.winerror = 32
                    raise exc
                return original_replace(path, target)

            with patch.object(Path, "replace", transient_lock):
                result = parse_file_process_worker(job, settings, base / "spool")

            self.assertEqual(result_attempts, 3)
            self.assertTrue(result.spool_path.is_file())

    def test_windows_sharing_violation_is_reported_as_file_in_use(self) -> None:
        exc = PermissionError(errno.EACCES, "sharing violation")
        exc.winerror = 32

        self.assertEqual(error_code_for_exception(exc), "FILE_IN_USE")
        self.assertEqual(user_message_for_exception(exc), "文件正被其他程序占用，请稍后重试")
        outcome = failed_parse_outcome(
            ParseJob(file_id=10, file_path=Path("locked.pdf")),
            exc,
        )
        self.assertEqual(outcome.status, "failed_retryable")

    def test_process_parse_checkpoint_preserves_completed_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "checkpoint.txt"
            source.write_text("CHECKPOINT_TEXT", encoding="utf-8")
            settings = AppSettings(enable_ocr=False)
            job = ParseJob(
                file_id=7,
                file_path=source,
                parser_name="text",
                lane="normal",
                queued_monotonic=1.0,
                started_monotonic=1.0,
            )
            spool_dir = base / "spool"
            checkpoint_path = partial_parse_path(job, spool_dir)

            parse_file_with_registry(
                job,
                ParserRegistry(settings),
                CancelToken(),
                settings,
                checkpoint_path=checkpoint_path,
            )

            recovered = load_partial_parse_checkpoint(job, spool_dir)
            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertEqual(recovered.status, "partial_success")
            self.assertIn("CHECKPOINT_TEXT", recovered.blocks[0].raw_text)
            self.assertFalse(checkpoint_path.exists())

    def test_completed_process_result_removes_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "complete.txt"
            source.write_text("COMPLETE_TEXT", encoding="utf-8")
            settings = AppSettings(enable_ocr=False)
            job = ParseJob(
                file_id=8,
                file_path=source,
                parser_name="text",
                lane="normal",
                queued_monotonic=1.0,
                started_monotonic=1.0,
            )
            spool_dir = base / "spool"

            result = parse_file_process_worker(job, settings, spool_dir)

            self.assertTrue(result.spool_path.is_file())
            self.assertFalse(partial_parse_path(job, spool_dir).exists())
            result.spool_path.unlink()

    def test_checkpoint_can_be_loaded_from_a_later_run_spool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "resume.txt"
            source.write_text("PERSISTENT_CHECKPOINT_TEXT", encoding="utf-8")
            checkpoint = base / "persistent" / "resume.partial.pickle"
            settings = AppSettings(enable_ocr=False)
            first = ParseJob(
                file_id=9,
                file_path=source,
                parser_name="text",
                parser_version="2",
                content_key="sha256:persistent",
                checkpoint_path=checkpoint,
                queued_monotonic=1.0,
                started_monotonic=1.0,
            )
            parse_file_with_registry(
                first,
                ParserRegistry(settings),
                CancelToken(),
                settings,
                checkpoint_path=partial_parse_path(first, base / "run-one"),
            )
            second = ParseJob(
                file_id=9,
                file_path=source,
                parser_name="text",
                parser_version="2",
                content_key="sha256:persistent",
                checkpoint_path=checkpoint,
            )

            recovered = load_partial_parse_checkpoint(
                second,
                base / "different-run",
                consume=False,
            )

            self.assertIsNotNone(recovered)
            assert recovered is not None
            self.assertIn("PERSISTENT_CHECKPOINT_TEXT", recovered.blocks[0].raw_text)

    def test_valid_spooled_artifact_is_written_without_reparsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "recover.txt"
            source.write_text("RECOVERED_SPOOL_HIT", encoding="utf-8")
            settings = AppSettings(enable_ocr=False)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            parser_name, parser_version = parser_identity_for_path(source, settings)
            file_id, _ = db.upsert_file_metadata(
                root_id,
                source,
                parser_version=parser_version,
            )
            task_id = db.create_parse_task(file_id, "old-run", "normal")
            db.mark_task_running(task_id)
            runtime_temp = base / "runtime"
            spool_path = runtime_temp / "process_results" / "old-run" / "artifact.pickle"
            spool_path.parent.mkdir(parents=True)
            outcome = ParseOutcome(
                file_id=file_id,
                file_path=source,
                blocks=[
                    ContentBlock(
                        file_path=str(source),
                        block_index=0,
                        block_type="text",
                        location_text="第 1 行",
                        raw_text="RECOVERED_SPOOL_HIT",
                        normalized_text=normalize_text("RECOVERED_SPOOL_HIT"),
                    )
                ],
                parser_name=parser_name,
                status="success",
                task_id=task_id,
                content_key=fingerprint_file(source).key,
                parser_version=parser_version,
            )
            with spool_path.open("wb") as stream:
                pickle.dump(outcome, stream, protocol=pickle.HIGHEST_PROTOCOL)
            db.mark_task_spooled(task_id, spool_path, sha256_path(spool_path))

            with patch("local_full_text_search.core.index_manager.TEMP_DIR", runtime_temp):
                summary = IndexManager(db, settings).index_root(root_id)

            self.assertEqual(summary.indexed, 1)
            self.assertEqual(summary.skipped, 1)
            self.assertFalse(spool_path.exists())
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text="RECOVERED_SPOOL_HIT")).total_confirmed,
                1,
            )
            with db.connect() as con:
                status = con.execute(
                    "SELECT status FROM parse_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()[0]
            self.assertEqual(status, "complete")


if __name__ == "__main__":
    unittest.main()
