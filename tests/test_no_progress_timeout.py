from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import (
    ParseJob,
    no_progress_timeout,
    process_progress_path,
    refresh_job_progress,
    write_process_progress,
)


class NoProgressTimeoutTests(unittest.TestCase):
    def test_timeout_is_selected_by_stage_and_expands_on_retry(self) -> None:
        settings = AppSettings(
            normal_no_progress_timeout_seconds=3,
            ocr_no_progress_timeout_seconds=7,
            archive_no_progress_timeout_seconds=11,
            legacy_no_progress_timeout_seconds=13,
            process_no_progress_timeout_seconds=5,
        )
        job = ParseJob(file_id=1, file_path=Path("sample.pdf"), lane="office_process")

        self.assertEqual(no_progress_timeout(settings, job), 5)
        job.progress_phase = "pdf_page_ocr"
        self.assertEqual(no_progress_timeout(settings, job), 7)
        job.retry_count = 2
        self.assertEqual(no_progress_timeout(settings, job), 28)

    def test_clock_moves_only_for_a_new_semantic_progress_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spool = Path(tmp)
            job = ParseJob(
                file_id=7,
                file_path=Path("large.pdf"),
                lane="ocr",
                started_monotonic=1.0,
                last_progress_monotonic=1.0,
            )
            path = process_progress_path(job, spool)
            payload = {
                "file_id": job.file_id,
                "progress_sequence": 1,
                "phase": "pdf_page",
                "completed": 3,
                "total": 100,
                "detail": "第 3 页",
            }
            write_process_progress(path, payload)
            refresh_job_progress(job, spool, 10.0)
            self.assertEqual(job.last_progress_monotonic, 10.0)
            self.assertEqual(job.progress_completed, 3)

            refresh_job_progress(job, spool, 20.0)
            self.assertEqual(job.last_progress_monotonic, 10.0)

            payload["progress_sequence"] = 2
            payload["completed"] = 4
            write_process_progress(path, payload)
            refresh_job_progress(job, spool, 30.0)
            self.assertEqual(job.last_progress_monotonic, 30.0)
            self.assertEqual(job.progress_completed, 4)


if __name__ == "__main__":
    unittest.main()
