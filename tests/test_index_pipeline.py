from __future__ import annotations

import tempfile
import threading
import os
import hashlib
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.task_manager import CancelToken


def test_parsing_starts_before_directory_discovery_finishes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        files = []
        for index in range(70):
            path = root / f"{index:03d}.txt"
            path.write_text(f"PIPELINE_UNIQUE_{index}", encoding="utf-8")
            files.append(path)
        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)
        parse_started = threading.Event()
        discovery_observed_parse = threading.Event()
        index_manager_module = __import__(
            "local_full_text_search.core.index_manager",
            fromlist=["schedule_parse_lanes"],
        )
        original_schedule = index_manager_module.schedule_parse_lanes

        class PipelineProbeManager(IndexManager):
            def _iter_discovered_files(
                self,
                root_path: Path,
                include_subfolders: bool,
                token: CancelToken,
                summary: object,
                progress_callback: object,
            ) -> Iterable[Path]:
                for index, path in enumerate(files, start=1):
                    summary.scanned += 1
                    yield path
                    if index == 64 and parse_started.wait(3):
                        discovery_observed_parse.set()

        def recording_schedule(*args: object, **kwargs: object) -> object:
            submitted = original_schedule(*args, **kwargs)
            if submitted:
                parse_started.set()
            return submitted

        settings = AppSettings(
            enable_ocr=False,
            enable_parse_cache=False,
            index_write_batch_size=16,
            parser_workers=2,
        )
        with patch(
            "local_full_text_search.core.index_manager.schedule_parse_lanes",
            recording_schedule,
        ):
            summary = PipelineProbeManager(db, settings).index_root(root_id)

        assert summary.failed == 0
        assert discovery_observed_parse.is_set()


def test_s0_01r_real_index_pipeline_uses_recoverable_planning_io() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        source = root / "sample.txt"
        source.write_text("RECOVERABLE_PLANNING_PIPELINE", encoding="utf-8")
        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)
        planning_pids: set[int] = set()

        def progress(payload: dict[str, object]) -> None:
            pid = int(payload.get("planning_worker_pid") or 0)
            if pid:
                planning_pids.add(pid)

        settings = AppSettings(
            enable_ocr=False,
            enable_parse_cache=False,
            planning_discovery_batch_size=1,
            parser_workers=1,
        )
        with (
            patch.object(
                db,
                "upsert_file_metadata_many",
                side_effect=AssertionError(
                    "scheduler must not stat/hash through database metadata API"
                ),
            ),
            patch(
                "local_full_text_search.core.index_manager.fingerprint_file_with_spool",
                side_effect=AssertionError(
                    "scheduler must not fingerprint in its own process"
                ),
            ),
        ):
            summary = IndexManager(db, settings).index_root(
                root_id,
                progress_callback=progress,
            )

        assert summary.failed == 0
        assert planning_pids
        assert os.getpid() not in planning_pids


def test_u0_02v_real_index_run_persists_replayable_eta_inputs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        for index in range(4):
            (root / f"eta-{index}.txt").write_text(
                f"ETA_REPLAY_INPUT_{index}",
                encoding="utf-8",
            )
        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)

        summary = IndexManager(
            db,
            AppSettings(enable_ocr=False, parser_workers=1),
        ).index_root(root_id)
        run = db.recent_index_runs_since("1970-01-01T00:00:00+00:00")[-1]
        metrics = run["summary"]["metrics"]
        events = metrics["eta_metrics"]["replay_events"]

        assert summary.failed == 0
        assert events[-1]["event_type"] == "finish"
        completions = [
            event for event in events if event["event_type"] == "completion"
        ]
        assert len(completions) == 4
        assert all(event["completed_cost"] > 0 for event in completions)
        assert all(event["service_seconds"] > 0 for event in completions)
        assert all(event["workers_by_lane"]["normal"] >= 1 for event in events)


def test_p0_01r_submitted_page_state_falls_back_when_fast_db_write_is_busy() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        source = root / "page.pdf"
        source.write_bytes(b"%PDF-1.4\n")
        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)
        file_id, _ = db.upsert_file_metadata(root_id, source)
        task_id = db.create_parse_task(file_id, "page-run", "pdf_native_page")
        manager = IndexManager(db, AppSettings(enable_ocr=False))

        with (
            patch.object(db, "try_mark_tasks_running", return_value=False),
            patch.object(
                db,
                "mark_tasks_running",
                wraps=db.mark_tasks_running,
            ) as durable_mark,
        ):
            manager._mark_submitted_tasks_running([task_id])

        durable_mark.assert_called_once_with([task_id])
        with db.connect() as connection:
            task = connection.execute(
                "SELECT status FROM parse_tasks WHERE id = ?",
                (task_id,),
            ).fetchone()
            attempt = connection.execute(
                "SELECT status FROM parse_task_attempts WHERE task_id = ?",
                (task_id,),
            ).fetchone()
        assert task["status"] == "running"
        assert attempt["status"] == "running"


def test_p0_02r_real_image_job_uses_durable_ocr_request_lifecycle() -> None:
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        image_path = root / "small.jpg"
        Image.new("RGB", (10, 10), "white").save(image_path)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)

        summary = IndexManager(
            db,
            AppSettings(
                enable_ocr=True,
                ocr_images=True,
                ocr_scanned_pdf=False,
                min_ocr_image_pixels=12_000,
                ocr_workers=1,
            ),
        ).index_root(root_id)
        with db.connect() as connection:
            row = connection.execute(
                "SELECT * FROM ocr_requests"
            ).fetchone()

        assert summary.failed == 0
        assert row is not None
        assert row["status"] == "confirmed"
        assert row["lease_owner"] is None
        assert int(row["width"]) == 10
        assert int(row["height"]) == 10
        assert row["content_sha256"] == digest
