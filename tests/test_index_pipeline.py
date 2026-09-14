from __future__ import annotations

import tempfile
import threading
import os
import hashlib
import sqlite3
import time
from collections.abc import Iterable
from pathlib import Path
from unittest.mock import patch

import pytest

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import (
    DatabaseManager,
    SQLiteBusyCircuit,
    SQLiteBusyCircuitOpen,
)
from local_full_text_search.core.index_manager import (
    IndexFinalizationWatchdog,
    IndexFinalizationWatchdogError,
    IndexManager,
    IndexSummary,
)
from local_full_text_search.core.errors import PlanningWorkerError
from local_full_text_search.core.task_manager import CancelToken


def test_global_fts_drain_retries_busy_and_covers_pending_files_from_all_roots(
    tmp_path: Path,
) -> None:
    class FakeDatabase:
        db_path = tmp_path / "candidate.db"

        def __init__(self) -> None:
            self.pending = [11, 22, 33]
            self.sync_calls: list[tuple[int, ...]] = []
            self.attempts = 0

        def pending_content_fts_file_ids(self) -> tuple[int, ...]:
            return tuple(self.pending)

        def sync_content_fts_for_files(self, file_ids, **_kwargs) -> int:
            self.attempts += 1
            self.sync_calls.append(tuple(int(value) for value in file_ids))
            if self.attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            self.pending = []
            return 8

    database = FakeDatabase()
    manager = IndexManager(database, AppSettings(enable_ocr=False))  # type: ignore[arg-type]

    manager._drain_pending_content_fts(CancelToken(), None)

    assert database.sync_calls == [(11, 22, 33), (11, 22, 33)]
    assert database.pending == []


def test_finalization_watchdog_detects_semantic_stall(tmp_path: Path) -> None:
    database = DatabaseManager(tmp_path / "watchdog.db")
    watchdog = IndexFinalizationWatchdog(
        database,
        run_id="watchdog-test",
        root_id=1,
        timeout_seconds=1.0,
        check_interval_seconds=0.1,
    )
    watchdog.start()
    try:
        watchdog.progress("cache_link", total=10, detail="test")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            try:
                watchdog.check()
            except IndexFinalizationWatchdogError:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("finalization watchdog did not fire")
    finally:
        watchdog.stop(outcome="test")


def test_finalization_busy_circuit_stops_repeated_lock_errors(tmp_path: Path) -> None:
    database = DatabaseManager(tmp_path / "busy-circuit.db")
    manager = IndexManager(database, AppSettings(enable_ocr=False))
    watchdog = IndexFinalizationWatchdog(
        database,
        run_id="busy-circuit-test",
        root_id=1,
        timeout_seconds=5.0,
        check_interval_seconds=0.1,
    )
    circuit = SQLiteBusyCircuit(
        "test_finalization_operation",
        max_attempts=2,
        max_seconds=60,
        database_path=database.db_path,
    )

    def always_busy() -> object:
        raise sqlite3.OperationalError("database is locked")

    watchdog.start()
    try:
        with pytest.raises(SQLiteBusyCircuitOpen):
            manager._run_finalization_operation(
                always_busy,
                watchdog=watchdog,
                run_id="busy-circuit-test",
                operation_name="test_finalization_operation",
                busy_circuit=circuit,
                batch_size=1,
            )
    finally:
        watchdog.stop(outcome="test")


def test_infrastructure_root_failure_stops_candidate_publish_path(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    manager = IndexManager(database, AppSettings(enable_ocr=False))

    with patch.object(
        manager,
        "index_root",
        side_effect=IndexFinalizationWatchdogError("counter rebuild stalled"),
    ):
        with pytest.raises(RuntimeError, match="基础设施阶段失败"):
            manager.index_enabled_roots(root_ids=[root_id])


def test_directory_planning_worker_interrupt_retries_and_skips_emitted_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    first = root / "first.txt"
    second = root / "second.txt"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    class InterruptingRunner:
        calls = 0

        def stream(self, *_args: object, **_kwargs: object):
            self.calls += 1
            if self.calls == 1:
                yield [str(first)]
                raise PlanningWorkerError("directory_enumeration 失败：KeyboardInterrupt")
            yield [str(first), str(second)]

    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    manager = IndexManager(database, AppSettings(enable_ocr=False))
    manager._planning_runner = InterruptingRunner()  # type: ignore[assignment]
    manager._planning_control_dir = tmp_path / "planning-control"
    summary = IndexSummary()

    with patch.object(
        manager,
        "_stat_paths_recoverable",
        return_value=([], []),
    ):
        paths = list(
            manager._iter_discovered_files(
                root,
                True,
                CancelToken(),
                summary,
                None,
            )
        )

    assert paths == [first, second]
    assert manager._planning_runner.calls == 2  # type: ignore[union-attr]
    assert summary.scanned == 2


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


def test_completed_results_are_committed_while_discovery_total_is_still_growing() -> None:
    """A large directory must not defer every writer submission until discovery ends."""

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        root = base / "files"
        root.mkdir()
        files = []
        for index in range(70):
            path = root / f"{index:03d}.txt"
            path.write_text(f"PIPELINE_STREAMING_{index}", encoding="utf-8")
            files.append(path)

        db = DatabaseManager(base / "index.db")
        db.initialize()
        root_id = db.add_root(root)
        discovery_finished = threading.Event()
        committed_during_discovery = threading.Event()
        database_commit_during_discovery = threading.Event()
        original_replace_many = db.replace_document_blocks_many

        class StreamingProbeManager(IndexManager):
            def _iter_discovered_files(
                self,
                root_path: Path,
                include_subfolders: bool,
                token: CancelToken,
                summary: object,
                progress_callback: object,
            ) -> Iterable[Path]:
                try:
                    for index, path in enumerate(files, start=1):
                        summary.scanned += 1
                        yield path
                        # The first planning batch contains 32 files.  Wait
                        # after it has been submitted so the next batch boundary
                        # has a deterministic completed future to drain.
                        if index == 33:
                            deadline = time.monotonic() + 10
                            while time.monotonic() < deadline:
                                normal_lane = (self._pause_lanes or {}).get("normal")
                                if normal_lane is not None and any(
                                    future.done() for future in normal_lane.futures
                                ):
                                    break
                                time.sleep(0.01)
                            else:
                                raise AssertionError(
                                    "first planning batch did not complete in time"
                                )
                        if index == 65:
                            assert database_commit_during_discovery.wait(10)
                finally:
                    discovery_finished.set()

        def recording_replace_many(*args: object, **kwargs: object) -> object:
            result = original_replace_many(*args, **kwargs)
            if not discovery_finished.is_set():
                database_commit_during_discovery.set()
            return result

        def capture_progress(payload: dict[str, object]) -> None:
            if (
                payload.get("stage") == "indexing"
                and int(payload.get("completed_files") or 0) > 0
                and not discovery_finished.is_set()
            ):
                committed_during_discovery.set()

        settings = AppSettings(
            enable_ocr=False,
            enable_parse_cache=False,
            index_write_batch_size=16,
            parser_workers=2,
        )
        with patch.object(
            db,
            "replace_document_blocks_many",
            side_effect=recording_replace_many,
        ):
            summary = StreamingProbeManager(db, settings).index_root(
                root_id,
                progress_callback=capture_progress,
            )

        assert summary.failed == 0
        assert summary.indexed == len(files)
        assert committed_during_discovery.is_set()
        assert database_commit_during_discovery.is_set()


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


def test_p0_01r_submitted_page_state_does_not_block_scheduler_when_db_is_busy() -> None:
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

            durable_mark.assert_not_called()
            with db.connect() as connection:
                task = connection.execute(
                    "SELECT status FROM parse_tasks WHERE id = ?",
                    (task_id,),
                ).fetchone()
                attempt = connection.execute(
                    "SELECT status FROM parse_task_attempts WHERE task_id = ?",
                    (task_id,),
                ).fetchone()
            assert task["status"] == "queued"
            assert attempt is None


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
