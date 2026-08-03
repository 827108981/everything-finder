from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import pytest

from local_full_text_search.core.errors import CancelledError, PauseRequestedError
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import (
    IndexManager,
    ParseJob,
    ParseLane,
    ParseOutcome,
    ProcessLaneWatchdog,
    parse_file_process_worker,
    schedule_parse_lanes,
)
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.core.run_control import (
    planning_pause_acknowledgements,
    pause_marker_path,
    request_process_pause,
    resume_processes,
)
from local_full_text_search.core.task_manager import CancelToken, ProcessRunControlToken


def test_process_token_returns_from_a_safe_point_when_pause_is_requested() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        control_dir = Path(tmp)
        token = ProcessRunControlToken(control_dir)

        token.wait_if_paused()
        request_process_pause(control_dir)
        with pytest.raises(PauseRequestedError):
            token.wait_if_paused()

        resume_processes(control_dir)
        token.wait_if_paused()
        assert not pause_marker_path(control_dir).exists()


def test_thread_token_wait_is_released_by_resume() -> None:
    token = CancelToken()
    token.pause()
    finished = threading.Event()

    def wait_for_resume() -> None:
        token.wait_if_paused()
        finished.set()

    thread = threading.Thread(target=wait_for_resume)
    thread.start()
    time.sleep(0.03)
    assert not finished.is_set()
    token.resume()
    thread.join(timeout=1)
    assert finished.is_set()


def test_cancel_releases_a_paused_thread_with_cancelled_error() -> None:
    token = CancelToken()
    token.pause()
    observed: list[type[BaseException]] = []

    def wait_for_resume() -> None:
        try:
            token.wait_if_paused()
        except BaseException as exc:  # pragma: no branch - assertion records exact type.
            observed.append(type(exc))

    thread = threading.Thread(target=wait_for_resume)
    thread.start()
    time.sleep(0.03)
    token.cancel()
    thread.join(timeout=1)

    assert observed == [CancelledError]


def test_process_parser_returns_a_paused_outcome_at_its_first_safe_point() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "sample.txt"
        source.write_text("safe pause", encoding="utf-8")
        request_process_pause(root)
        job = ParseJob(
            file_id=7,
            file_path=source,
            parser_name="text",
            content_key="test",
            started_monotonic=time.perf_counter(),
            queued_monotonic=time.perf_counter(),
        )

        descriptor = parse_file_process_worker(
            job,
            AppSettings(enable_ocr=False),
            root,
        )

        import pickle

        with descriptor.spool_path.open("rb") as stream:
            outcome = pickle.load(stream)
        assert outcome.status == "paused"
        assert outcome.error_code == "PAUSED_AT_SAFE_POINT"


def test_paused_scheduler_returns_without_blocking_the_drain_loop() -> None:
    token = CancelToken()
    token.pause()
    executor = ThreadPoolExecutor(max_workers=1)
    lane = ParseLane(
        "normal",
        executor,
        1,
        1024 * 1024,
        worker_count=1,
    )
    lane.pending.append(
        ParseJob(
            file_id=11,
            file_path=Path("pending.txt"),
            parser_name="text",
            memory_estimate_bytes=1024,
        )
    )
    finished = threading.Event()

    def schedule() -> None:
        schedule_parse_lanes(
            [lane],
            AppSettings(enable_ocr=False),
            token,
            Path.cwd(),
        )
        finished.set()

    thread = threading.Thread(target=schedule)
    thread.start()
    try:
        assert finished.wait(timeout=0.2)
        assert len(lane.pending) == 1
        assert not lane.futures
    finally:
        token.resume()
        thread.join(timeout=1)
        executor.shutdown(wait=True, cancel_futures=True)


def test_job_completed_after_pause_satisfies_its_required_acknowledgement(
    tmp_path: Path,
) -> None:
    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        AppSettings(enable_ocr=False),
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future: Future[object] = Future()
    job = ParseJob(
        file_id=17,
        task_id=23,
        file_path=tmp_path / "completed.txt",
        parser_name="text",
        progress_phase="text_chunk",
        progress_completed=4,
        progress_total=4,
    )
    lane = ParseLane("normal", executor, 1, 1024, worker_count=1)
    lane.futures.add(future)
    lane.jobs[future] = job
    manager._pause_lanes = {"normal": lane}

    try:
        manager.request_pause()
        future.set_result(None)
        outcome = ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=[],
            parser_name="text",
            status="success",
            task_id=job.task_id,
            progress_phase="complete",
            progress_completed=4,
            progress_total=4,
        )
        manager._record_completed_pause_requirement(job, outcome)

        status = manager.pause_status()
        assert status["safe"] is True
        assert status["required_acknowledgements"] == 1
        assert status["received_acknowledgements"] == 1
        assert status["acknowledgements"][0]["completed_during_pause"] is True
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_pdf_pause_acknowledgement_uses_the_page_number_as_cursor(
    tmp_path: Path,
) -> None:
    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        AppSettings(enable_ocr=True),
    )
    job = ParseJob(
        file_id=19,
        task_id=29,
        file_path=tmp_path / "scanned.pdf",
        parser_name="pdf",
        pdf_page_number=8,
        pdf_task_type="pdf_ocr_page",
        progress_completed=2,
        progress_total=1,
    )
    outcome = ParseOutcome(
        file_id=job.file_id,
        file_path=job.file_path,
        blocks=[],
        parser_name="pdf",
        status="paused",
        task_id=job.task_id,
        progress_completed=2,
        progress_total=1,
        resume_cursor=0,
    )

    manager._record_pause_acknowledgement(job, outcome)

    acknowledgement = manager.pause_status()["acknowledgements"][0]
    assert acknowledgement["safe_unit_type"] == "pdf_ocr_page"
    assert acknowledgement["cursor"] == 8


def _hang_without_progress() -> None:
    threading.Event().wait(60)


def test_process_watchdog_terminates_a_worker_that_never_reports_progress() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spool = Path(tmp)
        executor = ProcessPoolExecutor(max_workers=1)
        job = ParseJob(
            file_id=9,
            file_path=Path("hung.txt"),
            lane="normal",
            started_monotonic=time.perf_counter(),
            last_progress_monotonic=time.perf_counter(),
        )
        future = executor.submit(_hang_without_progress)
        lane = ParseLane(
            "normal",
            executor,
            1,
            1024,
            process_based=True,
            worker_count=1,
        )
        lane.futures.add(future)
        lane.jobs[future] = job
        watchdog = ProcessLaneWatchdog(
            {"normal": lane},
            spool,
            AppSettings(normal_no_progress_timeout_seconds=1),
        )
        watchdog.start()
        try:
            deadline = time.monotonic() + 4
            while not job.watchdog_timed_out and time.monotonic() < deadline:
                time.sleep(0.05)
            assert job.watchdog_timed_out
            deadline = time.monotonic() + 2
            processes = list((getattr(executor, "_processes", None) or {}).values())
            while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
                time.sleep(0.05)
            assert not any(process.is_alive() for process in processes)
        finally:
            watchdog.stop()
            executor.shutdown(wait=False, cancel_futures=True)


def test_p1_03r_watchdog_emits_real_recovery_states_before_termination() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        spool = Path(tmp)
        executor = ProcessPoolExecutor(max_workers=1)
        job = ParseJob(
            file_id=10,
            file_path=Path("hung-diagnostic.txt"),
            lane="normal",
            started_monotonic=time.perf_counter(),
            last_progress_monotonic=time.perf_counter(),
            progress_phase="text_parse",
            progress_cursor="line:42",
        )
        future = executor.submit(_hang_without_progress)
        lane = ParseLane(
            "normal",
            executor,
            1,
            1024,
            process_based=True,
            worker_count=1,
        )
        lane.futures.add(future)
        lane.jobs[future] = job
        events: list[dict[str, object]] = []
        watchdog = ProcessLaneWatchdog(
            {"normal": lane},
            spool,
            AppSettings(normal_no_progress_timeout_seconds=1),
            diagnostic_callback=events.append,
        )
        watchdog.start()
        try:
            deadline = time.monotonic() + 4
            while len(events) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert [event["state"] for event in events[:2]] == [
                "reclaiming_no_progress",
                "terminating_worker",
            ]
            assert all(event["source"] == "process_lane_watchdog" for event in events)
            assert events[0]["phase"] == "text_parse"
            assert events[0]["cursor"] == "line:42"
        finally:
            watchdog.stop()
            executor.shutdown(wait=False, cancel_futures=True)


def test_safe_pause_can_rebuild_idle_pools_and_apply_new_mode() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manager = IndexManager(
            DatabaseManager(base / "index.db"),
            AppSettings(parser_workers=1),
            run_context={"execution_mode": "normal"},
        )
        old_executor = ThreadPoolExecutor(max_workers=1)
        new_executor = ThreadPoolExecutor(max_workers=2)
        current_lane = ParseLane(
            "normal",
            old_executor,
            1,
            1024,
            worker_count=1,
        )
        replacement_lane = ParseLane(
            "normal",
            new_executor,
            2,
            2048,
            worker_count=2,
        )
        manager._pause_lanes = {"normal": current_lane}
        manager._pause_executors = [old_executor]
        manager._pause_process_executors = []
        manager._pause_spool_dir = base
        manager._current_metrics = IndexRunMetrics(run_id="paused-switch")
        manager._current_metrics.eta_metrics[
            "_run_started_monotonic"
        ] = time.perf_counter() - 5
        manager._create_lanes = lambda _jobs, _spool: (  # type: ignore[method-assign]
            {"normal": replacement_lane},
            [new_executor],
            [],
        )

        applied = manager.apply_settings_while_paused(
            AppSettings(parser_workers=2),
            execution_mode="performance",
            effective_profile={"mode": "performance", "normal_workers": 2},
        )

        assert applied
        assert current_lane.executor is new_executor
        assert current_lane.worker_count == 2
        assert current_lane.max_in_flight == 2
        assert manager.run_context["execution_mode"] == "performance"
        assert manager._current_metrics.profile_transitions[-1]["to"] == "performance"
        assert (
            manager._current_metrics.eta_metrics["replay_events"][-1][
                "event_type"
            ]
            == "mode_switch"
        )
        new_executor.shutdown(wait=True, cancel_futures=True)


def test_u0_03r_pause_and_resume_reach_the_planning_control_domain(
    tmp_path: Path,
) -> None:
    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        AppSettings(),
    )
    planning_control = tmp_path / "planning-control"
    manager._planning_control_dir = planning_control

    manager.request_pause()
    assert pause_marker_path(planning_control).is_file()

    manager.request_resume()
    assert not pause_marker_path(planning_control).exists()


def test_u0_03r_planning_worker_blocks_idle_and_acknowledges_pause(
    tmp_path: Path,
) -> None:
    token = ProcessRunControlToken(
        tmp_path,
        pause_behavior="block",
    )
    request_process_pause(tmp_path)
    finished = threading.Event()
    worker = threading.Thread(
        target=lambda: (
            token.wait_if_paused(),
            finished.set(),
        ),
    )
    worker.start()
    deadline = time.monotonic() + 2
    acknowledgements: dict[int, dict[str, object]] = {}
    while time.monotonic() < deadline:
        acknowledgements = planning_pause_acknowledgements(tmp_path)
        if acknowledgements:
            break
        time.sleep(0.02)

    assert os.getpid() in acknowledgements
    assert acknowledgements[os.getpid()]["holds_external_process"] is False
    assert not finished.is_set()

    resume_processes(tmp_path)
    worker.join(timeout=2)
    assert finished.is_set()
    assert planning_pause_acknowledgements(tmp_path) == {}


def test_u0_03r_planning_acknowledgement_contains_the_real_safe_cursor(
    tmp_path: Path,
) -> None:
    token = ProcessRunControlToken(
        tmp_path,
        pause_behavior="block",
    )
    token.set_pause_checkpoint(
        task_id="planning:hash:17",
        safe_unit_type="content_hash",
        cursor="offset:1048576",
        checkpoint_version=1048576,
        checkpoint_checksum="sha256:prefix",
    )
    request_process_pause(tmp_path)
    worker = threading.Thread(target=token.wait_if_paused)
    worker.start()
    deadline = time.monotonic() + 2
    acknowledgement: dict[str, object] = {}
    while time.monotonic() < deadline:
        acknowledgement = planning_pause_acknowledgements(
            tmp_path
        ).get(os.getpid(), {})
        if acknowledgement:
            break
        time.sleep(0.02)

    assert acknowledgement["task_id"] == "planning:hash:17"
    assert acknowledgement["safe_unit_type"] == "content_hash"
    assert acknowledgement["cursor"] == "offset:1048576"
    assert acknowledgement["checkpoint_version"] == 1048576
    assert (
        acknowledgement["checkpoint_checksum"]
        == "sha256:prefix"
    )

    resume_processes(tmp_path)
    worker.join(timeout=2)


def test_u0_03r_manager_accepts_an_idle_acknowledged_planning_worker(
    tmp_path: Path,
) -> None:
    class ActivePlanningRunner:
        @property
        def active_pids(self) -> tuple[int, ...]:
            return (43210,)

    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        AppSettings(enable_ocr=False),
    )
    manager._pause_lanes = {}
    manager._planning_runner = ActivePlanningRunner()  # type: ignore[assignment]
    manager._planning_control_dir = tmp_path / "planning-control"
    acknowledgement_dir = (
        manager._planning_control_dir / ".planning_pause_acknowledgements"
    )
    acknowledgement_dir.mkdir(parents=True)
    (acknowledgement_dir / "43210.json").write_text(
        json.dumps(
            {
                "task_id": "planning:43210",
                "worker_pid": 43210,
                "safe_unit_type": "planning_safe_point",
                "cursor": "file:17",
                "checkpoint_version": 17,
                "checkpoint_checksum": "checkpoint",
                "returned_at_epoch": time.time(),
                "holds_external_process": False,
            }
        ),
        encoding="utf-8",
    )
    manager._pause_state = "pausing"

    assert manager.is_safely_paused() is True
    status = manager.pause_status()
    assert status["state"] == "paused"
    assert status["planning_required_acknowledgements"] == 1
    assert status["planning_received_acknowledgements"] == 1
    assert status["planning_acknowledgements"] == [
        {
            "task_id": "planning:43210",
            "worker_pid": 43210,
            "safe_unit_type": "planning_safe_point",
            "cursor": "file:17",
            "checkpoint_version": 17,
            "checkpoint_checksum": "checkpoint",
            "returned_at_epoch": status["planning_acknowledgements"][0][
                "returned_at_epoch"
            ],
            "holds_external_process": False,
        }
    ]


def test_safe_pause_waits_for_every_active_worker_acknowledgement() -> None:
    from concurrent.futures import Future

    manager = IndexManager(
        DatabaseManager(Path(tempfile.mkdtemp()) / "index.db"),
        AppSettings(enable_ocr=False),
    )
    executor = ThreadPoolExecutor(max_workers=1)
    lane = ParseLane(
        "normal",
        executor,
        1,
        1024,
        worker_count=1,
    )
    future: Future[object] = Future()
    future.set_running_or_notify_cancel()
    job = ParseJob(
        file_id=17,
        task_id=170,
        file_path=Path("active.txt"),
        parser_name="text",
    )
    lane.futures.add(future)  # type: ignore[arg-type]
    lane.jobs[future] = job  # type: ignore[index]
    manager._pause_lanes = {"normal": lane}
    manager.request_pause()
    future.set_result(object())

    try:
        assert manager.is_safely_paused() is False
        manager._record_pause_acknowledgement(
            job,
            ParseOutcome(
                file_id=17,
                file_path=Path("active.txt"),
                blocks=[],
                parser_name="text",
                status="paused",
                worker_pid=1234,
            ),
        )
        assert manager.is_safely_paused() is True
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_u0_02v_pause_and_resume_are_written_to_eta_replay_trace() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        manager = IndexManager(
            DatabaseManager(Path(tmp) / "index.db"),
            AppSettings(),
        )
        manager._current_metrics = IndexRunMetrics(run_id="pause-trace")
        manager._current_metrics.eta_metrics[
            "_run_started_monotonic"
        ] = time.perf_counter() - 3
        manager._pause_lanes = {}

        manager.request_pause()
        manager.request_resume()

        events = manager._current_metrics.eta_metrics["replay_events"]
        assert [event["event_type"] for event in events] == ["pause", "resume"]
        assert events[0]["at_seconds"] >= 3


def test_metric_01r_resume_records_the_actual_paused_resource_window(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    manager = IndexManager(database, AppSettings(enable_ocr=False))
    manager._pause_lanes = {}
    manager._current_metrics = IndexRunMetrics(
        run_id="paused-resource-window"
    )

    manager.request_pause()
    assert manager.is_safely_paused() is True
    time.sleep(0.08)
    manager.request_resume()

    pause_metrics = manager._current_metrics.pause_metrics
    assert pause_metrics["paused_observation_seconds"] >= 0.05
    assert pause_metrics["paused_observation_progress_delta"] == 0
    assert pause_metrics["paused_read_bytes_delta"] >= 0
    assert pause_metrics["paused_database_write_count"] == 0
    assert pause_metrics["paused_cpu_average"] >= 0


def test_paused_mode_switch_failure_rolls_back_and_stays_paused() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        original_settings = AppSettings(parser_workers=1)
        manager = IndexManager(
            DatabaseManager(base / "index.db"),
            original_settings,
            run_context={"execution_mode": "normal"},
        )
        old_executor = ThreadPoolExecutor(max_workers=1)
        lane = ParseLane(
            "normal",
            old_executor,
            1,
            1024,
            worker_count=1,
        )
        manager._pause_lanes = {"normal": lane}
        manager._pause_executors = [old_executor]
        manager._pause_process_executors = []
        manager._pause_spool_dir = base
        manager._pause_state = "paused"
        manager._current_metrics = IndexRunMetrics(run_id="rollback")

        def fail_candidate(
            _jobs: list[ParseJob],
            _spool: Path,
        ) -> tuple[dict[str, ParseLane], list[object], list[object]]:
            raise RuntimeError("injected pool startup failure")

        manager._create_lanes = fail_candidate  # type: ignore[method-assign]

        applied = manager.apply_settings_while_paused(
            AppSettings(parser_workers=2),
            execution_mode="performance",
            effective_profile={"mode": "performance"},
        )

        assert applied is False
        assert manager.settings is original_settings
        assert manager.run_context["execution_mode"] == "normal"
        assert manager.pause_status()["state"] == "paused"
        assert (
            manager._current_metrics.pause_metrics[
                "mode_switch_failure_count"
            ]
            == 1
        )
        assert (
            manager._current_metrics.pause_metrics[
                "mode_switch_rollback_count"
            ]
            == 1
        )
        old_executor.shutdown(wait=True, cancel_futures=True)
