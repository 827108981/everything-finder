from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from local_full_text_search.core.errors import PlanningNoProgressError
from local_full_text_search.core.planning_worker import (
    PlanningProgressReporter,
    RecoverablePlanningRunner,
)
from local_full_text_search.core.run_control import (
    request_process_pause,
    resume_processes,
)
from local_full_text_search.core.task_manager import ProcessRunControlToken


def _hang_without_progress(
    _reporter: PlanningProgressReporter,
    child_pid_path: Path | None = None,
) -> None:
    child: subprocess.Popen[str] | None = None
    if child_pid_path is not None:
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            text=True,
        )
        child_pid_path.write_text(str(child.pid), encoding="ascii")
    try:
        time.sleep(60)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()


def _repeat_same_progress(reporter: PlanningProgressReporter) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        reporter.advance(
            phase="content_hash",
            completed=1,
            total=10,
            cursor="offset:4096",
            bytes_read=4096,
        )
        time.sleep(0.03)


def _healthy_long_progress(
    reporter: PlanningProgressReporter,
    duration: float,
) -> int:
    started = time.monotonic()
    completed = 0
    while time.monotonic() - started < duration:
        completed += 1
        reporter.advance(
            phase="content_hash",
            completed=completed,
            total=1000,
            cursor=f"offset:{completed * 4096}",
            bytes_read=completed * 4096,
        )
        time.sleep(0.04)
    return completed


def _healthy_follow_up(
    reporter: PlanningProgressReporter,
    value: int,
) -> int:
    reporter.advance(
        phase="file_stat",
        completed=1,
        total=1,
        cursor="complete",
    )
    return value * 2


def _report_pid(reporter: PlanningProgressReporter) -> int:
    reporter.advance(
        phase="file_stat",
        completed=1,
        total=1,
        cursor="complete",
    )
    return os.getpid()


def _stream_batches(
    reporter: PlanningProgressReporter,
    count: int,
    delay_seconds: float,
):
    for index in range(count):
        reporter.advance(
            phase="directory_enumeration",
            completed=index + 1,
            total=count,
            cursor=f"batch:{index + 1}",
        )
        yield [index]
        time.sleep(delay_seconds)


def _stream_then_hang(reporter: PlanningProgressReporter):
    reporter.advance(
        phase="directory_enumeration",
        completed=1,
        total=2,
        cursor="batch:1",
    )
    yield ["first"]
    time.sleep(60)


def _pause_aware_planning_task(
    reporter: PlanningProgressReporter,
    pause_control_dir: Path,
) -> str:
    token = ProcessRunControlToken(
        pause_control_dir,
        pause_behavior="block",
    )
    token.set_pause_checkpoint(
        task_id="planning:test",
        safe_unit_type="content_hash",
        cursor="offset:1048576",
        checkpoint_version=1048576,
        checkpoint_checksum="prefix",
    )
    reporter.advance(
        phase="content_hash",
        completed=1,
        total=2,
        cursor="offset:1048576",
        bytes_read=1048576,
        checkpoint_version=1048576,
    )
    token.wait_if_paused()
    reporter.advance(
        phase="content_hash",
        completed=2,
        total=2,
        cursor="offset:2097152",
        bytes_read=2097152,
        checkpoint_version=2097152,
    )
    return "continued"


def test_u0_03r_planning_watchdog_freezes_during_user_pause(
    tmp_path: Path,
) -> None:
    pause_control = tmp_path / "pause-control"
    runner = RecoverablePlanningRunner(
        tmp_path / "tasks",
        no_progress_timeout_seconds=0.20,
        startup_timeout_seconds=2,
        poll_interval_seconds=0.02,
        pause_control_dir=pause_control,
    )
    request_process_pause(pause_control)
    result: list[str] = []
    errors: list[BaseException] = []

    def execute() -> None:
        try:
            result.append(
                runner.run(
                    "pause-aware",
                    _pause_aware_planning_task,
                    pause_control,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=execute)
    thread.start()
    time.sleep(0.65)
    assert thread.is_alive()
    assert not errors

    resume_processes(pause_control)
    thread.join(timeout=3)
    assert result == ["continued"]
    assert not errors


def test_s0_01r_planning_hang_is_terminated_with_its_child_tree(
    tmp_path: Path,
) -> None:
    child_pid_path = tmp_path / "child.pid"
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.35,
        poll_interval_seconds=0.02,
    )

    with pytest.raises(PlanningNoProgressError):
        runner.run(
            "planning-hang",
            _hang_without_progress,
            child_pid_path,
        )

    child_pid = int(child_pid_path.read_text(encoding="ascii"))
    deadline = time.monotonic() + 2
    while psutil.pid_exists(child_pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not psutil.pid_exists(child_pid)
    assert runner.active_pids == ()


def test_s0_01r_duplicate_progress_does_not_refresh_timeout(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.30,
        poll_interval_seconds=0.02,
    )

    started = time.monotonic()
    with pytest.raises(PlanningNoProgressError):
        runner.run("duplicate-progress", _repeat_same_progress)
    elapsed = time.monotonic() - started

    assert elapsed < 1.2


def test_s0_01r_healthy_long_task_survives_total_duration(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.20,
        poll_interval_seconds=0.02,
    )

    completed = runner.run(
        "healthy-long-task",
        _healthy_long_progress,
        0.75,
    )

    assert completed >= 10
    assert runner.active_pids == ()


def test_s0_03r_runner_accepts_healthy_work_after_recycling_hang(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.25,
        poll_interval_seconds=0.02,
    )

    with pytest.raises(PlanningNoProgressError):
        runner.run("first-hang", _hang_without_progress)

    assert runner.run("healthy-after-recycle", _healthy_follow_up, 21) == 42
    assert runner.active_pids == ()


def test_s0_03r_planning_task_runs_outside_parent_process(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=1,
        poll_interval_seconds=0.02,
    )

    assert runner.run("pid-check", _report_pid) != os.getpid()


def test_s0_01r_stream_returns_batches_before_discovery_finishes(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.5,
        poll_interval_seconds=0.02,
    )

    started = time.monotonic()
    stream = runner.stream("directory-stream", _stream_batches, 3, 0.15)
    assert next(stream) == [0]
    first_elapsed = time.monotonic() - started
    assert first_elapsed < 0.6
    assert list(stream) == [[1], [2]]
    assert runner.active_pids == ()


def test_s0_03r_stream_hang_is_recycled_after_last_semantic_batch(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=0.25,
        poll_interval_seconds=0.02,
    )

    stream = runner.stream("directory-stream-hang", _stream_then_hang)
    assert next(stream) == ["first"]
    with pytest.raises(PlanningNoProgressError):
        next(stream)
    assert runner.active_pids == ()


def test_s0_01r_force_cancel_terminates_active_planning_process(
    tmp_path: Path,
) -> None:
    runner = RecoverablePlanningRunner(
        tmp_path / "control",
        no_progress_timeout_seconds=30,
        startup_timeout_seconds=2,
        poll_interval_seconds=0.02,
    )
    finished = threading.Event()

    def execute() -> None:
        try:
            runner.run("force-cancel", _hang_without_progress)
        except BaseException:
            pass
        finally:
            finished.set()

    thread = threading.Thread(target=execute)
    thread.start()
    deadline = time.monotonic() + 3
    while not runner.active_pids and time.monotonic() < deadline:
        time.sleep(0.02)
    assert runner.active_pids

    runner.cancel_active()

    assert finished.wait(3)
    thread.join(timeout=1)
    assert runner.active_pids == ()
