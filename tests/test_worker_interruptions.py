from __future__ import annotations

from concurrent.futures import Future
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import (
    ParseJob,
    ParseLane,
    drain_completed_lanes,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.workers.scan_worker import ScanWorker


def test_keyboard_interrupt_from_child_future_becomes_retryable_task_result() -> None:
    future: Future[object] = Future()
    future.set_exception(KeyboardInterrupt())
    job = ParseJob(file_id=7, file_path=Path("interrupted.png"), lane="ocr")
    lane = ParseLane(
        name="ocr",
        executor=object(),  # type: ignore[arg-type]
        max_in_flight=1,
        max_inflight_bytes=1024,
        process_based=True,
    )
    lane.futures.add(future)  # type: ignore[arg-type]
    lane.jobs[future] = job  # type: ignore[index]

    completed = drain_completed_lanes(
        [lane],
        CancelToken(),
        Path("."),
    )

    assert len(completed) == 1
    _lane_name, result_job, result, _descriptor_bytes = completed[0]
    assert result_job is job
    assert result.error_code == "WORKER_INTERRUPTED"
    assert result.status == "failed_retryable"
    assert not lane.futures
    assert not lane.jobs


def test_system_exit_from_child_future_does_not_escape_scheduler() -> None:
    future: Future[object] = Future()
    future.set_exception(SystemExit("child exited"))
    job = ParseJob(file_id=8, file_path=Path("interrupted.pdf"), lane="pdf")
    lane = ParseLane(
        name="pdf",
        executor=object(),  # type: ignore[arg-type]
        max_in_flight=1,
        max_inflight_bytes=1024,
        process_based=True,
    )
    lane.futures.add(future)  # type: ignore[arg-type]
    lane.jobs[future] = job  # type: ignore[index]

    completed = drain_completed_lanes([lane], CancelToken(), Path("."))

    assert completed[0][2].error_code == "WORKER_INTERRUPTED"


def test_scan_worker_terminal_signal_is_emitted_once_after_cleanup_boundary() -> None:
    worker = ScanWorker(Path("scan.db"), AppSettings(enable_ocr=False))
    finished: list[object] = []
    failed: list[object] = []
    cancelled: list[object] = []
    ended: list[object] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.cancelled.connect(cancelled.append)
    worker.ended.connect(ended.append)
    worker._terminal_status = "cancelled"
    worker._terminal_payload = {
        "status": "cancelled",
        "cancellation_source": "界面取消",
    }

    worker._emit_terminal_signals()
    worker._emit_terminal_signals()

    assert finished == []
    assert failed == []
    assert len(cancelled) == 1
    assert len(ended) == 1
    assert ended[0]["cancellation_source"] == "界面取消"  # type: ignore[index]
