from __future__ import annotations

import tempfile
from concurrent.futures import Future
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import (
    ParseJob,
    ParseLane,
    _new_process_executor,
    cpu_tokens_for_job,
    parse_pdf_batch_process_worker,
    schedule_parse_lanes,
    should_lower_process_priority,
)
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.core.task_manager import CancelToken


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls = 0
        self.submissions: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, *args: object, **kwargs: object) -> Future[object]:
        self.calls += 1
        self.submissions.append((args[0], tuple(args[1:])))
        return Future()


def job(file_id: int, size_mb: int, lane: str) -> ParseJob:
    return ParseJob(
        file_id=file_id,
        file_path=Path(f"file-{file_id}.txt"),
        lane=lane,
        size_bytes=size_mb * 1024 * 1024,
    )


def test_lane_and_global_byte_budgets_limit_submission() -> None:
    first_executor = RecordingExecutor()
    second_executor = RecordingExecutor()
    first = ParseLane("normal", first_executor, 4, 200 * 1024 * 1024)
    second = ParseLane("ocr", second_executor, 4, 200 * 1024 * 1024)
    first.pending.extend([job(1, 100, "normal"), job(2, 100, "normal")])
    second.pending.append(job(3, 100, "ocr"))
    settings = AppSettings(index_memory_budget_mb=128)

    with tempfile.TemporaryDirectory() as tmp:
        schedule_parse_lanes(
            [first, second],
            settings,
            CancelToken(),
            Path(tmp),
        )

    assert first_executor.calls == 1
    assert second_executor.calls == 0
    assert first.inflight_bytes <= 128 * 1024 * 1024


def test_process_lanes_sharing_executor_do_not_prefetch_behind_each_other() -> None:
    executor = RecordingExecutor()
    zip_lane = ParseLane("zip", executor, 1, 200 * 1024 * 1024, process_based=True)
    ocr_lane = ParseLane("ocr", executor, 1, 200 * 1024 * 1024, process_based=True)
    zip_lane.pending.append(job(1, 10, "zip"))
    ocr_lane.pending.append(job(2, 10, "ocr"))

    with tempfile.TemporaryDirectory() as tmp:
        schedule_parse_lanes(
            [zip_lane, ocr_lane],
            AppSettings(),
            CancelToken(),
            Path(tmp),
        )

    assert executor.calls == 1
    assert len(zip_lane.futures) == 1
    assert len(ocr_lane.futures) == 0


def test_cpu_token_budget_accounts_for_ocr_threads() -> None:
    normal_executor = RecordingExecutor()
    ocr_executor = RecordingExecutor()
    normal = ParseLane("normal", normal_executor, 1, 200 * 1024 * 1024)
    ocr = ParseLane("ocr", ocr_executor, 1, 200 * 1024 * 1024, process_based=True)
    normal.pending.append(job(1, 10, "normal"))
    ocr.pending.append(job(2, 10, "ocr"))
    settings = AppSettings(index_cpu_token_budget=2, ocr_cpu_threads=2)

    with tempfile.TemporaryDirectory() as tmp:
        schedule_parse_lanes([normal, ocr], settings, CancelToken(), Path(tmp))

    assert normal_executor.calls == 1
    assert ocr_executor.calls == 0


def test_persistent_ocr_executor_is_not_recycled_by_task_count() -> None:
    executor = _new_process_executor(
        AppSettings(process_max_tasks_per_child=16),
        1,
        persistent=True,
    )
    try:
        assert executor._max_tasks_per_child is None
    finally:
        executor.shutdown(wait=True, cancel_futures=True)


def test_performance_pdf_pages_are_submitted_as_one_source_batch() -> None:
    executor = RecordingExecutor()
    lane = ParseLane(
        "pdf",
        executor,
        1,
        1024 * 1024 * 1024,
        process_based=True,
        worker_count=1,
    )
    for page_number in range(1, 6):
        lane.pending.append(
            ParseJob(
                file_id=7,
                file_path=Path("large.pdf"),
                task_id=100 + page_number,
                lane="pdf",
                size_bytes=1024,
                memory_estimate_bytes=4096,
                pdf_document_task_id=55,
                pdf_page_number=page_number,
                pdf_task_type="pdf_native_page",
            )
        )
    settings = AppSettings(
        index_performance_preset="fastest",
        pdf_page_batch_size=4,
    )
    metrics = IndexRunMetrics(run_id="pdf-batch")

    with tempfile.TemporaryDirectory() as tmp:
        submitted = schedule_parse_lanes(
            [lane],
            settings,
            CancelToken(),
            Path(tmp),
            metrics=metrics,
        )

    assert submitted == [101, 102, 103, 104]
    assert executor.calls == 1
    assert executor.submissions[0][0] is parse_pdf_batch_process_worker
    leader = executor.submissions[0][1][0]
    assert [leader.pdf_page_number, *[job.pdf_page_number for job in leader.batch_jobs]] == [
        1,
        2,
        3,
        4,
    ]
    assert [job.pdf_page_number for job in lane.pending] == [5]
    assert metrics.pdf_metrics["pdf_dispatch_batch_count"] == 1
    assert metrics.pdf_metrics["pdf_dispatched_page_count"] == 4
    assert metrics.pdf_metrics["pdf_max_batch_pages"] == 4


def test_native_pdf_page_does_not_reserve_ocr_cpu_threads() -> None:
    settings = AppSettings(ocr_cpu_threads=4, enable_ocr=True, ocr_scanned_pdf=True)
    native_page = ParseJob(
        file_id=1,
        file_path=Path("native.pdf"),
        lane="pdf",
        pdf_task_type="pdf_native_page",
        pdf_page_number=1,
    )
    fallback_document = ParseJob(
        file_id=2,
        file_path=Path("fallback.pdf"),
        lane="pdf",
    )

    assert cpu_tokens_for_job(native_page, settings) == 1
    assert cpu_tokens_for_job(fallback_document, settings) == 4


def test_performance_workers_keep_normal_process_priority() -> None:
    assert should_lower_process_priority(
        AppSettings(index_performance_preset="fastest")
    ) is False
    assert should_lower_process_priority(
        AppSettings(index_performance_preset="balanced")
    ) is True
