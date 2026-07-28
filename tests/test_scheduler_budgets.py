from __future__ import annotations

import tempfile
from concurrent.futures import Future
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import ParseJob, ParseLane, schedule_parse_lanes
from local_full_text_search.core.task_manager import CancelToken


class RecordingExecutor:
    def __init__(self) -> None:
        self.calls = 0

    def submit(self, *args: object, **kwargs: object) -> Future[object]:
        self.calls += 1
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
