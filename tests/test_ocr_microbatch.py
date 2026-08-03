from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from concurrent.futures import Executor, Future
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import (
    ParseJob,
    ParseLane,
    ParseOutcome,
    parse_ocr_batch_process_worker,
    schedule_parse_lanes,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.core.errors import PauseRequestedError
from local_full_text_search.ocr.microbatch import (
    MicroBatchPlanner,
    OcrInferenceRequest,
    RecognitionMicroBatchCoordinator,
)
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.parser_registry import ParserRegistry
from local_full_text_search.parsers.pdf_parser import PdfParser


def _request(
    request_id: str,
    *,
    source: str,
    language: str = "ch",
    pixels: int = 100,
    submitted_ms: int = 0,
) -> OcrInferenceRequest:
    return OcrInferenceRequest(
        request_id=request_id,
        parent_task_id=source,
        region_id=request_id,
        source_order=0,
        page_number=1,
        region_top=0,
        region_left=0,
        language=language,
        model_fingerprint="model-v1",
        preprocess_fingerprint="prep-v1",
        pixel_count=pixels,
        memory_bytes=pixels * 4,
        submitted_ms=submitted_ms,
        payload=request_id,
    )


def test_p0_04r_microbatch_combines_compatible_cross_source_requests() -> None:
    planner = MicroBatchPlanner(
        max_requests=4,
        max_pixels=1_000,
        max_memory_bytes=10_000,
        max_wait_ms=40,
    )
    planner.submit(_request("pdf-page", source="pdf"))
    planner.submit(_request("image", source="image"))

    batch = planner.next_batch(now_ms=40)

    assert [request.request_id for request in batch] == [
        "pdf-page",
        "image",
    ]
    assert planner.metrics["inference_calls"] == 1
    assert planner.metrics["requests"] == 2


def test_p0_04r_microbatch_separates_language_and_oversize_request() -> None:
    planner = MicroBatchPlanner(
        max_requests=8,
        max_pixels=500,
        max_memory_bytes=5_000,
        max_wait_ms=20,
    )
    planner.submit(_request("ch", source="one", language="ch", pixels=100))
    planner.submit(_request("en", source="two", language="en", pixels=100))
    planner.submit(_request("huge", source="three", pixels=2_000))

    first = planner.next_batch(now_ms=20)
    second = planner.next_batch(now_ms=20)
    third = planner.next_batch(now_ms=20)

    assert len(first) == 1
    assert len(second) == 1
    assert len(third) == 1
    assert {first[0].request_id, second[0].request_id, third[0].request_id} == {
        "ch",
        "en",
        "huge",
    }
    assert planner.metrics["oversize_single_count"] == 1


def test_p0_04r_cancel_and_stable_output_order() -> None:
    planner = MicroBatchPlanner(
        max_requests=8,
        max_pixels=10_000,
        max_memory_bytes=100_000,
        max_wait_ms=20,
    )
    planner.submit(_request("later", source="one"))
    planner.submit(_request("cancelled", source="two"))
    planner.submit(_request("earlier", source="three"))
    planner.cancel("cancelled")
    batch = planner.next_batch(now_ms=20)
    results = {
        "later": "B",
        "earlier": "A",
    }

    ordered = planner.stable_results(batch, results)

    assert [item[0].request_id for item in ordered] == ["earlier", "later"]
    assert planner.metrics["cancelled_before_batch_count"] == 1


def test_p0_04r_runtime_coordinator_batches_two_live_sources() -> None:
    inference_calls: list[list[str]] = []

    def infer(crops: list[object]) -> list[tuple[str, float]]:
        values = [str(crop) for crop in crops]
        inference_calls.append(values)
        return [(value.upper(), 0.9) for value in values]

    coordinator = RecognitionMicroBatchCoordinator(
        infer,
        max_requests=8,
        max_pixels=10_000,
        max_memory_bytes=100_000,
        max_wait_ms=40,
    )
    barrier = threading.Barrier(3)
    results: dict[str, list[tuple[str, float | None]]] = {}

    def submit(source: str) -> None:
        barrier.wait()
        results[source] = coordinator.recognize(
            [source],
            pixel_counts=[100],
            compatibility_key=("ch", "model", "prep"),
        )

    first = threading.Thread(target=submit, args=("pdf-page",))
    second = threading.Thread(target=submit, args=("image",))
    first.start()
    second.start()
    barrier.wait()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive() and not second.is_alive()
    assert len(inference_calls) == 1
    assert set(inference_calls[0]) == {"pdf-page", "image"}
    assert results["pdf-page"] == [("PDF-PAGE", 0.9)]
    assert results["image"] == [("IMAGE", 0.9)]
    assert coordinator.metrics["recognize_requests"] == 2
    assert coordinator.metrics["recognize_inference_calls"] == 1
    assert coordinator.metrics["recognize_batch_count"] == 1
    assert coordinator.metrics["recognize_average_batch_size"] == 2.0


def test_p0_04r_pause_before_claim_removes_only_that_live_request() -> None:
    inference_calls: list[list[str]] = []

    def infer(crops: list[object]) -> list[str]:
        values = [str(crop) for crop in crops]
        inference_calls.append(values)
        return values

    coordinator = RecognitionMicroBatchCoordinator(
        infer,
        max_requests=8,
        max_pixels=10_000,
        max_memory_bytes=100_000,
        max_wait_ms=50,
    )
    pause_requested = threading.Event()
    barrier = threading.Barrier(3)
    observed: dict[str, object] = {}

    def submit_paused() -> None:
        def check() -> None:
            if pause_requested.is_set():
                raise PauseRequestedError("pause")

        barrier.wait()
        try:
            coordinator.recognize(
                ["paused"],
                pixel_counts=[100],
                compatibility_key=("ch", "model", "prep"),
                cancel_check=check,
            )
        except BaseException as exc:
            observed["paused_error"] = type(exc)

    def submit_healthy() -> None:
        barrier.wait()
        observed["healthy"] = coordinator.recognize(
            ["healthy"],
            pixel_counts=[100],
            compatibility_key=("ch", "model", "prep"),
        )

    paused = threading.Thread(target=submit_paused)
    healthy = threading.Thread(target=submit_healthy)
    paused.start()
    healthy.start()
    barrier.wait()
    time.sleep(0.01)
    pause_requested.set()
    paused.join(timeout=2)
    healthy.join(timeout=2)

    assert observed["paused_error"] is PauseRequestedError
    assert observed["healthy"] == ["healthy"]
    assert inference_calls == [["healthy"]]
    assert coordinator.metrics["cancelled_before_batch_count"] == 1


def test_p0_02r_parser_registries_can_share_one_live_ocr_engine() -> None:
    sentinel = object()
    first = ParserRegistry(AppSettings(), shared_ocr=sentinel)
    second = ParserRegistry(AppSettings(), shared_ocr=sentinel)
    ocr_parsers = [
        parser
        for registry in (first, second)
        for parser in registry.parsers
        if isinstance(parser, (ImageParser, PdfParser))
    ]

    assert ocr_parsers
    assert all(parser.ocr is sentinel for parser in ocr_parsers)


class _RecordingExecutor(Executor):
    def __init__(self) -> None:
        self.calls: list[tuple[object, tuple[object, ...]]] = []

    def submit(self, fn: object, /, *args: object, **kwargs: object) -> Future:
        future: Future = Future()
        self.calls.append((fn, args))
        return future


def test_p0_04r_scheduler_submits_cross_source_ocr_jobs_as_one_runtime_batch(
    tmp_path: Path,
) -> None:
    executor = _RecordingExecutor()
    jobs = [
        ParseJob(
            file_id=1,
            file_path=tmp_path / "page.pdf",
            task_id=11,
            lane="ocr",
            memory_estimate_bytes=100,
            pdf_page_number=1,
            pdf_source_digest="pdf-source",
        ),
        ParseJob(
            file_id=2,
            file_path=tmp_path / "image.jpg",
            task_id=12,
            lane="ocr",
            memory_estimate_bytes=100,
        ),
    ]
    lane = ParseLane(
        "ocr",
        executor,
        max_in_flight=1,
        max_inflight_bytes=10_000,
        process_based=True,
        worker_count=1,
    )
    lane.pending.extend(jobs)

    task_ids = schedule_parse_lanes(
        [lane],
        AppSettings(
            ocr_workers=1,
            ocr_microbatch_parent_jobs=4,
            index_memory_budget_mb=128,
        ),
        CancelToken(),
        tmp_path,
    )

    assert task_ids == [11, 12]
    assert len(executor.calls) == 1
    assert executor.calls[0][0] is parse_ocr_batch_process_worker
    leader = lane.jobs[next(iter(lane.jobs))]
    assert [leader.file_id, *[job.file_id for job in leader.batch_jobs]] == [
        1,
        2,
    ]


def test_p0_04r_scheduler_can_batch_two_pages_from_one_pdf(
    tmp_path: Path,
) -> None:
    executor = _RecordingExecutor()
    jobs = [
        ParseJob(
            file_id=1,
            file_path=tmp_path / "manual.pdf",
            task_id=page,
            lane="ocr",
            memory_estimate_bytes=100,
            pdf_page_number=page,
            pdf_source_digest="same-pdf",
        )
        for page in (1, 2)
    ]
    lane = ParseLane(
        "ocr",
        executor,
        max_in_flight=1,
        max_inflight_bytes=10_000,
        process_based=True,
        worker_count=1,
    )
    lane.pending.extend(jobs)

    task_ids = schedule_parse_lanes(
        [lane],
        AppSettings(
            ocr_workers=1,
            ocr_microbatch_parent_jobs=4,
            index_memory_budget_mb=128,
        ),
        CancelToken(),
        tmp_path,
    )

    assert task_ids == [1, 2]
    leader = lane.jobs[next(iter(lane.jobs))]
    assert [leader.pdf_page_number, leader.batch_jobs[0].pdf_page_number] == [
        1,
        2,
    ]


def test_p0_04r_process_batch_runs_parents_concurrently_with_one_shared_engine(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    import local_full_text_search.core.index_manager as index_manager

    sentinel = object()
    leader = ParseJob(
        file_id=1,
        file_path=tmp_path / "page.pdf",
        lane="ocr",
    )
    image = ParseJob(
        file_id=2,
        file_path=tmp_path / "image.jpg",
        lane="ocr",
    )
    leader.batch_jobs = (image,)
    barrier = threading.Barrier(2)
    observed_engines: list[object] = []

    def fake_parse(
        job: ParseJob,
        registry: ParserRegistry,
        _token: object,
        _settings: AppSettings,
        **_kwargs: object,
    ) -> ParseOutcome:
        observed_engines.append(registry.shared_ocr)
        barrier.wait(timeout=1)
        return ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=[],
            parser_name="fake_ocr",
            status="success",
            lane="ocr",
        )

    monkeypatch.setattr(
        index_manager,
        "_process_registry",
        SimpleNamespace(shared_ocr=sentinel),
    )
    monkeypatch.setattr(
        index_manager,
        "parse_file_with_registry",
        fake_parse,
    )

    results = parse_ocr_batch_process_worker(
        leader,
        AppSettings(ocr_microbatch_parent_jobs=2),
        tmp_path,
    )

    assert [result.file_id for result in results] == [1, 2]
    assert observed_engines == [sentinel, sentinel]
    assert len({result.worker_pid for result in results}) == 1
