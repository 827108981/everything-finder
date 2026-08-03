from __future__ import annotations

import math
import statistics
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass(frozen=True, slots=True)
class OcrInferenceRequest:
    request_id: str
    parent_task_id: str
    region_id: str
    source_order: int
    page_number: int
    region_top: int
    region_left: int
    language: str
    model_fingerprint: str
    preprocess_fingerprint: str
    pixel_count: int
    memory_bytes: int
    submitted_ms: int
    payload: Any

    @property
    def compatibility_key(self) -> tuple[str, str, str]:
        return (
            self.language,
            self.model_fingerprint,
            self.preprocess_fingerprint,
        )

    @property
    def stable_key(self) -> tuple[int, int, int, int, str]:
        return (
            int(self.source_order),
            int(self.page_number),
            int(self.region_top),
            int(self.region_left),
            self.region_id,
        )


class MicroBatchPlanner:
    """Deterministic bounded planner shared by detection and recognition."""

    def __init__(
        self,
        *,
        max_requests: int,
        max_pixels: int,
        max_memory_bytes: int,
        max_wait_ms: int,
    ) -> None:
        self.max_requests = max(1, int(max_requests))
        self.max_pixels = max(1, int(max_pixels))
        self.max_memory_bytes = max(1, int(max_memory_bytes))
        self.max_wait_ms = max(0, int(max_wait_ms))
        self._pending: list[OcrInferenceRequest] = []
        self._cancelled: set[str] = set()
        self.metrics: dict[str, int] = {
            "requests": 0,
            "inference_calls": 0,
            "batch_count": 0,
            "batch_items": 0,
            "pixels": 0,
            "oversize_single_count": 0,
            "cancelled_before_batch_count": 0,
        }

    def submit(self, request: OcrInferenceRequest) -> None:
        if any(item.request_id == request.request_id for item in self._pending):
            raise ValueError(f"Duplicate inference request: {request.request_id}")
        self._pending.append(request)
        self.metrics["requests"] += 1

    def cancel(self, request_id: str) -> None:
        self._cancelled.add(str(request_id))

    def next_batch(self, *, now_ms: int) -> list[OcrInferenceRequest]:
        retained: list[OcrInferenceRequest] = []
        for request in self._pending:
            if request.request_id in self._cancelled:
                self.metrics["cancelled_before_batch_count"] += 1
                continue
            retained.append(request)
        self._pending = retained
        if not self._pending:
            return []
        oldest = min(
            self._pending,
            key=lambda item: item.submitted_ms,
        )
        compatible = [
            request
            for request in self._pending
            if request.compatibility_key == oldest.compatibility_key
        ]
        compatible.sort(key=lambda item: item.submitted_ms)
        oldest_wait = max(0, int(now_ms) - int(oldest.submitted_ms))
        capacity_triggered = len(compatible) >= self.max_requests
        if oldest_wait < self.max_wait_ms and not capacity_triggered:
            return []
        first = compatible[0]
        if (
            first.pixel_count > self.max_pixels
            or first.memory_bytes > self.max_memory_bytes
        ):
            batch = [first]
            self.metrics["oversize_single_count"] += 1
        else:
            batch = []
            pixels = 0
            memory = 0
            for request in compatible:
                if len(batch) >= self.max_requests:
                    break
                next_pixels = pixels + max(1, int(request.pixel_count))
                next_memory = memory + max(1, int(request.memory_bytes))
                if batch and (
                    next_pixels > self.max_pixels
                    or next_memory > self.max_memory_bytes
                ):
                    break
                batch.append(request)
                pixels = next_pixels
                memory = next_memory
        claimed_ids = {request.request_id for request in batch}
        self._pending = [
            request
            for request in self._pending
            if request.request_id not in claimed_ids
        ]
        if batch:
            self.metrics["inference_calls"] += 1
            self.metrics["batch_count"] += 1
            self.metrics["batch_items"] += len(batch)
            self.metrics["pixels"] += sum(
                max(0, int(request.pixel_count))
                for request in batch
            )
        return batch

    @staticmethod
    def stable_results(
        batch: list[OcrInferenceRequest],
        results: dict[str, Any],
    ) -> list[tuple[OcrInferenceRequest, Any]]:
        return [
            (request, results[request.request_id])
            for request in sorted(batch, key=lambda item: item.stable_key)
            if request.request_id in results
        ]


@dataclass(slots=True)
class _LiveRecognitionItem:
    request_id: str
    crop: object
    pixel_count: int
    memory_bytes: int
    compatibility_key: tuple[str, str, str]
    submitted_at: float
    ready: threading.Event
    cancel_check: Callable[[], None]
    result: Any = None
    error: BaseException | None = None
    claimed: bool = False


class RecognitionMicroBatchCoordinator:
    """Thread-safe live inference aggregator used inside a persistent OCR worker."""

    def __init__(
        self,
        inference: Callable[[list[object]], Sequence[Any]],
        *,
        max_requests: int,
        max_pixels: int,
        max_memory_bytes: int,
        max_wait_ms: int,
    ) -> None:
        self.inference = inference
        self.max_requests = max(1, int(max_requests))
        self.max_pixels = max(1, int(max_pixels))
        self.max_memory_bytes = max(1, int(max_memory_bytes))
        self.max_wait_seconds = max(0.0, float(max_wait_ms) / 1000.0)
        self._condition = threading.Condition()
        self._pending: list[_LiveRecognitionItem] = []
        self._draining = False
        self._sequence = 0
        self._wait_samples_ms: list[float] = []
        self.metrics: dict[str, int | float] = {
            "recognize_requests": 0,
            "recognize_inference_calls": 0,
            "recognize_batch_count": 0,
            "recognize_batch_items": 0,
            "recognize_pixels": 0,
            "recognize_average_batch_size": 0.0,
            "microbatch_wait_ms_p50": 0.0,
            "microbatch_wait_ms_p95": 0.0,
            "microbatch_wait_ms_max": 0.0,
            "oversize_single_count": 0,
            "cancelled_before_batch_count": 0,
        }

    def recognize(
        self,
        crops: Sequence[object],
        *,
        pixel_counts: Sequence[int],
        compatibility_key: tuple[str, str, str],
        cancel_check: Callable[[], None] | None = None,
    ) -> list[Any]:
        if len(crops) != len(pixel_counts):
            raise ValueError("crops and pixel_counts must have the same length")
        if not crops:
            return []
        check = cancel_check or (lambda: None)
        check()
        now = time.monotonic()
        items: list[_LiveRecognitionItem] = []
        with self._condition:
            for crop, pixels in zip(crops, pixel_counts, strict=True):
                self._sequence += 1
                pixel_count = max(1, int(pixels))
                item = _LiveRecognitionItem(
                    request_id=f"live-{self._sequence}",
                    crop=crop,
                    pixel_count=pixel_count,
                    memory_bytes=max(1, pixel_count * 4),
                    compatibility_key=compatibility_key,
                    submitted_at=now,
                    ready=threading.Event(),
                    cancel_check=check,
                )
                self._pending.append(item)
                items.append(item)
            self.metrics["recognize_requests"] = int(
                self.metrics["recognize_requests"]
            ) + len(items)
            is_leader = not self._draining
            if is_leader:
                self._draining = True
            self._condition.notify_all()
        if is_leader:
            self._drain()
        while not all(item.ready.wait(timeout=0.05) for item in items):
            try:
                check()
            except BaseException as exc:
                with self._condition:
                    for item in items:
                        if item.ready.is_set() or item.claimed:
                            continue
                        if item in self._pending:
                            self._pending.remove(item)
                        item.error = exc
                        item.ready.set()
                        self.metrics["cancelled_before_batch_count"] = int(
                            self.metrics["cancelled_before_batch_count"]
                        ) + 1
                raise
        for item in items:
            if item.error is not None:
                raise item.error
        return [item.result for item in items]

    def _drain(self) -> None:
        try:
            while True:
                with self._condition:
                    if not self._pending:
                        self._draining = False
                        self._condition.notify_all()
                        return
                    oldest = min(
                        self._pending,
                        key=lambda item: item.submitted_at,
                    )
                    deadline = oldest.submitted_at + self.max_wait_seconds
                    while (
                        len(
                            [
                                item
                                for item in self._pending
                                if item.compatibility_key
                                == oldest.compatibility_key
                            ]
                        )
                        < self.max_requests
                        and time.monotonic() < deadline
                    ):
                        self._condition.wait(
                            timeout=max(0.0, deadline - time.monotonic())
                        )
                    for item in list(self._pending):
                        try:
                            item.cancel_check()
                        except BaseException as exc:
                            self._pending.remove(item)
                            item.error = exc
                            item.ready.set()
                            self.metrics[
                                "cancelled_before_batch_count"
                            ] = int(
                                self.metrics[
                                    "cancelled_before_batch_count"
                                ]
                            ) + 1
                    if oldest not in self._pending:
                        continue
                    compatible = [
                        item
                        for item in self._pending
                        if item.compatibility_key == oldest.compatibility_key
                    ]
                    compatible.sort(key=lambda item: item.submitted_at)
                    first = compatible[0]
                    if (
                        first.pixel_count > self.max_pixels
                        or first.memory_bytes > self.max_memory_bytes
                    ):
                        batch = [first]
                        self.metrics["oversize_single_count"] = int(
                            self.metrics["oversize_single_count"]
                        ) + 1
                    else:
                        batch = []
                        pixels = 0
                        memory = 0
                        for item in compatible:
                            next_pixels = pixels + item.pixel_count
                            next_memory = memory + item.memory_bytes
                            if len(batch) >= self.max_requests:
                                break
                            if batch and (
                                next_pixels > self.max_pixels
                                or next_memory > self.max_memory_bytes
                            ):
                                break
                            batch.append(item)
                            pixels = next_pixels
                            memory = next_memory
                    for item in batch:
                        item.claimed = True
                        self._pending.remove(item)
                started = time.monotonic()
                try:
                    results = list(
                        self.inference([item.crop for item in batch])
                    )
                    if len(results) != len(batch):
                        raise RuntimeError(
                            "OCR recognizer returned a different result count"
                        )
                except BaseException as exc:
                    for item in batch:
                        item.error = exc
                        item.ready.set()
                else:
                    for item, result in zip(batch, results, strict=True):
                        item.result = result
                        item.ready.set()
                completed_at = time.monotonic()
                waits = [
                    max(0.0, (started - item.submitted_at) * 1000.0)
                    for item in batch
                ]
                with self._condition:
                    self._wait_samples_ms.extend(waits)
                    self._wait_samples_ms = self._wait_samples_ms[-2048:]
                    self.metrics["recognize_inference_calls"] = int(
                        self.metrics["recognize_inference_calls"]
                    ) + 1
                    self.metrics["recognize_batch_count"] = int(
                        self.metrics["recognize_batch_count"]
                    ) + 1
                    self.metrics["recognize_batch_items"] = int(
                        self.metrics["recognize_batch_items"]
                    ) + len(batch)
                    self.metrics["recognize_pixels"] = int(
                        self.metrics["recognize_pixels"]
                    ) + sum(item.pixel_count for item in batch)
                    batches = max(
                        1,
                        int(self.metrics["recognize_batch_count"]),
                    )
                    self.metrics["recognize_average_batch_size"] = round(
                        int(self.metrics["recognize_batch_items"]) / batches,
                        3,
                    )
                    self.metrics["microbatch_wait_ms_p50"] = round(
                        statistics.median(self._wait_samples_ms),
                        3,
                    )
                    ordered_waits = sorted(self._wait_samples_ms)
                    p95_index = max(
                        0,
                        math.ceil(len(ordered_waits) * 0.95) - 1,
                    )
                    self.metrics["microbatch_wait_ms_p95"] = round(
                        ordered_waits[p95_index],
                        3,
                    )
                    self.metrics["microbatch_wait_ms_max"] = round(
                        max(ordered_waits),
                        3,
                    )
                    self._condition.notify_all()
        finally:
            with self._condition:
                self._draining = False
                for item in self._pending:
                    if not item.ready.is_set():
                        item.error = RuntimeError(
                            "OCR microbatch coordinator stopped"
                        )
                        item.ready.set()
                self._pending.clear()
                self._condition.notify_all()
