from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from local_full_text_search.core.index_time_estimator import IndexTimeEstimator


@dataclass(frozen=True, slots=True)
class EtaHistoryContext:
    """Exact compatibility boundary for historical parser throughput."""

    parser_name: str
    parser_version: str
    ocr_enabled: bool
    ocr_strategy: str
    ocr_model_fingerprint: str
    execution_mode: str
    hardware_tier: str
    disk_class: str
    extension: str
    size_bucket: str
    page_bucket: str

    @property
    def key(self) -> str:
        canonical = json.dumps(
            asdict(self),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True, slots=True)
class EtaHistorySample:
    context: EtaHistoryContext
    seconds_per_cost: float
    sample_count: int


class EtaHistoryStore:
    """Small atomic store which only returns exact-context ETA samples."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._samples: list[EtaHistorySample] = []
        self._load()

    def add(self, sample: EtaHistorySample) -> None:
        if sample.seconds_per_cost <= 0:
            raise ValueError("seconds_per_cost must be positive")
        if sample.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        self._samples.append(sample)
        self._samples = self._samples[-5000:]
        self._save()

    def rates_for(self, context: EtaHistoryContext) -> list[float]:
        expected = context.key
        return [
            float(sample.seconds_per_cost)
            for sample in self._samples
            if sample.context.key == expected
        ]

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if int(payload.get("schema_version") or 0) != self.SCHEMA_VERSION:
                return
            loaded: list[EtaHistorySample] = []
            for item in payload.get("samples") or []:
                context_payload = item.get("context")
                if not isinstance(context_payload, dict):
                    continue
                context = EtaHistoryContext(**context_payload)
                if str(item.get("context_key") or "") != context.key:
                    continue
                loaded.append(
                    EtaHistorySample(
                        context=context,
                        seconds_per_cost=float(item["seconds_per_cost"]),
                        sample_count=int(item["sample_count"]),
                    )
                )
            self._samples = loaded[-5000:]
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            self._samples = []

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "samples": [
                {
                    "context": asdict(sample.context),
                    "context_key": sample.context.key,
                    "seconds_per_cost": sample.seconds_per_cost,
                    "sample_count": sample.sample_count,
                }
                for sample in self._samples
            ],
        }
        temporary = self.path.with_name(f"{self.path.name}.tmp-{os.getpid()}")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.path)


@dataclass(frozen=True, slots=True)
class EtaReplayEvent:
    at_seconds: float
    event_type: str
    remaining_cost_by_lane: Mapping[str, float]
    active_elapsed_by_lane: Mapping[str, float] = field(default_factory=dict)
    workers_by_lane: Mapping[str, int] = field(default_factory=dict)
    lane: str = ""
    completed_cost: float = 0.0
    service_seconds: float = 0.0
    mode: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EtaReplayEvent:
        return cls(
            at_seconds=float(payload.get("at_seconds") or 0.0),
            event_type=str(payload.get("event_type") or "progress"),
            remaining_cost_by_lane=_float_map(
                payload.get("remaining_cost_by_lane")
            ),
            active_elapsed_by_lane=_float_map(
                payload.get("active_elapsed_by_lane")
            ),
            workers_by_lane=_int_map(payload.get("workers_by_lane")),
            lane=str(payload.get("lane") or ""),
            completed_cost=float(payload.get("completed_cost") or 0.0),
            service_seconds=float(payload.get("service_seconds") or 0.0),
            mode=str(payload.get("mode") or ""),
        )


@dataclass(frozen=True, slots=True)
class EtaReplayPoint:
    at_seconds: float
    event_type: str
    eta_seconds: int
    actual_remaining_seconds: float
    absolute_error_seconds: float
    absolute_percentage_error: float
    ready: bool
    mode: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EtaReplayReport:
    duration_seconds: float
    first_ready_seconds: float | None
    predictions: tuple[EtaReplayPoint, ...]
    median_absolute_percentage_error: float
    final_ten_minutes_median_absolute_percentage_error: float
    jump_count: int
    max_single_up_jump_seconds: int
    max_single_down_jump_seconds: int
    pause_frozen: bool
    mode_switch_recalibration_seconds: float | None
    worker_recycle_count: int

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["predictions"] = [point.to_dict() for point in self.predictions]
        return payload


def replay_eta(events: Sequence[EtaReplayEvent]) -> EtaReplayReport:
    ordered = sorted(events, key=lambda event: event.at_seconds)
    if not ordered:
        raise ValueError("ETA replay requires at least one event")
    if ordered[-1].event_type != "finish":
        raise ValueError("ETA replay must end with a finish event")
    if any(event.at_seconds < 0 for event in ordered):
        raise ValueError("ETA replay timestamps cannot be negative")
    if any(
        current.at_seconds < previous.at_seconds
        for previous, current in zip(ordered, ordered[1:])
    ):
        raise ValueError("ETA replay timestamps must be monotonic")

    duration = float(ordered[-1].at_seconds)
    pause_intervals = _pause_intervals(ordered, duration)
    workers = _first_workers(ordered)
    estimator = IndexTimeEstimator({}, workers)
    mode = "initial"
    predictions: list[EtaReplayPoint] = []
    first_ready: float | None = None
    worker_recycles = 0
    paused_eta: int | None = None
    pause_frozen = True
    mode_switch_at: float | None = None
    mode_switch_recalibration: float | None = None

    for event in ordered:
        if event.event_type == "finish":
            break
        if event.workers_by_lane:
            workers.update(
                {
                    str(lane): max(1, int(count))
                    for lane, count in event.workers_by_lane.items()
                }
            )
            estimator.workers_by_lane = dict(workers)
        if event.event_type == "worker_recycle":
            worker_recycles += 1
        if event.event_type == "mode_switch":
            mode = event.mode or "switched"
            mode_switch_at = event.at_seconds
            mode_switch_recalibration = None
            estimator = IndexTimeEstimator({}, workers)
        if (
            event.event_type == "completion"
            and event.lane
            and event.completed_cost > 0
            and event.service_seconds > 0
        ):
            estimator.observe(
                event.lane,
                event.completed_cost,
                event.service_seconds,
            )
        if event.event_type == "pause":
            estimator.pause()
        elif event.event_type == "resume":
            estimator.resume()

        estimate = estimator.estimate(
            event.remaining_cost_by_lane,
            event.active_elapsed_by_lane,
            now=event.at_seconds,
            force_recalibration=event.event_type in {"resume", "mode_switch"},
        )
        if estimate is None or not estimate.ready:
            continue
        if first_ready is None:
            first_ready = event.at_seconds
        if mode_switch_at is not None and mode_switch_recalibration is None:
            mode_switch_recalibration = max(
                0.0,
                event.at_seconds - mode_switch_at,
            )
        actual_remaining = _active_seconds_between(
            event.at_seconds,
            duration,
            pause_intervals,
        )
        absolute_error = abs(float(estimate.seconds) - actual_remaining)
        percentage_error = (
            absolute_error / actual_remaining
            if actual_remaining > 0
            else 0.0
        )
        point = EtaReplayPoint(
            at_seconds=event.at_seconds,
            event_type=event.event_type,
            eta_seconds=int(estimate.seconds),
            actual_remaining_seconds=round(actual_remaining, 6),
            absolute_error_seconds=round(absolute_error, 6),
            absolute_percentage_error=round(percentage_error, 6),
            ready=True,
            mode=mode,
        )
        predictions.append(point)
        if event.event_type == "pause":
            paused_eta = point.eta_seconds
        elif _inside_pause(event.at_seconds, pause_intervals):
            if paused_eta is not None and point.eta_seconds != paused_eta:
                pause_frozen = False
        elif event.event_type == "resume":
            paused_eta = None

    errors = [point.absolute_percentage_error for point in predictions]
    final_errors = [
        point.absolute_percentage_error
        for point in predictions
        if duration - point.at_seconds <= 600
    ]
    jumps = [
        current.eta_seconds - previous.eta_seconds
        for previous, current in zip(predictions, predictions[1:])
    ]
    jump_count = sum(
        1
        for index, jump in enumerate(jumps)
        if abs(jump)
        > max(30, int(predictions[index].eta_seconds * 0.25))
    )
    return EtaReplayReport(
        duration_seconds=duration,
        first_ready_seconds=first_ready,
        predictions=tuple(predictions),
        median_absolute_percentage_error=_median_or_zero(errors),
        final_ten_minutes_median_absolute_percentage_error=_median_or_zero(
            final_errors
        ),
        jump_count=jump_count,
        max_single_up_jump_seconds=max([0, *jumps]),
        max_single_down_jump_seconds=abs(min([0, *jumps])),
        pause_frozen=pause_frozen,
        mode_switch_recalibration_seconds=mode_switch_recalibration,
        worker_recycle_count=worker_recycles,
    )


def load_replay_events(payload: Mapping[str, object]) -> list[EtaReplayEvent]:
    raw_events = payload.get("events")
    if raw_events is None:
        raw_events = (
            (payload.get("metrics") or {}).get("eta_metrics", {}).get(
                "replay_events"
            )
            if isinstance(payload.get("metrics"), dict)
            else None
        )
    if not isinstance(raw_events, list):
        raise ValueError("input does not contain an ETA event list")
    return [
        EtaReplayEvent.from_dict(item)
        for item in raw_events
        if isinstance(item, dict)
    ]


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): max(0.0, float(item))
        for key, item in value.items()
    }


def _int_map(value: object) -> dict[str, int]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): max(1, int(item))
        for key, item in value.items()
    }


def _first_workers(events: Iterable[EtaReplayEvent]) -> dict[str, int]:
    for event in events:
        if event.workers_by_lane:
            return {
                str(lane): max(1, int(count))
                for lane, count in event.workers_by_lane.items()
            }
    return {}


def _pause_intervals(
    events: Sequence[EtaReplayEvent],
    duration: float,
) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    paused_at: float | None = None
    for event in events:
        if event.event_type == "pause" and paused_at is None:
            paused_at = event.at_seconds
        elif event.event_type == "resume" and paused_at is not None:
            intervals.append((paused_at, event.at_seconds))
            paused_at = None
    if paused_at is not None:
        intervals.append((paused_at, duration))
    return intervals


def _active_seconds_between(
    start: float,
    end: float,
    pause_intervals: Sequence[tuple[float, float]],
) -> float:
    paused = sum(
        max(0.0, min(end, pause_end) - max(start, pause_start))
        for pause_start, pause_end in pause_intervals
    )
    return max(0.0, end - start - paused)


def _inside_pause(
    at_seconds: float,
    pause_intervals: Sequence[tuple[float, float]],
) -> bool:
    return any(start <= at_seconds < end for start, end in pause_intervals)


def _median_or_zero(values: Sequence[float]) -> float:
    return round(float(statistics.median(values)), 6) if values else 0.0
