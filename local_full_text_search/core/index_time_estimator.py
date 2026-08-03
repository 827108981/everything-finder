from __future__ import annotations

import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True, slots=True)
class IndexEstimate:
    seconds: int
    lower_seconds: int
    upper_seconds: int
    sample_count: int = 0
    ready: bool = False
    confidence: float = 0.0

    def display_text(self) -> str:
        if not self.ready:
            return "正在估算…"
        if self.seconds < 60:
            return f"预计剩余约 {max(1, self.seconds)} 秒"
        return f"预计剩余约 {max(1, math.ceil(self.seconds / 60))} 分钟"


@dataclass(slots=True)
class _LaneSamples:
    completed_cost: float = 0.0
    service_seconds: float = 0.0
    count: int = 0
    ewma_rate: float = 0.0
    recent_rates: deque[float] = field(default_factory=lambda: deque(maxlen=21))


class IndexTimeEstimator:
    """Stable single-value ETA for the critical path of parallel parser lanes."""

    DEFAULT_SECONDS_PER_COST = {
        "normal": 0.04,
        "pdf": 0.12,
        "office_process": 0.06,
        "legacy_office": 0.15,
        "legacy_word": 0.15,
        "legacy_excel": 0.15,
        "legacy_powerpoint": 0.15,
        "zip": 0.50,
        "ocr": 1.00,
    }
    MIN_READY_SAMPLES = 3
    DISPLAY_UPDATE_SECONDS = 10.0
    MAX_NORMAL_STEP_RATIO = 0.20

    def __init__(
        self,
        total_cost_by_lane: Mapping[str, float],
        workers_by_lane: Mapping[str, int] | None = None,
    ) -> None:
        self.total_cost_by_lane = {
            str(lane): max(0.0, float(cost))
            for lane, cost in total_cost_by_lane.items()
        }
        self.workers_by_lane = {
            str(lane): max(1, int(workers))
            for lane, workers in (workers_by_lane or {}).items()
        }
        self._samples: dict[str, _LaneSamples] = {}
        self._lock = threading.Lock()
        self._display_seconds: int | None = None
        self._last_display_update = 0.0
        self._paused = False
        self._last_estimate: IndexEstimate | None = None

    def observe(self, lane: str, cost: float, service_seconds: float) -> None:
        normalized_cost = max(0.001, float(cost))
        normalized_seconds = max(0.001, float(service_seconds))
        rate = normalized_seconds / normalized_cost
        with self._lock:
            sample = self._samples.setdefault(str(lane), _LaneSamples())
            sample.completed_cost += normalized_cost
            sample.service_seconds += normalized_seconds
            sample.count += 1
            sample.recent_rates.append(rate)
            sample.ewma_rate = (
                rate
                if sample.ewma_rate <= 0
                else sample.ewma_rate * 0.70 + rate * 0.30
            )

    def pause(self) -> None:
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        with self._lock:
            self._paused = False
            self._last_display_update = 0.0

    def reset_for_context_change(
        self,
        workers_by_lane: Mapping[str, int] | None = None,
    ) -> None:
        """Discard incompatible speed samples after a mode/profile change."""

        with self._lock:
            if workers_by_lane is not None:
                self.workers_by_lane = {
                    str(lane): max(1, int(workers))
                    for lane, workers in workers_by_lane.items()
                }
            self._samples = {}
            self._display_seconds = None
            self._last_display_update = 0.0
            self._last_estimate = None

    def estimate(
        self,
        remaining_cost_by_lane: Mapping[str, float],
        active_elapsed_by_lane: Mapping[str, float] | None = None,
        *,
        now: float | None = None,
        force_recalibration: bool = False,
    ) -> IndexEstimate | None:
        observed_at = time.monotonic() if now is None else float(now)
        active_elapsed = active_elapsed_by_lane or {}
        with self._lock:
            if self._paused and self._last_estimate is not None:
                return self._last_estimate
            samples = {
                lane: _copy_samples(value)
                for lane, value in self._samples.items()
            }

        lane_seconds: list[float] = []
        longest_active = 0.0
        sample_count = 0
        sampled_lanes = 0
        active_lanes = 0
        for lane, raw_cost in remaining_cost_by_lane.items():
            remaining_cost = max(0.0, float(raw_cost))
            if remaining_cost <= 0:
                continue
            active_lanes += 1
            default_rate = self.DEFAULT_SECONDS_PER_COST.get(lane, 0.08)
            sample = samples.get(lane, _LaneSamples())
            sample_count += sample.count
            if sample.count:
                sampled_lanes += 1
            rate = _blended_rate(sample, default_rate)
            minimum_rate = default_rate if lane in {"ocr", "zip"} else default_rate * 0.25
            rate = max(minimum_rate, min(default_rate * 50.0, rate))
            workers = self.workers_by_lane.get(lane, 1)
            lane_seconds.append(remaining_cost * rate / workers)
            longest_active = max(
                longest_active,
                max(0.0, float(active_elapsed.get(lane, 0.0))),
            )

        if not lane_seconds:
            return None
        critical_seconds = max(lane_seconds)
        if longest_active >= 30.0:
            # A long in-flight unit is evidence that the static prior is too
            # optimistic, but it should not create the old 60%-180% jump.
            critical_seconds = max(
                critical_seconds,
                min(longest_active, max(60.0, critical_seconds * 4.0)),
            )

        ready = sample_count >= self.MIN_READY_SAMPLES and (
            sampled_lanes >= min(active_lanes, 2) or active_lanes == 1
        )
        confidence = min(
            1.0,
            sample_count / 12.0,
            sampled_lanes / max(1, active_lanes),
        )
        lower = max(1, int(critical_seconds * (0.75 - 0.10 * confidence)))
        upper = max(
            lower,
            math.ceil(critical_seconds * (1.45 - 0.20 * confidence)),
        )
        raw_seconds = max(1, math.ceil(critical_seconds))

        with self._lock:
            due = (
                self._display_seconds is None
                or force_recalibration
                or observed_at - self._last_display_update >= self.DISPLAY_UPDATE_SECONDS
            )
            if due:
                if self._display_seconds is None or force_recalibration:
                    self._display_seconds = raw_seconds
                else:
                    maximum_step = max(
                        5,
                        int(self._display_seconds * self.MAX_NORMAL_STEP_RATIO),
                    )
                    delta = raw_seconds - self._display_seconds
                    self._display_seconds += max(
                        -maximum_step,
                        min(maximum_step, delta),
                    )
                self._last_display_update = observed_at
            estimate = IndexEstimate(
                seconds=max(1, int(self._display_seconds or raw_seconds)),
                lower_seconds=lower,
                upper_seconds=upper,
                sample_count=sample_count,
                ready=ready,
                confidence=confidence,
            )
            self._last_estimate = estimate
            return estimate


def _copy_samples(value: _LaneSamples) -> _LaneSamples:
    copied = _LaneSamples(
        completed_cost=value.completed_cost,
        service_seconds=value.service_seconds,
        count=value.count,
        ewma_rate=value.ewma_rate,
    )
    copied.recent_rates.extend(value.recent_rates)
    return copied


def _blended_rate(sample: _LaneSamples, default_rate: float) -> float:
    if not sample.count:
        return default_rate
    aggregate = sample.service_seconds / max(0.001, sample.completed_cost)
    recent = (
        statistics.median(sample.recent_rates)
        if sample.recent_rates
        else aggregate
    )
    ewma = sample.ewma_rate or aggregate
    observed = recent * 0.45 + ewma * 0.35 + aggregate * 0.20
    prior_weight = min(0.80, sample.count / 8.0)
    return default_rate * (1.0 - prior_weight) + observed * prior_weight
