from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class IndexEstimate:
    lower_seconds: int
    upper_seconds: int
    sample_count: int = 0

    def display_text(self) -> str:
        if self.upper_seconds < 60:
            return f"预计剩余 {max(1, self.lower_seconds)}-{max(1, self.upper_seconds)} 秒"
        lower = max(1, self.lower_seconds // 60)
        upper = max(lower, math.ceil(self.upper_seconds / 60))
        return f"预计剩余 {lower}-{upper} 分钟"


@dataclass(slots=True)
class _LaneSamples:
    completed_cost: float = 0.0
    service_seconds: float = 0.0
    count: int = 0


class IndexTimeEstimator:
    """Estimate the critical path of independent parser lanes.

    Parse lanes run concurrently and have very different service rates. A
    single global completed/elapsed ratio is heavily biased by early small
    files and can report seconds while an OCR or ZIP task is still running.
    """

    DEFAULT_SECONDS_PER_COST = {
        "normal": 0.04,
        "office_process": 0.06,
        "legacy_office": 0.15,
        "zip": 0.50,
        "ocr": 1.00,
    }

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

    def observe(self, lane: str, cost: float, service_seconds: float) -> None:
        normalized_cost = max(0.001, float(cost))
        normalized_seconds = max(0.001, float(service_seconds))
        with self._lock:
            sample = self._samples.setdefault(str(lane), _LaneSamples())
            sample.completed_cost += normalized_cost
            sample.service_seconds += normalized_seconds
            sample.count += 1

    def estimate(
        self,
        remaining_cost_by_lane: Mapping[str, float],
        active_elapsed_by_lane: Mapping[str, float] | None = None,
    ) -> IndexEstimate | None:
        active_elapsed = active_elapsed_by_lane or {}
        with self._lock:
            samples = {
                lane: _LaneSamples(value.completed_cost, value.service_seconds, value.count)
                for lane, value in self._samples.items()
            }

        lane_seconds: list[float] = []
        longest_active = 0.0
        sample_count = 0
        for lane, raw_cost in remaining_cost_by_lane.items():
            remaining_cost = max(0.0, float(raw_cost))
            if remaining_cost <= 0:
                continue
            default_rate = self.DEFAULT_SECONDS_PER_COST.get(lane, 0.08)
            sample = samples.get(lane, _LaneSamples())
            sample_count += sample.count
            # A prior equivalent to 20 cost units prevents one very small file
            # from collapsing the estimate for hundreds of pending files.
            rate = (
                sample.service_seconds + default_rate * 20.0
            ) / max(0.001, sample.completed_cost + 20.0)
            minimum_rate = default_rate if lane in {"ocr", "zip"} else default_rate * 0.25
            rate = max(minimum_rate, min(default_rate * 50.0, rate))
            workers = self.workers_by_lane.get(lane, 1)
            lane_seconds.append(remaining_cost * rate / workers)
            longest_active = max(longest_active, max(0.0, float(active_elapsed.get(lane, 0.0))))

        if not lane_seconds:
            return None
        critical_seconds = max(lane_seconds)
        lower = max(1, int(critical_seconds * 0.60))
        upper = max(2, math.ceil(critical_seconds * 1.80))
        if longest_active >= 10.0:
            # Once a task has overrun the static model, stop showing a tiny
            # range. Its elapsed time is the best available uncertainty signal.
            lower = max(lower, min(60, int(longest_active * 0.10)))
            upper = max(upper, math.ceil(longest_active * 1.50))
        return IndexEstimate(lower, upper, sample_count)
