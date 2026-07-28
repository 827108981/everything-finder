from __future__ import annotations

import math
import statistics
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SearchEstimateContext:
    mode: str
    file_count: int
    scoped: bool
    extension_filtered: bool
    searches_content: bool
    ocr_fuzzy: bool
    case_sensitive: bool


@dataclass(frozen=True, slots=True)
class SearchEstimate:
    lower_ms: int
    upper_ms: int
    sample_count: int = 0

    def display_text(self) -> str:
        if self.upper_ms < 1_000:
            return "预计不到 1 秒"
        if self.upper_ms < 60_000:
            lower = max(1, math.floor(self.lower_ms / 1_000))
            upper = max(lower, math.ceil(self.upper_ms / 1_000))
            return f"预计 {lower}–{upper} 秒"
        lower = max(1, math.floor(self.lower_ms / 60_000))
        upper = max(lower, math.ceil(self.upper_ms / 60_000))
        return f"预计 {lower}–{upper} 分钟"


class SearchTimeEstimator:
    """Fast session-local estimates, calibrated with recent completed searches."""

    def __init__(self, history_size: int = 20) -> None:
        self._history_size = max(1, history_size)
        self._samples: dict[tuple[object, ...], deque[int]] = defaultdict(
            lambda: deque(maxlen=self._history_size)
        )

    def estimate(self, context: SearchEstimateContext) -> SearchEstimate:
        baseline_ms = self._baseline_ms(context)
        samples = self._samples[self._key(context)]
        if samples:
            observed_ms = statistics.median(samples)
            observed_weight = min(0.8, 0.45 + 0.08 * len(samples))
            center_ms = baseline_ms * (1 - observed_weight) + observed_ms * observed_weight
            spread = max(0.22, 0.48 - 0.04 * min(len(samples), 6))
        else:
            center_ms = baseline_ms
            spread = 0.55

        lower_ms = max(80, round(center_ms * (1 - spread)))
        upper_ms = max(lower_ms + 120, round(center_ms * (1 + spread)))
        return SearchEstimate(lower_ms, upper_ms, len(samples))

    def observe(self, context: SearchEstimateContext, elapsed_ms: int) -> None:
        if elapsed_ms > 0:
            self._samples[self._key(context)].append(int(elapsed_ms))

    @staticmethod
    def _baseline_ms(context: SearchEstimateContext) -> float:
        # SQLite FTS startup is nearly constant; result verification grows with index size.
        center_ms = 180 + min(max(context.file_count, 0), 1_000_000) * 0.025
        center_ms *= {
            "exact": 1.0,
            "phrase": 1.15,
            "all": 1.3,
            "any": 1.55,
            "regex": 3.8,
        }.get(context.mode, 1.25)
        if context.scoped:
            center_ms *= 0.72
        if context.extension_filtered:
            center_ms *= 0.76
        if not context.searches_content:
            center_ms *= 0.65
        if context.ocr_fuzzy:
            center_ms *= 1.7
        if context.case_sensitive:
            center_ms *= 1.12
        return min(max(center_ms, 140), 120_000)

    @staticmethod
    def _key(context: SearchEstimateContext) -> tuple[object, ...]:
        if context.file_count < 10_000:
            size_bucket = "small"
        elif context.file_count < 100_000:
            size_bucket = "medium"
        else:
            size_bucket = "large"
        return (
            context.mode,
            size_bucket,
            context.scoped,
            context.extension_filtered,
            context.searches_content,
            context.ocr_fuzzy,
            context.case_sensitive,
        )
