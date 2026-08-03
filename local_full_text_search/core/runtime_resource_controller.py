from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ResourceSample:
    timestamp: float
    total_cpu_percent: float
    app_cpu_percent: float
    memory_available_bytes: int
    app_rss_bytes: int
    disk_busy_percent: float
    network_read_latency_ms: float
    queue_depth: int
    active_tasks: int
    ocr_pending_pixels: int
    writer_queue_depth: int
    completion_rate: float
    worker_failure_rate: float
    paused: bool = False
    worker_rss_bytes: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResourceDecision:
    state: str
    reason: str
    target_ocr_inflight: int | None = None
    target_read_inflight_scale: float = 1.0
    allow_active_tasks_to_finish: bool = True
    changed: bool = False


class RuntimeResourceController:
    """Hysteretic runtime admission controller; it never kills active work."""

    def __init__(
        self,
        *,
        memory_budget_bytes: int,
        ocr_hard_max: int,
        initial_ocr_inflight: int = 1,
        consecutive_samples: int = 3,
        min_state_seconds: float = 10.0,
        resize_cooldown_seconds: float = 15.0,
    ) -> None:
        self.memory_budget_bytes = max(256 * 1024**2, int(memory_budget_bytes))
        self.ocr_hard_max = max(1, int(ocr_hard_max))
        self.current_ocr_inflight = max(
            1,
            min(self.ocr_hard_max, int(initial_ocr_inflight)),
        )
        self.consecutive_samples = max(1, int(consecutive_samples))
        self.min_state_seconds = max(0.0, float(min_state_seconds))
        self.resize_cooldown_seconds = max(
            0.0,
            float(resize_cooldown_seconds),
        )
        self.state = "stable"
        self._state_since: float | None = None
        self._candidate_state = "stable"
        self._candidate_count = 0
        self._last_resize_at: float | None = None

    def observe(self, sample: ResourceSample) -> ResourceDecision:
        if self._state_since is None:
            self._state_since = float(sample.timestamp)
        if sample.paused:
            return ResourceDecision(
                state=self.state,
                reason="paused_lock",
                target_read_inflight_scale=self._read_scale(self.state),
            )
        candidate, reason = self._classify(sample)
        if candidate == self.state:
            self._candidate_state = candidate
            self._candidate_count = 0
            return ResourceDecision(
                state=self.state,
                reason=reason,
                target_read_inflight_scale=self._read_scale(self.state),
            )
        if candidate != self._candidate_state:
            self._candidate_state = candidate
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        state_age = float(sample.timestamp) - float(self._state_since)
        if (
            self._candidate_count < self.consecutive_samples
            or state_age < self.min_state_seconds
        ):
            return ResourceDecision(
                state=self.state,
                reason="hysteresis_wait",
                target_read_inflight_scale=self._read_scale(self.state),
            )
        self.state = candidate
        self._state_since = float(sample.timestamp)
        self._candidate_count = 0
        target: int | None = None
        resize_allowed = (
            self._last_resize_at is None
            or float(sample.timestamp) - self._last_resize_at
            >= self.resize_cooldown_seconds
        )
        if resize_allowed:
            if candidate == "underutilized":
                desired = min(2, self.ocr_hard_max)
            elif candidate in {
                "memory_pressure",
                "cpu_pressure",
                "io_pressure",
                "recovery_cooldown",
            }:
                desired = 1
            else:
                desired = self.current_ocr_inflight
            if desired != self.current_ocr_inflight:
                target = desired
                self.current_ocr_inflight = desired
                self._last_resize_at = float(sample.timestamp)
        return ResourceDecision(
            state=self.state,
            reason=reason,
            target_ocr_inflight=target,
            target_read_inflight_scale=self._read_scale(self.state),
            changed=True,
        )

    def _classify(self, sample: ResourceSample) -> tuple[str, str]:
        rss_ratio = float(sample.app_rss_bytes) / max(
            1,
            self.memory_budget_bytes,
        )
        if (
            rss_ratio >= 0.90
            or int(sample.memory_available_bytes) < 512 * 1024**2
        ):
            return "memory_pressure", "memory_budget_or_available"
        if (
            float(sample.total_cpu_percent) >= 90.0
            or float(sample.app_cpu_percent) >= 90.0
        ):
            return "cpu_pressure", "sustained_cpu_pressure"
        if (
            float(sample.disk_busy_percent) >= 90.0
            or float(sample.network_read_latency_ms) >= 250.0
        ):
            return "io_pressure", "sustained_io_pressure"
        if float(sample.worker_failure_rate) > 0.05:
            return "recovery_cooldown", "worker_failure_cooldown"
        if (
            float(sample.total_cpu_percent) < 55.0
            and float(sample.app_cpu_percent) < 55.0
            and rss_ratio < 0.70
            and int(sample.memory_available_bytes) >= 1024 * 1024**2
            and float(sample.disk_busy_percent) < 70.0
            and int(sample.writer_queue_depth) <= 1
            and int(sample.queue_depth) > int(sample.active_tasks)
        ):
            return "underutilized", "sustained_spare_capacity"
        return "stable", "within_operating_band"

    @staticmethod
    def _read_scale(state: str) -> float:
        if state == "io_pressure":
            return 0.5
        if state in {"memory_pressure", "recovery_cooldown"}:
            return 0.75
        return 1.0
