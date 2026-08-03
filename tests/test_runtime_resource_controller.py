from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.runtime_resource_controller import (
    ResourceSample,
    RuntimeResourceController,
)
from local_full_text_search.core.index_manager import (
    IndexManager,
    ParseLane,
    ProcessResourceMonitor,
)
from local_full_text_search.models.index_metrics import IndexRunMetrics


def _sample(
    timestamp: float,
    *,
    cpu: float = 30.0,
    app_cpu: float = 25.0,
    available_mb: int = 8_000,
    app_rss_mb: int = 1_000,
    disk_busy: float = 20.0,
    queue_depth: int = 20,
    active: int = 1,
    paused: bool = False,
    network_latency_ms: float = 0.0,
) -> ResourceSample:
    return ResourceSample(
        timestamp=timestamp,
        total_cpu_percent=cpu,
        app_cpu_percent=app_cpu,
        memory_available_bytes=available_mb * 1024 * 1024,
        app_rss_bytes=app_rss_mb * 1024 * 1024,
        disk_busy_percent=disk_busy,
        network_read_latency_ms=network_latency_ms,
        queue_depth=queue_depth,
        active_tasks=active,
        ocr_pending_pixels=10_000_000,
        writer_queue_depth=0,
        completion_rate=1.0,
        worker_failure_rate=0.0,
        paused=paused,
    )


def test_p1_02r_underutilized_requires_hysteresis_before_ocr_expansion() -> None:
    controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        consecutive_samples=3,
        min_state_seconds=0,
        resize_cooldown_seconds=0,
    )

    assert controller.observe(_sample(0)).target_ocr_inflight is None
    assert controller.observe(_sample(1)).target_ocr_inflight is None
    decision = controller.observe(_sample(2))

    assert decision.state == "underutilized"
    assert decision.target_ocr_inflight == 2
    assert decision.reason == "sustained_spare_capacity"


def test_p1_02r_memory_pressure_shrinks_without_killing_active_tasks() -> None:
    controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        initial_ocr_inflight=2,
        consecutive_samples=2,
        min_state_seconds=0,
        resize_cooldown_seconds=0,
    )

    controller.observe(_sample(0, available_mb=300, app_rss_mb=3900))
    decision = controller.observe(
        _sample(1, available_mb=300, app_rss_mb=3900, active=2)
    )

    assert decision.state == "memory_pressure"
    assert decision.target_ocr_inflight == 1
    assert decision.allow_active_tasks_to_finish is True


def test_p1_02r_threshold_noise_does_not_flap_and_pause_locks_controller() -> None:
    controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        consecutive_samples=3,
        min_state_seconds=10,
        resize_cooldown_seconds=10,
    )
    states = [
        controller.observe(_sample(0, cpu=54)).state,
        controller.observe(_sample(1, cpu=58)).state,
        controller.observe(_sample(2, cpu=53)).state,
        controller.observe(_sample(3, cpu=59)).state,
    ]
    paused = controller.observe(_sample(20, paused=True))

    assert states == ["stable", "stable", "stable", "stable"]
    assert paused.state == controller.state
    assert paused.target_ocr_inflight is None
    assert paused.reason == "paused_lock"


def test_p1_02r_cpu_and_disk_pressure_never_expand() -> None:
    cpu_controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        consecutive_samples=1,
        min_state_seconds=0,
        resize_cooldown_seconds=0,
    )
    disk_controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        consecutive_samples=1,
        min_state_seconds=0,
        resize_cooldown_seconds=0,
    )

    cpu = cpu_controller.observe(_sample(0, cpu=96, app_cpu=92))
    disk = disk_controller.observe(_sample(0, disk_busy=98))

    assert cpu.state == "cpu_pressure"
    assert cpu.target_ocr_inflight != 2
    assert disk.state == "io_pressure"
    assert disk.target_read_inflight_scale < 1.0


def test_p1_02r_network_latency_reduces_read_admission() -> None:
    controller = RuntimeResourceController(
        memory_budget_bytes=4 * 1024**3,
        ocr_hard_max=2,
        consecutive_samples=1,
        min_state_seconds=0,
        resize_cooldown_seconds=0,
    )

    decision = controller.observe(
        _sample(0, network_latency_ms=320.0)
    )

    assert decision.state == "io_pressure"
    assert decision.reason == "sustained_io_pressure"
    assert decision.target_read_inflight_scale == 0.5


def test_p1_02r_resource_monitor_reports_network_probe_latency(
    tmp_path,
) -> None:
    monitor = ProcessResourceMonitor(network_probe_path=tmp_path)
    monitor._latest["network_read_latency_ms"] = 42.5
    monitor._latest["worker_rss_bytes"] = {123: 456_789}

    sample = monitor.snapshot(
        queue_depth=1,
        active_tasks=1,
        ocr_pending_pixels=0,
        writer_queue_depth=0,
        paused=False,
    )

    assert sample.network_read_latency_ms == 42.5
    assert sample.worker_rss_bytes == {123: 456_789}


def test_p1_02r_idle_safe_boundary_replaces_the_real_ocr_pool(
    tmp_path,
) -> None:
    settings = AppSettings(enable_ocr=True, ocr_workers=2)
    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        settings,
    )
    old_executor = ProcessPoolExecutor(max_workers=1)
    lane = ParseLane(
        "ocr",
        old_executor,
        1,
        256 * 1024 * 1024,
        process_based=True,
        persistent_process=True,
        worker_count=1,
    )
    lanes = {"ocr": lane}
    executors = [old_executor]
    process_executors = [old_executor]

    expanded, reason = manager._resize_idle_ocr_process_lane(
        lanes,
        executors,
        process_executors,
        tmp_path,
        target_workers=2,
        reason="sustained_spare_capacity",
    )

    assert expanded is True
    assert reason == "resized"
    assert lane.executor is not old_executor
    assert lane.worker_count == 2
    assert lane.max_in_flight == 2
    assert getattr(lane.executor, "_max_workers") == 2

    expanded_executor = lane.executor
    shrunk, reason = manager._resize_idle_ocr_process_lane(
        lanes,
        executors,
        process_executors,
        tmp_path,
        target_workers=1,
        reason="memory_budget_or_available",
    )

    assert shrunk is True
    assert reason == "resized"
    assert lane.executor is not expanded_executor
    assert lane.worker_count == 1
    assert lane.max_in_flight == 1
    assert getattr(lane.executor, "_max_workers") == 1
    lane.executor.shutdown(wait=True, cancel_futures=True)


def test_p1_02r_resize_event_contains_every_required_resource_sample(
    tmp_path,
) -> None:
    manager = IndexManager(
        DatabaseManager(tmp_path / "index.db"),
        AppSettings(),
    )
    manager._current_metrics = IndexRunMetrics(run_id="resource-event")
    manager._latest_resource_sample = _sample(
        123.0,
        cpu=44.0,
        app_cpu=31.0,
        available_mb=6_000,
        app_rss_mb=900,
        disk_busy=28.0,
        queue_depth=12,
        active=1,
        network_latency_ms=7.5,
    )
    manager._resource_resize_cooldown_seconds = 15.0

    manager._record_resource_pool_resize(
        old_workers=1,
        new_workers=2,
        reason="sustained_spare_capacity",
        success=True,
        rollback_reason="",
    )

    event = manager._current_metrics.profile_transitions[-1]
    assert {
        "timestamp",
        "from_workers",
        "to_workers",
        "reason",
        "cpu_percent",
        "app_cpu_percent",
        "rss_bytes",
        "memory_available_bytes",
        "disk_busy_percent",
        "network_read_latency_ms",
        "queue_depth",
        "active_tasks",
        "writer_queue_depth",
        "cooldown_seconds",
        "success",
        "rollback_reason",
    }.issubset(event)
