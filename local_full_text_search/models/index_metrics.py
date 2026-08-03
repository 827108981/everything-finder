from __future__ import annotations

from dataclasses import asdict, dataclass, field


def _hang_metric_defaults() -> dict[str, object]:
    return {
        "watchdog_scan_count": 0,
        "no_progress_timeout_count": 0,
        "planning_worker_timeout_count": 0,
        "parser_worker_timeout_count": 0,
        "worker_process_kill_count": 0,
        "child_process_kill_count": 0,
        "pool_rebuild_count": 0,
        "retry_count_by_error": {},
        "same_stall_signature_count": 0,
        "duplicate_progress_ignored_count": 0,
        "checkpoint_resume_count": 0,
        "checkpoint_resume_units_avoided": 0,
        "last_semantic_progress_age_seconds": 0.0,
    }


def _pdf_metric_defaults() -> dict[str, object]:
    return {
        "pdf_documents_total": 0,
        "pdf_pages_total": 0,
        "pdf_native_pages": 0,
        "pdf_ocr_candidate_pages": 0,
        "pdf_ocr_pages_completed": 0,
        "pdf_page_cache_hits": 0,
        "pdf_preview_render_ms": 0,
        "pdf_region_render_ms": 0,
        "pdf_full_page_fallback_count": 0,
        "pdf_200dpi_region_count": 0,
        "pdf_300dpi_upgrade_region_count": 0,
        "pdf_merge_ms": 0,
        "pdf_max_page_queue_wait_ms": 0,
        "pdf_dispatch_batch_count": 0,
        "pdf_dispatched_page_count": 0,
        "pdf_max_batch_pages": 0,
    }


def _ocr_metric_defaults() -> dict[str, object]:
    return {
        "ocr_worker_count_peak": 0,
        "ocr_model_load_count": 0,
        "ocr_model_load_ms": 0,
        "ocr_detect_requests": 0,
        "ocr_detect_calls": 0,
        "ocr_detect_pixels": 0,
        "ocr_recognize_requests": 0,
        "ocr_recognize_calls": 0,
        "ocr_recognize_pixels": 0,
        "ocr_microbatch_count": 0,
        "ocr_average_microbatch_size": 0.0,
        "ocr_crop_cache_hits": 0,
        "ocr_page_cache_hits": 0,
        "ocr_embedded_image_cache_hits": 0,
        "ocr_adaptive_split_count": 0,
        "ocr_unresolved_regions_peak": 0,
        "ocr_regions_resumed": 0,
        "detect_requests": 0,
        "detect_inference_calls": 0,
        "detect_batch_count": 0,
        "detect_average_batch_size": 0.0,
        "detect_pixels": 0,
        "recognize_requests": 0,
        "recognize_inference_calls": 0,
        "recognize_batch_count": 0,
        "recognize_average_batch_size": 0.0,
        "recognize_pixels": 0,
        "microbatch_wait_ms_p50": 0.0,
        "microbatch_wait_ms_p95": 0.0,
        "microbatch_wait_ms_max": 0.0,
        "oversize_single_count": 0,
        "cancelled_before_batch_count": 0,
    }


def _eta_metric_defaults() -> dict[str, object]:
    return {
        "eta_first_ready_seconds": 0.0,
        "eta_update_count": 0,
        "eta_jump_count": 0,
        "eta_absolute_error_seconds": 0.0,
        "eta_absolute_percentage_error": 0.0,
        "replay_events": [],
    }


def _pause_metric_defaults() -> dict[str, object]:
    return {
        "pause_request_to_submission_stop_ms": 0,
        "safe_pause_latency_seconds": 0.0,
        "paused_observation_seconds": 0.0,
        "paused_observation_progress_delta": 0,
        "paused_cpu_average": 0.0,
        "paused_read_bytes_delta": 0,
        "paused_database_write_count": 0,
        "resume_count": 0,
        "mode_switch_count": 0,
        "mode_switch_failure_count": 0,
        "mode_switch_rollback_count": 0,
    }


def _resource_metric_defaults() -> dict[str, object]:
    return {
        "sample_count": 0,
        "peak_total_cpu_percent": 0.0,
        "peak_app_cpu_percent": 0.0,
        "peak_rss_bytes": 0,
        "minimum_memory_available_bytes": 0,
        "peak_disk_busy_percent": 0.0,
        "peak_network_read_latency_ms": 0.0,
        "max_queue_depth": 0,
        "max_active_tasks": 0,
        "max_ocr_pending_pixels": 0,
        "max_writer_queue_depth": 0,
        "latest_worker_rss_bytes": {},
        "ocr_pool_resize_count": 0,
        "ocr_pool_resize_failure_count": 0,
        "rss_budget_bytes": 0,
        "rss_budget_exceeded": False,
    }


def _merge_defaults(
    defaults: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    return {**defaults, **dict(current)}


@dataclass(slots=True)
class FileTiming:
    file_id: int
    extension: str
    size_bytes: int
    queue_name: str
    queue_wait_ms: int = 0
    parse_ms: int = 0
    block_count: int = 0
    text_chars: int = 0
    spool_bytes: int = 0
    worker_pid: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IndexRunMetrics:
    run_id: str
    mode: str = "incremental"
    execution_mode: str = "normal"
    hardware: dict[str, object] = field(default_factory=dict)
    root_disk_classes: dict[str, str] = field(default_factory=dict)
    effective_profile: dict[str, object] = field(default_factory=dict)
    lane_worker_limits: dict[str, int] = field(default_factory=dict)
    discovered_files: int = 0
    discovered_bytes: int = 0
    scan_ms: int = 0
    fingerprint_ms: int = 0
    parse_ms_by_lane: dict[str, int] = field(default_factory=dict)
    lane_input_bytes: dict[str, int] = field(default_factory=dict)
    lane_output_blocks: dict[str, int] = field(default_factory=dict)
    normalize_ms: int = 0
    spool_write_ms: int = 0
    database_write_ms: int = 0
    fts_build_ms: int = 0
    total_ms: int = 0
    peak_rss_bytes: int = 0
    process_spawn_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dedup_candidate_count: int = 0
    dedup_full_hash_count: int = 0
    dedup_verified_source_count: int = 0
    dedup_parse_avoided_count: int = 0
    dedup_bytes_avoided: int = 0
    source_open_count: int = 0
    source_bytes_read: int = 0
    full_hash_count: int = 0
    full_hash_bytes: int = 0
    hash_reused_count: int = 0
    spool_write_bytes: int = 0
    spool_reuse_count: int = 0
    duplicate_source_read_avoided_bytes: int = 0
    ocr_metrics: dict[str, object] = field(
        default_factory=_ocr_metric_defaults
    )
    pdf_metrics: dict[str, object] = field(
        default_factory=_pdf_metric_defaults
    )
    zip_metrics: dict[str, object] = field(default_factory=dict)
    xlsx_metrics: dict[str, object] = field(default_factory=dict)
    legacy_office_metrics: dict[str, object] = field(default_factory=dict)
    hang_metrics: dict[str, object] = field(
        default_factory=_hang_metric_defaults
    )
    eta_metrics: dict[str, object] = field(
        default_factory=_eta_metric_defaults
    )
    pause_metrics: dict[str, object] = field(
        default_factory=_pause_metric_defaults
    )
    resource_metrics: dict[str, object] = field(
        default_factory=_resource_metric_defaults
    )
    fallback_and_throttle_events: list[dict[str, object]] = field(default_factory=list)
    profile_transitions: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hang_metrics = _merge_defaults(
            _hang_metric_defaults(),
            self.hang_metrics,
        )
        self.pdf_metrics = _merge_defaults(
            _pdf_metric_defaults(),
            self.pdf_metrics,
        )
        self.ocr_metrics = _merge_defaults(
            _ocr_metric_defaults(),
            self.ocr_metrics,
        )
        self.eta_metrics = _merge_defaults(
            _eta_metric_defaults(),
            self.eta_metrics,
        )
        self.pause_metrics = _merge_defaults(
            _pause_metric_defaults(),
            self.pause_metrics,
        )
        self.resource_metrics = _merge_defaults(
            _resource_metric_defaults(),
            self.resource_metrics,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
