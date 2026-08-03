from __future__ import annotations

from local_full_text_search.core.index_manager import (
    accumulate_required_ocr_block_metrics,
)
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import IndexRunMetrics


def test_metric_01r_default_payload_declares_every_required_phase2_metric() -> None:
    payload = IndexRunMetrics(run_id="required-keys").to_dict()

    required = {
        "hang_metrics": {
            "watchdog_scan_count",
            "no_progress_timeout_count",
            "planning_worker_timeout_count",
            "parser_worker_timeout_count",
            "worker_process_kill_count",
            "child_process_kill_count",
            "pool_rebuild_count",
            "retry_count_by_error",
            "same_stall_signature_count",
            "duplicate_progress_ignored_count",
            "checkpoint_resume_count",
            "checkpoint_resume_units_avoided",
            "last_semantic_progress_age_seconds",
        },
        "pdf_metrics": {
            "pdf_documents_total",
            "pdf_pages_total",
            "pdf_native_pages",
            "pdf_ocr_candidate_pages",
            "pdf_ocr_pages_completed",
            "pdf_page_cache_hits",
            "pdf_preview_render_ms",
            "pdf_region_render_ms",
            "pdf_full_page_fallback_count",
            "pdf_200dpi_region_count",
            "pdf_300dpi_upgrade_region_count",
            "pdf_merge_ms",
            "pdf_max_page_queue_wait_ms",
            "pdf_dispatch_batch_count",
            "pdf_dispatched_page_count",
            "pdf_max_batch_pages",
        },
        "ocr_metrics": {
            "ocr_worker_count_peak",
            "ocr_model_load_count",
            "ocr_model_load_ms",
            "ocr_detect_requests",
            "ocr_detect_calls",
            "ocr_detect_pixels",
            "ocr_recognize_requests",
            "ocr_recognize_calls",
            "ocr_recognize_pixels",
            "ocr_microbatch_count",
            "ocr_average_microbatch_size",
            "ocr_crop_cache_hits",
            "ocr_page_cache_hits",
            "ocr_embedded_image_cache_hits",
            "ocr_adaptive_split_count",
            "ocr_unresolved_regions_peak",
            "ocr_regions_resumed",
            "detect_requests",
            "detect_inference_calls",
            "detect_batch_count",
            "detect_average_batch_size",
            "detect_pixels",
            "recognize_requests",
            "recognize_inference_calls",
            "recognize_batch_count",
            "recognize_average_batch_size",
            "recognize_pixels",
            "microbatch_wait_ms_p50",
            "microbatch_wait_ms_p95",
            "microbatch_wait_ms_max",
            "oversize_single_count",
            "cancelled_before_batch_count",
        },
        "eta_metrics": {
            "eta_first_ready_seconds",
            "eta_update_count",
            "eta_jump_count",
            "eta_absolute_error_seconds",
            "eta_absolute_percentage_error",
        },
        "pause_metrics": {
            "pause_request_to_submission_stop_ms",
            "safe_pause_latency_seconds",
            "paused_observation_progress_delta",
            "paused_cpu_average",
            "paused_read_bytes_delta",
            "paused_database_write_count",
            "resume_count",
            "mode_switch_count",
            "mode_switch_failure_count",
            "mode_switch_rollback_count",
        },
        "resource_metrics": {
            "sample_count",
            "peak_total_cpu_percent",
            "peak_app_cpu_percent",
            "peak_rss_bytes",
            "minimum_memory_available_bytes",
            "peak_disk_busy_percent",
            "peak_network_read_latency_ms",
            "max_queue_depth",
            "max_active_tasks",
            "max_ocr_pending_pixels",
            "max_writer_queue_depth",
            "latest_worker_rss_bytes",
            "ocr_pool_resize_count",
            "ocr_pool_resize_failure_count",
            "rss_budget_bytes",
            "rss_budget_exceeded",
        },
    }
    for section, keys in required.items():
        assert keys.issubset(payload[section])


def test_run_metrics_persist_effective_profile_and_pipeline_counters() -> None:
    metrics = IndexRunMetrics(
        run_id="run",
        execution_mode="performance",
        root_disk_classes={"E:\\资料": "ssd", "Z:\\共享": "network"},
        effective_profile={"cpu_token_budget": 6},
        lane_worker_limits={"normal": 3, "ocr": 1},
        lane_input_bytes={"normal": 123},
        lane_output_blocks={"normal": 4},
        source_open_count=2,
        source_bytes_read=456,
        full_hash_count=1,
        full_hash_bytes=123,
        hash_reused_count=1,
        hang_metrics={
            "watchdog_scan_count": 2,
            "parser_worker_timeout_count": 1,
        },
        eta_metrics={"eta_first_ready_seconds": 8.0},
        pause_metrics={"safe_pause_latency_seconds": 1.2},
        resource_metrics={"sample_count": 3},
    )

    payload = metrics.to_dict()

    assert payload["execution_mode"] == "performance"
    assert payload["root_disk_classes"]["Z:\\共享"] == "network"
    assert payload["effective_profile"]["cpu_token_budget"] == 6
    assert payload["lane_worker_limits"]["ocr"] == 1
    assert payload["lane_input_bytes"]["normal"] == 123
    assert payload["lane_output_blocks"]["normal"] == 4
    assert payload["source_open_count"] == 2
    assert payload["full_hash_count"] == 1
    assert payload["hang_metrics"]["watchdog_scan_count"] == 2
    assert payload["eta_metrics"]["eta_first_ready_seconds"] == 8.0
    assert payload["pause_metrics"]["safe_pause_latency_seconds"] == 1.2
    assert payload["resource_metrics"]["sample_count"] == 3


def test_metric_01r_real_ocr_block_updates_required_public_metric_names() -> None:
    metrics = IndexRunMetrics(run_id="ocr-required-names")
    block = ContentBlock(
        file_path="scan.pdf",
        block_index=0,
        block_type="pdf_page_ocr",
        location_text="第 1 页",
        raw_text="OCR",
        normalized_text="ocr",
        source_type="ocr",
        extra={
            "ocr_embedded_image_cache_hits": 1,
            "ocr_exact_cache_hits": 2,
            "preview_detect_calls": 1,
            "preview_detect_pixels": 921_600,
            "recognize_requests": 7,
            "recognize_inference_calls": 2,
            "recognize_pixels": 123_456,
            "adaptive_regions_split": 3,
            "adaptive_regions_remaining_peak": 5,
            "checkpoint_regions_reused": 4,
            "pdf_full_page_fallback": True,
            "pdf_upgraded_regions": 2,
        },
    )

    accumulate_required_ocr_block_metrics(metrics, block)

    assert metrics.ocr_metrics["ocr_embedded_image_cache_hits"] == 1
    assert metrics.ocr_metrics["ocr_page_cache_hits"] == 2
    assert metrics.ocr_metrics["ocr_detect_requests"] == 1
    assert metrics.ocr_metrics["ocr_detect_calls"] == 1
    assert metrics.ocr_metrics["ocr_detect_pixels"] == 921_600
    assert metrics.ocr_metrics["ocr_recognize_requests"] == 7
    assert metrics.ocr_metrics["ocr_recognize_calls"] == 2
    assert metrics.ocr_metrics["ocr_recognize_pixels"] == 123_456
    assert metrics.ocr_metrics["ocr_adaptive_split_count"] == 3
    assert metrics.ocr_metrics["ocr_unresolved_regions_peak"] == 5
    assert metrics.ocr_metrics["ocr_regions_resumed"] == 4
    assert metrics.pdf_metrics["pdf_full_page_fallback_count"] == 1
    assert metrics.pdf_metrics["pdf_300dpi_upgrade_region_count"] == 2
