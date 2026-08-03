from __future__ import annotations

from pathlib import Path

from local_full_text_search.core.phase2_validation import (
    _mode_switch_ack_cursor,
    validate_ocr_backend_gate,
    validate_pdf_page_pipeline,
    validate_paused_mode_switch,
    validate_safe_pause,
)


def test_p0_02r_ocr_backend_gate_uses_one_real_lane_and_recovers_worker(
    tmp_path: Path,
) -> None:
    result = validate_ocr_backend_gate(tmp_path)

    assert result["mechanism_only"] is False
    scheduler = result["real_scheduler"]
    assert scheduler["passed"] is True, scheduler
    assert scheduler["confirmed_source_kinds"] == [
        "image",
        "pdf_page",
        "zip_image",
    ]
    assert scheduler["shared_worker_count"] == 1
    assert scheduler["model_state"] == "ready"
    assert scheduler["model_load_count_per_worker"] == 2
    assert scheduler["small_source_completed_before_pdf_drained"] is True
    assert scheduler["worker_pids_after_completion"] == []
    recovery = result["worker_crash_recovery"]
    assert recovery["passed"] is True, recovery
    assert recovery["crash_observed"] is True
    assert recovery["crashed_worker_model_state"] == "ready"
    assert recovery["crashed_worker_model_load_count"] == 2
    assert recovery["pool_rebuild_count"] == 1
    assert recovery["rebuild_reason_recorded"] is True
    assert recovery["replacement_worker_changed"] is True
    assert recovery["replacement_worker_model_load_count"] == 2
    assert recovery["source_success"] is True
    assert recovery["worker_pids_after_completion"] == []
    assert result["passed"] is True


def test_u0_04r_zero_resume_cursor_is_not_replaced_by_completed_bytes() -> None:
    assert (
        _mode_switch_ack_cursor(
            {
                "acknowledgements": [
                    {"cursor": 0, "completed_units": 99_840}
                ]
            }
        )
        == 0
    )


def test_p0_01r_pdf_validation_covers_cancel_and_mode_consistency(
    tmp_path: Path,
) -> None:
    result = validate_pdf_page_pipeline(tmp_path)

    assert result["passed"] is True, result
    assert result["cancel_gate"]["cancelled"] is True
    assert result["cancel_gate"]["confirmed_pages"] > 0
    assert result["cancel_gate"]["pending_pages"] > 0
    assert result["cancel_gate"]["published_blocks"] == 0
    assert result["cancel_gate"]["merge_completed"] == 0
    assert result["cancel_gate"]["search_ready"] is False
    assert result["mode_consistency"]["digest_matches"] is True
    assert result["mode_consistency"]["page_order_matches"] is True
    assert result["mode_consistency"]["search_hits_match"] is True


def test_u0_03r_safe_pause_validation_runs_a_real_xlsx_task(
    tmp_path: Path,
) -> None:
    result = validate_safe_pause(tmp_path)

    assert result["mechanism_only"] is False
    assert result["resource_idle_real_format_gate_required"] is False
    assert result["formats"]["xlsx"]["started"] is True
    assert result["formats"]["xlsx"]["safe_pause_confirmed"] is True
    assert result["formats"]["xlsx"]["observation_seconds"] >= 5
    assert result["formats"]["xlsx"]["progress_delta"] == 0
    assert result["formats"]["xlsx"]["database_write_delta"] == 0
    assert result["formats"]["xlsx"]["source_read_bytes_delta"] == 0
    assert result["formats"]["xlsx"]["resume_cursor_advanced"] is True
    assert result["formats"]["xlsx"]["duplicate_blocks"] == 0
    assert result["formats"]["xlsx"]["failed_delta"] == 0
    pdf = result["formats"]["pdf_native_page"]
    assert pdf["started"] is True
    assert pdf["safe_pause_confirmed"] is True
    assert pdf["observation_seconds"] >= 5
    assert pdf["progress_delta"] == 0
    assert pdf["database_write_delta"] == 0
    assert pdf["source_read_bytes_delta"] == 0
    assert pdf["pending_pages_while_paused"] > 0
    assert pdf["resume_pages_advanced"] is True
    assert pdf["ordered_pages"] == pdf["page_tasks"]
    assert pdf["duplicate_blocks"] == 0
    assert pdf["failed_delta"] == 0
    zip_member = result["formats"]["zip_member_xlsx"]
    assert zip_member["started"] is True
    assert zip_member["safe_pause_confirmed"] is True
    assert zip_member["observation_seconds"] >= 5
    assert zip_member["progress_delta"] == 0
    assert zip_member["database_write_delta"] == 0
    assert zip_member["source_read_bytes_delta"] == 0
    assert zip_member["resume_cursor_advanced"] is True
    assert zip_member["member_success"] is True
    assert zip_member["search_token_blocks"] == 1
    assert zip_member["duplicate_blocks"] == 0
    assert "zip_member" not in result["remaining_format_gates"]
    pptx = result["formats"]["pptx_slide"]
    assert pptx["started"] is True
    assert pptx["safe_pause_confirmed"] is True
    assert pptx["observation_seconds"] >= 5
    assert pptx["progress_delta"] == 0
    assert pptx["database_write_delta"] == 0
    assert pptx["source_read_bytes_delta"] == 0
    assert pptx["resume_cursor_advanced"] is True
    assert pptx["ordered_slides"] == pptx["slide_count"]
    assert pptx["search_token_blocks"] == 1
    assert pptx["duplicate_blocks"] == 0
    assert pptx["passed"] is True, pptx
    assert "ooxml_paragraph_or_slide" not in result["remaining_format_gates"]
    image_ocr = result["formats"]["image_ocr_batch"]
    assert image_ocr["started"] is True
    assert image_ocr["safe_pause_confirmed"] is True
    assert image_ocr["observation_seconds"] >= 5
    assert image_ocr["progress_delta"] == 0
    assert image_ocr["database_write_delta"] == 0
    assert image_ocr["source_read_bytes_delta"] == 0
    assert image_ocr["resume_cursor_advanced"] is True
    assert image_ocr["source_success"] is True
    assert image_ocr["ocr_text_chars"] > 0
    assert image_ocr["engine"] == "PaddleOCR"
    assert image_ocr["duplicate_blocks"] == 0
    assert image_ocr["worker_exited"] is True
    assert image_ocr["passed"] is True, image_ocr
    assert "image_ocr_batch" not in result["remaining_format_gates"]
    pdf_ocr = result["formats"]["pdf_ocr_region"]
    assert pdf_ocr["started"] is True
    assert pdf_ocr["safe_pause_confirmed"] is True
    assert pdf_ocr["observation_seconds"] >= 5
    assert pdf_ocr["progress_delta"] == 0
    assert pdf_ocr["database_write_delta"] == 0
    assert pdf_ocr["source_read_bytes_delta"] == 0
    assert pdf_ocr["pending_pages_while_paused"] > 0
    assert pdf_ocr["resume_cursor_advanced"] is True
    assert pdf_ocr["completed_ocr_pages"] == pdf_ocr["page_count"]
    assert pdf_ocr["merge_tasks"] == 1
    assert pdf_ocr["ordered_pages"] == pdf_ocr["page_count"]
    assert pdf_ocr["ocr_text_chars"] > 0
    assert pdf_ocr["duplicate_blocks"] == 0
    assert pdf_ocr["worker_exited"] is True
    assert pdf_ocr["passed"] is True, pdf_ocr
    assert "pdf_ocr_region" not in result["remaining_format_gates"]
    legacy = result["formats"]["legacy_office_conversion"]
    assert legacy["started"] is True
    assert legacy["external_process_seen"] is True
    assert legacy["safe_pause_confirmed"] is True
    assert legacy["office_pids_while_paused"] == []
    assert legacy["observation_seconds"] >= 5
    assert legacy["progress_delta"] == 0
    assert legacy["database_write_delta"] == 0
    assert legacy["source_read_bytes_delta"] == 0
    assert legacy["resume_cursor_advanced"] is True
    assert legacy["source_success"] is True
    assert legacy["search_token_blocks"] == 1
    assert legacy["conversion_cache_reused_after_resume"] is True
    assert legacy["duplicate_blocks"] == 0
    assert legacy["worker_exited"] is True
    assert legacy["office_pids_after_completion"] == []
    assert legacy["passed"] is True, legacy
    assert "legacy_office_conversion" not in result["remaining_format_gates"]
    content_hash = result["planning"]["content_hash"]
    assert content_hash["started"] is True
    assert content_hash["safe_pause_confirmed"] is True
    assert content_hash["observation_seconds"] >= 5
    assert content_hash["progress_delta"] == 0
    assert content_hash["database_write_delta"] == 0
    assert content_hash["source_read_bytes_delta"] == 0
    assert content_hash["paused_cpu_average"] <= 5
    assert content_hash["source_unchanged"] is True
    assert content_hash["acknowledgements"]
    assert content_hash["resume_cursor_advanced"] is True
    assert content_hash["source_success"] is True
    assert content_hash["duplicate_blocks"] == 0
    assert content_hash["summary"]["failed"] == 0
    assert content_hash["passed"] is True, content_hash
    directory = result["planning"]["directory_enumeration"]
    assert directory["started"] is True
    assert directory["safe_pause_confirmed"] is True
    assert directory["observation_seconds"] >= 5
    assert directory["progress_delta"] == 0
    assert directory["database_write_delta"] == 0
    assert directory["source_read_bytes_delta"] == 0
    assert directory["paused_cpu_average"] <= 5
    assert directory["source_unchanged"] is True
    assert directory["acknowledgements"]
    assert directory["resume_cursor_advanced"] is True
    assert directory["indexed_files"] == directory["source_files"]
    assert directory["summary"]["failed"] == 0
    assert directory["passed"] is True, directory
    assert "hash_offset" not in result["remaining_format_gates"]
    assert "directory_batch" not in result["remaining_format_gates"]
    assert result["passed"] is True, result


def test_u0_04r_mode_switch_validation_uses_a_real_resumable_xlsx(
    tmp_path: Path,
) -> None:
    result = validate_paused_mode_switch(tmp_path)

    assert result["passed"] is True, result
    assert result["mechanism_only"] is False
    assert result["real_long_format_gate_required"] is False
    assert result["format"] == "xlsx"
    assert result["switches"] == [
        "normal_to_performance",
        "performance_to_normal",
    ]
    assert result["remained_paused_after_each_switch"] is True
    assert result["cursor_advanced_after_each_resume"] is True
    assert result["duplicate_blocks"] == 0
    assert result["content_digest_matches_control"] is True
    assert result["search_hits_match_control"] is True
    assert result["rollback_injection_passed"] is True
    assert result["remaining_format_gates"] == []
    for format_name in ("image_ocr", "pdf_ocr_page", "zip_member"):
        format_result = result["formats"][format_name]
        assert format_result["passed"] is True, format_result
        assert format_result["switches"] == [
            "normal_to_performance",
            "performance_to_normal",
        ]
        assert format_result["remained_paused_after_each_switch"] is True
        assert format_result["cursor_advanced_after_each_resume"] is True
        assert format_result["duplicate_blocks"] == 0
        assert format_result["content_digest_matches_control"] is True
        assert format_result["search_hits_match_control"] is True
    legacy = result["formats"]["legacy_office"]
    assert legacy["passed"] is True, legacy
    assert legacy["switches"] == ["normal_to_performance"]
    assert legacy["remained_paused_after_each_switch"] is True
    assert legacy["cursor_advanced_after_each_resume"] is True
    assert legacy["duplicate_blocks"] == 0
    assert legacy["content_digest_matches_control"] is True
    assert legacy["search_hits_match_control"] is True
    assert result["formats"]["image_ocr"]["engine"] == "PaddleOCR"
    assert result["formats"]["image_ocr"]["ocr_text_chars"] > 0
    assert result["formats"]["pdf_ocr_page"]["completed_pages"] == 8
    assert result["formats"]["pdf_ocr_page"]["ordered_pages"] == 8
    assert result["formats"]["zip_member"]["member_success"] is True
    assert legacy["external_process_seen"] is True
    assert (
        legacy["conversion_cache_reused_after_resume"]
        is True
    )
    assert legacy["office_pids_after"] == []
