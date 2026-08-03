from __future__ import annotations

from app import INDEX_STATUS_LAYOUT_CASES

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QVBoxLayout

from local_full_text_search.ui.main_window import (
    FailedPage,
    IndexPage,
    format_index_scope_status,
    format_active_phase,
    format_remaining_single,
    performance_mode_notice,
    scheduler_diagnostic_label,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_index_status_uses_two_rows_and_keeps_file_details_separate() -> None:
    _app()
    page = IndexPage()
    page.set_task_running(True)
    page.set_scan_progress(
        209,
        309,
        2,
        r"E:\very\long\folder\微信图片_20260421082832_32_45.jpg",
        phase_label="正在解析并写入索引",
        eta_text="预计剩余约 11 分钟",
        active_elapsed_seconds=252,
        active_queue="ocr",
        active_file_count=2,
        active_phase="ocr_recognize_original_regions",
        active_completed_units=96,
        active_total_units=240,
        excluded_video=22,
    )

    assert isinstance(page.task_strip.layout(), QVBoxLayout)
    assert page.task_strip.layout().count() == 2
    assert "209/309" in page.task_label.text()
    assert page.task_label.text() == "普通模式·总体 209/309·失败 2·排除视频 22"
    assert "微信图片" not in page.task_label.text()
    assert "微信图片" in page.task_file.text()
    assert page.task_file.toolTip().endswith("微信图片_20260421082832_32_45.jpg")
    assert page.task_runtime.text().startswith("OCR 已运行")
    assert page.task_units.text() == "96/240"


def test_pdf_lane_ocr_phase_is_labeled_as_ocr_runtime() -> None:
    _app()
    page = IndexPage()
    page.set_scan_progress(
        35,
        117,
        0,
        r"E:\manuals\large-scanned-manual.pdf",
        active_elapsed_seconds=252,
        active_queue="pdf",
        active_phase="pdf_ocr_tile_recognize_microbatch",
        active_completed_units=35,
        active_total_units=117,
    )

    assert page.task_runtime.text().startswith("OCR 已运行")
    assert page.task_phase.text() == "OCR 批量识别文字区域"


def test_active_phase_has_readable_labels_and_safe_fallback() -> None:
    assert format_active_phase("pdf_ocr_tile_detect") == "OCR 分块检测文字区域"
    assert (
        format_active_phase("pdf_ocr_region_300dpi")
        == "PDF 300 DPI 精细识别"
    )
    assert format_active_phase("custom_phase") == "custom phase"


def test_single_eta_never_displays_a_range() -> None:
    assert format_remaining_single(0, False) == "正在估算…"
    assert format_remaining_single(40, True) == "预计剩余约 40 秒"
    assert format_remaining_single(601, True) == "预计剩余约 11 分钟"
    assert "-" not in format_remaining_single(601, True)


def test_mode_switch_is_enabled_only_after_safe_pause_confirmation() -> None:
    _app()
    page = IndexPage()
    requested: list[bool] = []
    pauses: list[str] = []
    page.mode_switch_requested.connect(requested.append)
    page.pause_requested.connect(lambda: pauses.append("pause"))
    page.set_task_running(True)

    page._performance_mode_clicked()
    assert requested == []

    page._toggle_pause()
    assert pauses == ["pause"]
    assert page.pause_state == "pausing"
    assert not page.start_button.isEnabled()
    assert not page.performance_button.isEnabled()

    page.set_pause_state("paused", "普通模式 · 已暂停")
    assert page.start_button.isEnabled()
    assert page.performance_button.isEnabled()
    page._performance_mode_clicked()
    page._normal_mode_clicked()

    assert requested == [True, False]
    assert page.task_eta.text() == "已暂停 · 继续后重新估算"


def test_performance_mode_notice_discloses_aggressive_idle_resource_use() -> None:
    notice = performance_mode_notice()

    assert "电脑空闲" in notice
    assert "CPU" in notice
    assert "内存" in notice
    assert "自动降低" in notice
def test_u0_01v_declares_every_required_high_dpi_layout_case() -> None:
    assert INDEX_STATUS_LAYOUT_CASES == (
        (1280, 800, 1.00, "ocr_running"),
        (1280, 800, 1.25, "ocr_running"),
        (1280, 800, 1.50, "ocr_running"),
        (1366, 768, 1.00, "pausing"),
        (1920, 1080, 1.50, "paused_switched"),
    )


def test_p1_03r_maps_every_real_scheduler_recovery_state() -> None:
    assert scheduler_diagnostic_label("reclaiming_no_progress") == "正在回收无进展任务"
    assert scheduler_diagnostic_label("terminating_worker") == "正在终止 worker"
    assert scheduler_diagnostic_label("rebuilding_pool") == "正在重建解析进程池"
    assert scheduler_diagnostic_label("checkpoint_resumed") == "已从检查点恢复"
    assert (
        scheduler_diagnostic_label("same_stall_retry_stopped")
        == "同一卡点重复发生，已停止自动重试"
    )


def test_p1_03r_displays_recovery_and_parallel_lane_progress() -> None:
    _app()
    page = IndexPage()
    page.set_scan_progress(
        80,
        309,
        1,
        r"E:\manuals\slow.pdf",
        active_elapsed_seconds=320,
        active_queue="pdf",
        active_file_count=4,
        active_phase="pdf_ocr_tile_recognize_microbatch",
        active_completed_units=37,
        active_total_units=240,
        no_progress_seconds=21,
        retry_count=1,
        diagnostic_state="rebuilding_pool",
        diagnostic_reason="PDF OCR 页在安全游标 37 无进展",
        representative_is_slowest=True,
        other_active_lane_count=3,
        other_recent_progress_seconds=1,
    )

    assert "正在重建解析进程池" in page.task_phase.text()
    assert "另有 3 个车道仍在处理" in page.task_phase.toolTip()
    assert "最近有效进展 1 秒前" in page.task_phase.toolTip()
    assert "PDF OCR 页在安全游标 37 无进展" in page.task_phase.toolTip()


def test_fts_02r_failed_page_separates_scope_states_and_supports_bulk_actions() -> None:
    _app()
    page = FailedPage()
    excluded_ids: list[list[int]] = []
    restored_ids: list[list[int]] = []
    page.exclude_requested.connect(excluded_ids.append)
    page.restore_requested.connect(restored_ids.append)
    blocking = {
        "id": 11,
        "path": r"E:\scope\broken.zip",
        "filename": "broken.zip",
        "extension": ".zip",
        "parse_status": "failed",
        "parse_error_code": "ZIP_CORRUPTED",
        "parse_error_message": "bad central directory",
        "parser_name": "zip",
        "indexed_at": "2026-07-31T10:00:00Z",
        "progress_phase": "zip_members",
        "progress_cursor": "4",
        "recovery_advice": "repair or exclude",
    }
    excluded = {
        **blocking,
        "id": 21,
        "file_id": 11,
        "scope_state": "manual_excluded",
        "reason": "field file cannot be repaired",
        "created_at": "2026-07-31T10:01:00Z",
        "current_parse_status": "failed",
        "current_error_code": "ZIP_CORRUPTED",
        "current_error_message": "bad central directory",
    }
    metadata = {
        **blocking,
        "id": 31,
        "path": r"E:\scope\empty.zip",
        "filename": "empty.zip",
        "parse_status": "metadata_only",
        "parse_error_code": "ZIP_NO_SUPPORTED_MEMBER",
    }

    page.set_rows([blocking], [excluded], [metadata])

    assert page.scope_tabs.tabText(0) == "阻断项 1"
    assert page.scope_tabs.tabText(1) == "已人工排除 1"
    assert page.scope_tabs.tabText(2) == "仅元数据完成 1"
    checkbox = page.table.item(0, 0)
    checkbox.setCheckState(Qt.CheckState.Checked)
    page.exclude_button.click()
    assert excluded_ids == [[11]]

    page.scope_tabs.setCurrentIndex(1)
    checkbox = page.table.item(0, 0)
    checkbox.setCheckState(Qt.CheckState.Checked)
    page.restore_button.click()
    assert restored_ids == [[11]]
    assert "field file cannot be repaired" in page.table.item(0, 7).text()


def test_failed_page_exposes_force_complete_control() -> None:
    app = _app()
    page = FailedPage()
    page.resize(1040, 720)
    requests: list[bool] = []
    page.force_complete_requested.connect(lambda: requests.append(True))

    page.set_rows([])
    page.show()
    app.processEvents()
    assert page.force_complete_button.isHidden() is False
    assert page.force_complete_button.isEnabled() is False
    assert page.force_complete_button.toolTip() == "当前索引已经就绪"

    page.set_rows(
        [
            {
                "id": 41,
                "path": r"E:\scope\stale.pdf",
                "filename": "stale.pdf",
                "extension": ".pdf",
                "parse_status": "pending",
            }
        ]
    )
    page.set_readiness(
        {
            "ready": False,
            "repairable": True,
            "not_ready_reasons": ["blocking_files"],
        }
    )
    page.show()
    app.processEvents()

    assert page.force_complete_button.text() == "强力完成本次索引"
    assert page.force_complete_button.isHidden() is False
    button_bottom = page.force_complete_button.mapTo(
        page,
        page.force_complete_button.rect().bottomLeft(),
    ).y()
    assert button_bottom < page.table.geometry().top()
    assert page.force_complete_button.geometry().right() <= page.width()
    assert page.force_complete_button.isEnabled() is True
    assert "1 个阻断项" in page.force_complete_button.toolTip()
    page.set_index_running(True)
    assert page.force_complete_button.isEnabled() is False
    assert "正在运行" in page.force_complete_button.toolTip()
    page.set_index_running(False)
    page.force_complete_button.click()
    assert requests == [True]
    page.close()


def test_failed_page_enables_force_complete_for_zero_blocker_repair() -> None:
    app = _app()
    page = FailedPage()
    page.set_rows([])
    page.set_readiness(
        {
            "ready": False,
            "repairable": True,
            "not_ready_reasons": [
                "unfinished_tasks",
                "content_fts_dirty",
                "unpublished_candidate",
            ],
        }
    )
    page.show()
    app.processEvents()

    assert page.force_complete_button.isHidden() is False
    assert page.force_complete_button.isEnabled() is True
    assert "残留解析任务" in page.force_complete_button.toolTip()
    assert "全文索引待发布" in page.force_complete_button.toolTip()
    page.close()


def test_fts_02r_scope_completion_status_discloses_manual_exclusions() -> None:
    text = format_index_scope_status(
        {
            "complete_files": 158,
            "eligible_files": 158,
            "blocking_files": 0,
            "manual_excluded_files": 28,
            "video_excluded": 2,
            "ready": True,
        }
    )

    assert "当前范围完成 158/158" in text
    assert "已人工排除 28 个" in text
    assert "排除视频 2 个" in text
    assert "全部文件解析成功" not in text


def test_scope_status_discloses_zero_blocker_repair_reasons() -> None:
    text = format_index_scope_status(
        {
            "complete_files": 7387,
            "eligible_files": 7387,
            "blocking_files": 0,
            "unfinished_tasks": 12,
            "unpublished_candidates": 1,
            "content_fts_dirty": True,
            "ready": False,
        }
    )

    assert "文件已完成 7387/7387" in text
    assert "索引状态待修复" in text
    assert "残留任务 12" in text
    assert "未发布索引 1" in text
    assert "全文索引待发布" in text
    assert "待处理 0" not in text
