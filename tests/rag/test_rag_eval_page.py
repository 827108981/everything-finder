from __future__ import annotations

import json

from PySide6.QtWidgets import QApplication, QPlainTextEdit

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.search_index_coordinator import SearchIndexCoordinator
from local_full_text_search.services.settings_service import SettingsService
import local_full_text_search.ui.main_window as main_window_module
from local_full_text_search.ui.main_window import MainWindow, PAGE_INDEX
from local_full_text_search.ui.rag_eval_page import RagEvalPage


def test_rag_eval_page_offers_modes_and_stage_views() -> None:
    app = QApplication.instance() or QApplication([])
    page = RagEvalPage()

    assert page.tabs.count() == 5
    assert page.tabs.tabText(0) == "概览"
    assert page.tabs.tabText(1) == "逐题结果"
    assert page.tabs.tabText(2) == "检索链路"
    assert page.tabs.tabText(4) == "历史对比"
    assert page.mode.itemData(0) == "retrieval-only"
    assert page.mode.itemData(1) == "full"
    assert page.stop_button.isEnabled() is False
    page.deleteLater()
    app.processEvents()


def test_stage_table_keeps_ranking_columns_visible_and_shows_cross_stage_trace() -> None:
    app = QApplication.instance() or QApplication([])
    page = RagEvalPage()
    result = {
        "candidate_stages": {
            "fusion": [{"chunk_id": "gold", "filename": "manual.pdf", "location": "第 4 页", "fused_score": 0.04}],
            "rerank": [{"chunk_id": "gold", "filename": "manual.pdf", "location": "第 4 页", "rerank_score": 0.91}],
            "evidence": [
                {"chunk_id": "other", "selected": False, "final_rank": None, "reason": "未通过门槛"},
                {"chunk_id": "gold", "selected": True, "final_rank": 1, "reason": "通过证据门禁"},
            ],
        },
        "gold_mappings": [
            {"kind": "positive", "confidence": "exact", "chunk_ids": ["gold"]}
        ],
    }

    page._fill_stage_table("evidence", result)

    assert page.evidence_table.horizontalHeaderItem(0).text() == "阶段名次"
    assert page.evidence_table.horizontalHeaderItem(5).text() == "向量名次"
    assert page.evidence_table.item(0, 0).text() == "未入选"
    assert page.evidence_table.item(1, 0).text() == "1"
    page.evidence_table.selectRow(1)
    app.processEvents()
    details = QPlainTextEdit()
    page._show_candidate_detail("evidence", page.evidence_table, details)
    assert "融合 #1" in details.toPlainText()
    page.deleteLater()
    app.processEvents()


def test_main_window_routes_to_rag_eval_page(tmp_path, monkeypatch) -> None:
    app = QApplication.instance() or QApplication([])
    # 关闭主窗口会永久停止应用级协调器；测试使用独立实例，避免污染后续索引测试。
    coordinator = SearchIndexCoordinator()
    monkeypatch.setattr(
        main_window_module,
        "get_search_index_coordinator",
        lambda: coordinator,
    )
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    window = MainWindow(
        database,
        AppSettings(monitor_file_changes=False),
        SettingsService(tmp_path / "settings.json"),
        defer_initial_refresh=True,
    )

    window.switch_page("rag_eval")

    assert window.stack.currentIndex() == PAGE_INDEX["rag_eval"]
    assert window.stack.currentWidget() is window.rag_eval_page
    assert window.sidebar.buttons["rag_eval"].text() == "RAG 批量评测"
    assert window.top_bar.title.text() == "RAG 批量评测"
    window.close()
    window.deleteLater()
    app.processEvents()


def test_history_comparison_reports_stage_deltas(tmp_path) -> None:
    app = QApplication.instance() or QApplication([])
    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"
    old_dir.mkdir()
    new_dir.mkdir()
    for directory, run_id, hit_rate in ((old_dir, "old-run", 0.5), (new_dir, "new-run", 0.75)):
        (directory / "run.json").write_text(json.dumps({
            "run_id": run_id, "dataset_sha256": "same", "ai_bundle_id": "bundle",
            "mode": "retrieval-only", "settings": {"limits": {"fusion_top_k": 30}},
        }), encoding="utf-8")
        (directory / "summary.json").write_text(json.dumps({
            "behavior_accuracy": None,
            "stages": {"fusion": {
                "gold_hit_rate": hit_rate, "mean_evidence_recall": hit_rate,
                "hard_negative_candidate_count": 1,
            }},
        }), encoding="utf-8")
    page = RagEvalPage()
    page.history_old_path.setText(str(old_dir))
    page.history_new_path.setText(str(new_dir))

    page._compare_runs()

    assert "old-run" in page.history_result.toPlainText()
    assert "+25.0 个百分点" in page.history_result.toPlainText()
    page.deleteLater()
    app.processEvents()
