from __future__ import annotations

import os
import json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QFileDialog

from local_full_text_search.rag.models import RagAnswer, SourceRef
from local_full_text_search.rag.index_builder import RagBuildSummary
from local_full_text_search.ui.rag_diagnostic_dialog import _CandidateStage
from local_full_text_search.ui.rag_page import RagPage


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_page_preserves_model_citation_ids_and_source_payload() -> None:
    _app()
    page = RagPage()
    source = SourceRef(
        source_generation=4,
        stable_chunk_id="chunk-1",
        document_id=9,
        file_id=12,
        root_id=3,
        file_path=r"C:\资料\恢复指南.pdf",
        filename="恢复指南.pdf",
        extension=".pdf",
        block_ids=(17,),
        location_text="第 2 页",
        page_number=2,
    )
    page.show_answer(
        RagAnswer(
            answer_text="依据见第二条来源 [S2]。",
            sources=(source,),
            insufficient_evidence=False,
            source_ids=("S2",),
        )
    )

    item = page.sources.item(0)
    original_payload = dict(item.data(Qt.ItemDataRole.UserRole))
    opened = []
    page.source_open_requested.connect(lambda path, location: opened.append((path, location)))
    page._open_source(item)

    assert item.text().startswith("[S2]")
    assert opened[0][0] == source.file_path
    assert opened[0][1]["page_number"] == 2
    assert item.data(Qt.ItemDataRole.UserRole) == original_payload


def test_page_shows_warning_when_answer_uses_older_ai_index() -> None:
    _app()
    page = RagPage()
    page.show_answer(RagAnswer(
        answer_text="基于旧资料的回答。",
        sources=(),
        insufficient_evidence=False,
        warnings=("全文索引已有更新；本次回答仍基于已发布的旧 AI 索引，可能不包含最新资料。",),
    ))
    assert "旧 AI 索引" in page.status_label.text()


def test_page_explains_when_existing_ai_index_is_reused() -> None:
    _app()
    page = RagPage()
    page.show_build_summary(RagBuildSummary(
        source_generation=8,
        bundle_id="bundle-8",
        chunk_count=11769,
        vector_count=11769,
        elapsed_ms=3800,
        bundle_path="C:/rag/bundle-8",
        unchanged=True,
    ))
    assert "资料未变化" in page.status_label.text()


def test_page_shows_reused_local_models_and_chroma_state() -> None:
    _app()
    page = RagPage()
    page.show_local_state({
        "models_ready": True,
        "index_ready": True,
        "chunk_count": 12,
        "vector_count": 12,
        "using_previous_index": True,
    })

    assert "已复用本地 BGE 模型" in page.status_label.text()
    assert "12 个 Chroma 向量" in page.status_label.text()
    assert "继续使用旧 AI 索引" in page.status_label.text()


def test_diagnostic_dialog_shows_candidates_context_and_key_free_request(
    tmp_path, monkeypatch
) -> None:
    app = _app()
    page = RagPage()
    page.start_diagnostics("rag-query-test", "检查点有什么用？")
    candidate = {
        "chunk_id": "chunk-1",
        "filename": "恢复手册.txt",
        "file_path": "C:/资料/恢复手册.txt",
        "location": "第 2 段",
        "source_type": "native_text",
        "text": "检查点可以恢复任务。",
        "lexical_rank": 1,
        "vector_rank": 2,
        "vector_score": 0.82,
        "fused_score": 0.03,
        "rerank_score": 0.94,
        "selected": True,
        "reason": "通过筛选",
    }
    page.show_progress({
        "stage": "checking_evidence",
        "message": "正在筛选证据",
        "lexical_candidates": [candidate],
        "lexical_terms": ["检查点", "任务恢复"],
        "lexical_fts_expression": '"检查点" OR "任务恢复"',
        "vector_candidates": [candidate],
        "fused_candidates": [candidate],
        "reranked_candidates": [candidate],
        "evidence_candidates": [candidate],
        "llm_context": "[S1]\n内容：检查点可以恢复任务。",
        "api_style": "responses",
        "provider_host": "api.example.test",
        "api_request": {
            "model": "test-model",
            "instructions": "只根据证据回答。",
            "input": "问题：检查点有什么用？\n[S1] 检查点可以恢复任务。",
        },
    })

    dialog = page.diagnostics_dialog
    assert dialog.lexical.table.rowCount() == 1
    assert dialog.rerank.table.item(0, 1).text().find("重排 0.940000") >= 0
    assert dialog.rerank.table.columnCount() == 5
    assert dialog.rerank.table.horizontalHeaderItem(1).text() == "排名与分数"
    assert dialog.rerank.table.item(0, 1).toolTip() == dialog.rerank.table.item(0, 1).text()
    assert "chunk-1" in dialog.rerank.details.toPlainText()
    assert "检查点可以恢复任务" in dialog.evidence.details.toPlainText()
    assert "[S1]" in dialog.context_text.toPlainText()
    assert "test-model" in dialog.request_text.toPlainText()
    assert "api_key" not in dialog.request_text.toPlainText().lower()
    assert "RRF" in dialog.fusion.layout().itemAt(0).widget().text()
    assert "检查点" in dialog.lexical.query_label.text()

    source = SourceRef(
        source_generation=4,
        stable_chunk_id="chunk-1",
        document_id=9,
        file_id=12,
        root_id=3,
        file_path="C:/资料/恢复手册.txt",
        filename="恢复手册.txt",
        extension=".txt",
        block_ids=(17,),
        location_text="第 2 段",
    )
    page.show_answer(RagAnswer(
        answer_text="检查点用于恢复任务 [S1]。",
        sources=(source,),
        insufficient_evidence=False,
        source_ids=("S1",),
        trace_id="trace-test",
        provider_name="OpenAI Compatible",
        model_name="test-model",
    ))
    assert "检查点用于恢复任务" in dialog.final_text.toPlainText()
    assert "test-model" in dialog.final_text.toPlainText()
    assert "恢复手册.txt" in dialog.final_text.toPlainText()
    safe_export = dialog.diagnostic_json()
    serialized_safe_export = json.dumps(safe_export, ensure_ascii=False)
    assert "C:/资料/恢复手册.txt" not in serialized_safe_export
    assert "检查点可以恢复任务。" not in serialized_safe_export
    assert "api_key" not in serialized_safe_export.lower()

    dialog.copy_text_button.click()
    copied_text = QApplication.clipboard().text()
    assert "===== 关键词召回 =====" in copied_text
    assert "===== 最终结果 =====" in copied_text
    assert "检查点用于恢复任务" in copied_text
    assert "api_key" not in copied_text.lower()

    page.show()
    dialog.show()
    app.processEvents()
    original_tab = dialog.tabs.currentIndex()
    screenshot = dialog.capture_all_tabs()
    assert not screenshot.isNull()
    assert screenshot.height() > dialog.tabs.height()
    assert dialog.tabs.currentIndex() == original_tab
    dialog.copy_screenshot_button.click()
    assert not QApplication.clipboard().pixmap().isNull()
    assert "已复制全部" in dialog.export_status.text()

    screenshot_path = tmp_path / "完整检索诊断.png"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(screenshot_path), "PNG 图片 (*.png)"),
    )
    dialog.save_screenshot_button.click()
    assert screenshot_path.exists()
    assert screenshot_path.stat().st_size > 0
    assert "已保存" in dialog.export_status.text()

    json_path = tmp_path / "检索诊断.json"
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *_args, **_kwargs: (str(json_path), "JSON 文件 (*.json)"),
    )
    dialog.save_json_button.click()
    exported = json.loads(json_path.read_text(encoding="utf-8"))
    assert exported["result"]["answer"] == "检查点用于恢复任务 [S1]。"
    assert "text" not in exported["stages"]["checking_evidence"]["evidence_candidates"][0]
    assert "file_path" not in exported["stages"]["checking_evidence"]["evidence_candidates"][0]
    assert "已保存" in dialog.export_status.text()

    page.start_diagnostics("rag-query-next", "下一个问题")
    assert dialog.lexical.table.rowCount() == 0
    assert dialog.request_text.toPlainText() == ""
    assert dialog.final_text.toPlainText() == ""
    page.diagnostics_button.click()
    assert dialog.isVisible()
    dialog.close()
    page.close()


def test_diagnostic_rank_and_score_remain_visible_with_long_chunk_id() -> None:
    app = _app()
    stage = _CandidateStage("候选诊断")
    stage.resize(700, 500)
    stage.set_candidates([{
        "chunk_id": "c" * 64,
        "filename": "很长的资料文件名称" * 8 + ".pdf",
        "location": "第 1446 页",
        "text": "相关正文",
        "vector_rank": 1,
        "vector_score": 0.641639,
    }])
    stage.show()
    app.processEvents()

    header = stage.table.horizontalHeader()
    assert header.sectionViewportPosition(1) >= 0
    assert header.sectionViewportPosition(1) + header.sectionSize(1) <= stage.table.viewport().width()
    assert "相似度 0.641639" in stage.table.item(0, 1).text()
    assert "c" * 64 in stage.details.toPlainText()
    stage.close()


def test_page_distinguishes_local_evidence_refusal_from_llm_refusal() -> None:
    _app()
    page = RagPage()
    page.start_diagnostics("rag-query-no-evidence", "资料里有答案吗？")
    page.show_answer(RagAnswer("资料中没有足够依据。", (), True))
    assert "未调用 LLM" in page.status_label.text()

    page.start_diagnostics("rag-query-model-refusal", "资料里有答案吗？")
    page.show_answer(RagAnswer(
        "资料中没有足够依据。", (), True, provider_name="OpenAI Compatible"
    ))
    assert "模型判断" in page.status_label.text()
    assert "LLM 判断" in page.diagnostics_dialog.summary.text()
