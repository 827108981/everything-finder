from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.models import Citation, RagAnswer, SourceRef, SourceSpan
from local_full_text_search.ui.main_window import SettingsPage
from local_full_text_search.ui.rag_page import RagPage


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_rag_page_builds_query_and_opens_a_source(tmp_path) -> None:
    _app()
    page = RagPage(AppSettings(rag_enabled=True))
    page.set_roots([{"id": 5, "path": str(tmp_path), "enabled": 1}])
    page.root_combo.setCurrentIndex(1)
    page.pdf_check.setChecked(False)
    page.txt_check.setChecked(True)
    page.question.setText("索引中断后怎样恢复？")
    requested = []
    page.ask_requested.connect(requested.append)
    page.submit_question()
    assert len(requested) == 1
    assert requested[0].root_ids == (5,)
    assert requested[0].extensions == (".docx", ".txt")
    assert requested[0].question == "索引中断后怎样恢复？"

    source_path = tmp_path / "notes.txt"
    source_path.write_text("恢复步骤", encoding="utf-8")
    source = SourceRef(
        generation=7,
        stable_chunk_id="chunk-1",
        file_id=1,
        root_id=5,
        document_id=2,
        file_path=str(source_path),
        filename=source_path.name,
        extension=".txt",
        spans=(SourceSpan(11, 0, 0, 4),),
        block_index_start=0,
        block_index_end=0,
        location_text="第 3 行",
        line_start=3,
        line_end=3,
    )
    page.set_answer(
        RagAnswer(
            "answered",
            "按检查点继续构建。[S1]",
            (Citation("S1", source, "hash", "恢复步骤"),),
            "trace-1",
            7,
        )
    )
    opened = []
    page.open_location_requested.connect(lambda path, location: opened.append((path, location)))
    card = page.sources_layout.itemAt(0).widget()
    button = card.findChildren(type(page.ask_button))[0]
    button.click()
    assert opened[0] == (str(source_path), source)
    assert page.answer.toPlainText() == "按检查点继续构建。[S1]"


def test_settings_page_keeps_api_key_out_of_app_settings(tmp_path) -> None:
    _app()
    settings = AppSettings()
    page = SettingsPage(settings)
    page.rag_enabled.setChecked(True)
    page.rag_llm_base_url.setText("https://api.example.test/v1")
    page.rag_llm_model.setText("example-chat")
    page.rag_llm_api_key.setText("secret-from-user")
    saved_keys = []
    page.api_key_save_requested.connect(saved_keys.append)
    page.rag_llm_api_key_save.click()
    page.apply()
    assert saved_keys == ["secret-from-user"]
    assert page.rag_llm_api_key.text() == ""
    assert settings.rag_llm_base_url == "https://api.example.test/v1"
    assert settings.rag_llm_model == "example-chat"
    assert "secret-from-user" not in str(settings.to_dict())
    assert "rag_llm_api_key" not in settings.to_dict()
