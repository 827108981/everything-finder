from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.ui import main_window
from local_full_text_search.ui.main_window import MainWindow, SettingsPage


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_connection_test_does_not_save_or_clear_the_api_key() -> None:
    _app()
    settings = AppSettings()
    page = SettingsPage(settings)
    saved = []
    api_keys = []
    connection_tests = []
    page.save_requested.connect(saved.append)
    page.rag_api_key_save_requested.connect(api_keys.append)
    page.rag_connection_test_requested.connect(
        lambda current_settings, key, config_saved, key_saved: connection_tests.append(
            (current_settings, key, config_saved, key_saved)
        )
    )

    page.rag_enabled.setChecked(True)
    page.rag_api_base_url.setText("https://provider.example/v1")
    page.rag_api_model.setText("chat-model")
    page.rag_api_style.setCurrentIndex(page.rag_api_style.findData("responses"))
    page.rag_api_key.setText("do-not-write-this-to-settings.json")
    page.test_rag_connection()

    assert saved == []
    assert api_keys == []
    assert settings.rag_api_base_url != "https://provider.example/v1"
    snapshot, key, config_saved, key_saved = connection_tests[0]
    assert snapshot is not settings
    assert snapshot.rag_enabled is True
    assert snapshot.rag_api_base_url == "https://provider.example/v1"
    assert snapshot.rag_api_model == "chat-model"
    assert snapshot.rag_api_style == "responses"
    assert "do-not-write-this-to-settings.json" not in str(snapshot.to_dict())
    assert key == "do-not-write-this-to-settings.json"
    assert config_saved is False
    assert key_saved is False
    assert page.rag_api_key.text() == "do-not-write-this-to-settings.json"

    page.save_rag_api_key()
    assert api_keys == ["do-not-write-this-to-settings.json"]
    assert page.rag_api_key.text() == "do-not-write-this-to-settings.json"


def test_rag_index_batch_controls_are_saved() -> None:
    _app()
    settings = AppSettings()
    page = SettingsPage(settings)
    saved = []
    page.save_requested.connect(saved.append)

    page.rag_embedding_batch_size.setValue(48)
    page.rag_index_write_batch_size.setValue(192)
    page.apply()

    assert settings.rag_embedding_batch_size == 48
    assert settings.rag_index_write_batch_size == 192
    assert settings.rag_answer_review_enabled is True
    assert saved == [settings]


def test_answer_review_setting_is_saved() -> None:
    _app()
    settings = AppSettings(rag_answer_review_enabled=False)
    page = SettingsPage(settings)
    saved = []
    page.save_requested.connect(saved.append)

    page.rag_answer_review_enabled.setChecked(True)
    page.apply()

    assert settings.rag_answer_review_enabled is True
    assert saved == [settings]


def test_question_is_blocked_when_tested_api_settings_are_not_saved(monkeypatch) -> None:
    _app()
    settings = AppSettings(rag_enabled=True, rag_api_model="5.6 Sol")
    page = SettingsPage(settings)
    page.rag_api_model.setText("gpt-5.6-sol")
    messages = []
    events = []
    fake_window = SimpleNamespace(
        settings=settings,
        settings_page=page,
        rag_page=SimpleNamespace(show_failure=messages.append),
    )
    monkeypatch.setattr(
        main_window,
        "_log_ui_lifecycle_event",
        lambda event, **fields: events.append((event, fields)),
    )

    assert page.has_unsaved_rag_api_settings() is True
    MainWindow.ask_rag(fake_window, "免疫发光检测原理是什么")

    assert len(messages) == 1
    assert "保存设置" in messages[0]
    assert "诊断编号" in messages[0]
    assert settings.rag_api_model == "5.6 Sol"
    assert events[-1][0] == "rag.query.rejected"
    assert events[-1][1]["error_code"] == "RAG_API_SETTINGS_UNSAVED"
    assert events[-1][1]["saved_model"] == "5.6 Sol"
    assert events[-1][1]["draft_model"] == "gpt-5.6-sol"
