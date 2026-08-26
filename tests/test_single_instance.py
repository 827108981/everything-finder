from __future__ import annotations

import os
import uuid

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from local_full_text_search.services.single_instance import (
    SingleInstanceGuard,
    activate_window,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_duplicate_instance_notifies_primary_without_acquiring_server() -> None:
    app = _app()
    server_name = f"lfts-test-{uuid.uuid4().hex}"
    primary = SingleInstanceGuard(server_name)
    duplicate = SingleInstanceGuard(server_name)
    activations: list[bool] = []
    primary.activation_requested.connect(lambda: activations.append(True))
    try:
        assert primary.acquire() is True
        assert duplicate.acquire() is False
        for _ in range(10):
            app.processEvents()
            if activations:
                break
        assert activations == [True]
        assert primary.owns_server is True
        assert duplicate.owns_server is False
    finally:
        duplicate.close()
        primary.close()


def test_activate_window_restores_and_focuses_existing_window() -> None:
    calls: list[str] = []

    class Window:
        def isMinimized(self) -> bool:
            return True

        def showNormal(self) -> None:
            calls.append("showNormal")

        def show(self) -> None:
            calls.append("show")

        def raise_(self) -> None:
            calls.append("raise")

        def activateWindow(self) -> None:
            calls.append("activate")

    activate_window(Window(), log=calls.append)

    assert calls[:3] == ["showNormal", "raise", "activate"]
    assert "已有窗口已恢复并请求置前" in calls


def test_application_checks_single_instance_before_database_initialization() -> None:
    source = open("app.py", encoding="utf-8").read()
    assert source.index('diagnostics.stage_started("检查重复启动")') < source.index(
        'diagnostics.stage_started("读取设置")'
    )
    assert source.index('diagnostics.stage_started("检查重复启动")') < source.index(
        'diagnostics.stage_started("初始化索引数据库")'
    )
