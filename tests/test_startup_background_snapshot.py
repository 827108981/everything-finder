from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.ui.main_window import MainWindow


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _wait_until(
    predicate: object,
    *,
    timeout: float = 4.0,
) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())  # type: ignore[operator]


def _window(tmp_path: Path) -> MainWindow:
    _app()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root = tmp_path / "files"
    root.mkdir()
    database.add_root(root)
    settings = AppSettings(
        enable_ocr=False,
        monitor_file_changes=False,
    )
    settings_service = SettingsService(tmp_path / "settings.json")
    settings_service.save(settings)
    window = MainWindow(
        database,
        settings,
        settings_service,
        defer_initial_refresh=True,
    )
    window.show()
    return window


def test_deferred_startup_snapshot_keeps_qt_event_loop_responsive(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    original = DatabaseManager.index_readiness

    def deliberately_slow_readiness(self: DatabaseManager) -> dict[str, object]:
        time.sleep(0.5)
        return original(self)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        DatabaseManager,
        "index_readiness",
        deliberately_slow_readiness,
    )
    window = _window(tmp_path)
    heartbeat = {"count": 0}
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(
        lambda: heartbeat.__setitem__("count", heartbeat["count"] + 1)
    )
    timer.start()

    assert _wait_until(lambda: heartbeat["count"] >= 5, timeout=0.35)
    assert window.database_snapshot_thread is not None
    assert _wait_until(
        lambda: window.database_snapshot_thread is None,
        timeout=4.0,
    )

    timer.stop()
    window.hide()


def test_startup_incremental_scan_waits_for_snapshot_completion(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    original = DatabaseManager.index_readiness

    def deliberately_slow_readiness(self: DatabaseManager) -> dict[str, object]:
        time.sleep(0.35)
        return original(self)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        DatabaseManager,
        "index_readiness",
        deliberately_slow_readiness,
    )
    window = _window(tmp_path)
    scan_calls: list[bool] = []
    monkeypatch.setattr(  # type: ignore[attr-defined]
        window,
        "start_scan",
        lambda *args, **kwargs: scan_calls.append(True),
    )

    window.schedule_startup_incremental_scan()
    _wait_until(lambda: False, timeout=0.15)
    assert scan_calls == []
    assert _wait_until(lambda: bool(scan_calls), timeout=4.0)

    window.hide()
