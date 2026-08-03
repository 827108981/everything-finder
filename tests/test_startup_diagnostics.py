from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.services.startup_diagnostics import (
    StartupDiagnostics,
    classify_startup_error,
)


def test_startup_diagnostics_records_completed_startup(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)

    diagnostics.begin()
    diagnostics.stage_started("加载模块")
    diagnostics.stage_completed()
    diagnostics.mark_window_visible()

    records = [
        json.loads(line)
        for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]
    state = json.loads(diagnostics.state_path.read_text(encoding="utf-8"))
    assert "START" in events
    assert "STAGE_COMPLETED" in events
    assert "WINDOW_VISIBLE" in events
    assert state["status"] == "complete"
    for record in records:
        assert record["app_version"]
        assert record["executable"]
        assert record["cwd"]
        assert record["settings_path"].endswith("settings.json")
        assert record["database_path"].endswith("search_index.db")
        assert record["log_path"] == str(diagnostics.log_path)
        assert record["last_status"]


def test_startup_diagnostics_exposes_previous_failed_stage(tmp_path: Path) -> None:
    first = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)
    first.begin()
    first.stage_started("初始化索引数据库")
    first.stage_failed("初始化索引数据库", RuntimeError("database locked"))

    second = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)
    second.begin()

    assert second.previous_failure is not None
    assert second.previous_failure["stage"] == "初始化索引数据库"


def test_faulthandler_is_enabled_before_start_record(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)

    with patch(
        "local_full_text_search.services.startup_diagnostics.faulthandler.enable"
    ) as enable:
        diagnostics.begin()

    enable.assert_called_once()
    events = [
        json.loads(line)["event"]
        for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[:2] == ["FAULTHANDLER_ENABLED", "START"]


def test_startup_error_categories_are_actionable(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path)
    diagnostics.begin()

    assert classify_startup_error(
        ModuleNotFoundError("No module named PySide6"),
        "加载图形界面",
    )[0] == "CRITICAL_COMPONENT_MISSING"
    assert classify_startup_error(
        PermissionError("Access is denied"),
        "初始化索引数据库",
    )[0] == "USER_DATA_NOT_WRITABLE"
    assert classify_startup_error(
        sqlite3.OperationalError("database is locked"),
        "初始化索引数据库",
    )[0] == "DATABASE_LOCKED"
    assert classify_startup_error(
        sqlite3.DatabaseError("database disk image is malformed"),
        "初始化索引数据库",
    )[0] == "DATABASE_CORRUPT"
    assert classify_startup_error(
        OSError("Could not load Qt platform DLL"),
        "加载图形界面",
    )[0] == "QT_OR_DLL_LOAD_FAILED"

    diagnostics.current_stage = "初始化索引数据库"
    message = diagnostics.format_error_message(
        sqlite3.OperationalError("database is locked")
    )
    assert "DATABASE_LOCKED" in message
    assert str(diagnostics.log_path) in message
