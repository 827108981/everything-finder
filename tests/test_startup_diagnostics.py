from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.services.startup_diagnostics import (
    StartupDiagnostics,
    classify_startup_error,
    frozen_runtime_error_message,
    frozen_runtime_issues,
)


def test_startup_diagnostics_records_completed_startup(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)

    diagnostics.begin()
    diagnostics.stage_started("加载模块")
    diagnostics.stage_completed()
    diagnostics.mark_window_visible()

    running_state = json.loads(diagnostics.state_path.read_text(encoding="utf-8"))
    assert running_state["status"] == "running"
    diagnostics.mark_clean_exit("test_completed", exit_code=0)

    records = [
        json.loads(line)
        for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [record["event"] for record in records]
    state = json.loads(diagnostics.state_path.read_text(encoding="utf-8"))
    assert "START" in events
    assert "STAGE_COMPLETED" in events
    assert "WINDOW_VISIBLE" in events
    assert "APP_CLEAN_EXIT" in events
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


def test_startup_diagnostics_distinguishes_interrupt_from_clean_exit(
    tmp_path: Path,
) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)
    diagnostics.begin()
    diagnostics.stage_started("加载图形界面")
    diagnostics.mark_interrupted("IDE_OR_CONSOLE_INTERRUPT", exit_code=130)

    records = [
        json.loads(line)
        for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["event"] == "APP_INTERRUPTED"
    assert records[-1]["source"] == "IDE_OR_CONSOLE_INTERRUPT"
    assert json.loads(diagnostics.state_path.read_text(encoding="utf-8"))["status"] == "interrupted"


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


def test_slow_stage_only_records_diagnostics_without_modal_dialog(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=1)
    diagnostics.begin()
    diagnostics.stage_started("初始化索引数据库")
    with patch.object(diagnostics, "show_native_error") as show_error:
        diagnostics._report_slow_stage("初始化索引数据库")
    show_error.assert_not_called()
    records = [
        json.loads(line)
        for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(record["event"] == "STAGE_SLOW" for record in records)


def test_current_schema_fast_start_does_not_rebuild_fts(tmp_path: Path) -> None:
    from local_full_text_search.core.database import DatabaseManager

    path = tmp_path / "search.db"
    database = DatabaseManager(path)
    database.initialize()
    database.initialize(startup_fast=True)
    assert database._startup_fast_path_ready()


def test_fast_start_defers_additive_schema_repair_without_copying_large_database(
    tmp_path: Path,
) -> None:
    from local_full_text_search.core.database import DatabaseManager

    path = tmp_path / "search.db"
    database = DatabaseManager(path)
    database.initialize()
    with database.connect() as con:
        con.execute("DROP INDEX idx_content_blocks_index_version")
        con.execute(
            """
            CREATE TABLE short_tokens (
                token TEXT NOT NULL,
                block_id INTEGER NOT NULL,
                position_count INTEGER DEFAULT 1,
                PRIMARY KEY(token, block_id)
            )
            """
        )

    reopened = DatabaseManager(path)
    reopened.initialize(startup_fast=True)

    assert reopened.startup_migration_deferred is True
    assert not list(tmp_path.glob("search.schema-v9*.backup.db"))
    assert reopened.list_roots() == []

    progress: list[dict[str, object]] = []
    reopened.initialize(progress_callback=progress.append)
    assert reopened._startup_fast_path_ready()
    assert any(
        item.get("stage") == "database_migration_schema" for item in progress
    )


def test_fast_start_does_not_open_existing_database_on_the_gui_thread(
    tmp_path: Path,
) -> None:
    from local_full_text_search.core.database import DatabaseManager

    path = tmp_path / "search.db"
    database = DatabaseManager(path)
    database.initialize()
    reopened = DatabaseManager(path)
    reopened._startup_fast_path_ready = lambda: (_ for _ in ()).throw(
        AssertionError("GUI fast startup must not open SQLite")
    )

    reopened.initialize(startup_fast=True)

    assert reopened.startup_migration_deferred is True


def test_frozen_runtime_check_reports_missing_boot_artifacts(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "_internal"
    runtime.mkdir()
    (tmp_path / "app.exe").write_bytes(b"exe")
    with patch("local_full_text_search.services.startup_diagnostics.sys.frozen", True, create=True), patch(
        "local_full_text_search.services.startup_diagnostics.sys.executable",
        str(tmp_path / "app.exe"),
    ), patch(
        "local_full_text_search.services.startup_diagnostics.sys._MEIPASS",
        str(runtime),
        create=True,
    ):
        issues = frozen_runtime_issues()

    assert "Python 嵌入运行库" in issues
    assert "Python 标准库" in issues
    assert "应用模块" in issues
    assert "Qt 运行库" in issues
    assert "请退出程序并重新安装完整发布目录" in frozen_runtime_error_message(issues)
