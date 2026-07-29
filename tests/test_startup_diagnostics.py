from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.services.startup_diagnostics import StartupDiagnostics


def test_startup_diagnostics_records_completed_startup(tmp_path: Path) -> None:
    diagnostics = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)

    diagnostics.begin()
    diagnostics.stage_started("加载模块")
    diagnostics.stage_completed()
    diagnostics.mark_window_visible()

    events = [json.loads(line)["event"] for line in diagnostics.log_path.read_text(encoding="utf-8").splitlines()]
    state = json.loads(diagnostics.state_path.read_text(encoding="utf-8"))
    assert "START" in events
    assert "STAGE_COMPLETED" in events
    assert "WINDOW_VISIBLE" in events
    assert state["status"] == "complete"


def test_startup_diagnostics_exposes_previous_failed_stage(tmp_path: Path) -> None:
    first = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)
    first.begin()
    first.stage_started("初始化索引数据库")
    first.stage_failed("初始化索引数据库", RuntimeError("database locked"))

    second = StartupDiagnostics(base_dir=tmp_path, timeout_seconds=60)
    second.begin()

    assert second.previous_failure is not None
    assert second.previous_failure["stage"] == "初始化索引数据库"
