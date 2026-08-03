from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.core.hang_validation import (
    HANG_RECOVERY_SCENARIOS,
    run_hang_recovery_validation,
)


def test_s0_03r_real_hang_recovery_matrix_recycles_and_runs_health_check(
    tmp_path: Path,
) -> None:
    output = tmp_path / "hang-recovery.json"

    report = run_hang_recovery_validation(
        output,
        timeout_seconds=0.35,
    )

    assert output.is_file()
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert set(persisted["scenarios"]) == set(HANG_RECOVERY_SCENARIOS)
    assert report["passed"] is True
    assert report["residual_pids"] == []
    for result in report["scenarios"].values():
        assert result["timed_out"] is True
        assert result["old_pid_exited"] is True
        assert result["healthy_follow_up"] is True
        assert "worker_recycle" in result["eta_event_types"]
