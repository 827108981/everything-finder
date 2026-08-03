from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.corpus_validation import (
    compare_index_databases,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.eta_replay import load_replay_events
from local_full_text_search.core.index_manager import IndexManager


def _index(root: Path, database_path: Path) -> None:
    database = DatabaseManager(database_path)
    database.initialize()
    root_id = database.add_root(root)
    summary = IndexManager(
        database,
        AppSettings(enable_ocr=False),
    ).index_root(root_id)
    assert summary.failed == 0


def test_compare_index_accepts_identical_independent_indexes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "golden.txt").write_text(
        "拔掉 3 个传感器\nDispersion",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    _index(root, baseline)
    _index(root, candidate)
    query_file = tmp_path / "queries.json"
    query_file.write_text(
        json.dumps(["拔掉 3 个传感器", "Dispersion"], ensure_ascii=False),
        encoding="utf-8",
    )

    report = compare_index_databases(
        baseline,
        candidate,
        query_file,
    )

    assert report["passed"] is True
    assert report["file_inventory_equal"] is True
    assert report["content_digest_equal"] is True
    assert report["query_results_equal"] is True


def test_compare_index_rejects_missing_content_and_query_hits(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    source = root / "golden.txt"
    source.write_text("拔掉 3 个传感器", encoding="utf-8")
    baseline = tmp_path / "baseline.db"
    candidate = tmp_path / "candidate.db"
    _index(root, baseline)
    source.write_text("内容已改变", encoding="utf-8")
    _index(root, candidate)
    query_file = tmp_path / "queries.json"
    query_file.write_text(
        json.dumps(["拔掉 3 个传感器"], ensure_ascii=False),
        encoding="utf-8",
    )

    report = compare_index_databases(
        baseline,
        candidate,
        query_file,
    )

    assert report["passed"] is False
    assert report["content_digest_equal"] is False
    assert report["query_results_equal"] is False


def test_p1_04r_cold_command_persists_run_metrics_for_eta_and_audit(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    root.mkdir()
    for index in range(4):
        (root / f"eta-{index}.txt").write_text(
            f"COLD_ETA_EVIDENCE_{index}",
            encoding="utf-8",
        )
    output = tmp_path / "cold-result.json"
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "app.py"),
            "--benchmark-cold-index",
            str(root),
            "--output",
            str(output),
            "--performance",
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["index_run"]["status"] == "complete"
    assert payload["metrics"] == payload["index_run"]["summary"]["metrics"]
    events = load_replay_events(payload)
    assert events[-1].event_type == "finish"
    assert any(event.event_type == "completion" for event in events)
    assert "resource_metrics" in payload["metrics"]
    assert "ocr_metrics" in payload["metrics"]
    assert "pause_metrics" in payload["metrics"]
    assert payload["performance_mode"] is True
    assert payload["effective_profile"]["mode"] == "performance"
    assert payload["settings"]["index_performance_preset"] == "fastest"
    assert payload["metrics"]["execution_mode"] == "performance"
