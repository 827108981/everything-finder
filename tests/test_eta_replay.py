from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from local_full_text_search.core.eta_replay import (
    EtaHistoryContext,
    EtaHistorySample,
    EtaHistoryStore,
    EtaReplayEvent,
    replay_eta,
)


def _context(**changes: object) -> EtaHistoryContext:
    values: dict[str, object] = {
        "parser_name": "pdf",
        "parser_version": "7",
        "ocr_enabled": True,
        "ocr_strategy": "adaptive-960-regions-v2",
        "ocr_model_fingerprint": "model-sha256",
        "execution_mode": "performance",
        "hardware_tier": "cpu-6c12t-16g",
        "disk_class": "local-ssd",
        "extension": ".pdf",
        "size_bucket": "100m-1g",
        "page_bucket": "101-500",
    }
    values.update(changes)
    return EtaHistoryContext(**values)


def test_u0_02v_history_context_isolates_every_required_dimension() -> None:
    baseline = _context()
    dimensions = {
        "parser_name": "image",
        "parser_version": "8",
        "ocr_enabled": False,
        "ocr_strategy": "disabled",
        "ocr_model_fingerprint": "another-model",
        "execution_mode": "normal",
        "hardware_tier": "cpu-4c8t-8g",
        "disk_class": "network",
        "extension": ".jpg",
        "size_bucket": "10m-100m",
        "page_bucket": "21-100",
    }

    for field_name, alternative in dimensions.items():
        assert _context(**{field_name: alternative}).key != baseline.key


def test_u0_02v_history_store_never_crosses_context_boundaries(
    tmp_path: Path,
) -> None:
    store = EtaHistoryStore(tmp_path / "eta_history.json")
    local_ocr = _context()
    network_ocr = _context(disk_class="network")
    local_without_ocr = _context(ocr_enabled=False, ocr_strategy="disabled")
    store.add(EtaHistorySample(local_ocr, seconds_per_cost=1.25, sample_count=8))
    store.add(EtaHistorySample(network_ocr, seconds_per_cost=9.0, sample_count=20))
    store.add(EtaHistorySample(local_without_ocr, seconds_per_cost=0.1, sample_count=20))

    reloaded = EtaHistoryStore(tmp_path / "eta_history.json")

    assert reloaded.rates_for(local_ocr) == [1.25]
    assert reloaded.rates_for(network_ocr) == [9.0]
    assert reloaded.rates_for(local_without_ocr) == [0.1]
    payload = json.loads((tmp_path / "eta_history.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1


def test_u0_02v_replay_reports_error_jumps_pause_and_mode_recalibration() -> None:
    events = [
        EtaReplayEvent(
            at_seconds=0,
            event_type="progress",
            remaining_cost_by_lane={"ocr": 100},
            workers_by_lane={"ocr": 1},
        ),
        EtaReplayEvent(
            at_seconds=10,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 90},
        ),
        EtaReplayEvent(
            at_seconds=20,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 80},
        ),
        EtaReplayEvent(
            at_seconds=30,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 70},
        ),
        EtaReplayEvent(
            at_seconds=40,
            event_type="pause",
            remaining_cost_by_lane={"ocr": 60},
        ),
        EtaReplayEvent(
            at_seconds=55,
            event_type="progress",
            remaining_cost_by_lane={"ocr": 60},
        ),
        EtaReplayEvent(
            at_seconds=60,
            event_type="resume",
            remaining_cost_by_lane={"ocr": 60},
        ),
        EtaReplayEvent(
            at_seconds=70,
            event_type="mode_switch",
            mode="normal",
            remaining_cost_by_lane={"ocr": 50},
            workers_by_lane={"ocr": 1},
        ),
        EtaReplayEvent(
            at_seconds=80,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 40},
        ),
        EtaReplayEvent(
            at_seconds=90,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 30},
        ),
        EtaReplayEvent(
            at_seconds=100,
            event_type="completion",
            lane="ocr",
            completed_cost=10,
            service_seconds=10,
            remaining_cost_by_lane={"ocr": 20},
        ),
        EtaReplayEvent(
            at_seconds=120,
            event_type="finish",
            remaining_cost_by_lane={},
        ),
    ]

    report = replay_eta(events)

    assert report.first_ready_seconds == 30
    assert report.predictions
    assert all(point.absolute_error_seconds >= 0 for point in report.predictions)
    assert all(point.absolute_percentage_error >= 0 for point in report.predictions)
    assert report.median_absolute_percentage_error >= 0
    assert report.final_ten_minutes_median_absolute_percentage_error >= 0
    assert report.pause_frozen is True
    assert report.mode_switch_recalibration_seconds == 30
    assert report.worker_recycle_count == 0
    assert report.max_single_up_jump_seconds >= 0
    assert report.max_single_down_jump_seconds >= 0


def test_u0_02v_worker_recycle_is_counted_without_resetting_samples() -> None:
    events = [
        EtaReplayEvent(
            at_seconds=0,
            event_type="completion",
            lane="normal",
            completed_cost=1,
            service_seconds=1,
            remaining_cost_by_lane={"normal": 5},
            workers_by_lane={"normal": 1},
        ),
        EtaReplayEvent(
            at_seconds=1,
            event_type="worker_recycle",
            lane="normal",
            remaining_cost_by_lane={"normal": 4},
        ),
        EtaReplayEvent(
            at_seconds=2,
            event_type="completion",
            lane="normal",
            completed_cost=1,
            service_seconds=1,
            remaining_cost_by_lane={"normal": 3},
        ),
        EtaReplayEvent(
            at_seconds=3,
            event_type="completion",
            lane="normal",
            completed_cost=1,
            service_seconds=1,
            remaining_cost_by_lane={"normal": 2},
        ),
        EtaReplayEvent(
            at_seconds=5,
            event_type="finish",
            remaining_cost_by_lane={},
        ),
    ]

    report = replay_eta(events)

    assert report.worker_recycle_count == 1
    assert report.first_ready_seconds == 3


def test_u0_02v_replay_tool_runs_directly_outside_the_repository(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    source = tmp_path / "events.json"
    output = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "at_seconds": 0,
                        "event_type": "completion",
                        "lane": "normal",
                        "completed_cost": 1,
                        "service_seconds": 1,
                        "remaining_cost_by_lane": {"normal": 1},
                        "workers_by_lane": {"normal": 1},
                    },
                    {
                        "at_seconds": 1,
                        "event_type": "finish",
                        "remaining_cost_by_lane": {},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(repository / "tools" / "replay_index_eta.py"),
            str(source),
            "--output",
            str(output),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr or completed.stdout
    assert json.loads(output.read_text(encoding="utf-8"))["duration_seconds"] == 1
