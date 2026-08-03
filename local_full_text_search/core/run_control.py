from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from local_full_text_search.core.task_manager import PAUSE_MARKER_NAME

PLANNING_PAUSE_ACK_DIR_NAME = ".planning_pause_acknowledgements"


def pause_marker_path(control_dir: Path) -> Path:
    return Path(control_dir) / PAUSE_MARKER_NAME


def request_process_pause(control_dir: Path) -> Path:
    directory = Path(control_dir)
    directory.mkdir(parents=True, exist_ok=True)
    marker = pause_marker_path(directory)
    temporary = marker.with_suffix(f".tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_text(
        json.dumps(
            {
                "requested_at": time.time(),
                "request_pid": os.getpid(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    temporary.replace(marker)
    return marker


def resume_processes(control_dir: Path) -> None:
    pause_marker_path(control_dir).unlink(missing_ok=True)


def planning_pause_acknowledgements(
    control_dir: Path,
) -> dict[int, dict[str, object]]:
    directory = (
        Path(control_dir) / PLANNING_PAUSE_ACK_DIR_NAME
    )
    acknowledgements: dict[int, dict[str, object]] = {}
    if not directory.is_dir():
        return acknowledgements
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            pid = int(payload.get("worker_pid") or 0)
            if pid > 0:
                acknowledgements[pid] = dict(payload)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return acknowledgements
