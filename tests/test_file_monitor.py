from __future__ import annotations

import threading
import time
from pathlib import Path

from local_full_text_search.core.file_monitor import FileMonitor
from local_full_text_search.core.task_manager import CancelToken


def test_monitor_coalesces_bursts_into_one_callback() -> None:
    received: list[Path] = []
    callback_fired = threading.Event()
    monitor = FileMonitor(
        lambda path: (received.append(path), callback_fired.set()),
        debounce_seconds=0.03,
    )

    for index in range(25):
        monitor._schedule(Path(f"changed-{index}.txt"))

    assert callback_fired.wait(0.5)
    assert received == [Path("changed-24.txt")]
    monitor.stop()


def test_monitor_stop_cancels_pending_callback() -> None:
    received: list[Path] = []
    monitor = FileMonitor(received.append, debounce_seconds=0.05)

    monitor._schedule(Path("pending.txt"))
    monitor.stop()
    time.sleep(0.1)

    assert received == []


def test_force_cancel_is_visible_to_workers() -> None:
    token = CancelToken()

    token.cancel(force=True)

    assert token.cancelled
    assert token.force_cancelled
