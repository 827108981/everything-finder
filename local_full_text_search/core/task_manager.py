from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

from .errors import CancelledError, PauseRequestedError


PAUSE_MARKER_NAME = ".pause_requested"
PLANNING_PAUSE_ACK_DIR_NAME = ".planning_pause_acknowledgements"


class CancelToken:
    """Thread-safe pause/cancel token checked by scanners, parsers and searches."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._force_cancelled = threading.Event()
        self._paused = threading.Event()
        self._state_changed = threading.Condition()

    def cancel(self, *, force: bool = False) -> None:
        if force:
            self._force_cancelled.set()
        self._cancelled.set()
        with self._state_changed:
            self._state_changed.notify_all()

    def pause(self) -> None:
        self._paused.set()
        with self._state_changed:
            self._state_changed.notify_all()

    def resume(self) -> None:
        self._paused.clear()
        with self._state_changed:
            self._state_changed.notify_all()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def force_cancelled(self) -> bool:
        return self._force_cancelled.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("任务已取消")

    def wait_if_paused(self) -> None:
        with self._state_changed:
            while self.paused:
                self.throw_if_cancelled()
                self._state_changed.wait(timeout=0.1)

    def wait_until_resumed(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._state_changed:
            while self.paused:
                self.throw_if_cancelled()
                if deadline is None:
                    self._state_changed.wait(timeout=0.1)
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._state_changed.wait(timeout=min(0.1, remaining))
        return True


class ProcessRunControlToken:
    """File-backed pause token that works in spawned Windows parser workers.

    A process returns from its current parser at the next cooperative safe
    point instead of blocking inside the worker. The parent can then declare a
    true idle pause and rebuild pools before resuming in a different profile.
    """

    def __init__(
        self,
        control_dir: Path,
        *,
        pause_behavior: str = "return",
    ) -> None:
        self.control_dir = Path(control_dir)
        self._cancelled = False
        self.pause_behavior = str(pause_behavior)
        self._pause_checkpoint: dict[str, object] = {
            "task_id": f"planning:{os.getpid()}",
            "safe_unit_type": "planning_safe_point",
            "cursor": "",
            "checkpoint_version": 0,
            "checkpoint_checksum": "",
        }

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    @property
    def force_cancelled(self) -> bool:
        return False

    @property
    def paused(self) -> bool:
        return (self.control_dir / PAUSE_MARKER_NAME).is_file()

    def throw_if_cancelled(self) -> None:
        if self._cancelled:
            raise CancelledError("任务已取消")
        if self.paused:
            raise PauseRequestedError("任务已在安全检查点暂停")

    def wait_if_paused(self) -> None:
        if not self.paused:
            return
        if self.pause_behavior != "block":
            raise PauseRequestedError("任务已在安全检查点暂停")
        acknowledgement_dir = (
            self.control_dir / PLANNING_PAUSE_ACK_DIR_NAME
        )
        acknowledgement_dir.mkdir(parents=True, exist_ok=True)
        acknowledgement = acknowledgement_dir / f"{os.getpid()}.json"
        temporary = acknowledgement.with_suffix(
            f".tmp.{uuid.uuid4().hex}"
        )
        payload = {
            **self._pause_checkpoint,
            "worker_pid": os.getpid(),
            "returned_at_epoch": time.time(),
            "holds_external_process": False,
        }
        try:
            temporary.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                encoding="utf-8",
            )
            os.replace(temporary, acknowledgement)
            while self.paused:
                if self._cancelled:
                    raise CancelledError("任务已取消")
                time.sleep(0.05)
        finally:
            temporary.unlink(missing_ok=True)
            acknowledgement.unlink(missing_ok=True)

    def set_pause_checkpoint(
        self,
        *,
        task_id: str,
        safe_unit_type: str,
        cursor: str,
        checkpoint_version: int,
        checkpoint_checksum: str = "",
    ) -> None:
        self._pause_checkpoint = {
            "task_id": str(task_id),
            "safe_unit_type": str(safe_unit_type),
            "cursor": str(cursor),
            "checkpoint_version": max(
                0,
                int(checkpoint_version),
            ),
            "checkpoint_checksum": str(checkpoint_checksum),
        }

    def cancel(self, *, force: bool = False) -> None:
        _ = force
        self._cancelled = True

    def pause(self) -> None:
        raise RuntimeError("ProcessRunControlToken is controlled by its marker file")

    def resume(self) -> None:
        raise RuntimeError("ProcessRunControlToken is controlled by its marker file")
