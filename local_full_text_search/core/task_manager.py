from __future__ import annotations

import threading
import time

from .errors import CancelledError


class CancelToken:
    """Thread-safe pause/cancel token checked by scanners, parsers and searches."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._paused = threading.Event()

    def cancel(self) -> None:
        self._cancelled.set()

    def pause(self) -> None:
        self._paused.set()

    def resume(self) -> None:
        self._paused.clear()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def paused(self) -> bool:
        return self._paused.is_set()

    def throw_if_cancelled(self) -> None:
        if self.cancelled:
            raise CancelledError("任务已取消")

    def wait_if_paused(self) -> None:
        while self.paused:
            self.throw_if_cancelled()
            time.sleep(0.1)
