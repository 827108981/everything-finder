from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class FileMonitor:
    """Optional watchdog wrapper with a simple debounce callback."""

    def __init__(self, callback: Callable[[Path], None], debounce_seconds: float = 2.0) -> None:
        self.callback = callback
        self.debounce_seconds = debounce_seconds
        self._observer = None
        self._timers: dict[str, threading.Timer] = {}

    def start(self, roots: list[Path]) -> None:
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer
        except ImportError:
            logger.warning("watchdog 未安装，文件监控未启用")
            return

        monitor = self

        class Handler(FileSystemEventHandler):
            def on_any_event(self, event: object) -> None:
                src_path = getattr(event, "src_path", None)
                is_directory = bool(getattr(event, "is_directory", False))
                if src_path and not is_directory:
                    monitor._schedule(Path(src_path))

        self._observer = Observer()
        handler = Handler()
        for root in roots:
            self._observer.schedule(handler, str(root), recursive=True)
        self._observer.start()

    def stop(self) -> None:
        for timer in self._timers.values():
            timer.cancel()
        self._timers.clear()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    def _schedule(self, path: Path) -> None:
        key = str(path)
        existing = self._timers.get(key)
        if existing:
            existing.cancel()
        timer = threading.Timer(self.debounce_seconds, self.callback, args=(path,))
        self._timers[key] = timer
        timer.start()
