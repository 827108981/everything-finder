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
        self._timer: threading.Timer | None = None
        self._pending_path: Path | None = None
        self._lock = threading.Lock()

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
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=0.75)
            self._observer = None
        with self._lock:
            timer = self._timer
            self._timer = None
            self._pending_path = None
        if timer is not None:
            timer.cancel()

    def _schedule(self, path: Path) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._pending_path = path
            timer = threading.Timer(self.debounce_seconds, self._fire)
            timer.daemon = True
            self._timer = timer
        timer.start()

    def _fire(self) -> None:
        with self._lock:
            path = self._pending_path
            self._timer = None
            self._pending_path = None
        if path is None:
            return
        try:
            self.callback(path)
        except Exception:
            logger.exception("File monitor callback failed for %s", path)
