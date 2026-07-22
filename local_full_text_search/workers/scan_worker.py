from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.task_manager import CancelToken


class ScanWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path, settings: AppSettings) -> None:
        super().__init__()
        self.db_path = db_path
        self.settings = settings
        self.token = CancelToken()

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            db.initialize()
            manager = IndexManager(db, self.settings)
            summary = manager.index_enabled_roots(self.token, self.progress.emit)
            self.finished.emit(summary)
        except Exception:
            self.failed.emit(traceback.format_exc())

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()

    @Slot()
    def pause(self) -> None:
        self.token.pause()

    @Slot()
    def resume(self) -> None:
        self.token.resume()
