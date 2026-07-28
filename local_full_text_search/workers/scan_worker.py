from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.task_manager import CancelToken

logger = logging.getLogger(__name__)


class ScanWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path, settings: AppSettings) -> None:
        super().__init__()
        self.db_path = db_path
        self.settings = settings
        self.token = CancelToken()
        self.manager: IndexManager | None = None

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            db.initialize()
            self.manager = IndexManager(db, self.settings)
            summary = self.manager.index_enabled_roots(self.token, self.progress.emit)
            self.finished.emit(summary)
        except Exception as exc:
            logger.exception("Index scan failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            self.manager = None

    @Slot()
    def cancel(self, *, force: bool = False) -> None:
        self.token.cancel(force=force)
        if force and self.manager is not None:
            self.manager.force_terminate_processes()

    @Slot()
    def pause(self) -> None:
        self.token.pause()

    @Slot()
    def resume(self) -> None:
        self.token.resume()
