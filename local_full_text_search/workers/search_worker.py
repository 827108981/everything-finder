from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.search_query import SearchQuery

logger = logging.getLogger(__name__)


class SearchWorker(QObject):
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, db_path: Path, query: SearchQuery) -> None:
        super().__init__()
        self.db_path = db_path
        self.query = query
        self.token = CancelToken()

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            engine = SearchEngine(db)
            self.finished.emit(engine.search(self.query, self.token))
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Search failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
