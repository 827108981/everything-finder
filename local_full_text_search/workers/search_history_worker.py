from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager

logger = logging.getLogger(__name__)


class SearchHistoryWorker(QObject):
    """Persist search-history changes away from the Qt GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        db_path: Path,
        *,
        query_text: str = "",
        clear: bool = False,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.query_text = query_text
        self.clear = clear

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            if self.clear:
                db.clear_search_history()
                histories: list[str] = []
            else:
                db.add_search_history(self.query_text)
                histories = db.search_history()
            self.finished.emit(histories)
        except Exception as exc:
            logger.warning("Search history update deferred: %s", exc)
            self.failed.emit(str(exc) or exc.__class__.__name__)
