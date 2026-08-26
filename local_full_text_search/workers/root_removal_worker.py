from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager

logger = logging.getLogger(__name__)


class RootRemovalWorker(QObject):
    """Remove one search scope outside the Qt GUI thread."""

    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path, root_id: int) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.root_id = int(root_id)

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit({"stage": "removing_scope", "phase_label": "正在删除搜索范围索引"})
            database = DatabaseManager(self.db_path)
            database.remove_root(self.root_id)
            self.progress.emit({"stage": "refreshing", "phase_label": "正在刷新搜索索引"})
            self.finished.emit({"root_id": self.root_id})
        except Exception as exc:
            logger.exception("Search scope removal failed: root_id=%s", self.root_id)
            self.failed.emit(str(exc) or exc.__class__.__name__)
