from __future__ import annotations

import logging
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.model_manager import RagModelManager
from local_full_text_search.rag.models import RagBuildCancelled

logger = logging.getLogger(__name__)


class RagIndexWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    ended = Signal()

    def __init__(
        self,
        generation_base_path: Path,
        settings: AppSettings,
        model_manager: RagModelManager,
    ) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path)
        self.settings = settings
        self.model_manager = model_manager
        self.token = CancelToken()

    @Slot()
    def run(self) -> None:
        try:
            from local_full_text_search.rag.index_builder import RagIndexBuilder

            self.progress.emit({"stage": "loading_model", "message": "正在加载本地 Embedding 模型"})
            embedding = self.model_manager.embedding_provider(self.settings)
            builder = RagIndexBuilder(self.generation_base_path)
            report = builder.build(embedding, self.token, self.progress.emit)
            self.finished.emit(report)
        except (CancelledError, RagBuildCancelled):
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("RAG sidecar build failed")
            logger.error("RAG build traceback: %s", traceback.format_exc())
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            self.ended.emit()

    @Slot()
    def cancel(self) -> None:
        self.token.cancel(reason="RAG_INDEX_CANCELLED")
