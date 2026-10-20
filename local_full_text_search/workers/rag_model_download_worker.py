from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec


class RagModelDownloadWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.settings = settings
        self.token = CancelToken()

    @Slot()
    def run(self) -> None:
        manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
        specs = (
            ModelSpec(
                self.settings.rag_embedding_repo_id,
                self.settings.rag_embedding_revision,
                "embedding",
            ),
            ModelSpec(
                self.settings.rag_reranker_repo_id,
                self.settings.rag_reranker_revision,
                "reranker",
            ),
        )
        try:
            paths = {}
            for spec in specs:
                self.token.throw_if_cancelled()
                self.progress.emit(
                    {"stage": "downloading_model", "model": spec.repo_id, "kind": spec.kind}
                )
                paths[spec.kind] = str(
                    manager.download(spec, self.token, self.progress.emit)
                )
            self.finished.emit(paths)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
