from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_builder import RagIndexBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.lexical_index_builder import LexicalIndexBuilder
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.services.logging_service import log_event


logger = logging.getLogger(__name__)


class RagIndexWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, generation_base_path: Path, settings) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path).resolve()
        self.settings = settings
        self.token = CancelToken()
        self.run_id = f"rag-index-{uuid.uuid4().hex}"

    def _log(self, event: str, **fields: object) -> None:
        try:
            log_event(
                event,
                component="rag_index_worker",
                run_id=self.run_id,
                **fields,
            )
        except Exception:
            logger.debug("Unable to write RAG index diagnostic: %s", event, exc_info=True)

    @Slot()
    def run(self) -> None:
        started = time.perf_counter()
        stage_started = started
        stage = "embedding_model_resolution"
        self._log(
            "rag.index.worker_started",
            embedding_model_id=self.settings.rag_embedding_repo_id,
            embedding_revision=self.settings.rag_embedding_revision,
            embedding_batch_size=self.settings.rag_embedding_batch_size,
            write_batch_size=self.settings.rag_index_write_batch_size,
        )
        self._log("rag.index.stage_started", stage=stage)
        try:
            manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
            embedding_path = manager.resolve_local_path(
                ModelSpec(
                    self.settings.rag_embedding_repo_id,
                    self.settings.rag_embedding_revision,
                    "embedding",
                )
            )
            self._log(
                "rag.index.stage_completed",
                stage=stage,
                duration_ms=int((time.perf_counter() - stage_started) * 1000),
                model_id=self.settings.rag_embedding_repo_id,
                revision=self.settings.rag_embedding_revision,
            )
            stage = "embedding_model_loading"
            stage_started = time.perf_counter()
            self._log("rag.index.stage_started", stage=stage)
            embedding = LocalBgeEmbeddingProvider(
                embedding_path,
                repo_id=self.settings.rag_embedding_repo_id,
                revision=self.settings.rag_embedding_revision,
                batch_size=self.settings.rag_embedding_batch_size,
            )
            self._log(
                "rag.index.stage_completed",
                stage=stage,
                duration_ms=int((time.perf_counter() - stage_started) * 1000),
                model_id=embedding.model_id,
                revision=embedding.model_revision,
                batch_size=embedding.batch_size,
            )
            source_store = IndexGenerationStore(self.generation_base_path)
            rag_store = RagIndexStore(self.generation_base_path.parent / "rag")
            builder = RagIndexBuilder(source_store, rag_store, embedding, self.settings)
            stage = "index_build"
            self._log("rag.index.stage_started", stage=stage)
            summary = builder.build(self.token, self.progress.emit, run_id=self.run_id)
            if str(getattr(self.settings, "rag_lexical_backend", "legacy")) == "word_v2":
                stage = "lexical_index"
                stage_started = time.perf_counter()
                self.progress.emit({
                    "stage": stage, "message": "正在准备中文词项关键词索引",
                })
                source_bundle = rag_store.active()
                if source_bundle.bundle_id != summary.bundle_id:
                    raise RuntimeError("AI 索引在关键词索引准备期间已切换，请重试")
                lexical, reused = LexicalIndexBuilder(rag_store).prepare(
                    source_bundle, self.token,
                )
                self._log(
                    "rag.index.lexical_ready", stage=stage,
                    source_bundle_id=lexical.source_bundle_id,
                    index_fingerprint=lexical.index_fingerprint,
                    chunk_count=lexical.chunk_count, reused=reused,
                    duration_ms=int((time.perf_counter() - stage_started) * 1000),
                )
                self.progress.emit({
                    "stage": stage,
                    "message": "中文词项关键词索引已复用" if reused else "中文词项关键词索引已建立",
                })
            self.finished.emit(summary)
        except CancelledError:
            self._log(
                "rag.index.cancelled",
                stage=stage,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            self.cancelled.emit()
        except Exception as exc:
            self._log(
                "rag.index.failed",
                stage=stage,
                error_type=type(exc).__name__,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
            )
            self.failed.emit(str(exc) or type(exc).__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
