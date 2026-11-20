from __future__ import annotations

import logging
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.diagnostics import (
    log_rag_exception,
    new_rag_request_id,
    provider_host,
)
from local_full_text_search.rag.errors import RagError
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.services.logging_service import log_event


logger = logging.getLogger(__name__)


class RagQueryWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        generation_base_path: Path,
        query: RagQuery,
        settings,
        *,
        request_id: str | None = None,
    ) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path).resolve()
        self.query = query
        self.settings = settings
        self.token = CancelToken()
        self.request_id = request_id or new_rag_request_id("query")
        self.current_stage = "queued"
        self._stage_started_at = 0.0
        self.error_details = ""

    def _log(self, event: str, **fields: object) -> None:
        # 诊断日志不能反过来让用户的问答失败。
        try:
            log_event(
                event,
                component="rag_query_worker",
                request_id=self.request_id,
                **fields,
            )
        except Exception:
            logger.debug("Unable to write RAG query diagnostic: %s", event, exc_info=True)

    def _set_stage(self, stage: str, **metrics: object) -> None:
        now = time.monotonic()
        if self._stage_started_at:
            self._log(
                "rag.query.stage_completed",
                stage=self.current_stage,
                duration_ms=int((now - self._stage_started_at) * 1_000),
            )
        self.current_stage = stage
        self._stage_started_at = now
        self._log("rag.query.stage_started", stage=stage, **metrics)

    def _on_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            stage = str(payload.get("stage") or "")
            if stage:
                allowed_metrics = (
                    "source_generation",
                    "active_source_generation",
                    "rag_bundle_id",
                    "candidate_count",
                    "input_count",
                    "output_count",
                    "duplicate_count",
                    "conflict_count",
                    "required_constraint_count",
                    "lexical_count",
                    "vector_count",
                    "reranked_count",
                    "accepted_count",
                    "source_count",
                    "context_chars",
                    "answer_chars",
                    "citation_count",
                    "claim_count",
                    "supported_count",
                    "review_verdict",
                    "review_enabled",
                    "model",
                    "api_style",
                    "timeout_seconds",
                    "reason",
                )
                self._set_stage(
                    stage,
                    **{key: payload[key] for key in allowed_metrics if key in payload},
                )
        self.progress.emit(payload)

    @Slot()
    def run(self) -> None:
        started_at = time.monotonic()
        secret_store = None
        common_fields = {
            "provider_host": provider_host(self.settings.rag_api_base_url),
            "model": self.settings.rag_api_model,
            "api_style": self.settings.rag_api_style,
        }
        self._log(
            "rag.query.started",
            stage="model_resolution",
            question_chars=len(self.query.text),
            **common_fields,
        )
        try:
            self._set_stage("model_resolution")
            manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
            embedding_path = manager.resolve_local_path(
                ModelSpec(
                    self.settings.rag_embedding_repo_id,
                    self.settings.rag_embedding_revision,
                    "embedding",
                )
            )
            reranker_path = manager.resolve_local_path(
                ModelSpec(
                    self.settings.rag_reranker_repo_id,
                    self.settings.rag_reranker_revision,
                    "reranker",
                )
            )
            self._set_stage("embedding_model_loading")
            embedding = LocalBgeEmbeddingProvider(
                embedding_path,
                repo_id=self.settings.rag_embedding_repo_id,
                revision=self.settings.rag_embedding_revision,
                batch_size=self.settings.rag_embedding_batch_size,
            )
            self._set_stage("reranker_model_loading")
            reranker = LocalBgeReranker(
                reranker_path,
                repo_id=self.settings.rag_reranker_repo_id,
                revision=self.settings.rag_reranker_revision,
                batch_size=self.settings.rag_reranker_batch_size,
                max_length=self.settings.rag_reranker_max_length,
            )
            secret_store = WindowsCredentialSecretStore()
            service = AnswerService(
                IndexGenerationStore(self.generation_base_path),
                RagIndexStore(self.generation_base_path.parent / "rag"),
                embedding,
                reranker,
                OpenAICompatibleGateway(),
                secret_store,
                self.settings,
                progress_callback=self._on_progress,
            )
            answer = service.answer(self.query, self.token, trace_id=self.request_id)
            self._log(
                "rag.query.stage_completed",
                stage=self.current_stage,
                duration_ms=int((time.monotonic() - self._stage_started_at) * 1_000),
            )
            self._log(
                "rag.query.succeeded",
                stage="completed",
                duration_ms=int((time.monotonic() - started_at) * 1_000),
                insufficient_evidence=answer.insufficient_evidence,
                source_count=len(answer.sources),
                **common_fields,
            )
            self.finished.emit(answer)
        except CancelledError:
            self._log(
                "rag.query.cancelled",
                stage=self.current_stage,
                duration_ms=int((time.monotonic() - started_at) * 1_000),
                **common_fields,
            )
            self.cancelled.emit()
        except Exception as exc:
            secret = ""
            if secret_store is not None:
                try:
                    secret = secret_store.retrieve(self.settings.rag_api_secret_ref) or ""
                except Exception:
                    pass
            if secret:
                try:
                    message, self.error_details = log_rag_exception(
                        "rag.query.failed",
                        exc,
                        request_id=self.request_id,
                        component="rag_query_worker",
                        secrets=[secret],
                        stage=self.current_stage,
                        stage_duration_ms=int(
                            (time.monotonic() - self._stage_started_at) * 1_000
                        ),
                        duration_ms=int((time.monotonic() - started_at) * 1_000),
                        **common_fields,
                    )
                except Exception:
                    logger.debug("Unable to write RAG query failure", exc_info=True)
                    message = str(exc) if isinstance(exc, RagError) else type(exc).__name__
            else:
                # 尚未取得 Key 时只记录受控字段，避免把未知异常正文写进日志。
                self._log(
                    "rag.query.failed",
                    stage=self.current_stage,
                    duration_ms=int((time.monotonic() - started_at) * 1_000),
                    error_type=type(exc).__name__,
                    credential_available=False,
                    **common_fields,
                )
                message = str(exc) if isinstance(exc, RagError) else type(exc).__name__
            self.failed.emit(message or type(exc).__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
