from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.runner import RagEvalConfig, RagEvalRunner
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


class RagEvalWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        generation_base_path: Path,
        settings,
        config: RagEvalConfig,
        *,
        preflight_only: bool = False,
    ) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path).resolve()
        self.settings = settings
        self.config = config
        self.preflight_only = bool(preflight_only)
        self.token = CancelToken()
        self.current_stage = "queued"
        self.error_details = ""
        self.partial_result: dict[str, Any] | None = None
        self._secret_values: list[str] = []

    @Slot()
    def run(self) -> None:
        started = time.monotonic()
        runner = RagEvalRunner(
            self.generation_base_path,
            self.settings,
            self.token,
            progress_callback=self._emit_progress,
        )
        try:
            self.current_stage = "dataset_and_index_preflight"
            self._emit_progress({"type": "stage", "stage": self.current_stage, "message": "正在校验测试集和已发布 AI 索引"})
            prepared = runner.prepare(self.config)
            self._emit_progress({
                "type": "preflight_complete",
                "bundle_id": prepared.bundle_id,
                "dataset_path": str(prepared.dataset.path),
                "dataset_sha256": prepared.dataset.sha256,
                "coverage": prepared.coverage,
                "case_ids": [case.case_id for case in prepared.cases],
            })
            if self.preflight_only:
                self.finished.emit({"type": "preflight_complete", "bundle_id": prepared.bundle_id, "coverage": prepared.coverage})
                return

            self.token.throw_if_cancelled()
            self.current_stage = "loading_local_models"
            self._emit_progress({"type": "stage", "stage": self.current_stage, "message": "正在从本地缓存加载 Embedding 与 Reranker 模型"})
            manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
            embedding_path = manager.resolve_local_path(ModelSpec(
                self.settings.rag_embedding_repo_id,
                self.settings.rag_embedding_revision,
                "embedding",
            ))
            reranker_path = manager.resolve_local_path(ModelSpec(
                self.settings.rag_reranker_repo_id,
                self.settings.rag_reranker_revision,
                "reranker",
            ))
            embedding = LocalBgeEmbeddingProvider(
                embedding_path,
                repo_id=self.settings.rag_embedding_repo_id,
                revision=self.settings.rag_embedding_revision,
                batch_size=self.settings.rag_embedding_batch_size,
            )
            reranker = LocalBgeReranker(
                reranker_path,
                repo_id=self.settings.rag_reranker_repo_id,
                revision=self.settings.rag_reranker_revision,
                batch_size=self.settings.rag_reranker_batch_size,
                max_length=self.settings.rag_reranker_max_length,
            )
            secret_store = WindowsCredentialSecretStore()
            gateway = OpenAICompatibleGateway()

            if self.config.mode == "full":
                self.current_stage = "checking_credentials"
                if not str(getattr(self.settings, "rag_api_model", "")).strip():
                    raise ValueError("设置页尚未填写 LLM 模型名")
                api_key = secret_store.retrieve(self.settings.rag_api_secret_ref)
                if not api_key:
                    raise ValueError("设置页没有已保存的 API Key；请先保存 Key 后再运行完整问答评测")
                self._secret_values.append(str(api_key))
                runner.redaction_values = tuple(self._secret_values)

            def service_factory(callback):
                return AnswerService(
                    IndexGenerationStore(self.generation_base_path),
                    RagIndexStore(self.generation_base_path.parent / "rag"),
                    embedding,
                    reranker,
                    gateway,
                    secret_store,
                    self.settings,
                    progress_callback=callback,
                )

            self.current_stage = "evaluating_cases"
            self.partial_result = runner.run(self.config, prepared, service_factory)
            self.finished.emit(self.partial_result)
        except CancelledError:
            self.partial_result = runner.last_report
            self._emit_progress({
                "type": "cancelled",
                "stage": self.current_stage,
                "duration_ms": int((time.monotonic() - started) * 1000),
                "partial_result": self.partial_result,
            })
            self.cancelled.emit()
        except Exception as exc:
            self.current_stage = self.current_stage or "unknown"
            safe_message = self._redact(str(exc))
            self.error_details = f"阶段：{self.current_stage}\n错误类型：{type(exc).__name__}\n{safe_message}"
            self.failed.emit(safe_message or type(exc).__name__)

    def _emit_progress(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage") or payload.get("type") or "")
        if stage:
            self.current_stage = stage
        self.progress.emit(payload)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()

    def _redact(self, value: str) -> str:
        for secret in self._secret_values:
            if secret:
                value = value.replace(secret, "[已隐藏]")
        return value
