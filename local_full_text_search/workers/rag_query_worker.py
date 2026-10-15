from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.model_manager import CredentialStore
from local_full_text_search.rag.models import RagQuery

logger = logging.getLogger(__name__)


class RagQueryWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()
    ended = Signal()

    def __init__(
        self,
        generation_base_path: Path,
        settings: AppSettings,
        query: RagQuery,
        model_manager: RagModelManager,
    ) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path)
        self.settings = settings
        self.query = query
        self.model_manager = model_manager
        self.token = CancelToken()
        self.gateway = None

    @Slot()
    def run(self) -> None:
        try:
            from local_full_text_search.rag.answer_service import AnswerService
            from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway

            api_key = CredentialStore().get_api_key()
            gateway = OpenAICompatibleGateway(
                self.settings.rag_llm_base_url,
                api_key,
                self.settings.rag_llm_model,
            )
            self.gateway = gateway
            self.progress.emit({"stage": "loading_model", "message": "正在加载或首次下载本地 Embedding 模型"})
            embedding = self.model_manager.embedding_provider(self.settings)
            context_budget = min(
                int(self.settings.rag_context_max_tokens),
                max(
                    256,
                    int(self.settings.rag_llm_context_tokens)
                    - int(self.settings.rag_answer_max_tokens)
                    - 512,
                ),
            )
            service = AnswerService(
                self.generation_base_path,
                context_max_tokens=context_budget,
                answer_max_tokens=self.settings.rag_answer_max_tokens,
            )
            answer = service.answer(
                self.query,
                embedding=embedding,
                llm=gateway,
                token=self.token,
                progress=self.progress.emit,
            )
            self.finished.emit(answer)
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("RAG question failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            if "gateway" in locals():
                gateway.close()
            self.gateway = None
            self.ended.emit()

    @Slot()
    def cancel(self) -> None:
        self.token.cancel(reason="RAG_QUERY_CANCELLED")
        gateway = self.gateway
        if gateway is not None:
            gateway.cancel()
