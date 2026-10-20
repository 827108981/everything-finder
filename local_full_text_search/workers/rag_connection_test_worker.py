from __future__ import annotations

import time

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.diagnostics import (
    log_rag_exception,
    new_rag_request_id,
    provider_host,
)
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.services.logging_service import log_event


class RagConnectionTestWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        settings,
        api_key: str,
        *,
        request_id: str | None = None,
        api_key_is_saved: bool = False,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.api_key = api_key
        self.token = CancelToken()
        self.request_id = request_id or new_rag_request_id("connection-test")
        self.api_key_is_saved = bool(api_key_is_saved)
        self.error_details = ""

    @Slot()
    def run(self) -> None:
        started_at = time.monotonic()
        common_fields = {
            "stage": "validate_and_create_client",
            "provider_host": provider_host(self.settings.rag_api_base_url),
            "model": str(self.settings.rag_api_model),
            "api_style": str(self.settings.rag_api_style),
            "timeout_seconds": int(self.settings.rag_api_timeout_seconds),
            "api_key_source": "saved_credential" if self.api_key_is_saved else "current_input",
        }
        log_event(
            "rag.connection_test.started",
            component="rag_connection_test_worker",
            request_id=self.request_id,
            **common_fields,
        )
        try:
            log_event(
                "rag.connection_test.request_started",
                component="rag_connection_test_worker",
                request_id=self.request_id,
                stage="llm_generation_request",
                provider_host=common_fields["provider_host"],
                model=common_fields["model"],
                api_style=common_fields["api_style"],
            )
            answer = OpenAICompatibleGateway().generate(
                GenerationRequest(
                    question="连接测试。请回复：连接成功。",
                    context="",
                    model=self.settings.rag_api_model,
                    api_base_url=self.settings.rag_api_base_url,
                    api_style=self.settings.rag_api_style,
                    api_key=self.api_key,
                    timeout_seconds=self.settings.rag_api_timeout_seconds,
                    max_output_tokens=64,
                ),
                self.token,
            )
            log_event(
                "rag.connection_test.succeeded",
                component="rag_connection_test_worker",
                request_id=self.request_id,
                stage="response_received",
                duration_ms=int((time.monotonic() - started_at) * 1_000),
                provider_host=common_fields["provider_host"],
                model=common_fields["model"],
                api_style=common_fields["api_style"],
            )
            self.finished.emit(answer)
        except CancelledError:
            log_event(
                "rag.connection_test.cancelled",
                component="rag_connection_test_worker",
                request_id=self.request_id,
                stage="cancelled",
                duration_ms=int((time.monotonic() - started_at) * 1_000),
                provider_host=common_fields["provider_host"],
                model=common_fields["model"],
                api_style=common_fields["api_style"],
            )
            self.cancelled.emit()
        except Exception as exc:
            message, self.error_details = log_rag_exception(
                "rag.connection_test.failed",
                exc,
                request_id=self.request_id,
                component="rag_connection_test_worker",
                secrets=[self.api_key],
                stage="llm_generation_request",
                duration_ms=int((time.monotonic() - started_at) * 1_000),
                provider_host=common_fields["provider_host"],
                model=common_fields["model"],
                api_style=common_fields["api_style"],
                timeout_seconds=common_fields["timeout_seconds"],
                api_key_source=common_fields["api_key_source"],
            )
            self.failed.emit(message or type(exc).__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
