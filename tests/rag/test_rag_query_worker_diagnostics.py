from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag import diagnostics
from local_full_text_search.rag.errors import RagGenerationError
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.workers import rag_query_worker as worker_module


def test_query_failure_records_generation_stage_and_redacts_key(monkeypatch) -> None:
    records = []
    secret = "test-secret-never-log"

    def capture(event, **fields):
        records.append((event, fields))

    monkeypatch.setattr(worker_module, "log_event", capture)
    monkeypatch.setattr(diagnostics, "log_event", capture)

    class FakeManager:
        def __init__(self, cache_dir):
            pass

        def resolve_local_path(self, spec):
            return Path("unused-model")

    class ProviderError(Exception):
        status_code = 404
        body = {"error": {"type": "model_not_found"}}
        response = SimpleNamespace(
            status_code=404,
            headers={"x-request-id": "provider-request-example"},
        )

    class FakeAnswerService:
        def __init__(self, *args, progress_callback, **kwargs):
            self.progress_callback = progress_callback

        def answer(self, query, token, *, trace_id=None):
            self.progress_callback({"stage": "checking_index", "message": "检查索引"})
            self.progress_callback(
                {
                    "stage": "generating",
                    "message": "生成答案",
                    "candidate_count": 3,
                    "api_request": {"input": "private-document-text"},
                }
            )
            raise RagGenerationError("模型不受支持") from ProviderError(secret)

    monkeypatch.setattr(worker_module, "LocalModelManager", FakeManager)
    monkeypatch.setattr(worker_module, "LocalBgeEmbeddingProvider", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_module, "LocalBgeReranker", lambda *args, **kwargs: object())
    monkeypatch.setattr(worker_module, "IndexGenerationStore", lambda *args: object())
    monkeypatch.setattr(worker_module, "RagIndexStore", lambda *args: object())
    monkeypatch.setattr(worker_module, "WindowsCredentialSecretStore", lambda: SimpleNamespace(retrieve=lambda ref: secret))
    monkeypatch.setattr(worker_module, "AnswerService", FakeAnswerService)

    worker = worker_module.RagQueryWorker(
        Path("unused-generation.db"),
        RagQuery(text="如何恢复索引？"),
        AppSettings(rag_api_model="gpt-5.6-sol", rag_api_base_url="https://example.test"),
        request_id="rag-query-example",
    )
    failures = []
    worker.failed.connect(failures.append)
    worker.run()

    failure_events = [fields for event, fields in records if event == "rag.query.failed"]
    assert failures == ["模型不受支持"]
    assert len(failure_events) == 1
    failure = failure_events[0]
    assert failure["request_id"] == "rag-query-example"
    assert failure["stage"] == "generating"
    assert failure["http_status"] == 404
    assert failure["provider_error_type"] == "model_not_found"
    assert failure["provider_request_id"] == "provider-request-example"
    assert any(
        event == "rag.query.stage_started"
        and fields["stage"] == "generating"
        and fields["candidate_count"] == 3
        for event, fields in records
    )
    assert secret not in str(records)
    assert "private-document-text" not in str(records)
