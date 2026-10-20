from __future__ import annotations

from types import SimpleNamespace

from local_full_text_search.rag import diagnostics


def test_rag_exception_log_redacts_key_and_records_provider_request_id(monkeypatch) -> None:
    records = []
    monkeypatch.setattr(diagnostics, "log_event", lambda *args, **kwargs: records.append((args, kwargs)))

    class ProviderError(Exception):
        status_code = 401
        body = {"error": {"type": "model_not_found", "code": "model_not_found"}}
        response = SimpleNamespace(
            status_code=401,
            headers={"x-request-id": "provider-request-123"},
        )

    secret = "test-secret-12345"
    message, stacktrace = diagnostics.log_rag_exception(
        "rag.connection_test.failed",
        ProviderError(f"unauthorized {secret}"),
        request_id="rag-connection-test-example",
        component="test",
        secrets=[secret],
        stage="llm_request",
    )

    assert secret not in message
    assert secret not in stacktrace
    assert records[0][1]["request_id"] == "rag-connection-test-example"
    assert records[0][1]["provider_request_id"] == "provider-request-123"
    assert records[0][1]["provider_error_type"] == "model_not_found"
    assert records[0][1]["provider_error_code"] == "model_not_found"
    assert secret not in records[0][1]["error_message"]
    assert secret not in records[0][1]["stacktrace"]
