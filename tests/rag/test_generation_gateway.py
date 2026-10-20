from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.errors import RagConfigurationError, RagGenerationError
from local_full_text_search.rag.generation.base import AnswerReviewRequest, GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    OpenAICompatibleGateway,
    REVIEW_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    provider_request_payload,
    provider_review_payload,
)


def _request(api_style: str = "chat_completions") -> GenerationRequest:
    return GenerationRequest(
        question="检查点有什么用？",
        context="[S1]\n文件：恢复手册.txt\n内容：检查点可以恢复任务。",
        model="test-model",
        api_base_url="https://api.example.test/v1",
        api_style=api_style,
        api_key="secret-that-must-not-leak",
    )


@pytest.mark.parametrize("api_style", ["chat_completions", "responses"])
def test_openai_compatible_gateway_supports_both_api_styles(api_style) -> None:
    calls = []

    class _FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **params: calls.append(("chat", params))
                    or SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content="依据为检查点 [S1]。"))]
                    )
                )
            )
            self.responses = SimpleNamespace(
                create=lambda **params: calls.append(("responses", params))
                or SimpleNamespace(output_text="依据为检查点 [S1]。")
            )

    answer = OpenAICompatibleGateway(_FakeClient).generate(_request(api_style), CancelToken())

    assert answer.text == "依据为检查点 [S1]。"
    assert calls[0][1]["api_key"] == "secret-that-must-not-leak"
    request_call = calls[1][1]
    assert request_call == provider_request_payload(_request(api_style))
    assert "secret-that-must-not-leak" not in str(request_call)
    assert "检查点可以恢复任务" in str(request_call)
    if api_style == "responses":
        assert request_call["instructions"] == SYSTEM_PROMPT
        assert request_call["store"] is False
    else:
        assert request_call["messages"][0]["content"] == SYSTEM_PROMPT


def test_gateway_masks_api_key_from_provider_errors() -> None:
    class _Unauthorized(Exception):
        status_code = 401

    class _FakeClient:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **params: (_ for _ in ()).throw(
                    _Unauthorized("bad key secret-that-must-not-leak")
                ))
            )

    with pytest.raises(RagGenerationError, match="API Key 无效") as raised:
        OpenAICompatibleGateway(_FakeClient).generate(_request(), CancelToken())
    assert "secret-that-must-not-leak" not in str(raised.value)


def test_gateway_rejects_invalid_url_before_network_call() -> None:
    request = replace(_request(), api_base_url="file:///etc/passwd")
    with pytest.raises(RagConfigurationError, match="HTTP/HTTPS"):
        OpenAICompatibleGateway(lambda **kwargs: pytest.fail("must not create client")).generate(
            request, CancelToken()
        )


def test_gateway_identifies_model_not_found_instead_of_generic_endpoint_404() -> None:
    class _MissingModel(Exception):
        status_code = 404
        body = {"error": {"type": "model_not_found"}}

    class _FakeClient:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(
                create=lambda **params: (_ for _ in ()).throw(_MissingModel("unknown model"))
            )

    request = replace(_request("responses"), model="5.6 Sol")
    with pytest.raises(RagGenerationError, match="模型“5.6 Sol”不受当前服务商账号支持"):
        OpenAICompatibleGateway(_FakeClient).generate(request, CancelToken())


@pytest.mark.parametrize("api_style", ["chat_completions", "responses"])
def test_gateway_supports_answer_review_for_both_api_styles(api_style) -> None:
    review_json = (
        '{"verdict":"pass","summary":"引用有效","claims":['
        '{"claim":"检查底物","citations":["S1"],"status":"supported",'
        '"reason":"证据写明检查底物"}],"final_answer":"检查底物 [S1]。"}'
    )
    calls = []

    class _FakeClient:
        def __init__(self, **kwargs):
            calls.append(("client", kwargs))
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **params: calls.append(("chat", params))
                    or SimpleNamespace(
                        choices=[SimpleNamespace(message=SimpleNamespace(content=review_json))]
                    )
                )
            )
            self.responses = SimpleNamespace(
                create=lambda **params: calls.append(("responses", params))
                or SimpleNamespace(output_text=review_json)
            )

    request = AnswerReviewRequest(
        question="底物异常如何处理？",
        query_plan_summary="intent=troubleshooting; object=底物",
        context="[S1]\n内容：检查底物。",
        draft_answer="检查底物 [S1]。",
        citation_check_summary="引用编号有效",
        allowed_source_ids=("S1",),
        model="test-model",
        api_base_url="https://api.example.test/v1",
        api_style=api_style,
        api_key="secret-that-must-not-leak",
    )

    result = OpenAICompatibleGateway(_FakeClient).review_answer(request, CancelToken())

    assert result.verdict == "pass"
    assert result.final_answer == "检查底物 [S1]。"
    request_call = calls[1][1]
    assert request_call == provider_review_payload(request)
    assert "secret-that-must-not-leak" not in str(request_call)
    assert "底物" in str(request_call)
    if api_style == "responses":
        assert request_call["instructions"] == REVIEW_SYSTEM_PROMPT
        assert request_call["store"] is False
    else:
        assert request_call["messages"][0]["content"] == REVIEW_SYSTEM_PROMPT
