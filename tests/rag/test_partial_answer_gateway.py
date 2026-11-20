from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import provider_request_payload
from tools.partial_answer_gateway import partial_provider_payload


def request(style):
    return GenerationRequest("为何偏低？如果另一种情况呢？", "[S1]\n内容：第一种情况的依据",
                             "model", "https://unused.invalid/v1", style, "secret",
                             query_plan_summary="conditions")


def test_only_system_instruction_changes_for_chat_completions():
    original = provider_request_payload(request("chat_completions"))
    candidate = partial_provider_payload(request("chat_completions"))
    assert candidate["messages"][1:] == original["messages"][1:]
    assert candidate["model"] == original["model"]
    assert candidate["max_tokens"] == original["max_tokens"]
    assert "某一项有直接、适用的原文依据" in candidate["messages"][0]["content"]


def test_only_system_instruction_changes_for_responses():
    original = provider_request_payload(request("responses"))
    candidate = partial_provider_payload(request("responses"))
    assert candidate["input"] == original["input"]
    assert candidate["model"] == original["model"]
    assert candidate["max_output_tokens"] == original["max_output_tokens"]
    assert "如果全部所问都没有直接依据" in candidate["instructions"]
