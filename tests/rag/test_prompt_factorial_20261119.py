from dataclasses import replace
from types import SimpleNamespace

import pytest

from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    SYSTEM_PROMPT, generation_system_prompt, provider_request_payload,
)
from local_full_text_search.rag.evaluation.answer_judge import _identifier_binding_check


@pytest.mark.parametrize("style", ["responses", "chat_completions"])
def test_prompt_variants_only_change_system_instructions(style):
    request = GenerationRequest("普通用户问题", "[S1]\n内容：可用资料", "test", "https://example.test/v1",
                                style, "never-in-payload")
    payloads = {v: provider_request_payload(replace(request, prompt_variant=v))
                for v in ("baseline", "behavior", "completeness", "combined")}
    assert generation_system_prompt("baseline") == SYSTEM_PROMPT
    for variant, payload in payloads.items():
        assert "never-in-payload" not in str(payload)
        if style == "responses":
            assert payload["input"] == payloads["baseline"]["input"]
        else:
            assert payload["messages"][1] == payloads["baseline"]["messages"][1]
    assert len({generation_system_prompt(v) for v in payloads}) == 4


@pytest.mark.parametrize("question,body,missing", [
    ("ZX-7300i温度怎么查？", "机型间有差异，例如ZX7300i温度采集板。", False),
    ("ZX-7300i温度怎么查？", "机型ZX7300温度采集板。", True),
    ("ZX-7300i温度怎么查？", "机型ZX7300m温度采集板。", True),
    ("ZX-7300i温度怎么查？", "其他机型支持该功能。", True),
    ("物料MAT-040-000765-00是什么？", "物料MAT04000076500", True),
    ("部件ZX81在哪里？", "相同功能的部件ZX82。", True),
    ("报警AB-123是什么意思？", "报警AB123：通道异常。", True),
])
def test_model_format_equivalence_preserves_suffix_and_other_identifiers(question, body, missing):
    result = _identifier_binding_check(SimpleNamespace(question=question), "见资料[S1]", "[S1]\n内容：" + body)
    assert bool(result["identifiers_missing_from_cited_sources"]) == missing


def test_uncited_model_scope_is_not_a_proof():
    context = "[S1]\n内容：该诊断失败的原因。\n[S2]\n内容：机型ZX-7300i的另一项功能。"
    result = _identifier_binding_check(SimpleNamespace(question="ZX-7300i诊断为何失败？"), "可能原因见[S1]", context)
    assert result["status"] == "review"


def test_experiment_prompts_have_no_gold_question_or_answer_constants():
    for variant in ("behavior", "completeness", "combined"):
        prompt = generation_system_prompt(variant)
        assert not any(value in prompt for value in ("rag-biz", "CSDB", "LVC", "BM10", "22.8", "115±5"))
