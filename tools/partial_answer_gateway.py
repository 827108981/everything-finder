"""Opt-in generation prompt candidate for isolated answer-layer comparisons."""

from __future__ import annotations

from local_full_text_search.rag.generation.base import ModelAnswer
from local_full_text_search.rag.generation.openai_compatible import (
    OpenAICompatibleGateway, SYSTEM_PROMPT, provider_request_payload,
)


PARTIAL_RULE = """逐项核对用户实际所问。某一项有直接、适用的原文依据时，回答这一项并引用对应来源；另一项缺依据时，仅说明该项资料不足，不要把整题都拒绝。
原因、处理方向和条件分支按问题需要表达，不把一种可能性写成唯一诊断，不添加用户未问的机型参数或操作。每一条引用只支持它紧邻的事实。回答前检查所问步骤、先后顺序及必要数值单位是否遗漏。"""


def partial_provider_payload(request):
    payload = provider_request_payload(request)
    old_rule = "如果证据无法回答问题，请只回答“资料中没有足够依据。”。"
    new_rule = "如果全部所问都没有直接依据，请只回答“资料中没有足够依据。”。"
    system = SYSTEM_PROMPT.replace(old_rule, new_rule) + "\n" + PARTIAL_RULE
    if request.api_style == "chat_completions":
        payload["messages"][0]["content"] = system
    else:
        payload["instructions"] = system
    return payload


class PartialAnswerGateway(OpenAICompatibleGateway):
    def generate(self, request, token):
        self._validate(request)
        text = self._request_text(partial_provider_payload(request), request, token)
        return ModelAnswer(text, "OpenAI Compatible", request.model)
