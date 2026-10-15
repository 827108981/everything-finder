from __future__ import annotations

import json
from collections.abc import Callable

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.generation.base import ModelHealth


class MockLLMGateway:
    def __init__(self, response: dict[str, object] | Callable[[list[dict[str, str]]], dict[str, object]] | None = None) -> None:
        self.response = response or {
            "insufficient": False,
            "answer": "测试答案。[S1]",
            "used_sources": ["S1"],
        }
        self.calls = 0

    def health(self) -> ModelHealth:
        return ModelHealth(True, "mock-llm", "Mock LLM ready")

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int:
        return max(1, sum(len(message.get("content", "")) for message in messages) // 2)

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancel_token: CancelToken,
    ) -> str:
        cancel_token.throw_if_cancelled()
        self.calls += 1
        response = self.response(messages) if callable(self.response) else self.response
        return json.dumps(response, ensure_ascii=False)
