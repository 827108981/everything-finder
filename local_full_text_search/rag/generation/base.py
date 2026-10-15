from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from local_full_text_search.core.task_manager import CancelToken


class LLMUnavailableError(RuntimeError):
    """Raised when the configured LLM API cannot answer."""


@dataclass(frozen=True, slots=True)
class ModelHealth:
    available: bool
    model_id: str = ""
    message: str = ""


class LLMGateway(Protocol):
    def health(self) -> ModelHealth: ...

    def count_chat_tokens(self, messages: list[dict[str, str]]) -> int: ...

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        cancel_token: CancelToken,
    ) -> str: ...
