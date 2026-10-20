from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import AnswerReviewResult


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    question: str
    context: str
    model: str
    api_base_url: str
    api_style: str
    api_key: str
    timeout_seconds: int = 90
    max_output_tokens: int = 1000
    query_plan_summary: str = ""


@dataclass(frozen=True, slots=True)
class AnswerReviewRequest:
    question: str
    query_plan_summary: str
    context: str
    draft_answer: str
    citation_check_summary: str
    allowed_source_ids: tuple[str, ...]
    model: str
    api_base_url: str
    api_style: str
    api_key: str
    timeout_seconds: int = 90
    max_output_tokens: int = 1400


@dataclass(frozen=True, slots=True)
class ModelAnswer:
    text: str
    provider_name: str
    model_name: str


class LLMGateway(Protocol):
    def generate(self, request: GenerationRequest, token: CancelToken) -> ModelAnswer: ...

    def review_answer(
        self, request: AnswerReviewRequest, token: CancelToken
    ) -> AnswerReviewResult: ...
