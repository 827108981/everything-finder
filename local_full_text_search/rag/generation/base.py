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
    # Explicit experimental override; normal callers retain the accepted prompt.
    prompt_variant: str = "baseline"


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
class TextCompletionRequest:
    system_prompt: str
    user_prompt: str
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


@dataclass(frozen=True, slots=True)
class ClarificationAssessment:
    needs_clarification: bool
    question: str = ""
    missing_fields: tuple[str, ...] = ()
    evidence: tuple[tuple[str, str], ...] = ()
    reason: str = ""
    decision_type: str = ""
    request_quote: str = ""


class LLMGateway(Protocol):
    def generate(self, request: GenerationRequest, token: CancelToken) -> ModelAnswer: ...

    def review_answer(
        self, request: AnswerReviewRequest, token: CancelToken
    ) -> AnswerReviewResult: ...
