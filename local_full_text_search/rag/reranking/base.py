from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import EvidenceChunk


class Reranker(Protocol):
    def rerank(
        self,
        question: str,
        candidates: Sequence[EvidenceChunk],
        top_n: int,
        token: CancelToken,
    ) -> list[EvidenceChunk]: ...
