from __future__ import annotations

from typing import Protocol

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import EvidenceChunk, QueryPlan, RagQuery


class Retriever(Protocol):
    def retrieve(
        self,
        query: RagQuery,
        token: CancelToken,
        plan: QueryPlan | None = None,
    ) -> list[EvidenceChunk]: ...
