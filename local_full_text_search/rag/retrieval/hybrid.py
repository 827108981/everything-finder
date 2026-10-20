from __future__ import annotations

from typing import Callable

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, QueryPlan, RagQuery
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.query_interpreter import QueryInterpreter


class HybridRetriever:
    def __init__(
        self,
        lexical: LexicalRetriever,
        vector: VectorRetriever,
        *,
        rrf_k: int = 60,
        progress_callback: Callable[..., None] | None = None,
    ):
        self.lexical = lexical
        self.vector = vector
        self.rrf_k = max(1, int(rrf_k))
        self.progress_callback = progress_callback
        self.query_interpreter = QueryInterpreter()

    def retrieve(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        plan = plan or self.query_interpreter.interpret(query.text, token)
        if self.progress_callback:
            self.progress_callback(
                "lexical_retrieval",
                "正在执行关键词检索",
                lexical_terms=list(plan.lexical_terms),
                lexical_fts_expression=plan.lexical_expression,
            )
        lexical = self.lexical.retrieve(query, token, plan)
        token.throw_if_cancelled()
        if self.progress_callback:
            self.progress_callback(
                "vector_retrieval",
                "正在执行向量检索",
                lexical_count=len(lexical),
                lexical_candidates=[candidate_snapshot(item) for item in lexical],
                lexical_terms=list(plan.lexical_terms),
                lexical_fts_expression=plan.lexical_expression,
            )
        vector = self.vector.retrieve(query, token, plan)
        if self.progress_callback:
            self.progress_callback(
                "fusion",
                "正在合并关键词与向量候选",
                vector_count=len(vector),
                vector_candidates=[candidate_snapshot(item) for item in vector],
            )
        merged: dict[str, EvidenceChunk] = {}
        for item in lexical:
            token.throw_if_cancelled()
            key = item.chunk.stable_chunk_id
            target = merged.setdefault(key, EvidenceChunk(chunk=item.chunk))
            target.lexical_rank = item.lexical_rank
            target.lexical_score = item.lexical_score
            target.fused_score += 1.0 / (self.rrf_k + int(item.lexical_rank or 1))
        for item in vector:
            token.throw_if_cancelled()
            key = item.chunk.stable_chunk_id
            target = merged.setdefault(key, EvidenceChunk(chunk=item.chunk))
            target.vector_rank = item.vector_rank
            target.vector_score = item.vector_score
            target.fused_score += 1.0 / (self.rrf_k + int(item.vector_rank or 1))
        ordered = sorted(
            merged.values(),
            key=lambda item: (-item.fused_score, item.chunk.stable_chunk_id),
        )
        # 留出空间给后续去重；否则高排名重复块会让去重后候选量不足。
        return ordered[: max(1, int(query.fusion_top_k) * 2)]
