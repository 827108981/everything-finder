from __future__ import annotations

from collections import defaultdict

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.base import EmbeddingProvider
from local_full_text_search.rag.models import RagQuery, RankedEvidence
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.retrieval.vector import VectorRetriever


class HybridRetriever:
    def __init__(
        self,
        lexical: LexicalRetriever | None = None,
        vector: VectorRetriever | None = None,
        *,
        rrf_k: int = 60,
        branch_top_n: int = 30,
    ) -> None:
        self.lexical = lexical or LexicalRetriever()
        self.vector = vector or VectorRetriever()
        self.rrf_k = max(1, int(rrf_k))
        self.branch_top_n = max(1, min(100, int(branch_top_n)))

    def retrieve(
        self,
        connection,
        query: RagQuery,
        embedding: EmbeddingProvider,
        token: CancelToken,
        *,
        top_n: int = 20,
    ) -> list[RankedEvidence]:
        lexical_hits = self.lexical.retrieve(connection, query, self.branch_top_n)
        token.throw_if_cancelled()
        vector_hits = self.vector.retrieve(
            connection,
            query,
            embedding,
            token,
            self.branch_top_n,
        )
        fused: dict[str, dict[str, object]] = {}
        for rank, evidence in enumerate(lexical_hits, start=1):
            entry = fused.setdefault(evidence.chunk.stable_chunk_id, {"evidence": evidence, "score": 0.0})
            entry["score"] = float(entry["score"]) + 1.0 / (self.rrf_k + rank)
            entry["lexical_rank"] = rank
        for rank, evidence in enumerate(vector_hits, start=1):
            entry = fused.setdefault(evidence.chunk.stable_chunk_id, {"evidence": evidence, "score": 0.0})
            entry["score"] = float(entry["score"]) + 1.0 / (self.rrf_k + rank)
            entry["vector_rank"] = rank
        ordered = sorted(
            fused.values(),
            key=lambda entry: (
                -float(entry["score"]),
                int(entry.get("lexical_rank", 10**9)),
                int(entry.get("vector_rank", 10**9)),
                str(entry["evidence"].chunk.stable_chunk_id),
            ),
        )
        diversified: list[RankedEvidence] = []
        per_document: dict[int, int] = defaultdict(int)
        per_page: dict[tuple[int, int | None], int] = defaultdict(int)
        for entry in ordered:
            evidence = entry["evidence"]
            source = evidence.selected_source
            page_key = (source.document_id, source.page_number)
            if per_document[source.document_id] >= 3 or per_page[page_key] >= 2:
                continue
            per_document[source.document_id] += 1
            per_page[page_key] += 1
            diversified.append(
                RankedEvidence(
                    evidence=evidence,
                    lexical_rank=(int(entry["lexical_rank"]) if "lexical_rank" in entry else None),
                    vector_rank=(int(entry["vector_rank"]) if "vector_rank" in entry else None),
                    fused_score=float(entry["score"]),
                )
            )
            if len(diversified) >= max(1, int(top_n)):
                break
        return diversified
