from __future__ import annotations


from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, QueryPlan, RagQuery


class VectorRetriever:
    def __init__(self, store: RagIndexStore, bundle, vector_store, embedding) -> None:
        self.store = store
        self.bundle = bundle
        self.vector_store = vector_store
        self.embedding = embedding

    def retrieve(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        token.throw_if_cancelled()
        vector = self.embedding.embed_query(plan.semantic_query if plan else query.text)
        token.throw_if_cancelled()
        hits = self.vector_store.query(
            vector,
            max(query.vector_top_k * 4, query.vector_top_k),
            root_ids=query.root_ids,
            extensions=query.extensions,
            include_ocr=query.include_ocr,
        )
        connection = self.store.connect_read_only(self.bundle.metadata_path)
        try:
            results: list[EvidenceChunk] = []
            seen: set[str] = set()
            for hit in hits:
                token.throw_if_cancelled()
                chunk_id = str(hit.metadata.get("stable_chunk_id") or hit.point_id)
                if chunk_id in seen:
                    continue
                chunk = self.store.load_chunk(connection, chunk_id)
                if chunk is None:
                    continue
                if not self.store._matches_filters(
                    chunk,
                    query.root_ids,
                    query.extensions,
                    query.include_ocr,
                    query.min_ocr_confidence,
                ):
                    continue
                seen.add(chunk_id)
                results.append(
                    EvidenceChunk(
                        chunk=chunk,
                        vector_rank=len(results) + 1,
                        vector_score=float(hit.score),
                    )
                )
                if len(results) >= query.vector_top_k:
                    break
            return results
        finally:
            connection.close()
