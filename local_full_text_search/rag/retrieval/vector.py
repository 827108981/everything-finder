from __future__ import annotations

import time

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import (
    EvidenceChunk, QueryPlan, RagQuery, RetrievalRouteHit, VectorUnitHit,
)
from local_full_text_search.rag.vector_index_store import VectorIndexBundle, VectorIndexStore


class VectorRetriever:
    def __init__(
        self, store: RagIndexStore, bundle, vector_store, embedding, *,
        focused_enabled: bool = False, backend: str = "legacy",
        unit_index: VectorIndexBundle | None = None, oversample_factor: int = 8,
    ) -> None:
        self.store = store
        self.bundle = bundle
        self.vector_store = vector_store
        self.embedding = embedding
        self.focused_enabled = focused_enabled
        self.backend = str(backend)
        if self.backend not in {"legacy", "unit_v2"}:
            raise ValueError(f"不支持的向量检索后端：{self.backend}")
        if self.backend == "unit_v2" and unit_index is None:
            raise ValueError("unit_v2 必须绑定细粒度向量索引")
        self.unit_index = unit_index
        self.oversample_factor = max(1, int(oversample_factor))
        self.route_diagnostics: list[dict[str, object]] = []

    def retrieve(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        token.throw_if_cancelled()
        variants = [("semantic_original", plan.semantic_query if plan else query.text)]
        if self.focused_enabled and plan:
            for variant in plan.semantic_variants:
                if variant.route_id == "semantic_focused" and variant.query_text != variants[0][1]:
                    variants.append((variant.route_id, variant.query_text))
        if plan and plan.adapter_active:
            seen={value for _,value in variants}
            for n,sub in enumerate(plan.sub_queries,1):
                if sub.text not in seen:
                    variants.append((f'answer_target_{n}',sub.text))
                    seen.add(sub.text)
        route_hits = []
        self.route_diagnostics = []
        for route_id, route_text in variants:
            token.throw_if_cancelled()
            started = time.monotonic()
            try:
                vector = self.embedding.embed_query(route_text)
                token.throw_if_cancelled()
                multiplier = self.oversample_factor if self.backend == "unit_v2" else 4
                requested = max(query.vector_top_k * multiplier, query.vector_top_k)
                hits = self._query_hits(vector, requested, query)
                if self.backend == "unit_v2":
                    available = self.vector_store.count()
                    while (
                        len({str(hit.metadata.get("stable_chunk_id") or hit.point_id) for hit in hits})
                        < query.vector_top_k and requested < available
                    ):
                        requested = min(available, requested * 2)
                        hits = self._query_hits(vector, requested, query)
                route_hits.append((route_id, hits))
                self.route_diagnostics.append({
                    "route_id": route_id, "candidate_count": len(hits),
                    "parent_candidate_count": len({
                        str(hit.metadata.get("stable_chunk_id") or hit.point_id) for hit in hits
                    }),
                    "requested_unit_depth": requested,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "query_text": route_text, "status": "completed",
                    "backend": self.backend,
                    "vector_index_fingerprint": (
                        self.unit_index.index_fingerprint if self.unit_index else ""
                    ),
                })
            except Exception:
                if route_id == "semantic_original" or token.cancelled:
                    raise
                # 补充路线故障时原始问题仍可检索；诊断由调用方记录降级。
                self.route_diagnostics.append({
                    "route_id": route_id, "candidate_count": 0,
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "query_text": route_text, "status": "degraded_route",
                })
                break
        connection = self.store.connect_read_only(self.bundle.metadata_path)
        try:
            merged: dict[str, EvidenceChunk] = {}
            unit_details = self._load_unit_details(route_hits)
            for route_id, hits in route_hits:
                route_count = 0
                seen_parents: set[str] = set()
                for raw_rank, hit in enumerate(hits, 1):
                    token.throw_if_cancelled()
                    chunk_id = str(hit.metadata.get("stable_chunk_id") or hit.point_id)
                    chunk = self.store.load_chunk(connection, chunk_id)
                    if chunk is None or not self.store._matches_filters(
                        chunk, query.root_ids, query.extensions, query.include_ocr,
                        query.min_ocr_confidence,
                    ):
                        continue
                    target = merged.setdefault(chunk_id, EvidenceChunk(chunk=chunk))
                    if self.backend == "unit_v2":
                        detail = unit_details.get(str(hit.point_id), {})
                        target.vector_unit_hits += (VectorUnitHit(
                            unit_id=str(hit.point_id), route_id=route_id, raw_rank=raw_rank,
                            score=float(hit.score), text=str(detail.get("text") or ""),
                            kind=str(detail.get("kind") or hit.metadata.get("unit_kind") or "window"),
                            start_char=int(detail.get("start_char") or 0),
                            end_char=int(detail.get("end_char") or 0),
                            token_count=int(detail.get("token_count") or 0),
                            embedding_text=str(detail.get("embedding_text") or "") if detail.get("representation_metadata") is not None else "",
                            representation_metadata=detail.get("representation_metadata"),
                        ),)
                    if chunk_id in seen_parents:
                        continue
                    seen_parents.add(chunk_id)
                    route_count += 1
                    target.vector_route_hits += (RetrievalRouteHit(
                        route_id, route_count, float(hit.score), "vector_similarity"
                    ),)
                    target.vector_route_rrf_score = (
                        (target.vector_route_rrf_score or 0.0) + 1.0 / (60 + route_count)
                    )
                    if route_id == "semantic_original" or target.vector_score is None:
                        target.vector_score = float(hit.score)
                    if route_count >= query.vector_top_k:
                        break
            if len(route_hits) == 1:
                ordered = sorted(merged.values(), key=lambda item: (
                    item.vector_route_hits[0].rank, item.chunk.stable_chunk_id,
                ))
            else:
                ordered = sorted(merged.values(), key=lambda item: (
                    -float(item.vector_route_rrf_score or 0.0), item.chunk.stable_chunk_id,
                ))
            results = ordered[:query.vector_top_k]
            for rank, item in enumerate(results, 1):
                item.vector_rank = rank
            return results
        finally:
            connection.close()

    def _load_unit_details(self, route_hits) -> dict[str, dict[str, object]]:
        if self.backend != "unit_v2" or self.unit_index is None:
            return {}
        unit_ids = list(dict.fromkeys(
            str(hit.point_id) for _, hits in route_hits for hit in hits
        ))
        return VectorIndexStore.load_units(self.unit_index.database_path, unit_ids)

    def _query_hits(self, vector, top_k: int, query: RagQuery):
        return self.vector_store.query(
            vector, top_k, root_ids=query.root_ids, extensions=query.extensions,
            include_ocr=query.include_ocr,
        )
