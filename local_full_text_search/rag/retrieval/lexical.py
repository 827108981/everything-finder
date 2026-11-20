from __future__ import annotations

from dataclasses import replace
import time
import re

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.index_store import RagBundle
from local_full_text_search.rag.lexical_index_store import LexicalIndexStore
from local_full_text_search.rag.models import EvidenceChunk, RagQuery, RetrievalRouteHit
from local_full_text_search.rag.query_analyzer import QueryAnalysis, QueryAnalyzer
from local_full_text_search.rag.models import QueryPlan
from local_full_text_search.rag.retrieval.lexical_analyzer import get_lexical_analyzer


class LexicalRetriever:
    def __init__(
        self, store: RagIndexStore, metadata_path, *, multi_route_enabled: bool = False,
        backend: str = "legacy", bundle: RagBundle | None = None,
        title_weight: float = 0.2, complete_match_enabled: bool = True,
        k1: float = 1.2, b: float = 0.75,
    ) -> None:
        self.store = store
        self.metadata_path = metadata_path
        self.analyzer = QueryAnalyzer()
        self.v2_analyzer = get_lexical_analyzer() if backend == "word_v2" else None
        self.last_analysis: QueryAnalysis | QueryPlan | None = None
        self.multi_route_enabled = multi_route_enabled
        self.backend = str(backend or "legacy")
        self.bundle = bundle
        self.title_weight = float(title_weight)
        self.complete_match_enabled = bool(complete_match_enabled)
        self.k1 = float(k1)
        self.b = float(b)
        self.route_diagnostics: list[dict[str, object]] = []
        if self.backend not in {"legacy", "word_v2"}:
            raise ValueError(f"不支持的关键词检索后端：{self.backend}")
        if self.backend == "word_v2" and self.multi_route_enabled:
            raise ValueError("word_v2 与旧关键词多路线不能同时启用")
        if self.backend == "word_v2" and self.bundle is None:
            raise ValueError("word_v2 必须绑定明确的 AI 索引 bundle")

    def retrieve(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        results = self._retrieve_original(query, token, plan)
        if not plan or not plan.adapter_active:
            return results
        from local_full_text_search.rag.identifier_evidence import identifiers, exact, prioritize
        # Independent exact route on the published metadata FTS, for both backends.
        # Original lexical search always runs; filters are identical on all routes.
        merged = {i.chunk.stable_chunk_id:i for i in results}
        connection = self.store.connect_read_only(self.metadata_path)
        try:
            soft_ids=tuple(dict.fromkeys(e.text for e in plan.unknown_entities if e.type == 'identifier_candidate' and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{1,30}',e.text)))[:3]
            for value in dict.fromkeys((*identifiers(plan),*soft_ids)):
                token.throw_if_cancelled()
                expression='"'+value.replace('"','""')+'"' if len(value)>=3 else ''
                p=replace(plan,lexical_expression=expression,lexical_terms=(value,),fallback_terms=(value,))
                hits=self.store.lexical_search(connection,value,top_k=query.lexical_top_k,
                    root_ids=query.root_ids,extensions=query.extensions,include_ocr=query.include_ocr,
                    min_ocr_confidence=query.min_ocr_confidence,analysis=p)
                hits=[i for i in hits if exact(i.chunk.text,value)]
                self.route_diagnostics.append({'route_id':'identifier_exact' if value not in soft_ids else 'unknown_identifier_hint','query_text':value,
                    'candidate_count':len(hits),'reason':'Exact body token; no inferred identity or meaning'})
                for rank,i in enumerate(hits,1):
                    target=merged.setdefault(i.chunk.stable_chunk_id,i)
                    target.lexical_route_hits+=(RetrievalRouteHit('identifier_exact:'+value,rank,i.lexical_score,'bm25_raw'),)
        finally:
            connection.close()
        from local_full_text_search.rag.source_consensus import supplement, reserve
        results=reserve(prioritize(plan,supplement(self,query,plan,list(merged.values()),token)))[:query.lexical_top_k]
        for rank,i in enumerate(results,1): i.lexical_rank=rank
        return results

    def _retrieve_original(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        token.throw_if_cancelled()
        if self.backend == "word_v2":
            assert self.bundle is not None and self.v2_analyzer is not None
            lexical_store = LexicalIndexStore(self.store.root)
            lexical_bundle = lexical_store.active(
                self.bundle.bundle_id, self.v2_analyzer.fingerprint,
            )
            diagnostics: dict[str, object] = {}
            results = lexical_store.search(
                lexical_bundle, self.store, self.bundle, self.v2_analyzer, query,
                title_weight=self.title_weight,
                complete_match_enabled=self.complete_match_enabled,
                k1=self.k1, b=self.b, diagnostics=diagnostics,
            )
            self.route_diagnostics = [{"route_id": "lexical_word_v2", **diagnostics}]
            token.throw_if_cancelled()
            return results
        connection = self.store.connect_read_only(self.metadata_path)
        try:
            self.last_analysis = plan or self.analyzer.analyze(query.text)
            self.route_diagnostics = []
            routes = plan.lexical_variants if self.multi_route_enabled and plan and plan.lexical_variants else ()
            if not routes:
                started = time.monotonic()
                search_diagnostics: dict[str, object] = {}
                results = self.store.lexical_search(
                    connection, query.text, top_k=query.lexical_top_k,
                    root_ids=query.root_ids, extensions=query.extensions,
                    include_ocr=query.include_ocr,
                    min_ocr_confidence=query.min_ocr_confidence,
                    analysis=self.last_analysis,
                    diagnostics=search_diagnostics,
                )
                self.route_diagnostics.append({
                    "route_id": "lexical_legacy", "candidate_count": len(results),
                    "duration_ms": int((time.monotonic() - started) * 1000),
                    "expression": self.last_analysis.fts_expression,
                    **search_diagnostics,
                })
            else:
                merged: dict[str, EvidenceChunk] = {}
                seen_expressions: set[tuple[str, tuple[str, ...]]] = set()
                for route in routes:
                    token.throw_if_cancelled()
                    route_key = (route.expression, route.signal_texts if not route.expression else ())
                    if route_key in seen_expressions:
                        continue
                    seen_expressions.add(route_key)
                    route_plan = replace(plan, lexical_expression=route.expression,
                                         fallback_terms=route.signal_texts or plan.fallback_terms)
                    started = time.monotonic()
                    search_diagnostics = {}
                    route_results = self.store.lexical_search(
                        connection, query.text, top_k=query.lexical_top_k,
                        root_ids=query.root_ids, extensions=query.extensions,
                        include_ocr=query.include_ocr,
                        min_ocr_confidence=query.min_ocr_confidence,
                        analysis=route_plan,
                        diagnostics=search_diagnostics,
                    )
                    self.route_diagnostics.append({
                        "route_id": route.route_id, "candidate_count": len(route_results),
                        "duration_ms": int((time.monotonic() - started) * 1000),
                        "expression": route.expression, "query_text": route.query_text,
                        **search_diagnostics,
                    })
                    for item in route_results:
                        chunk_id = item.chunk.stable_chunk_id
                        target = merged.setdefault(chunk_id, EvidenceChunk(chunk=item.chunk))
                        rank = int(item.lexical_rank or 1)
                        target.lexical_route_rrf_score = (
                            (target.lexical_route_rrf_score or 0.0) + 1.0 / (60 + rank)
                        )
                        target.lexical_route_hits += (RetrievalRouteHit(
                            route.route_id, rank, item.lexical_score, "bm25_raw"),)
                        if target.lexical_score is None or rank < min(
                            hit.rank for hit in target.lexical_route_hits[:-1]
                        ):
                            target.lexical_score = item.lexical_score
                results = sorted(merged.values(), key=lambda item: (
                    -float(item.lexical_route_rrf_score or 0.0), item.chunk.stable_chunk_id,
                ))[:query.lexical_top_k]
                for rank, item in enumerate(results, 1):
                    item.lexical_rank = rank
            token.throw_if_cancelled()
            return results
        finally:
            connection.close()
