from __future__ import annotations

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, RagQuery
from local_full_text_search.rag.query_analyzer import QueryAnalysis, QueryAnalyzer
from local_full_text_search.rag.models import QueryPlan


class LexicalRetriever:
    def __init__(self, store: RagIndexStore, metadata_path) -> None:
        self.store = store
        self.metadata_path = metadata_path
        self.analyzer = QueryAnalyzer()
        self.last_analysis: QueryAnalysis | QueryPlan | None = None

    def retrieve(
        self, query: RagQuery, token: CancelToken, plan: QueryPlan | None = None
    ) -> list[EvidenceChunk]:
        token.throw_if_cancelled()
        connection = self.store.connect_read_only(self.metadata_path)
        try:
            self.last_analysis = plan or self.analyzer.analyze(query.text)
            results = self.store.lexical_search(
                connection,
                query.text,
                top_k=query.lexical_top_k,
                root_ids=query.root_ids,
                extensions=query.extensions,
                include_ocr=query.include_ocr,
                min_ocr_confidence=query.min_ocr_confidence,
                analysis=self.last_analysis,
            )
            token.throw_if_cancelled()
            return results
        finally:
            connection.close()
