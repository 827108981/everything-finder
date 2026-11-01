from __future__ import annotations

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import RagQuery, SourceBlock
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever


def _chunk(index: int, filename: str, text: str):
    block = SourceBlock(
        source_generation=1, block_id=index, document_id=index, file_id=index,
        root_id=1, file_path=f"E:/资料/{filename}", filename=filename,
        extension=".pdf", document_key=f"doc-{index}", block_index=0,
        text=text, location_text="第 1 页", page_number=1,
    )
    return next(ChunkBuilder().iter_chunks([block]))


def test_unknown_short_identifier_uses_boundary_and_filename(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    candidate, connection = store.create_candidate(
        1, embedding_repo_id="test/embedding", embedding_revision="a" * 40,
        embedding_dimension=512, reranker_repo_id="test/reranker", reranker_revision="b" * 40,
    )
    try:
        for index in range(1, 11):
            store.add_chunk(connection, _chunk(index, f"AJ80-{index}.pdf", "接口连接说明"))
        store.add_chunk(connection, _chunk(11, "J8接口.pdf", "连接说明"))
        connection.commit()
        plan = QueryInterpreter().interpret("J8 接口连接到哪里？")
        assert any(item.route_id == "lexical_short_identifier" for item in plan.lexical_variants)
        retriever = LexicalRetriever(store, candidate.metadata_path, multi_route_enabled=True)
        result = retriever.retrieve(RagQuery(text="J8 接口连接到哪里？", lexical_top_k=5), CancelToken(), plan)
        assert any(item.chunk.source.filename == "J8接口.pdf" for item in result)
        assert all("J80" not in item.chunk.source.filename for item in result if item.lexical_route_hits and
                   all(hit.route_id == "lexical_short_identifier" for hit in item.lexical_route_hits))
        assert any(hit.route_id == "lexical_short_identifier" for item in result for hit in item.lexical_route_hits)
    finally:
        store.abort(candidate, connection)


def test_short_identifier_pages_past_unrelated_like_hits(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    candidate, connection = store.create_candidate(
        1, embedding_repo_id="test/embedding", embedding_revision="a" * 40,
        embedding_dimension=512, reranker_repo_id="test/reranker", reranker_revision="b" * 40,
    )
    try:
        for index in range(1, 1051):
            store.add_chunk(connection, _chunk(index, f"AJ80-{index}.pdf", "J80 接口说明"))
        store.add_chunk(connection, _chunk(1051, "J8接口.pdf", "J8 连接说明"))
        connection.commit()
        plan = QueryInterpreter().interpret("J8 接口连接到哪里？")
        route = next(item for item in plan.lexical_variants
                     if item.route_id == "lexical_short_identifier")
        from dataclasses import replace
        search_plan = replace(plan, lexical_expression="", fallback_terms=route.signal_texts)
        diagnostics: dict[str, object] = {}
        result = store.lexical_search(
            connection, "J8", top_k=5, analysis=search_plan,
            diagnostics=diagnostics,
        )
        assert [item.chunk.source.filename for item in result] == ["J8接口.pdf"]
        assert diagnostics["scanned_count"] > 1000
        assert diagnostics["candidate_pool_exhausted"] is False
    finally:
        store.abort(candidate, connection)
