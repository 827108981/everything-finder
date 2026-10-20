from __future__ import annotations

from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import SourceBlock
from local_full_text_search.rag.query_analyzer import QueryAnalyzer


def _chunk(text: str, block_id: int, filename: str):
    block = SourceBlock(
        source_generation=1,
        block_id=block_id,
        document_id=block_id,
        file_id=block_id,
        root_id=1,
        file_path=f"C:/资料/{filename}",
        filename=filename,
        extension=".pdf",
        document_key=f"doc-{block_id}",
        block_index=0,
        text=text,
        location_text=f"第 {block_id} 页",
        page_number=block_id,
    )
    return next(ChunkBuilder().iter_chunks([block]))


def test_query_analyzer_removes_question_shell_and_expands_stable_status() -> None:
    analysis = QueryAnalyzer().analyze("如何解决免疫底物本底不通过的问题？")

    assert "底物本底" in analysis.terms
    assert "如何解决" not in analysis.fts_expression
    assert "未通过" in analysis.fallback_terms
    assert " OR " in analysis.fts_expression


def test_bm25_ranks_analyzed_keyword_matches(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    candidate, connection = store.create_candidate(
        1,
        embedding_repo_id="test/embedding",
        embedding_revision="a" * 40,
        embedding_dimension=512,
        reranker_repo_id="test/reranker",
        reranker_revision="b" * 40,
    )
    store.add_chunk(
        connection,
        _chunk("底物本底检测未通过处理思路：检查底物瓶和管路。", 1, "故障手册.pdf"),
    )
    store.add_chunk(
        connection,
        _chunk("免疫底物需要避光保存。", 2, "保存说明.pdf"),
    )
    connection.commit()
    try:
        results = store.lexical_search(
            connection,
            "如何解决免疫底物本底不通过的问题？",
            top_k=10,
        )
        assert [item.chunk.source.filename for item in results] == [
            "故障手册.pdf",
            "保存说明.pdf",
        ]
        assert results[0].lexical_rank == 1
        assert results[0].lexical_score is not None
        assert results[0].lexical_score < results[1].lexical_score
    finally:
        store.abort(candidate, connection)
