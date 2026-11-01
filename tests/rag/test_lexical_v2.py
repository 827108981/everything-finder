from __future__ import annotations

import math

import pytest

from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.errors import LexicalIndexNotReadyError
from local_full_text_search.rag.lexical_index_builder import LexicalIndexBuilder
from local_full_text_search.rag.models import RagQuery, SourceBlock
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.retrieval.lexical_analyzer import LexicalAnalyzer
from local_full_text_search.rag.retrieval.lexical_scoring import (
    bm25_term_score, inverse_document_frequency,
)


def _chunk(index: int, filename: str, text: str, *, root_id: int = 1, path_prefix: str = "E:/资料"):
    block = SourceBlock(
        source_generation=1, block_id=index, document_id=index, file_id=index,
        root_id=root_id, file_path=f"{path_prefix}/{filename}", filename=filename,
        extension=".pdf", document_key=f"doc-{index}", block_index=0,
        text=text, location_text="第 1 页", page_number=1,
    )
    return next(ChunkBuilder().iter_chunks([block]))


def _prepared_store(tmp_path, rows):
    store = RagIndexStore(tmp_path / "rag")
    bundle, connection = store.create_candidate(
        1, embedding_repo_id="test/embedding", embedding_revision="a" * 40,
        embedding_dimension=8, reranker_repo_id="test/reranker", reranker_revision="b" * 40,
    )
    for row in rows:
        store.add_chunk(connection, row)
    connection.commit()
    lexical, reused = LexicalIndexBuilder(store).prepare(bundle, CancelToken())
    return store, bundle, connection, lexical, reused


def _search(store, bundle, text: str, **query_args):
    retriever = LexicalRetriever(store, bundle.metadata_path, backend="word_v2", bundle=bundle)
    return retriever, retriever.retrieve(RagQuery(text=text, **query_args), CancelToken())


def test_analyzer_keeps_short_words_identifiers_and_deduplicates_query_terms() -> None:
    analyzed = LexicalAnalyzer().analyze_query("P02 制冷 泵，制冷；J8不能写成J80")
    terms = [item.normalized for item in analyzed.tokens]
    assert "p02" in terms
    assert "制冷" in terms
    assert "泵" in terms
    assert "j8" in terms and "j80" in terms
    assert terms.count("制冷") == 1
    assert "制冷" in analyzed.duplicate_terms


def test_scoring_formula_is_independently_calculable() -> None:
    expected_idf = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    assert inverse_document_frequency(3, 1) == pytest.approx(expected_idf)
    expected = expected_idf * 2 * 2.2 / (2 + 1.2 * (1 - 0.75 + 0.75 * 10 / 8))
    assert bm25_term_score(
        document_count=3, document_frequency=1, term_frequency=2,
        field_length=10, average_field_length=8, k1=1.2, b=0.75,
    ) == pytest.approx(expected)


def test_v2_ranks_body_evidence_and_exposes_contributions(tmp_path) -> None:
    rows = [
        _chunk(1, "BS-2800M维修手册.pdf", "试剂盘采用半导体制冷方式，控制目标温度。"),
        _chunk(2, "制冷资料.pdf", "免疫分析仪清洗液缓冲容器说明。"),
        _chunk(3, "其他资料.pdf", "试剂盘日常维护说明。"),
    ]
    store, bundle, connection, _, _ = _prepared_store(tmp_path, rows)
    try:
        retriever, results = _search(store, bundle, "BS-2800M试剂盘采用什么制冷方式？")
        assert results[0].chunk.source.filename == "BS-2800M维修手册.pdf"
        assert results[0].lexical_score_kind == "lexical_bm25_fields_v2"
        terms = {item["term"] for item in results[0].lexical_contributions}
        # 通用词典可能把“试剂盘”切成“试剂/盘”；测试检索信号，不绑定业务词典切法。
        assert {"bs-2800m", "试剂", "盘", "制冷"} <= terms
        assert retriever.route_diagnostics[0]["actual_terms"]
        assert retriever.route_diagnostics[0]["backend"] == "word_v2"
    finally:
        connection.close()


def test_identifier_boundary_does_not_match_longer_identifier(tmp_path) -> None:
    rows = [
        _chunk(1, "J8接口.pdf", "J8连接到控制板。"),
        _chunk(2, "AJ80接口.pdf", "AJ80连接到控制板。"),
        _chunk(3, "J80接口.pdf", "J80连接到控制板。"),
    ]
    store, bundle, connection, _, _ = _prepared_store(tmp_path, rows)
    try:
        _, results = _search(store, bundle, "J8连接到哪里？")
        matched = [item.chunk.source.filename for item in results if any(
            row["term"] == "j8" for row in item.lexical_contributions
        )]
        assert matched == ["J8接口.pdf"]
    finally:
        connection.close()


def test_complete_text_locator_repairs_context_dependent_tokenization(tmp_path) -> None:
    # 查询侧“乒乓球拍”是一个词，正文“乒乓球拍卖完了”会切成“乒乓球/拍卖…”。
    # 只有连续原文核验可以补回，不能把分散出现的字拼成命中。
    rows = [
        _chunk(1, "目标.pdf", "仓库里的乒乓球拍卖完了。"),
        _chunk(2, "干扰.pdf", "乒乓球已经售完，球拍仍有库存。"),
    ]
    store, bundle, connection, _, _ = _prepared_store(tmp_path, rows)
    try:
        retriever, results = _search(store, bundle, "乒乓球拍")
        assert results[0].chunk.source.filename == "目标.pdf"
        assert any(
            row["term"] == "乒乓球拍" and row["match_methods"] == ["complete_text_locator"]
            for row in results[0].lexical_contributions
        )
        assert "乒乓球拍" in retriever.route_diagnostics[0]["complete_match_terms"]
    finally:
        connection.close()


def test_filters_apply_before_topk_and_same_index_is_reused(tmp_path) -> None:
    rows = [
        *[_chunk(index, f"根一{index}.pdf", "P02接口连接位置说明。", root_id=1)
          for index in range(1, 61)],
        _chunk(100, "目标.pdf", "P02接口连接位置说明。", root_id=2),
    ]
    store, bundle, connection, lexical, reused = _prepared_store(tmp_path, rows)
    try:
        assert reused is False
        same, reused_again = LexicalIndexBuilder(store).prepare(bundle, CancelToken())
        assert reused_again is True
        assert same.database_path == lexical.database_path
        _, results = _search(store, bundle, "P02接口连接位置", root_ids=(2,), lexical_top_k=1)
        assert [item.chunk.source.filename for item in results] == ["目标.pdf"]
    finally:
        connection.close()


def test_cancelled_build_does_not_publish(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    bundle, connection = store.create_candidate(
        1, embedding_repo_id="test", embedding_revision="a", embedding_dimension=2,
        reranker_repo_id="test", reranker_revision="b",
    )
    store.add_chunk(connection, _chunk(1, "资料.pdf", "短词制冷与泵"))
    connection.commit()
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        LexicalIndexBuilder(store).prepare(bundle, token)
    assert not (store.root / "lexical" / bundle.bundle_id / "active.json").exists()
    connection.close()


def test_v2_missing_index_and_legacy_route_conflict_are_explicit(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    bundle, connection = store.create_candidate(
        1, embedding_repo_id="test", embedding_revision="a", embedding_dimension=2,
        reranker_repo_id="test", reranker_revision="b",
    )
    store.add_chunk(connection, _chunk(1, "资料.pdf", "制冷方式"))
    connection.commit()
    missing = LexicalRetriever(store, bundle.metadata_path, backend="word_v2", bundle=bundle)
    with pytest.raises(LexicalIndexNotReadyError, match="关键词 V2 索引未就绪"):
        missing.retrieve(RagQuery(text="制冷"), CancelToken())
    with pytest.raises(ValueError, match="不能同时启用"):
        LexicalRetriever(
            store, bundle.metadata_path, backend="word_v2", bundle=bundle,
            multi_route_enabled=True,
        )
    connection.close()
