from __future__ import annotations

from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.models import EvidenceChunk, SourceBlock
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.reranking.support import (
    build_rerank_query,
    compute_answer_support,
    infer_question_intent,
)
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker


def _candidate(text: str, chunk_id: str) -> EvidenceChunk:
    block = SourceBlock(
        source_generation=1,
        block_id=1,
        document_id=1,
        file_id=1,
        root_id=1,
        file_path=f"C:/资料/{chunk_id}.pdf",
        filename=f"{chunk_id}.pdf",
        extension=".pdf",
        document_key=chunk_id,
        block_index=0,
        text=text,
        location_text="第 1 页",
        page_number=1,
    )
    return EvidenceChunk(chunk=next(ChunkBuilder().iter_chunks([block])))


def test_support_uses_question_intent_and_direct_answer_expression() -> None:
    plan = QueryInterpreter().interpret("BS-2800 生化分析仪的参考光路有什么作用？")
    assert infer_question_intent(plan) == "purpose"
    direct = _candidate(
        "参考光路用于监控光源稳定性，光线经过滤光片到达光电检测器。", "direct"
    )
    operation = _candidate("参考光路更换操作：拆下旧部件并安装新部件。", "operation")
    direct_score, direct_reasons = compute_answer_support(plan, direct)
    operation_score, operation_reasons = compute_answer_support(plan, operation)
    assert direct_score > operation_score
    assert any("用于" in reason for reason in direct_reasons)
    assert any("未发现" in reason for reason in operation_reasons)


def test_rerank_query_keeps_original_question_and_structured_intent() -> None:
    plan = QueryInterpreter().interpret("BS-2800 生化分析仪的参考光路有什么作用？")
    query = build_rerank_query(plan)
    assert "参考光路有什么作用" in query
    assert "回答关系：purpose" in query


def test_rank_guard_limits_catastrophic_downward_move() -> None:
    reranker = object.__new__(LocalBgeReranker)
    reranker.max_rank_drop = 3
    candidates = []
    for rank, score in ((1, 0.99), (2, 0.98), (3, 0.97), (4, 0.96), (5, 0.95)):
        item = _candidate(f"正文 {rank}", str(rank))
        item.fused_rank = rank
        item.rerank_score_combined = score
        candidates.append(item)
    # 模拟融合第 1 名被重排模型放到列表末尾。
    candidates[0], candidates[-1] = candidates[-1], candidates[0]
    guarded = reranker._apply_rank_guard(candidates)
    assert guarded.index(candidates[0]) <= 3
    assert all(
        position <= int(item.fused_rank) + reranker.max_rank_drop
        for position, item in enumerate(guarded, 1)
    )
