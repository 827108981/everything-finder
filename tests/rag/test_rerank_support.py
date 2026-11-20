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
from local_full_text_search.core.task_manager import CancelToken
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
import math
import sys


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


def test_legacy_input_bge_ignores_weights_guard_and_document_quota(monkeypatch):
    class Tensor:
        def __init__(self, values): self.values=values
        def reshape(self, *args): return self
        def cpu(self): return self
        def tolist(self): return self.values
    class Model:
        config=SimpleNamespace(num_labels=1)
        def to(self, device): return self
        def eval(self): pass
        def __call__(self, **encoded):
            return SimpleNamespace(logits=Tensor([float(text) for text in encoded['texts']]))
    class Tokenizer:
        def __call__(self, questions, texts, **kwargs): return {'texts':texts}
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()),
        AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=lambda *a,**k:Model())))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(inference_mode=nullcontext,
        sigmoid=lambda tensor:Tensor([1/(1+math.exp(-x)) for x in tensor.values])))
    ranker=LocalBgeReranker(None,repo_id='fixture',revision='fixed',
        cross_encoder_weight=0,retrieval_prior_weight=10,answer_support_weight=100,max_rank_drop=1,input_mode='truncate')
    candidates=[_candidate(str(n),str(n)) for n in range(6)]
    for i,c in enumerate(candidates):
        c.chunk=replace(c.chunk,document_key='same-document')
        # All six have the same document_key; old per-file quota would lose three.
        assert c.chunk.document_key==candidates[0].chunk.document_key
        c.fused_rank=i+1
        c.answer_support_score=1-i/6
    result=ranker.rerank('普通问题',candidates,6,CancelToken())
    assert [c.chunk.text for c in result]==['5','4','3','2','1','0']
    assert tuple(result)==ranker.last_scored_candidates
    assert all(c.rerank_score_combined==c.rerank_score for c in result)
    assert all(c.rerank_diagnostics['strategy']=='bge_base' for c in result)
    assert ranker.cross_encoder_weight==1 and ranker.max_rank_drop==0


def test_bge_ties_use_fusion_score_then_stable_id_only():
    ranker=object.__new__(LocalBgeReranker)
    candidates=[_candidate('相同正文',key) for key in ('z','a','b')]
    for c in candidates:
        c.rerank_score=.5
        c.fused_score=0
        c.answer_support_score=1
    candidates[0].fused_score=.1
    expected=sorted(candidates,key=lambda c:(-c.fused_score,c.chunk.stable_chunk_id))
    assert ranker._order_scored('问题',candidates,CancelToken())==expected


