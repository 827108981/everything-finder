from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import EvidenceChunk, RagQuery, RagChunk, SourceRef
from local_full_text_search.rag.query_interpreter import QueryInterpreter


def _candidate(chunk_id: str, text: str, rerank_score: float) -> EvidenceChunk:
    source = SourceRef(
        source_generation=1,
        stable_chunk_id=chunk_id,
        document_id=1,
        file_id=1,
        root_id=1,
        file_path=f"C:/资料/{chunk_id}.pdf",
        filename=f"{chunk_id}.pdf",
        extension=".pdf",
        block_ids=(1,),
        location_text="第 1 页",
    )
    chunk = RagChunk(
        stable_chunk_id=chunk_id,
        document_key=chunk_id,
        ordinal=1,
        text=text,
        normalized_text=text,
        text_hash=chunk_id,
        source=source,
    )
    return EvidenceChunk(
        chunk=chunk,
        lexical_rank=1,
        vector_score=0.8,
        fused_score=0.02,
        rerank_score=rerank_score,
    )


def test_single_project_gold_case_does_not_promote_all_project_failure() -> None:
    fixture = Path(__file__).parent / "fixtures" / "gold_cases.jsonl"
    case = json.loads(fixture.read_text(encoding="utf-8").splitlines()[0])
    plan = QueryInterpreter().interpret(case["question"])
    correct = _candidate(
        "correct",
        "单个项目发光值异常时，检查该项目对应参数、试剂和反应杯。",
        rerank_score=0.72,
    )
    misleading = _candidate(
        "misleading",
        "所有项目均定标失败时，应检查整体定标状态和校准液。",
        rerank_score=0.98,
    )
    ranked = ConstraintMatcher().evaluate(plan, [misleading, correct])

    decision = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=case["question"], final_top_k=8), ranked, plan=plan
    )

    ids = {item.chunk.stable_chunk_id for item in decision.accepted}
    violations = sum(
        1
        for item in decision.accepted
        if any(phrase in item.chunk.text for phrase in case["negative_evidence_contains"])
    )
    assert decision.insufficient is False
    assert "correct" in ids
    assert "misleading" not in ids
    assert ("correct" in ids) is case["expected_correct_candidate_survives"]
    assert violations == case["expected_constraint_violations_at_8"]
