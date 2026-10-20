from __future__ import annotations

import json

import pytest

from local_full_text_search.rag.answer_review import AnswerReviewParser
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.clarification import ClarificationPolicy
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.errors import RagGenerationError
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import (
    ConstraintKind,
    EvidenceChunk,
    RagChunk,
    RagQuery,
    SourceRef,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter


def _candidate(
    chunk_id: str,
    text: str,
    *,
    document_key: str = "doc-1",
    text_hash: str | None = None,
    rerank: float = 0.8,
    lexical_rank: int | None = 1,
    vector_score: float | None = 0.7,
) -> EvidenceChunk:
    source = SourceRef(
        source_generation=1,
        stable_chunk_id=chunk_id,
        document_id=1,
        file_id=1,
        root_id=1,
        file_path=f"C:/docs/{document_key}.pdf",
        filename=f"{document_key}.pdf",
        extension=".pdf",
        block_ids=(1,),
        location_text="第 1 页",
    )
    chunk = RagChunk(
        stable_chunk_id=chunk_id,
        document_key=document_key,
        ordinal=1,
        text=text,
        normalized_text=text,
        text_hash=text_hash or f"hash-{chunk_id}",
        source=source,
    )
    return EvidenceChunk(
        chunk=chunk,
        lexical_rank=lexical_rank,
        vector_score=vector_score,
        fused_score=0.02,
        rerank_score=rerank,
    )


def test_query_plan_extracts_scope_and_real_terms_without_arbitrary_four_grams() -> None:
    plan = QueryInterpreter().interpret("只有某个项目的发光值异常，这个时候该怎么排查")

    assert plan.intent == "troubleshooting"
    assert any(c.kind is ConstraintKind.SCOPE and c.value == "single" for c in plan.constraints)
    assert any(c.kind is ConstraintKind.OBJECT and c.value == "发光值" for c in plan.constraints)
    assert "发光值" in plan.lexical_terms
    assert "发光值异常" in plan.lexical_terms
    assert "项目的发" not in plan.lexical_terms
    assert plan.semantic_query == plan.original_text
    assert plan.missing_fields == ("stage",)


@pytest.mark.parametrize(
    ("candidate_text", "expected"),
    [
        ("只有一个项目的发光值异常，检查对应参数。", "match"),
        ("几乎所有项目均出现定标失败。", "conflict"),
        ("检查异常项目对应的试剂和反应杯。", "unknown"),
        ("并非所有项目都异常，需要单独检查。", "unknown"),
    ],
)
def test_constraint_matcher_judges_single_project_scope(candidate_text, expected) -> None:
    plan = QueryInterpreter().interpret("只有某个项目的发光值异常怎么排查")
    candidate = _candidate("chunk-1", candidate_text)

    result = ConstraintMatcher().evaluate(plan, [candidate])[0]
    scope = next(item for item in result.constraint_decision.judgements if item.constraint.kind is ConstraintKind.SCOPE)

    assert scope.status.value == expected
    assert result.constraint_decision.required_conflict_count == (1 if expected == "conflict" else 0)


def test_constraint_matcher_treats_low_as_abnormal_and_high_as_low_conflict() -> None:
    analyzer = QueryInterpreter()
    abnormal_plan = analyzer.interpret("发光值异常如何处理")
    low_candidate = _candidate("low", "发光值偏低，应检查底物和反应杯。")
    ConstraintMatcher().evaluate(abnormal_plan, [low_candidate])
    direction = next(
        item for item in low_candidate.constraint_decision.judgements
        if item.constraint.kind is ConstraintKind.DIRECTION
    )
    assert direction.status.value == "match"

    low_plan = analyzer.interpret("发光值偏低如何处理")
    high_candidate = _candidate("high", "发光值偏高，应检查校准曲线。")
    ConstraintMatcher().evaluate(low_plan, [high_candidate])
    direction = next(
        item for item in high_candidate.constraint_decision.judgements
        if item.constraint.kind is ConstraintKind.DIRECTION
    )
    assert direction.status.value == "conflict"


def test_candidate_deduplicator_removes_same_document_near_duplicates_but_keeps_other_docs() -> None:
    first = _candidate("a", "定标失败时检查底物、反应杯和管路是否堵塞")
    near = _candidate("b", "定标失败时，检查底物、反应杯和管路是否堵塞。")
    other_doc = _candidate("c", first.chunk.text, document_key="doc-2")

    result = CandidateDeduplicator(jaccard_threshold=0.92).deduplicate(
        [first, near, other_doc], limit=10
    )

    assert [item.chunk.stable_chunk_id for item in result.candidates] == ["a", "c"]
    assert result.duplicates[0].removed_chunk_id == "b"
    assert result.duplicates[0].kept_chunk_id == "a"
    assert result.duplicates[0].method == "same_document_3gram_jaccard"


def test_evidence_policy_rejects_high_score_conflict_and_promotes_matching_candidate() -> None:
    plan = QueryInterpreter().interpret("只有某个项目的发光值异常怎么排查")
    conflict = _candidate("all", "所有项目均定标失败，检查总校准状态。", rerank=0.99)
    match = _candidate("single", "只有一个项目发光值异常，检查该项目参数。", rerank=0.72)
    candidates = ConstraintMatcher().evaluate(plan, [conflict, match])

    decision = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text, final_top_k=1), candidates, plan=plan
    )

    assert decision.insufficient is False
    assert [item.chunk.stable_chunk_id for item in decision.accepted] == ["single"]
    assert decision.rejected[0][0] == "all"
    assert "冲突" in decision.rejected[0][1]


def test_clarification_policy_requests_when_candidate_stages_compete() -> None:
    plan = QueryInterpreter().interpret("发光值异常怎么排查")
    candidates = [
        _candidate("cal", "定标时发光值异常，检查定标液。"),
        _candidate("qc", "质控时发光值异常，检查质控品。"),
    ]

    decision = ClarificationPolicy().evaluate_competing_stages(plan, candidates)

    assert decision.required is True
    assert "定标、质控" in decision.question


def _review_payload(**updates):
    payload = {
        "verdict": "pass",
        "summary": "结论均有资料支持",
        "claims": [
            {
                "claim": "检查该项目的参数。",
                "citations": ["S1"],
                "status": "supported",
                "reason": "证据明确写出该步骤",
            }
        ],
        "final_answer": "检查该项目的参数。[S1]",
    }
    payload.update(updates)
    return json.dumps(payload, ensure_ascii=False)


def test_answer_review_parser_accepts_valid_revised_answer_and_citation_allowlist() -> None:
    result = AnswerReviewParser().parse(
        _review_payload(verdict="revised"),
        allowed_source_ids={"S1"},
        provider_name="test",
        model_name="test-model",
    ).result

    assert result.verdict == "revised"
    assert result.final_answer.endswith("[S1]")
    assert result.claims[0].status == "supported"


@pytest.mark.parametrize(
    "raw_text",
    [
        "说明如下：" + _review_payload(),
        _review_payload(final_answer="凭经验检查。 [S9]"),
        _review_payload(verdict="pass", claims=[{
            "claim": "无依据结论",
            "citations": [],
            "status": "not_supported",
            "reason": "证据未提及",
        }]),
    ],
)
def test_answer_review_parser_rejects_malformed_or_unsafe_result(raw_text) -> None:
    with pytest.raises(RagGenerationError):
        AnswerReviewParser().parse(raw_text, allowed_source_ids={"S1"})


def test_answer_review_parser_rejects_unsupported_claim_left_in_final_answer() -> None:
    raw = _review_payload(
        verdict="revised",
        claims=[{
            "claim": "检查该项目的参数。",
            "citations": [],
            "status": "not_supported",
            "reason": "证据没有提及",
        }],
    )
    with pytest.raises(RagGenerationError, match="未获支持"):
        AnswerReviewParser().parse(raw, allowed_source_ids={"S1"})
