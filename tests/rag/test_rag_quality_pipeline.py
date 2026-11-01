from __future__ import annotations

import json

import pytest

from local_full_text_search.rag.answer_review import AnswerReviewParser
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.clarification import ClarificationPolicy
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.context_builder import ContextBuilder
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
from local_full_text_search.rag.trace import candidate_snapshot


def _candidate(
    chunk_id: str,
    text: str,
    *,
    document_key: str = "doc-1",
    text_hash: str | None = None,
    rerank: float = 0.8,
    lexical_rank: int | None = 1,
    vector_score: float | None = 0.7,
    extension: str = ".pdf",
) -> EvidenceChunk:
    source = SourceRef(
        source_generation=1,
        stable_chunk_id=chunk_id,
        document_id=1,
        file_id=1,
        root_id=1,
        file_path=f"C:/docs/{document_key}{extension}",
        filename=f"{document_key}{extension}",
        extension=extension,
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
    # 未识别到固定阶段本身不应强制用户补充；召回仍保留多个阶段证据。
    assert plan.missing_fields == ()


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


def test_candidate_deduplicator_groups_layout_equivalent_cross_file_text() -> None:
    first = _candidate("a", "定标失败时检查底物、反应杯和管路是否堵塞")
    near = _candidate("b", "定标失败时，检查底物、反应杯和管路是否堵塞。")
    other_doc = _candidate("c", "定标失败时，检查底物、反应杯和管路是否堵塞。", document_key="doc-2")

    result = CandidateDeduplicator(jaccard_threshold=0.92).deduplicate(
        [first, near, other_doc], limit=10
    )

    assert [item.chunk.stable_chunk_id for item in result.candidates] == ["a"]
    assert result.duplicates[0].removed_chunk_id == "b"
    assert result.duplicates[0].kept_chunk_id == "a"
    assert result.duplicates[0].method == "same_document_3gram_jaccard"
    assert result.candidates[0].all_sources == (first.chunk.source, other_doc.chunk.source)
    assert [item.status for item in result.dispositions] == [
        "kept", "duplicate_same_document_near_text", "grouped_layout_equivalent_text",
    ]


def test_cross_format_layout_can_merge_but_changed_fact_cannot() -> None:
    first = _candidate("pdf", "高液位传感器触发后，停止吸液；流量为 12 mL/min。", document_key="pdf")
    formatted = _candidate("ppt", "• 高液位传感器触发后\n停止吸液\n流量为 12 mL/min", document_key="ppt", extension=".pptx")
    changed = _candidate("xlsx", "高液位传感器触发后，开始吸液；流量为 12 mL/min。", document_key="xlsx", extension=".xlsx")
    changed_number = _candidate("pdf-2", "高液位传感器触发后，停止吸液；流量为 13 mL/min。", document_key="pdf-2")

    result = CandidateDeduplicator().deduplicate([first, formatted, changed, changed_number], limit=10)

    assert [item.chunk.stable_chunk_id for item in result.candidates] == ["pdf", "xlsx", "pdf-2"]
    assert result.dispositions[1].status == "grouped_layout_equivalent_text"
    assert result.candidates[0].all_member_chunk_ids == ("pdf", "ppt")


def test_cross_format_identity_keeps_factual_operators_and_units() -> None:
    from local_full_text_search.rag.content_identity import canonical_content

    assert canonical_content("• 流量 ≥ 12 %") != canonical_content("流量 < 12 %")
    assert canonical_content("流量 12 %") != canonical_content("流量 12")
    assert canonical_content("版本 V1.1") != canonical_content("版本 V11")
    assert canonical_content("A-B") != canonical_content("AB")


def test_explicit_device_requires_applicable_evidence_before_context() -> None:
    plan = QueryInterpreter().interpret("超声影像诊断仪的探头发射功率合格范围是多少？")
    unrelated = _candidate("other", "CL-8000i 超声频率异常处理方法，检查探头。", document_key="immuno")
    applicable = _candidate("device", "超声影像诊断仪探头发射功率合格范围为 10 至 20。", document_key="ultrasound")
    checked = ConstraintMatcher().evaluate(plan, [unrelated, applicable])

    decision = EvidencePolicy().evaluate(RagQuery(text=plan.original_text, final_top_k=8), checked, plan=plan)

    assert [item.chunk.stable_chunk_id for item in decision.accepted] == ["device", "other"]
    assert checked[0].constraint_decision.support_status == "supported"
    assert checked[1].constraint_decision.support_status == "unknown"
    context = ContextBuilder().build(decision.accepted)
    assert "设备适用性：正文与来源已共同确认" in context.text
    assert "设备适用性：未确认，仅作背景" in context.text

    only_unknown = EvidencePolicy().evaluate(
        RagQuery(text=plan.original_text), [unrelated], plan=plan,
    )
    assert only_unknown.insufficient
    assert only_unknown.reason == "required_constraint_not_confirmed:device=超声影像诊断仪"


def test_exact_text_group_collects_sources_after_limit_without_score_boost() -> None:
    first = _candidate("a", "检查点可以恢复任务。", text_hash="same")
    first.fused_score = 0.05
    cutoff = _candidate("c", "完全不同的新正文。", text_hash="other")
    same = _candidate("b", "检查点可以恢复任务。", document_key="doc-2", text_hash="same")
    same.fused_score = 0.99
    repeated = _candidate("a", "检查点可以恢复任务。", text_hash="same")
    result = CandidateDeduplicator().deduplicate([first, cutoff, same, repeated], limit=1)

    assert result.candidates == (first,)
    assert first.fused_score == 0.05
    assert first.all_member_chunk_ids == ("a", "b")
    assert first.all_sources == (first.chunk.source, same.chunk.source)
    assert [item.status for item in result.dispositions] == [
        "kept", "fusion_top_k_cutoff", "grouped_exact_text_hash",
        "duplicate_stable_chunk_id",
    ]
    assert [item.input_rank for item in result.dispositions] == [1, 2, 3, 4]
    assert result.dispositions[2].output_rank == 1
    assert result.dispositions[2].source_merged is True
    assert result.cutoffs[0].chunk_id == "c"
    assert "fusion_top_k=1" in result.cutoffs[0].reason
    assert "c" not in {item.removed_chunk_id for item in result.duplicates}
    assert len(result.exact_groups) == len(result.near_duplicates) + 1
    snapshot = candidate_snapshot(first, dedup_status="kept")
    assert snapshot["source_count"] == 2
    assert snapshot["member_chunk_ids"] == ["a", "b"]


def test_exact_text_group_uses_one_evidence_slot_and_one_context_id() -> None:
    first = _candidate("a", "检查点可以恢复任务。", text_hash="same")
    second = _candidate("b", first.chunk.text, document_key="doc-2", text_hash="same")
    result = CandidateDeduplicator().deduplicate([first, second], limit=1)
    context = ContextBuilder().build(result.candidates)

    assert len(result.candidates) == 1
    assert context.text.count(first.chunk.text) == 1
    assert context.text.count("[S1]") == 1
    assert len(context.source_map["S1"]) == 2
    assert context.sources == (first.chunk.source, second.chunk.source)
    assert CitationValidator().validate("按检查点恢复。[S1]", context.source_map) == ("S1",)


def test_context_defensively_merges_same_hash_sources() -> None:
    first = _candidate("a", "检查点可以恢复任务。", text_hash="same")
    second = _candidate("b", first.chunk.text, document_key="doc-2", text_hash="same")
    context = ContextBuilder().build([first, second])

    assert len(context.source_map) == 1
    assert len(context.source_map["S1"]) == 2
    assert context.entries[0]["defensive_merge"] is True
    assert "防御性合并" in context.entries[0]["reason"]


def test_repeated_chunk_id_after_cutoff_is_recorded_once_per_input() -> None:
    kept = _candidate("a", "正文甲", text_hash="hash-a")
    cutoff = _candidate("b", "正文乙", text_hash="hash-b")
    repeat = _candidate("b", "正文乙", text_hash="hash-b")
    result = CandidateDeduplicator().deduplicate([kept, cutoff, repeat], limit=1)

    assert len(result.dispositions) == 3
    assert [item.status for item in result.dispositions] == [
        "kept", "fusion_top_k_cutoff", "duplicate_stable_chunk_id",
    ]
    assert result.dispositions[-1].output_rank is None
    assert len(result.duplicates) == 0


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


def test_clarification_policy_does_not_force_stage_for_general_question() -> None:
    plan = QueryInterpreter().interpret("发光值异常怎么排查")
    candidates = [
        _candidate("cal", "定标时发光值异常，检查定标液。"),
        _candidate("qc", "质控时发光值异常，检查质控品。"),
    ]

    decision = ClarificationPolicy().evaluate_competing_stages(plan, candidates)

    assert decision.required is False


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


def test_answer_review_insufficient_normalizes_freeform_refusal() -> None:
    result = AnswerReviewParser().parse(
        _review_payload(verdict="insufficient", final_answer="目前无法确认，请查看其他资料。",
                        claims=[]),
        allowed_source_ids={"S1"},
    ).result
    assert result.final_answer == "资料中没有足够依据。"


def test_answer_review_missing_final_citation_fails_closed() -> None:
    result = AnswerReviewParser().parse(
        _review_payload(verdict="revised", final_answer="检查该项目的参数。"),
        allowed_source_ids={"S1"},
    ).result

    assert result.verdict == "insufficient"
    assert result.final_answer == "资料中没有足够依据。"
    assert "没有引用来源" in result.summary


def test_answer_review_unassociated_final_citation_fails_closed() -> None:
    result = AnswerReviewParser().parse(
        _review_payload(verdict="revised", final_answer="检查该项目的参数。[S2]"),
        allowed_source_ids={"S1", "S2"},
    ).result

    assert result.verdict == "insufficient"
    assert result.final_answer == "资料中没有足够依据。"
    assert "未关联到 supported" in result.summary


def test_answer_review_accepts_one_valid_json_with_preamble() -> None:
    parsed = AnswerReviewParser().parse(
        "审核结果：\n```json\n" + _review_payload() + "\n```\n结束。",
        allowed_source_ids={"S1"},
    )
    assert parsed.result.verdict == "pass"


@pytest.mark.parametrize(
    "raw_text",
    [
        _review_payload() + "\n" + _review_payload(),
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
