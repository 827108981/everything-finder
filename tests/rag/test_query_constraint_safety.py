"""P0：解析错误不能升级为误删正确证据。"""

from __future__ import annotations

import pytest

from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import (
    ConstraintKind, ConstraintPolarity, ConstraintStrength, EvidenceChunk,
    QueryConstraint, RagChunk, RagQuery, SourceRef,
)
from local_full_text_search.rag.query_analyzer import QueryAnalyzer
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.trace import candidate_snapshot


def _candidate(chunk_id: str, text: str, score: float = 0.8) -> EvidenceChunk:
    source = SourceRef(
        source_generation=1, stable_chunk_id=chunk_id, document_id=1,
        file_id=1, root_id=1, file_path=f"C:/资料/{chunk_id}.pdf",
        filename=f"{chunk_id}.pdf", extension=".pdf", block_ids=(1,),
        location_text="第 1 页",
    )
    return EvidenceChunk(
        chunk=RagChunk(chunk_id, chunk_id, 1, text, text, chunk_id, source),
        lexical_rank=1, vector_score=0.8, rerank_score=score,
    )


def _of(plan, kind: ConstraintKind, value: str):
    return next(item for item in plan.constraints if item.kind is kind and item.value == value)


def test_equal_condition_decisions_keep_rerank_order() -> None:
    plan = QueryInterpreter().interpret("BS-2800M 生化分析仪的试剂盘采用什么制冷方式？")
    first = _candidate("first", "BS-2800M 生化分析仪试剂盘采用压缩机制冷。", score=0.7)
    second = _candidate("second", "BS-2800M 生化分析仪试剂盘采用压缩机制冷。", score=0.9)
    first.rerank_score_combined = 0.8
    second.rerank_score_combined = 0.7

    checked = ConstraintMatcher().evaluate(plan, [first, second])

    assert [item.chunk.stable_chunk_id for item in checked] == ["first", "second"]
    assert checked[0].constraint_score == checked[1].constraint_score


@pytest.mark.parametrize(
    ("question", "kind", "value", "polarity", "strength"),
    [
        ("发光值偏低如何处理", ConstraintKind.DIRECTION, "low", "affirmed", "hard"),
        ("发光值不是偏低，而是偏高", ConstraintKind.DIRECTION, "high", "affirmed", "hard"),
        ("发光值并非偏高", ConstraintKind.DIRECTION, "high", "negated", "ambiguous"),
        ("项目A偏低，项目B偏高", ConstraintKind.DIRECTION, "low", "affirmed", "ambiguous"),
        ("发光值好像有点偏高", ConstraintKind.DIRECTION, "high", "affirmed", "soft"),
        ("刚才说错了，不是偏高，是偏低", ConstraintKind.DIRECTION, "low", "affirmed", "hard"),
        ("手册写“偏低”，但实际发光值偏高", ConstraintKind.DIRECTION, "low", "unknown", "ambiguous"),
        ("并非所有项目都异常", ConstraintKind.SCOPE, "all", "negated", "ambiguous"),
        ("只有一个项目发光值异常", ConstraintKind.SCOPE, "single", "affirmed", "hard"),
        ("发光值异常怎么办", ConstraintKind.DIRECTION, "abnormal", "affirmed", "hard"),
    ],
)
def test_p0_acceptance_cases_have_safe_polarity_and_strength(
    question, kind, value, polarity, strength,
) -> None:
    plan = QueryInterpreter().interpret(question)
    condition = _of(plan, kind, value)
    assert (condition.polarity.value, condition.strength.value) == (polarity, strength)
    assert condition.confidence is None
    assert condition.source_text and condition.reason
    assert plan.original_text == plan.semantic_query == question
    if question == "发光值异常怎么办":
        assert plan.missing_fields == ()
    for category in (ConstraintKind.SCOPE, ConstraintKind.STAGE,
                     ConstraintKind.DIRECTION, ConstraintKind.MODEL):
        hard_values = {item.value for item in plan.constraints
                       if item.kind is category and item.is_hard}
        assert len(hard_values) <= 1


def test_negation_and_correction_keep_only_positive_direction_in_fts() -> None:
    analysis = QueryAnalyzer().analyze("发光值不是偏低，而是偏高")
    assert "发光值偏高" in analysis.terms
    assert "偏低" not in analysis.fts_expression
    assert "下降" not in analysis.fts_expression
    assert "偏低" not in analysis.fallback_terms
    assert _of(QueryInterpreter().interpret("发光值不是偏低，而是偏高"),
               ConstraintKind.DIRECTION, "low").polarity is ConstraintPolarity.NEGATED


def test_interpreter_preserves_raw_question_including_surrounding_space() -> None:
    original = "  发光值不是偏低，而是偏高？\n"
    plan = QueryInterpreter().interpret(original)
    assert plan.original_text == original
    assert plan.semantic_query == original
    assert "偏低" not in plan.lexical_expression


@pytest.mark.parametrize("question", [
    "发光值并不是偏高", "发光值不属于偏高", "发光值不是偏高，是偏低",
])
def test_adjacent_negation_never_becomes_a_hard_positive(question) -> None:
    plan = QueryInterpreter().interpret(question)
    high = _of(plan, ConstraintKind.DIRECTION, "high")
    assert high.polarity is ConstraintPolarity.NEGATED
    assert not high.is_hard


def test_interpreter_downgrades_conflicting_optional_parse() -> None:
    class OptionalParser:
        def parse(self, text, token):
            return {"constraints": [QueryConstraint(
                ConstraintKind.DIRECTION, "high", source_text="模型推断偏高",
            )]}

    plan = QueryInterpreter(llm_parser=OptionalParser()).interpret("发光值偏低")
    assert not any(item.is_hard for item in plan.constraints
                   if item.kind is ConstraintKind.DIRECTION)
    assert all(item.strength is ConstraintStrength.AMBIGUOUS
               for item in plan.constraints if item.kind is ConstraintKind.DIRECTION)
    assert plan.semantic_query == "发光值偏低"


def test_matcher_keeps_unknown_soft_conflict_and_ambiguous_candidates() -> None:
    matcher = ConstraintMatcher()
    hard_plan = QueryInterpreter().interpret("发光值偏高")
    unknown = matcher.evaluate(hard_plan, [_candidate("unknown", "发光值状态待确认。")])[0]
    assert unknown.constraint_decision.accepted
    assert unknown.constraint_decision.required_unknown_count >= 1
    uncovered = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=hard_plan.original_text), [unknown], plan=hard_plan,
    )
    assert uncovered.insufficient
    assert uncovered.reason.startswith("required_constraint_not_confirmed:")
    assert "关键硬条件未获证据确认" in uncovered.rejected[0][1]

    soft_plan = QueryInterpreter().interpret("发光值好像有点偏高")
    low = _candidate("low", "发光值偏低，应检查底物。", score=0.99)
    high = _candidate("high", "发光值偏高，应检查校准曲线。", score=0.6)
    ranked = matcher.evaluate(soft_plan, [low, high])
    assert [item.chunk.stable_chunk_id for item in ranked] == ["high", "low"]
    assert low.constraint_decision.accepted
    assert low.constraint_decision.soft_conflict_count == 1
    assert low.constraint_decision.required_conflict_count == 0

    ambiguous = QueryInterpreter().interpret("项目A偏低，项目B偏高")
    ambiguous_candidates = matcher.evaluate(
        ambiguous, [_candidate("other", "项目A发光值偏低。")]
    )
    for item in ambiguous_candidates:
        assert item.constraint_decision.accepted
    decision = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=ambiguous.original_text), ambiguous_candidates, plan=ambiguous,
    )
    assert [item.chunk.stable_chunk_id for item in decision.accepted] == ["other"]


def test_ac02_query_plan_to_evidence_policy_preserves_high_and_rejects_low() -> None:
    plan = QueryInterpreter().interpret("发光值不是偏低，而是偏高")
    high = _candidate("high", "发光值偏高，应检查校准曲线。", score=0.7)
    low = _candidate("low", "发光值偏低，应检查底物。", score=0.99)
    ranked = ConstraintMatcher().evaluate(plan, [low, high])
    result = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text, final_top_k=2), ranked, plan=plan,
    )
    assert [item.chunk.stable_chunk_id for item in result.accepted] == ["high"]
    assert result.rejected[0][0] == "low"
    assert "偏高" in result.rejected[0][1] and "偏低" in result.rejected[0][1]
    snapshot = candidate_snapshot(low)
    assert snapshot["constraint_decision"]["hard_rejected"] is True
    assert snapshot["constraint_decision"]["judgements"][0]["polarity"] == "negated"
    plan_snapshot = AnswerService._plan_snapshot(plan)
    assert plan_snapshot["original_text"] == plan.original_text
    assert plan_snapshot["negated_constraints"][0]["value"] == "low"
    assert plan_snapshot["analyzer_version"] == "query-plan-v3"
    # 即使调用方复用了带旧判断的候选，关闭门禁也不能再次硬拒绝。
    disabled = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text), [low], plan=None,
    )
    assert disabled.accepted == (low,)


def test_evidence_policy_does_not_hard_reject_soft_conflict() -> None:
    plan = QueryInterpreter().interpret("发光值好像有点偏高")
    low = _candidate("low", "发光值偏低，应检查底物。")
    ConstraintMatcher().evaluate(plan, [low])
    result = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text), [low], plan=plan,
    )
    assert not result.insufficient
    assert result.accepted == (low,)
    assert result.rejected == ()


def test_evidence_policy_explains_topk_and_only_hard_coverage() -> None:
    plan = QueryInterpreter().interpret("发光值好像有点偏高")
    first = _candidate("first", "发光值偏低，应检查底物。", score=0.9)
    second = _candidate("second", "发光值偏高，应检查校准曲线。", score=0.8)
    ConstraintMatcher().evaluate(plan, [first, second])
    result = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text, final_top_k=1), [first, second], plan=plan,
    )
    assert not result.insufficient
    assert len(result.accepted) == 1
    assert any("最终 TopK 截断" in reason for _, reason in result.rejected)


def test_same_sentence_with_two_devices_is_unknown_not_confirmed() -> None:
    plan = QueryInterpreter().interpret("生化分析仪的光度计如何校准？")
    mixed = _candidate("mixed", "生化分析仪与免疫分析仪的光度计校准步骤如下。")
    checked = ConstraintMatcher().evaluate(plan, [mixed])[0]
    assert checked.constraint_decision.accepted
    assert checked.constraint_decision.support_status == "unknown"
    result = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text), [checked], plan=plan,
    )
    assert result.insufficient
    assert result.reason.startswith("required_constraint_not_confirmed:device=")


def test_open_question_topic_blocks_cross_domain_transfer_without_domain_dictionary() -> None:
    from dataclasses import replace

    plan = QueryInterpreter().interpret("生化如何对光度计进行校准？")
    assert _of(plan, ConstraintKind.TOPIC, "生化").is_hard

    def titled(chunk_id: str, filename: str) -> EvidenceChunk:
        candidate = _candidate(chunk_id, "光度计校准时先检查参考光路。")
        source = replace(candidate.chunk.source, filename=filename,
                         file_path=f"C:/资料/{filename}")
        candidate.chunk = replace(candidate.chunk, source=source)
        return candidate

    unrelated = titled("immune", "免疫光学.pdf")
    related = titled("bio", "生化光学.pdf")
    ranked = ConstraintMatcher().evaluate(plan, [unrelated, related])
    assert [row.chunk.stable_chunk_id for row in ranked] == ["bio", "immune"]
    assert related.constraint_decision.support_status == "supported"
    assert unrelated.constraint_decision.support_status == "unknown"
    title_only = _candidate("title-only", "其他设备的日常维护说明。")
    title_only.chunk = replace(title_only.chunk, source=replace(
        title_only.chunk.source, filename="生化光学.pdf",
        file_path="C:/资料/生化光学.pdf",
    ))
    assert ConstraintMatcher().evaluate(plan, [title_only])[0].constraint_decision.support_status == "unknown"
    result = EvidencePolicy(min_rerank_score=0.1).evaluate(
        RagQuery(text=plan.original_text), [unrelated], plan=plan,
    )
    assert result.insufficient
    assert result.reason == "required_constraint_not_confirmed:topic=生化"
