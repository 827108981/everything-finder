from __future__ import annotations

from collections import Counter
from typing import Any

from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch
from local_full_text_search.rag.evaluation.gold_loader import GoldCase
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


STAGE_ORDER = ("lexical", "vector", "fusion", "deduplicated", "rerank", "constraints", "evidence", "context")
STAGE_LABELS = {
    "lexical": "关键词召回",
    "vector": "向量召回",
    "fusion": "融合排序",
    "deduplicated": "去重",
    "rerank": "重排",
    "constraints": "业务条件检查",
    "evidence": "最终证据筛选",
    "context": "送入生成上下文",
}

METRIC_DEFINITIONS = {
    "gold_hit_rate": "Gold 命中率：非拒答且声明 Gold 的计分题中，该阶段完整找到至少一条正确证据的比例。",
    "mean_evidence_recall": "平均证据覆盖率：本题完整找到的 Gold 条数 ÷ 本题声明 Gold 条数，再对有效题取平均；索引未定位的 Gold 仍在分母。",
    "mean_reciprocal_rank": "MRR：每题首条完整 Gold 的完成名次取倒数；未命中贡献 0，再对有效题取平均。",
    "mean_first_gold_rank": "首位正确证据平均排名：每题第一条 Gold 证据的名次取平均；越小越好。",
    "hard_negative_candidate_count": "指定反例候选数：该阶段出现的已标注反例总次数，不是准确率；越多表示误召回风险越高。",
}


def score_case(
    case: GoldCase,
    stages: dict[str, list[dict[str, Any]]],
    mappings: list[GoldEvidenceMatch],
    *,
    answer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    positives = [item for item in mappings if item.kind == "positive" and item.chunk_ids]
    negatives = [item for item in mappings if item.kind == "hard_negative" and item.chunk_ids]
    stage_results: dict[str, Any] = {}
    first_hit: dict[str, str] = {}
    for stage_name in STAGE_ORDER:
        stage_results[stage_name] = score_stage(stage_name, stages.get(stage_name, []), mappings)
        for evidence_id in stage_results[stage_name]["gold_completion_ranks"]:
            first_hit.setdefault(evidence_id, stage_name)
    behavior = _classify_actual_behavior(answer)
    if answer and answer.get("mode") == "full" and case.expected_behavior == "partial_answer":
        behavior = "answer_needs_fact_review"
    text = str((answer or {}).get("text") or "")
    forbidden_found = [claim for claim in case.forbidden_claims if claim and _contains_normalized(text, claim)]
    lexical_fact_hints = [fact for fact in case.required_facts if _contains_normalized(text, fact)]
    return {
        "id": case.case_id,
        "group_id": case.group_id,
        "question": case.question,
        "expected_behavior": case.expected_behavior,
        "actual_behavior": behavior,
        "behavior_match": (
            None
            if answer and (answer.get("mode") == "retrieval-only" or case.expected_behavior == "partial_answer")
            else behavior == case.expected_behavior
        ),
        "stages": stage_results,
        "first_gold_hit": first_hit,
        "gold_mappings": [
            {
                "evidence_id": item.evidence_id,
                "kind": item.kind,
                "file": item.file,
                "page": item.page,
                "chunk_ids": list(item.chunk_ids),
                "declared_chunk_ids": list(item.declared_chunk_ids),
                "equivalent_chunk_ids": list(item.equivalent_chunk_ids),
                "suspected_chunk_ids": list(item.suspected_chunk_ids),
                "conflict_chunk_ids": list(item.conflict_chunk_ids),
                "alternatives": [
                    {"alternative_id": alt.alternative_id, "match_type": alt.match_type,
                     "confirmed": alt.confirmed,
                     "parts": [
                         {"chunk_id": part.chunk_id, "quote_start": part.quote_start,
                          "quote_end": part.quote_end, "source_start": part.source_start,
                          "source_end": part.source_end,
                          "quote_original_start": part.quote_original_start,
                          "quote_original_end": part.quote_original_end,
                          "source_original_start": part.source_original_start,
                          "source_original_end": part.source_original_end} for part in alt.parts
                     ]} for alt in item.alternatives
                ],
                "confidence": item.confidence,
                "reason": item.reason,
            }
            for item in mappings
        ],
        "index_coverage": {
            "positive_total": len([item for item in mappings if item.kind == "positive"]),
            "positive_located": len(positives),
            "negative_total": len([item for item in mappings if item.kind == "hard_negative"]),
            "negative_located": len(negatives),
            "needs_review": [item.evidence_id for item in mappings if item.confidence == "review"],
        },
        "required_fact_text_hints": {
            "matched": lexical_fact_hints,
            "unmatched": [fact for fact in case.required_facts if fact not in lexical_fact_hints],
            "interpretation": "仅为字面命中提示，不作为语义正确性判分",
        },
        "forbidden_claims_found": forbidden_found,
        "forbidden_claims_match_type": "normalized_literal",
        "answer": answer or {},
    }


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [item for item in results if item.get("error") is None]
    behavior_rows = [item for item in complete if item.get("behavior_match") is not None]
    behaviors = Counter((item.get("expected_behavior"), item.get("actual_behavior")) for item in behavior_rows)
    stage_summary: dict[str, Any] = {}
    for stage in STAGE_ORDER:
        rows = [item.get("stages", {}).get(stage, {}) for item in complete]
        with_gold = [
            row for result, row in zip(complete, rows)
            if result.get("expected_behavior") != "refuse"
            and row.get("gold_evidence_total_declared", row.get("gold_evidence_total_located", 0)) > 0
        ]
        stage_summary[stage] = {
            "cases_with_gold": len(with_gold),
            "gold_hit_cases": sum(bool(row.get("gold_hit")) for row in with_gold),
            "gold_hit_rate": (
                sum(bool(row.get("gold_hit")) for row in with_gold) / len(with_gold)
                if with_gold else None
            ),
            "mean_reciprocal_rank": (
                sum(float(row.get("reciprocal_rank") or 0.0) for row in with_gold) / len(with_gold)
                if with_gold else None
            ),
            "mean_first_gold_rank": (
                sum(float(row["first_gold_rank"]) for row in with_gold if row.get("first_gold_rank") is not None)
                / len([row for row in with_gold if row.get("first_gold_rank") is not None])
                if any(row.get("first_gold_rank") is not None for row in with_gold) else None
            ),
            "mean_evidence_recall": (
                sum(float(row["gold_evidence_recall"]) for row in with_gold if row.get("gold_evidence_recall") is not None)
                / len([row for row in with_gold if row.get("gold_evidence_recall") is not None])
                if any(row.get("gold_evidence_recall") is not None for row in with_gold) else None
            ),
            "hard_negative_candidate_count": sum(int(row.get("hard_negative_found", 0)) for row in rows),
        }
    positives = [item for item in complete if item.get("expected_behavior") != "refuse"]
    return {
        "case_count": len(results),
        "completed_case_count": len(complete),
        "behavior_confusion": [
            {"expected": expected, "actual": actual, "count": count}
            for (expected, actual), count in sorted(behaviors.items())
        ],
        "behavior_accuracy": (
            sum(bool(item.get("behavior_match")) for item in behavior_rows) / len(behavior_rows)
            if behavior_rows else None
        ),
        "answer_cases": len(positives),
        "refusal_cases": sum(item.get("expected_behavior") == "refuse" for item in complete),
        "forbidden_claim_case_count": sum(bool(item.get("forbidden_claims_found")) for item in complete),
        "api_call_count": sum(int(item.get("api_call_count") or 0) for item in results),
        "stages": stage_summary,
        "metric_definitions": dict(METRIC_DEFINITIONS),
    }


def build_rank_history(
    stages: dict[str, list[dict[str, Any]]],
) -> dict[str, list[tuple[str, int]]]:
    """按 chunk ID 汇总候选在各阶段的实际名次。"""

    history: dict[str, list[tuple[str, int]]] = {}
    for stage in STAGE_ORDER:
        candidates = stages.get(stage, [])
        if stage == "evidence" and any("selected" in item for item in candidates):
            candidates = [item for item in candidates if item.get("selected") is True]
            candidates = sorted(
                candidates,
                key=lambda item: int(item.get("final_rank") or 0),
            )
        for position, candidate in enumerate(candidates, 1):
            chunk_id = str(candidate.get("chunk_id") or "")
            if not chunk_id:
                continue
            rank = position
            if stage == "evidence" and candidate.get("final_rank") is not None:
                rank = int(candidate["final_rank"])
            history.setdefault(chunk_id, []).append((stage, rank))
    return history


def _classify_actual_behavior(answer: dict[str, Any] | None) -> str:
    if not answer:
        return "not_run"
    if answer.get("mode") == "retrieval-only":
        return "not_run"
    status = str(answer.get("status") or "")
    text = str(answer.get("text") or "").strip()
    if status == "needs_clarification":
        return "clarify"
    refusal_phrases = (
        "资料中没有足够依据", "无法从当前资料", "当前资料没有提供", "资料未提供",
        "无法根据提供的资料", "无法从所提供资料",
    )
    if status == "insufficient" or any(phrase in text for phrase in refusal_phrases):
        return "refuse"
    if not text:
        return "empty"
    # 核心 RagAnswerStatus 没有“部分回答”；最终覆盖率交给语义判分/人工确认。
    return "answer"


def _contains_normalized(answer: str, fact: str) -> bool:
    from local_full_text_search.rag.evaluation.evidence_matcher import normalize_evidence_text

    normalized_fact = normalize_evidence_text(fact)
    return bool(normalized_fact and normalized_fact in normalize_evidence_text(answer))
