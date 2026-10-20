from __future__ import annotations

from collections import Counter
from typing import Any

from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch
from local_full_text_search.rag.evaluation.gold_loader import GoldCase


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
    "gold_hit_rate": "Gold 命中率：该阶段至少找到一条已定位的正确证据的题目比例。",
    "mean_evidence_recall": "平均证据召回率：找回的已定位正确证据数 ÷ 本题已定位正确证据数，再对题目取平均；受 Gold 映射覆盖率限制。",
    "mean_reciprocal_rank": "MRR：每题取首条正确证据排名的倒数 1/rank，再对有已定位正确证据的题目取平均；越靠前越高。",
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
        candidates = stages.get(stage_name, [])
        # “证据筛选”阶段的输入列表也包含被拒候选；只把真正通过门禁的证据
        # 计入该阶段的 Recall/MRR，拒绝项仍保留在候选明细供排查。
        if stage_name == "evidence" and any("selected" in item for item in candidates):
            candidates = [item for item in candidates if item.get("selected") is True]
        ids = [str(item.get("chunk_id") or "") for item in candidates]
        id_set = set(ids)
        hit_evidence = [item for item in positives if id_set.intersection(item.chunk_ids)]
        gold_ranks = [
            index for index, chunk_id in enumerate(ids, 1)
            if any(chunk_id in evidence.chunk_ids for evidence in positives)
        ]
        for item in hit_evidence:
            first_hit.setdefault(item.evidence_id, stage_name)
        negative_hits = [item for item in negatives if id_set.intersection(item.chunk_ids)]
        stage_results[stage_name] = {
            "candidate_count": len(candidates),
            "gold_evidence_found": len(hit_evidence),
            "gold_evidence_total_located": len(positives),
            "gold_evidence_recall": (len(hit_evidence) / len(positives)) if positives else None,
            "gold_hit": bool(hit_evidence),
            "first_gold_rank": min(gold_ranks) if gold_ranks else None,
            "reciprocal_rank": (1.0 / min(gold_ranks)) if gold_ranks else 0.0 if positives else None,
            "hard_negative_found": len(negative_hits),
            "hard_negative_ids": [item.evidence_id for item in negative_hits],
        }
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
        with_gold = [row for row in rows if row.get("gold_evidence_total_located", 0) > 0]
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
        "api_call_count": sum(int(item.get("api_call_count") or 0) for item in complete),
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
