"""保存每条候选在实际记录阶段之间的去向和可解释的退出原因。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CandidateTransition:
    chunk_id: str
    from_stage: str
    to_stage: str
    status: str
    reason_code: str
    reason: str
    rank_before: int | None
    rank_after: int | None
    representative_chunk_id: str | None = None


def _ranked(stage: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if stage == "evidence":
        return [row for row in rows if row.get("selected") is True]
    if stage == "constraints":
        return [row for row in rows if (row.get("constraint_decision") or {}).get("accepted") is not False]
    return rows


def build_candidate_transitions(
    stages: dict[str, list[dict[str, Any]]], events: list[dict[str, Any]],
    *, reranker_top_k: int,
) -> list[dict[str, Any]]:
    dispositions = {
        str(row.get("chunk_id") or ""): row
        for event in events for row in event.get("candidate_dispositions", [])
        if isinstance(row, dict)
    }
    context_excluded = {
        str(row.get("chunk_id") or ""): row
        for event in events for row in event.get("context_excluded", [])
        if isinstance(row, dict)
    }
    pairs = (
        ("lexical", "fusion"), ("vector", "fusion"),
        ("fusion", "deduplicated"), ("deduplicated", "rerank"),
        ("rerank", "constraints"), ("constraints", "evidence"),
        ("evidence", "context"),
    )
    transitions: list[dict[str, Any]] = []
    for before, after in pairs:
        if before not in stages or after not in stages:
            continue
        source = _ranked(before, stages[before])
        target = _ranked(after, stages[after])
        positions: dict[str, tuple[int, str]] = {}
        for rank, item in enumerate(target, 1):
            representative = str(item.get("chunk_id") or "")
            for chunk_id in {representative, *map(str, item.get("member_chunk_ids", []))}:
                if chunk_id:
                    positions[chunk_id] = (rank, representative)
        for rank, item in enumerate(source, 1):
            chunk_id = str(item.get("chunk_id") or "")
            if not chunk_id:
                continue
            found = positions.get(chunk_id)
            reason_code = "present"
            reason = "进入下一阶段"
            status = "present"
            representative = found[1] if found else None
            if found and representative != chunk_id:
                status = "merged"
                reason_code = "merged_layout_equivalent"
                reason = f"已并入代表候选 {representative}"
            elif not found:
                status = "removed"
                if after == "fusion":
                    reason_code, reason = "fusion_top_k_cutoff", "未进入融合已记录深度"
                elif after == "deduplicated":
                    disposition = dispositions.get(chunk_id, {})
                    representative = str(disposition.get("representative_chunk_id") or "") or None
                    if representative and representative != chunk_id:
                        status, reason_code = "merged", "merged_layout_equivalent"
                    else:
                        reason_code = "fusion_top_k_cutoff"
                    reason = str(disposition.get("reason") or "未进入去重保留深度")
                elif after == "rerank":
                    reason_code = "reranker_top_k_cutoff" if rank > reranker_top_k else "reranker_not_selected"
                    reason = "超过重排入口 TopK" if rank > reranker_top_k else "重排未选中"
                elif after == "constraints":
                    rejected = next((row for row in stages[after] if row.get("chunk_id") == chunk_id), {})
                    decision = rejected.get("constraint_decision") or {}
                    if decision.get("accepted") is False:
                        reason_code, reason = "constraint_conflict", str(decision.get("reason") or "条件冲突")
                    else:
                        reason_code, reason = "reranker_not_selected", "没有进入条件检查"
                elif after == "evidence":
                    rejected = next((row for row in stages[after] if row.get("chunk_id") == chunk_id), {})
                    reason = str(rejected.get("reason") or "未进入最终证据")
                    decision = rejected.get("constraint_decision") or {}
                    if decision.get("accepted") is False:
                        reason_code = "constraint_conflict"
                    elif "文档" in reason or "每份" in reason:
                        reason_code = "document_limit"
                    elif "分数" in reason or "阈值" in reason:
                        reason_code = "below_score_threshold"
                    else:
                        reason_code = "final_top_k_cutoff"
                elif after == "context":
                    excluded = context_excluded.get(chunk_id, {})
                    reason_code = str(excluded.get("reason_code") or "context_budget_cutoff")
                    reason = str(excluded.get("reason") or "上下文字符预算未纳入")
            if after == "context" and found:
                included = target[found[0] - 1]
                if included.get("truncated"):
                    reason_code, reason = "context_text_truncated", "正文仅部分送入上下文"
            transitions.append(asdict(CandidateTransition(
                chunk_id, before, after, status, reason_code, reason,
                rank, found[0] if found else None, representative,
            )))
    return transitions
