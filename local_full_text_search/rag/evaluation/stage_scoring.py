"""逐阶段按完整引文覆盖计分；疑似来源与部分片段仅用于诊断。"""

from __future__ import annotations

from typing import Any
import hashlib

from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch


SCORER_VERSION = "source-quote-v3"

def _part_rank(part, quote: str, candidates: list[dict[str, Any]], stage: str, *, require_text: bool = False) -> int | None:
    expected = canonical_content(quote)[part.quote_start:part.quote_end]
    for rank, candidate in enumerate(candidates, 1):
        ids = {str(candidate.get("chunk_id") or "")}
        ids.update(str(value) for value in candidate.get("member_chunk_ids", []))
        if part.chunk_id not in ids:
            continue
        actual = candidate.get("included_text") if stage == "context" else candidate.get("text")
        if require_text and not isinstance(actual, str):
            continue
        # 旧运行没有保存正文时只能回退 ID；新版运行必须核对实际文字。
        if actual is not None and expected and expected not in canonical_content(str(actual)):
            continue
        if require_text and getattr(part, "source_quote_occurrences", 1) > 1:
            # A shared number in a projected row does not locate its source row.
            if hashlib.sha256(canonical_content(actual).encode("utf-8")).hexdigest() != part.source_canonical_hash:
                continue
        return rank
    return None


def score_stage(
    stage: str, candidates: list[dict[str, Any]], mappings: list[GoldEvidenceMatch],
) -> dict[str, Any]:
    if stage == "evidence" and any("selected" in item for item in candidates):
        candidates = [item for item in candidates if item.get("selected") is True]
    positives = [item for item in mappings if item.kind == "positive"]
    negatives = [item for item in mappings if item.kind == "hard_negative"]
    complete_ranks: dict[str, int] = {}
    first_fragment_ranks: dict[str, int] = {}
    match_types: dict[str, str] = {}
    for evidence in positives:
        for alternative in evidence.alternatives:
            if not alternative.confirmed:
                continue
            part_ranks = [_part_rank(part, evidence.quote, candidates, stage,
                                     require_text=stage == "context") for part in alternative.parts]
            present_ranks = [rank for rank in part_ranks if rank is not None]
            if present_ranks:
                first_fragment_ranks[evidence.evidence_id] = min(
                    first_fragment_ranks.get(evidence.evidence_id, min(present_ranks)), min(present_ranks)
                )
            if len(present_ranks) == len(part_ranks) and part_ranks:
                completed = max(present_ranks)
                if evidence.evidence_id not in complete_ranks or completed < complete_ranks[evidence.evidence_id]:
                    complete_ranks[evidence.evidence_id] = completed
                    match_types[evidence.evidence_id] = alternative.match_type
        if evidence.alternatives:
            continue
        # 历史数据适配：没有区间映射时沿用 ID，仅用于解释旧结果。
        for rank, candidate in enumerate(candidates, 1):
            ids = {str(candidate.get("chunk_id") or "")}
            ids.update(str(value) for value in candidate.get("member_chunk_ids", []))
            if ids.intersection(evidence.chunk_ids):
                if stage == "context":
                    actual = candidate.get("included_text")
                    if not isinstance(actual, str) or not canonical_content(evidence.quote) or canonical_content(evidence.quote) not in canonical_content(actual):
                        continue
                complete_ranks[evidence.evidence_id] = rank
                first_fragment_ranks[evidence.evidence_id] = rank
                match_types[evidence.evidence_id] = "legacy_quote_source" if stage == "context" else "legacy_id_only"
                break
    negative_ids: list[str] = []
    page_only_ids: list[str] = []
    unresolved_ids: list[str] = []
    negative_ranks: dict[str, int] = {}
    for evidence in negatives:
        source_hit = any(
            {str(item.get("chunk_id") or ""), *map(str, item.get("member_chunk_ids", []))}
            .intersection(evidence.chunk_ids)
            for item in candidates
        )
        confirmed = [a for a in evidence.alternatives if a.confirmed]
        completion_ranks = []
        for alternative in confirmed:
            ranks = [_part_rank(part, evidence.quote, candidates, stage, require_text=True)
                     for part in alternative.parts]
            if ranks and all(rank is not None for rank in ranks):
                completion_ranks.append(max(ranks))
        complete = bool(completion_ranks)
        if complete:
            negative_ids.append(evidence.evidence_id)
            negative_ranks[evidence.evidence_id] = min(completion_ranks)
        elif source_hit:
            page_only_ids.append(evidence.evidence_id)
            if not confirmed or any(getattr(p, "source_quote_occurrences", 1) > 1 for a in confirmed for p in a.parts) or not any(
                isinstance(c.get("included_text") if stage == "context" else c.get("text"), str)
                for c in candidates
                if {str(c.get("chunk_id") or ""), *map(str, c.get("member_chunk_ids", []))}.intersection(evidence.chunk_ids)
            ):
                unresolved_ids.append(evidence.evidence_id)
    first_gold = min(complete_ranks.values()) if complete_ranks else None
    total = len(positives)
    return {
        "candidate_count": len(candidates),
        "gold_evidence_found": len(complete_ranks),
        "gold_evidence_total_located": sum(bool(item.chunk_ids) for item in positives),
        "gold_evidence_total_declared": total,
        "gold_evidence_recall": len(complete_ranks) / total if total else None,
        "gold_hit": bool(complete_ranks),
        "first_gold_rank": first_gold,
        "first_gold_fragment_rank": min(first_fragment_ranks.values()) if first_fragment_ranks else None,
        "reciprocal_rank": 1.0 / first_gold if first_gold else 0.0 if total else None,
        "gold_completion_ranks": complete_ranks,
        "gold_match_types": match_types,
        "hard_negative_found": len(negative_ids),
        "hard_negative_ids": negative_ids,
        "hard_negative_completion_ranks": negative_ranks,
        "hard_negative_page_only_ids": page_only_ids,
        "hard_negative_unresolved_ids": unresolved_ids,
        "scorer_version": SCORER_VERSION,
    }
