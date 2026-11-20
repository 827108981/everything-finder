"""Offline ranking diagnostic on frozen candidates; does not alter application or Gold."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from local_full_text_search.rag.evaluation.evidence_matcher import (
    EvidenceAlternative, EvidencePart, GoldEvidenceMatch,
)
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


RUN = Path("reports/rerank_dev_holdout_20261101")
OUT = RUN / "scope_bonus_diagnostic.json"
MODEL = re.compile(r"(?i)(?<![A-Z0-9])(?:CL|BS)[-‐‑–— ]?\d{3,4}[A-Z]?|(?<![A-Z0-9])M\d{4}")


def models(value: str) -> set[str]:
    return {re.sub(r"[^A-Z0-9]", "", m.group().upper()) for m in MODEL.finditer(value)}


def scope_signal(question: str, candidate: dict) -> tuple[float, list[str]]:
    """Conservative generic source applicability: explicit model and corpus line only."""
    path = str(candidate["file_path"])
    query_models = models(question)
    path_models = models(path)
    bonus = 0.0
    why = []
    if len(query_models) == 1 and path_models:
        if query_models & path_models:
            bonus += 0.02
            why.append("named model in source path")
        else:
            bonus -= 0.02
            why.append("different named model in source path")
    query_lines = {x for x in ("生化", "免疫") if x in question}
    source_lines = {x for x in ("生化", "免疫") if path.startswith("E:\\" + x + "资料\\")}
    if len(query_lines) == 1 and len(source_lines) == 1:
        if query_lines == source_lines:
            bonus += 0.01
            why.append("same corpus line")
        else:
            bonus -= 0.01
            why.append("different corpus line")
    return bonus, why


def mappings(record: dict, gold: dict) -> list[GoldEvidenceMatch]:
    result = []
    for m in record["gold_mappings"]:
        seq = int(m["evidence_id"].rsplit(":", 1)[1]) - 1
        evidence = (gold["gold_evidence"] if m["kind"] == "positive" else gold["hard_negatives"])[seq]
        result.append(GoldEvidenceMatch(
            evidence_id=m["evidence_id"], kind=m["kind"], file=m["file"],
            page=m["page"], quote=str(evidence.get("quote") or ""),
            chunk_ids=tuple(m["chunk_ids"]), confidence=m["confidence"],
            alternatives=tuple(EvidenceAlternative(
                a["alternative_id"], a["match_type"],
                tuple(EvidencePart(**part) for part in a["parts"]), a["confirmed"]
            ) for a in m.get("alternatives", [])),
        ))
    return result


def main() -> None:
    protocol = json.loads((RUN / "protocol.json").read_text(encoding="utf-8"))
    gold = {}
    for cohort, data in protocol["datasets"].items():
        path = Path(data["path"])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == data["sha256"]
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            gold[case["id"]] = case
    results = []
    for line in (RUN / "cases.jsonl").open(encoding="utf-8"):
        record = json.loads(line)
        if record["arm"] != "pure":
            continue
        by_id = {x["chunk_id"]: x for x in record["candidate_stages"]["rerank"]}
        original = [by_id[x["chunk_id"]] for x in record["candidate_stages"]["deduplicated"] if x["chunk_id"] in by_id]
        assert len(original) == len(by_id)
        ranked = sorted(original, key=lambda x: (
            -(float(x["rerank_score"]) + scope_signal(record["question"], x)[0]),
            -float(x["fused_score"]), x["chunk_id"],
        ))
        # The stored rerank stage is the complete scored order before document diversity.
        assert [x["chunk_id"] for x in sorted(original, key=lambda x: (
            -float(x["rerank_score"]), -float(x["fused_score"]), x["chunk_id"]
        ))][:20] == [x["chunk_id"] for x in record["candidate_stages"]["rerank"]][:20]
        gold_maps = mappings(record, gold[record["id"]])
        scores = {str(k): score_stage("rerank", ranked[:k], gold_maps) for k in (1, 3, 8, 20)}
        results.append({
            "id": record["id"], "cohort": record["cohort"], "question": record["question"],
            "status": record["review_status"], "scores": scores,
            "baseline": {str(k): record["rerank_topk"][str(k)] for k in (1, 3, 8, 20)},
            "first_eight": [{"rank": i, "file": x["file_path"], "location": x["location"],
                             "raw_score": x["rerank_score"], "scope_delta": scope_signal(record["question"], x)[0],
                             "scope_reason": scope_signal(record["question"], x)[1]}
                            for i, x in enumerate(ranked[:8], 1)],
        })
    assert len(results) == 60
    summary = {}
    for cohort in ("development", "holdout_pending"):
        items = [x for x in results if x["cohort"] == cohort and gold[x["id"]]["expected_behavior"] != "refuse"]
        summary[cohort] = {}
        for k in (1, 3, 8):
            summary[cohort][str(k)] = {
                name: {"hit": sum(x[field][str(k)]["gold_hit"] for x in items),
                       "mean_coverage": sum(x[field][str(k)]["gold_evidence_recall"] for x in items) / len(items),
                       "mrr": sum(x[field][str(k)]["reciprocal_rank"] for x in items) / len(items),
                       "negatives": sum(x[field][str(k)]["hard_negative_found"] for x in items)}
                for name, field in (("pure", "baseline"), ("pure_scope", "scores"))
            }
        summary[cohort]["changes"] = [{"id": x["id"],
            "pure_rank": x["baseline"]["20"]["first_gold_rank"],
            "scope_rank": x["scores"]["20"]["first_gold_rank"],
            "pure_coverage8": x["baseline"]["8"]["gold_evidence_recall"],
            "scope_coverage8": x["scores"]["8"]["gold_evidence_recall"],
            "new_negatives": sorted(set(x["scores"]["8"]["hard_negative_ids"]) -
                                    set(x["baseline"]["8"]["hard_negative_ids"]))}
            for x in items if x["scores"]["20"]["first_gold_rank"] != x["baseline"]["20"]["first_gold_rank"]
            or x["scores"]["8"]["gold_evidence_recall"] != x["baseline"]["8"]["gold_evidence_recall"]]
    OUT.write_text(json.dumps({"bundle_id": protocol["bundle_id"], "parameters": {"model_match": 0.02,
        "model_mismatch": -0.02, "line_match": 0.01, "line_mismatch": -0.01},
        "note": "Post-hoc diagnostic; holdout already inspected; Gold incomplete for equivalent sources.",
        "summary": summary, "cases": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
