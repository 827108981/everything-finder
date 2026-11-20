"""Verify and summarize a frozen, fixed-context evidence delivery diagnostic."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.index_store import RagIndexStore

from tools.rag_eval_common import ROOT, lines, read, sha, write
from tools.rag_trace_audit import audit_case_sources
from tools.validate_evidence_delivery import OUT, HISTORIC, NATIVE, PRIOR, DIAGNOSTIC_IDS, verify_freeze


def main() -> None:
    protocol = verify_freeze()
    contexts = {row["id"]: row for row in lines(OUT / "diagnostic_contexts.jsonl")}
    runs = lines(OUT / "diagnostic_runs.jsonl")
    assert len(runs) == len(DIAGNOSTIC_IDS) * 3
    assert {(row["id"], row["repeat"]) for row in runs} == {
        (case_id, repeat) for case_id in DIAGNOSTIC_IDS for repeat in (1, 2, 3)
    }
    cases = {case.case_id: case for case in load_gold_dataset(
        ROOT / "tests/rag/fixtures/rag_gold_v4_2_review.json", reviewed_only=False).cases}
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    with store.connect_read_only(store.active().metadata_path) as conn:
        chunk_rows = load_chunk_rows(conn)
    audit_rows = []
    for case_id in DIAGNOSTIC_IDS:
        record = contexts[case_id]
        for arm in ("A", "B"):
            audit_rows.extend({"arm": arm, **row} for row in audit_case_sources(
                cases[case_id], chunk_rows, record[f"{arm}_entries"]
            ))
    with (OUT / "diagnostic_source_audit.jsonl").open("x", encoding="utf-8") as target:
        for row in audit_rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    by_case = defaultdict(list)
    for row in runs:
        by_case[row["id"]].append(row)
    per_case = []
    for case_id in DIAGNOSTIC_IDS:
        ordered = sorted(by_case[case_id], key=lambda item: item["repeat"])
        context = contexts[case_id]
        assert all(row["arms"][arm]["context_sha256"] ==
                   __import__("hashlib").sha256(context[f"{arm}_context"].encode()).hexdigest()
                   for row in ordered for arm in ("A", "B"))
        evidence = [row for row in audit_rows if row["id"] == case_id]
        per_case.append({
            "id": case_id, "group_id": cases[case_id].raw["group_id"],
            "A": [row["arms"]["A"].get("judgement", {}).get("verdict", "execution_or_judge_error")
                  for row in ordered],
            "B": [row["arms"]["B"].get("judgement", {}).get("verdict", "execution_or_judge_error")
                  for row in ordered],
            "B_source_binding": [row["arms"]["B"].get("judgement", {}).get("source_binding", {}).get("status")
                                 for row in ordered],
            "A_evidence_complete": sum(row["actual_context_quote"] for row in evidence if row["arm"] == "A"),
            "B_evidence_complete": sum(row["actual_context_quote"] for row in evidence if row["arm"] == "B"),
            "total_evidence": sum(row["arm"] == "A" for row in evidence),
            "A_entries": len(context["A_entries"]), "B_entries": len(context["B_entries"]),
            "A_chars": len(context["A_context"]), "B_chars": len(context["B_context"]),
            "removed_chunk_ids": context["removed_chunk_ids"],
            "added_chunk_ids": context["added_chunk_ids"],
            "source_confirmed": all(row["original_quote"] and row["index_quote"] for row in evidence),
        })
    write(OUT / "diagnostic_summary.json", {
        "bundle": protocol["published_ai_bundle"],
        "model": protocol["model"],
        "case_count": len(per_case), "paired_runs": len(runs),
        "per_case": per_case,
        "verdicts": {arm: dict(Counter(row["arms"][arm].get("judgement", {}).get("verdict", "execution_or_judge_error")
                                      for row in runs)) for arm in ("A", "B")},
        "source_rows": len(audit_rows),
        "original_and_index_verified": all(row["original_quote"] and row["index_quote"] for row in audit_rows),
        "all_B_evidence_delivered": all(row["actual_context_quote"] for row in audit_rows if row["arm"] == "B"),
        "all_contexts_under_original_limits": all(item["B_entries"] <= 8 and item["B_chars"] <= 12000
                                                   for item in per_case),
    })
    write(OUT / "final_manifest.json", {
        "published_ai_bundle": protocol["published_ai_bundle"],
        "frozen_input_hashes_verified": True,
        "settings_sha256_at_start": protocol["settings_sha256"],
        "settings_sha256_at_end": sha(Path(next(path for path in read(OUT / "input_manifest_before.json")
                                              if path.endswith("settings.json")))),
        "result_hashes": {str(path): sha(path) for path in sorted(OUT.iterdir()) if path.is_file()},
    })
    print(read(OUT / "diagnostic_summary.json"))


if __name__ == "__main__":
    main()
