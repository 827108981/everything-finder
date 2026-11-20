"""Freeze and audit the branch retrieval experiment before changing it."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.index_store import RagIndexStore

from tools.branch_candidate_v2 import validate
from tools.rag_eval_common import ROOT, read, sha, write
from tools.rag_trace_audit import original_text
from tools.validate_post_gold_selection import reproduce


OUT = ROOT / "reports/rag_general_branch_scope_20261111_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
BASE = ROOT / "reports/rag_gold_v4_3_retest_20261110/runs.jsonl"
OLD = ROOT / "reports/rag_conditional_fusion_pilot_20261110_01/runs.jsonl"
PLANS = ROOT / "reports/rag_conditional_fusion_cohort_20261110_01/plans.json"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    cases = load_gold_dataset(GOLD, reviewed_only=False).cases
    statuses = Counter(case.review_status for case in cases)
    with store.connect_read_only(bundle.metadata_path) as connection:
        replayed, _, _ = reproduce(settings, store, connection)
        replay = {"cases": len(replayed), "exact_context_and_entries": len(replayed),
                  "bundle_id": bundle.bundle_id}
        write(OUT / "baseline_replay.json", replay)
        old = [json.loads(line) for line in OLD.open(encoding="utf-8") if line.strip()]
        audited = []
        for row in old:
            if row["id"] != "rag-biz-008":
                continue
            entries = row.get("entries", [])
            source = next((entry for entry in entries if entry["source_id"] == "S3"), None)
            answer = row.get("answer", {}).get("text", "")
            expansion = bool(source and "[S3]" in answer)
            source_check = {}
            if expansion:
                cid = source["chunk_id"]
                chunk = store.load_chunk(connection, cid)
                assert chunk is not None
                path = Path(chunk.source.file_path)
                page = chunk.source.page_number
                page_text = original_text(path, page)
                quote = "胶管未安装到位脱落或者老化断裂"
                source_check = {
                    "chunk_id": cid, "source_id": source["source_id"],
                    "file": str(path), "page": page, "location": source["location"],
                    "file_sha256": sha(path),
                    "original_quote": canonical_content(quote) in canonical_content(page_text),
                    "index_quote": canonical_content(quote) in canonical_content(chunk.text),
                    "actual_quote": canonical_content(quote) in canonical_content(source["included_text"]),
                    "source_model_scope": "CL-8000i/CL-8200i",
                    "scope_in_original": "CL-8000i" in page_text and "CL" in page_text,
                    "scope_in_actual": "CL-8000i" in source["included_text"],
                    "scope_qualification_in_answer": "CL-8000i" in answer or "8200i" in answer,
                    "claim_with_s3": re.findall(r"[^\n]*\[S3\]", answer),
                }
            audited.append({
                "id": row["id"], "arm": row["arm"], "repeat": row["repeat"],
                "question": row.get("question") or next(c.question for c in cases if c.case_id == row["id"]),
                "answer": answer,
                "automatic_verdict": row.get("judgement", {}).get("verdict"),
                "automatic_source_binding": row.get("judgement", {}).get("source_binding"),
                "source_check": source_check,
                "manual_classification": (
                    "scope_not_proven" if expansion and not source_check["scope_qualification_in_answer"]
                    else "scope_qualified" if expansion else "no_added_scope_claim"
                ),
                "preexisting_missing_software_step": "软件中控制蠕动泵" not in answer
                and "软件界面控制蠕动泵" not in answer,
            })
    with (OUT / "claim_audit.jsonl").open("w", encoding="utf-8") as file:
        for row in audited:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    plans = read(PLANS)
    fallback = []
    for row in plans:
        if row["branches"]:
            reason = "valid"
        else:
            data = json.loads(row["raw"])
            branches = data.get("branches", [])
            reason = (
                "no_branches_proposed" if not branches else
                "first_branch_inherited_nonempty" if branches[0].get("inherited_quote") else
                "different_objects_or_methods_not_covered_by_validator"
            )
        fallback.append({"id": row["id"], "question": row["question"],
                         "raw": row["raw"], "status": reason,
                         "validated_count": len(validate(row["question"], row["raw"]))})
    write(OUT / "planner_coverage_audit.json", fallback)
    files = [GOLD, ROOT / "tests/rag/fixtures/rag_gold_v4_3.jsonl",
             BASE, OLD, PLANS, SETTINGS_PATH,
             ROOT / "tools/branch_candidate_v2.py",
             ROOT / "tools/conditional_fusion_candidate.py"]
    write(OUT / "protocol.json", {
        "bundle_id": bundle.bundle_id, "bundle_metadata": str(bundle.metadata_path),
        "source_generation": bundle.source_generation,
        "model": settings.rag_api_model, "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip(),
        "files_sha256": {str(path): sha(path) for path in files},
        "gold_statuses": dict(statuses),
        "past_run_rows": len(old), "audit_rows": len(audited),
        "baseline_replay": replay,
        "candidate_limit": "one initial and at most one evidence-driven revision",
        "decision_gate": "cross-business improvement, no confirmed new material error, source binding",
    })
    write(OUT / "dataset_manifest.json", {
        "formal_reviewed": sum(c.review_status == "reviewed" for c in cases),
        "formal_excluded": [c.case_id for c in cases if c.review_status != "reviewed"],
        "historical_regression": ["rag_acceptance_v1_review.json",
                                  "rag_row_holdout_v1_review.json",
                                  "rag_row_holdout_v2_review.json"],
        "historical_independent_now_development":
            "reports/rag_evidence_answer_native_recheck_20261109/acceptance_review.json",
    })
    (OUT / "working_tree_before.diff").write_text(subprocess.check_output(
        ["git", "diff", "--binary"], text=True, cwd=ROOT, errors="replace"), encoding="utf-8")
    (OUT / "working_tree_at_audit.txt").write_text(subprocess.check_output(
        ["git", "status", "--short"], text=True, cwd=ROOT), encoding="utf-8")
    print("bundle", bundle.bundle_id, "replay", len(replayed), "audit", len(audited), flush=True)


if __name__ == "__main__":
    main()
