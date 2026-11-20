"""Verify frozen evidence-delivery inputs and hash the final batch artifacts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from tools.rag_eval_common import ROOT, lines, read, sha, write
from tools.recheck_evidence_delivery_batch import verify as verify_complete_batch
from tools.validate_evidence_delivery import verify_freeze


OUT = ROOT / "reports/rag_evidence_delivery_20261109_01"
BATCH = OUT / "complete_batch_v2"


def main() -> None:
    frozen = verify_freeze()
    complete = verify_complete_batch()
    rows = list(lines(BATCH / "runs.jsonl"))
    keys = [(row["id"], row["repeat"]) for row in rows]
    assert len(rows) == 30 and len(set(keys)) == 30
    counts: dict[str, dict[str, Counter]] = {}
    for row in rows:
        assert row["id"] in complete["case_ids"]
        cohort = row["cohort"]
        counts.setdefault(cohort, {"A": Counter(), "B": Counter()})
        for arm in "AB":
            result = row["arms"][arm]
            assert "request_snapshot" in result and "judgement" in result
            assert hashlib.sha256(json.dumps(
                result["request_snapshot"], ensure_ascii=False,
                sort_keys=True).encode("utf-8")).hexdigest() == result["request_sha256"]
            counts[cohort][arm][result["judgement"]["verdict"]] += 1
    assert len([row for row in rows if row["cohort"] == "D"]) == 12
    assert len([row for row in rows if row["cohort"] == "C"]) == 18
    artifacts = [
        OUT / name for name in (
            "fact_gap_audit.jsonl", "fact_gap_summary.json",
            "diagnostic_contexts.jsonl", "diagnostic_source_audit.jsonl",
            "diagnostic_runs.jsonl", "controls.json", "control_runs.jsonl",
            "report.md", "failure_stage_matrix.md", "source_layout_audit.md",
            "diagnostic_gate.md", "execution_log.md", "verification.md",
        )
    ] + [BATCH / "protocol.json", BATCH / "runs.jsonl"]
    result = {
        "published_ai_bundle": frozen["published_ai_bundle"],
        "frozen_inputs_verified": True,
        "complete_batch_verified": True,
        "case_count": len(complete["case_ids"]),
        "paired_runs": len(rows),
        "generation_calls": 2 * len(rows),
        "cohort_verdicts": {group: {arm: dict(verdicts)
                                    for arm, verdicts in arms.items()}
                            for group, arms in counts.items()},
        "artifact_sha256": {str(path): sha(path) for path in artifacts},
    }
    write(OUT / "end_manifest.json", result)
    print(json.dumps({key: result[key] for key in (
        "published_ai_bundle", "case_count", "paired_runs",
        "generation_calls", "cohort_verdicts")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
