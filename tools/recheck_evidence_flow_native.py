"""Native RAG A/B rerun after an external settings-file rewrite."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from tools.rag_eval_common import ROOT, lines, read, sha, write
from tools.validate_evidence_answer_flow import FIXTURES, Experiment, needs_repeat


OLD = ROOT / "reports/rag_evidence_answer_flow_20261108"
OUT = ROOT / "reports/rag_evidence_answer_native_recheck_20261109"


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def public_settings(settings: AppSettings) -> dict:
    raw = asdict(settings)
    result = {key: value for key, value in raw.items()
              if key.startswith("rag_") and not any(word in key for word in ("secret", "api_base_url"))}
    result["api_base_url_sha256"] = hashlib.sha256(settings.rag_api_base_url.encode("utf-8")).hexdigest()
    result["api_secret_ref_sha256"] = hashlib.sha256(settings.rag_api_secret_ref.encode("utf-8")).hexdigest()
    return result


def cases():
    def load(path):
        return load_gold_dataset(path, reviewed_only=False).cases

    formal = [("formal", case) for case in load(FIXTURES[0]) if case.review_status == "reviewed"]
    regression = [("regression_pending", case) for path in FIXTURES[1:] for case in load(path)]
    independent = [("independent_pending", case) for case in load(OUT / "acceptance_review.json")]
    assert len(formal) == 48 and len(regression) == 30 and len(independent) == 12
    assert all(case.review_status == "pending" for _, case in independent)
    assert len({case.case_id for _, case in formal + regression + independent}) == 90
    return formal + regression + independent


def frozen_files(exp: Experiment) -> list[Path]:
    return [
        Path(__file__).resolve(), ROOT / "tools/validate_evidence_answer_flow.py",
        ROOT / "tools/rag_eval_common.py", ROOT / "tools/rag_trace_audit.py",
        OUT / "acceptance_review.json", OUT / "manual_repeat_ids.json", *FIXTURES,
        exp.store.manifest_path, exp.bundle.metadata_path,
        *(ROOT / "local_full_text_search/rag").rglob("*.py"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("native", "repeat", "finish"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    accepted = OUT / "acceptance_review.json"
    manual = OUT / "manual_repeat_ids.json"
    if not accepted.exists():
        accepted.write_bytes((OLD / "acceptance_review.json").read_bytes())
    if not manual.exists():
        manual.write_bytes((OLD / "manual_repeat_ids.json").read_bytes())

    exp = Experiment(OUT, args.workers)
    assert exp.bundle.bundle_id == read(OLD / "protocol.json")["bundle"]
    cohort_cases = cases()
    freeze = {str(path): sha(path) for path in frozen_files(exp)}
    protocol_path = OUT / "protocol.json"
    effective = public_settings(exp.settings)
    if not protocol_path.exists():
        assert args.stage == "native"
        write(protocol_path, {
            "started": now(),
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "bundle": exp.bundle.bundle_id,
            "settings_file_sha256_at_start": sha(SETTINGS_PATH),
            "effective_rag_settings": effective,
            "input_sha256": freeze,
            "cases": [{"cohort": cohort, "id": case.case_id, "group_id": case.raw.get("group_id")}
                      for cohort, case in cohort_cases],
            "candidate": "Unchanged two-stage evidence-map implementation from the previous full experiment.",
            "comparison": "Both arms use the same Experiment settings instance; model calls are independent.",
            "repeats": "Two more paired runs for all first-round differences, uncertainty, or frozen manual IDs.",
            "independent_status": "Reuses previous 12 pending questions; a reproduction sample, not a fresh holdout.",
        })
    else:
        protocol = read(protocol_path)
        assert protocol["bundle"] == exp.bundle.bundle_id
        assert protocol["effective_rag_settings"] == effective, "Effective RAG settings changed"
        assert protocol["input_sha256"] == freeze, "Frozen code, index, or Gold changed"

    target = OUT / "e2e_runs.jsonl"
    if args.stage == "native":
        exp.run(target, [(cohort, case, 1) for cohort, case in cohort_cases])
    else:
        first = {row["id"]: row for row in lines(target) if row["repeat"] == 1}
        assert len(first) == 90
        selected = {case_id for case_id, row in first.items() if needs_repeat(row)}
        selected.update(read(manual))
        assert selected <= set(first)
        if args.stage == "repeat":
            write(OUT / "repeat_ids.json", sorted(selected))
            exp.run(target, [(cohort, case, repeat) for cohort, case in cohort_cases
                             if case.case_id in selected for repeat in (2, 3)])
        else:
            data = lines(target)
            assert len(data) == 90 + 2 * len(selected)
            assert all(row.get("same_context") for row in data)
            assert all(not row["arms"][arm].get("error") for row in data for arm in ("A", "B"))
            assert {row["id"] for row in data if row["repeat"] > 1} == selected
            assert all(sha(path) == expected for path, expected in freeze.items())
            current_settings = AppSettings.from_dict(read(SETTINGS_PATH))
            write(OUT / "final_manifest.json", {
                "finished": now(), "bundle": exp.bundle.bundle_id,
                "model": exp.settings.rag_api_model,
                "settings_file_sha256_at_start": read(protocol_path)["settings_file_sha256_at_start"],
                "settings_file_sha256_at_end": sha(SETTINGS_PATH),
                "effective_rag_settings_equal_at_end": public_settings(current_settings) == effective,
                "code_index_gold_hashes_verified": True,
                "first_pairs": dict(Counter(row["cohort"] for row in data if row["repeat"] == 1)),
                "repeat_cases": len(selected), "total_pairs": len(data), "same_context_pairs": len(data),
            })
            print(f"finished: {len(data)} pairs; {len(selected)} repeat cases")


if __name__ == "__main__":
    main()
