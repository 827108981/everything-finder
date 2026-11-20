"""Frozen question cohort for isolated conditional fusion retrieval checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever

from tools.branch_candidate_v2 import PROMPT, validate
from tools.conditional_fusion_candidate import merge_with_route_coverage
from tools.rag_eval_common import ROOT, sha, write
from tools.rag_trace_audit import audit_case_sources
from tools.validate_evidence_answer_flow import Experiment


OUT = ROOT / "reports/rag_conditional_fusion_cohort_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
SOURCE = ROOT / "reports/rag_gold_v4_3_retest_20261110/runs.jsonl"
IDS = tuple(f"rag-biz-{number:03d}" for number in (8, 19, 22, 27, 31, 36, 40, 41, 44, 48))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(OUT, 1)
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    assert all(exp.bundle.bundle_id == cases[cid].corpus_snapshot for cid in IDS)
    previous = {}
    with SOURCE.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("id") in IDS and row.get("repeat") == 1:
                previous[row["id"]] = row
    assert set(previous) == set(IDS)
    write(OUT / "protocol.json", {
        "ids": IDS, "bundle_id": exp.bundle.bundle_id,
        "gold_sha256": sha(GOLD), "baseline_sha256": sha(SOURCE),
        "settings_sha256": sha(SETTINGS_PATH),
        "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "candidate_sha256": sha(ROOT / "tools/conditional_fusion_candidate.py"),
        "note": "Cohort frozen before planner calls. A is saved native context; B changes only hybrid candidate coverage. No answer generation.",
    })
    plan_path = OUT / "plans.json"
    if plan_path.exists():
        plans = json.loads(plan_path.read_text(encoding="utf-8"))
        assert [row["id"] for row in plans] == list(IDS)
    else:
        gateway = OpenAICompatibleGateway(max_transient_retries=2)
        plans = []
        for cid in IDS:
            case = cases[cid]
            try:
                raw = gateway.complete_text(TextCompletionRequest(
                    system_prompt=PROMPT, user_prompt=case.question,
                    model=exp.settings.rag_api_model,
                    api_base_url=exp.settings.rag_api_base_url,
                    api_style=exp.settings.rag_api_style, api_key=exp.key,
                    timeout_seconds=exp.settings.rag_api_timeout_seconds,
                    max_output_tokens=800,
                ), CancelToken())
                branches = validate(case.question, raw)
                error = ""
            except Exception as exc:
                raw, branches, error = "", (), exp.redact(exc)
            plans.append({"id": cid, "question": case.question, "raw": raw,
                          "branches": [vars(branch) for branch in branches], "error": error})
            print("plan", cid, len(branches), error[:80], flush=True)
        write(plan_path, plans)

    results = []
    for plan in plans:
        cid = plan["id"]
        case = cases[cid]
        original = previous[cid]
        baseline_entries = original["entries"]
        baseline_audit = audit_case_sources(case, exp.chunk_rows, baseline_entries)
        if not plan["branches"]:
            results.append({"id": cid, "status": "fallback", "reason": plan["error"] or "invalid_or_no_branches",
                            "a": baseline_audit, "b": baseline_audit,
                            "a_context": original["context"], "b_context": original["context"],
                            "route_audit": []})
            continue
        branches = validate(case.question, plan["raw"])
        assert branches
        routes = []

        class Probe(HybridRetriever):
            def retrieve(self, query, token, interpreted=None):
                original_candidates = super().retrieve(query, token, interpreted)
                route_candidates = {}
                for position, branch in enumerate(branches, 1):
                    route_query = replace(query, text=branch.condition_quote)
                    route_candidates[f"branch_{position}"] = super().retrieve(
                        route_query, token, QueryInterpreter().interpret(route_query.text, token))
                merged, audit = merge_with_route_coverage(
                    original_candidates, route_candidates, limit=len(original_candidates), per_route=10)
                routes.extend(audit)
                return merged

        with patch("local_full_text_search.rag.answer_service.HybridRetriever", Probe):
            run = exp.native(case, None, CancelToken(), retrieve=True)
        if run.get("error"):
            results.append({"id": cid, "status": "error", "error": run["error"],
                            "a": baseline_audit, "route_audit": routes})
            print("retrieve", cid, "error", run["error"][:80], flush=True)
            continue
        actual_audit = audit_case_sources(case, exp.chunk_rows, run["entries"])
        results.append({"id": cid, "status": "tested", "a": baseline_audit, "b": actual_audit,
                        "a_context": original["context"], "b_context": run["context"],
                        "a_entries": baseline_entries, "b_entries": run["entries"],
                        "route_audit": routes})
        before = sum(item["actual_context_quote"] for item in baseline_audit)
        after = sum(item["actual_context_quote"] for item in actual_audit)
        print("retrieve", cid, before, after,
              "changed", original["context"] != run["context"], flush=True)
    write(OUT / "static_results.json", results)


if __name__ == "__main__":
    main()
