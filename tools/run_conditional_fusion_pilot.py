"""Real-answer pilot for the isolated conditional retrieval candidate."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import patch

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever

from tools.branch_candidate_v2 import validate
from tools.conditional_fusion_candidate import merge_with_route_coverage
from tools.rag_eval_common import ROOT, sha, write
from tools.rag_trace_audit import audit_case_sources
from tools.scoped_applicability_candidate import ScopedApplicability
from tools.validate_conditional_fusion_v2 import ScopedPolicy
from tools.validate_evidence_answer_flow import Experiment


OUT = ROOT / "reports/rag_conditional_fusion_pilot_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
PLANS = ROOT / "reports/rag_conditional_fusion_cohort_20261110_01/plans.json"
IDS = ("rag-biz-008", "rag-biz-031", "rag-biz-040")
ARMS = ("A", "R", "RC")
REPEATS = 3


def run(exp, case, branches, arm):
    token = CancelToken()
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    matcher = ScopedApplicability(tuple(b.condition_quote for b in branches), mode="C") if arm == "RC" else None
    policy = ScopedPolicy(exp.settings, matcher) if matcher else None
    events = []
    route_audit = []

    class Probe(HybridRetriever):
        def retrieve(self, query, current_token, plan=None):
            original = super().retrieve(query, current_token, plan)
            if arm == "A":
                return original
            routes = {}
            for position, branch in enumerate(branches, 1):
                route_query = replace(query, text=branch.condition_quote)
                routes[f"branch_{position}"] = super().retrieve(
                    route_query, current_token,
                    QueryInterpreter().interpret(route_query.text, current_token))
            merged, audit = merge_with_route_coverage(original, routes,
                                                       limit=len(original), per_route=10)
            route_audit.extend(audit)
            return merged

    def progress(event):
        events.append(event)

    with patch("local_full_text_search.rag.answer_service.HybridRetriever", Probe):
        service = AnswerService(
            exp.source_store, exp.store, exp.embedding, exp.ranker, gateway,
            exp.secrets, exp.settings, progress_callback=progress,
            constraint_matcher=matcher, evidence_policy=policy,
        )
        s = exp.settings
        query = RagQuery(text=case.question, include_ocr=True,
                         min_ocr_confidence=s.ocr_min_confidence,
                         lexical_top_k=s.rag_lexical_top_k, vector_top_k=s.rag_vector_top_k,
                         fusion_top_k=s.rag_fusion_top_k, reranker_top_k=s.rag_reranker_top_k,
                         final_top_k=s.rag_final_top_k)
        answer = _answer_record(service.answer(query, token))
    context_event = next(event for event in reversed(events) if "llm_context" in event)
    context = context_event["llm_context"]
    entries = context_event["context_entries"]
    judgement = GoldAnswerJudge(gateway, exp.settings, exp.key).evaluate(
        case, {**answer, "evidence_context": context}, token)
    return {"arm": arm, "answer": answer, "judgement": judgement,
            "context": context, "entries": entries, "route_audit": route_audit,
            "source_audit": audit_case_sources(case, exp.chunk_rows, entries),
            "condition_bindings": matcher.last_bindings if matcher else []}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(OUT, 1)
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    plans = {row["id"]: row for row in json.loads(PLANS.read_text(encoding="utf-8"))}
    assert all(exp.bundle.bundle_id == cases[cid].corpus_snapshot for cid in IDS)
    assert all(len(validate(cases[cid].question, plans[cid]["raw"])) == 2 for cid in IDS)
    write(OUT / "protocol.json", {
        "ids": IDS, "arms": ARMS, "repeats": REPEATS,
        "bundle_id": exp.bundle.bundle_id, "gold_sha256": sha(GOLD),
        "planner_output_sha256": sha(PLANS),
        "fusion_candidate_sha256": sha(ROOT / "tools/conditional_fusion_candidate.py"),
        "scope_candidate_sha256": sha(ROOT / "tools/scoped_applicability_candidate.py"),
        "stage": "real AnswerService answer with frozen validated planner output; not full planner reliability test",
    })
    path = OUT / "runs.jsonl"
    completed = set()
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            completed = {(row["id"], row["repeat"], row["arm"])
                         for line in stream if (row := json.loads(line))}
    with path.open("a", encoding="utf-8") as stream:
        for repeat in range(1, REPEATS + 1):
            for cid in IDS:
                case = cases[cid]
                branches = validate(case.question, plans[cid]["raw"])
                for arm in ARMS[repeat % len(ARMS):] + ARMS[:repeat % len(ARMS)]:
                    if (cid, repeat, arm) in completed:
                        continue
                    try:
                        data = run(exp, case, branches, arm)
                    except Exception as exc:
                        data = {"arm": arm, "error": exp.redact(exc)}
                    row = {"id": cid, "repeat": repeat, **data}
                    stream.write(json.dumps(row, ensure_ascii=False).replace(exp.key, "[REDACTED]") + "\n")
                    stream.flush()
                    print(cid, repeat, arm,
                          data.get("judgement", {}).get("verdict", data.get("error", ""))[:100],
                          flush=True)


if __name__ == "__main__":
    main()
