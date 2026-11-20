"""Isolated native-chain A/R/C/RC check for v4.3 conditional retrieval."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from tools.branch_candidate_v2 import PROMPT, validate
from tools.conditional_fusion_candidate import merge_with_route_coverage
from tools.rag_eval_common import ROOT, read, sha, write
from tools.scoped_applicability_candidate import ScopedApplicability
from tools.validate_evidence_answer_flow import Experiment


OUT = ROOT / "reports/rag_conditional_fusion_v2_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
SOURCE = ROOT / "reports/rag_gold_v4_3_retest_20261110/runs.jsonl"
FROZEN = '{"branches":[{"condition_quote":"ALB、GLU 等小样本量项目重复性变差，而 ALT、AST 较稳定","inherited_quote":""},{"condition_quote":"所有项目都变差","inherited_quote":"重复性"}]}'


class ScopedPolicy(EvidencePolicy):
    def __init__(self, settings, matcher):
        super().__init__(
            min_vector_score=settings.rag_min_vector_score,
            min_rerank_score=settings.rag_min_rerank_score,
            max_per_document=settings.rag_max_chunks_per_document,
        )
        self.matcher = matcher

    def evaluate(self, query, candidates, *, plan=None):
        return super().evaluate(query, candidates,
                                plan=self.matcher.policy_plan(plan) if plan else None)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(OUT, 1)
    case = next(c for c in load_gold_dataset(GOLD, reviewed_only=False).cases if c.case_id == "rag-biz-040")
    branches = validate(case.question, FROZEN)
    assert len(branches) == 2
    previous = next(json.loads(line) for line in SOURCE.open(encoding="utf-8")
                    if '"id": "rag-biz-040"' in line[:100] and '"repeat": 1' in line[:350])
    assert exp.bundle.bundle_id == case.corpus_snapshot
    write(OUT / "protocol.json", {
        "bundle_id": exp.bundle.bundle_id,
        "gold_sha256": sha(GOLD), "source_sha256": sha(SOURCE),
        "settings_sha256": sha(SETTINGS_PATH),
        "planner_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "planner_output": FROZEN, "condition_queries": [b.condition_quote for b in branches],
        "fusion_candidate_sha256": sha(ROOT / "tools/conditional_fusion_candidate.py"),
        "scoped_condition_candidate_sha256": sha(ROOT / "tools/scoped_applicability_candidate.py"),
        "arms": {"A": "original", "R": "route coverage fusion", "C": "scoped condition only",
                 "RC": "route coverage and scoped condition"},
        "note": "Fixed validated planner output for local chain ablation. No answer generation or adoption claim.",
    })
    targets = {
        14: next(x["chunk_id"] for x in previous["candidate_stages"]["fusion"]
                 if x["filename"] == "生化试剂应用及常见问题-SH-20190110.pptx" and x["slide_number"] == 14),
        16: next(x["chunk_id"] for x in previous["candidate_stages"]["vector"]
                 if x["filename"] == "生化试剂应用及常见问题-SH-20190110.pptx" and x["slide_number"] == 16),
    }
    output = []
    for arm in ("A", "R", "C", "RC"):
        events, route_audit = [], []
        matcher = ScopedApplicability(tuple(b.condition_quote for b in branches), mode="C") if arm in ("C", "RC") else None
        policy = ScopedPolicy(exp.settings, matcher) if matcher else None

        class ProbeHybrid(HybridRetriever):
            def retrieve(self, query, token, plan=None):
                original = super().retrieve(query, token, plan)
                if arm not in ("R", "RC"):
                    return original
                routes = {}
                for position, branch in enumerate(branches, 1):
                    branch_query = replace(query, text=branch.condition_quote)
                    routes[f"branch_{position}"] = super().retrieve(
                        branch_query, token, QueryInterpreter().interpret(branch.condition_quote, token)
                    )
                merged, trace = merge_with_route_coverage(original, routes, limit=len(original), per_route=10)
                route_audit.extend(trace)
                return merged

        def progress(event):
            events.append(event)

        with patch("local_full_text_search.rag.answer_service.HybridRetriever", ProbeHybrid):
            service = AnswerService(
                exp.source_store, exp.store, exp.embedding, exp.ranker, None,
                exp.secrets, exp.settings, progress_callback=progress,
                constraint_matcher=matcher, evidence_policy=policy,
            )
            s = exp.settings
            query = RagQuery(text=case.question, include_ocr=True,
                             min_ocr_confidence=s.ocr_min_confidence,
                             lexical_top_k=s.rag_lexical_top_k, vector_top_k=s.rag_vector_top_k,
                             fusion_top_k=s.rag_fusion_top_k,
                             reranker_top_k=s.rag_reranker_top_k,
                             final_top_k=s.rag_final_top_k)
            result = service.retrieve(query, CancelToken())
        context_event = next(event for event in reversed(events) if "llm_context" in event)
        entries = context_event["context_entries"]
        if arm == "A":
            assert context_event["llm_context"] == previous["context"]
        ranks = {}
        for stage, key in (("fusion", "fused_candidates"), ("dedup", "deduplicated_candidates"),
                           ("rerank", "reranked_candidates"),
                           ("constraints", "constraint_candidates"),
                           ("evidence", "evidence_candidates")):
            row = next((event[key] for event in events if key in event), [])
            ranks[stage] = {str(page): next((i for i, candidate in enumerate(row, 1)
                                             if candidate["chunk_id"] == cid), None)
                            for page, cid in targets.items()}
        included = {str(page): next((entry["included_text"] for entry in entries
                                     if entry["chunk_id"] == cid), "")
                    for page, cid in targets.items()}
        record = {"arm": arm, "status": result.status.value, "reason": result.reason,
                  "ranks": ranks, "included": included,
                  "selected_ids": [entry["chunk_id"] for entry in entries],
                  "route_audit": route_audit,
                  "context": context_event["llm_context"], "entries": entries,
                  "condition_bindings": matcher.last_bindings if matcher else [],
                  "event_stages": [event["stage"] for event in events]}
        output.append(record)
        print(arm, ranks, {p: bool(v) for p, v in included.items()}, result.reason, flush=True)
    write(OUT / "static_arms.json", output)


if __name__ == "__main__":
    main()
