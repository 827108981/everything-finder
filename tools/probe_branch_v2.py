"""Inspect native retrieval ranks for question-only branch queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from unittest.mock import patch

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from tools.rag_eval_common import ROOT, read, sha, write
from tools.validate_evidence_answer_flow import Experiment
from tools.branch_candidate_v2 import validate, PROMPT


OUT = ROOT / "reports/rag_post_gold_upstream_20261110_02"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
QUESTION_ID = "rag-biz-040"
FROZEN_PLAN = '{"branches":[{"condition_quote":"ALB、GLU 等小样本量项目重复性变差，而 ALT、AST 较稳定","inherited_quote":""},{"condition_quote":"所有项目都变差","inherited_quote":"重复性"}]}'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(OUT, 1)
    case = next(c for c in load_gold_dataset(GOLD, reviewed_only=False).cases if c.case_id == QUESTION_ID)
    assert exp.bundle.bundle_id == case.corpus_snapshot
    branches = validate(case.question, FROZEN_PLAN)
    assert len(branches) == 2
    write(OUT / "protocol.json", {
        "bundle_id": exp.bundle.bundle_id,
        "gold_sha256": sha(GOLD),
        "source_report_sha256": sha(ROOT / "reports/rag_gold_v4_3_retest_20261110/runs.jsonl"),
        "planner_prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
        "original_model_output": FROZEN_PLAN,
        "validated_branches": [vars(branch) for branch in branches],
        "kind": "read-only native retrieval diagnosis; no answer-model calls",
    })
    old = next(json.loads(line) for line in (ROOT / "reports/rag_gold_v4_3_retest_20261110/runs.jsonl").open(encoding="utf-8")
               if f'"id": "{QUESTION_ID}"' in line[:100] and '"repeat": 1' in line[:350])
    targets = {14: next(x["chunk_id"] for x in old["candidate_stages"]["fusion"]
                        if x["filename"] == "生化试剂应用及常见问题-SH-20190110.pptx" and x["slide_number"] == 14),
               16: next(x["chunk_id"] for x in old["candidate_stages"]["vector"]
                        if x["filename"] == "生化试剂应用及常见问题-SH-20190110.pptx" and x["slide_number"] == 16)}
    reports = []
    captured = {}

    class Probe(HybridRetriever):
        def retrieve(self, query, token, plan=None):
            fused = super().retrieve(query, token, plan)
            captured[query.text] = {
                "lexical": [candidate_snapshot(x) for x in self.lexical.retrieve(query, token, plan)],
                "vector": [candidate_snapshot(x) for x in self.vector.retrieve(query, token, plan)],
                "fused": [candidate_snapshot(x) for x in fused],
            }
            return fused

    with patch("local_full_text_search.rag.answer_service.HybridRetriever", Probe):
        queries = tuple(dict.fromkeys((case.question, *[branch.query_text for branch in branches],
                                       *[branch.condition_quote for branch in branches])))
        for query_text in queries:
            run = exp.native(replace(case, question=query_text), None, CancelToken(), retrieve=True)
            assert not run.get("error"), run.get("error")
            stage = captured[query_text]
            ranks = {str(page): {
                kind: next((i for i, item in enumerate(stage[kind], 1)
                            if item["chunk_id"] == chunk_id), None)
                for kind in ("lexical", "vector", "fused")}
                for page, chunk_id in targets.items()}
            reports.append({"question": query_text, "ranks": ranks,
                            "counts": {name: len(value) for name, value in stage.items()},
                            "native_context": run["context"]})
            print(query_text[:45], ranks, flush=True)
    write(OUT / "route_ranks.json", {"targets": targets, "rows": reports})


if __name__ == "__main__":
    main()
