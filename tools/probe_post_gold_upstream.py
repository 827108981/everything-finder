"""Read-only oracle probe: can the unchanged downstream retain a missing page?"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.reranking.support import compute_answer_support
from local_full_text_search.rag.trace import candidate_snapshot
from tools.rag_eval_common import ROOT, read, sha, write
from tools.validate_evidence_answer_flow import Experiment
from tools.validate_post_gold_selection import restore


BASE = ROOT / "reports/rag_gold_v4_3_retest_20261110"
OUT = ROOT / "reports/rag_post_gold_upstream_20261110_01"
TARGETS = {
    "rag-biz-019": ("化学发光定标处理思路分析.pptx", 13, 14),
    "rag-biz-040": ("生化试剂应用及常见问题-SH-20190110.pptx", 14, 16),
}


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_chain(exp, question, fused):
    s = exp.settings
    query = RagQuery(
        text=question, lexical_top_k=s.rag_lexical_top_k,
        vector_top_k=s.rag_vector_top_k, fusion_top_k=s.rag_fusion_top_k,
        reranker_top_k=s.rag_reranker_top_k, final_top_k=s.rag_final_top_k,
    )
    plan = QueryInterpreter().interpret(question)
    dedup = CandidateDeduplicator(jaccard_threshold=s.rag_dedup_jaccard_threshold)
    deduped = list(dedup.deduplicate(fused, limit=query.fusion_top_k).candidates)
    rerank_input = deduped[:query.reranker_top_k]
    for item in deduped:
        item.answer_support_score, item.answer_support_reasons = compute_answer_support(plan, item)
    exp.ranker.rerank(question, rerank_input, query.reranker_top_k, CancelToken())
    scored = list(exp.ranker.last_scored_candidates)
    constrained = ConstraintMatcher().evaluate(plan, scored)
    policy = EvidencePolicy(
        min_vector_score=s.rag_min_vector_score,
        min_rerank_score=s.rag_min_rerank_score,
        max_per_document=s.rag_max_chunks_per_document,
    )
    evidence = policy.evaluate(query, constrained, plan=plan)
    context = ContextBuilder().build(evidence.accepted, max_chars=s.rag_context_max_chars)
    return {
        "fused": [candidate_snapshot(item) for item in fused],
        "deduplicated": [candidate_snapshot(item) for item in deduped],
        "rerank_input": [candidate_snapshot(item) for item in rerank_input],
        "reranked": [candidate_snapshot(item) for item in scored],
        "constraints": [candidate_snapshot(item) for item in constrained],
        "evidence": [candidate_snapshot(item, selected=item in evidence.accepted,
                                         reason=dict(evidence.rejected).get(item.chunk.stable_chunk_id, "selected"))
                     for item in constrained],
        "decision_reason": evidence.reason,
        "context": context.text, "entries": list(context.entries),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    exp = Experiment(OUT, 1)
    assert exp.bundle.bundle_id == read(BASE / "protocol.json")["bundle"]
    write(OUT / "protocol.json", {
        "bundle_id": exp.bundle.bundle_id,
        "method": "oracle insert of indexed missing page into fixed fusion budget; no runtime algorithm or answer generation",
        "upstream_source": str(BASE / "runs.jsonl"),
        "baseline_sha256": sha(BASE / "runs.jsonl"),
        "settings_sha256": sha(SETTINGS_PATH),
        "source_page_ids": TARGETS,
        "not_an_adoption_candidate": True,
    })
    previous = {row["id"]: row for row in rows(BASE / "runs.jsonl")
                if row["repeat"] == 1 and row["id"] in TARGETS}
    with exp.store.connect_read_only(exp.bundle.metadata_path) as conn:
        cache = {}
        for case_id, (name, anchor_page, target_page) in TARGETS.items():
            old = previous[case_id]
            plan = QueryInterpreter().interpret(old["question"])
            fused = [restore(snap, plan, exp.store, conn, cache)
                     for snap in old["candidate_stages"]["fusion"]]
            anchor = next(item for item in fused if item.chunk.source.filename == name
                          and item.chunk.source.slide_number == anchor_page)
            target = next((item for item in fused if item.chunk.source.filename == name
                           and item.chunk.source.slide_number == target_page), None)
            target_in_original_fusion = target is not None
            if target is None:
                snap = next(
                    value for value in old["candidate_stages"]["vector"]
                    if value["filename"] == name and value["slide_number"] == target_page
                )
                target = restore(snap, plan, exp.store, conn, cache)
                target.fused_score = 1.0 / (exp.settings.rag_rrf_k + target.vector_rank)
                target.fused_rank = len(fused) + 1
            new = [item for item in fused if item.chunk.stable_chunk_id != target.chunk.stable_chunk_id]
            new.insert(new.index(anchor) + 1, target)
            new = new[:len(fused)]
            for rank, item in enumerate(new, 1):
                item.fused_rank = rank
            result = run_chain(exp, old["question"], copy.deepcopy(new))
            target_id = target.chunk.stable_chunk_id
            stage_rank = {}
            for stage in ("fused", "deduplicated", "rerank_input", "reranked", "constraints", "evidence"):
                stage_rank[stage] = next((i for i, item in enumerate(result[stage], 1)
                                         if item["chunk_id"] == target_id), None)
            included = next((entry for entry in result["entries"] if entry["chunk_id"] == target_id), None)
            record = {
                "id": case_id, "question": old["question"],
                "target_file": name, "target_page": target_page,
                "target_id": target_id,
                "target_in_original_fusion": target_in_original_fusion,
                "oracle_added_target": True,
                "original_context": old["context"],
                "baseline_fused_anchor_rank": next(i for i, item in enumerate(fused, 1) if item.chunk.stable_chunk_id == anchor.chunk.stable_chunk_id),
                "stage_rank": stage_rank,
                "actual_included_target": included,
                "context": result["context"], "entries": result["entries"],
                "decision_reason": result["decision_reason"],
                "source_continuity_claim": "not established by page adjacency alone",
            }
            write(OUT / f"probe_{case_id}.json", record)
            print(case_id, stage_rank, bool(included), result["decision_reason"], flush=True)


if __name__ == "__main__":
    main()
