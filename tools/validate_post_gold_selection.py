"""Replay the frozen v4.3 candidate pool and test a source-bound selector."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evidence_policy import EvidenceDecision, EvidencePolicy
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.models import (
    CandidateDecision, ConstraintJudgement, ConstraintStatus, EvidenceChunk, RagQuery,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from tools.rag_eval_common import ROOT, read, sha, write
from tools.source_bound_selection import propose_selection
from tools.source_bound_selection import SELECTOR_PROMPT, REMOVAL_PROMPT


BASE = ROOT / "reports/rag_gold_v4_3_retest_20261110"
OUT = ROOT / "reports/rag_post_gold_selection_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
INDEX = Path("E:/everythingfinder-db/rag")


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def snapshots():
    for row in rows(BASE / "runs.jsonl"):
        if row["repeat"] == 1:
            yield row


def restore(snap, plan, store, conn, cache):
    def chunk(cid):
        if cid not in cache:
            cache[cid] = store.load_chunk(conn, cid)
        assert cache[cid] is not None
        return cache[cid]

    ch = chunk(snap["chunk_id"])
    assert ch.text == snap["text"]
    fields = (
        "lexical_rank", "lexical_score", "lexical_score_kind", "vector_rank",
        "vector_score", "fused_rank", "fused_score", "rerank_score",
        "rerank_score_combined", "answer_support_score", "constraint_score",
    )
    item = EvidenceChunk(chunk=ch, **{key: snap[key] for key in fields if key in snap})
    item.member_chunk_ids = tuple(snap.get("member_chunk_ids", []))
    item.sources = tuple(chunk(source["chunk_id"]).source for source in snap.get("sources", []))
    decision = snap.get("constraint_decision")
    if decision:
        judgements = []
        for value in decision["judgements"]:
            condition = next(
                part for part in plan.constraints
                if part.kind.value == value["kind"]
                and part.value == value["value"]
                and part.source_text == value["source_text"]
                and part.strength.value == value["strength"]
                and part.polarity.value == value["polarity"]
            )
            judgements.append(ConstraintJudgement(
                condition, ConstraintStatus(value["status"]),
                value["reason"], value["matched_text"],
            ))
        item.constraint_decision = CandidateDecision(
            ch.stable_chunk_id, decision["accepted"], decision["reason"],
            tuple(judgements),
            **{key: decision[key] for key in (
                "required_match_count", "required_unknown_count",
                "required_conflict_count", "soft_match_count",
                "soft_conflict_count", "support_status",
            )},
            support_spans=tuple(decision["support_spans"]),
        )
    return item


def reproduce(settings, store, conn):
    policy = EvidencePolicy(
        min_vector_score=settings.rag_min_vector_score,
        min_rerank_score=settings.rag_min_rerank_score,
        max_per_document=settings.rag_max_chunks_per_document,
    )
    builder = ContextBuilder()
    cache = {}
    result = []
    for old in snapshots():
        plan = QueryInterpreter().interpret(old["question"])
        query = RagQuery(text=old["question"], final_top_k=settings.rag_final_top_k)
        candidates = [restore(s, plan, store, conn, cache)
                      for s in old["candidate_stages"].get("constraints", [])]
        decision = policy.evaluate(query, candidates, plan=plan)
        context = builder.build(decision.accepted, max_chars=settings.rag_context_max_chars)
        assert context.text == old["context"], f"baseline context mismatch: {old['id']}"
        assert list(context.entries) == old["entries"], f"baseline entries mismatch: {old['id']}"
        result.append((old, plan, query, candidates, decision, context))
    assert len(result) == 91
    return result, policy, builder


def report_protocol(settings, bundle, reproduced):
    fixture_files = [
        GOLD, ROOT / "tests/rag/fixtures/rag_gold_v4_3.jsonl",
        ROOT / "tests/rag/fixtures/rag_acceptance_v1_review.json",
        ROOT / "tests/rag/fixtures/rag_row_holdout_v1_review.json",
        ROOT / "tests/rag/fixtures/rag_row_holdout_v2_review.json",
        BASE / "runs.jsonl", BASE / "protocol.json", SETTINGS_PATH,
        store_manifest := Path(bundle.metadata_path), Path(__file__),
    ]
    protocol = {
        "bundle_id": bundle.bundle_id,
        "model": settings.rag_api_model,
        "baseline_rows": len(reproduced),
        "formal_reviewed": 48,
        "excluded_from_formal": ["rag-biz-007", "rag-biz-024"],
        "same_upstream": True,
        "single_variable": "final evidence selection only",
        "candidate_version": "source-bound-additive-v1",
        "max_chunks": settings.rag_final_top_k,
        "max_chars": settings.rag_context_max_chars,
        "baseline_exact_context_and_entries": True,
        "first_static_gate": "006 and 023 needed facts in actual text; no confirmed lost required fact or wrong-object support across 91 rows",
        "next_gate": "real AnswerService paired answer, then frozen fresh independent questions and full regression",
        "prior_counterexample_cases": ["rag-biz-015", "rag-biz-022"],
        "prior_frozen_candidates": "row supplement, grouped MMR, quota-free and scoped-branch prototypes remain disabled",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_sha256": {str(path): sha(path) for path in fixture_files},
        "index_metadata_path": str(store_manifest),
    }
    write(OUT / "protocol.json", protocol)
    (OUT / "working_tree_before.diff").write_text(
        subprocess.check_output(["git", "diff", "--binary"], text=True, errors="replace"),
        encoding="utf-8",
    )
    (OUT / "working_tree_before.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True),
        encoding="utf-8",
    )


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--baseline-only", action="store_true")
    args.add_argument("--targets-only", action="store_true")
    args.add_argument("--workers", type=int, default=4)
    args.add_argument("--resume", action="store_true")
    opts = args.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    store = RagIndexStore(INDEX)
    bundle = store.active()
    assert bundle.bundle_id == read(BASE / "protocol.json")["bundle"]
    with store.connect_read_only(bundle.metadata_path) as conn:
        reproduced, policy, builder = reproduce(settings, store, conn)
        if not (OUT / "protocol.json").exists():
            report_protocol(settings, bundle, reproduced)
        write(OUT / "baseline_replay.json", {
            "bundle_id": bundle.bundle_id,
            "cases": len(reproduced),
            "exact_context_and_entries": len(reproduced),
            "by_cohort": dict(Counter(old["cohort"] for old, *_ in reproduced)),
        })
        if opts.baseline_only:
            print("BASELINE", len(reproduced), "exact", flush=True)
            return
        key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
        if not key:
            raise RuntimeError("saved model credential unavailable")
        gateway = OpenAICompatibleGateway(max_transient_retries=2)
        out_path = OUT / ("targets_v3.jsonl" if opts.targets_only else "static_v2.jsonl")
        prior = rows(out_path) if opts.resume and out_path.exists() else []
        if out_path.exists() and not opts.resume and not opts.targets_only:
            raise FileExistsError(f"Use --resume after interruption: {out_path}")
        done = {record["id"] for record in prior}
        selected_rows = [row for row in reproduced
                         if (not opts.targets_only or row[0]["id"] in {"rag-biz-006", "rag-biz-023", "rag-biz-028"})
                         and row[0]["id"] not in done]
        if not opts.targets_only and not opts.resume:
            write(OUT / "candidate_protocol_v2.json", {
                "bundle_id": bundle.bundle_id,
                "cohorts": dict(Counter(row[0]["cohort"] for row in reproduced)),
                "fixed_upstream_and_baseline_replay": True,
                "candidate_prompt_sha256": hashlib.sha256(SELECTOR_PROMPT.encode()).hexdigest(),
                "removal_prompt_sha256": hashlib.sha256(REMOVAL_PROMPT.encode()).hexdigest(),
                "candidate_code_sha256": sha(ROOT / "tools/source_bound_selection.py"),
                "harness_sha256": sha(__file__),
                "settings_sha256": sha(SETTINGS_PATH),
                "gold_sha256": sha(GOLD),
                "baseline_runs_sha256": sha(BASE / "runs.jsonl"),
                "upstream_candidates": "replayed first-round constraints-stage snapshots, no retrieval/reranking recomputation",
                "candidate_fixed_before_full_static": True,
                "revision_reason": "First full static replay found 028 would drop a directly relevant low-signal troubleshooting table in favor of generic board diagnostics.",
            })
        chunk_rows = load_chunk_rows(conn)
        cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
        for path in (ROOT / "tests/rag/fixtures").glob("rag_*_review.json"):
            if path == GOLD or path.name.startswith("rag_gold_"):
                continue
            for case in load_gold_dataset(path, reviewed_only=False).cases:
                cases[case.case_id] = case
        old_independent = ROOT / "reports/rag_evidence_answer_native_recheck_20261109/acceptance_review.json"
        for case in load_gold_dataset(old_independent, reviewed_only=False).cases:
            cases[case.case_id] = case
        mappings = {case_id: map_case_evidence(case, chunk_rows)
                    for case_id, case in cases.items() if case_id in {row[0]["id"] for row in selected_rows}}

        def work(record):
            old, plan, query, candidates, baseline, old_context = record
            result = propose_selection(
                query, candidates, plan, baseline=baseline, policy=policy,
                builder=builder, max_chars=settings.rag_context_max_chars,
                gateway=gateway, model=settings.rag_api_model,
                api_base_url=settings.rag_api_base_url,
                api_style=settings.rag_api_style, api_key=key,
                timeout_seconds=settings.rag_api_timeout_seconds,
                token=CancelToken(),
            )
            context = builder.build(result.decision.accepted, max_chars=settings.rag_context_max_chars)
            by_id = {item.chunk.stable_chunk_id: item for item in candidates}
            stage = [
                {**candidate_snapshot(by_id[entry["chunk_id"]], selected=True),
                 "included_text": entry["included_text"], "truncated": entry["truncated"]}
                for entry in context.entries
            ]
            before_stage = [
                {**candidate_snapshot(by_id[entry["chunk_id"]], selected=True),
                 "included_text": entry["included_text"], "truncated": entry["truncated"]}
                for entry in old_context.entries
            ]
            before_score = score_stage("context", before_stage, mappings[old["id"]])
            after_score = score_stage("context", stage, mappings[old["id"]])
            rec = {
                "id": old["id"], "cohort": old["cohort"], "question": old["question"],
                "selection_reason": result.reason, "raw_selector": result.raw,
                "removal_audit": result.removal_audit,
                "checked_support": result.checked,
                "before_ids": [entry["chunk_id"] for entry in old_context.entries],
                "after_ids": [entry["chunk_id"] for entry in context.entries],
                "added": [entry["chunk_id"] for entry in context.entries
                          if entry["chunk_id"] not in {e["chunk_id"] for e in old_context.entries}],
                "removed": [entry["chunk_id"] for entry in old_context.entries
                            if entry["chunk_id"] not in {e["chunk_id"] for e in context.entries}],
                "before_context_sha256": hashlib.sha256(old_context.text.encode()).hexdigest(),
                "after_context": context.text, "after_entries": list(context.entries),
                "before_score": before_score, "after_score": after_score,
                "lost_gold": sorted(set(before_score["gold_completion_ranks"]) - set(after_score["gold_completion_ranks"])),
                "gained_gold": sorted(set(after_score["gold_completion_ranks"]) - set(before_score["gold_completion_ranks"])),
                "new_negative": sorted(set(after_score["hard_negative_ids"]) - set(before_score["hard_negative_ids"])),
            }
            return rec

        output = list(prior)
        with ThreadPoolExecutor(max_workers=1 if opts.targets_only else opts.workers) as pool:
            futures = [pool.submit(work, record) for record in selected_rows]
            for future in as_completed(futures):
                rec = future.result()
                output.append(rec)
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(rec["id"], rec["selection_reason"], len(rec["added"]),
                      len(rec["removed"]), len(rec["lost_gold"]), flush=True)
        print("SELECTED", len(output), flush=True)


if __name__ == "__main__":
    main()
