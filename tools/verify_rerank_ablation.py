"""Frozen-candidate reranking experiment; never changes application settings or Gold."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import EvidenceChunk
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import (
    GoldEvidenceMatch, EvidenceAlternative, EvidencePart,
)
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_gold_dataset(Path("tests/rag/fixtures/rag_gold_v3.jsonl"))
    gold = {c.case_id: c for c in dataset.cases}
    saved = [json.loads(line) for line in (args.source / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    run = json.loads((args.source / "run.json").read_text(encoding="utf-8"))
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    assert bundle.bundle_id == run["ai_bundle_id"]
    conn = store.connect_read_only(bundle.metadata_path)
    reranker = object.__new__(LocalBgeReranker)
    reranker.max_rank_drop = 3
    live = None
    if args.live:
        spec = ModelSpec(run["settings"]["reranker_model"], run["settings"]["reranker_revision"], "reranker")
        path = LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(spec)
        live = LocalBgeReranker(path, repo_id=spec.repo_id, revision=spec.revision)
    results = []
    started = time.monotonic()
    for record in saved:
        case = gold[record["id"]]
        assert case.question == record["question"]
        snapshots = record["candidate_stages"]["rerank"]
        # These are exactly the scored input candidates; ordering comes from the saved dedup stage.
        byid = {s["chunk_id"]: s for s in snapshots}
        original = [s for s in record["candidate_stages"]["deduplicated"] if s["chunk_id"] in byid]
        assert len(original) == len(snapshots)
        items = []
        for s in original:
            scored = byid[s["chunk_id"]]
            chunk = store.load_chunk(conn, s["chunk_id"])
            assert chunk is not None and chunk.text == scored["text"]
            item = EvidenceChunk(chunk=chunk)
            for key in ("lexical_rank", "vector_rank", "fused_rank", "fused_score", "rerank_score", "answer_support_score"):
                setattr(item, key, scored.get(key))
            items.append(item)
        mappings = []
        for m in record["gold_mappings"]:
            seq = int(m["evidence_id"].rsplit(":", 1)[1]) - 1
            evidence = (case.gold_evidence if m["kind"] == "positive" else case.hard_negatives)[seq]
            quote = str(evidence.get("quote") or "")
            mappings.append(GoldEvidenceMatch(
                evidence_id=m["evidence_id"], kind=m["kind"], file=m["file"], page=m["page"], quote=quote,
                chunk_ids=tuple(m["chunk_ids"]), confidence=m["confidence"],
                alternatives=tuple(EvidenceAlternative(a["alternative_id"], a["match_type"],
                    tuple(EvidencePart(**p) for p in a["parts"]), a["confirmed"]) for a in m.get("alternatives", [])),
            ))
        def rank(scores):
            return sorted(range(len(items)), key=lambda i: (-scores[i], -items[i].fused_score, items[i].chunk.stable_chunk_id))
        pure = [float(x.rerank_score) for x in items]
        combined = [.65 * x.rerank_score + .05 * retrieval_prior(x) + .30 * float(x.answer_support_score or 0) for x in items]
        orders = {"A_original": list(range(len(items))), "B_model": rank(pure), "C_mixed": rank(combined)}
        guarded = reranker._apply_rank_guard([items[i] for i in orders["C_mixed"]])
        orders["D_current"] = [items.index(x) for x in guarded]
        assert [items[i].chunk.stable_chunk_id for i in orders["D_current"]] == [s["chunk_id"] for s in snapshots]
        orders["C1_model_prior"] = rank([.65*x.rerank_score+.05*retrieval_prior(x) for x in items])
        orders["C2_model_support"] = rank([.65*x.rerank_score+.30*float(x.answer_support_score or 0) for x in items])
        orders["B1_model_guarded"] = [items.index(x) for x in reranker._apply_rank_guard([items[i] for i in orders["B_model"]])]
        def evaluate(order):
            rows = [byid[items[i].chunk.stable_chunk_id] for i in order]
            return {str(k): score_stage("rerank", rows[:k], mappings) for k in (1, 3, 5, 8, 20)}
        row = {"id": case.case_id, "question": case.question, "group_id": case.group_id,
               "cohort": "original18" if int(case.case_id[-3:]) <= 18 else "added32",
               "expected_behavior": case.expected_behavior,
               "arms": {arm: evaluate(order) for arm, order in orders.items()},
               "candidate_count": len(items), "candidates": [], "live_ms": 0}
        for key in ("gold_evidence_found", "gold_evidence_recall", "first_gold_rank", "hard_negative_found"):
            assert row["arms"]["D_current"]["20"][key] == record["stages"]["rerank"][key], (case.case_id, key)
        if live and items:
            t = time.monotonic()
            live.rerank(case.question, items, len(items), CancelToken())
            fresh = {x.chunk.stable_chunk_id: x.rerank_score for x in live.last_scored_candidates}
            row["live_ms"] = round((time.monotonic()-t)*1000)
            row["max_score_difference"] = max(abs(fresh[x.chunk.stable_chunk_id]-pure[i]) for i,x in enumerate(items))
            row["live_order_matches"] = rank([fresh[x.chunk.stable_chunk_id] for x in items]) == orders["B_model"]
            window_scores = [fresh[x.chunk.stable_chunk_id] for x in items]
            window_ms = 0
            for i, item in enumerate(items):
                tokenizer = live.tokenizer
                query_tokens = tokenizer.encode(case.question, add_special_tokens=False)
                text_tokens = tokenizer.encode(item.chunk.text, add_special_tokens=False)
                pair_len = len(query_tokens) + len(text_tokens) + tokenizer.num_special_tokens_to_add(pair=True)
                actual = tokenizer(case.question, item.chunk.text, truncation=True, max_length=512)
                candidate_row = {"chunk_id": item.chunk.stable_chunk_id, "file": item.chunk.source.file_path,
                                 "location": item.chunk.source.location_text, "pair_tokens": pair_len,
                                 "truncated": pair_len > 512, "model_score": pure[i],
                                 "live_score": fresh[item.chunk.stable_chunk_id], "support_score": item.answer_support_score,
                                 "combined_score": combined[i],
                                 "ranks": {arm: order.index(i)+1 for arm,order in orders.items()}}
                # Score all overlapping windows for long passages; no Gold is used to choose windows.
                if pair_len > 512:
                    budget = 512-len(query_tokens)-tokenizer.num_special_tokens_to_add(pair=True)
                    windows = [text_tokens[j:j+budget] for j in range(0, len(text_tokens), max(1,budget-64))]
                    passages = [tokenizer.decode(w, skip_special_tokens=True) for w in windows]
                    encoded = tokenizer([case.question]*len(passages), passages, padding=True, truncation=True, max_length=512, return_tensors="pt")
                    t = time.monotonic()
                    with live._torch.inference_mode():
                        scores = live._torch.sigmoid(live.model(**encoded).logits.reshape(-1)).cpu().tolist()
                    window_ms += round((time.monotonic()-t)*1000)
                    window_scores[i] = max(scores)
                    candidate_row["window_scores"] = scores
                    candidate_row["retained_text"] = tokenizer.decode(actual["input_ids"], skip_special_tokens=True)
                row["candidates"].append(candidate_row)
            row["window_ms"] = window_ms
            orders["E_windows_model"] = rank(window_scores)
            row["arms"]["E_windows_model"] = evaluate(orders["E_windows_model"])
            for i, c in enumerate(row["candidates"]):
                c["ranks"]["E_windows_model"] = orders["E_windows_model"].index(i)+1
        results.append(row)
        dump(args.out / "cases.json", results)
        print(case.case_id, "done", row["live_ms"], flush=True)
    conn.close()
    summary = {}
    for cohort in ("all", "original18", "added32"):
        selected = [r for r in results if (cohort == "all" or r["cohort"] == cohort) and r["expected_behavior"] != "refuse"]
        summary[cohort] = {}
        for arm in results[0]["arms"]:
            summary[cohort][arm] = {}
            for k in (1,3,5,8,20):
                scores = [r["arms"][arm][str(k)] for r in selected]
                summary[cohort][arm][str(k)] = {"n": len(scores), "hit": sum(s["gold_hit"] for s in scores),
                    "coverage": sum(s["gold_evidence_recall"] for s in scores)/len(scores),
                    "mrr": sum(s["reciprocal_rank"] for s in scores)/len(scores),
                    "negative_occurrences": sum(s["hard_negative_found"] for s in scores),
                    "miss_ids": [r["id"] for r in selected if not r["arms"][arm][str(k)]["gold_hit"]]}
    dump(args.out / "summary.json", summary)
    dump(args.out / "protocol.json", {"source_run": str(args.source), "source_sha256": hashlib.sha256((args.source/'cases.jsonl').read_bytes()).hexdigest(),
        "dataset_sha256": dataset.sha256, "bundle_id": bundle.bundle_id, "model": run["settings"]["reranker_model"],
        "revision": run["settings"]["reranker_revision"], "live": args.live, "wall_seconds": time.monotonic()-started,
        "scope": "fixed historical rerank inputs, no downstream constraints or quotas; all 50 reviewed cases; not held-out",
        "window_policy": "512 total tokens, 64 passage token overlap, max window score; only overlength passages",
        "current_order_reproduction": "exact for all cases"})
    print(json.dumps(summary["all"], ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
