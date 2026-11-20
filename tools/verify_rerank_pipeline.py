"""Run the production retrieval pipeline twice, sharing verified raw model scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from tools import run_rag_experiment as experiment
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--verified-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--downstream", action="store_true", help="Isolate conditional sorting and document quotas on the pure-model baseline")
    parser.add_argument("--only-arm", default="", help="Run one named experimental arm")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    original = [json.loads(line) for line in (args.source / "cases.jsonl").read_text(encoding="utf-8").splitlines()]
    verified = json.loads((args.verified_scores / "cases.json").read_text(encoding="utf-8"))
    protocol = json.loads((args.verified_scores / "protocol.json").read_text(encoding="utf-8"))
    assert len(original) == len(verified) == 50
    assert all(row.get("max_score_difference") == 0 and row.get("live_order_matches") for row in verified)
    cache = {}
    for old, live in zip(original, verified, strict=True):
        assert old["id"] == live["id"] and old["question"] == live["question"]
        texts = {c["chunk_id"]: c["text"] for c in old["candidate_stages"]["rerank"]}
        for candidate in live["candidates"]:
            cache[(old["question"], candidate["chunk_id"], digest(texts[candidate["chunk_id"]]))] = candidate["live_score"]
    stats = {}

    class CachedReranker(LocalBgeReranker):
        def __init__(self, model_path, **kwargs):
            assert kwargs["repo_id"] == protocol["model"] and kwargs["revision"] == protocol["revision"]
            assert kwargs["max_length"] == 512
            self.model_id = kwargs["repo_id"]
            self.revision = kwargs["revision"]
            self.max_rank_drop = kwargs["max_rank_drop"]
            self.cross_encoder_weight = kwargs["cross_encoder_weight"]
            self.retrieval_prior_weight = kwargs["retrieval_prior_weight"]
            self.answer_support_weight = kwargs["answer_support_weight"]
            self.last_scored_candidates = ()
            self.fallback = None
            self.model_path, self.kwargs = model_path, kwargs

        def rerank(self, question, candidates, top_n, token):
            missing = [c for c in candidates if (question, c.chunk.stable_chunk_id, digest(c.chunk.text)) not in cache]
            if missing:
                if self.fallback is None:
                    self.fallback = LocalBgeReranker(self.model_path, **self.kwargs)
                self.fallback.rerank(question, missing, len(missing), token)
                for c in self.fallback.last_scored_candidates:
                    cache[(question, c.chunk.stable_chunk_id, digest(c.chunk.text))] = c.rerank_score
            stats["fresh_pairs"] += len(missing)
            stats["cached_pairs"] += len(candidates)-len(missing)
            for c in candidates:
                token.throw_if_cancelled()
                c.rerank_score = cache[(question, c.chunk.stable_chunk_id, digest(c.chunk.text))]
                c.rerank_score_combined = (
                    self.cross_encoder_weight*c.rerank_score
                    + self.retrieval_prior_weight*retrieval_prior(c)
                    + self.answer_support_weight*float(c.answer_support_score or 0)
                )
            ordered = sorted(candidates, key=lambda c: (-c.rerank_score_combined, -float(c.fused_score), c.chunk.stable_chunk_id))
            ordered = self._apply_rank_guard(ordered)
            self.last_scored_candidates = tuple(ordered)
            return self._apply_document_diversity(ordered, max(1, int(top_n)))

    experiment.LocalBgeReranker = CachedReranker
    production_service = experiment.AnswerService
    variant = {"preserve_order": False, "max_per_document": 3}

    class PreserveOrderMatcher(ConstraintMatcher):
        def evaluate(self, plan, candidates):
            super().evaluate(plan, candidates)
            # Keep all original judgements, conflicts and applicability labels;
            # change only the order, not the downstream hard gates.
            return tuple(candidates)

    class ObjectNeutralSortMatcher(ConstraintMatcher):
        def evaluate(self, plan, candidates):
            ordered = super().evaluate(plan, candidates)
            if not any(c.kind.value == "object" for c in plan.constraints):
                return ordered
            def signals(item):
                judgements = [j for j in item.constraint_decision.judgements if j.constraint.kind.value != "object"]
                hard = [j for j in judgements if j.constraint.is_hard]
                soft = [j for j in judgements if j.constraint.polarity.value == "affirmed" and j.constraint.strength.value == "soft"]
                unknown = any(j.status.value == "unknown" for j in hard)
                score = (2*sum(j.status.value == "match" for j in hard)
                         + sum(j.status.value == "match" for j in soft)
                         - sum(j.status.value == "conflict" for j in soft))
                return unknown, score
            if not any(c.is_hard for c in plan.constraints):
                return tuple(sorted(candidates, key=lambda x: (
                    -signals(x)[1], -float(x.rerank_score_combined or 0), x.chunk.stable_chunk_id)))
            return tuple(sorted(candidates, key=lambda x: (
                not x.constraint_decision.accepted, signals(x)[0], -signals(x)[1])))

    def service_with_variant(*positional, **kwargs):
        if variant.get("neutral_object_sort"):
            kwargs["constraint_matcher"] = ObjectNeutralSortMatcher()
        elif variant["preserve_order"]:
            kwargs["constraint_matcher"] = PreserveOrderMatcher()
        settings = positional[6]
        kwargs["evidence_policy"] = EvidencePolicy(
            min_vector_score=settings.rag_min_vector_score,
            min_rerank_score=settings.rag_min_rerank_score,
            max_per_document=variant["max_per_document"],
        )
        return production_service(*positional, **kwargs)

    if args.downstream:
        experiment.AnswerService = service_with_variant
    common = {
        "generation_base_path": "E:/everythingfinder-db/search_index.db",
        "expected_bundle_id": protocol["bundle_id"],
        "rag_lexical_backend": "word_v2", "rag_vector_backend": "unit_v2",
        "rag_lexical_multi_route_enabled": False, "rag_vector_focused_query_enabled": False,
        "rag_lexical_top_k": 50, "rag_vector_top_k": 50, "rag_fusion_top_k": 30,
        "rag_reranker_top_k": 20, "rag_final_top_k": 8, "rag_context_max_chars": 12000,
        "rag_answer_review_enabled": False,
    }
    arms = (("current", (0.65, 0.05, 0.30, 3)), ("pure", (1.0, 0.0, 0.0, 0)))
    if args.downstream:
        arms = tuple((name, (1.0, 0.0, 0.0, 0)) for name in (
            "pure_baseline", "preserve_order", "quota8", "both_diagnostic",
            "object_neutral_sort",
        ))
    if args.only_arm:
        arms = tuple(x for x in arms if x[0] == args.only_arm)
        if not arms:
            raise ValueError("Unknown experimental arm")
    for arm, weights in arms:
        variant["preserve_order"] = arm in ("preserve_order", "both_diagnostic")
        variant["max_per_document"] = 8 if arm in ("quota8", "both_diagnostic") else 3
        variant["neutral_object_sort"] = arm == "object_neutral_sort"
        config = dict(common)
        config.update(zip(("rag_rerank_cross_encoder_weight", "rag_rerank_retrieval_prior_weight", "rag_rerank_answer_support_weight", "rag_rerank_max_rank_drop"), weights))
        path = args.out / (arm+".json")
        path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        stats.clear()
        stats.update(cached_pairs=0, fresh_pairs=0)
        sys.argv = ["run_rag_experiment", "--cases", "tests/rag/fixtures/rag_gold_v3.jsonl", "--experiment-config", str(path),
                    "--mode", "retrieval-only", "--out", str(args.out/arm)]
        print("START", arm, flush=True)
        code = experiment.main()
        assert code == 0
        runs = sorted((args.out/arm).glob("*/run.json"))
        assert len(runs) == 1
        (runs[0].parent/"score_cache.json").write_text(json.dumps({
            **stats, "verified_score_source": str(args.verified_scores),
            "key": "exact question, chunk ID, SHA256 of full candidate text; pinned model revision and length",
            "scope": "all production retrieval and downstream stages rerun; raw reranker scores reused when identical",
            "performance_warning": "cached run timing is not uncached production latency",
        }, indent=2), encoding="utf-8")
        if args.downstream:
            (runs[0].parent/"downstream_variant.json").write_text(json.dumps({
                **variant, "hard_constraint_judgements": "unchanged",
                "hard_conflict_gates": "unchanged", "critical_condition_confirmation": "unchanged",
                "clarification_policy": "unchanged", "final_top_k": 8, "context_max_chars": 12000,
            }, indent=2), encoding="utf-8")
        print("DONE", arm, stats, flush=True)


if __name__ == "__main__":
    main()
