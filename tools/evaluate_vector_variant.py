"""在固定 AI bundle 上比较生产向量索引与隔离向量索引的检索结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.settings_service import SettingsService


def _evidence_ranks(case: dict, rank_by_chunk: dict[str, int], key: str) -> list[int | None]:
    return [
        min((rank_by_chunk[chunk_id] for chunk_id in evidence["mapping"]["chunk_ids"]
             if chunk_id in rank_by_chunk), default=None)
        for evidence in case[key]
    ]


def _metrics(cases: list[dict]) -> dict:
    scored = [case for case in cases if case["expected_behavior"] != "refuse"]
    firsts = [case["first_gold_rank"] for case in scored]
    return {
        "scored_cases": len(scored),
        "hit_at_1": sum(rank is not None and rank <= 1 for rank in firsts),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in firsts),
        "hit_at_50": sum(rank is not None and rank <= 50 for rank in firsts),
        "mrr": sum(1 / rank if rank else 0 for rank in firsts) / max(1, len(scored)),
        "mean_evidence_coverage": sum(
            sum(rank is not None for rank in case["gold_ranks"]) / max(1, len(case["gold_ranks"]))
            for case in scored
        ) / max(1, len(scored)),
        "hard_negative_in_top50": sum(
            sum(rank is not None for rank in case["hard_negative_ranks"])
            for case in cases
        ),
        "total_retrieval_ms": sum(case["retrieval_ms"] for case in cases),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--candidate-vector-root", type=Path, required=True)
    parser.add_argument("--report-detail", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.report_detail]
    store = RagIndexStore(args.generation_base_path.resolve().parent / "rag")
    source = store.active()
    if source.bundle_id != args.expected_bundle_id:
        raise ValueError("当前 bundle 与指定值不一致")
    settings = SettingsService().load()
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(
            settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding",
        )),
        repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    base_policy = VectorUnitPolicy.from_settings(settings)
    if not base_policy.include_source_context:
        raise ValueError("本机当前设置不是带来源上下文的生产基线")
    baseline = VectorIndexStore(store.root).active(
        source.bundle_id, embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision,
        embedding_dimension=embedding.dimension,
        policy_fingerprint=base_policy.fingerprint,
    )
    settings.rag_vector_unit_include_source_context = False
    candidate_policy = VectorUnitPolicy.from_settings(settings)
    candidate = VectorIndexStore(args.candidate_vector_root.resolve()).active(
        source.bundle_id, embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision,
        embedding_dimension=embedding.dimension,
        policy_fingerprint=candidate_policy.fingerprint,
    )
    vector_stores = {
        "baseline": ChromaVectorStore(baseline.chroma_path),
        "body_only": ChromaVectorStore(candidate.chroma_path),
    }
    indexes = {"baseline": baseline, "body_only": candidate}
    try:
        dataset_results = []
        for report in reports:
            if report["run"]["ai_bundle_id"] != source.bundle_id:
                raise ValueError("评测报告与当前 bundle 不一致")
            dataset = {"dataset_path": report["run"]["dataset_path"], "cases": []}
            for case in report["cases"]:
                if case["query"]["semantic_query"] != case["question"]:
                    raise ValueError(f"{case['id']} 的语义查询与原问题不同")
                query = RagQuery(
                    text=case["question"], vector_top_k=50,
                    include_ocr=True,
                )
                compared = {}
                for name in ("baseline", "body_only"):
                    retriever = VectorRetriever(
                        store, source, vector_stores[name], embedding,
                        backend="unit_v2", unit_index=indexes[name],
                        oversample_factor=8,
                    )
                    started = time.monotonic()
                    results = retriever.retrieve(query, CancelToken())
                    rank_by_chunk = {
                        result.chunk.stable_chunk_id: int(result.vector_rank)
                        for result in results
                    }
                    gold_ranks = _evidence_ranks(case, rank_by_chunk, "gold_evidence")
                    negative_ranks = _evidence_ranks(case, rank_by_chunk, "hard_negatives")
                    compared[name] = {
                        "gold_ranks": gold_ranks,
                        "hard_negative_ranks": negative_ranks,
                        "first_gold_rank": min(
                            (rank for rank in gold_ranks if rank is not None), default=None,
                        ),
                        "retrieval_ms": int((time.monotonic() - started) * 1000),
                        "candidate_count": len(results),
                        "raw_unit_depth": retriever.route_diagnostics[0]["requested_unit_depth"],
                    }
                expected = [
                    min(evidence["stages"]["vector"], default=None)
                    for evidence in case["gold_evidence"]
                ]
                if compared["baseline"]["gold_ranks"] != expected:
                    raise AssertionError(
                        f"{case['id']} 基线复现失败："
                        f"{compared['baseline']['gold_ranks']} != {expected}"
                    )
                dataset["cases"].append({
                    "id": case["id"], "question": case["question"],
                    "expected_behavior": case["expected_behavior"],
                    "baseline": compared["baseline"], "body_only": compared["body_only"],
                })
                print(json.dumps({
                    "id": case["id"],
                    "baseline": compared["baseline"]["gold_ranks"],
                    "body_only": compared["body_only"]["gold_ranks"],
                }, ensure_ascii=False), flush=True)
            dataset["summary"] = {
                name: _metrics([case[name] | {"expected_behavior": case["expected_behavior"]}
                                for case in dataset["cases"]])
                for name in ("baseline", "body_only")
            }
            dataset_results.append(dataset)
        output = {
            "source_bundle_id": source.bundle_id,
            "model_id": embedding.model_id,
            "model_revision": embedding.model_revision,
            "baseline_index_fingerprint": baseline.index_fingerprint,
            "body_only_index_fingerprint": candidate.index_fingerprint,
            "baseline_unit_count": baseline.unit_count,
            "body_only_unit_count": candidate.unit_count,
            "datasets": dataset_results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"output": str(args.out), "summaries": [item["summary"] for item in dataset_results]},
                         ensure_ascii=False), flush=True)
    finally:
        for vector_store in vector_stores.values():
            vector_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
