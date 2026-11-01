"""在基线向量 Top-N 候选与 Gold 组成的固定池中测试原子事实向量排名。

固定候选池能验证 Gold 的相似度提升是否会被竞争候选同步提升抵消；
因为没有覆盖全语料，本结果不能解释为正式 Recall 或全库名次。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.services.settings_service import SettingsService
from tools.probe_atomic_fact_gold import _embedding_text, atomic_facts


def _mapped_chunk_ids(evidence_rows: list[dict]) -> set[str]:
    return {
        str(chunk_id)
        for evidence in evidence_rows
        for chunk_id in evidence["mapping"]["chunk_ids"]
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--report-detail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--pool-depth", type=int, default=50)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    report = json.loads(args.report_detail.read_text(encoding="utf-8"))
    selected = [case for case in report["cases"] if case["id"] in set(args.case_id)]
    missing = sorted(set(args.case_id) - {case["id"] for case in selected})
    if missing:
        raise ValueError(f"报告中没有题目：{', '.join(missing)}")

    settings = SettingsService().load()
    store = RagIndexStore(args.generation_base_path.resolve().parent / "rag")
    source = store.active()
    if source.bundle_id != args.expected_bundle_id:
        raise ValueError("当前 AI bundle 与实验指定值不一致")
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(
            settings.rag_embedding_repo_id,
            settings.rag_embedding_revision,
            "embedding",
        )),
        repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    source_db = store.connect_read_only(source.metadata_path)
    try:
        results = []
        for case in selected:
            baseline_candidates = case["stage_data"]["vector"]["candidates"][:args.pool_depth]
            baseline_ids = [str(item["chunk_id"]) for item in baseline_candidates]
            gold_ids = _mapped_chunk_ids(case["gold_evidence"])
            negative_ids = _mapped_chunk_ids(case["hard_negatives"])
            pool_ids = list(dict.fromkeys([*baseline_ids, *sorted(gold_ids), *sorted(negative_ids)]))

            parent_fact_texts: dict[str, list[str]] = {}
            parent_facts: dict[str, list[str]] = {}
            for chunk_id in pool_ids:
                chunk = store.load_chunk(source_db, chunk_id)
                if chunk is None:
                    raise ValueError(f"候选父 chunk 不存在：{chunk_id}")
                facts = atomic_facts(
                    chunk.text,
                    embedding.count_tokens,
                    min_tokens=args.min_tokens,
                    max_tokens=args.max_tokens,
                )
                parent_facts[chunk_id] = facts
                parent_fact_texts[chunk_id] = [_embedding_text(chunk, fact) for fact in facts]

            unique_texts = list(dict.fromkeys(
                text for texts in parent_fact_texts.values() for text in texts
            ))
            ordered_texts = sorted(unique_texts, key=embedding.count_tokens)
            encoded = embedding.embed_documents(ordered_texts)
            vector_by_text = dict(zip(ordered_texts, encoded, strict=True))
            query_vector = np.asarray(embedding.embed_query(case["question"]), dtype=np.float32)
            ranked = []
            for chunk_id in pool_ids:
                scores = [
                    float(np.dot(query_vector, np.asarray(vector_by_text[text])))
                    for text in parent_fact_texts[chunk_id]
                ]
                best_index = int(np.argmax(scores))
                ranked.append({
                    "chunk_id": chunk_id,
                    "score": scores[best_index],
                    "best_fact": parent_facts[chunk_id][best_index],
                    "fact_count": len(scores),
                    "in_baseline_top_n": chunk_id in baseline_ids,
                    "is_gold": chunk_id in gold_ids,
                    "is_hard_negative": chunk_id in negative_ids,
                })
            ranked.sort(key=lambda item: (-item["score"], item["chunk_id"]))
            rank_by_chunk = {item["chunk_id"]: rank for rank, item in enumerate(ranked, 1)}
            gold_rows = []
            for evidence in case["gold_evidence"]:
                ranks = [
                    rank_by_chunk[chunk_id]
                    for chunk_id in evidence["mapping"]["chunk_ids"]
                    if chunk_id in rank_by_chunk
                ]
                gold_rows.append({
                    "evidence_id": evidence["evidence_id"],
                    "baseline_vector_ranks": evidence["stages"]["vector"],
                    "atomic_pool_rank": min(ranks) if ranks else None,
                })
            negative_rows = []
            for evidence in case["hard_negatives"]:
                ranks = [
                    rank_by_chunk[chunk_id]
                    for chunk_id in evidence["mapping"]["chunk_ids"]
                    if chunk_id in rank_by_chunk
                ]
                negative_rows.append({
                    "evidence_id": evidence["evidence_id"],
                    "baseline_vector_ranks": evidence["stages"]["vector"],
                    "atomic_pool_rank": min(ranks) if ranks else None,
                })
            result = {
                "id": case["id"],
                "question": case["question"],
                "baseline_depth": len(baseline_ids),
                "pool_parent_count": len(pool_ids),
                "atomic_fact_count": len(unique_texts),
                "gold": gold_rows,
                "hard_negatives": negative_rows,
                "top_candidates": [
                    {key: item[key] for key in (
                        "chunk_id", "score", "fact_count", "in_baseline_top_n",
                        "is_gold", "is_hard_negative", "best_fact",
                    )}
                    for item in ranked[:10]
                ],
            }
            results.append(result)
            print(json.dumps({
                "id": case["id"],
                "pool_parents": len(pool_ids),
                "atomic_facts": len(unique_texts),
                "gold": gold_rows,
                "hard_negatives": negative_rows,
            }, ensure_ascii=False), flush=True)
        output = {
            "scope": "fixed_baseline_top_n_plus_gold_pool_not_full_corpus_ranking",
            "source_bundle_id": source.bundle_id,
            "model_id": embedding.model_id,
            "model_revision": embedding.model_revision,
            "pool_depth": args.pool_depth,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "cases": results,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        source_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
