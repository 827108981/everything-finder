"""在已发布向量索引上验证原问题与现有 focused 查询的独立召回及融合。

复用评测报告中已记录的原问题 Top-50，focused 路线重新查询全库；
查询文本完全来自报告中的 QueryPlan，不读取金标生成查询。
"""

from __future__ import annotations

import argparse
import json
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


def _ranks(evidence_rows: list[dict], ranks_by_chunk: dict[str, int]) -> list[int | None]:
    return [
        min((ranks_by_chunk[chunk_id] for chunk_id in evidence["mapping"]["chunk_ids"]
             if chunk_id in ranks_by_chunk), default=None)
        for evidence in evidence_rows
    ]


def _metrics(cases: list[dict], route: str) -> dict:
    eligible = [case for case in cases if case["expected_behavior"] != "refuse"]
    firsts = [min((rank for rank in case[route]["gold"] if rank), default=None)
              for case in eligible]
    return {
        "cases": len(eligible),
        "hit_at_5": sum(rank is not None and rank <= 5 for rank in firsts),
        "hit_at_50": sum(rank is not None and rank <= 50 for rank in firsts),
        "mrr": sum(1 / rank if rank else 0 for rank in firsts) / len(eligible),
        "mean_evidence_coverage": sum(
            sum(rank is not None for rank in case[route]["gold"])
            / max(1, len(case[route]["gold"]))
            for case in eligible
        ) / len(eligible),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--report-detail", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    settings = SettingsService().load()
    store = RagIndexStore(args.generation_base_path.resolve().parent / "rag")
    source = store.active()
    if source.bundle_id != args.expected_bundle_id:
        raise ValueError("AI bundle 已改变，不能比较不同语料")
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(
            settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding"
        )),
        repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    index = VectorIndexStore(store.root).active(
        source.bundle_id,
        embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision,
        embedding_dimension=embedding.dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint,
    )
    vector_store = ChromaVectorStore(index.chroma_path)
    try:
        datasets = []
        for path in args.report_detail:
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["run"]["ai_bundle_id"] != source.bundle_id:
                raise ValueError(f"报告 AI bundle 不一致：{path}")
            cases = []
            for case in report["cases"]:
                original_text = case["query"]["semantic_query"]
                if original_text != case["question"]:
                    raise ValueError(f"原问题查询不一致：{case['id']}")
                original = case["stage_data"]["vector"]["candidates"]
                if len(original) != 50:
                    raise ValueError(f"基线不是完整 Top-50：{case['id']}")
                base_ids = [item["chunk_id"] for item in original]
                base_ranks = {chunk_id: rank for rank, chunk_id in enumerate(base_ids, 1)}
                expected = [min(e["stages"]["vector"], default=None)
                            for e in case["gold_evidence"]]
                if _ranks(case["gold_evidence"], base_ranks) != expected:
                    raise AssertionError(f"基线名次无法复现：{case['id']}")

                focused = next((item["query_text"] for item in case["query"]["semantic_variants"]
                                if item["route_id"] == "semantic_focused"), None)
                if focused:
                    retriever = VectorRetriever(
                        store, source, vector_store, embedding,
                        backend="unit_v2", unit_index=index, oversample_factor=8,
                    )
                    hits = retriever.retrieve(
                        RagQuery(text=focused, vector_top_k=50, include_ocr=True),
                        CancelToken(),
                    )
                    focus_ids = [item.chunk.stable_chunk_id for item in hits]
                else:
                    # 未生成 focused 变体的题使用原排序占位，并明确标记。
                    focus_ids = base_ids
                focus_ranks = {chunk_id: rank for rank, chunk_id in enumerate(focus_ids, 1)}
                all_ids = set(base_ids) | set(focus_ids)
                merged_ids = sorted(all_ids, key=lambda chunk_id: (
                    -(1 / (60 + base_ranks[chunk_id]) if chunk_id in base_ranks else 0)
                    -(1 / (60 + focus_ranks[chunk_id]) if chunk_id in focus_ranks else 0),
                    chunk_id,
                ))[:50]
                merged_ranks = {chunk_id: rank for rank, chunk_id in enumerate(merged_ids, 1)}
                routes = {}
                for name, ranks in (("original", base_ranks), ("focused", focus_ranks),
                                    ("rrf", merged_ranks)):
                    routes[name] = {
                        "gold": _ranks(case["gold_evidence"], ranks),
                        "hard_negatives": _ranks(case["hard_negatives"], ranks),
                    }
                row = {
                    "id": case["id"], "question": case["question"],
                    "expected_behavior": case["expected_behavior"],
                    "focused_query": focused,
                    "focused_available": bool(focused),
                    "focused_depth": len(focus_ids),
                    **routes,
                }
                cases.append(row)
                print(json.dumps({
                    "id": case["id"],
                    "original": routes["original"]["gold"],
                    "focused": routes["focused"]["gold"],
                    "rrf": routes["rrf"]["gold"],
                }, ensure_ascii=False), flush=True)
            datasets.append({
                "source_report": str(path), "cases": cases,
                "summary": {name: _metrics(cases, name) for name in ("original", "focused", "rrf")},
            })
        result = {
            "scope": "full_corpus_query_variants_on_existing_vector_index",
            "bundle_id": source.bundle_id,
            "model": embedding.model_id,
            "model_revision": embedding.model_revision,
            "index_fingerprint": index.index_fingerprint,
            "datasets": datasets,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        vector_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
