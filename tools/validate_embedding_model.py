"""在隔离目录构建一个候选 Embedding 向量索引，并做纯向量召回对比。

该工具只读取已发布 AI bundle 的正文，不改写生产 active.json；输出中的指标
只代表向量召回（不含关键词、融合、重排和 LLM），便于判断模型是否改善了
向量层本身。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import aggregate, score_case
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


def _build_settings(args: argparse.Namespace) -> AppSettings:
    settings = AppSettings()
    settings.rag_vector_backend = "unit_v2"
    settings.rag_vector_unit_target_tokens = int(args.target_tokens)
    settings.rag_vector_unit_max_tokens = int(args.max_tokens)
    settings.rag_vector_unit_overlap_tokens = int(args.overlap_tokens)
    settings.rag_vector_unit_include_source_context = True
    settings.rag_vector_unit_oversample_factor = 8
    settings.rag_embedding_batch_size = int(args.batch_size)
    settings.rag_embedding_repo_id = args.model_id
    settings.rag_embedding_revision = args.revision
    return settings


def _evaluate(
    *, cases_path: Path, source_store: RagIndexStore, source_bundle,
    embedding, unit_index, vector_store, top_k: int,
) -> dict[str, object]:
    dataset = load_gold_dataset(cases_path, reviewed_only=False)
    cases = [case for case in dataset.cases if case.review_status == "reviewed"]
    connection = source_store.connect_read_only(source_bundle.metadata_path)
    try:
        chunk_rows = load_chunk_rows(connection)
    finally:
        connection.close()
    mappings = {case.case_id: tuple(map_case_evidence(case, chunk_rows)) for case in cases}
    retriever = VectorRetriever(
        source_store, source_bundle, vector_store, embedding,
        backend="unit_v2", unit_index=unit_index, oversample_factor=8,
    )
    rows: list[dict[str, object]] = []
    for case in cases:
        started = time.monotonic()
        query = RagQuery(text=case.question, vector_top_k=top_k)
        hits = retriever.retrieve(query, CancelToken())
        candidates = [
            {
                "chunk_id": hit.chunk.stable_chunk_id,
                "member_chunk_ids": list(hit.all_member_chunk_ids),
                "text": hit.chunk.text,
                "rank": rank,
                "score": hit.vector_score,
            }
            for rank, hit in enumerate(hits, 1)
        ]
        scored = score_case(case, {"vector": candidates}, list(mappings[case.case_id]))
        rows.append({
            "id": case.case_id,
            "question": case.question,
            "expected_behavior": case.expected_behavior,
            "vector": scored["stages"]["vector"],
            "top_candidates": candidates,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "route_diagnostics": list(retriever.route_diagnostics),
        })
    # aggregate() 需要 score_case 的完整顶层结果；此处只补齐 vector 层，
    # 其它阶段留空，避免把纯向量结果误报成完整 RAG 结果。
    complete = []
    for row, case in zip(rows, cases):
        complete.append({
            "id": case.case_id,
            "expected_behavior": case.expected_behavior,
            "error": None,
            "stages": {"vector": row["vector"]},
        })
    summary = aggregate(complete)
    return {
        "cases_path": str(cases_path),
        "case_count": len(cases),
        "summary": summary,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--cases", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--target-tokens", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--overlap-tokens", type=int, default=32)
    args = parser.parse_args()

    settings = _build_settings(args)
    source_store = RagIndexStore(args.generation_base_path.expanduser().resolve().parent / "rag")
    source_bundle = source_store.active()
    if source_bundle.bundle_id != args.expected_bundle_id:
        raise RuntimeError(
            f"当前 AI bundle 为 {source_bundle.bundle_id}，不是指定的 {args.expected_bundle_id}"
        )
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    model_path = manager.resolve_local_path(ModelSpec(args.model_id, args.revision, "embedding"))
    embedding = LocalBgeEmbeddingProvider(
        model_path, repo_id=args.model_id, revision=args.revision,
        batch_size=args.batch_size,
    )
    model_slug = args.model_id.replace("/", "__") + "__" + args.revision[:12]
    vector_root = args.out.expanduser().resolve() / model_slug
    vector_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    last_percent = {-1}

    def on_progress(event: dict[str, object]) -> None:
        percent = int(event.get("percent") or -1)
        if percent >= 0 and (percent == 100 or percent // 5 > max(last_percent)):
            last_percent.add(percent // 5)
            print(json.dumps({
                "stage": event.get("stage"), "percent": percent,
                "parents": event.get("processed_parent_chunks"),
                "units": event.get("unit_count"),
            }, ensure_ascii=False), flush=True)
    unit_index, reused = VectorIndexBuilder(
        source_store, embedding, settings, vector_root=vector_root,
    ).prepare(source_bundle, CancelToken(), on_progress)
    vector_store = ChromaVectorStore(unit_index.chroma_path)
    try:
        results = []
        for cases_path in args.cases:
            results.append(_evaluate(
                cases_path=cases_path, source_store=source_store,
                source_bundle=source_bundle, embedding=embedding,
                unit_index=unit_index, vector_store=vector_store, top_k=args.top_k,
            ))
    finally:
        vector_store.close()
        try:
            from chromadb.api.shared_system_client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
        except (ImportError, AttributeError):
            pass
    output = {
        "model_id": args.model_id,
        "revision": args.revision,
        "dimension": embedding.dimension,
        "source_bundle_id": source_bundle.bundle_id,
        "vector_index": {
            "path": str(unit_index.path),
            "fingerprint": unit_index.index_fingerprint,
            "parent_chunk_count": unit_index.parent_chunk_count,
            "unit_count": unit_index.unit_count,
            "reused": reused,
        },
        "top_k": args.top_k,
        "total_duration_ms": int((time.monotonic() - started) * 1000),
        "datasets": results,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / (model_slug + "__validation.json")
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model_id": args.model_id, "dimension": embedding.dimension,
                      "unit_count": unit_index.unit_count, "output": str(path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
