"""比较金标向量在“带来源上下文”和“仅正文”编码下的相似度。

这是全量重建前的低成本方向性实验。它只回答金标自身的向量是否变得更接近问题，
不能代替全库排名或 Recall 评测。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.retrieval.vector_units import (
    VectorUnitBuilder,
    VectorUnitPolicy,
)
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.settings_service import SettingsService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--report-detail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    report = json.loads(args.report_detail.read_text(encoding="utf-8"))
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
    baseline_policy = VectorUnitPolicy.from_settings(settings)
    baseline = VectorIndexStore(store.root).active(
        source.bundle_id,
        embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision,
        embedding_dimension=embedding.dimension,
        policy_fingerprint=baseline_policy.fingerprint,
    )
    body_policy = VectorUnitPolicy(
        target_tokens=baseline_policy.target_tokens,
        max_tokens=baseline_policy.max_tokens,
        overlap_tokens=baseline_policy.overlap_tokens,
        include_source_context=False,
    )
    unit_builder = VectorUnitBuilder(embedding.count_tokens, body_policy)
    source_db = store.connect_read_only(source.metadata_path)
    vector_db = sqlite3.connect(
        f"file:{baseline.database_path}?mode=ro", uri=True
    )
    vectors = ChromaVectorStore(baseline.chroma_path)
    try:
        cases = []
        for case in report["cases"]:
            query_vector = np.asarray(
                embedding.embed_query(case["question"]), dtype=np.float32
            )
            evidence_rows = []
            for evidence in case["gold_evidence"]:
                chunk_ids = list(evidence["mapping"]["chunk_ids"])
                unit_ids: list[str] = []
                body_texts: list[str] = []
                baseline_texts: list[str] = []
                for chunk_id in chunk_ids:
                    chunk = store.load_chunk(source_db, chunk_id)
                    if chunk is None:
                        raise ValueError(f"金标父 chunk 不存在：{chunk_id}")
                    for unit in unit_builder.build(chunk):
                        unit_ids.append(unit.unit_id)
                        body_texts.append(unit.text)
                        row = vector_db.execute(
                            "SELECT embedding_text FROM vector_units WHERE unit_id=?",
                            (unit.unit_id,),
                        ).fetchone()
                        if row is None:
                            raise ValueError(f"生产向量单元不存在：{unit.unit_id}")
                        baseline_texts.append(str(row[0]))
                baseline_vectors = vectors.get_embeddings(unit_ids)
                body_vectors = embedding.embed_documents(body_texts)
                current_scores = [
                    float(np.dot(query_vector, np.asarray(baseline_vectors[unit_id])))
                    for unit_id in unit_ids
                ]
                body_scores = [
                    float(np.dot(query_vector, np.asarray(vector)))
                    for vector in body_vectors
                ]
                current_max = max(current_scores)
                body_max = max(body_scores)
                evidence_rows.append({
                    "evidence_id": evidence["evidence_id"],
                    "reported_vector_ranks": evidence["stages"]["vector"],
                    "unit_count": len(unit_ids),
                    "units_already_body_only": sum(
                        before == after
                        for before, after in zip(baseline_texts, body_texts, strict=True)
                    ),
                    "current_max_cosine": current_max,
                    "body_only_max_cosine": body_max,
                    "delta": body_max - current_max,
                })
            cases.append({
                "id": case["id"],
                "question": case["question"],
                "evidence": evidence_rows,
            })
            print(json.dumps({
                "id": case["id"],
                "evidence": [
                    {
                        "rank": item["reported_vector_ranks"],
                        "current": round(item["current_max_cosine"], 6),
                        "body_only": round(item["body_only_max_cosine"], 6),
                        "delta": round(item["delta"], 6),
                    }
                    for item in evidence_rows
                ],
            }, ensure_ascii=False), flush=True)
        output = {
            "scope": "gold_similarity_probe_only_not_full_corpus_ranking",
            "source_bundle_id": source.bundle_id,
            "model_id": embedding.model_id,
            "model_revision": embedding.model_revision,
            "baseline_policy_fingerprint": baseline_policy.fingerprint,
            "body_only_policy_fingerprint": body_policy.fingerprint,
            "cases": cases,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    finally:
        vectors.close()
        vector_db.close()
        source_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
