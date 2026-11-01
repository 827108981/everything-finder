"""用句子级原子事实近似，比较 Gold 与问题的向量相似度。

本工具只编码 Gold 父 chunk，用于判断原子事实方向是否值得继续做候选池或全库实验；
它不计算全库排名，也不代表 Recall。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.settings_service import SettingsService


_BOUNDARY = re.compile(r"(?<=[。！？!?；;])|\n+")
_LEADING_BULLET = re.compile(r"^[\s•●○▪■□✓✔➢▶►·・\-—]+")


def _split_by_tokens(text: str, count_tokens, max_tokens: int) -> list[str]:
    """把超长句按模型 token 上限切开，优先在逗号、冒号和空格处断开。"""
    remaining = text.strip()
    output: list[str] = []
    while remaining:
        if count_tokens(remaining) <= max_tokens:
            output.append(remaining)
            break
        low, high, best = 1, len(remaining), 1
        while low <= high:
            middle = (low + high) // 2
            if count_tokens(remaining[:middle]) <= max_tokens:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        boundaries = [remaining.rfind(mark, 0, best) for mark in ("，", ",", "：", ":", " ")]
        boundary = max(boundaries)
        cut = boundary + 1 if boundary >= best // 2 else best
        output.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [item for item in output if item]


def atomic_facts(
    text: str,
    count_tokens,
    *,
    min_tokens: int = 8,
    max_tokens: int = 64,
) -> list[str]:
    """按自然句和换行拆分，并合并过短标题或残片。

    规则不使用型号、领域词或测试集内容；token 数由当前 Embedding tokenizer 计算。
    """
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    fragments = [
        _LEADING_BULLET.sub("", item).strip()
        for item in _BOUNDARY.split(normalized)
    ]
    fragments = [item for item in fragments if item]
    expanded: list[str] = []
    for fragment in fragments:
        expanded.extend(_split_by_tokens(fragment, count_tokens, max_tokens))

    facts: list[str] = []
    pending: list[str] = []
    for fragment in expanded:
        if count_tokens(fragment) < min_tokens:
            pending.append(fragment)
            if count_tokens("\n".join(pending)) < min_tokens:
                continue
            candidate = "\n".join(pending)
            pending.clear()
            facts.extend(_split_by_tokens(candidate, count_tokens, max_tokens))
            continue
        if pending:
            candidate = "\n".join([*pending, fragment])
            pending.clear()
            facts.extend(_split_by_tokens(candidate, count_tokens, max_tokens))
        else:
            facts.append(fragment)
    if pending:
        tail = "\n".join(pending)
        if facts and count_tokens(f"{facts[-1]}\n{tail}") <= max_tokens:
            facts[-1] = f"{facts[-1]}\n{tail}"
        else:
            facts.extend(_split_by_tokens(tail, count_tokens, max_tokens))
    return facts


def _embedding_text(chunk, fact: str) -> str:
    title = Path(chunk.source.filename).stem.strip()
    location = chunk.source.location_text.strip()
    prefix = [f"来源：{title}" if title else "", f"位置：{location}" if location else ""]
    return "\n".join([item for item in prefix if item] + [f"正文：{fact}"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--report-detail", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-tokens", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()
    if args.min_tokens < 1 or args.max_tokens < args.min_tokens:
        raise ValueError("原子事实 token 范围无效")

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
    policy = VectorUnitPolicy.from_settings(settings)
    baseline = VectorIndexStore(store.root).active(
        source.bundle_id,
        embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision,
        embedding_dimension=embedding.dimension,
        policy_fingerprint=policy.fingerprint,
    )
    source_db = store.connect_read_only(source.metadata_path)
    vector_db = sqlite3.connect(f"file:{baseline.database_path}?mode=ro", uri=True)
    vectors = ChromaVectorStore(baseline.chroma_path)
    try:
        cases = []
        for case in report["cases"]:
            query_vector = np.asarray(embedding.embed_query(case["question"]), dtype=np.float32)
            evidence_rows = []
            for evidence in case["gold_evidence"]:
                baseline_unit_ids: list[str] = []
                facts: list[str] = []
                fact_embedding_texts: list[str] = []
                for chunk_id in evidence["mapping"]["chunk_ids"]:
                    chunk = store.load_chunk(source_db, chunk_id)
                    if chunk is None:
                        raise ValueError(f"金标父 chunk 不存在：{chunk_id}")
                    baseline_unit_ids.extend(
                        str(row[0]) for row in vector_db.execute(
                            "SELECT unit_id FROM vector_units WHERE parent_chunk_id=? ORDER BY ordinal",
                            (chunk_id,),
                        )
                    )
                    chunk_facts = atomic_facts(
                        chunk.text,
                        embedding.count_tokens,
                        min_tokens=args.min_tokens,
                        max_tokens=args.max_tokens,
                    )
                    facts.extend(chunk_facts)
                    fact_embedding_texts.extend(_embedding_text(chunk, fact) for fact in chunk_facts)
                baseline_vectors = vectors.get_embeddings(baseline_unit_ids)
                current_scores = [
                    float(np.dot(query_vector, np.asarray(baseline_vectors[unit_id])))
                    for unit_id in baseline_unit_ids
                ]
                fact_vectors = embedding.embed_documents(fact_embedding_texts)
                fact_scores = [
                    float(np.dot(query_vector, np.asarray(vector)))
                    for vector in fact_vectors
                ]
                best_index = int(np.argmax(fact_scores))
                current_max = max(current_scores)
                atomic_max = fact_scores[best_index]
                evidence_rows.append({
                    "evidence_id": evidence["evidence_id"],
                    "reported_vector_ranks": evidence["stages"]["vector"],
                    "baseline_unit_count": len(baseline_unit_ids),
                    "atomic_fact_count": len(facts),
                    "current_max_cosine": current_max,
                    "atomic_max_cosine": atomic_max,
                    "delta": atomic_max - current_max,
                    "best_atomic_fact": facts[best_index],
                    "best_atomic_fact_tokens": embedding.count_tokens(facts[best_index]),
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
                        "atomic": round(item["atomic_max_cosine"], 6),
                        "delta": round(item["delta"], 6),
                        "facts": item["atomic_fact_count"],
                    }
                    for item in evidence_rows
                ],
            }, ensure_ascii=False), flush=True)
        output = {
            "scope": "gold_atomic_similarity_probe_only_not_full_corpus_ranking",
            "source_bundle_id": source.bundle_id,
            "model_id": embedding.model_id,
            "model_revision": embedding.model_revision,
            "min_tokens": args.min_tokens,
            "max_tokens": args.max_tokens,
            "cases": cases,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        vectors.close()
        vector_db.close()
        source_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
