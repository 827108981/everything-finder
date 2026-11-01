"""在固定候选池中比较多个 Embedding 模型的语义排序能力。

候选池由当前模型每题 Top-50 的首个命中单元，加上该题全部 Gold/指定负例
父块的所有向量单元组成。该结果用于模型诊断，不等同于全库 Recall@50。
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec


def _load_rows(paths: list[Path]) -> list[dict[str, object]]:
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    row["_source_run"] = str(path.parent)
                    rows.append(row)
    return rows


def _evidence_parent_sets(row: dict[str, object], kind: str) -> list[set[str]]:
    groups = []
    for item in row.get("gold_mappings", []):
        if item.get("kind") != kind:
            continue
        ids = {str(value) for value in item.get("chunk_ids", []) if value}
        if ids:
            groups.append(ids)
    return groups


def _prepare_pools(rows: list[dict[str, object]], database: Path, depth: int):
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    pools = []
    all_unit_ids: set[str] = set()
    try:
        for row in rows:
            parent_units: dict[str, set[str]] = {}
            vector_rows = row.get("candidate_stages", {}).get("vector", [])[:depth]
            for candidate in vector_rows:
                parent = str(candidate.get("chunk_id") or "")
                hits = candidate.get("vector_unit_hits") or []
                if parent and hits:
                    parent_units.setdefault(parent, set()).add(str(hits[0]["unit_id"]))
            gold_groups = _evidence_parent_sets(row, "positive")
            negative_groups = _evidence_parent_sets(row, "hard_negative")
            special_parents = set().union(*gold_groups, *negative_groups) if (gold_groups or negative_groups) else set()
            for offset in range(0, len(special_parents), 400):
                batch = list(special_parents)[offset:offset + 400]
                if not batch:
                    continue
                marks = ",".join("?" for _ in batch)
                for found in connection.execute(
                    f"SELECT unit_id,parent_chunk_id FROM vector_units WHERE parent_chunk_id IN ({marks})",
                    batch,
                ):
                    parent_units.setdefault(str(found["parent_chunk_id"]), set()).add(str(found["unit_id"]))
            unit_ids = set().union(*parent_units.values()) if parent_units else set()
            all_unit_ids.update(unit_ids)
            pools.append({
                "id": row["id"], "question": row["question"],
                "expected_behavior": row["expected_behavior"],
                "parent_units": parent_units,
                "gold_groups": gold_groups, "negative_groups": negative_groups,
                "baseline_vector": row.get("stages", {}).get("vector", {}),
                "source_run": row["_source_run"],
            })
        texts = {}
        ids = sorted(all_unit_ids)
        for offset in range(0, len(ids), 500):
            batch = ids[offset:offset + 500]
            marks = ",".join("?" for _ in batch)
            for found in connection.execute(
                f"SELECT unit_id,embedding_text FROM vector_units WHERE unit_id IN ({marks})", batch,
            ):
                texts[str(found["unit_id"])] = str(found["embedding_text"])
    finally:
        connection.close()
    return pools, texts


def _evaluate_model(model_id: str, revision: str, pools, texts, batch_size: int):
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    provider = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(model_id, revision, "embedding")),
        repo_id=model_id, revision=revision, batch_size=batch_size,
    )
    unit_ids = sorted(texts)
    started = time.monotonic()
    vectors: dict[str, np.ndarray] = {}
    for offset in range(0, len(unit_ids), batch_size):
        batch = unit_ids[offset:offset + batch_size]
        encoded = provider.embed_documents([texts[unit_id] for unit_id in batch])
        vectors.update({unit_id: np.asarray(value, dtype=np.float32) for unit_id, value in zip(batch, encoded)})
        if offset == 0 or offset + len(batch) == len(unit_ids) or offset // batch_size % 5 == 0:
            print(json.dumps({"model": model_id, "encoded": offset + len(batch),
                              "total": len(unit_ids)}, ensure_ascii=False), flush=True)
    encode_ms = int((time.monotonic() - started) * 1000)
    results = []
    for pool in pools:
        query_started = time.monotonic()
        query = np.asarray(provider.embed_query(pool["question"]), dtype=np.float32)
        parent_scores = {
            parent: max(float(np.dot(query, vectors[unit_id])) for unit_id in ids)
            for parent, ids in pool["parent_units"].items() if ids
        }
        ranked = sorted(parent_scores, key=lambda parent: (-parent_scores[parent], parent))
        rank_by_parent = {parent: rank for rank, parent in enumerate(ranked, 1)}
        gold_ranks = [
            min((rank_by_parent[parent] for parent in group if parent in rank_by_parent), default=None)
            for group in pool["gold_groups"]
        ]
        negative_ranks = [
            min((rank_by_parent[parent] for parent in group if parent in rank_by_parent), default=None)
            for group in pool["negative_groups"]
        ]
        present_gold = [rank for rank in gold_ranks if rank is not None]
        first_gold = min(present_gold) if present_gold else None
        present_negative = [rank for rank in negative_ranks if rank is not None]
        results.append({
            "id": pool["id"], "question": pool["question"],
            "expected_behavior": pool["expected_behavior"],
            "pool_parent_count": len(ranked),
            "first_gold_rank_in_pool": first_gold,
            "gold_ranks_in_pool": gold_ranks,
            "gold_recall_at_5_in_pool": (
                sum(rank is not None and rank <= 5 for rank in gold_ranks) / len(gold_ranks)
                if gold_ranks else None
            ),
            "first_hard_negative_rank_in_pool": min(present_negative) if present_negative else None,
            "hard_negative_above_first_gold": (
                sum(rank < first_gold for rank in present_negative) if first_gold else None
            ),
            "baseline_first_gold_rank_full_top50": pool["baseline_vector"].get("first_gold_rank"),
            "baseline_gold_recall_full_top50": pool["baseline_vector"].get("gold_evidence_recall"),
            "query_duration_ms": int((time.monotonic() - query_started) * 1000),
            "top10": [
                {"rank": rank, "parent_chunk_id": parent, "score": parent_scores[parent]}
                for rank, parent in enumerate(ranked[:10], 1)
            ],
            "source_run": pool["source_run"],
        })
    scored = [row for row in results if row["expected_behavior"] != "refuse" and row["gold_ranks_in_pool"]]
    summary = {
        "case_count": len(results), "scored_case_count": len(scored),
        "gold_hit_at_5_cases_in_pool": sum(
            row["first_gold_rank_in_pool"] is not None and row["first_gold_rank_in_pool"] <= 5
            for row in scored
        ),
        "mrr_in_pool": (
            sum(1.0 / row["first_gold_rank_in_pool"] if row["first_gold_rank_in_pool"] else 0.0 for row in scored)
            / len(scored) if scored else None
        ),
        "mean_first_gold_rank_in_pool": (
            sum(row["first_gold_rank_in_pool"] for row in scored if row["first_gold_rank_in_pool"])
            / sum(bool(row["first_gold_rank_in_pool"]) for row in scored)
            if any(row["first_gold_rank_in_pool"] for row in scored) else None
        ),
        "mean_gold_recall_at_5_in_pool": (
            sum(row["gold_recall_at_5_in_pool"] for row in scored) / len(scored) if scored else None
        ),
        "hard_negatives_above_gold": sum(row["hard_negative_above_first_gold"] or 0 for row in scored),
        "document_encode_duration_ms": encode_ms,
        "mean_query_duration_ms": (
            sum(row["query_duration_ms"] for row in results) / len(results) if results else None
        ),
        "unique_unit_count": len(unit_ids),
    }
    return {"model_id": model_id, "revision": revision, "dimension": provider.dimension,
            "summary": summary, "cases": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--vector-units-db", type=Path, required=True)
    parser.add_argument("--model", action="append", nargs=2, metavar=("MODEL_ID", "REVISION"), required=True)
    parser.add_argument("--candidate-depth", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = _load_rows(args.cases_jsonl)
    pools, texts = _prepare_pools(rows, args.vector_units_db, args.candidate_depth)
    output = {
        "method": "fixed_candidate_pool",
        "limitation": (
            "候选池为当前模型 Top-N 每个父块的首个向量单元，并补入全部 Gold/指定负例单元；"
            "只能比较池内语义排序，不能解释为全库 Recall@N。"
        ),
        "candidate_depth": args.candidate_depth,
        "dataset_case_count": len(rows),
        "unique_unit_count": len(texts),
        "models": [
            _evaluate_model(model_id, revision, pools, texts, args.batch_size)
            for model_id, revision in args.model
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.out), "unit_count": len(texts)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
