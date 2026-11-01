"""比较 unit_v2 的精确余弦排名与 Chroma 原始排名。

该工具是离线诊断工具，不改变生产检索配置，也不修改索引。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.settings_service import SettingsService


def _load_settings():
    try:
        return SettingsService().load()
    except Exception:
        return AppSettings()


def _parent_id(metadata: dict[str, object], point_id: str) -> str:
    return str(metadata.get("stable_chunk_id") or point_id)


def _parent_ranks(units: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_rank, item in enumerate(units, 1):
        parent = _parent_id(item["metadata"], str(item["unit_id"]))
        if parent in seen:
            continue
        seen.add(parent)
        output.append({
            "parent_chunk_id": parent,
            "parent_rank": len(output) + 1,
            "best_unit_rank": raw_rank,
            "best_unit_id": str(item["unit_id"]),
            "score": float(item["score"]),
        })
    return output


def _first_rank(items: list[dict[str, object]], parents: set[str]) -> int | None:
    for item in items:
        if str(item["parent_chunk_id"]) in parents:
            return int(item["parent_rank"])
    return None


def _first_unit_rank(items: list[dict[str, object]], parents: set[str]) -> int | None:
    for raw_rank, item in enumerate(items, 1):
        if _parent_id(item["metadata"], str(item["unit_id"])) in parents:
            return raw_rank
    return None


def _gold_parent_ids(case: dict[str, object]) -> set[str]:
    ids: set[str] = set()
    for mapping in case.get("gold_mappings", []):
        if str(mapping.get("kind")) != "positive":
            continue
        for key in ("declared_chunk_ids", "equivalent_chunk_ids"):
            ids.update(str(item) for item in (mapping.get(key) or []))
    return ids


def _read_chroma_units(store: ChromaVectorStore) -> tuple[np.ndarray, list[dict[str, object]]]:
    collection = store._require_collection()
    total = int(collection.count())
    all_embeddings = []
    rows: list[dict[str, object]] = []
    # Chroma/Rust 使用 SQLite 参数绑定；一次读取数万条会超过变量上限。
    for offset in range(0, total, 1000):
        payload = collection.get(
            limit=min(1000, total - offset),
            offset=offset,
            include=["embeddings", "metadatas"],
        )
        ids = [str(item) for item in (payload.get("ids") or [])]
        metadatas = list(payload.get("metadatas") or [])
        embeddings = np.asarray(payload.get("embeddings"), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(ids):
            raise RuntimeError(
                f"Chroma 向量读取异常：shape={embeddings.shape}, ids={len(ids)}"
            )
        all_embeddings.append(embeddings)
        rows.extend(
            {"unit_id": point_id, "metadata": dict(metadata or {})}
            for point_id, metadata in zip(ids, metadatas)
        )
    return np.concatenate(all_embeddings, axis=0), rows


def _rank_exact(matrix: np.ndarray, rows: list[dict[str, object]], query: list[float], depth: int):
    scores = matrix @ np.asarray(query, dtype=np.float32)
    order = np.argsort(-scores, kind="stable")[:depth]
    return [
        {
            "unit_id": rows[int(index)]["unit_id"],
            "metadata": rows[int(index)]["metadata"],
            "score": float(scores[int(index)]),
        }
        for index in order
    ]


def _rank_chroma(store: ChromaVectorStore, query: list[float], depth: int):
    hits = store.query(query, depth)
    return [
        {
            "unit_id": hit.point_id,
            "metadata": hit.metadata,
            "score": float(hit.score),
        }
        for hit in hits
    ]


def run(run_dir: Path, out_dir: Path, raw_k: int, parent_k: int) -> dict[str, object]:
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    run_config = manifest["experiment_config"]
    vector_index = json.loads(
        (run_dir / "run.json").read_text(encoding="utf-8")
    )["settings"]["vector_index"]
    database_path = Path(str(vector_index["database_path"]))
    chroma_path = database_path.parent / "chroma"
    active_path = database_path.parent.parent / "active.json"
    active = json.loads(active_path.read_text(encoding="utf-8"))
    if int(active["unit_count"]) != int(vector_index["unit_count"]):
        raise RuntimeError("active.json 与运行记录的 unit_count 不一致")

    settings = _load_settings()
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    model_path = manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding",
    ))
    embedding = LocalBgeEmbeddingProvider(
        model_path,
        repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )

    cases = []
    with (run_dir / "cases.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                cases.append(json.loads(line))

    store = ChromaVectorStore(chroma_path)
    try:
        matrix, rows = _read_chroma_units(store)
        if matrix.shape[0] != int(vector_index["unit_count"]):
            raise RuntimeError(
                f"Chroma count 与运行记录不一致：{matrix.shape[0]} != {vector_index['unit_count']}"
            )
        details = []
        for case in cases:
            query = embedding.embed_query(str(case["question"]))
            exact = _rank_exact(matrix, rows, query, max(raw_k, parent_k * 8))
            chroma = _rank_chroma(store, query, max(raw_k, parent_k * 8))
            exact_parents = _parent_ranks(exact)
            chroma_parents = _parent_ranks(chroma)
            gold = _gold_parent_ids(case)
            exact_top_raw = {str(item["unit_id"]) for item in exact[:raw_k]}
            chroma_top_raw = {str(item["unit_id"]) for item in chroma[:raw_k]}
            exact_top_parents = {
                str(item["parent_chunk_id"]) for item in exact_parents[:parent_k]
            }
            chroma_top_parents = {
                str(item["parent_chunk_id"]) for item in chroma_parents[:parent_k]
            }
            exact_unit_rank = _first_unit_rank(exact, gold) if gold else None
            chroma_unit_rank = _first_unit_rank(chroma, gold) if gold else None
            exact_parent_rank = _first_rank(exact_parents, gold) if gold else None
            chroma_parent_rank = _first_rank(chroma_parents, gold) if gold else None
            if not gold:
                diagnosis = "无已声明 Gold，跳过 Gold 排名"
            elif exact_unit_rank is None and chroma_unit_rank is None:
                diagnosis = "精确计算和 Chroma 都未进入诊断深度：Embedding/查询/切块问题"
            elif exact_unit_rank is not None and chroma_unit_rank is None:
                diagnosis = "精确计算命中但 Chroma 未命中：疑似 ANN 漏召回"
            elif exact_unit_rank is not None and chroma_unit_rank is not None:
                diagnosis = "两者均命中：不是 Chroma 漏召回，需看 Embedding 分数和父块归并"
            else:
                diagnosis = (
                    "Chroma 在诊断深度内返回但精确排名超出诊断深度："
                    "属于近似搜索的深层排序差异；不代表生产 Top-K 漏召回"
                )
            details.append({
                "id": case["id"],
                "question": case["question"],
                "gold_parent_ids": sorted(gold),
                "exact_unit_rank": exact_unit_rank,
                "chroma_unit_rank": chroma_unit_rank,
                "exact_parent_rank": exact_parent_rank,
                "chroma_parent_rank": chroma_parent_rank,
                "unit_rank_delta_chroma_minus_exact": (
                    None if exact_unit_rank is None or chroma_unit_rank is None
                    else chroma_unit_rank - exact_unit_rank
                ),
                "parent_rank_delta_chroma_minus_exact": (
                    None if exact_parent_rank is None or chroma_parent_rank is None
                    else chroma_parent_rank - exact_parent_rank
                ),
                "raw_overlap_at_k": len(exact_top_raw & chroma_top_raw),
                "raw_recall_at_k": len(exact_top_raw & chroma_top_raw) / max(1, raw_k),
                "parent_overlap_at_k": len(exact_top_parents & chroma_top_parents),
                "parent_recall_at_k": len(exact_top_parents & chroma_top_parents) / max(1, parent_k),
                "diagnosis": diagnosis,
                "exact_top_units": exact[:raw_k],
                "chroma_top_units": chroma[:raw_k],
            })
    finally:
        store.close()

    valid = [item for item in details if item["gold_parent_ids"]]
    summary = {
        "run_dir": str(run_dir),
        "bundle_id": manifest["bundle_id"],
        "vector_index_fingerprint": vector_index["index_fingerprint"],
        "embedding_model": run_config.get("embedding_model"),
        "unit_count": int(vector_index["unit_count"]),
        "parent_chunk_count": int(vector_index["parent_chunk_count"]),
        "raw_k": raw_k,
        "parent_k": parent_k,
        "case_count": len(details),
        "gold_case_count": len(valid),
        "mean_raw_recall_at_k": sum(item["raw_recall_at_k"] for item in details) / max(1, len(details)),
        "mean_parent_recall_at_k": sum(item["parent_recall_at_k"] for item in details) / max(1, len(details)),
        "gold_exact_hit_cases": sum(item["exact_unit_rank"] is not None for item in valid),
        "gold_chroma_hit_cases": sum(item["chroma_unit_rank"] is not None for item in valid),
        "ann_suspected_cases": sum("ANN 漏召回" in item["diagnosis"] for item in valid),
        "embedding_or_query_cases": sum("Embedding/查询" in item["diagnosis"] for item in valid),
        "details": details,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--raw-k", type=int, default=50)
    parser.add_argument("--parent-k", type=int, default=50)
    args = parser.parse_args()
    summary = run(args.run_dir.resolve(), args.out_dir.resolve(), args.raw_k, args.parent_k)
    print(json.dumps({
        key: summary[key] for key in (
            "case_count", "gold_case_count", "unit_count", "parent_chunk_count",
            "mean_raw_recall_at_k", "mean_parent_recall_at_k",
            "gold_exact_hit_cases", "gold_chroma_hit_cases",
            "ann_suspected_cases", "embedding_or_query_cases",
        )
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
