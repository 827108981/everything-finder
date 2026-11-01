"""在冻结的父 chunk 候选池中比较现有窗口与保留上下文的较短窗口。

候选池固定为原向量 Top-N、Gold 和指定负例；所有候选接受相同变换。
Gold 仅用于事后标注，不参与文本变换、嵌入或查询构造。
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import replace
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.reporting.matching import normalize_text
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.settings_service import SettingsService


_PDF_WRAP = re.compile(r"(?<=[\u4e00-\u9fff])\n(?=[\u4e00-\u9fff])")


def _new_units(chunk, builder: VectorUnitBuilder) -> list[tuple[str, str]]:
    text = chunk.text
    if chunk.source.extension.lower() == ".pdf":
        # 只修复两个汉字之间的物理换行，保留标点后的段落边界。
        text = _PDF_WRAP.sub("", text)
    heading = next((line.strip() for line in text.splitlines() if line.strip()), "")
    units = builder.build(replace(chunk, text=text))
    return [
        (
            f"章节：{heading}\n{unit.embedding_text}"
            if heading and heading not in unit.text else unit.embedding_text,
            unit.text,
        )
        for unit in units
    ]


def _quote_coverage(quote: str, snippet: str) -> float:
    needle = normalize_text(quote)
    haystack = normalize_text(snippet)
    if not needle or not haystack:
        return 0.0
    match = SequenceMatcher(None, needle, haystack, autojunk=False).find_longest_match(
        0, len(needle), 0, len(haystack),
    )
    return match.size / len(needle)


def _evidence_rows(
    evidence: list[dict], order: dict[str, int], best_text: dict[str, str],
    unit_candidates: dict[str, list[tuple[float, str]]],
    parent_scores: dict[str, float],
) -> list[dict]:
    rows = []
    for item in evidence:
        ids = item["mapping"]["chunk_ids"]
        ranks = [order[chunk_id] for chunk_id in ids if chunk_id in order]
        ranked_id = min(ids, key=lambda chunk_id: order.get(chunk_id, 10**9))
        supporting = sorted(
            (
                (score, _quote_coverage(item["quote"], snippet), snippet)
                for chunk_id in ids
                for score, snippet in unit_candidates.get(chunk_id, [])
                if _quote_coverage(item["quote"], snippet) >= 0.6
            ),
            key=lambda value: -value[0],
        )
        support_score = supporting[0][0] if supporting else None
        rows.append({
            "evidence_id": item["evidence_id"],
            "rank": min(ranks) if ranks else None,
            "reported_full_corpus_rank": min(item["stages"]["vector"], default=None),
            "best_snippet_quote_coverage": _quote_coverage(
                item["quote"], best_text.get(ranked_id, ""),
            ),
            "best_snippet": best_text.get(ranked_id, ""),
            "supporting_snippet_score": support_score,
            "supporting_snippet_quote_coverage": supporting[0][1] if supporting else None,
            "supporting_snippet": supporting[0][2] if supporting else None,
            "supporting_snippet_pool_rank": (
                1 + sum(score > support_score for score in parent_scores.values())
                if support_score is not None else None
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", required=True)
    parser.add_argument("--report-detail", type=Path, required=True)
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-depth", type=int, default=10)
    parser.add_argument("--target-tokens", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--overlap-tokens", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(max(1, args.torch_threads))
    report = json.loads(args.report_detail.read_text(encoding="utf-8"))
    cases = [item for item in report["cases"] if item["id"] in set(args.case_id)]
    if len(cases) != len(set(args.case_id)):
        raise ValueError("请求的题目不在报告中")
    settings = SettingsService().load()
    store = RagIndexStore(args.generation_base_path.resolve().parent / "rag")
    source = store.active()
    if source.bundle_id != args.expected_bundle_id:
        raise ValueError("AI bundle 已改变")
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(
            settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding"
        )), repo_id=settings.rag_embedding_repo_id, revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    baseline = VectorIndexStore(store.root).active(
        source.bundle_id, embedding_repo_id=embedding.model_id,
        embedding_revision=embedding.model_revision, embedding_dimension=embedding.dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint,
    )
    policy = VectorUnitPolicy(
        target_tokens=args.target_tokens, max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens, include_source_context=True,
    )
    builder = VectorUnitBuilder(embedding.count_tokens, policy)
    source_db = store.connect_read_only(source.metadata_path)
    unit_db = sqlite3.connect(f"file:{baseline.database_path}?mode=ro", uri=True)
    vectors = ChromaVectorStore(baseline.chroma_path)
    output = {
        "scope": "fixed_baseline_top_n_plus_gold_and_negatives_not_full_corpus_recall",
        "bundle_id": source.bundle_id,
        "model": embedding.model_id,
        "model_revision": embedding.model_revision,
        "index_fingerprint": baseline.index_fingerprint,
        "candidate_pool_depth": args.pool_depth,
        "window_policy": {
            "target_tokens": args.target_tokens, "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "pdf_wrap_repair": "join Han-to-Han physical line breaks",
            "context": "source + location + first source line as heading if absent",
        },
        "cases": [],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for case in cases:
            started = time.monotonic()
            base_candidates = case["stage_data"]["vector"]["candidates"][:args.pool_depth]
            top_ids = [item["chunk_id"] for item in base_candidates]
            labeled_ids = [
                chunk_id
                for kind in ("gold_evidence", "hard_negatives")
                for evidence in case[kind]
                for chunk_id in evidence["mapping"]["chunk_ids"]
            ]
            parent_ids = list(dict.fromkeys([*top_ids, *labeled_ids]))
            base_unit_ids: dict[str, list[str]] = {}
            base_unit_texts: dict[str, list[str]] = {}
            new_texts: dict[str, list[tuple[str, str]]] = {}
            for chunk_id in parent_ids:
                chunk = store.load_chunk(source_db, chunk_id)
                if chunk is None:
                    raise ValueError(f"缺少父 chunk：{chunk_id}")
                rows = unit_db.execute(
                    "SELECT unit_id,text FROM vector_units WHERE parent_chunk_id=? ORDER BY ordinal",
                    (chunk_id,),
                ).fetchall()
                base_unit_ids[chunk_id] = [row[0] for row in rows]
                base_unit_texts[chunk_id] = [row[1] for row in rows]
                new_texts[chunk_id] = _new_units(chunk, builder)
            all_ids = list(dict.fromkeys(uid for ids in base_unit_ids.values() for uid in ids))
            base_vectors = vectors.get_embeddings(all_ids)
            if len(base_vectors) != len(all_ids):
                raise ValueError("基线向量缺失")
            unique_new = sorted({text for units in new_texts.values() for text, _ in units},
                                key=embedding.count_tokens)
            print(json.dumps({"id": case["id"], "stage": "encoding",
                              "parents": len(parent_ids), "new_units": len(unique_new)},
                             ensure_ascii=False), flush=True)
            encoded = embedding.embed_documents(unique_new)
            new_vectors = dict(zip(unique_new, encoded, strict=True))
            focused = next((item["query_text"] for item in case["query"]["semantic_variants"]
                            if item["route_id"] == "semantic_focused"), None)
            query_texts = {"original": case["question"]}
            if focused:
                query_texts["focused"] = focused
            rank_sets = {}
            for query_name, query_text in query_texts.items():
                query_vector = np.asarray(embedding.embed_query(query_text), dtype=np.float32)
                for doc_name in ("baseline", "contextual_small_window"):
                    scored: list[tuple[str, float, str]] = []
                    unit_candidates: dict[str, list[tuple[float, str]]] = {}
                    for chunk_id in parent_ids:
                        if doc_name == "baseline":
                            pairs = list(zip(base_unit_ids[chunk_id], base_unit_texts[chunk_id], strict=True))
                            candidates = [
                                (float(np.dot(query_vector, np.asarray(base_vectors[uid], dtype=np.float32))), text)
                                for uid, text in pairs
                            ]
                        else:
                            candidates = [
                                (float(np.dot(query_vector, np.asarray(new_vectors[text], dtype=np.float32))), unit_text)
                                for text, unit_text in new_texts[chunk_id]
                            ]
                        score, best_text = max(candidates, key=lambda item: item[0])
                        unit_candidates[chunk_id] = candidates
                        scored.append((chunk_id, score, best_text))
                    scored.sort(key=lambda item: (-item[1], item[0]))
                    order = {chunk_id: rank for rank, (chunk_id, _, _) in enumerate(scored, 1)}
                    best_texts = {chunk_id: text for chunk_id, _, text in scored}
                    parent_scores = {chunk_id: score for chunk_id, score, _ in scored}
                    name = f"{doc_name}_{query_name}"
                    rank_sets[name] = {
                        "gold": _evidence_rows(
                            case["gold_evidence"], order, best_texts,
                            unit_candidates, parent_scores,
                        ),
                        "hard_negatives": _evidence_rows(
                            case["hard_negatives"], order, best_texts,
                            unit_candidates, parent_scores,
                        ),
                        "top_three": [
                            {"chunk_id": chunk_id, "score": score, "text": text[:300]}
                            for chunk_id, score, text in scored[:3]
                        ],
                    }
            row = {
                "id": case["id"], "question": case["question"],
                "focused_query": focused,
                "pool_parent_count": len(parent_ids),
                "baseline_unit_count": len(all_ids),
                "new_unit_count": len(unique_new),
                "elapsed_seconds": round(time.monotonic() - started, 2),
                "groups": rank_sets,
            }
            output["cases"].append(row)
            args.out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({
                "id": case["id"], "stage": "complete", "elapsed_seconds": row["elapsed_seconds"],
                "ranks": {key: [item["rank"] for item in val["gold"]]
                          for key, val in rank_sets.items()},
            }, ensure_ascii=False), flush=True)
    finally:
        vectors.close()
        unit_db.close()
        source_db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
