from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_full_text_search.config.constants import CACHE_DIR, DB_PATH
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.rag.embeddings.fastembed_local import FastEmbedLocalProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.retrieval.vector import VectorRetriever


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate RAG chunk retrieval against JSONL gold sources.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--index-base", type=Path, default=DB_PATH)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "rag-fastembed")
    parser.add_argument("--modes", default="lexical,vector,hybrid")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    cases = []
    for line_number, line in enumerate(args.cases.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
        except json.JSONDecodeError as exc:
            parser.error(f"{args.cases}:{line_number}: 无效 JSON：{exc}")
        if not str(case.get("question") or "").strip():
            parser.error(f"{args.cases}:{line_number}: question 不能为空")
        if not isinstance(case.get("relevant_sources"), list) or not case["relevant_sources"]:
            parser.error(f"{args.cases}:{line_number}: relevant_sources 必须是非空列表")
        cases.append(case)
    if not cases:
        parser.error("评测集为空")

    provider = FastEmbedLocalProvider(
        args.model,
        cache_dir=args.cache_dir,
        local_files_only=args.offline,
    )
    active = IndexGenerationStore(args.index_base).active()
    if active is None:
        parser.error("当前没有已发布搜索 generation")
    store = RagIndexStore(args.index_base)
    modes = tuple(dict.fromkeys(mode.strip() for mode in args.modes.split(",") if mode.strip()))
    if not modes or any(mode not in {"lexical", "vector", "hybrid"} for mode in modes):
        parser.error("modes 仅支持 lexical、vector、hybrid")
    retrievers = {
        "lexical": LexicalRetriever(),
        "vector": VectorRetriever(),
        "hybrid": HybridRetriever(),
    }
    results: dict[str, list[dict[str, object]]] = {mode: [] for mode in modes}
    total_hits = {mode: 0 for mode in modes}
    reciprocal_ranks = {mode: 0.0 for mode in modes}

    with store.acquire_matching(active.generation, provider.identity) as handle:
        for case_index, case in enumerate(cases, start=1):
            query = RagQuery(
                question=str(case["question"]),
                root_ids=tuple(int(value) for value in case.get("root_ids", [])),
                extensions=tuple(case.get("extensions") or (".pdf", ".docx", ".txt")),
                include_ocr=bool(case.get("include_ocr", True)),
                min_ocr_confidence=float(case.get("min_ocr_confidence", 0.60)),
                top_k=args.top_k,
            )
            for mode in modes:
                retriever = retrievers[mode]
                if mode == "lexical":
                    hits = retriever.retrieve(handle.connection, query, args.top_k)
                elif mode == "vector":
                    hits = retriever.retrieve(handle.connection, query, provider, _NoCancel(), args.top_k)
                else:
                    ranked = retriever.retrieve(
                        handle.connection,
                        query,
                        provider,
                        _NoCancel(),
                        top_n=args.top_k,
                    )
                    hits = [item.evidence for item in ranked]
                relevant_rank = _first_relevant_rank(hits, case["relevant_sources"])
                if relevant_rank is not None:
                    total_hits[mode] += 1
                    reciprocal_ranks[mode] += 1.0 / relevant_rank
                results[mode].append(
                    {
                        "id": str(case.get("id") or case_index),
                        "question": case["question"],
                        "relevant_rank": relevant_rank,
                        "hit": relevant_rank is not None,
                        "retrieved": [
                            {
                                "stable_chunk_id": item.chunk.stable_chunk_id,
                                "file_path": item.selected_source.file_path,
                                "location_text": item.selected_source.location_text,
                                "page_number": item.selected_source.page_number,
                                "line_start": item.selected_source.line_start,
                                "line_end": item.selected_source.line_end,
                            }
                            for item in hits
                        ],
                    }
                )
            print(f"evaluated {case_index}/{len(cases)}", flush=True)

    summary = {
        "generation": active.generation,
        "model": provider.identity.model_id,
        "model_revision": provider.identity.revision,
        "case_count": len(cases),
        "top_k": args.top_k,
        "metrics": {
            mode: {
                f"recall@{args.top_k}": total_hits[mode] / len(cases),
                "mrr": reciprocal_ranks[mode] / len(cases),
            }
            for mode in modes
        },
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "metrics": summary["metrics"]}, ensure_ascii=False))
    return 0


def _first_relevant_rank(hits, gold_sources: list[object]) -> int | None:
    normalized_gold = [_source_spec(item) for item in gold_sources]
    for rank, hit in enumerate(hits, start=1):
        source = hit.selected_source
        for expected in normalized_gold:
            if expected["path"] and Path(source.file_path).as_posix().casefold() != expected["path"]:
                continue
            if expected["page_number"] is not None and source.page_number != expected["page_number"]:
                continue
            if expected["line_start"] is not None and source.line_start != expected["line_start"]:
                continue
            if expected["line_end"] is not None and source.line_end != expected["line_end"]:
                continue
            return rank
    return None


def _source_spec(value: object) -> dict[str, object]:
    if isinstance(value, str):
        path = value
        result: dict[str, object] = {}
    elif isinstance(value, dict):
        path = str(value.get("path") or "")
        result = value
    else:
        path, result = "", {}
    normalized = Path(path).as_posix().casefold() if path else ""
    return {
        "path": normalized,
        "page_number": _optional_int(result.get("page_number")),
        "line_start": _optional_int(result.get("line_start")),
        "line_end": _optional_int(result.get("line_end")),
    }


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


class _NoCancel:
    cancelled = False
    reason = ""

    def throw_if_cancelled(self) -> None:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
