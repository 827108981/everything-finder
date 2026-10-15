from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_full_text_search.config.constants import CACHE_DIR, DB_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.fastembed_local import FastEmbedLocalProvider
from local_full_text_search.rag.index_builder import RagIndexBuilder
from local_full_text_search.rag.models import RagProgress


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the RAG sidecar for the active published search generation."
    )
    parser.add_argument("--index-base", type=Path, default=DB_PATH)
    parser.add_argument("--model", default="BAAI/bge-small-zh-v1.5")
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR / "rag-fastembed")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true", help="use the cached model without network access")
    args = parser.parse_args()

    def report(payload: RagProgress | dict[str, object]) -> None:
        if isinstance(payload, RagProgress):
            value = {
                "stage": payload.stage,
                "message": payload.message,
                "completed": payload.completed,
                "total": payload.total,
                "indeterminate": payload.indeterminate,
            }
        else:
            value = dict(payload)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    try:
        print(json.dumps({"stage": "loading_model", "message": "加载或下载 Embedding 模型"}, ensure_ascii=False), flush=True)
        provider = FastEmbedLocalProvider(
            args.model,
            cache_dir=args.cache_dir,
            threads=args.threads,
            local_files_only=args.offline,
        )
        result = RagIndexBuilder(args.index_base).build(provider, CancelToken(), report)
        print(
            json.dumps(
                {
                    "stage": "published",
                    "generation": result.generation,
                    "chunks": result.chunk_count,
                    "sources": result.source_count,
                    "reused_embeddings": result.reused_embeddings,
                    "generated_embeddings": result.generated_embeddings,
                    "elapsed_ms": result.elapsed_ms,
                    "fts_mode": result.fts_mode,
                    "model": provider.identity.model_id,
                    "model_revision": provider.identity.revision,
                    "dimension": provider.identity.dimension,
                },
                ensure_ascii=False,
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps({"stage": "failed", "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
