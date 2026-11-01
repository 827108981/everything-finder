"""为当前已发布的 AI bundle 构建或复用关键词 V2 附属索引。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.lexical_index_builder import LexicalIndexBuilder


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-base-path", type=Path, required=True,
        help="普通索引数据库路径；AI 索引位于其同级 rag 目录",
    )
    parser.add_argument("--expected-bundle-id", default="")
    args = parser.parse_args()
    rag_store = RagIndexStore(args.generation_base_path.expanduser().resolve().parent / "rag")
    source = rag_store.active()
    if args.expected_bundle_id and source.bundle_id != args.expected_bundle_id:
        raise ValueError("实际 AI 索引 bundle_id 与指定值不一致，已停止构建")
    bundle, reused = LexicalIndexBuilder(rag_store).prepare(source, CancelToken())
    print(json.dumps({
        "source_bundle_id": bundle.source_bundle_id,
        "index_fingerprint": bundle.index_fingerprint,
        "analyzer_fingerprint": bundle.analyzer_fingerprint,
        "chunk_count": bundle.chunk_count,
        "database_path": str(bundle.database_path),
        "reused": reused,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
