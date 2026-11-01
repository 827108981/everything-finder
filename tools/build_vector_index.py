"""为当前已发布的 AI bundle 构建或复用向量单元 V2 附属索引。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder
from local_full_text_search.services.settings_service import SettingsService


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-base-path", type=Path, required=True)
    parser.add_argument("--expected-bundle-id", default="")
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--overlap-tokens", type=int, default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=None)
    parser.add_argument("--torch-threads", type=int, default=None,
                        help="仅用于本次构建的 PyTorch CPU 线程数；不修改应用设置")
    parser.add_argument("--embedding-repo-id", default=None,
                        help="候选 Embedding 模型；仅用于本次构建，不修改应用设置")
    parser.add_argument("--embedding-revision", default=None,
                        help="候选 Embedding 固定 revision")
    parser.add_argument("--body-only", action="store_true",
                        help="实验：向量只编码原正文，不附加文件名和位置；配合隔离输出目录使用")
    parser.add_argument("--vector-output-root", type=Path, default=None,
                        help="向量索引隔离根目录；不指定时使用生产 AI bundle 的 vector 目录")
    args = parser.parse_args()

    try:
        settings = SettingsService().load()
    except Exception:
        settings = AppSettings()
    if args.target_tokens is not None:
        settings.rag_vector_unit_target_tokens = args.target_tokens
    if args.max_tokens is not None:
        settings.rag_vector_unit_max_tokens = args.max_tokens
    if args.overlap_tokens is not None:
        settings.rag_vector_unit_overlap_tokens = args.overlap_tokens
    if args.embedding_batch_size is not None:
        settings.rag_embedding_batch_size = args.embedding_batch_size
    if args.embedding_repo_id is not None:
        settings.rag_embedding_repo_id = args.embedding_repo_id
    if args.embedding_revision is not None:
        settings.rag_embedding_revision = args.embedding_revision
    if args.torch_threads is not None:
        import torch

        torch.set_num_threads(max(1, int(args.torch_threads)))
    if args.body_only:
        if args.vector_output_root is None:
            raise ValueError("--body-only 必须指定 --vector-output-root，避免改动生产索引指针")
        settings.rag_vector_unit_include_source_context = False
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    model_path = manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding",
    ))
    embedding = LocalBgeEmbeddingProvider(
        model_path, repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    rag_store = RagIndexStore(args.generation_base_path.expanduser().resolve().parent / "rag")
    source = rag_store.active()
    if args.expected_bundle_id and source.bundle_id != args.expected_bundle_id:
        raise ValueError("实际 AI 索引 bundle_id 与指定值不一致，已停止构建")
    reported_bucket = -1

    def report(progress: dict[str, object]) -> None:
        nonlocal reported_bucket
        percent = int(progress.get("percent") or 0)
        bucket = percent // 5
        if bucket > reported_bucket or progress.get("reused") is not None:
            reported_bucket = bucket
            print(json.dumps(progress, ensure_ascii=False), flush=True)

    bundle, reused = VectorIndexBuilder(
        rag_store, embedding, settings,
        vector_root=args.vector_output_root.expanduser().resolve() if args.vector_output_root else None,
    ).prepare(
        source, CancelToken(), report,
    )
    print(json.dumps({
        "source_bundle_id": bundle.source_bundle_id,
        "index_fingerprint": bundle.index_fingerprint,
        "policy_fingerprint": bundle.policy_fingerprint,
        "parent_chunk_count": bundle.parent_chunk_count,
        "unit_count": bundle.unit_count,
        "database_path": str(bundle.database_path),
        "reused": reused,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
