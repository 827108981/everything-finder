"""使用本机已发布 AI 索引和已缓存模型执行可复现的纯检索评测。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR, SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.runner import RagEvalConfig, RagEvalRunner
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-db", type=Path, required=True, help="普通索引 search_index.db 基路径")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--lexical-multi-route", action="store_true")
    parser.add_argument("--vector-focused", action="store_true")
    parser.add_argument("--case-ids", nargs="*", default=[])
    args = parser.parse_args()

    # 只读本机配置；纯检索模式不会读取 API Key 或调用 LLM。
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    settings.rag_lexical_multi_route_enabled = args.lexical_multi_route
    settings.rag_vector_focused_query_enabled = args.vector_focused
    settings.rag_answer_review_enabled = False
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding_path = manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding",
    ))
    reranker_path = manager.resolve_local_path(ModelSpec(
        settings.rag_reranker_repo_id, settings.rag_reranker_revision, "reranker",
    ))
    embedding = LocalBgeEmbeddingProvider(
        embedding_path, repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision, batch_size=settings.rag_embedding_batch_size,
    )
    reranker = LocalBgeReranker(
        reranker_path, repo_id=settings.rag_reranker_repo_id,
        revision=settings.rag_reranker_revision, batch_size=settings.rag_reranker_batch_size,
        max_length=settings.rag_reranker_max_length,
    )
    token = CancelToken()
    runner = RagEvalRunner(args.index_db, settings, token)
    config = RagEvalConfig(
        dataset_path=args.cases, mode="retrieval-only", output_root=args.out,
        case_ids=tuple(args.case_ids),
    )
    prepared = runner.prepare(config)
    print(f"预检通过：{len(prepared.cases)} 题，bundle={prepared.bundle_id}", flush=True)

    def service_factory(callback):
        return AnswerService(
            IndexGenerationStore(args.index_db),
            RagIndexStore(args.index_db.parent / "rag"),
            embedding, reranker, OpenAICompatibleGateway(max_transient_retries=0),
            WindowsCredentialSecretStore(), settings, progress_callback=callback,
        )

    result = runner.run(config, prepared, service_factory)
    print(json.dumps({
        "run_dir": str(result.get("run_dir") or ""),
        "state": result.get("state"),
        "completed_case_count": len(result.get("results") or []),
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
