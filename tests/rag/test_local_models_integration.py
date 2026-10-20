from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("LFTS_TEST_REAL_RAG_MODELS") != "1",
    reason="set LFTS_TEST_REAL_RAG_MODELS=1 after downloading the pinned models",
)

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker


def test_downloaded_chinese_embedding_and_cross_encoder_models_run_offline():
    settings = AppSettings()
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding_path = manager.resolve_local_path(
        ModelSpec(
            settings.rag_embedding_repo_id,
            settings.rag_embedding_revision,
            "embedding",
        )
    )
    reranker_path = manager.resolve_local_path(
        ModelSpec(
            settings.rag_reranker_repo_id,
            settings.rag_reranker_revision,
            "reranker",
        )
    )

    embedding = LocalBgeEmbeddingProvider(
        embedding_path,
        repo_id=settings.rag_embedding_repo_id,
        revision=settings.rag_embedding_revision,
    )
    question = "武汉市索引服务的申请条件是什么？"
    relevant_text = "申请人需提交身份证明和项目资料。"
    unrelated_text = "图书馆开放时间为工作日八点至十七点。"
    query_vector = embedding.embed_query(question)
    document_vectors = embedding.embed_documents([relevant_text, unrelated_text])

    import numpy as np

    assert len(query_vector) == 512
    assert float(np.dot(query_vector, document_vectors[0])) > float(
        np.dot(query_vector, document_vectors[1])
    )

    candidates = [
        SimpleNamespace(
            chunk=SimpleNamespace(
                text=text,
                document_key=stable_id,
                stable_chunk_id=stable_id,
            ),
            fused_score=1.0,
            rerank_score=None,
        )
        for stable_id, text in (
            ("relevant", relevant_text),
            ("unrelated", unrelated_text),
        )
    ]
    reranker = LocalBgeReranker(
        reranker_path,
        repo_id=settings.rag_reranker_repo_id,
        revision=settings.rag_reranker_revision,
    )
    ranked = reranker.rerank(question, candidates, 2, CancelToken())

    assert ranked[0].chunk.stable_chunk_id == "relevant"
    assert ranked[0].rerank_score > ranked[1].rerank_score
