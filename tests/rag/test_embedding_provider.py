from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider


class _FakeSentenceTransformer:
    def __init__(self, _path, device="cpu") -> None:
        self.device = device
        self.max_seq_length = 2048
        self.encoded_texts = []
        self.tokenizer = lambda text, **_kwargs: {"input_ids": list(range(len(text) + 2))}

    @staticmethod
    def get_embedding_dimension() -> int:
        return 1024

    def encode(self, texts, **_kwargs):
        self.encoded_texts.append(list(texts))
        return np.ones((len(texts), 1024), dtype=np.float32)


def _install_fake(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=_FakeSentenceTransformer),
    )


def test_embedding_provider_accepts_model_dimension_instead_of_hardcoding_512(
    tmp_path, monkeypatch,
) -> None:
    _install_fake(monkeypatch)
    provider = LocalBgeEmbeddingProvider(
        tmp_path, repo_id="BAAI/bge-large-zh-v1.5", revision="fixed",
    )
    assert provider.dimension == 1024
    assert len(provider.embed_query("测试问题")) == 1024
    assert len(provider.embed_documents(["正文一", "正文二"])) == 2


def test_qwen3_and_bge_use_their_respective_query_instructions(tmp_path, monkeypatch) -> None:
    _install_fake(monkeypatch)
    qwen = LocalBgeEmbeddingProvider(
        tmp_path, repo_id="Qwen/Qwen3-Embedding-0.6B", revision="fixed",
    )
    qwen.embed_query("如何校准？")
    assert qwen._model.encoded_texts[-1][0] == (
        "Instruct: Given a question, retrieve passages that answer the question\n"
        "Query: 如何校准？"
    )

    bge = LocalBgeEmbeddingProvider(
        tmp_path, repo_id="BAAI/bge-large-zh-v1.5", revision="fixed",
    )
    bge.embed_query("如何校准？")
    assert bge._model.encoded_texts[-1][0] == "为这个句子生成表示以用于检索相关文章：如何校准？"


def test_document_embedding_reduces_batch_after_memory_error(tmp_path, monkeypatch) -> None:
    _install_fake(monkeypatch)
    provider = LocalBgeEmbeddingProvider(
        tmp_path, repo_id="BAAI/bge-large-zh-v1.5", revision="fixed", batch_size=8,
    )
    attempted = []

    def encode(texts, **kwargs):
        attempted.append(int(kwargs.get("batch_size") or 0))
        if kwargs.get("batch_size") == 8:
            raise RuntimeError("DefaultCPUAllocator: not enough memory")
        return np.ones((len(texts), 1024), dtype=np.float32)

    provider._model.encode = encode
    vectors = provider.embed_documents(["正文一", "正文二", "正文三"])

    assert len(vectors) == 3
    assert attempted == [8, 4]
    assert provider.batch_size == 4
