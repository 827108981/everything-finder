from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

import local_full_text_search.rag.index_builder as index_builder_module
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_builder import RagIndexBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import RagChunk, SourceRef
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


@dataclass(frozen=True)
class _Generation:
    generation: int
    path: Path


class _SourceStore:
    def __init__(self, path: Path) -> None:
        self.current = _Generation(3, path)

    @contextmanager
    def acquire_active(self):
        yield self.current, object()

    def active(self):
        return self.current


class _BlockReader:
    def iter_blocks(self, *_args, **_kwargs):
        return iter((object(),))


class _ChunkBuilder:
    def __init__(self, count: int) -> None:
        self.count = count
        self.generation = 3
        self.id_prefix = "chunk"
        self.text_overrides: dict[int, str] = {}

    def iter_chunks(self, _blocks):
        for index in range(self.count):
            stable_id = f"{self.id_prefix}-{index:04d}"
            source = SourceRef(
                source_generation=self.generation,
                stable_chunk_id=stable_id,
                document_id=1,
                file_id=1,
                root_id=1,
                file_path="C:/资料/批次测试.txt",
                filename="批次测试.txt",
                extension=".txt",
                block_ids=(index + 1,),
                location_text=f"第 {index + 1} 段",
            )
            text = self.text_overrides.get(index, f"用于验证批次写入的正文块 {index}")
            yield RagChunk(
                stable_chunk_id=stable_id,
                document_key="batch-test-document",
                ordinal=index,
                text=text,
                normalized_text=text,
                text_hash=f"hash-{index:04d}",
                source=source,
            )


class _Embedding:
    model_id = "test/embedding"
    model_revision = "a" * 40
    dimension = 3

    def __init__(self) -> None:
        self.call_sizes: list[int] = []

    def embed_documents(self, texts):
        self.call_sizes.append(len(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


def test_index_build_separates_embedding_and_write_batch_sizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_sizes: list[int] = []

    class _RecordingVectorStore:
        def __init__(self, *args, **kwargs) -> None:
            self.inner = ChromaVectorStore(*args, **kwargs)

        def upsert(self, records) -> None:
            write_sizes.append(len(records))
            self.inner.upsert(records)

        def count(self) -> int:
            return self.inner.count()

        def close(self) -> None:
            self.inner.close()

    monkeypatch.setattr(
        index_builder_module, "ChromaVectorStore", _RecordingVectorStore
    )
    settings = AppSettings(
        rag_embedding_batch_size=32,
        rag_index_write_batch_size=128,
    )
    embedding = _Embedding()
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    builder = RagIndexBuilder(
        source_store,
        rag_store,
        embedding,
        settings,
        block_reader=_BlockReader(),
        chunk_builder=_ChunkBuilder(130),
    )

    summary = builder.build(CancelToken(), run_id="test-rag-batching")

    assert embedding.call_sizes == [32, 32, 32, 32, 2]
    assert write_sizes == [128, 2]
    assert summary.chunk_count == summary.vector_count == 130
    assert rag_store.active().source_generation == 3


def test_unchanged_source_reuses_published_bundle_without_embedding(tmp_path: Path) -> None:
    settings = AppSettings(rag_embedding_batch_size=2, rag_index_write_batch_size=2)
    embedding = _Embedding()
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    chunks = _ChunkBuilder(3)
    builder = RagIndexBuilder(
        source_store, rag_store, embedding, settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    )
    first = builder.build(CancelToken())
    assert embedding.call_sizes == [2, 1]

    # 全文 generation 变了，但实际正文和引用元数据都没有变化。
    source_store.current = _Generation(4, source_store.current.path)
    chunks.generation = 4
    progress = []
    second = builder.build(CancelToken(), progress.append)

    assert second.bundle_id == first.bundle_id
    assert second.source_generation == 4
    assert embedding.call_sizes == [2, 1]
    assert rag_store.active().verified_source_generation == 4
    assert progress[-1]["message"] == "资料未变化，沿用已有 AI 索引"
    assert len(list((tmp_path / "rag").glob("rag.g*"))) == 1


def test_changed_chunk_only_embeds_new_text(tmp_path: Path) -> None:
    settings = AppSettings(rag_embedding_batch_size=2, rag_index_write_batch_size=2)
    embedding = _Embedding()
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    chunks = _ChunkBuilder(3)
    builder = RagIndexBuilder(
        source_store, rag_store, embedding, settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    )
    first = builder.build(CancelToken())
    original_calls = list(embedding.call_sizes)
    chunks.text_overrides[1] = "这是修改后的正文，必须重新计算向量"

    second = builder.build(CancelToken())

    assert second.bundle_id != first.bundle_id
    assert second.chunk_count == second.vector_count == 3
    assert embedding.call_sizes[len(original_calls):] == [1]
    assert rag_store.active().bundle_id == second.bundle_id
    old_store = ChromaVectorStore(Path(first.bundle_path) / "chroma")
    new_store = ChromaVectorStore(Path(second.bundle_path) / "chroma")
    try:
        assert new_store.get_embeddings(["chunk-0000"])["chunk-0000"] == (
            old_store.get_embeddings(["chunk-0000"])["chunk-0000"]
        )
    finally:
        old_store.close()
        new_store.close()


def test_changed_chunk_ids_reuse_vectors_by_exact_text(tmp_path: Path) -> None:
    settings = AppSettings(rag_embedding_batch_size=2, rag_index_write_batch_size=2)
    embedding = _Embedding()
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    chunks = _ChunkBuilder(3)
    builder = RagIndexBuilder(
        source_store, rag_store, embedding, settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    )
    builder.build(CancelToken())
    call_count = len(embedding.call_sizes)
    chunks.id_prefix = "new-id"

    summary = builder.build(CancelToken())

    assert len(embedding.call_sizes) == call_count
    assert summary.chunk_count == 3
    store = ChromaVectorStore(Path(summary.bundle_path) / "chroma")
    try:
        assert store.count() == 3
        assert len(store.get_embeddings(["new-id-0000"])) == 1
    finally:
        store.close()


def test_embedding_revision_change_forces_full_reembedding(tmp_path: Path) -> None:
    settings = AppSettings(rag_embedding_batch_size=2, rag_index_write_batch_size=2)
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    chunks = _ChunkBuilder(3)
    RagIndexBuilder(
        source_store, rag_store, _Embedding(), settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    ).build(CancelToken())

    class _NewRevisionEmbedding(_Embedding):
        model_revision = "b" * 40

    embedding = _NewRevisionEmbedding()
    RagIndexBuilder(
        source_store, rag_store, embedding, settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    ).build(CancelToken())

    assert embedding.call_sizes == [2, 1]
    assert rag_store.active().embedding_revision == "b" * 40


def test_cancelled_incremental_build_preserves_published_bundle(tmp_path: Path) -> None:
    settings = AppSettings(rag_embedding_batch_size=2, rag_index_write_batch_size=2)
    source_store = _SourceStore(tmp_path / "published-source.db")
    rag_store = RagIndexStore(tmp_path / "rag")
    chunks = _ChunkBuilder(3)
    first = RagIndexBuilder(
        source_store, rag_store, _Embedding(), settings,
        block_reader=_BlockReader(), chunk_builder=chunks,
    ).build(CancelToken())
    chunks.text_overrides[1] = "取消前需要生成向量的新内容"
    token = CancelToken()

    class _CancellingEmbedding(_Embedding):
        def embed_documents(self, texts):
            token.cancel()
            return super().embed_documents(texts)

    with pytest.raises(CancelledError):
        RagIndexBuilder(
            source_store, rag_store, _CancellingEmbedding(), settings,
            block_reader=_BlockReader(), chunk_builder=chunks,
        ).build(token)

    assert rag_store.active().bundle_id == first.bundle_id
    # Windows 上 Chroma 可能暂时持有候选文件句柄；关键是不切换发布清单。
    assert rag_store.active().path == Path(first.bundle_path)
