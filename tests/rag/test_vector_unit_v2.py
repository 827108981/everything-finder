from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.errors import VectorIndexNotReadyError
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import RagQuery, SourceBlock
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder
from local_full_text_search.rag.vector_index_store import VectorIndexBundle, VectorIndexStore


def _chunk(index: int, text: str, filename: str = "通用设备资料.pdf"):
    block = SourceBlock(
        source_generation=1, block_id=index, document_id=index, file_id=index,
        root_id=1, file_path=f"E:/资料/{filename}", filename=filename,
        extension=".pdf", document_key=f"doc-{index}", block_index=0,
        text=text, location_text="第 3 页", page_number=3,
    )
    return next(ChunkBuilder(target_chars=4000, max_chars=5000).iter_chunks([block]))


def _count_tokens(text: str) -> int:
    # 测试替身把每个非空白字符视作 token，并计入两个特殊 token。
    return len("".join(text.split())) + 2


class _Embedding:
    model_id = "test/embedding"
    model_revision = "revision-1"
    dimension = 3
    count_tokens = staticmethod(_count_tokens)

    @staticmethod
    def embed_documents(texts):
        return [[1.0, float(index + 1), 0.5] for index, _ in enumerate(texts)]

    @staticmethod
    def embed_query(_text):
        return [1.0, 1.0, 0.5]


def test_unit_builder_keeps_adjacent_short_lines_and_never_silently_truncates() -> None:
    chunk = _chunk(1, "模块名称\n连接器\nP02\n制冷循环泵\n额定电压 DC24V\n" + "维护说明。" * 40)
    policy = VectorUnitPolicy(
        target_tokens=36, max_tokens=48, overlap_tokens=8,
        include_source_context=True,
    )
    units = VectorUnitBuilder(_count_tokens, policy).build(chunk)
    assert len(units) > 1
    assert all(item.token_count <= 48 for item in units)
    assert any("P02" in item.text and "制冷循环泵" in item.text for item in units)
    assert any(item.kind == "line_window" for item in units)
    assert all(item.text in chunk.text for item in units)


def test_policy_fingerprint_changes_for_every_semantic_setting() -> None:
    baseline = VectorUnitPolicy()
    assert baseline.fingerprint != replace(baseline, target_tokens=129).fingerprint
    assert baseline.fingerprint != replace(baseline, max_tokens=193).fingerprint
    assert baseline.fingerprint != replace(baseline, overlap_tokens=31).fingerprint
    assert baseline.fingerprint != replace(baseline, include_source_context=False).fingerprint


def test_vector_index_is_bound_to_bundle_model_policy_and_reused(tmp_path) -> None:
    rag_store = RagIndexStore(tmp_path / "rag")
    bundle, connection = rag_store.create_candidate(
        1, embedding_repo_id=_Embedding.model_id,
        embedding_revision=_Embedding.model_revision, embedding_dimension=3,
        reranker_repo_id="test/reranker", reranker_revision="r1",
    )
    chunks = [_chunk(1, "第一段内容。\n第二段事实。"), _chunk(2, "另一份通用资料。")]
    for chunk in chunks:
        rag_store.add_chunk(connection, chunk)
    connection.commit()
    bundle = replace(bundle, chunk_count=len(chunks))
    settings = AppSettings(
        rag_vector_unit_target_tokens=16, rag_vector_unit_max_tokens=32,
        rag_vector_unit_overlap_tokens=4, rag_embedding_batch_size=2,
    )
    try:
        first, reused = VectorIndexBuilder(rag_store, _Embedding(), settings).prepare(
            bundle, CancelToken()
        )
        second, reused_again = VectorIndexBuilder(rag_store, _Embedding(), settings).prepare(
            bundle, CancelToken()
        )
        assert reused is False and reused_again is True
        assert first.index_fingerprint == second.index_fingerprint
        assert first.parent_chunk_count == 2
        assert first.unit_count >= 2
        isolated_root = tmp_path / "candidate-model"
        isolated, isolated_reused = VectorIndexBuilder(
            rag_store, _Embedding(), settings, vector_root=isolated_root,
        ).prepare(bundle, CancelToken())
        assert isolated_reused is False
        assert isolated.path.is_relative_to(isolated_root / "vector")
        assert isolated.index_fingerprint == first.index_fingerprint
        changed_policy = VectorUnitPolicy(17, 32, 4, True)
        with pytest.raises(VectorIndexNotReadyError):
            VectorIndexStore(rag_store.root).active(
                bundle.bundle_id, embedding_repo_id=_Embedding.model_id,
                embedding_revision=_Embedding.model_revision, embedding_dimension=3,
                policy_fingerprint=changed_policy.fingerprint,
            )
    finally:
        connection.close()


class _Connection:
    def close(self):
        pass


class _ParentStore:
    def __init__(self, chunks):
        self.chunks = {item.stable_chunk_id: item for item in chunks}

    def connect_read_only(self, _path):
        return _Connection()

    def load_chunk(self, _connection, chunk_id):
        return self.chunks.get(chunk_id)

    def _matches_filters(self, *_args):
        return True


class _UnitVectors:
    def __init__(self, parent_ids):
        self.parent_ids = parent_ids

    def query(self, *_args, **_kwargs):
        return [
            SimpleNamespace(point_id="u1", score=0.91, metadata={"stable_chunk_id": self.parent_ids[0]}),
            SimpleNamespace(point_id="u2", score=0.88, metadata={"stable_chunk_id": self.parent_ids[0]}),
            SimpleNamespace(point_id="u3", score=0.82, metadata={"stable_chunk_id": self.parent_ids[1]}),
        ]

    def count(self):
        return 3


def test_unit_hits_collapse_to_parent_rank_and_remain_traceable(tmp_path) -> None:
    chunks = [_chunk(1, "父正文一。"), _chunk(2, "父正文二。")]
    database = tmp_path / "vector_units.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE vector_units(unit_id TEXT,parent_chunk_id TEXT,ordinal INTEGER,kind TEXT,"
        "text TEXT,embedding_text TEXT,start_char INTEGER,end_char INTEGER,token_count INTEGER)"
    )
    connection.executemany("INSERT INTO vector_units VALUES(?,?,?,?,?,?,?,?,?)", [
        ("u1", chunks[0].stable_chunk_id, 0, "line_window", "局部事实 A", "局部事实 A", 0, 6, 8),
        ("u2", chunks[0].stable_chunk_id, 1, "line_window", "局部事实 B", "局部事实 B", 7, 13, 8),
        ("u3", chunks[1].stable_chunk_id, 0, "sentence_window", "另一事实", "另一事实", 0, 4, 6),
    ])
    connection.commit()
    connection.close()
    unit_index = VectorIndexBundle(
        "bundle", "digest", "test", "rev", 3, "policy", "index", tmp_path,
        database, tmp_path / "chroma", 2, 3, "now",
    )
    retriever = VectorRetriever(
        _ParentStore(chunks), SimpleNamespace(metadata_path="unused"),
        _UnitVectors([item.stable_chunk_id for item in chunks]), _Embedding(),
        backend="unit_v2", unit_index=unit_index,
    )
    rows = retriever.retrieve(RagQuery(text="问题", vector_top_k=2), CancelToken())
    assert [item.vector_rank for item in rows] == [1, 2]
    assert len(rows[0].vector_unit_hits) == 2
    assert rows[0].vector_score == pytest.approx(0.91)
    snapshot = candidate_snapshot(rows[0])
    assert snapshot["vector_unit_hits"][0]["text"] == "局部事实 A"
    assert snapshot["vector_unit_hits"][1]["raw_rank"] == 2


def test_unit_search_deepens_until_enough_distinct_parents(tmp_path) -> None:
    chunks = [_chunk(1, "父正文一。"), _chunk(2, "父正文二。")]
    database = tmp_path / "vector_units.db"
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE vector_units(unit_id TEXT,parent_chunk_id TEXT,ordinal INTEGER,kind TEXT,"
        "text TEXT,embedding_text TEXT,start_char INTEGER,end_char INTEGER,token_count INTEGER)"
    )
    connection.executemany("INSERT INTO vector_units VALUES(?,?,?,?,?,?,?,?,?)", [
        ("u1", chunks[0].stable_chunk_id, 0, "window", "一", "一", 0, 1, 3),
        ("u2", chunks[0].stable_chunk_id, 1, "window", "二", "二", 1, 2, 3),
        ("u3", chunks[1].stable_chunk_id, 0, "window", "三", "三", 0, 1, 3),
    ])
    connection.commit()
    connection.close()
    unit_index = VectorIndexBundle(
        "bundle", "digest", "test", "rev", 3, "policy", "index", tmp_path,
        database, tmp_path / "chroma", 2, 3, "now",
    )

    class _CollidingVectors(_UnitVectors):
        requested = []

        def query(self, _vector, top_k, **_kwargs):
            self.requested.append(top_k)
            rows = super().query()
            return rows[:top_k]

    vectors = _CollidingVectors([item.stable_chunk_id for item in chunks])
    retriever = VectorRetriever(
        _ParentStore(chunks), SimpleNamespace(metadata_path="unused"), vectors,
        _Embedding(), backend="unit_v2", unit_index=unit_index, oversample_factor=1,
    )
    rows = retriever.retrieve(RagQuery(text="问题", vector_top_k=2), CancelToken())
    assert len(rows) == 2
    assert vectors.requested == [2, 3]
    assert retriever.route_diagnostics[0]["parent_candidate_count"] == 2
