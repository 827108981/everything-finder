from __future__ import annotations

from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import SourceBlock


def _chunk():
    block = SourceBlock(
        source_generation=7,
        block_id=1,
        document_id=1,
        file_id=1,
        root_id=1,
        file_path="/資料/索引恢复.txt",
        filename="索引恢复.txt",
        extension=".txt",
        document_key="doc-1",
        block_index=1,
        text="索引中断后，可以检查检查点并继续处理。",
        location_text="第 1 行",
        line_start=1,
        line_end=1,
    )
    return next(iter(ChunkBuilder().iter_chunks([block])))


def test_manifest_publishes_matching_sqlite_bundle(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    candidate, connection = store.create_candidate(
        7,
        embedding_repo_id="test/embedding",
        embedding_revision="a" * 40,
        embedding_dimension=512,
        reranker_repo_id="test/reranker",
        reranker_revision="b" * 40,
    )
    store.add_chunk(connection, _chunk())
    store.add_vector_ref(connection, _chunk().stable_chunk_id)
    candidate.chroma_path.joinpath("placeholder").write_text("test")
    published = store.publish(candidate, connection, vector_count=1)

    active = store.active()
    assert active.bundle_id == published.bundle_id
    assert active.source_generation == 7
    assert active.verified_source_generation == 7
    assert active.chunk_count == active.vector_count == 1
    assert active.metadata_path.is_file()

    verified = store.mark_source_compatible(active, 8)
    assert verified.source_generation == 7
    assert store.active().verified_source_generation == 8


def test_lexical_retrieval_hydrates_source_and_short_query(tmp_path) -> None:
    store = RagIndexStore(tmp_path / "rag")
    candidate, connection = store.create_candidate(
        7,
        embedding_repo_id="test/embedding",
        embedding_revision="a" * 40,
        embedding_dimension=512,
        reranker_repo_id="test/reranker",
        reranker_revision="b" * 40,
    )
    chunk = _chunk()
    store.add_chunk(connection, chunk)
    connection.commit()
    try:
        results = store.lexical_search(connection, "索引", top_k=5)
        assert len(results) == 1
        assert results[0].chunk.source.file_path == "/資料/索引恢复.txt"
        assert results[0].lexical_rank == 1
    finally:
        store.abort(candidate, connection)
