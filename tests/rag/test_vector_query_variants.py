from __future__ import annotations

from types import SimpleNamespace

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.models import RagQuery, SourceBlock
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.vector import VectorRetriever


def _chunk(index: int):
    block = SourceBlock(
        source_generation=1, block_id=index, document_id=index, file_id=index,
        root_id=1, file_path=f"E:/资料/{index}.pdf", filename=f"{index}.pdf",
        extension=".pdf", document_key=f"doc-{index}", block_index=0,
        text=f"P02 泵资料 {index}", location_text="第 1 页", page_number=1,
    )
    return next(ChunkBuilder().iter_chunks([block]))


class _Connection:
    def close(self):
        pass


class _Store:
    def __init__(self, chunks):
        self.chunks = {item.stable_chunk_id: item for item in chunks}

    def connect_read_only(self, _path):
        return _Connection()

    def load_chunk(self, _connection, chunk_id):
        return self.chunks.get(chunk_id)

    def _matches_filters(self, *_args):
        return True


class _Embedding:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return text


class _Vectors:
    def __init__(self, chunks, *, fail_focused=False):
        self.chunks = chunks
        self.fail_focused = fail_focused

    def query(self, vector, *_args, **_kwargs):
        if "是什么" not in vector and self.fail_focused:
            raise OSError("focused route unavailable")
        chunk = self.chunks[0] if "是什么" in vector else self.chunks[1]
        return [SimpleNamespace(point_id=chunk.stable_chunk_id, metadata={}, score=0.8)]


def test_focused_query_merges_inside_one_vector_channel() -> None:
    plan = QueryInterpreter().interpret("P02 是什么泵？")
    assert len(plan.semantic_variants) == 2
    chunks = [_chunk(1), _chunk(2)]
    embedding = _Embedding()
    retriever = VectorRetriever(
        _Store(chunks), SimpleNamespace(metadata_path="unused"),
        _Vectors(chunks), embedding, focused_enabled=True,
    )
    rows = retriever.retrieve(RagQuery(text=plan.original_text, vector_top_k=5), CancelToken(), plan)
    assert len(rows) == 2
    assert {hit.route_id for row in rows for hit in row.vector_route_hits} == {
        "semantic_original", "semantic_focused"
    }
    assert embedding.queries == [variant.query_text for variant in plan.semantic_variants]


def test_failed_focused_query_keeps_original_results_and_records_degrade() -> None:
    plan = QueryInterpreter().interpret("P02 是什么泵？")
    chunks = [_chunk(1), _chunk(2)]
    retriever = VectorRetriever(
        _Store(chunks), SimpleNamespace(metadata_path="unused"),
        _Vectors(chunks, fail_focused=True), _Embedding(), focused_enabled=True,
    )
    rows = retriever.retrieve(RagQuery(text=plan.original_text), CancelToken(), plan)
    assert len(rows) == 1
    assert retriever.route_diagnostics[-1]["status"] == "degraded_route"
