from __future__ import annotations

import pytest

chromadb = pytest.importorskip("chromadb")

from local_full_text_search.rag.models import VectorRecord
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


def test_chroma_persists_hnsw_vectors_and_applies_metadata_filters(tmp_path) -> None:
    store = ChromaVectorStore(tmp_path / "vectors", create=True)
    store.upsert(
        [
            VectorRecord("a", (1.0, 0.0, 0.0), {"root_id": 1, "extension": ".pdf", "source_type": "native_text"}),
            VectorRecord("b", (0.0, 1.0, 0.0), {"root_id": 2, "extension": ".txt", "source_type": "ocr"}),
        ]
    )
    assert store.count() == 2
    hits = store.query((1.0, 0.0, 0.0), 5, root_ids=(1,), include_ocr=False)
    assert [item.point_id for item in hits] == ["a"]
    assert hits[0].score == pytest.approx(1.0, abs=1e-4)
    store.close()

    reopened = ChromaVectorStore(tmp_path / "vectors")
    assert reopened.count() == 2
    assert reopened.get_embeddings(["a"])["a"] == pytest.approx([1.0, 0.0, 0.0])
    reopened.close()
