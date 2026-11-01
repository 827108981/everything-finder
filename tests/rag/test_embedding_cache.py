from __future__ import annotations

import sqlite3

import pytest

from local_full_text_search.rag.embedding_cache import EmbeddingVectorCache


def test_cache_reuses_exact_text_only_with_same_model_revision_and_dimension(tmp_path) -> None:
    path = tmp_path / "embedding_cache.db"
    first = EmbeddingVectorCache(
        path, model_id="model-a", model_revision="rev-1", dimension=3,
    )
    first.put_many({"相同正文": [1.0, 0.5, -0.25]})
    assert first.get_many(["相同正文", "其他正文"])["相同正文"] == pytest.approx(
        [1.0, 0.5, -0.25]
    )
    first.close()

    other_revision = EmbeddingVectorCache(
        path, model_id="model-a", model_revision="rev-2", dimension=3,
    )
    assert other_revision.get_many(["相同正文"]) == {}
    other_revision.close()

    other_dimension = EmbeddingVectorCache(
        path, model_id="model-a", model_revision="rev-1", dimension=2,
    )
    assert other_dimension.get_many(["相同正文"]) == {}
    other_dimension.close()


def test_cache_does_not_store_plaintext_and_ignores_invalid_blob(tmp_path) -> None:
    path = tmp_path / "embedding_cache.db"
    cache = EmbeddingVectorCache(
        path, model_id="model-a", model_revision="rev-1", dimension=3,
    )
    cache.put_many({"敏感资料正文": [1.0, 2.0, 3.0]})
    digest = cache.text_hash("敏感资料正文")
    cache.connection.execute(
        "UPDATE embedding_cache SET vector_blob=? WHERE text_sha256=?",
        (sqlite3.Binary(b"broken"), digest),
    )
    cache.connection.commit()
    assert cache.get_many(["敏感资料正文"]) == {}
    cache.close()
    assert "敏感资料正文".encode("utf-8") not in path.read_bytes()


def test_cache_rejects_invalid_vectors(tmp_path) -> None:
    cache = EmbeddingVectorCache(
        tmp_path / "embedding_cache.db",
        model_id="model-a",
        model_revision="rev-1",
        dimension=3,
    )
    with pytest.raises(ValueError):
        cache.put_many({"正文": [1.0, 2.0]})
    with pytest.raises(ValueError):
        cache.put_many({"正文": [1.0, float("nan"), 3.0]})
    cache.close()
