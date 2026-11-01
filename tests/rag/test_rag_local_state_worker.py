from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.workers import rag_local_state_worker as worker_module


def test_startup_state_reuses_cached_models_and_published_chroma(monkeypatch, tmp_path) -> None:
    class _Models:
        def __init__(self, _cache_dir):
            pass

        def resolve_local_path(self, spec):
            return tmp_path / spec.kind

    bundle = SimpleNamespace(
        bundle_id="g000008-existing",
        chroma_path=tmp_path / "chroma",
        chunk_count=3,
        vector_count=3,
        source_generation=8,
        verified_source_generation=8,
        embedding_repo_id="BAAI/bge-small-zh-v1.5",
        embedding_revision="7999e1d3359715c523056ef9478215996d62a620",
        embedding_dimension=512,
    )

    class _RagStore:
        def __init__(self, _root):
            pass

        def active(self):
            return bundle

    class _Vectors:
        def __init__(self, _path):
            pass

        def count(self):
            return 3

        def close(self):
            pass

    class _SourceStore:
        def __init__(self, _path):
            pass

        def active(self):
            return SimpleNamespace(generation=8)

    monkeypatch.setattr(worker_module, "LocalModelManager", _Models)
    monkeypatch.setattr(worker_module, "RagIndexStore", _RagStore)
    monkeypatch.setattr(worker_module, "ChromaVectorStore", _Vectors)
    monkeypatch.setattr(worker_module, "IndexGenerationStore", _SourceStore)

    outcomes = []
    worker = worker_module.RagLocalStateWorker(
        Path(tmp_path / "search.db"), AppSettings(rag_enabled=True)
    )
    worker.finished.connect(outcomes.append)
    worker.run()

    assert outcomes[0]["models_ready"] is True
    assert outcomes[0]["index_ready"] is True
    assert outcomes[0]["bundle_id"] == "g000008-existing"
    assert outcomes[0]["vector_count"] == 3
    assert outcomes[0]["configured_embedding_dimension"] == 512
    assert outcomes[0]["index_embedding_model"] == "BAAI/bge-small-zh-v1.5"


def test_startup_state_accepts_manifest_attached_unit_v2_without_parent_vectors(
    monkeypatch, tmp_path,
) -> None:
    class _Models:
        def __init__(self, _cache_dir):
            pass

        def resolve_local_path(self, spec):
            return tmp_path / spec.kind

    bundle = SimpleNamespace(
        bundle_id="g000033-unit-v2",
        chroma_path=tmp_path / "unused-parent-chroma",
        chunk_count=11_769,
        vector_count=0,
        source_generation=33,
        verified_source_generation=33,
        embedding_repo_id="BAAI/bge-large-zh-v1.5",
        embedding_revision="79e7739b6ab944e86d6171e44d24c997fc1e0116",
        embedding_dimension=1024,
        parent_vector_mode="metadata_only",
        vector_unit_directory="attached-unit-index",
        vector_unit_index_fingerprint="index-fingerprint",
    )

    class _RagStore:
        def __init__(self, _root):
            pass

        def active(self):
            return bundle

    class _UnitStore:
        def __init__(self, _root):
            pass

        def attached(self, *_args, **_kwargs):
            return SimpleNamespace(
                chroma_path=tmp_path / "unit-chroma", unit_count=52_282,
            )

    class _Vectors:
        def __init__(self, path):
            self.path = path

        def count(self):
            assert self.path == tmp_path / "unit-chroma"
            return 52_282

        def close(self):
            pass

    class _SourceStore:
        def __init__(self, _path):
            pass

        def active(self):
            return SimpleNamespace(generation=33)

    monkeypatch.setattr(worker_module, "LocalModelManager", _Models)
    monkeypatch.setattr(worker_module, "RagIndexStore", _RagStore)
    monkeypatch.setattr(worker_module, "VectorIndexStore", _UnitStore)
    monkeypatch.setattr(worker_module, "ChromaVectorStore", _Vectors)
    monkeypatch.setattr(worker_module, "IndexGenerationStore", _SourceStore)
    monkeypatch.setattr(
        worker_module, "get_lexical_analyzer",
        lambda: SimpleNamespace(fingerprint="analyzer-v2"),
    )

    class _LexicalStore:
        def __init__(self, _root):
            pass

        def active(self, bundle_id, fingerprint):
            assert (bundle_id, fingerprint) == (bundle.bundle_id, "analyzer-v2")

    monkeypatch.setattr(worker_module, "LexicalIndexStore", _LexicalStore)
    settings = AppSettings(
        rag_enabled=True,
        rag_vector_backend="unit_v2",
        rag_lexical_backend="word_v2",
        rag_embedding_repo_id=bundle.embedding_repo_id,
        rag_embedding_revision=bundle.embedding_revision,
    )
    outcomes = []
    worker = worker_module.RagLocalStateWorker(tmp_path / "search.db", settings)
    worker.finished.connect(outcomes.append)
    worker.run()

    assert outcomes[0]["index_ready"] is True
    assert outcomes[0]["chunk_count"] == 11_769
    assert outcomes[0]["vector_count"] == 52_282

    class _MissingLexicalStore(_LexicalStore):
        def active(self, _bundle_id, _fingerprint):
            raise RuntimeError("关键词 V2 索引未就绪")

    monkeypatch.setattr(worker_module, "LexicalIndexStore", _MissingLexicalStore)
    missing = []
    worker = worker_module.RagLocalStateWorker(tmp_path / "search.db", settings)
    worker.finished.connect(missing.append)
    worker.run()
    assert missing[0]["index_ready"] is False
    assert "关键词 V2 索引未就绪" in missing[0]["index_error"]
