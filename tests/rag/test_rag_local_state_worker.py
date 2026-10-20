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
