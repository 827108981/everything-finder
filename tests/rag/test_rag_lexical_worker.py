from __future__ import annotations

from types import SimpleNamespace

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.workers import rag_index_worker as worker_module


def test_unchanged_ai_bundle_prepares_lexical_index(monkeypatch, tmp_path) -> None:
    bundle = SimpleNamespace(bundle_id="g000033-source")
    summary = SimpleNamespace(bundle_id=bundle.bundle_id, unchanged=True)
    calls = []

    class _Models:
        def __init__(self, _cache_dir):
            pass

        def resolve_local_path(self, _spec):
            return tmp_path

    class _Embedding:
        def __init__(self, *_args, **_kwargs):
            self.model_id = "test-model"
            self.model_revision = "test-revision"
            self.batch_size = 1

    class _RagStore:
        def __init__(self, _root):
            pass

        def active(self):
            return bundle

    class _Builder:
        def __init__(self, *_args):
            pass

        def build(self, *_args, **_kwargs):
            return summary

    class _LexicalBuilder:
        def __init__(self, _store):
            pass

        def prepare(self, source, _token):
            calls.append(source.bundle_id)
            return SimpleNamespace(
                source_bundle_id=source.bundle_id, index_fingerprint="index-v2",
                chunk_count=3,
            ), True

    monkeypatch.setattr(worker_module, "LocalModelManager", _Models)
    monkeypatch.setattr(worker_module, "LocalBgeEmbeddingProvider", _Embedding)
    monkeypatch.setattr(worker_module, "RagIndexStore", _RagStore)
    monkeypatch.setattr(worker_module, "RagIndexBuilder", _Builder)
    monkeypatch.setattr(worker_module, "LexicalIndexBuilder", _LexicalBuilder)
    monkeypatch.setattr(worker_module, "IndexGenerationStore", lambda _path: object())

    worker = worker_module.RagIndexWorker(
        tmp_path / "search.db", AppSettings(rag_lexical_backend="word_v2"),
    )
    finished = []
    failed = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)
    worker.run()

    assert calls == [bundle.bundle_id]
    assert finished == [summary]
    assert failed == []
