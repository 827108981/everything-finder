from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from local_full_text_search.core.compact_snapshot_builder import build_compact_snapshot
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.config.constants import CACHE_DIR
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.mock import MockEmbeddingProvider
from local_full_text_search.rag.embeddings.fastembed_local import FastEmbedLocalProvider
from local_full_text_search.rag.generation.mock import MockLLMGateway
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_builder import RagIndexBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import CredentialStore
from local_full_text_search.rag.models import RagIndexNotReadyError, RagQuery
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.retrieval.vector import VectorRetriever


def _publish_search_generation(tmp_path: Path) -> tuple[Path, int, int]:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    candidate = store.begin_candidate()
    db = DatabaseManager(candidate.path)
    db.initialize()
    root = tmp_path / "indexed"
    root.mkdir()
    root_id = db.add_root(root)
    path = root / "recovery.txt"
    path.write_text("fixture", encoding="utf-8")
    file_id, _changed = db.upsert_file_metadata(root_id, path)
    db.replace_file_blocks(
        file_id,
        path.name,
        str(path),
        [
            ContentBlock(
                file_path=str(path),
                block_index=0,
                block_type="text",
                location_text="第 1-3 行",
                raw_text=(
                    "索引中断后可以从上一个已发布 generation 恢复。\n"
                    "系统会检查候选数据库，通过 readiness 校验后再原子发布。\n"
                    "发生错误时旧版索引仍可搜索。"
                ),
                normalized_text="索引中断后可以从上一个已发布 generation 恢复。",
                line_start=1,
                line_end=3,
            )
        ],
        parser_name="text",
    )
    published = store.publish(
        candidate,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-10-15T00:00:00+00:00",
    )
    return base, published.generation, root_id


def test_mock_embedding_builds_generation_aligned_sidecar_and_hybrid_answer(tmp_path: Path) -> None:
    base, generation, root_id = _publish_search_generation(tmp_path)
    embedding = MockEmbeddingProvider(dimension=48)
    report = RagIndexBuilder(base).build(embedding, CancelToken())
    assert report.generation == generation
    assert report.chunk_count >= 1
    assert report.generated_embeddings == report.chunk_count

    active = RagIndexStore(base).active()
    assert active is not None
    assert active.generation == generation
    assert active.model == embedding.identity

    service = AnswerService(base, hybrid=HybridRetriever())
    answer = service.answer(
        RagQuery(
            "索引中断后怎样恢复？",
            root_ids=(root_id,),
            extensions=(".txt",),
            top_k=4,
        ),
        embedding=embedding,
        llm=MockLLMGateway(
            {
                "insufficient": False,
                "answer": "可以从上一个已发布 generation 恢复。[S1]",
                "used_sources": ["S1"],
            }
        ),
        token=CancelToken(),
    )
    assert answer.status == "answered"
    assert answer.source_generation == generation
    assert answer.citations[0].source.file_path.endswith("recovery.txt")
    assert answer.citations[0].source.root_id == root_id
    assert "已发布 generation" in answer.citations[0].excerpt


def test_sidecar_rejects_wrong_generation_and_model(tmp_path: Path) -> None:
    base, generation, _root_id = _publish_search_generation(tmp_path)
    embedding = MockEmbeddingProvider(dimension=24)
    RagIndexBuilder(base).build(embedding, CancelToken())
    store = RagIndexStore(base)
    with pytest.raises(RagIndexNotReadyError, match="generation"):
        with store.acquire_matching(generation + 1, embedding.identity):
            pass
    wrong_model = MockEmbeddingProvider(dimension=32).identity
    with pytest.raises(RagIndexNotReadyError, match="模型版本"):
        with store.acquire_matching(generation, wrong_model):
            pass


def test_inadequate_filters_return_without_calling_llm(tmp_path: Path) -> None:
    base, generation, root_id = _publish_search_generation(tmp_path)
    embedding = MockEmbeddingProvider(dimension=32)
    RagIndexBuilder(base).build(embedding, CancelToken())
    llm = MockLLMGateway()
    answer = AnswerService(base).answer(
        RagQuery(
            "索引如何恢复？",
            root_ids=(root_id + 1000,),
            extensions=(".txt",),
        ),
        embedding=embedding,
        llm=llm,
        token=CancelToken(),
    )
    assert answer.status == "insufficient"
    assert answer.source_generation == generation
    assert llm.calls == 0


def test_corrupt_sidecar_candidate_does_not_replace_manifest(tmp_path: Path) -> None:
    base, generation, _root_id = _publish_search_generation(tmp_path)
    embedding = MockEmbeddingProvider(dimension=24)
    store = RagIndexStore(base)
    RagIndexBuilder(base).build(embedding, CancelToken())
    before = store.manifest_path.read_bytes()
    active_before = store.active()
    assert active_before is not None
    candidate = store.begin_candidate(
        source_generation=generation,
        source_database=IndexGenerationStore(base).active().path,
        model=embedding.identity,
        chunk_policy_version="test-v1",
    )
    with store.connect_candidate(candidate) as connection:
        connection.execute("INSERT INTO rag_chunks(stable_chunk_id,text,normalized_text,text_hash,"
                           "chunk_policy_version,source_type) VALUES('orphan','text','text','hash','test-v1','native_text')")
    with pytest.raises(sqlite3.DatabaseError, match="incomplete candidate"):
        store.publish(candidate)
    store.abort(candidate, "fixture failure")
    after = store.active()
    assert store.manifest_path.read_bytes() == before
    assert after is not None and after.path == active_before.path
    assert after.generation == generation


def test_openai_compatible_gateway_uses_configured_key_model_and_stream() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["Authorization"] == "Bearer test-secret"
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "test-model"}]})
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["model"] == "test-model"
            assert body["max_completion_tokens"] == 128
            assert "max_tokens" not in body
            return httpx.Response(
                200,
                text=(
                    'data: {"choices":[{"delta":{"content":"{\\"insufficient\\":true,\\"answer\\":\\"\\",\\"used_sources\\":[]}"}}]}\n\n'
                    "data: [DONE]\n\n"
                ),
                headers={"content-type": "text/event-stream"},
            )
        return httpx.Response(404)

    gateway = OpenAICompatibleGateway(
        "https://api.example.test/v1",
        "test-secret",
        "test-model",
        transport=httpx.MockTransport(handler),
    )
    assert gateway.test_connection().available
    result = gateway.generate(
        [{"role": "user", "content": "test"}],
        max_tokens=128,
        cancel_token=CancelToken(),
    )
    assert json.loads(result)["insufficient"] is True
    assert len(calls) == 2
    gateway.close()


def test_api_gateway_requires_https_for_remote_and_rejects_secrets_in_url() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleGateway("http://api.example.test/v1", "key", "model")
    with pytest.raises(ValueError, match="不能包含"):
        OpenAICompatibleGateway("https://api.example.test/v1?key=secret", "key", "model")


def test_credential_store_keeps_secret_out_of_settings_serialization() -> None:
    class FakeKeyring:
        errors = SimpleNamespace(PasswordDeleteError=type("PasswordDeleteError", (Exception,), {}))

        def __init__(self) -> None:
            self.values: dict[tuple[str, str], str] = {}

        def get_password(self, service: str, username: str) -> str | None:
            return self.values.get((service, username))

        def set_password(self, service: str, username: str, value: str) -> None:
            self.values[(service, username)] = value

        def delete_password(self, service: str, username: str) -> None:
            if (service, username) not in self.values:
                raise self.errors.PasswordDeleteError()
            del self.values[(service, username)]

    from local_full_text_search.config.defaults import AppSettings

    fake = FakeKeyring()
    store = CredentialStore(fake)
    store.save_api_key("test-secret")
    assert store.get_api_key() == "test-secret"
    assert "test-secret" not in json.dumps(AppSettings().to_dict())
    store.delete_api_key()
    assert store.get_api_key() == ""


def test_downloaded_fastembed_model_builds_real_vector_sidecar_offline(tmp_path: Path) -> None:
    pytest.importorskip("fastembed")
    cache_dir = CACHE_DIR / "rag-fastembed"
    snapshot_root = cache_dir / "models--Qdrant--bge-small-zh-v1.5" / "snapshots"
    if not snapshot_root.is_dir() or not list(snapshot_root.glob("*/model_optimized.onnx")):
        pytest.skip("FastEmbed BGE model has not been downloaded")
    base, generation, root_id = _publish_search_generation(tmp_path)
    embedding = FastEmbedLocalProvider(
        "BAAI/bge-small-zh-v1.5",
        cache_dir=cache_dir,
        threads=2,
        local_files_only=True,
    )
    report = RagIndexBuilder(base).build(embedding, CancelToken())
    assert report.generation == generation
    assert report.chunk_count > 0
    assert report.generated_embeddings == report.chunk_count

    with RagIndexStore(base).acquire_matching(generation, embedding.identity) as handle:
        hits = VectorRetriever().retrieve(
            handle.connection,
            RagQuery("索引中断怎样从之前版本恢复？", root_ids=(root_id,), top_k=4),
            embedding,
            CancelToken(),
            top_n=4,
        )
    assert hits
    assert hits[0].selected_source.file_path.endswith("recovery.txt")
