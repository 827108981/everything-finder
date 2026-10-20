from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("chromadb")

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.errors import RagIndexStaleError
from local_full_text_search.rag.generation.base import ModelAnswer
from local_full_text_search.rag.index_builder import RagIndexBuilder
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import (
    AnswerReviewResult,
    ClaimReview,
    EvidenceChunk,
    RagQuery,
    RagAnswerStatus,
)
from local_full_text_search.rag.secrets import MemorySecretStore


@dataclass(frozen=True)
class _Generation:
    generation: int
    path: Path


class _SourceGenerationStore:
    def __init__(self, path: Path, generation: int = 3) -> None:
        self.current = _Generation(generation, path)

    @contextmanager
    def acquire_active(self):
        yield self.current, object()

    def active(self):
        return self.current


class _Embedding:
    model_id = "test/embedding"
    model_revision = "a" * 40
    dimension = 3

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


class _DifferentRevisionEmbedding(_Embedding):
    model_revision = "f" * 40


class _Reranker:
    def __init__(self, score: float) -> None:
        self.score = score

    def rerank(self, question, candidates, top_n, token):
        token.throw_if_cancelled()
        output = list(candidates[:top_n])
        for item in output:
            item.rerank_score = self.score
        return output


class _Gateway:
    def __init__(self, answer: str = "检查点可用于恢复索引任务 [S1]。") -> None:
        self.answer_text = answer
        self.calls = 0
        self.review_calls = 0
        self.last_request = None
        self.reviewed_answer = None

    def generate(self, request, token):
        token.throw_if_cancelled()
        self.calls += 1
        self.last_request = request
        return ModelAnswer(self.answer_text, "test-provider", request.model)

    def review_answer(self, request, token):
        token.throw_if_cancelled()
        self.review_calls += 1
        final = self.reviewed_answer or request.draft_answer
        verdict = "revised" if self.reviewed_answer else "pass"
        return AnswerReviewResult(
            verdict,
            "模拟复核完成",
            (ClaimReview("检查点可恢复任务", ("S1",), "supported", "证据包含该步骤"),),
            final,
            "test-provider",
            request.model,
        )


def _create_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, root_id INTEGER, path TEXT, filename TEXT,
            extension TEXT, content_hash TEXT
        );
        CREATE TABLE blocks (
            id INTEGER PRIMARY KEY, file_id INTEGER, document_id INTEGER,
            block_index INTEGER, raw_text TEXT, location_text TEXT,
            page_number INTEGER, slide_number INTEGER, sheet_name TEXT,
            cell_start TEXT, cell_end TEXT, line_start INTEGER, line_end INTEGER,
            source_type TEXT, ocr_confidence REAL
        );
        CREATE VIEW search_blocks AS SELECT * FROM blocks;
        INSERT INTO files VALUES(
            11, 2, 'C:/资料/索引恢复手册.pdf', '索引恢复手册.pdf', '.pdf', 'doc-hash'
        );
        INSERT INTO blocks VALUES(
            21, 11, 5, 0,
            '索引任务中断后，程序会检查已保存的检查点，并从上次成功的位置继续处理；发布前还会核对正文块和搜索索引数量。',
            '第 4 页', 4, NULL, NULL, NULL, NULL, NULL, NULL, 'native_text', NULL
        );
        """
    )
    connection.commit()
    connection.close()


def _create_scope_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE files (
            id INTEGER PRIMARY KEY, root_id INTEGER, path TEXT, filename TEXT,
            extension TEXT, content_hash TEXT
        );
        CREATE TABLE blocks (
            id INTEGER PRIMARY KEY, file_id INTEGER, document_id INTEGER,
            block_index INTEGER, raw_text TEXT, location_text TEXT,
            page_number INTEGER, slide_number INTEGER, sheet_name TEXT,
            cell_start TEXT, cell_end TEXT, line_start INTEGER, line_end INTEGER,
            source_type TEXT, ocr_confidence REAL
        );
        CREATE VIEW search_blocks AS SELECT * FROM blocks;
        INSERT INTO files VALUES(
            11, 2, 'C:/资料/单项目排查.pdf', '单项目排查.pdf', '.pdf', 'single-hash'
        );
        INSERT INTO files VALUES(
            12, 2, 'C:/资料/全部项目故障.pdf', '全部项目故障.pdf', '.pdf', 'all-hash'
        );
        INSERT INTO blocks VALUES(
            21, 11, 101, 0,
            '单个项目发光值异常时，检查该项目对应参数、试剂和反应杯。',
            '第 1 页', 1, NULL, NULL, NULL, NULL, NULL, NULL, 'native_text', NULL
        );
        INSERT INTO blocks VALUES(
            22, 12, 102, 0,
            '所有项目均出现发光值异常和定标失败时，应检查整体定标状态和校准液。',
            '第 2 页', 2, NULL, NULL, NULL, NULL, NULL, NULL, 'native_text', NULL
        );
        """
    )
    connection.commit()
    connection.close()


@pytest.fixture
def indexed_rag(tmp_path):
    snapshot = tmp_path / "generation-3.db"
    _create_snapshot(snapshot)
    source_store = _SourceGenerationStore(snapshot)
    rag_store = RagIndexStore(tmp_path / "rag")
    settings = AppSettings(
        rag_enabled=True,
        rag_api_model="test-model",
        rag_min_vector_score=0.1,
        rag_min_rerank_score=0.5,
        rag_embedding_batch_size=2,
        rag_answer_review_enabled=False,
    )
    builder = RagIndexBuilder(source_store, rag_store, _Embedding(), settings)
    summary = builder.build(CancelToken())
    assert summary.chunk_count == summary.vector_count == 1
    yield source_store, rag_store, settings


def test_full_build_retrieve_rerank_cite_and_generate(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    gateway = _Gateway()
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        gateway,
        secrets,
        settings,
        progress_callback=progress.append,
    )

    answer = service.answer(
        RagQuery(text="索引任务中断后如何恢复？"),
        CancelToken(),
    )

    assert answer.insufficient_evidence is False
    assert answer.source_ids == ("S1",)
    assert answer.sources[0].filename == "索引恢复手册.pdf"
    assert answer.sources[0].page_number == 4
    assert "[S1]" in answer.answer_text
    assert gateway.calls == 1
    assert gateway.last_request.api_key == "test-secret"
    assert "检查点" in gateway.last_request.context
    stages = [item["stage"] for item in progress]
    assert stages == [
        "checking_index",
        "query_interpreting",
        "query_interpreted",
        "clarification_check",
        "retrieving",
        "lexical_retrieval",
        "vector_retrieval",
        "fusion",
        "deduplicating",
        "reranking",
        "reranked",
        "checking_constraints",
        "checking_evidence",
        "building_context",
        "credential_read",
        "generating",
        "draft_generated",
        "validating",
        "completed",
    ]
    assert next(item for item in progress if item["stage"] == "reranking")["candidate_count"] > 0
    assert next(item for item in progress if item["stage"] == "generating")["context_chars"] > 0
    lexical_step = next(item for item in progress if item["stage"] == "vector_retrieval")
    vector_step = next(item for item in progress if item["stage"] == "fusion")
    fusion_step = next(item for item in progress if item["stage"] == "reranking")
    evidence_step = next(item for item in progress if item["stage"] == "checking_evidence")
    request_step = next(item for item in progress if item["stage"] == "generating")
    assert isinstance(lexical_step["lexical_candidates"], list)
    assert "检查点" in vector_step["vector_candidates"][0]["text"]
    assert vector_step["vector_candidates"][0]["vector_score"] is not None
    assert fusion_step["fused_candidates"][0]["fused_score"] > 0
    assert evidence_step["evidence_candidates"][0]["selected"] is True
    assert request_step["api_request"]["model"] == "test-model"
    assert "检查点" in str(request_step["api_request"])
    assert "test-secret" not in str(progress)


def test_retrieval_mode_uses_the_same_pipeline_without_reading_key_or_calling_llm(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    secrets = MemorySecretStore()
    gateway = _Gateway()
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        gateway,
        secrets,
        settings,
        progress_callback=progress.append,
    )

    result = service.retrieve(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())

    assert result.status == RagAnswerStatus.ANSWERED
    assert result.insufficient_evidence is False
    assert result.sources[0].filename == "索引恢复手册.pdf"
    assert gateway.calls == 0
    assert not any(item["stage"] in {"credential_read", "generating"} for item in progress)
    assert any(item["stage"] == "retrieval_completed" for item in progress)


def test_low_reranker_score_refuses_without_calling_llm(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    gateway = _Gateway()
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.01),
        gateway,
        secrets,
        settings,
        progress_callback=progress.append,
    )

    answer = service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())

    assert answer.insufficient_evidence is True
    assert answer.sources == ()
    assert gateway.calls == 0
    evidence = next(item for item in progress if item["stage"] == "checking_evidence")
    assert evidence["evidence_candidates"][0]["selected"] is False
    assert "重排分数低于" in evidence["evidence_candidates"][0]["reason"]
    assert not any("api_request" in item for item in progress)


def test_generation_mismatch_uses_published_index_with_warning(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    source_store.current = _Generation(4, source_store.current.path)
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        _Gateway(),
        secrets,
        settings,
        progress_callback=progress.append,
    )

    answer = service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())
    assert answer.insufficient_evidence is False
    assert answer.sources[0].source_generation == 3
    assert any("旧 AI 索引" in warning for warning in answer.warnings)
    retrieving = next(item for item in progress if item["stage"] == "retrieving")
    assert retrieving["source_generation"] == 3
    assert retrieving["active_source_generation"] == 4


def test_verified_unchanged_generation_answers_without_stale_warning(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    original_bundle = rag_store.active().bundle_id
    source_store.current = _Generation(4, source_store.current.path)
    summary = RagIndexBuilder(source_store, rag_store, _Embedding(), settings).build(CancelToken())
    assert summary.unchanged is True
    assert summary.bundle_id == original_bundle
    assert rag_store.active().verified_source_generation == 4

    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    service = AnswerService(
        source_store, rag_store, _Embedding(), _Reranker(0.97), _Gateway(), secrets, settings,
    )
    answer = service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())
    assert answer.insufficient_evidence is False
    assert answer.warnings == ("未启用 LLM 答案二次复核。",)


def test_embedding_model_mismatch_blocks_answer(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    service = AnswerService(
        source_store,
        rag_store,
        _DifferentRevisionEmbedding(),
        _Reranker(0.97),
        _Gateway(),
        MemorySecretStore(),
        settings,
    )

    with pytest.raises(RagIndexStaleError, match="向量模型与 AI 索引不匹配"):
        service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())


def test_unknown_citation_is_rejected(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        _Gateway("没有来源的结论 [S9]"),
        secrets,
        settings,
    )

    from local_full_text_search.rag.errors import RagGenerationError

    with pytest.raises(RagGenerationError, match="无法验证"):
        service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())


def test_answer_review_calls_llm_twice_and_publishes_reviewed_text(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    settings.rag_answer_review_enabled = True
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    gateway = _Gateway()
    gateway.reviewed_answer = "先从已保存的检查点继续处理。[S1]"
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        gateway,
        secrets,
        settings,
        progress_callback=progress.append,
    )

    answer = service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())

    assert answer.answer_text == gateway.reviewed_answer
    assert answer.audit.review.verdict == "revised"
    assert answer.audit.draft_answer == gateway.answer_text
    assert gateway.calls == 1
    assert gateway.review_calls == 1
    assert answer.source_ids == ("S1",)
    review_step = next(item for item in progress if item["stage"] == "reviewing")
    assert review_step["review_request"]["model"] == "test-model"
    assert "test-secret" not in str(progress)


def test_answer_review_failure_does_not_return_unreviewed_draft(indexed_rag) -> None:
    source_store, rag_store, settings = indexed_rag
    settings.rag_answer_review_enabled = True
    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")

    class FailedReviewGateway(_Gateway):
        def review_answer(self, request, token):
            raise RuntimeError("审查服务不可用")

    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        _Reranker(0.97),
        FailedReviewGateway(),
        secrets,
        settings,
    )

    with pytest.raises(RuntimeError, match="审查服务不可用"):
        service.answer(RagQuery(text="索引任务中断后如何恢复？"), CancelToken())


def test_gold_single_project_case_filters_higher_rerank_all_project_evidence(tmp_path) -> None:
    snapshot = tmp_path / "scope-generation.db"
    _create_scope_snapshot(snapshot)
    source_store = _SourceGenerationStore(snapshot)
    rag_store = RagIndexStore(tmp_path / "rag-scope")
    settings = AppSettings(
        rag_enabled=True,
        rag_api_model="test-model",
        rag_min_vector_score=0.1,
        rag_min_rerank_score=0.1,
        rag_final_top_k=1,
        rag_answer_review_enabled=False,
    )
    RagIndexBuilder(source_store, rag_store, _Embedding(), settings).build(CancelToken())

    class ScopeReranker:
        def __init__(self):
            self.last_scored_candidates = ()

        def rerank(self, question, candidates, top_n, token):
            scored = list(candidates[:top_n])
            for item in scored:
                item.rerank_score = 0.99 if "所有项目" in item.chunk.text else 0.72
            scored.sort(key=lambda item: -(item.rerank_score or 0))
            self.last_scored_candidates = tuple(scored)
            return scored[:top_n]

    secrets = MemorySecretStore()
    secrets.save(settings.rag_api_secret_ref, "test-secret")
    progress = []
    service = AnswerService(
        source_store,
        rag_store,
        _Embedding(),
        ScopeReranker(),
        _Gateway("先检查该项目对应的参数和试剂 [S1]。"),
        secrets,
        settings,
        progress_callback=progress.append,
    )

    answer = service.answer(
        RagQuery(text="只有某个项目的发光值异常，这个时候该怎么排查", final_top_k=1),
        CancelToken(),
    )

    assert answer.sources[0].filename == "单项目排查.pdf"
    constraints = next(item for item in progress if item["stage"] == "checking_constraints")
    all_project = next(
        item for item in constraints["constraint_candidates"]
        if item["filename"] == "全部项目故障.pdf"
    )
    assert all_project["rerank_score"] == 0.99
    assert all_project["constraint_decision"]["required_conflict_count"] >= 1
    evidence = next(item for item in progress if item["stage"] == "checking_evidence")
    assert all(
        item["filename"] != "全部项目故障.pdf" or item["selected"] is False
        for item in evidence["evidence_candidates"]
    )
