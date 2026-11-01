from __future__ import annotations

import gc
import hashlib
import json
import logging
import sqlite3
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.block_reader import BlockReader
from local_full_text_search.rag.chunk_builder import ChunkBuilder
from local_full_text_search.rag.embedding_cache import EmbeddingVectorCache
from local_full_text_search.rag.embeddings.local_bge import MAX_EMBEDDING_BATCH_SIZE
from local_full_text_search.rag.errors import (
    RagCorpusTooLargeError,
    RagIndexCorruptError,
    RagIndexMissingError,
    RagSourceChangedError,
)
from local_full_text_search.rag.index_store import RagBundle, RagIndexStore
from local_full_text_search.rag.models import VectorRecord
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.services.logging_service import log_event


logger = logging.getLogger(__name__)
MAX_INDEX_WRITE_BATCH_SIZE = 256


@dataclass(frozen=True, slots=True)
class RagBuildSummary:
    source_generation: int
    bundle_id: str
    chunk_count: int
    vector_count: int
    elapsed_ms: int
    bundle_path: str
    unchanged: bool = False
    unit_count: int = 0


class RagIndexBuilder:
    def __init__(
        self,
        source_store: IndexGenerationStore,
        rag_store: RagIndexStore,
        embedding,
        settings,
        *,
        block_reader: BlockReader | None = None,
        chunk_builder: ChunkBuilder | None = None,
    ) -> None:
        self.source_store = source_store
        self.rag_store = rag_store
        self.embedding = embedding
        self.settings = settings
        self.block_reader = block_reader or BlockReader()
        self.chunk_builder = chunk_builder or ChunkBuilder(
            target_chars=settings.rag_chunk_target_chars,
            max_chars=settings.rag_chunk_max_chars,
            overlap_chars=settings.rag_chunk_overlap_chars,
        )
        self.embedding_batch_size = min(
            MAX_EMBEDDING_BATCH_SIZE,
            max(1, int(settings.rag_embedding_batch_size)),
        )
        self._effective_embedding_batch_size = self.embedding_batch_size
        self._length_bucketing = bool(
            getattr(settings, "rag_embedding_length_bucketing", True)
        )
        self.write_batch_size = min(
            MAX_INDEX_WRITE_BATCH_SIZE,
            max(1, int(settings.rag_index_write_batch_size)),
        )

    def build(
        self,
        token: CancelToken,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        *,
        run_id: str | None = None,
    ) -> RagBuildSummary:
        started = time.perf_counter()
        run_id = run_id or f"rag-index-{uuid.uuid4().hex}"
        candidate: RagBundle | None = None
        connection = None
        vector_store = None
        previous_vector_store = None
        embedding_cache = None
        chunk_count = 0
        total_chunk_count = 0
        batch_index = 0
        reused_count = 0
        embedded_count = 0
        resumed_count = 0
        resumed_ids: tuple[str, ...] = ()
        resuming_candidate = False
        unit_v2 = str(getattr(self.settings, "rag_vector_backend", "legacy")) == "unit_v2"
        parent_vector_mode = "metadata_only" if unit_v2 else "chroma"

        def report(stage: str, message: str, **extra: object) -> None:
            if progress_callback:
                progress_callback({"stage": stage, "message": message, **extra})

        with self.source_store.acquire_active() as (source, _lease):
            self._log_event(
                "rag.index.build_started",
                run_id=run_id,
                source_generation=source.generation,
                embedding_model_id=self.embedding.model_id,
                embedding_revision=self.embedding.model_revision,
                embedding_batch_size=self.embedding_batch_size,
                write_batch_size=self.write_batch_size,
            )
            previous_bundle = None
            previous_text_ids: dict[str, str] = {}
            try:
                previous_bundle = self.rag_store.active()
                if (
                    self._same_embedding_model(previous_bundle)
                    and previous_bundle.parent_vector_mode == parent_vector_mode
                    and (not unit_v2 or self._attached_vector_index(previous_bundle) is not None)
                ):
                    if not unit_v2:
                        previous_vector_store = ChromaVectorStore(
                            previous_bundle.chroma_path
                        )
                    if self._unchanged_bundle(source, previous_bundle, previous_vector_store, token):
                        active_now = self.source_store.active()
                        if active_now is None or active_now.generation != source.generation:
                            raise RagSourceChangedError("全文索引在校验期间已切换，请重新构建 AI 索引")
                        previous_bundle = self.rag_store.mark_source_compatible(
                            previous_bundle, source.generation
                        )
                        elapsed = int((time.perf_counter() - started) * 1000)
                        self._log_event(
                            "rag.index.build_unchanged",
                            run_id=run_id,
                            source_generation=source.generation,
                            rag_source_generation=previous_bundle.source_generation,
                            bundle_id=previous_bundle.bundle_id,
                            chunk_count=previous_bundle.chunk_count,
                            reused_count=previous_bundle.chunk_count,
                            embedded_count=0,
                            elapsed_ms=elapsed,
                        )
                        report("completed", "资料未变化，沿用已有 AI 索引", chunk_count=previous_bundle.chunk_count)
                        if previous_vector_store is not None:
                            previous_vector_store.close()
                            previous_vector_store = None
                        return RagBuildSummary(
                            source_generation=source.generation,
                            bundle_id=previous_bundle.bundle_id,
                            chunk_count=previous_bundle.chunk_count,
                            vector_count=previous_bundle.vector_count,
                            elapsed_ms=elapsed,
                            bundle_path=str(previous_bundle.path),
                            unchanged=True,
                            unit_count=int(previous_bundle.vector_unit_count or 0),
                        )
                    if previous_vector_store is not None:
                        previous_text_ids = self._previous_text_ids(previous_bundle)
                else:
                    previous_bundle = None
            except (RagIndexMissingError, RagIndexCorruptError) as exc:
                self._log_event(
                    "rag.index.reuse_unavailable",
                    run_id=run_id,
                    error_type=type(exc).__name__,
                )
                if previous_vector_store is not None:
                    previous_vector_store.close()
                    previous_vector_store = None
            except BaseException:
                if previous_vector_store is not None:
                    previous_vector_store.close()
                raise
            report("planning", "正在统计本次需要生成的内容块")
            planning_started = time.perf_counter()
            total_chunk_count = self._count_chunks(source, token)
            build_signature = self._build_signature(source.generation)
            self._log_event(
                "rag.index.stage_completed",
                run_id=run_id,
                stage="chunk_planning",
                duration_ms=int((time.perf_counter() - planning_started) * 1000),
                total_chunk_count=total_chunk_count,
                build_signature=build_signature,
            )
            report(
                "planning",
                "内容块统计完成",
                chunk_count=0,
                total_chunk_count=total_chunk_count,
            )
            self._log_event(
                "rag.index.stage_started", run_id=run_id, stage="candidate_create"
            )
            candidate_started = time.perf_counter()
            try:
                resumed = None
                if bool(getattr(self.settings, "rag_resume_partial_build", True)):
                    resumed = self.rag_store.find_resumable_candidate(
                        source.generation,
                        embedding_repo_id=self.embedding.model_id,
                        embedding_revision=self.embedding.model_revision,
                        embedding_dimension=self.embedding.dimension,
                        build_signature=build_signature,
                    )
                if resumed is not None:
                    candidate, connection = resumed
                    if candidate.total_chunk_count != total_chunk_count:
                        self.rag_store.abort(candidate, connection)
                        candidate = None
                        connection = None
                    else:
                        resumed_count = candidate.chunk_count
                        resuming_candidate = True
                        resumed_ids = tuple(
                            str(row[0])
                            for row in connection.execute(
                                "SELECT stable_chunk_id FROM rag_chunks ORDER BY rowid"
                            )
                        )
                if candidate is None:
                    candidate, connection = self.rag_store.create_candidate(
                        source.generation,
                        embedding_repo_id=self.embedding.model_id,
                        embedding_revision=self.embedding.model_revision,
                        embedding_dimension=self.embedding.dimension,
                        reranker_repo_id=self.settings.rag_reranker_repo_id,
                        reranker_revision=self.settings.rag_reranker_revision,
                        hnsw_m=self.settings.rag_hnsw_m,
                        construction_ef=self.settings.rag_hnsw_construction_ef,
                        search_ef=self.settings.rag_hnsw_search_ef,
                        build_signature=build_signature,
                        total_chunk_count=total_chunk_count,
                        parent_vector_mode=parent_vector_mode,
                    )
            except BaseException:
                if previous_vector_store is not None:
                    previous_vector_store.close()
                raise
            self._log_event(
                "rag.index.stage_completed",
                run_id=run_id,
                stage="candidate_create",
                duration_ms=int((time.perf_counter() - candidate_started) * 1000),
                resumed_chunk_count=resumed_count,
            )
            try:
                token.throw_if_cancelled()
                report(
                    "vector_store",
                    (
                        "正在恢复父正文元数据" if resumed_count and unit_v2
                        else "正在写入父正文元数据" if unit_v2
                        else "正在恢复未完成的向量索引" if resumed_count
                        else "正在创建 Chroma HNSW 向量索引"
                    ),
                    chunk_count=resumed_count,
                    total_chunk_count=total_chunk_count,
                    resumed_chunk_count=resumed_count,
                )
                self._log_event(
                    "rag.index.stage_started",
                    run_id=run_id,
                    stage="vector_store_create",
                )
                vector_store_started = time.perf_counter()
                if not unit_v2:
                    vector_store = ChromaVectorStore(
                        candidate.chroma_path,
                        create=not resuming_candidate,
                        hnsw_m=self.settings.rag_hnsw_m,
                        construction_ef=self.settings.rag_hnsw_construction_ef,
                        search_ef=self.settings.rag_hnsw_search_ef,
                    )
                    embedding_cache = EmbeddingVectorCache(
                        self.rag_store.root / "embedding_cache.db",
                        model_id=self.embedding.model_id,
                        model_revision=self.embedding.model_revision,
                        dimension=self.embedding.dimension,
                    )
                if (
                    not unit_v2
                    and resuming_candidate
                    and vector_store.count() != resumed_count
                ):
                    # SQLite 与 Chroma 跨库提交无法做到一个事务；若上次进程在两者之间
                    # 异常退出，宁可丢弃候选，也不能带着不一致数据继续发布。
                    vector_store.close()
                    vector_store = None
                    self.rag_store.abort(candidate, connection)
                    candidate, connection = self.rag_store.create_candidate(
                        source.generation,
                        embedding_repo_id=self.embedding.model_id,
                        embedding_revision=self.embedding.model_revision,
                        embedding_dimension=self.embedding.dimension,
                        reranker_repo_id=self.settings.rag_reranker_repo_id,
                        reranker_revision=self.settings.rag_reranker_revision,
                        hnsw_m=self.settings.rag_hnsw_m,
                        construction_ef=self.settings.rag_hnsw_construction_ef,
                        search_ef=self.settings.rag_hnsw_search_ef,
                        build_signature=build_signature,
                        total_chunk_count=total_chunk_count,
                        parent_vector_mode=parent_vector_mode,
                    )
                    resumed_count = 0
                    resumed_ids = ()
                    resuming_candidate = False
                    vector_store = ChromaVectorStore(
                        candidate.chroma_path,
                        create=True,
                        hnsw_m=self.settings.rag_hnsw_m,
                        construction_ef=self.settings.rag_hnsw_construction_ef,
                        search_ef=self.settings.rag_hnsw_search_ef,
                    )
                self._log_event(
                    "rag.index.stage_completed",
                    run_id=run_id,
                    stage="vector_store_create",
                    duration_ms=int((time.perf_counter() - vector_store_started) * 1000),
                )
                blocks = self.block_reader.iter_blocks(
                    source.path,
                    source.generation,
                    include_ocr=True,
                    min_ocr_confidence=float(self.settings.ocr_min_confidence),
                )
                chunks = self.chunk_builder.iter_chunks(blocks)
                batch = []
                collect_started = time.perf_counter()
                embedding_phase_started = time.perf_counter()
                chunk_count = resumed_count
                batch_index = resumed_count // self.write_batch_size
                if resumed_count:
                    report(
                        "embedding",
                        f"已恢复 {resumed_count:,} 个内容块，继续生成向量",
                        chunk_count=resumed_count,
                        total_chunk_count=total_chunk_count,
                        resumed_chunk_count=resumed_count,
                    )
                source_chunk_index = 0
                for chunk in chunks:
                    token.throw_if_cancelled()
                    if source_chunk_index < resumed_count:
                        if resumed_ids[source_chunk_index] != chunk.stable_chunk_id:
                            raise RagIndexCorruptError(
                                "未完成 AI 索引与当前分块顺序不一致，无法安全续建"
                            )
                        source_chunk_index += 1
                        continue
                    source_chunk_index += 1
                    batch.append(chunk)
                    if len(batch) >= self.write_batch_size:
                        self._check_chunk_limit(chunk_count, len(batch))
                        collect_ms = int((time.perf_counter() - collect_started) * 1000)
                        batch_index += 1
                        if unit_v2:
                            indexed, reused, embedded = self._index_metadata_batch(
                                batch, connection, token,
                                run_id=run_id, batch_index=batch_index,
                                chunk_read_and_chunk_ms=collect_ms,
                            )
                        else:
                            indexed, reused, embedded = self._index_batch(
                                batch,
                                connection,
                                vector_store,
                                token,
                                previous_vector_store=previous_vector_store,
                                previous_text_ids=previous_text_ids,
                                embedding_cache=embedding_cache,
                                run_id=run_id,
                                batch_index=batch_index,
                                chunk_read_and_chunk_ms=collect_ms,
                            )
                        chunk_count += indexed
                        reused_count += reused
                        embedded_count += embedded
                        batch.clear()
                        collect_started = time.perf_counter()
                        report(
                            "embedding",
                            "正在写入父正文元数据（父向量已跳过）" if unit_v2 else "正在分块并生成向量",
                            batch_index=batch_index,
                            write_batch_size=self.write_batch_size,
                            embedding_batch_size=self._effective_embedding_batch_size,
                            **self._progress_stats(
                                chunk_count,
                                total_chunk_count,
                                resumed_count,
                                embedding_phase_started,
                            ),
                        )
                        if chunk_count > int(self.settings.rag_max_chunks):
                            raise RagCorpusTooLargeError(
                                f"RAG 正文块超过 {int(self.settings.rag_max_chunks):,}，请缩小资料范围后重试"
                            )
                if batch:
                    self._check_chunk_limit(chunk_count, len(batch))
                    collect_ms = int((time.perf_counter() - collect_started) * 1000)
                    batch_index += 1
                    if unit_v2:
                        indexed, reused, embedded = self._index_metadata_batch(
                            batch, connection, token,
                            run_id=run_id, batch_index=batch_index,
                            chunk_read_and_chunk_ms=collect_ms,
                        )
                    else:
                        indexed, reused, embedded = self._index_batch(
                            batch,
                            connection,
                            vector_store,
                            token,
                            previous_vector_store=previous_vector_store,
                            previous_text_ids=previous_text_ids,
                            embedding_cache=embedding_cache,
                            run_id=run_id,
                            batch_index=batch_index,
                            chunk_read_and_chunk_ms=collect_ms,
                        )
                    chunk_count += indexed
                    reused_count += reused
                    embedded_count += embedded
                    report(
                        "embedding",
                        "正在完成父正文元数据" if unit_v2 else "正在完成最后一批向量",
                        batch_index=batch_index,
                        write_batch_size=self.write_batch_size,
                        embedding_batch_size=self._effective_embedding_batch_size,
                        **self._progress_stats(
                            chunk_count,
                            total_chunk_count,
                            resumed_count,
                            embedding_phase_started,
                        ),
                    )
                if chunk_count > int(self.settings.rag_max_chunks):
                    raise RagCorpusTooLargeError(
                        f"RAG 正文块超过 {int(self.settings.rag_max_chunks):,}，请缩小资料范围后重试"
                    )
                token.throw_if_cancelled()
                report(
                    "validating",
                    "正在校验父正文元数据" if unit_v2 else "正在校验正文与向量数量",
                )
                self._log_event(
                    "rag.index.stage_started",
                    run_id=run_id,
                    stage="validation",
                )
                validation_started = time.perf_counter()
                vector_count = 0 if unit_v2 else vector_store.count()
                stored_chunk_count = int(
                    connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                )
                if stored_chunk_count != chunk_count:
                    raise RuntimeError(
                        f"父正文数量不一致：预期 {chunk_count}，实际 {stored_chunk_count}"
                    )
                if not unit_v2 and vector_count != chunk_count:
                    raise RuntimeError(
                        f"Chroma 数量不一致：预期 {chunk_count}，实际 {vector_count}"
                    )
                attached_vector_index = None
                if unit_v2:
                    report(
                        "vector_unit_index",
                        "正在构建与父正文候选绑定的细粒度向量索引",
                        processed_parent_chunks=0,
                        total_parent_chunks=chunk_count,
                        unit_count=0,
                    )
                    attached_vector_index, _reused = VectorIndexBuilder(
                        self.rag_store, self.embedding, self.settings
                    ).prepare(
                        replace(candidate, chunk_count=chunk_count, vector_count=0),
                        token,
                        progress_callback,
                        publish_pointer=False,
                    )
                    # 先发布新 bundle 私有目录下的兼容指针，最后再切换唯一权威的
                    # rag.manifest；旧父索引不会看到这个新 bundle，因此没有半发布窗口。
                    VectorIndexStore.publish_pointer(attached_vector_index)
                active_now = self.source_store.active()
                if active_now is None or active_now.generation != source.generation:
                    raise RagSourceChangedError("全文索引在构建期间已切换，请重新构建 AI 索引")
                self._log_event(
                    "rag.index.stage_completed",
                    run_id=run_id,
                    stage="validation",
                    duration_ms=int((time.perf_counter() - validation_started) * 1000),
                    chunk_count=chunk_count,
                    vector_count=vector_count,
                )
                report("publishing", "正在发布 AI 索引")
                self._log_event(
                    "rag.index.stage_started",
                    run_id=run_id,
                    stage="publish",
                )
                publish_started = time.perf_counter()
                published = self.rag_store.publish(
                    candidate,
                    connection,
                    vector_count=vector_count,
                    attached_vector_index=attached_vector_index,
                )
                connection = None
                self._log_event(
                    "rag.index.stage_completed",
                    run_id=run_id,
                    stage="publish",
                    duration_ms=int((time.perf_counter() - publish_started) * 1000),
                    bundle_id=published.bundle_id,
                )
                elapsed = int((time.perf_counter() - started) * 1000)
                self._log_event(
                    "rag.index.build_completed",
                    run_id=run_id,
                    source_generation=source.generation,
                    bundle_id=published.bundle_id,
                    chunk_count=chunk_count,
                    vector_count=vector_count,
                    batch_count=batch_index,
                    reused_count=reused_count,
                    embedded_count=embedded_count,
                    unit_count=(
                        attached_vector_index.unit_count
                        if attached_vector_index is not None else 0
                    ),
                    elapsed_ms=elapsed,
                )
                report(
                    "completed",
                    "AI 索引已就绪",
                    chunk_count=chunk_count,
                    total_chunk_count=total_chunk_count,
                    percent=100.0,
                )
                return RagBuildSummary(
                    source_generation=source.generation,
                    bundle_id=published.bundle_id,
                    chunk_count=published.chunk_count,
                    vector_count=published.vector_count,
                    elapsed_ms=elapsed,
                    bundle_path=str(published.path),
                    unit_count=(
                        int(attached_vector_index.unit_count)
                        if attached_vector_index is not None else 0
                    ),
                )
            except BaseException as exc:
                cancelled = isinstance(exc, CancelledError)
                self._log_event(
                    "rag.index.build_cancelled" if cancelled else "rag.index.build_failed",
                    level=logging.INFO if cancelled else logging.ERROR,
                    run_id=run_id,
                    source_generation=source.generation,
                    batch_count=batch_index,
                    chunk_count=chunk_count,
                    error_type=type(exc).__name__,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
                if vector_store is not None:
                    vector_store.close()
                    vector_store = None
                if candidate is not None:
                    if cancelled and bool(
                        getattr(self.settings, "rag_resume_partial_build", True)
                    ):
                        if connection is not None:
                            connection.commit()
                            connection.close()
                            connection = None
                        self._log_event(
                            "rag.index.candidate_suspended",
                            level=logging.INFO,
                            run_id=run_id,
                            bundle_id=candidate.bundle_id,
                            chunk_count=chunk_count,
                            total_chunk_count=total_chunk_count,
                        )
                    else:
                        self.rag_store.abort(candidate, connection)
                        connection = None
                raise
            finally:
                if vector_store is not None:
                    vector_store.close()
                if previous_vector_store is not None:
                    previous_vector_store.close()
                if embedding_cache is not None:
                    embedding_cache.close()

    def _index_batch(
        self,
        chunks,
        connection,
        vector_store,
        token: CancelToken,
        *,
        previous_vector_store,
        previous_text_ids: dict[str, str],
        embedding_cache: EmbeddingVectorCache | None,
        run_id: str,
        batch_index: int,
        chunk_read_and_chunk_ms: int,
    ) -> tuple[int, int, int]:
        token.throw_if_cancelled()
        embedding_started = time.perf_counter()
        embedding_ms = 0
        embedding_batch_count = 0
        unique_chunk_texts = list(dict.fromkeys(chunk.text for chunk in chunks))
        try:
            vectors_by_text: dict[str, list[float]] = (
                embedding_cache.get_many(unique_chunk_texts)
                if embedding_cache is not None else {}
            )
        except (OSError, sqlite3.Error, ValueError):
            vectors_by_text = {}
        reusable_texts = {
            chunk.text: previous_text_ids[chunk.text]
            for chunk in chunks
            if chunk.text in previous_text_ids
        }
        if previous_vector_store is not None and reusable_texts:
            old_vectors = previous_vector_store.get_embeddings(list(set(reusable_texts.values())))
            vectors_by_text.update(
                (text, old_vectors[point_id])
                for text, point_id in reusable_texts.items()
                if point_id in old_vectors
                and len(old_vectors[point_id]) == self.embedding.dimension
            )
        reused_count = sum(chunk.text in vectors_by_text for chunk in chunks)
        new_texts = list(dict.fromkeys(
            chunk.text for chunk in chunks if chunk.text not in vectors_by_text
        ))
        if self._length_bucketing:
            # 同一批中按长度分桶，减少短文本被补齐到最长文本所产生的无效计算。
            # 结果最终按正文映射回 chunk，因此排序不会改变索引内容或向量归属。
            new_texts.sort(key=len)
        offset = 0
        while offset < len(new_texts):
            token.throw_if_cancelled()
            batch_size = min(
                self._effective_embedding_batch_size, len(new_texts) - offset
            )
            text_batch = new_texts[offset : offset + batch_size]
            encode_started = time.perf_counter()
            try:
                encoded = self.embedding.embed_documents(text_batch)
            except BaseException as exc:
                if not self._is_memory_error(exc) or batch_size <= 1:
                    raise
                reduced = max(1, batch_size // 2)
                self._effective_embedding_batch_size = min(
                    self._effective_embedding_batch_size, reduced
                )
                self._clear_embedding_memory()
                self._log_event(
                    "rag.index.embedding_batch_reduced",
                    level=logging.WARNING,
                    run_id=run_id,
                    previous_batch_size=batch_size,
                    effective_batch_size=self._effective_embedding_batch_size,
                    error_type=type(exc).__name__,
                )
                continue
            embedding_ms += int((time.perf_counter() - encode_started) * 1000)
            embedding_batch_count += 1
            if len(encoded) != len(text_batch):
                raise ValueError("Embedding 返回的向量条数与 chunk 条数不一致")
            vectors_by_text.update(zip(text_batch, encoded, strict=True))
            if embedding_cache is not None:
                try:
                    embedding_cache.put_many(dict(zip(text_batch, encoded, strict=True)))
                except (OSError, sqlite3.Error, ValueError):
                    pass
            offset += len(text_batch)
        vectors = [vectors_by_text[chunk.text] for chunk in chunks]

        prepare_started = time.perf_counter()
        records = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            token.throw_if_cancelled()
            if len(vector) != self.embedding.dimension:
                raise ValueError(
                    f"向量维度错误：{len(vector)}，预期 {self.embedding.dimension}"
                )
            records.append(
                VectorRecord(
                    point_id=chunk.stable_chunk_id,
                    embedding=tuple(float(item) for item in vector),
                    metadata={
                        "stable_chunk_id": chunk.stable_chunk_id,
                        "source_generation": chunk.source.source_generation,
                        "root_id": chunk.source.root_id,
                        "extension": chunk.source.extension or "",
                        "source_type": chunk.source.source_type,
                        "ocr_confidence": float(chunk.source.ocr_confidence or 0.0),
                        "document_id": chunk.source.document_id or 0,
                    },
                )
            )
        prepare_ms = int((time.perf_counter() - prepare_started) * 1000)

        sqlite_started = time.perf_counter()
        for chunk in chunks:
            token.throw_if_cancelled()
            self.rag_store.add_chunk(connection, chunk)
            self.rag_store.add_vector_ref(connection, chunk.stable_chunk_id)
        sqlite_insert_ms = int((time.perf_counter() - sqlite_started) * 1000)

        chroma_started = time.perf_counter()
        vector_store.upsert(records)
        chroma_upsert_ms = int((time.perf_counter() - chroma_started) * 1000)

        commit_started = time.perf_counter()
        connection.commit()
        sqlite_commit_ms = int((time.perf_counter() - commit_started) * 1000)
        self._log_event(
            "rag.index.batch_completed",
            run_id=run_id,
            batch_index=batch_index,
            chunk_count=len(chunks),
            chunk_read_and_chunk_ms=chunk_read_and_chunk_ms,
            embedding_batch_size=self._effective_embedding_batch_size,
            embedding_batch_count=embedding_batch_count,
            length_bucketing=self._length_bucketing,
            reused_count=reused_count,
            embedded_count=len(new_texts),
            embedding_ms=embedding_ms,
            prepare_ms=prepare_ms,
            sqlite_insert_ms=sqlite_insert_ms,
            chroma_upsert_ms=chroma_upsert_ms,
            sqlite_commit_ms=sqlite_commit_ms,
            batch_elapsed_ms=int((time.perf_counter() - embedding_started) * 1000),
        )
        return len(chunks), reused_count, len(new_texts)

    def _index_metadata_batch(
        self,
        chunks,
        connection,
        token: CancelToken,
        *,
        run_id: str,
        batch_index: int,
        chunk_read_and_chunk_ms: int,
    ) -> tuple[int, int, int]:
        started = time.perf_counter()
        for chunk in chunks:
            token.throw_if_cancelled()
            self.rag_store.add_chunk(connection, chunk)
        connection.commit()
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self._log_event(
            "rag.index.batch_completed",
            run_id=run_id,
            batch_index=batch_index,
            chunk_count=len(chunks),
            chunk_read_and_chunk_ms=chunk_read_and_chunk_ms,
            embedding_batch_size=0,
            embedding_batch_count=0,
            length_bucketing=False,
            reused_count=0,
            embedded_count=0,
            embedding_ms=0,
            prepare_ms=0,
            sqlite_insert_ms=elapsed_ms,
            chroma_upsert_ms=0,
            sqlite_commit_ms=0,
            parent_vector_mode="metadata_only",
            batch_elapsed_ms=elapsed_ms,
        )
        return len(chunks), 0, 0

    def _count_chunks(self, source, token: CancelToken) -> int:
        blocks = self.block_reader.iter_blocks(
            source.path,
            source.generation,
            include_ocr=True,
            min_ocr_confidence=float(self.settings.ocr_min_confidence),
        )
        count = 0
        for _chunk in self.chunk_builder.iter_chunks(blocks):
            token.throw_if_cancelled()
            count += 1
            self._check_chunk_limit(count - 1, 1)
        if count <= 0:
            raise RagIndexCorruptError("全文索引中没有可供 AI 索引使用的正文块")
        return count

    def _build_signature(self, source_generation: int) -> str:
        payload = {
            "source_generation": int(source_generation),
            "embedding_repo_id": self.embedding.model_id,
            "embedding_revision": self.embedding.model_revision,
            "embedding_dimension": int(self.embedding.dimension),
            "chunk_target_chars": int(self.settings.rag_chunk_target_chars),
            "chunk_max_chars": int(self.settings.rag_chunk_max_chars),
            "chunk_overlap_chars": int(self.settings.rag_chunk_overlap_chars),
            "include_ocr": True,
            "ocr_min_confidence": float(self.settings.ocr_min_confidence),
            "hnsw_m": int(self.settings.rag_hnsw_m),
            "hnsw_construction_ef": int(self.settings.rag_hnsw_construction_ef),
            "hnsw_search_ef": int(self.settings.rag_hnsw_search_ef),
            "vector_backend": str(getattr(self.settings, "rag_vector_backend", "legacy")),
            "vector_unit_policy": VectorUnitPolicy.from_settings(self.settings).fingerprint,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _progress_stats(
        chunk_count: int,
        total_chunk_count: int,
        resumed_count: int,
        phase_started: float,
    ) -> dict[str, object]:
        elapsed = max(0.001, time.perf_counter() - phase_started)
        completed_now = max(0, int(chunk_count) - int(resumed_count))
        chunks_per_second = completed_now / elapsed if completed_now else 0.0
        remaining = max(0, int(total_chunk_count) - int(chunk_count))
        eta_seconds = (
            int(round(remaining / chunks_per_second)) if chunks_per_second > 0 else None
        )
        percent = (
            min(100.0, 100.0 * int(chunk_count) / int(total_chunk_count))
            if total_chunk_count
            else 0.0
        )
        return {
            "chunk_count": int(chunk_count),
            "total_chunk_count": int(total_chunk_count),
            "resumed_chunk_count": int(resumed_count),
            "percent": percent,
            "chunks_per_second": chunks_per_second,
            "eta_seconds": eta_seconds,
        }

    @staticmethod
    def _is_memory_error(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "out of memory",
                "not enough memory",
                "cannot allocate memory",
                "defaultcpuallocator",
                "bad allocation",
            )
        )

    @staticmethod
    def _clear_embedding_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass

    def _same_embedding_model(self, bundle: RagBundle) -> bool:
        return (
            bundle.embedding_repo_id == self.embedding.model_id
            and bundle.embedding_revision == self.embedding.model_revision
            and bundle.embedding_dimension == self.embedding.dimension
        )

    def _previous_text_ids(self, bundle: RagBundle) -> dict[str, str]:
        connection = self.rag_store.connect_read_only(bundle.metadata_path)
        try:
            return {
                str(row["text"]): str(row["stable_chunk_id"])
                for row in connection.execute("SELECT stable_chunk_id,text FROM rag_chunks")
            }
        finally:
            connection.close()

    def _unchanged_bundle(self, source, bundle: RagBundle, vector_store, token: CancelToken) -> bool:
        if vector_store is None:
            if bundle.parent_vector_mode != "metadata_only" or bundle.vector_count != 0:
                return False
        elif vector_store.count() != bundle.vector_count:
            return False
        connection = self.rag_store.connect_read_only(bundle.metadata_path)
        try:
            meta = connection.execute("SELECT * FROM rag_meta WHERE id=1").fetchone()
            if meta is None or any((
                str(meta["reranker_repo_id"]) != self.settings.rag_reranker_repo_id,
                str(meta["reranker_revision"]) != self.settings.rag_reranker_revision,
                int(meta["hnsw_m"]) != int(self.settings.rag_hnsw_m),
                int(meta["hnsw_construction_ef"]) != int(self.settings.rag_hnsw_construction_ef),
                int(meta["hnsw_search_ef"]) != int(self.settings.rag_hnsw_search_ef),
            )):
                return False
            blocks = self.block_reader.iter_blocks(
                source.path,
                source.generation,
                include_ocr=True,
                min_ocr_confidence=float(self.settings.ocr_min_confidence),
            )
            count = 0
            for chunk in self.chunk_builder.iter_chunks(blocks):
                token.throw_if_cancelled()
                count += 1
                self._check_chunk_limit(count - 1, 1)
                if count > bundle.chunk_count:
                    return False
                old = self.rag_store.load_chunk(connection, chunk.stable_chunk_id)
                if old is None or replace(old, source=replace(
                    old.source, source_generation=chunk.source.source_generation
                )) != chunk:
                    return False
            return count == bundle.chunk_count
        finally:
            connection.close()

    def _attached_vector_index(self, bundle: RagBundle):
        if not (
            bundle.vector_unit_directory
            and bundle.vector_unit_index_fingerprint
            and bundle.vector_unit_policy_fingerprint
        ):
            return None
        policy = VectorUnitPolicy.from_settings(self.settings)
        if bundle.vector_unit_policy_fingerprint != policy.fingerprint:
            return None
        try:
            attached = VectorIndexStore(self.rag_store.root).attached(
                bundle.bundle_id,
                bundle.vector_unit_directory,
                embedding_repo_id=self.embedding.model_id,
                embedding_revision=self.embedding.model_revision,
                embedding_dimension=self.embedding.dimension,
                policy_fingerprint=policy.fingerprint,
                index_fingerprint=bundle.vector_unit_index_fingerprint,
            )
            vectors = ChromaVectorStore(attached.chroma_path)
            try:
                if vectors.count() != attached.unit_count:
                    return None
            finally:
                vectors.close()
            return attached
        except Exception:
            return None

    def _check_chunk_limit(self, current_count: int, next_batch_count: int) -> None:
        maximum = int(self.settings.rag_max_chunks)
        if current_count + next_batch_count > maximum:
            raise RagCorpusTooLargeError(
                f"RAG 正文块超过 {maximum:,}，请缩小资料范围后重试"
            )

    @staticmethod
    def _log_event(event: str, **fields: object) -> None:
        try:
            log_event(event, component="rag_index_builder", **fields)
        except Exception:
            # 诊断日志不可影响候选索引的构建和发布。
            logger.debug("Unable to write RAG index diagnostic: %s", event, exc_info=True)
