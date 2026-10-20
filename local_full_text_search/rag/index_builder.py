from __future__ import annotations

import logging
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
from local_full_text_search.rag.embeddings.local_bge import MAX_EMBEDDING_BATCH_SIZE
from local_full_text_search.rag.errors import (
    RagCorpusTooLargeError,
    RagIndexCorruptError,
    RagIndexMissingError,
    RagSourceChangedError,
)
from local_full_text_search.rag.index_store import RagBundle, RagIndexStore
from local_full_text_search.rag.models import VectorRecord
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
        chunk_count = 0
        batch_index = 0
        reused_count = 0
        embedded_count = 0

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
                if self._same_embedding_model(previous_bundle):
                    previous_vector_store = ChromaVectorStore(previous_bundle.chroma_path)
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
                        )
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
            self._log_event(
                "rag.index.stage_started",
                run_id=run_id,
                stage="candidate_create",
            )
            candidate_started = time.perf_counter()
            try:
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
            )
            try:
                token.throw_if_cancelled()
                report("vector_store", "正在创建 Chroma HNSW 向量索引")
                self._log_event(
                    "rag.index.stage_started",
                    run_id=run_id,
                    stage="vector_store_create",
                )
                vector_store_started = time.perf_counter()
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
                for chunk in chunks:
                    token.throw_if_cancelled()
                    batch.append(chunk)
                    if len(batch) >= self.write_batch_size:
                        self._check_chunk_limit(chunk_count, len(batch))
                        collect_ms = int((time.perf_counter() - collect_started) * 1000)
                        batch_index += 1
                        indexed, reused, embedded = self._index_batch(
                            batch,
                            connection,
                            vector_store,
                            token,
                            previous_vector_store=previous_vector_store,
                            previous_text_ids=previous_text_ids,
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
                            "正在分块并生成向量",
                            chunk_count=chunk_count,
                            batch_index=batch_index,
                            write_batch_size=self.write_batch_size,
                            embedding_batch_size=self.embedding_batch_size,
                        )
                        if chunk_count > int(self.settings.rag_max_chunks):
                            raise RagCorpusTooLargeError(
                                f"RAG 正文块超过 {int(self.settings.rag_max_chunks):,}，请缩小资料范围后重试"
                            )
                if batch:
                    self._check_chunk_limit(chunk_count, len(batch))
                    collect_ms = int((time.perf_counter() - collect_started) * 1000)
                    batch_index += 1
                    indexed, reused, embedded = self._index_batch(
                        batch,
                        connection,
                        vector_store,
                        token,
                        previous_vector_store=previous_vector_store,
                        previous_text_ids=previous_text_ids,
                        run_id=run_id,
                        batch_index=batch_index,
                        chunk_read_and_chunk_ms=collect_ms,
                    )
                    chunk_count += indexed
                    reused_count += reused
                    embedded_count += embedded
                    report(
                        "embedding",
                        "正在完成最后一批向量",
                        chunk_count=chunk_count,
                        batch_index=batch_index,
                        write_batch_size=self.write_batch_size,
                        embedding_batch_size=self.embedding_batch_size,
                    )
                if chunk_count > int(self.settings.rag_max_chunks):
                    raise RagCorpusTooLargeError(
                        f"RAG 正文块超过 {int(self.settings.rag_max_chunks):,}，请缩小资料范围后重试"
                    )
                token.throw_if_cancelled()
                report("validating", "正在校验正文与向量数量")
                self._log_event(
                    "rag.index.stage_started",
                    run_id=run_id,
                    stage="validation",
                )
                validation_started = time.perf_counter()
                vector_count = vector_store.count()
                if vector_count != chunk_count:
                    raise RuntimeError(
                        f"Chroma 数量不一致：预期 {chunk_count}，实际 {vector_count}"
                    )
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
                    candidate, connection, vector_count=vector_count
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
                    elapsed_ms=elapsed,
                )
                report("completed", "AI 索引已就绪", chunk_count=chunk_count)
                return RagBuildSummary(
                    source_generation=source.generation,
                    bundle_id=published.bundle_id,
                    chunk_count=published.chunk_count,
                    vector_count=published.vector_count,
                    elapsed_ms=elapsed,
                    bundle_path=str(published.path),
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
                if candidate is not None:
                    self.rag_store.abort(candidate, connection)
                raise
            finally:
                if vector_store is not None:
                    vector_store.close()
                if previous_vector_store is not None:
                    previous_vector_store.close()

    def _index_batch(
        self,
        chunks,
        connection,
        vector_store,
        token: CancelToken,
        *,
        previous_vector_store,
        previous_text_ids: dict[str, str],
        run_id: str,
        batch_index: int,
        chunk_read_and_chunk_ms: int,
    ) -> tuple[int, int, int]:
        token.throw_if_cancelled()
        embedding_started = time.perf_counter()
        embedding_ms = 0
        embedding_batch_count = 0
        vectors_by_text: dict[str, list[float]] = {}
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
        for offset in range(0, len(new_texts), self.embedding_batch_size):
            token.throw_if_cancelled()
            text_batch = new_texts[offset : offset + self.embedding_batch_size]
            encode_started = time.perf_counter()
            encoded = self.embedding.embed_documents(text_batch)
            embedding_ms += int((time.perf_counter() - encode_started) * 1000)
            embedding_batch_count += 1
            if len(encoded) != len(text_batch):
                raise ValueError("Embedding 返回的向量条数与 chunk 条数不一致")
            vectors_by_text.update(zip(text_batch, encoded, strict=True))
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
            embedding_batch_size=self.embedding_batch_size,
            embedding_batch_count=embedding_batch_count,
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
        if vector_store.count() != bundle.vector_count:
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
