from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.block_reader import BlockReader, DocumentBlocks
from local_full_text_search.rag.chunk_builder import ChunkBuilder, ChunkPolicy
from local_full_text_search.rag.embeddings.base import EmbeddingProvider, normalize_vectors
from local_full_text_search.rag.index_store import RagBuildReport, RagIndexStore
from local_full_text_search.rag.models import RagBuildCancelled, RagProgress, RagSourceError


class RagIndexBuilder:
    def __init__(
        self,
        generation_base_path: Path,
        *,
        target_chars: int = 380,
        max_chars: int = 600,
        overlap_chars: int = 48,
        embedding_batch_size: int = 32,
        commit_chunks: int = 256,
    ) -> None:
        self.generation_base_path = Path(generation_base_path).resolve()
        self.store = RagIndexStore(self.generation_base_path)
        self.generation_store = IndexGenerationStore(self.generation_base_path)
        self.target_chars = max(128, int(target_chars))
        self.max_chars = max(self.target_chars, int(max_chars))
        self.overlap_chars = max(0, min(int(overlap_chars), self.max_chars - 1))
        self.embedding_batch_size = max(1, min(64, int(embedding_batch_size)))
        self.commit_chunks = max(1, int(commit_chunks))

    def build(
        self,
        embedding: EmbeddingProvider,
        token: CancelToken,
        progress: Callable[[RagProgress], None] | None = None,
    ) -> RagBuildReport:
        started = time.perf_counter()
        candidate = None
        try:
            with BlockReader(self.generation_base_path).open_active() as reader:
                token.throw_if_cancelled()
                generation = reader.generation
                policy_version = f"evidence-v2-{embedding.identity.revision[:12]}"
                published = self.store.active()
                if (
                    published is not None
                    and published.generation == generation
                    and published.model == embedding.identity
                    and published.chunk_policy_version == policy_version
                ):
                    try:
                        report = self.store.published_report(published)
                        self._emit(
                            progress,
                            "ready",
                            "当前 generation 的智能索引已是最新版本",
                            report.chunk_count,
                            report.chunk_count,
                            False,
                        )
                        return report
                    except sqlite3.Error:
                        # A corrupt published sidecar is replaced by a fresh
                        # candidate only after the source generation is pinned.
                        pass
                policy = ChunkPolicy(
                    version=policy_version,
                    target_chars=self.target_chars,
                    max_chars=self.max_chars,
                    overlap_chars=self.overlap_chars,
                    embedding_model_revision=embedding.identity.revision,
                    max_embedding_tokens=embedding.identity.max_tokens,
                )
                chunk_builder = ChunkBuilder(policy, token_counter=embedding.count_tokens)
                candidate = self.store.begin_candidate(
                    source_generation=generation,
                    source_database=reader.active.path,
                    model=embedding.identity,
                    chunk_policy_version=policy_version,
                )
                counts = {
                    "chunk_count": 0,
                    "source_count": 0,
                    "reused_embeddings": 0,
                    "generated_embeddings": 0,
                }
                last_commit = 0
                previous_context = self.store.acquire_compatible_previous(embedding.identity)
                with previous_context as previous, self.store.connect_candidate(candidate) as target:
                    previous_connection = previous[0] if previous is not None else None
                    self._emit(progress, "chunking", "正在读取并切分证据块", 0, 0, True)
                    for document in reader.iter_documents():
                        token.throw_if_cancelled()
                        chunks = chunk_builder.build(document)
                        for offset in range(0, len(chunks), self.embedding_batch_size):
                            token.throw_if_cancelled()
                            batch = chunks[offset : offset + self.embedding_batch_size]
                            cached = self._cached_vectors(previous_connection, batch, embedding)
                            missing = [chunk for chunk in batch if chunk.stable_chunk_id not in cached]
                            generated: dict[str, np.ndarray] = {}
                            if missing:
                                vectors = embedding.embed_documents(
                                    [chunk.text for chunk in missing],
                                    token,
                                )
                                vectors = normalize_vectors(vectors, embedding.identity.dimension)
                                if len(vectors) != len(missing):
                                    raise ValueError("Embedding provider returned the wrong number of vectors")
                                generated = {
                                    chunk.stable_chunk_id: vectors[index]
                                    for index, chunk in enumerate(missing)
                                }
                            for chunk in batch:
                                token.throw_if_cancelled()
                                vector = cached.get(chunk.stable_chunk_id)
                                if vector is not None:
                                    counts["reused_embeddings"] += 1
                                else:
                                    vector = generated[chunk.stable_chunk_id]
                                    counts["generated_embeddings"] += 1
                                self._assert_sources_reconstruct(chunk, document)
                                self._insert_chunk(target, chunk, vector)
                                counts["chunk_count"] += 1
                                counts["source_count"] += len(chunk.sources)
                            if counts["chunk_count"] - last_commit >= self.commit_chunks:
                                self._update_build_meta(target, counts, started)
                                target.commit()
                                last_commit = counts["chunk_count"]
                                self._emit(
                                    progress,
                                    "embedding",
                                    f"已生成 {counts['generated_embeddings']:,} / "
                                    f"复用 {counts['reused_embeddings']:,} 个向量",
                                    counts["chunk_count"],
                                    0,
                                    True,
                                )
                    token.throw_if_cancelled()
                    if counts["chunk_count"] <= 0:
                        raise ValueError("当前索引没有可用于 RAG 的 PDF、DOCX 或 TXT 正文")
                    meta = target.execute("SELECT fts_mode FROM rag_meta WHERE id=1").fetchone()
                    if str(meta["fts_mode"]) == "trigram":
                        target.execute("INSERT INTO rag_chunks_fts(rag_chunks_fts) VALUES('rebuild')")
                    self._update_build_meta(target, counts, started)
                    target.commit()
                token.throw_if_cancelled()
                self._emit(progress, "validating", "正在校验并发布智能索引", counts["chunk_count"], counts["chunk_count"], False)
                active = self.generation_store.active()
                if active is None or active.generation != generation:
                    raise RuntimeError("搜索索引已更新，本轮 RAG 候选已过期")
                report = self.store.validate_candidate(candidate)
                self.store.publish(candidate, validated_report=report)
                self.store.cleanup_retired()
                self._emit(progress, "complete", "智能问答索引已就绪", report.chunk_count, report.chunk_count, False)
                return report
        except BaseException as exc:
            if candidate is not None:
                try:
                    self.store.abort(candidate, str(exc))
                except Exception:
                    pass
            if token.cancelled:
                raise RagBuildCancelled(token.reason or "RAG 索引构建已取消") from exc
            raise

    @staticmethod
    def _cached_vectors(
        connection,
        chunks,
        embedding: EmbeddingProvider,
    ) -> dict[str, np.ndarray]:
        if connection is None or not chunks:
            return {}
        identifiers = [chunk.stable_chunk_id for chunk in chunks]
        placeholders = ",".join("?" for _ in identifiers)
        rows = connection.execute(
            "SELECT c.stable_chunk_id, c.text_hash, e.vector_blob, e.dimension "
            "FROM rag_chunks c JOIN rag_embeddings e ON e.chunk_rowid=c.chunk_rowid "
            f"WHERE c.stable_chunk_id IN ({placeholders})",
            tuple(identifiers),
        ).fetchall()
        expected_hashes = {chunk.stable_chunk_id: chunk.text_hash for chunk in chunks}
        result: dict[str, np.ndarray] = {}
        for row in rows:
            stable_id = str(row["stable_chunk_id"])
            if (
                str(row["text_hash"]) != expected_hashes.get(stable_id)
                or int(row["dimension"]) != embedding.identity.dimension
            ):
                continue
            vector = np.frombuffer(row["vector_blob"], dtype="<f4").copy()
            if vector.shape != (embedding.identity.dimension,) or not np.isfinite(vector).all():
                continue
            norm = float(np.linalg.norm(vector))
            if norm <= 0:
                continue
            result[stable_id] = np.ascontiguousarray(vector / norm, dtype=np.float32)
        return result

    @staticmethod
    def _assert_sources_reconstruct(chunk, document: DocumentBlocks) -> None:
        blocks = {block.block_id: block for block in document.blocks}
        for source in chunk.sources:
            rebuilt: list[str] = []
            for span in source.spans:
                block = blocks.get(span.block_id)
                if block is None or span.raw_end > len(block.raw_text):
                    raise RagSourceError("chunk source span does not exist in active generation")
                rebuilt.append(
                    span.separator_before + block.raw_text[span.raw_start : span.raw_end]
                )
            if "".join(rebuilt) != chunk.text:
                raise RagSourceError("chunk text cannot be reconstructed from source spans")

    @staticmethod
    def _insert_chunk(connection, chunk, vector: np.ndarray) -> None:
        cursor = connection.execute(
            "INSERT INTO rag_chunks(stable_chunk_id,text,normalized_text,text_hash,"
            "chunk_policy_version,source_type,ocr_confidence) VALUES(?,?,?,?,?,?,?)",
            (
                chunk.stable_chunk_id,
                chunk.text,
                chunk.normalized_text,
                chunk.text_hash,
                chunk.chunk_policy_version,
                chunk.sources[0].source_type,
                chunk.sources[0].ocr_confidence,
            ),
        )
        chunk_rowid = int(cursor.lastrowid)
        for source in chunk.sources:
            connection.execute(
                "INSERT INTO rag_sources(chunk_rowid,root_id,file_id,document_id,file_path,filename,"
                "extension,location_text,page_number,line_start,line_end,source_type,ocr_confidence,spans_json) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk_rowid,
                    source.root_id,
                    source.file_id,
                    source.document_id,
                    source.file_path,
                    source.filename,
                    source.extension,
                    source.location_text,
                    source.page_number,
                    source.line_start,
                    source.line_end,
                    source.source_type,
                    source.ocr_confidence,
                    json.dumps(
                        [
                            {
                                "block_id": span.block_id,
                                "block_index": span.block_index,
                                "raw_start": span.raw_start,
                                "raw_end": span.raw_end,
                                "separator_before": span.separator_before,
                            }
                            for span in source.spans
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
        little_endian = np.ascontiguousarray(vector, dtype="<f4")
        norm = float(np.linalg.norm(little_endian))
        connection.execute(
            "INSERT INTO rag_embeddings(chunk_rowid,vector_blob,dimension,norm) VALUES(?,?,?,?)",
            (chunk_rowid, little_endian.tobytes(), int(little_endian.size), norm),
        )

    @staticmethod
    def _update_build_meta(connection, counts: dict[str, int], started: float) -> None:
        connection.execute(
            "UPDATE rag_meta SET reused_embeddings=?, generated_embeddings=?, elapsed_ms=? WHERE id=1",
            (
                counts["reused_embeddings"],
                counts["generated_embeddings"],
                int((time.perf_counter() - started) * 1000),
            ),
        )

    @staticmethod
    def _emit(
        callback: Callable[[RagProgress], None] | None,
        stage: str,
        message: str,
        completed: int,
        total: int,
        indeterminate: bool,
    ) -> None:
        if callback is not None:
            callback(RagProgress(stage, message, completed, total, indeterminate))
