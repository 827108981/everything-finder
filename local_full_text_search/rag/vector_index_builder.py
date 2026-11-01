from __future__ import annotations

import hashlib
import gc
import json
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from typing import Callable

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embedding_cache import EmbeddingVectorCache
from local_full_text_search.rag.index_store import RagBundle, RagIndexStore
from local_full_text_search.rag.models import VectorRecord
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import (
    VECTOR_INDEX_SCHEMA_VERSION, VectorIndexBundle, VectorIndexStore,
)
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


class VectorIndexBuilder:
    """为已发布 AI bundle 构建独立细粒度向量索引，不改写原 bundle。"""

    def __init__(self, rag_store: RagIndexStore, embedding, settings, *, vector_root=None) -> None:
        self.rag_store = rag_store
        self.embedding = embedding
        self.settings = settings
        self.policy = VectorUnitPolicy.from_settings(settings)
        # 验证候选模型时允许把向量索引写入隔离目录；默认路径保持原有
        # 生产行为。源 AI bundle 仍由 rag_store 读取，不会复制或改写正文索引。
        self.store = VectorIndexStore(vector_root or rag_store.root)

    def prepare(
        self, bundle: RagBundle, token: CancelToken,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
        *, publish_pointer: bool = True,
    ) -> tuple[VectorIndexBundle, bool]:
        token.throw_if_cancelled()
        source_digest = self._source_digest(bundle, token)
        fingerprint = hashlib.sha256(json.dumps({
            "schema": VECTOR_INDEX_SCHEMA_VERSION,
            "source_bundle_id": bundle.bundle_id,
            "source_content_digest": source_digest,
            "embedding_repo_id": self.embedding.model_id,
            "embedding_revision": self.embedding.model_revision,
            "embedding_dimension": self.embedding.dimension,
            "policy_fingerprint": self.policy.fingerprint,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        if publish_pointer:
            try:
                active = self.store.active(
                    bundle.bundle_id, embedding_repo_id=self.embedding.model_id,
                    embedding_revision=self.embedding.model_revision,
                    embedding_dimension=self.embedding.dimension,
                    policy_fingerprint=self.policy.fingerprint,
                )
                if (
                    active.source_content_digest == source_digest
                    and active.index_fingerprint == fingerprint
                ):
                    vectors = ChromaVectorStore(active.chroma_path)
                    try:
                        if vectors.count() == active.unit_count:
                            if progress_callback:
                                progress_callback({
                                    "stage": "vector_unit_index",
                                    "message": "细粒度向量索引未变化，继续复用",
                                    "parent_chunk_count": active.parent_chunk_count,
                                    "unit_count": active.unit_count,
                                    "reused": True,
                                })
                            return active, True
                    finally:
                        vectors.close()
            except Exception:
                pass

        parent = self.store.bundle_root(bundle.bundle_id)
        parent.mkdir(parents=True, exist_ok=True)
        # Chroma/Rust 在 Windows 进程内可能持续持有目录句柄，完整构建后重命名目录会
        # 触发 WinError 5。每次直接写入唯一、尚未被 active.json 引用的最终目录；
        # 数据库 state 校验通过后只原子替换 active.json，半成品始终不可见。
        final = self._resumable_path(parent, fingerprint) or (
            parent / f"{fingerprint[:20]}-{uuid.uuid4().hex[:8]}"
        )
        resumed = final.is_dir()
        if not resumed:
            final.mkdir()
        database_path = final / "vector_units.db"
        connection = sqlite3.connect(database_path)
        connection.row_factory = sqlite3.Row
        vectors = None
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            if resumed:
                meta = connection.execute("SELECT * FROM vector_index_meta WHERE id=1").fetchone()
                created_at = str(meta["created_at"])
                vectors = ChromaVectorStore(final / "chroma")
            else:
                self._create_schema(connection)
                connection.execute(
                    "INSERT INTO vector_index_meta VALUES(1,?,?,?,?,?,?,?,?,?,?,0,0)",
                    (VECTOR_INDEX_SCHEMA_VERSION, "building", bundle.bundle_id, source_digest,
                     self.embedding.model_id, self.embedding.model_revision, self.embedding.dimension,
                     self.policy.fingerprint, fingerprint, created_at),
                )
                vectors = ChromaVectorStore(
                    final / "chroma", create=True,
                    hnsw_m=int(getattr(self.settings, "rag_hnsw_m", 16)),
                    construction_ef=int(getattr(self.settings, "rag_hnsw_construction_ef", 200)),
                    search_ef=int(getattr(self.settings, "rag_hnsw_search_ef", 100)),
                )
            parent_count, unit_count = self._populate(
                connection, vectors, bundle, token, progress_callback
            )
            if parent_count != bundle.chunk_count or unit_count <= 0 or vectors.count() != unit_count:
                raise ValueError(
                    f"向量 V2 数量不一致：parents={parent_count}/{bundle.chunk_count}, "
                    f"units={unit_count}, vectors={vectors.count()}"
                )
            connection.execute(
                "UPDATE vector_index_meta SET state='published',parent_chunk_count=?,unit_count=? WHERE id=1",
                (parent_count, unit_count),
            )
            connection.commit()
            if str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower() != "ok":
                raise ValueError("向量 V2 SQLite 完整性检查失败")
            vectors.close()
            vectors = None
            connection.close()
            # Chroma 在同一进程按路径缓存 System；Windows 下若不释放，目录仍被占用。
            # 这里只发生在离线构建发布点，不在在线检索路径清理全局缓存。
            try:
                from chromadb.api.shared_system_client import SharedSystemClient
                SharedSystemClient.clear_system_cache()
            except (ImportError, AttributeError):
                pass
            gc.collect()
            published = VectorIndexBundle(
                source_bundle_id=bundle.bundle_id, source_content_digest=source_digest,
                embedding_repo_id=self.embedding.model_id,
                embedding_revision=self.embedding.model_revision,
                embedding_dimension=self.embedding.dimension,
                policy_fingerprint=self.policy.fingerprint, index_fingerprint=fingerprint,
                path=final, database_path=final / "vector_units.db", chroma_path=final / "chroma",
                parent_chunk_count=parent_count, unit_count=unit_count, created_at=created_at,
            )
            if publish_pointer:
                self.store.publish_pointer(published)
            if progress_callback:
                progress_callback({
                    "stage": "vector_unit_index",
                    "message": "细粒度向量索引已发布",
                    "parent_chunk_count": parent_count, "unit_count": unit_count,
                    "percent": 100, "reused": False,
                })
            return published, False
        except BaseException:
            if vectors is not None:
                vectors.close()
            try:
                connection.close()
            except sqlite3.Error:
                pass
            # 保留未被 active.json 引用的 building 目录，下一次可按确定性 unit_id 续建。
            raise

    def _populate(
        self, target: sqlite3.Connection, vectors: ChromaVectorStore,
        bundle: RagBundle, token: CancelToken,
        progress_callback: Callable[[dict[str, object]], None] | None,
    ) -> tuple[int, int]:
        source = self.rag_store.connect_read_only(bundle.metadata_path)
        unit_builder = VectorUnitBuilder(self.embedding.count_tokens, self.policy)
        parent_count = 0
        unit_count = int(target.execute("SELECT COUNT(*) FROM vector_units").fetchone()[0])
        pending_units = []
        # 一次收集更多单元，再让 Embedding provider 按其安全批量（本机实测为16）
        # 内部分批。这样 sentence-transformers 能在更大范围内按长度排序，减少 padding。
        write_batch_size = max(
            16, min(256, int(getattr(self.settings, "rag_index_write_batch_size", 128)))
        )
        length_bucketing = bool(
            getattr(self.settings, "rag_embedding_length_bucketing", True)
        )
        populate_started = time.perf_counter()
        cache = EmbeddingVectorCache(
            self.rag_store.root / "embedding_cache.db",
            model_id=self.embedding.model_id,
            model_revision=self.embedding.model_revision,
            dimension=self.embedding.dimension,
        )

        def flush() -> None:
            nonlocal unit_count
            if not pending_units:
                return
            token.throw_if_cancelled()
            ordered_units = (
                sorted(pending_units, key=lambda item: len(item.embedding_text))
                if length_bucketing else list(pending_units)
            )
            unique_texts = list(dict.fromkeys(item.embedding_text for item in ordered_units))
            try:
                vectors_by_text = cache.get_many(unique_texts)
            except (OSError, sqlite3.Error, ValueError):
                vectors_by_text = {}
            missing_texts = [text for text in unique_texts if text not in vectors_by_text]
            if missing_texts:
                encoded = self.embedding.embed_documents(missing_texts)
                new_vectors = dict(zip(missing_texts, encoded, strict=True))
                vectors_by_text.update(new_vectors)
                try:
                    cache.put_many(new_vectors)
                except (OSError, sqlite3.Error, ValueError):
                    pass
            embeddings = [vectors_by_text[item.embedding_text] for item in ordered_units]
            records = []
            for item, embedding in zip(ordered_units, embeddings, strict=True):
                row = unit_rows[item.parent_chunk_id]
                metadata = json.loads(row["source_json"])
                target.execute(
                    "INSERT OR REPLACE INTO vector_units VALUES(?,?,?,?,?,?,?,?,?)",
                    (item.unit_id, item.parent_chunk_id, item.ordinal, item.kind, item.text,
                     item.embedding_text, item.start_char, item.end_char, item.token_count),
                )
                records.append(VectorRecord(item.unit_id, tuple(embedding), {
                    "stable_chunk_id": item.parent_chunk_id, "unit_id": item.unit_id,
                    "unit_ordinal": item.ordinal, "unit_kind": item.kind,
                    "root_id": int(metadata.get("root_id") or 0),
                    "extension": str(metadata.get("extension") or "").lower(),
                    "source_type": str(metadata.get("source_type") or "native_text"),
                }))
            vectors.upsert(records)
            unit_count += len(pending_units)
            pending_units.clear()
            target.commit()

        try:
            rows = source.execute("SELECT * FROM rag_chunks ORDER BY stable_chunk_id")
            unit_rows: dict[str, sqlite3.Row] = {}
            for row in rows:
                token.throw_if_cancelled()
                chunk = self.rag_store.load_chunk(source, str(row["stable_chunk_id"]))
                if chunk is None:
                    raise ValueError(f"无法读取父 chunk：{row['stable_chunk_id']}")
                parent_count += 1
                if progress_callback and parent_count % 64 == 0:
                    elapsed = max(0.001, time.perf_counter() - populate_started)
                    parents_per_second = parent_count / elapsed
                    remaining_parents = max(0, bundle.chunk_count - parent_count)
                    progress_callback({
                        "stage": "vector_unit_index",
                        "message": "正在生成并写入细粒度向量单元",
                        "processed_parent_chunks": parent_count,
                        "total_parent_chunks": bundle.chunk_count,
                        "unit_count": unit_count + len(pending_units),
                        "percent": min(99, int(parent_count * 100 / max(1, bundle.chunk_count))),
                        "parents_per_second": parents_per_second,
                        "eta_seconds": int(round(remaining_parents / parents_per_second)),
                    })
                unit_rows[chunk.stable_chunk_id] = row
                built = unit_builder.build(chunk)
                if not built:
                    raise ValueError(f"父 chunk 未产生向量单元：{chunk.stable_chunk_id}")
                existing = {
                    str(found[0]) for found in target.execute(
                        "SELECT unit_id FROM vector_units WHERE parent_chunk_id=?",
                        (chunk.stable_chunk_id,),
                    ).fetchall()
                }
                expected = {item.unit_id for item in built}
                if existing == expected:
                    continue
                if existing:
                    target.execute(
                        "DELETE FROM vector_units WHERE parent_chunk_id=?",
                        (chunk.stable_chunk_id,),
                    )
                    unit_count -= len(existing)
                for item in built:
                    pending_units.append(item)
                    if len(pending_units) >= write_batch_size:
                        flush()
            flush()
            return parent_count, unit_count
        finally:
            source.close()
            cache.close()

    @staticmethod
    def _resumable_path(parent, fingerprint: str):
        for candidate in sorted(parent.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True):
            database = candidate / "vector_units.db"
            if not candidate.is_dir() or not database.is_file():
                continue
            try:
                connection = sqlite3.connect(database)
                connection.row_factory = sqlite3.Row
                try:
                    meta = connection.execute(
                        "SELECT state,index_fingerprint FROM vector_index_meta WHERE id=1"
                    ).fetchone()
                    if (
                        meta is not None and str(meta["state"]) == "building"
                        and str(meta["index_fingerprint"]) == fingerprint
                        and (candidate / "chroma").is_dir()
                    ):
                        return candidate
                finally:
                    connection.close()
            except sqlite3.Error:
                continue
        return None

    def _source_digest(self, bundle: RagBundle, token: CancelToken) -> str:
        digest = hashlib.sha256()
        connection = self.rag_store.connect_read_only(bundle.metadata_path)
        try:
            for index, row in enumerate(connection.execute(
                "SELECT stable_chunk_id,text_hash,source_json FROM rag_chunks ORDER BY stable_chunk_id"
            ), 1):
                if index % 256 == 0:
                    token.throw_if_cancelled()
                digest.update(json.dumps(
                    [str(row["stable_chunk_id"]), str(row["text_hash"]), str(row["source_json"])],
                    ensure_ascii=False, separators=(",", ":"),
                ).encode("utf-8"))
                digest.update(b"\n")
            return digest.hexdigest()
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            PRAGMA journal_mode=DELETE;
            CREATE TABLE vector_index_meta(
                id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL,
                state TEXT NOT NULL, source_bundle_id TEXT NOT NULL,
                source_content_digest TEXT NOT NULL, embedding_repo_id TEXT NOT NULL,
                embedding_revision TEXT NOT NULL, embedding_dimension INTEGER NOT NULL,
                policy_fingerprint TEXT NOT NULL, index_fingerprint TEXT NOT NULL,
                created_at TEXT NOT NULL, parent_chunk_count INTEGER NOT NULL,
                unit_count INTEGER NOT NULL
            );
            CREATE TABLE vector_units(
                unit_id TEXT PRIMARY KEY, parent_chunk_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, kind TEXT NOT NULL, text TEXT NOT NULL,
                embedding_text TEXT NOT NULL, start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL, token_count INTEGER NOT NULL,
                UNIQUE(parent_chunk_id,ordinal)
            );
            CREATE INDEX idx_vector_units_parent ON vector_units(parent_chunk_id,ordinal);
        """)
