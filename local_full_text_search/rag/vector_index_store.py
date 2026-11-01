from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from local_full_text_search.rag.errors import VectorIndexNotReadyError


VECTOR_INDEX_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class VectorIndexBundle:
    source_bundle_id: str
    source_content_digest: str
    embedding_repo_id: str
    embedding_revision: str
    embedding_dimension: int
    policy_fingerprint: str
    index_fingerprint: str
    path: Path
    database_path: Path
    chroma_path: Path
    parent_chunk_count: int
    unit_count: int
    created_at: str


class VectorIndexStore:
    def __init__(self, rag_root: Path) -> None:
        self.root = Path(rag_root).resolve() / "vector"

    def bundle_root(self, source_bundle_id: str) -> Path:
        return self.root / source_bundle_id

    def active(
        self, source_bundle_id: str, *, embedding_repo_id: str,
        embedding_revision: str, embedding_dimension: int, policy_fingerprint: str,
    ) -> VectorIndexBundle:
        pointer = self.bundle_root(source_bundle_id) / "active.json"
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
            expected = {
                "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
                "source_bundle_id": source_bundle_id,
                "embedding_repo_id": embedding_repo_id,
                "embedding_revision": embedding_revision,
                "embedding_dimension": int(embedding_dimension),
                "policy_fingerprint": policy_fingerprint,
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    raise ValueError(f"{key} mismatch")
            path = (pointer.parent / str(payload["directory"])).resolve()
            if path.parent != pointer.parent.resolve():
                raise ValueError("invalid vector index path")
            database_path = path / "vector_units.db"
            connection = self.connect_read_only(database_path)
            try:
                meta = connection.execute("SELECT * FROM vector_index_meta WHERE id=1").fetchone()
                if meta is None or meta["state"] != "published":
                    raise ValueError("vector unit index is not published")
                for key, value in expected.items():
                    if key == "schema_version":
                        actual = int(meta[key])
                    elif key == "embedding_dimension":
                        actual = int(meta[key])
                    else:
                        actual = str(meta[key])
                        value = str(value)
                    if actual != value:
                        raise ValueError(f"database {key} mismatch")
                if str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower() != "ok":
                    raise ValueError("vector unit database quick_check failed")
            finally:
                connection.close()
            return VectorIndexBundle(
                source_bundle_id=source_bundle_id,
                source_content_digest=str(payload["source_content_digest"]),
                embedding_repo_id=embedding_repo_id,
                embedding_revision=embedding_revision,
                embedding_dimension=int(embedding_dimension),
                policy_fingerprint=policy_fingerprint,
                index_fingerprint=str(payload["index_fingerprint"]),
                path=path, database_path=database_path, chroma_path=path / "chroma",
                parent_chunk_count=int(payload["parent_chunk_count"]),
                unit_count=int(payload["unit_count"]), created_at=str(payload.get("created_at") or ""),
            )
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error, json.JSONDecodeError) as exc:
            raise VectorIndexNotReadyError(
                "向量 V2 索引未就绪或与当前 AI 索引/模型/切分策略不一致；请先构建对应索引"
            ) from exc

    def attached(
        self,
        source_bundle_id: str,
        directory: str,
        *,
        embedding_repo_id: str,
        embedding_revision: str,
        embedding_dimension: int,
        policy_fingerprint: str,
        index_fingerprint: str,
    ) -> VectorIndexBundle:
        """读取父 RAG manifest 已绑定的细粒度索引，不依赖另一个活动指针。"""
        try:
            parent = self.bundle_root(source_bundle_id).resolve()
            path = (parent / str(directory)).resolve()
            if path.parent != parent or not path.is_dir():
                raise ValueError("invalid attached vector index path")
            database_path = path / "vector_units.db"
            connection = self.connect_read_only(database_path)
            try:
                meta = connection.execute(
                    "SELECT * FROM vector_index_meta WHERE id=1"
                ).fetchone()
                if meta is None or str(meta["state"]) != "published":
                    raise ValueError("attached vector index is not published")
                expected = {
                    "source_bundle_id": source_bundle_id,
                    "embedding_repo_id": embedding_repo_id,
                    "embedding_revision": embedding_revision,
                    "embedding_dimension": int(embedding_dimension),
                    "policy_fingerprint": policy_fingerprint,
                    "index_fingerprint": index_fingerprint,
                }
                for key, value in expected.items():
                    actual = int(meta[key]) if key == "embedding_dimension" else str(meta[key])
                    wanted = int(value) if key == "embedding_dimension" else str(value)
                    if actual != wanted:
                        raise ValueError(f"database {key} mismatch")
                if str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower() != "ok":
                    raise ValueError("vector unit database quick_check failed")
                parent_chunk_count = int(meta["parent_chunk_count"])
                unit_count = int(meta["unit_count"])
                source_content_digest = str(meta["source_content_digest"])
                created_at = str(meta["created_at"])
            finally:
                connection.close()
            if not (path / "chroma").is_dir() or unit_count <= 0:
                raise ValueError("attached vector index files are incomplete")
            return VectorIndexBundle(
                source_bundle_id=source_bundle_id,
                source_content_digest=source_content_digest,
                embedding_repo_id=embedding_repo_id,
                embedding_revision=embedding_revision,
                embedding_dimension=int(embedding_dimension),
                policy_fingerprint=policy_fingerprint,
                index_fingerprint=index_fingerprint,
                path=path,
                database_path=database_path,
                chroma_path=path / "chroma",
                parent_chunk_count=parent_chunk_count,
                unit_count=unit_count,
                created_at=created_at,
            )
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error) as exc:
            raise VectorIndexNotReadyError(
                "AI 索引绑定的细粒度向量版本损坏或与当前模型/策略不一致"
            ) from exc

    @staticmethod
    def connect_read_only(path: Path) -> sqlite3.Connection:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        connection = sqlite3.connect(
            f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True, timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    @staticmethod
    def publish_pointer(bundle: VectorIndexBundle) -> None:
        pointer = bundle.path.parent / "active.json"
        payload = {
            "schema_version": VECTOR_INDEX_SCHEMA_VERSION,
            "source_bundle_id": bundle.source_bundle_id,
            "source_content_digest": bundle.source_content_digest,
            "embedding_repo_id": bundle.embedding_repo_id,
            "embedding_revision": bundle.embedding_revision,
            "embedding_dimension": bundle.embedding_dimension,
            "policy_fingerprint": bundle.policy_fingerprint,
            "index_fingerprint": bundle.index_fingerprint,
            "directory": bundle.path.name,
            "parent_chunk_count": bundle.parent_chunk_count,
            "unit_count": bundle.unit_count,
            "created_at": bundle.created_at,
        }
        temporary = pointer.with_suffix(".json.tmp")
        with temporary.open("wb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, pointer)

    @staticmethod
    def load_units(database_path: Path, unit_ids: list[str]) -> dict[str, dict[str, object]]:
        if not unit_ids:
            return {}
        connection = VectorIndexStore.connect_read_only(database_path)
        try:
            output: dict[str, dict[str, object]] = {}
            for offset in range(0, len(unit_ids), 500):
                batch = unit_ids[offset:offset + 500]
                marks = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"SELECT * FROM vector_units WHERE unit_id IN ({marks})", batch,
                ).fetchall()
                output.update({str(row["unit_id"]): dict(row) for row in rows})
            return output
        finally:
            connection.close()
