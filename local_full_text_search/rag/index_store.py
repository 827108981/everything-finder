from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from local_full_text_search.rag.models import ModelIdentity, RagIndexNotReadyError


@dataclass(frozen=True, slots=True)
class CandidateRagIndex:
    generation: int
    path: Path
    source_database: Path
    model: ModelIdentity
    chunk_policy_version: str
    owner_nonce: str


@dataclass(frozen=True, slots=True)
class PublishedRagIndex:
    generation: int
    path: Path
    model: ModelIdentity
    chunk_policy_version: str
    fts_mode: str


@dataclass(frozen=True, slots=True)
class RagBuildReport:
    generation: int
    chunk_count: int
    source_count: int
    reused_embeddings: int
    generated_embeddings: int
    elapsed_ms: int
    fts_mode: str


@dataclass(frozen=True, slots=True)
class RagReadHandle:
    generation: int
    model: ModelIdentity
    fts_mode: str
    connection: sqlite3.Connection


class _RagLease:
    def __init__(self, registry: "_RagLeaseRegistry", key: tuple[Path, str]) -> None:
        self._registry = registry
        self._key = key
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._registry.release(self._key)


class _RagLeaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._leases: dict[tuple[Path, str], int] = {}

    def acquire(self, key: tuple[Path, str]) -> _RagLease:
        with self._lock:
            self._leases[key] = self._leases.get(key, 0) + 1
        return _RagLease(self, key)

    def release(self, key: tuple[Path, str]) -> None:
        with self._lock:
            count = self._leases.get(key, 0) - 1
            if count <= 0:
                self._leases.pop(key, None)
            else:
                self._leases[key] = count

    def active(self, key: tuple[Path, str]) -> int:
        with self._lock:
            return self._leases.get(key, 0)


_LEASES_LOCK = threading.Lock()
_SHARED_LEASES: dict[Path, _RagLeaseRegistry] = {}


class RagIndexStore:
    """Atomic, generation-aligned SQLite sidecars for RAG."""

    SCHEMA_VERSION = 1

    def __init__(self, generation_base_path: Path) -> None:
        self.generation_base_path = Path(generation_base_path).resolve()
        self.root = self.generation_base_path.parent / "rag-index"
        self.manifest_path = self.root / "rag.manifest.json"
        self.manifest_backup_path = self.root / "rag.manifest.json.bak"
        self.lock_path = self.root / "rag.build.lock"
        self._owner_nonce: str | None = None
        with _LEASES_LOCK:
            self._leases = _SHARED_LEASES.setdefault(self.root, _RagLeaseRegistry())

    def active(self) -> PublishedRagIndex | None:
        manifest = self._read_manifest()
        if manifest is None:
            return None
        try:
            generation = int(manifest["source_generation"])
            filename = str(manifest["active_file"])
            path = (self.root / filename).resolve()
            if path.parent != self.root or not path.is_file():
                return None
            model = ModelIdentity(
                str(manifest["embedding_model_id"]),
                str(manifest["embedding_model_revision"]),
                int(manifest["embedding_dimension"]),
                int(manifest["embedding_max_tokens"]),
            )
            connection = self._connect_readonly(path)
            try:
                row = connection.execute(
                    "SELECT source_generation, state, embedding_model_id, "
                    "embedding_model_revision, embedding_dimension, "
                    "embedding_max_tokens, chunk_policy_version, fts_mode "
                    "FROM rag_meta WHERE id = 1"
                ).fetchone()
                if row is None or str(row["state"]) != "published":
                    return None
                if int(row["source_generation"]) != generation:
                    return None
                if (
                    str(row["embedding_model_id"]) != model.model_id
                    or str(row["embedding_model_revision"]) != model.revision
                    or int(row["embedding_dimension"]) != model.dimension
                ):
                    return None
                return PublishedRagIndex(
                    generation=generation,
                    path=path,
                    model=model,
                    chunk_policy_version=str(row["chunk_policy_version"]),
                    fts_mode=str(row["fts_mode"]),
                )
            finally:
                connection.close()
        except (KeyError, OSError, ValueError, TypeError, sqlite3.Error):
            return None

    def published_report(self, active: PublishedRagIndex) -> RagBuildReport:
        connection = self._connect_readonly(active.path)
        try:
            check = str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower()
            if check != "ok":
                raise sqlite3.DatabaseError(f"published sidecar quick_check failed: {check}")
            chunks = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
            sources = int(connection.execute("SELECT COUNT(*) FROM rag_sources").fetchone()[0])
            embeddings = int(connection.execute("SELECT COUNT(*) FROM rag_embeddings").fetchone()[0])
            meta = connection.execute(
                "SELECT reused_embeddings,generated_embeddings,elapsed_ms,fts_mode "
                "FROM rag_meta WHERE id=1"
            ).fetchone()
            if meta is None:
                raise sqlite3.DatabaseError("published sidecar metadata is missing")
            if chunks <= 0 or sources < chunks or embeddings != chunks:
                raise sqlite3.DatabaseError("published sidecar has incomplete chunk/source/vector rows")
            return RagBuildReport(
                generation=active.generation,
                chunk_count=chunks,
                source_count=sources,
                reused_embeddings=int(meta["reused_embeddings"]),
                generated_embeddings=int(meta["generated_embeddings"]),
                elapsed_ms=int(meta["elapsed_ms"]),
                fts_mode=str(meta["fts_mode"]),
            )
        finally:
            connection.close()

    @contextmanager
    def acquire_matching(
        self,
        source_generation: int,
        model: ModelIdentity,
    ) -> Iterator[RagReadHandle]:
        active = self.active()
        if active is None:
            raise RagIndexNotReadyError("智能问答索引尚未发布")
        if active.generation != int(source_generation):
            raise RagIndexNotReadyError(
                f"智能索引 generation={active.generation}，当前搜索索引 generation={source_generation}"
            )
        if active.model != model:
            raise RagIndexNotReadyError("Embedding 模型版本变化，需要重建智能问答索引")
        lease_key = (active.path, active.model.revision)
        lease = self._leases.acquire(lease_key)
        connection: sqlite3.Connection | None = None
        try:
            if not active.path.is_file():
                raise RagIndexNotReadyError("智能问答索引文件已被清理")
            connection = self._connect_readonly(active.path)
            connection.execute("BEGIN")
            row = connection.execute(
                "SELECT source_generation, state, embedding_model_id, "
                "embedding_model_revision, embedding_dimension, embedding_max_tokens, fts_mode "
                "FROM rag_meta WHERE id = 1"
            ).fetchone()
            if (
                row is None
                or int(row["source_generation"]) != int(source_generation)
                or str(row["state"]) != "published"
                or str(row["embedding_model_id"]) != model.model_id
                or str(row["embedding_model_revision"]) != model.revision
                or int(row["embedding_dimension"]) != model.dimension
                or int(row["embedding_max_tokens"]) != model.max_tokens
            ):
                raise RagIndexNotReadyError("智能索引元数据与当前模型/搜索版本不一致")
            yield RagReadHandle(
                generation=active.generation,
                model=active.model,
                fts_mode=str(row["fts_mode"]),
                connection=connection,
            )
        finally:
            if connection is not None:
                connection.close()
            lease.release()

    def begin_candidate(
        self,
        *,
        source_generation: int,
        source_database: Path,
        model: ModelIdentity,
        chunk_policy_version: str,
    ) -> CandidateRagIndex:
        self.root.mkdir(parents=True, exist_ok=True)
        self._recover_stale_build()
        nonce = uuid.uuid4().hex
        payload = {
            "pid": os.getpid(),
            "owner_nonce": nonce,
            "source_generation": int(source_generation),
            "started_at": time.time(),
        }
        try:
            with self.lock_path.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except FileExistsError as exc:
            raise RuntimeError("已有 RAG sidecar 正在构建") from exc
        self._owner_nonce = nonce
        candidate = CandidateRagIndex(
            generation=int(source_generation),
            path=self.root / f"rag.gen-{source_generation:06d}.{model.revision[:12]}.building.db",
            source_database=Path(source_database).resolve(),
            model=model,
            chunk_policy_version=str(chunk_policy_version),
            owner_nonce=nonce,
        )
        self._remove_sqlite_files(candidate.path)
        try:
            self._create_schema(candidate)
        except Exception:
            self._remove_sqlite_files(candidate.path)
            self.lock_path.unlink(missing_ok=True)
            self._owner_nonce = None
            raise
        return candidate

    @contextmanager
    def connect_candidate(self, candidate: CandidateRagIndex) -> Iterator[sqlite3.Connection]:
        self._assert_candidate_owner(candidate)
        connection = sqlite3.connect(candidate.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=NORMAL")
        try:
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def compatible_previous(self, model: ModelIdentity) -> tuple[sqlite3.Connection, PublishedRagIndex] | None:
        active = self.active()
        if active is None or active.model != model:
            return None
        return self._connect_readonly(active.path), active

    def publish(
        self,
        candidate: CandidateRagIndex,
        *,
        validated_report: RagBuildReport | None = None,
    ) -> PublishedRagIndex:
        self._assert_candidate_owner(candidate)
        report = validated_report or self.validate_candidate(candidate)
        connection = sqlite3.connect(candidate.path, timeout=30)
        try:
            connection.execute(
                "UPDATE rag_meta SET state='published', published_at=? WHERE id=1",
                (_utc_now(),),
            )
            connection.commit()
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        finally:
            connection.close()
        final_name = (
            f"rag.gen-{candidate.generation:06d}.{candidate.model.revision[:12]}"
            f".{candidate.owner_nonce[:12]}.db"
        )
        final_path = self.root / final_name
        os.replace(candidate.path, final_path)
        manifest = {
            "format_version": self.SCHEMA_VERSION,
            "source_generation": candidate.generation,
            "source_database": candidate.source_database.name,
            "active_file": final_name,
            "embedding_model_id": candidate.model.model_id,
            "embedding_model_revision": candidate.model.revision,
            "embedding_dimension": candidate.model.dimension,
            "embedding_max_tokens": candidate.model.max_tokens,
            "chunk_policy_version": candidate.chunk_policy_version,
            "fts_mode": report.fts_mode,
            "chunk_count": report.chunk_count,
            "published_at": _utc_now(),
        }
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        self.manifest_backup_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.lock_path.unlink(missing_ok=True)
        self._owner_nonce = None
        return PublishedRagIndex(
            generation=candidate.generation,
            path=final_path,
            model=candidate.model,
            chunk_policy_version=candidate.chunk_policy_version,
            fts_mode=report.fts_mode,
        )

    def validate_candidate(self, candidate: CandidateRagIndex) -> RagBuildReport:
        self._assert_candidate_owner(candidate)
        connection = sqlite3.connect(candidate.path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            quick = str(connection.execute("PRAGMA quick_check").fetchone()[0]).lower()
            if quick != "ok":
                raise sqlite3.DatabaseError(f"RAG sidecar quick_check failed: {quick}")
            foreign_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_errors:
                raise sqlite3.DatabaseError("RAG sidecar foreign_key_check failed")
            meta = connection.execute("SELECT * FROM rag_meta WHERE id=1").fetchone()
            if meta is None or int(meta["source_generation"]) != candidate.generation:
                raise sqlite3.DatabaseError("candidate generation mismatch")
            if (
                str(meta["embedding_model_id"]) != candidate.model.model_id
                or str(meta["embedding_model_revision"]) != candidate.model.revision
                or int(meta["embedding_dimension"]) != candidate.model.dimension
                or int(meta["embedding_max_tokens"]) != candidate.model.max_tokens
            ):
                raise sqlite3.DatabaseError("candidate model identity mismatch")
            chunks = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
            sources = int(connection.execute("SELECT COUNT(*) FROM rag_sources").fetchone()[0])
            embeddings = int(connection.execute("SELECT COUNT(*) FROM rag_embeddings").fetchone()[0])
            if chunks <= 0 or sources < chunks or embeddings != chunks:
                raise sqlite3.DatabaseError(
                    f"incomplete candidate: chunks={chunks}, sources={sources}, embeddings={embeddings}"
                )
            bad_blob = connection.execute(
                "SELECT 1 FROM rag_embeddings WHERE dimension != ? OR length(vector_blob) != ? LIMIT 1",
                (candidate.model.dimension, candidate.model.dimension * 4),
            ).fetchone()
            if bad_blob is not None:
                raise sqlite3.DatabaseError("candidate embedding blob has invalid dimension")
            fts_mode = str(meta["fts_mode"])
            if fts_mode == "trigram":
                sample = connection.execute(
                    "SELECT normalized_text FROM rag_chunks WHERE length(normalized_text) >= 3 LIMIT 1"
                ).fetchone()
                if sample is not None:
                    token = str(sample[0])[:3].replace('"', '""')
                    connection.execute(
                        "SELECT rowid FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ? LIMIT 1",
                        (f'"{token}"',),
                    ).fetchone()
            return RagBuildReport(
                generation=candidate.generation,
                chunk_count=chunks,
                source_count=sources,
                reused_embeddings=int(meta["reused_embeddings"]),
                generated_embeddings=int(meta["generated_embeddings"]),
                elapsed_ms=int(meta["elapsed_ms"]),
                fts_mode=fts_mode,
            )
        finally:
            connection.close()

    def abort(self, candidate: CandidateRagIndex, reason: str = "") -> None:
        if self._owner_nonce and candidate.owner_nonce != self._owner_nonce:
            raise RuntimeError("RAG candidate owner mismatch")
        self._remove_sqlite_files(candidate.path)
        self.lock_path.unlink(missing_ok=True)
        self._owner_nonce = None

    @contextmanager
    def acquire_compatible_previous(self, model: ModelIdentity) -> Iterator[tuple[sqlite3.Connection, PublishedRagIndex] | None]:
        active = self.active()
        if active is None or active.model != model:
            yield None
            return
        key = (active.path, active.model.revision)
        lease = self._leases.acquire(key)
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect_readonly(active.path)
            connection.execute("BEGIN")
            yield connection, active
        finally:
            if connection is not None:
                connection.close()
            lease.release()

    def cleanup_retired(self, keep: int = 2) -> list[str]:
        active = self.active()
        active_name = active.path.name if active else ""
        paths = sorted(
            self.root.glob("rag.gen-*.db"),
            key=lambda path: path.stat().st_mtime if path.exists() else 0,
            reverse=True,
        )
        removed: list[str] = []
        keep_names = {path.name for path in paths[: max(1, int(keep))]} | {active_name}
        for path in paths:
            if path.name in keep_names:
                continue
            if self._leases.active((path, self._model_revision_from_name(path.name))) > 0:
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                continue
        return removed

    def _create_schema(self, candidate: CandidateRagIndex) -> None:
        connection = sqlite3.connect(candidate.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        fts_mode = "trigram"
        connection.executescript(
            """
            CREATE TABLE rag_meta (
              id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL,
              source_generation INTEGER NOT NULL, source_database TEXT NOT NULL,
              state TEXT NOT NULL CHECK(state IN ('building','published')),
              fts_mode TEXT NOT NULL CHECK(fts_mode IN ('trigram','like')),
              chunk_policy_version TEXT NOT NULL, embedding_model_id TEXT NOT NULL,
              embedding_model_revision TEXT NOT NULL, embedding_dimension INTEGER NOT NULL,
              embedding_max_tokens INTEGER NOT NULL, created_at TEXT NOT NULL,
              published_at TEXT, reused_embeddings INTEGER NOT NULL DEFAULT 0,
              generated_embeddings INTEGER NOT NULL DEFAULT 0,
              elapsed_ms INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE rag_chunks (
              chunk_rowid INTEGER PRIMARY KEY, stable_chunk_id TEXT NOT NULL UNIQUE,
              text TEXT NOT NULL, normalized_text TEXT NOT NULL, text_hash TEXT NOT NULL,
              chunk_policy_version TEXT NOT NULL, source_type TEXT NOT NULL,
              ocr_confidence REAL
            );
            CREATE TABLE rag_sources (
              id INTEGER PRIMARY KEY,
              chunk_rowid INTEGER NOT NULL REFERENCES rag_chunks(chunk_rowid) ON DELETE CASCADE,
              root_id INTEGER NOT NULL, file_id INTEGER NOT NULL, document_id INTEGER NOT NULL,
              file_path TEXT NOT NULL, filename TEXT NOT NULL, extension TEXT NOT NULL,
              location_text TEXT NOT NULL, page_number INTEGER, line_start INTEGER,
              line_end INTEGER, source_type TEXT NOT NULL, ocr_confidence REAL,
              spans_json TEXT NOT NULL, UNIQUE(chunk_rowid, file_id)
            );
            CREATE INDEX idx_rag_sources_filter
              ON rag_sources(root_id, extension, file_id, chunk_rowid);
            CREATE INDEX idx_rag_sources_document ON rag_sources(document_id, chunk_rowid);
            CREATE TABLE rag_embeddings (
              chunk_rowid INTEGER PRIMARY KEY REFERENCES rag_chunks(chunk_rowid) ON DELETE CASCADE,
              vector_blob BLOB NOT NULL, dimension INTEGER NOT NULL, norm REAL NOT NULL CHECK(norm > 0)
            );
            CREATE TABLE rag_build_runs (
              run_id TEXT PRIMARY KEY, source_generation INTEGER NOT NULL, status TEXT NOT NULL,
              chunk_count INTEGER NOT NULL DEFAULT 0, reused_embeddings INTEGER NOT NULL DEFAULT 0,
              generated_embeddings INTEGER NOT NULL DEFAULT 0, elapsed_ms INTEGER NOT NULL DEFAULT 0,
              error_code TEXT, error_message TEXT
            );
            """
        )
        try:
            connection.execute(
                "CREATE VIRTUAL TABLE rag_chunks_fts USING fts5("
                "normalized_text, content='rag_chunks', content_rowid='chunk_rowid', tokenize='trigram')"
            )
        except sqlite3.OperationalError:
            fts_mode = "like"
        connection.execute(
            "INSERT INTO rag_meta(id, schema_version, source_generation, source_database, state, "
            "fts_mode, chunk_policy_version, embedding_model_id, embedding_model_revision, "
            "embedding_dimension, embedding_max_tokens, created_at) "
            "VALUES(1, ?, ?, ?, 'building', ?, ?, ?, ?, ?, ?, ?)",
            (
                self.SCHEMA_VERSION,
                candidate.generation,
                candidate.source_database.name,
                fts_mode,
                candidate.chunk_policy_version,
                candidate.model.model_id,
                candidate.model.revision,
                candidate.model.dimension,
                candidate.model.max_tokens,
                _utc_now(),
            ),
        )
        connection.commit()
        connection.close()

    def _assert_candidate_owner(self, candidate: CandidateRagIndex) -> None:
        if self._owner_nonce != candidate.owner_nonce:
            raise RuntimeError("RAG candidate owner mismatch")
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise RuntimeError("RAG build lock is missing or damaged") from exc
        if (
            str(payload.get("owner_nonce")) != candidate.owner_nonce
            or int(payload.get("pid") or 0) != os.getpid()
            or int(payload.get("source_generation") or 0) != candidate.generation
        ):
            raise RuntimeError("RAG candidate lock owner changed")

    def _recover_stale_build(self) -> None:
        if not self.lock_path.exists():
            return
        try:
            payload = json.loads(self.lock_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
            nonce = str(payload.get("owner_nonce") or "")
        except (OSError, ValueError, TypeError):
            raise RuntimeError("RAG build lock is damaged; inspect rag-index directory")
        try:
            import psutil

            alive = pid > 0 and psutil.pid_exists(pid)
        except ImportError:
            alive = True  # Unknown process state must not authorize deleting a candidate.
        if alive:
            raise RuntimeError("已有 RAG sidecar 正在构建")
        generation = int(payload.get("source_generation") or 0)
        pattern = f"rag.gen-{generation:06d}.*.building.db*"
        for path in self.root.glob(pattern):
            self._remove_sqlite_files(path)
        self.lock_path.unlink(missing_ok=True)

    def _read_manifest(self) -> dict[str, object] | None:
        for path in (self.manifest_path, self.manifest_backup_path):
            try:
                value = json.loads(path.read_text(encoding="utf-8"))
                if int(value.get("format_version") or 0) == self.SCHEMA_VERSION:
                    return value
            except (OSError, ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _connect_readonly(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{Path(path).resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=3,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        connection.execute("PRAGMA busy_timeout=3000")
        return connection

    @staticmethod
    def _remove_sqlite_files(path: Path) -> None:
        for target in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm"), Path(str(path) + "-journal")):
            target.unlink(missing_ok=True)

    @staticmethod
    def _model_revision_from_name(name: str) -> str:
        try:
            return name.split(".")[2]
        except IndexError:
            return ""


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()
