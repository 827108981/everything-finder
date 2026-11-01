from __future__ import annotations

import json
import re
import os
import sqlite3
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.rag.errors import RagIndexCorruptError, RagIndexMissingError
from local_full_text_search.rag.models import EvidenceChunk, QueryPlan, RagChunk, SourceRef
from local_full_text_search.rag.query_analyzer import QueryAnalysis, QueryAnalyzer


RAG_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RagBundle:
    generation: int
    source_generation: int
    bundle_id: str
    path: Path
    metadata_path: Path
    chroma_path: Path
    chunk_count: int
    vector_count: int
    created_at: str
    embedding_repo_id: str = ""
    embedding_revision: str = ""
    embedding_dimension: int = 0
    verified_source_generation: int = 0
    build_signature: str = ""
    total_chunk_count: int = 0
    parent_vector_mode: str = "chroma"
    vector_unit_directory: str = ""
    vector_unit_index_fingerprint: str = ""
    vector_unit_policy_fingerprint: str = ""
    vector_unit_count: int = 0


class RagIndexStore:
    """Stores immutable SQLite+Chroma bundles selected by an atomic manifest."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "rag.manifest.json"

    def create_candidate(
        self,
        source_generation: int,
        *,
        embedding_repo_id: str,
        embedding_revision: str,
        embedding_dimension: int,
        reranker_repo_id: str,
        reranker_revision: str,
        hnsw_m: int = 16,
        construction_ef: int = 200,
        search_ef: int = 100,
        build_signature: str = "",
        total_chunk_count: int = 0,
        parent_vector_mode: str = "chroma",
    ) -> tuple[RagBundle, sqlite3.Connection]:
        self.root.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(timezone.utc).isoformat()
        bundle_id = f"g{int(source_generation):06d}-{uuid.uuid4().hex[:10]}"
        path = self.root / f"rag.{bundle_id}"
        path.mkdir(parents=False, exist_ok=False)
        metadata_path = path / "metadata.db"
        chroma_path = path / "chroma"
        chroma_path.mkdir()
        connection = sqlite3.connect(metadata_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("PRAGMA synchronous = FULL")
        connection.executescript(
            """
            CREATE TABLE rag_meta (
                id INTEGER PRIMARY KEY CHECK(id=1),
                schema_version INTEGER NOT NULL,
                state TEXT NOT NULL,
                source_generation INTEGER NOT NULL,
                bundle_id TEXT NOT NULL,
                embedding_repo_id TEXT NOT NULL,
                embedding_revision TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                vector_backend TEXT NOT NULL,
                vector_metric TEXT NOT NULL,
                hnsw_m INTEGER NOT NULL,
                hnsw_construction_ef INTEGER NOT NULL,
                hnsw_search_ef INTEGER NOT NULL,
                reranker_repo_id TEXT NOT NULL,
                reranker_revision TEXT NOT NULL,
                build_signature TEXT NOT NULL DEFAULT '',
                total_chunk_count INTEGER NOT NULL DEFAULT 0,
                chunk_count INTEGER NOT NULL DEFAULT 0,
                vector_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                published_at TEXT
            );
            CREATE TABLE rag_chunks (
                stable_chunk_id TEXT PRIMARY KEY,
                document_key TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                text TEXT NOT NULL,
                normalized_text TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                source_json TEXT NOT NULL,
                source_type TEXT NOT NULL,
                ocr_confidence REAL
            );
            CREATE VIRTUAL TABLE rag_chunks_fts USING fts5(
                stable_chunk_id UNINDEXED,
                normalized_text,
                filename,
                file_path,
                location_text,
                tokenize='trigram'
            );
            CREATE TABLE rag_vector_refs (
                stable_chunk_id TEXT PRIMARY KEY,
                point_id TEXT NOT NULL UNIQUE,
                FOREIGN KEY(stable_chunk_id) REFERENCES rag_chunks(stable_chunk_id)
                    ON DELETE CASCADE
            );
            CREATE INDEX idx_rag_chunks_document ON rag_chunks(document_key, ordinal);
            CREATE INDEX idx_rag_chunks_source ON rag_chunks(source_type, ocr_confidence);
            """
        )
        connection.execute(
            """INSERT INTO rag_meta(
                id,schema_version,state,source_generation,bundle_id,
                embedding_repo_id,embedding_revision,embedding_dimension,
                vector_backend,vector_metric,hnsw_m,hnsw_construction_ef,
                hnsw_search_ef,reranker_repo_id,reranker_revision,
                build_signature,total_chunk_count,created_at
            ) VALUES(1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                RAG_SCHEMA_VERSION,
                "building",
                int(source_generation),
                bundle_id,
                embedding_repo_id,
                embedding_revision,
                int(embedding_dimension),
                str(parent_vector_mode),
                "cosine",
                int(hnsw_m),
                int(construction_ef),
                int(search_ef),
                reranker_repo_id,
                reranker_revision,
                str(build_signature),
                max(0, int(total_chunk_count)),
                created_at,
            ),
        )
        connection.commit()
        return (
            RagBundle(
                generation=int(source_generation),
                source_generation=int(source_generation),
                bundle_id=bundle_id,
                path=path,
                metadata_path=metadata_path,
                chroma_path=chroma_path,
                chunk_count=0,
                vector_count=0,
                created_at=created_at,
                embedding_repo_id=embedding_repo_id,
                embedding_revision=embedding_revision,
                embedding_dimension=int(embedding_dimension),
                build_signature=str(build_signature),
                total_chunk_count=max(0, int(total_chunk_count)),
                parent_vector_mode=str(parent_vector_mode),
            ),
            connection,
        )

    def find_resumable_candidate(
        self,
        source_generation: int,
        *,
        embedding_repo_id: str,
        embedding_revision: str,
        embedding_dimension: int,
        build_signature: str,
    ) -> tuple[RagBundle, sqlite3.Connection] | None:
        """返回与本次构建完全相同的未完成候选。

        只读取 ``state=building`` 的候选，正式索引仍完全由 manifest 决定。
        缺文件、旧结构或参数不一致的目录不会被复用。
        """
        if not self.root.is_dir() or not build_signature:
            return None
        candidates = sorted(
            (path for path in self.root.glob(f"rag.g{int(source_generation):06d}-*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for path in candidates:
            metadata_path = path / "metadata.db"
            chroma_path = path / "chroma"
            if not metadata_path.is_file() or not chroma_path.is_dir():
                continue
            connection: sqlite3.Connection | None = None
            try:
                connection = sqlite3.connect(metadata_path, timeout=30)
                connection.row_factory = sqlite3.Row
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("PRAGMA journal_mode = DELETE")
                connection.execute("PRAGMA synchronous = FULL")
                columns = {
                    str(row[1]) for row in connection.execute("PRAGMA table_info(rag_meta)")
                }
                if not {"build_signature", "total_chunk_count"}.issubset(columns):
                    connection.close()
                    continue
                meta = connection.execute("SELECT * FROM rag_meta WHERE id=1").fetchone()
                if (
                    meta is None
                    or str(meta["state"]) != "building"
                    or int(meta["schema_version"]) != RAG_SCHEMA_VERSION
                    or int(meta["source_generation"]) != int(source_generation)
                    or str(meta["embedding_repo_id"]) != str(embedding_repo_id)
                    or str(meta["embedding_revision"]) != str(embedding_revision)
                    or int(meta["embedding_dimension"]) != int(embedding_dimension)
                    or str(meta["build_signature"]) != str(build_signature)
                ):
                    connection.close()
                    continue
                chunk_count = int(
                    connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0]
                )
                ref_count = int(
                    connection.execute("SELECT COUNT(*) FROM rag_vector_refs").fetchone()[0]
                )
                parent_vector_mode = str(meta["vector_backend"])
                if (
                    parent_vector_mode == "metadata_only" and ref_count != 0
                ) or (
                    parent_vector_mode != "metadata_only" and chunk_count != ref_count
                ):
                    connection.close()
                    continue
                return (
                    RagBundle(
                        generation=int(source_generation),
                        source_generation=int(source_generation),
                        bundle_id=str(meta["bundle_id"]),
                        path=path,
                        metadata_path=metadata_path,
                        chroma_path=chroma_path,
                        chunk_count=chunk_count,
                        vector_count=ref_count,
                        created_at=str(meta["created_at"]),
                        embedding_repo_id=str(meta["embedding_repo_id"]),
                        embedding_revision=str(meta["embedding_revision"]),
                        embedding_dimension=int(meta["embedding_dimension"]),
                        build_signature=str(meta["build_signature"]),
                        total_chunk_count=int(meta["total_chunk_count"]),
                        parent_vector_mode=parent_vector_mode,
                    ),
                    connection,
                )
            except (OSError, sqlite3.Error, KeyError, TypeError, ValueError):
                if connection is not None:
                    try:
                        connection.close()
                    except sqlite3.Error:
                        pass
        return None

    def add_chunk(self, connection: sqlite3.Connection, chunk: RagChunk) -> None:
        source_json = json.dumps(
            {
                "source_generation": chunk.source.source_generation,
                "stable_chunk_id": chunk.source.stable_chunk_id,
                "document_id": chunk.source.document_id,
                "file_id": chunk.source.file_id,
                "root_id": chunk.source.root_id,
                "file_path": chunk.source.file_path,
                "filename": chunk.source.filename,
                "extension": chunk.source.extension,
                "block_ids": list(chunk.source.block_ids),
                "location_text": chunk.source.location_text,
                "page_number": chunk.source.page_number,
                "slide_number": chunk.source.slide_number,
                "sheet_name": chunk.source.sheet_name,
                "cell_start": chunk.source.cell_start,
                "cell_end": chunk.source.cell_end,
                "line_start": chunk.source.line_start,
                "line_end": chunk.source.line_end,
                "source_type": chunk.source.source_type,
                "ocr_confidence": chunk.source.ocr_confidence,
            },
            ensure_ascii=False,
        )
        connection.execute(
            """INSERT INTO rag_chunks(
                stable_chunk_id,document_key,ordinal,text,normalized_text,text_hash,
                source_json,source_type,ocr_confidence
            ) VALUES(?,?,?,?,?,?,?,?,?)""",
            (
                chunk.stable_chunk_id,
                chunk.document_key,
                chunk.ordinal,
                chunk.text,
                chunk.normalized_text,
                chunk.text_hash,
                source_json,
                chunk.source.source_type,
                chunk.source.ocr_confidence,
            ),
        )
        connection.execute(
            """INSERT INTO rag_chunks_fts(
                stable_chunk_id,normalized_text,filename,file_path,location_text
            ) VALUES(?,?,?,?,?)""",
            (
                chunk.stable_chunk_id,
                chunk.normalized_text,
                chunk.source.filename,
                chunk.source.file_path,
                chunk.source.location_text,
            ),
        )

    @staticmethod
    def add_vector_ref(connection: sqlite3.Connection, stable_chunk_id: str) -> None:
        connection.execute(
            "INSERT INTO rag_vector_refs(stable_chunk_id,point_id) VALUES(?,?)",
            (stable_chunk_id, stable_chunk_id),
        )

    def publish(
        self,
        bundle: RagBundle,
        connection: sqlite3.Connection,
        *,
        vector_count: int,
        attached_vector_index=None,
    ) -> RagBundle:
        quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
        if str(quick_check).lower() != "ok":
            raise RagIndexCorruptError(f"RAG SQLite 完整性检查失败：{quick_check}")
        chunk_count = int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
        ref_count = int(connection.execute("SELECT COUNT(*) FROM rag_vector_refs").fetchone()[0])
        if bundle.parent_vector_mode == "metadata_only":
            if ref_count != 0 or int(vector_count) != 0:
                raise RagIndexCorruptError(
                    "仅元数据父索引不应包含父块向量引用："
                    f"refs={ref_count}, vectors={vector_count}"
                )
            if attached_vector_index is None:
                raise RagIndexCorruptError("细粒度向量索引尚未完成，不能发布父索引")
            if (
                str(attached_vector_index.source_bundle_id) != bundle.bundle_id
                or int(attached_vector_index.parent_chunk_count) != chunk_count
                or int(attached_vector_index.unit_count) <= 0
            ):
                raise RagIndexCorruptError("细粒度向量索引与父索引候选不一致")
        elif chunk_count != ref_count or chunk_count != int(vector_count):
            raise RagIndexCorruptError(
                f"chunk/vector 数量不一致：chunks={chunk_count}, refs={ref_count}, vectors={vector_count}"
            )
        if chunk_count <= 0:
            raise RagIndexCorruptError("没有可发布的 RAG 正文块")
        published_at = datetime.now(timezone.utc).isoformat()
        connection.execute(
            "UPDATE rag_meta SET state='published',chunk_count=?,vector_count=?,published_at=? WHERE id=1",
            (chunk_count, int(vector_count), published_at),
        )
        connection.commit()
        connection.close()
        payload = {
            "schema_version": RAG_SCHEMA_VERSION,
            "bundle_id": bundle.bundle_id,
            "directory": bundle.path.name,
            "source_generation": bundle.source_generation,
            "chunk_count": chunk_count,
            "vector_count": int(vector_count),
            "created_at": bundle.created_at,
            "published_at": published_at,
            "parent_vector_mode": bundle.parent_vector_mode,
        }
        if attached_vector_index is not None:
            payload["vector_unit_index"] = {
                "directory": str(attached_vector_index.path.name),
                "source_content_digest": str(attached_vector_index.source_content_digest),
                "embedding_repo_id": str(attached_vector_index.embedding_repo_id),
                "embedding_revision": str(attached_vector_index.embedding_revision),
                "embedding_dimension": int(attached_vector_index.embedding_dimension),
                "policy_fingerprint": str(attached_vector_index.policy_fingerprint),
                "index_fingerprint": str(attached_vector_index.index_fingerprint),
                "parent_chunk_count": int(attached_vector_index.parent_chunk_count),
                "unit_count": int(attached_vector_index.unit_count),
                "created_at": str(attached_vector_index.created_at),
            }
        payload["verified_source_generation"] = bundle.source_generation
        self._replace_manifest(payload)
        return RagBundle(
            generation=bundle.source_generation,
            source_generation=bundle.source_generation,
            bundle_id=bundle.bundle_id,
            path=bundle.path,
            metadata_path=bundle.metadata_path,
            chroma_path=bundle.chroma_path,
            chunk_count=chunk_count,
            vector_count=int(vector_count),
            created_at=bundle.created_at,
            embedding_repo_id=bundle.embedding_repo_id,
            embedding_revision=bundle.embedding_revision,
            embedding_dimension=bundle.embedding_dimension,
            verified_source_generation=bundle.source_generation,
            build_signature=bundle.build_signature,
            total_chunk_count=bundle.total_chunk_count,
            parent_vector_mode=bundle.parent_vector_mode,
            vector_unit_directory=(
                str(attached_vector_index.path.name) if attached_vector_index is not None else ""
            ),
            vector_unit_index_fingerprint=(
                str(attached_vector_index.index_fingerprint)
                if attached_vector_index is not None else ""
            ),
            vector_unit_policy_fingerprint=(
                str(attached_vector_index.policy_fingerprint)
                if attached_vector_index is not None else ""
            ),
            vector_unit_count=(
                int(attached_vector_index.unit_count) if attached_vector_index is not None else 0
            ),
        )

    def mark_source_compatible(self, bundle: RagBundle, source_generation: int) -> RagBundle:
        """Record that a newer full-text snapshot has identical RAG content and citations."""
        current = self.active()
        if current.bundle_id != bundle.bundle_id:
            raise RagIndexCorruptError("AI 索引在校验期间已切换，请重新检查")
        if current.verified_source_generation == int(source_generation):
            return current
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        payload["verified_source_generation"] = int(source_generation)
        self._replace_manifest(payload)
        return replace(bundle, verified_source_generation=int(source_generation))

    def _replace_manifest(self, payload: dict[str, object]) -> None:
        temporary = self.manifest_path.with_suffix(".json.tmp")
        with temporary.open("wb") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)

    def abort(self, bundle: RagBundle, connection: sqlite3.Connection | None = None) -> None:
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                pass
        candidate = Path(bundle.path).resolve()
        if candidate.parent == self.root and candidate.name.startswith("rag.g"):
            import shutil

            shutil.rmtree(candidate, ignore_errors=True)

    def active(self) -> RagBundle:
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if int(manifest.get("schema_version", -1)) != RAG_SCHEMA_VERSION:
                raise ValueError("unsupported manifest schema version")
            directory = str(manifest["directory"])
            path = (self.root / directory).resolve()
            if path.parent != self.root or not path.is_dir():
                raise ValueError("invalid bundle path")
            metadata_path = path / "metadata.db"
            chroma_path = path / "chroma"
            if not metadata_path.is_file() or not chroma_path.is_dir():
                raise ValueError("bundle files are incomplete")
            connection = self.connect_read_only(metadata_path)
            try:
                meta = connection.execute("SELECT * FROM rag_meta WHERE id=1").fetchone()
                if meta is None or meta["state"] != "published":
                    raise ValueError("bundle is not published")
                if int(meta["schema_version"]) != RAG_SCHEMA_VERSION:
                    raise ValueError("unsupported bundle schema version")
                if str(meta["bundle_id"]) != str(manifest["bundle_id"]):
                    raise ValueError("manifest and database bundle IDs differ")
                if int(meta["source_generation"]) != int(manifest["source_generation"]):
                    raise ValueError("manifest and database generation differ")
                if int(meta["chunk_count"]) != int(manifest["chunk_count"]):
                    raise ValueError("manifest and database counts differ")
                if int(meta["vector_count"]) != int(manifest["vector_count"]):
                    raise ValueError("manifest and database vector counts differ")
                embedding_repo_id = str(meta["embedding_repo_id"])
                embedding_revision = str(meta["embedding_revision"])
                embedding_dimension = int(meta["embedding_dimension"])
                parent_vector_mode = str(meta["vector_backend"])
            finally:
                connection.close()
            vector_unit = manifest.get("vector_unit_index")
            if not isinstance(vector_unit, dict):
                vector_unit = {}
            return RagBundle(
                generation=int(manifest["source_generation"]),
                source_generation=int(manifest["source_generation"]),
                bundle_id=str(manifest["bundle_id"]),
                path=path,
                metadata_path=metadata_path,
                chroma_path=chroma_path,
                chunk_count=int(manifest["chunk_count"]),
                vector_count=int(manifest["vector_count"]),
                created_at=str(manifest.get("created_at") or ""),
                embedding_repo_id=embedding_repo_id,
                embedding_revision=embedding_revision,
                embedding_dimension=embedding_dimension,
                verified_source_generation=int(
                    manifest.get("verified_source_generation", manifest["source_generation"])
                ),
                parent_vector_mode=str(
                    manifest.get("parent_vector_mode") or parent_vector_mode or "chroma"
                ),
                vector_unit_directory=str(vector_unit.get("directory") or ""),
                vector_unit_index_fingerprint=str(
                    vector_unit.get("index_fingerprint") or ""
                ),
                vector_unit_policy_fingerprint=str(
                    vector_unit.get("policy_fingerprint") or ""
                ),
                vector_unit_count=int(vector_unit.get("unit_count") or 0),
            )
        except FileNotFoundError as exc:
            raise RagIndexMissingError("尚未构建 AI 索引") from exc
        except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
            raise RagIndexCorruptError(f"RAG 索引清单或数据库损坏：{exc}") from exc

    @staticmethod
    def connect_read_only(path: Path) -> sqlite3.Connection:
        connection = sqlite3.connect(
            f"file:{Path(path).resolve().as_posix()}?mode=ro",
            uri=True,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def load_chunk(self, connection: sqlite3.Connection, stable_chunk_id: str) -> RagChunk | None:
        row = connection.execute(
            "SELECT * FROM rag_chunks WHERE stable_chunk_id=?", (stable_chunk_id,)
        ).fetchone()
        if row is None:
            return None
        source_data = json.loads(row["source_json"])
        source_data["block_ids"] = tuple(source_data.get("block_ids") or ())
        source = SourceRef(**source_data)
        return RagChunk(
            stable_chunk_id=str(row["stable_chunk_id"]),
            document_key=str(row["document_key"]),
            ordinal=int(row["ordinal"]),
            text=str(row["text"]),
            normalized_text=str(row["normalized_text"]),
            text_hash=str(row["text_hash"]),
            source=source,
        )

    def lexical_search(
        self,
        connection: sqlite3.Connection,
        query: str,
        *,
        top_k: int,
        root_ids: tuple[int, ...] = (),
        extensions: tuple[str, ...] = (),
        include_ocr: bool = True,
        min_ocr_confidence: float = 0.60,
        analysis: QueryAnalysis | QueryPlan | None = None,
        diagnostics: dict[str, object] | None = None,
    ) -> list[EvidenceChunk]:
        text = str(query or "").strip()
        if not text:
            return []
        analysis = analysis or QueryAnalyzer().analyze(text)
        short_identifier = None
        if not analysis.fts_expression and len(analysis.fallback_terms) == 1:
            possible = str(analysis.fallback_terms[0])
            if len(possible) < 3 and any(char.isalpha() for char in possible) and any(
                char.isdigit() for char in possible
            ):
                short_identifier = possible
        if analysis.fts_expression:
            sql = """
                SELECT stable_chunk_id, bm25(rag_chunks_fts) AS rank
                FROM rag_chunks_fts WHERE rag_chunks_fts MATCH ?
                ORDER BY rank, stable_chunk_id LIMIT ? OFFSET ?
            """
            parameters: list[object] = [analysis.fts_expression]
            scan_limit = max(1, int(top_k) * 4)
        else:
            like_terms = analysis.fallback_terms or (analysis.normalized_text,)
            clauses = []
            parameters: list[object] = []
            for term in like_terms:
                escaped = term.replace("%", "\\%").replace("_", "\\_")
                clauses.append("(normalized_text LIKE ? ESCAPE '\\' OR source_json LIKE ? ESCAPE '\\')" if short_identifier else "normalized_text LIKE ? ESCAPE '\\'")
                parameters.append(f"%{escaped}%")
                if short_identifier:
                    parameters.append(f"%{escaped}%")
            sql = (
                "SELECT stable_chunk_id, 0.0 AS rank FROM rag_chunks WHERE "
                + " OR ".join(clauses)
                + " ORDER BY stable_chunk_id LIMIT ? OFFSET ?"
            )
            # 短编号的 LIKE 会先找到大量 J80/AJ8 等伪命中，必须分页复核边界。
            scan_limit = max(10_000, int(top_k) * 200) if short_identifier else max(1, int(top_k) * 4)
        results: list[EvidenceChunk] = []
        scanned = 0
        exhausted = False
        page_size = min(200, scan_limit)
        identifier_pattern = (re.compile(
            rf"(?<![A-Za-z0-9]){re.escape(short_identifier)}(?![A-Za-z0-9])", re.I
        ) if short_identifier else None)
        while scanned < scan_limit and len(results) < max(1, int(top_k)):
            rows = connection.execute(
                sql, (*parameters, min(page_size, scan_limit - scanned), scanned)
            ).fetchall()
            if not rows:
                exhausted = True
                break
            scanned += len(rows)
            for row in rows:
                chunk = self.load_chunk(connection, str(row["stable_chunk_id"]))
                if chunk is None or not self._matches_filters(
                    chunk, root_ids, extensions, include_ocr, min_ocr_confidence
                ):
                    continue
                if identifier_pattern and not identifier_pattern.search(chunk.text + " " + chunk.source.filename):
                    continue
                results.append(
                    EvidenceChunk(
                        chunk=chunk,
                        lexical_rank=len(results) + 1,
                        lexical_score=float(row["rank"]),
                    )
                )
                if len(results) >= max(1, int(top_k)):
                    break
            if len(rows) < page_size:
                exhausted = True
                break
        if diagnostics is not None:
            diagnostics.update({
                "scanned_count": scanned,
                "scan_limit": scan_limit,
                "candidate_pool_exhausted": not exhausted and scanned >= scan_limit and len(results) < max(1, int(top_k)),
            })
        return results

    @staticmethod
    def _matches_filters(
        chunk: RagChunk,
        root_ids: tuple[int, ...],
        extensions: tuple[str, ...],
        include_ocr: bool,
        min_ocr_confidence: float,
    ) -> bool:
        source = chunk.source
        if root_ids and source.root_id not in root_ids:
            return False
        if extensions and source.extension.lower() not in {item.lower() for item in extensions}:
            return False
        if source.source_type == "ocr":
            if not include_ocr:
                return False
            if source.ocr_confidence is not None and source.ocr_confidence < min_ocr_confidence:
                return False
        return True
