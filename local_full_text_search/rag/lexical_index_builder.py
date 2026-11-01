from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagBundle, RagIndexStore
from local_full_text_search.rag.lexical_index_store import (
    LEXICAL_SCHEMA_VERSION, LexicalIndexBundle, LexicalIndexStore,
)
from local_full_text_search.rag.retrieval.lexical_analyzer import LexicalAnalyzer, get_lexical_analyzer


class LexicalIndexBuilder:
    def __init__(self, rag_store: RagIndexStore, analyzer: LexicalAnalyzer | None = None) -> None:
        self.rag_store = rag_store
        self.analyzer = analyzer or get_lexical_analyzer()
        self.store = LexicalIndexStore(rag_store.root)

    def prepare(self, bundle: RagBundle, token: CancelToken) -> tuple[LexicalIndexBundle, bool]:
        token.throw_if_cancelled()
        source_digest = self._source_digest(bundle, token)
        fingerprint = hashlib.sha256(json.dumps({
            "schema": LEXICAL_SCHEMA_VERSION,
            "source_bundle_id": bundle.bundle_id,
            "source_content_digest": source_digest,
            "analyzer_fingerprint": self.analyzer.fingerprint,
        }, sort_keys=True).encode("utf-8")).hexdigest()
        try:
            active = self.store.active(bundle.bundle_id, self.analyzer.fingerprint)
            if active.source_content_digest == source_digest and active.index_fingerprint == fingerprint:
                return active, True
        except Exception:
            pass

        parent = self.store.bundle_root(bundle.bundle_id)
        parent.mkdir(parents=True, exist_ok=True)
        final = parent / fingerprint[:20]
        if final.is_dir():
            shutil.rmtree(final)
        temporary = parent / f".tmp-{uuid.uuid4().hex}"
        temporary.mkdir()
        database = temporary / "lexical.db"
        connection = sqlite3.connect(database)
        connection.row_factory = sqlite3.Row
        created_at = datetime.now(timezone.utc).isoformat()
        try:
            self._create_schema(connection)
            connection.execute(
                "INSERT INTO lexical_meta VALUES(1,?,?,?,?,?,?,?,0)",
                (LEXICAL_SCHEMA_VERSION, "building", bundle.bundle_id, source_digest,
                 self.analyzer.fingerprint, fingerprint, created_at),
            )
            self._populate(connection, bundle, token)
            chunk_count = int(connection.execute("SELECT COUNT(*) FROM lexical_documents").fetchone()[0])
            source_count = self._source_chunk_count(bundle)
            if chunk_count != source_count or chunk_count <= 0:
                raise ValueError(f"关键词索引正文数不一致：lexical={chunk_count}, source={source_count}")
            connection.execute(
                "UPDATE lexical_terms SET document_frequency=(SELECT COUNT(DISTINCT document_id) "
                "FROM lexical_occurrences o WHERE o.term_id=lexical_terms.term_id)"
            )
            connection.execute(
                "UPDATE lexical_meta SET state='published',chunk_count=? WHERE id=1", (chunk_count,)
            )
            connection.commit()
            quick = connection.execute("PRAGMA quick_check").fetchone()[0]
            if str(quick).lower() != "ok":
                raise ValueError(f"关键词索引完整性检查失败：{quick}")
            connection.close()
            os.replace(temporary, final)
            published = LexicalIndexBundle(
                bundle.bundle_id, source_digest, self.analyzer.fingerprint, fingerprint,
                final, final / "lexical.db", chunk_count, created_at,
            )
            self.store.publish_pointer(published)
            return published, False
        except Exception:
            try:
                connection.close()
            except sqlite3.Error:
                pass
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _source_digest(self, bundle: RagBundle, token: CancelToken) -> str:
        digest = hashlib.sha256()
        connection = self.rag_store.connect_read_only(bundle.metadata_path)
        try:
            rows = connection.execute(
                "SELECT stable_chunk_id,text_hash,source_json FROM rag_chunks ORDER BY stable_chunk_id"
            )
            for index, row in enumerate(rows, 1):
                if index % 256 == 0:
                    token.throw_if_cancelled()
                source = json.loads(row["source_json"])
                line = [str(row["stable_chunk_id"]), str(row["text_hash"]), str(source.get("filename") or "")]
                digest.update(json.dumps(line, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
                digest.update(b"\n")
            return digest.hexdigest()
        finally:
            connection.close()

    def _source_chunk_count(self, bundle: RagBundle) -> int:
        connection = self.rag_store.connect_read_only(bundle.metadata_path)
        try:
            return int(connection.execute("SELECT COUNT(*) FROM rag_chunks").fetchone()[0])
        finally:
            connection.close()

    @staticmethod
    def _create_schema(connection: sqlite3.Connection) -> None:
        connection.executescript("""
            PRAGMA foreign_keys=ON;
            PRAGMA journal_mode=DELETE;
            CREATE TABLE lexical_meta(
                id INTEGER PRIMARY KEY CHECK(id=1), schema_version INTEGER NOT NULL,
                state TEXT NOT NULL, source_bundle_id TEXT NOT NULL,
                source_content_digest TEXT NOT NULL, analyzer_fingerprint TEXT NOT NULL,
                index_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            );
            CREATE TABLE lexical_documents(
                document_id INTEGER PRIMARY KEY, stable_chunk_id TEXT NOT NULL UNIQUE,
                body_length INTEGER NOT NULL,
                title_length INTEGER NOT NULL, root_id INTEGER NOT NULL,
                extension TEXT NOT NULL, source_type TEXT NOT NULL, ocr_confidence REAL
            );
            CREATE TABLE lexical_terms(
                term_id INTEGER PRIMARY KEY, term_kind TEXT NOT NULL, term_text TEXT NOT NULL,
                document_frequency INTEGER NOT NULL DEFAULT 0,
                UNIQUE(term_kind,term_text)
            );
            CREATE TABLE lexical_occurrences(
                term_id INTEGER NOT NULL, document_id INTEGER NOT NULL, field TEXT NOT NULL,
                term_frequency INTEGER NOT NULL,
                PRIMARY KEY(term_id,document_id,field),
                FOREIGN KEY(term_id) REFERENCES lexical_terms(term_id),
                FOREIGN KEY(document_id) REFERENCES lexical_documents(document_id)
            ) WITHOUT ROWID;
        """)

    def _populate(self, target: sqlite3.Connection, bundle: RagBundle, token: CancelToken) -> None:
        source = self.rag_store.connect_read_only(bundle.metadata_path)
        term_ids: dict[tuple[str, str], int] = {}
        try:
            rows = source.execute("SELECT stable_chunk_id,text,source_json FROM rag_chunks ORDER BY stable_chunk_id")
            for index, row in enumerate(rows, 1):
                if index % 64 == 0:
                    token.throw_if_cancelled()
                    target.commit()
                metadata = json.loads(row["source_json"])
                chunk_id = str(row["stable_chunk_id"])
                body = str(row["text"])
                title = Path(str(metadata.get("filename") or "")).stem
                body_tokens = self.analyzer.analyze_text(body)
                title_tokens = self.analyzer.analyze_text(title)
                cursor = target.execute(
                    "INSERT INTO lexical_documents(stable_chunk_id,body_length,title_length,root_id,extension,source_type,ocr_confidence) "
                    "VALUES(?,?,?,?,?,?,?)",
                    (chunk_id, len(body_tokens), len(title_tokens), int(metadata.get("root_id") or 0),
                     str(metadata.get("extension") or "").lower(), str(metadata.get("source_type") or ""),
                     metadata.get("ocr_confidence")),
                )
                document_id = int(cursor.lastrowid)
                for field, text, tokens in (("body", body, body_tokens), ("title", title, title_tokens)):
                    token_counts = Counter(item.normalized for item in tokens)
                    for term_text, frequency in token_counts.items():
                        term_id = self._term_id(target, term_ids, "token", term_text)
                        target.execute(
                            "INSERT INTO lexical_occurrences VALUES(?,?,?,?)",
                            (term_id, document_id, field, int(frequency)),
                        )
        finally:
            source.close()

    @staticmethod
    def _term_id(
        connection: sqlite3.Connection, cache: dict[tuple[str, str], int], kind: str, text: str,
    ) -> int:
        key = (kind, text)
        existing = cache.get(key)
        if existing is not None:
            return existing
        connection.execute(
            "INSERT OR IGNORE INTO lexical_terms(term_kind,term_text) VALUES(?,?)", key,
        )
        term_id = int(connection.execute(
            "SELECT term_id FROM lexical_terms WHERE term_kind=? AND term_text=?", key,
        ).fetchone()[0])
        cache[key] = term_id
        return term_id
