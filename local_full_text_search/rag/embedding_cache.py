from __future__ import annotations

import hashlib
import math
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


class EmbeddingVectorCache:
    """按固定模型版本和精确正文缓存向量，不保存资料正文。"""

    def __init__(
        self,
        path: Path,
        *,
        model_id: str,
        model_revision: str,
        dimension: int,
    ) -> None:
        self.path = Path(path).resolve()
        self.model_id = str(model_id)
        self.model_revision = str(model_revision)
        self.dimension = int(dimension)
        if self.dimension <= 0:
            raise ValueError("向量缓存维度必须为正数")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=30)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=NORMAL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS embedding_cache(
                model_id TEXT NOT NULL,
                model_revision TEXT NOT NULL,
                dimension INTEGER NOT NULL,
                text_sha256 TEXT NOT NULL,
                vector_blob BLOB NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY(model_id,model_revision,dimension,text_sha256)
            ) WITHOUT ROWID
            """
        )
        self.connection.commit()

    @staticmethod
    def text_hash(text: str) -> str:
        return hashlib.sha256(str(text).encode("utf-8")).hexdigest()

    def get_many(self, texts: Sequence[str]) -> dict[str, list[float]]:
        by_hash: dict[str, str] = {}
        for text in texts:
            value = str(text)
            digest = self.text_hash(value)
            previous = by_hash.get(digest)
            if previous is not None and previous != value:
                raise ValueError("正文 SHA-256 冲突，拒绝复用向量")
            by_hash[digest] = value
        if not by_hash:
            return {}
        output: dict[str, list[float]] = {}
        hashes = list(by_hash)
        for offset in range(0, len(hashes), 400):
            batch = hashes[offset : offset + 400]
            marks = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""SELECT text_sha256,vector_blob FROM embedding_cache
                    WHERE model_id=? AND model_revision=? AND dimension=?
                      AND text_sha256 IN ({marks})""",
                (self.model_id, self.model_revision, self.dimension, *batch),
            ).fetchall()
            for digest, blob in rows:
                raw = bytes(blob)
                if len(raw) != self.dimension * 4:
                    continue
                vector = list(struct.unpack(f"<{self.dimension}f", raw))
                if not all(math.isfinite(item) for item in vector):
                    continue
                text = by_hash.get(str(digest))
                if text is not None:
                    output[text] = vector
        return output

    def put_many(self, vectors_by_text: Mapping[str, Sequence[float]]) -> None:
        if not vectors_by_text:
            return
        created_at = datetime.now(timezone.utc).isoformat()
        rows = []
        for text, vector in vectors_by_text.items():
            values = tuple(float(item) for item in vector)
            if len(values) != self.dimension or not all(
                math.isfinite(item) for item in values
            ):
                raise ValueError("拒绝写入维度错误或包含非有限值的缓存向量")
            rows.append((
                self.model_id,
                self.model_revision,
                self.dimension,
                self.text_hash(str(text)),
                sqlite3.Binary(struct.pack(f"<{self.dimension}f", *values)),
                created_at,
            ))
        self.connection.executemany(
            """INSERT OR REPLACE INTO embedding_cache(
                model_id,model_revision,dimension,text_sha256,vector_blob,created_at
            ) VALUES(?,?,?,?,?,?)""",
            rows,
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "EmbeddingVectorCache":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
