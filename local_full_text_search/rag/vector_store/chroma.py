from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from local_full_text_search.rag.errors import RagIndexCorruptError, RagModelMissingError
from local_full_text_search.rag.models import VectorHit, VectorRecord


class ChromaVectorStore:
    COLLECTION_NAME = "rag_chunks"

    def __init__(
        self,
        persist_dir: Path,
        *,
        create: bool = False,
        hnsw_m: int = 16,
        construction_ef: int = 200,
        search_ef: int = 100,
    ) -> None:
        try:
            import chromadb
        except ImportError as exc:
            raise RagModelMissingError(
                "缺少 Chroma 向量数据库依赖，请安装 requirements-rag.txt"
            ) from exc
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = None
        if create:
            configuration = {
                "hnsw": {
                    "space": "cosine",
                    "max_neighbors": int(hnsw_m),
                    "ef_construction": int(construction_ef),
                    "ef_search": int(search_ef),
                }
            }
            self._collection = self._client.create_collection(
                name=self.COLLECTION_NAME,
                embedding_function=None,
                configuration=configuration,
            )
        else:
            try:
                self._collection = self._client.get_collection(
                    name=self.COLLECTION_NAME,
                    embedding_function=None,
                )
            except Exception as exc:
                raise RagIndexCorruptError(f"无法打开 Chroma collection：{exc}") from exc

    def upsert(self, records: Sequence[VectorRecord]) -> None:
        if not records:
            return
        self._require_collection().upsert(
            ids=[record.point_id for record in records],
            embeddings=[list(record.embedding) for record in records],
            metadatas=[dict(record.metadata) for record in records],
        )

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        *,
        root_ids: Sequence[int] = (),
        extensions: Sequence[str] = (),
        include_ocr: bool = True,
    ) -> list[VectorHit]:
        where_parts: list[dict[str, object]] = []
        if root_ids:
            ids = [int(item) for item in root_ids]
            where_parts.append({"root_id": ids[0] if len(ids) == 1 else {"$in": ids}})
        if extensions:
            values = [str(item).lower() for item in extensions]
            where_parts.append(
                {"extension": values[0] if len(values) == 1 else {"$in": values}}
            )
        if not include_ocr:
            where_parts.append({"source_type": {"$ne": "ocr"}})
        where = None
        if len(where_parts) == 1:
            where = where_parts[0]
        elif where_parts:
            where = {"$and": where_parts}
        result = self._require_collection().query(
            query_embeddings=[list(vector)],
            n_results=max(1, int(top_k)),
            where=where,
            include=["distances", "metadatas"],
        )
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        return [
            VectorHit(
                point_id=str(point_id),
                score=1.0 - float(distance),
                metadata=dict(metadata or {}),
            )
            for point_id, distance, metadata in zip(ids, distances, metadatas)
        ]

    def count(self) -> int:
        return int(self._require_collection().count())

    def get_embeddings(self, point_ids: Sequence[str]) -> dict[str, list[float]]:
        if not point_ids:
            return {}
        response = self._require_collection().get(
            ids=list(point_ids), include=["embeddings"]
        )
        embeddings = response.get("embeddings")
        if embeddings is None:
            return {}
        return {
            str(point_id): [float(value) for value in embedding]
            for point_id, embedding in zip(response.get("ids", []), embeddings)
        }

    def close(self) -> None:
        self._collection = None
        self._client = None

    def _require_collection(self):
        if self._collection is None:
            raise RagIndexCorruptError("Chroma collection 已关闭")
        return self._collection
