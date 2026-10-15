from __future__ import annotations

import heapq

import numpy as np

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.base import EmbeddingProvider
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.common import allowed_source_exists, load_evidence


class VectorRetriever:
    def __init__(self, scan_batch_size: int = 2048, candidate_multiplier: int = 2) -> None:
        self.scan_batch_size = max(64, int(scan_batch_size))
        self.candidate_multiplier = max(1, min(5, int(candidate_multiplier)))

    def retrieve(
        self,
        connection,
        query: RagQuery,
        embedding: EmbeddingProvider,
        token: CancelToken,
        top_n: int = 30,
    ):
        query_vector = embedding.embed_query(query.question, token)
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query_vector.shape != (embedding.identity.dimension,) or not np.isfinite(query_vector).all():
            raise ValueError("query embedding has invalid dimension or value")
        query_norm = float(np.linalg.norm(query_vector))
        if query_norm <= 0:
            raise ValueError("query embedding is a zero vector")
        query_vector = query_vector / query_norm
        allowed_sql, allowed_params = allowed_source_exists(query)
        wanted = max(1, int(top_n)) * self.candidate_multiplier
        heap: list[tuple[float, int]] = []
        last_rowid = 0
        while True:
            token.throw_if_cancelled()
            rows = connection.execute(
                "SELECT e.chunk_rowid,e.vector_blob,e.dimension FROM rag_embeddings e "
                "JOIN rag_chunks c ON c.chunk_rowid=e.chunk_rowid "
                f"WHERE e.chunk_rowid>? AND {allowed_sql} "
                "ORDER BY e.chunk_rowid LIMIT ?",
                (last_rowid, *allowed_params, self.scan_batch_size),
            ).fetchall()
            if not rows:
                break
            last_rowid = int(rows[-1]["chunk_rowid"])
            vector_values: list[np.ndarray] = []
            rowids: list[int] = []
            for row in rows:
                if int(row["dimension"]) != embedding.identity.dimension:
                    raise ValueError("stored vector dimension does not match active embedding model")
                vector = np.frombuffer(row["vector_blob"], dtype="<f4")
                if vector.shape != (embedding.identity.dimension,) or not np.isfinite(vector).all():
                    raise ValueError("stored vector has invalid shape or values")
                vector_values.append(vector)
                rowids.append(int(row["chunk_rowid"]))
            matrix = np.stack(vector_values)
            scores = matrix @ query_vector
            for rowid, score_value in zip(rowids, scores, strict=True):
                score = float(score_value)
                item = (score, -rowid)
                if len(heap) < wanted:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
        ranked = sorted(heap, reverse=True)
        output = []
        seen: set[str] = set()
        for _score, negative_rowid in ranked:
            token.throw_if_cancelled()
            evidence = load_evidence(connection, -negative_rowid, query)
            if evidence is None or evidence.chunk.stable_chunk_id in seen:
                continue
            output.append(evidence)
            seen.add(evidence.chunk.stable_chunk_id)
            if len(output) >= max(1, int(top_n)):
                break
        return output
