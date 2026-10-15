from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import ModelIdentity


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> ModelIdentity: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(
        self,
        texts: Sequence[str],
        token: CancelToken | None = None,
    ) -> np.ndarray: ...

    def embed_query(self, text: str, token: CancelToken | None = None) -> np.ndarray: ...


def normalize_vectors(vectors: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != int(dimension):
        raise ValueError(
            f"embedding dimension mismatch: expected {dimension}, got {values.shape}"
        )
    if not np.isfinite(values).all():
        raise ValueError("embedding contains a non-finite value")
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= 0):
        raise ValueError("embedding contains a zero vector")
    return np.ascontiguousarray(values / norms[:, None], dtype=np.float32)
