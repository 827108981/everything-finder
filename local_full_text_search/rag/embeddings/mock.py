from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.base import normalize_vectors
from local_full_text_search.rag.models import ModelIdentity

_MOCK_TERMS = re.compile(r"[a-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


class MockEmbeddingProvider:
    """Deterministic lexical-like vectors for tests; never use for real answers."""

    def __init__(self, dimension: int = 96) -> None:
        self._identity = ModelIdentity("mock-hash", "mock-v1", int(dimension), 4096)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def count_tokens(self, text: str) -> int:
        return max(1, len(_MOCK_TERMS.findall(normalize_text(text))))

    def embed_documents(
        self,
        texts: Sequence[str],
        token: CancelToken | None = None,
    ) -> np.ndarray:
        if token is not None:
            token.throw_if_cancelled()
        return normalize_vectors(
            np.stack([self._embed(text) for text in texts])
            if texts
            else np.empty((0, self.identity.dimension), dtype=np.float32),
            self.identity.dimension,
        )

    def embed_query(self, text: str, token: CancelToken | None = None) -> np.ndarray:
        if token is not None:
            token.throw_if_cancelled()
        return self._embed(text)

    def _embed(self, text: str) -> np.ndarray:
        dimension = self.identity.dimension
        vector = np.zeros(dimension, dtype=np.float32)
        terms = _MOCK_TERMS.findall(normalize_text(text))
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % dimension
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        if not np.any(vector):
            vector[0] = 1.0
        return normalize_vectors(vector, dimension)[0]
