from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from local_full_text_search.rag.errors import RagModelMissingError, RagModelValidationError


MAX_EMBEDDING_BATCH_SIZE = 64


class LocalBgeEmbeddingProvider:
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_path: Path,
        *,
        repo_id: str,
        revision: str,
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RagModelMissingError(
                "缺少 sentence-transformers/PyTorch，请安装 requirements-rag.txt"
            ) from exc
        self._model_id = repo_id
        self._revision = revision
        self._batch_size = min(MAX_EMBEDDING_BATCH_SIZE, max(1, int(batch_size)))
        try:
            self._model = SentenceTransformer(str(model_path), device=device)
            get_dimension = getattr(self._model, "get_embedding_dimension", None)
            dimension = (
                get_dimension()
                if callable(get_dimension)
                else self._model.get_sentence_embedding_dimension()
            )
        except Exception as exc:
            raise RagModelValidationError(f"无法加载 Embedding 模型：{exc}") from exc
        if int(dimension or 0) != 512:
            raise RagModelValidationError(
                f"Embedding 模型维度应为 512，实际为 {dimension}"
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return 512

    @property
    def batch_size(self) -> int:
        return self._batch_size

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validated(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        vectors = self._model.encode(
            [self.QUERY_PREFIX + str(text)],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return self._validated(vectors, 1)[0]

    @staticmethod
    def _validated(vectors, expected_rows: int) -> list[list[float]]:
        import numpy as np

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape != (expected_rows, 512):
            raise RagModelValidationError(f"Embedding 输出形状异常：{array.shape}")
        if not np.isfinite(array).all():
            raise RagModelValidationError("Embedding 输出包含 NaN 或无穷值")
        return array.tolist()
