from __future__ import annotations

import gc
from collections.abc import Sequence
from pathlib import Path

from local_full_text_search.rag.errors import RagModelMissingError, RagModelValidationError


MAX_EMBEDDING_BATCH_SIZE = 64


class LocalBgeEmbeddingProvider:
    QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："
    QWEN3_QUERY_PROMPT = (
        "Instruct: Given a question, retrieve passages that answer the question\n"
        "Query: "
    )

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
        self._uses_qwen3_query_prompt = "qwen3-embedding" in repo_id.lower()
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
        self._dimension = int(dimension or 0)
        if self._dimension <= 0:
            raise RagModelValidationError(
                f"Embedding 模型维度必须为正数，实际为 {dimension}"
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_revision(self) -> str:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_sequence_length(self) -> int:
        return int(getattr(self._model, "max_seq_length", 512) or 512)

    def count_tokens(self, text: str) -> int:
        """使用模型自己的 tokenizer 计数，避免按字符数猜测截断边界。"""
        tokenizer = getattr(self._model, "tokenizer", None)
        if tokenizer is None:
            raise RagModelValidationError("Embedding 模型没有可用 tokenizer")
        encoded = tokenizer(
            str(text), add_special_tokens=True, truncation=False,
            return_attention_mask=False, return_token_type_ids=False,
        )
        return len(encoded.get("input_ids") or ())

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        while True:
            try:
                vectors = self._model.encode(
                    list(texts),
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                break
            except BaseException as exc:
                if not self._is_memory_error(exc) or self._batch_size <= 1:
                    raise
                self._batch_size = max(1, self._batch_size // 2)
                gc.collect()
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except (ImportError, RuntimeError):
                    pass
        return self._validated(vectors, len(texts))

    def embed_query(self, text: str) -> list[float]:
        query_text = str(text)
        encode_kwargs = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if self._uses_qwen3_query_prompt:
            # Qwen3-Embedding 的 query prompt 与文档向量 prompt 不同；
            # 显式加入任务指令，避免把查询和正文编码成同一种语气。
            query_text = self.QWEN3_QUERY_PROMPT + query_text
        else:
            query_text = self.QUERY_PREFIX + query_text
        vectors = self._model.encode(
            [query_text], **encode_kwargs,
        )
        return self._validated(vectors, 1)[0]

    def _validated(self, vectors, expected_rows: int) -> list[list[float]]:
        import numpy as np

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape != (expected_rows, self._dimension):
            raise RagModelValidationError(f"Embedding 输出形状异常：{array.shape}")
        if not np.isfinite(array).all():
            raise RagModelValidationError("Embedding 输出包含 NaN 或无穷值")
        return array.tolist()

    @staticmethod
    def _is_memory_error(exc: BaseException) -> bool:
        if isinstance(exc, MemoryError):
            return True
        message = str(exc).lower()
        return any(marker in message for marker in (
            "out of memory",
            "not enough memory",
            "cannot allocate memory",
            "defaultcpuallocator",
            "bad allocation",
        ))
