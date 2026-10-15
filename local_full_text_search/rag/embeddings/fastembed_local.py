from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.embeddings.base import normalize_vectors
from local_full_text_search.rag.models import ModelIdentity

_TOKEN_LIMIT = re.compile(r"(\d+)\s+input tokens", re.IGNORECASE)


class FastEmbedLocalProvider:
    """Downloads a supported ONNX embedding model and runs it locally on CPU."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        *,
        cache_dir: Path,
        batch_size: int = 32,
        threads: int = 4,
        local_files_only: bool = False,
    ) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError("缺少 FastEmbed 依赖，请按 requirements-rag.txt 安装") from exc
        self.model_name = str(model_name)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = max(1, min(128, int(batch_size)))
        from fastembed import TextEmbedding

        descriptions = {
            item.get("model"): item
            for item in TextEmbedding.list_supported_models()
            if isinstance(item, dict)
        }
        spec = descriptions.get(self.model_name)
        if spec is None:
            raise ValueError(f"FastEmbed 不支持模型：{self.model_name}")
        hf_repo = str((spec.get("sources") or {}).get("hf") or "")
        snapshot_root = self.cache_dir / f"models--{hf_repo.replace('/', '--')}" / "snapshots"
        model_file = str(spec.get("model_file") or "model_optimized.onnx")
        cached = bool(
            hf_repo
            and snapshot_root.is_dir()
            and any((snapshot / model_file).is_file() for snapshot in snapshot_root.iterdir())
        )
        try:
            self.model = TextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir),
                threads=max(1, int(threads)),
                local_files_only=bool(local_files_only or cached),
            )
        except Exception as exc:
            raise RuntimeError(f"Embedding 模型下载或加载失败：{exc}") from exc
        model_dir = Path(str(getattr(self.model.model, "_model_dir", ""))).resolve()
        revision = model_dir.name
        if len(revision) < 12:
            raise RuntimeError("FastEmbed 下载目录没有可识别的模型 revision")
        dimension = int(self.model.embedding_size)
        description = str(spec.get("description") or "")
        max_tokens_match = _TOKEN_LIMIT.search(description)
        max_tokens = int(max_tokens_match.group(1)) if max_tokens_match else 512
        self._identity = ModelIdentity(self.model_name, revision, dimension, max_tokens)

    @property
    def identity(self) -> ModelIdentity:
        return self._identity

    def count_tokens(self, text: str) -> int:
        return int(self.model.token_count(str(text)))

    def embed_documents(
        self,
        texts: Sequence[str],
        token: CancelToken | None = None,
    ) -> np.ndarray:
        vectors: list[np.ndarray] = []
        values = list(texts)
        for start in range(0, len(values), self.batch_size):
            if token is not None:
                token.throw_if_cancelled()
            batch = values[start : start + self.batch_size]
            for text in batch:
                if self.count_tokens(text) > self.identity.max_tokens:
                    raise ValueError("文本超过 Embedding 模型 token 上限，拒绝静默截断")
            vectors.extend(
                np.asarray(vector, dtype=np.float32)
                for vector in self.model.passage_embed(batch, batch_size=self.batch_size)
            )
        if not vectors:
            return np.empty((0, self.identity.dimension), dtype=np.float32)
        return normalize_vectors(np.stack(vectors), self.identity.dimension)

    def embed_query(self, text: str, token: CancelToken | None = None) -> np.ndarray:
        if token is not None:
            token.throw_if_cancelled()
        if self.count_tokens(text) > self.identity.max_tokens:
            raise ValueError("问题超过 Embedding 模型 token 上限")
        vectors = list(self.model.query_embed([text]))
        if len(vectors) != 1:
            raise RuntimeError("FastEmbed query_embed 返回了错误的向量数量")
        return normalize_vectors(np.asarray(vectors[0]), self.identity.dimension)[0]
