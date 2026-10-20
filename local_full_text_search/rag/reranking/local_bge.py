from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from local_full_text_search.rag.errors import RagModelMissingError, RagModelValidationError
from local_full_text_search.rag.models import EvidenceChunk


class LocalBgeReranker:
    def __init__(
        self,
        model_path: Path,
        *,
        repo_id: str,
        revision: str,
        batch_size: int = 4,
        max_length: int = 512,
        device: str = "cpu",
    ) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
        except ImportError as exc:
            raise RagModelMissingError(
                "缺少 Transformers/PyTorch，请安装 requirements-rag.txt"
            ) from exc
        self._torch = torch
        self.model_id = repo_id
        self.revision = revision
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(32, int(max_length))
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path), local_files_only=True
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path), local_files_only=True
            ).to(device)
            self.model.eval()
            if int(getattr(self.model.config, "num_labels", 1)) != 1:
                raise ValueError("reranker 应输出一个相关度 logit")
        except Exception as exc:
            raise RagModelValidationError(f"无法加载 Cross-Encoder：{exc}") from exc
        self.last_scored_candidates: tuple[EvidenceChunk, ...] = ()

    def rerank(
        self,
        question: str,
        candidates: Sequence[EvidenceChunk],
        top_n: int,
        token,
    ) -> list[EvidenceChunk]:
        scored: list[EvidenceChunk] = []
        for start in range(0, len(candidates), self.batch_size):
            token.throw_if_cancelled()
            batch = list(candidates[start : start + self.batch_size])
            encoded = self.tokenizer(
                [question] * len(batch),
                [item.chunk.text for item in batch],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(-1)
                scores = self._torch.sigmoid(logits).cpu().tolist()
            for item, score in zip(batch, scores, strict=True):
                item.rerank_score = float(score)
                scored.append(item)
        scored.sort(
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                -float(item.fused_score),
                item.chunk.stable_chunk_id,
            )
        )
        # 在同一次问答中保留完整重排顺序，供界面展示最终截断前的候选。
        self.last_scored_candidates = tuple(scored)
        return self._apply_document_diversity(scored, max(1, int(top_n)))

    @staticmethod
    def _apply_document_diversity(
        candidates: Sequence[EvidenceChunk], top_n: int, max_per_document: int = 3
    ) -> list[EvidenceChunk]:
        counts: dict[str, int] = {}
        output: list[EvidenceChunk] = []
        for item in candidates:
            key = item.chunk.document_key
            if counts.get(key, 0) >= max_per_document:
                continue
            counts[key] = counts.get(key, 0) + 1
            output.append(item)
            if len(output) >= top_n:
                break
        return output
