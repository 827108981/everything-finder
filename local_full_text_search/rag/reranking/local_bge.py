from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from local_full_text_search.rag.errors import RagModelMissingError, RagModelValidationError
from local_full_text_search.rag.models import EvidenceChunk
from local_full_text_search.rag.reranking.support import retrieval_prior


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
        cross_encoder_weight: float = 0.65,
        retrieval_prior_weight: float = 0.05,
        answer_support_weight: float = 0.30,
        max_rank_drop: int = 3,
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
        weights = [
            max(0.0, float(cross_encoder_weight)),
            max(0.0, float(retrieval_prior_weight)),
            max(0.0, float(answer_support_weight)),
        ]
        total_weight = sum(weights) or 1.0
        self.cross_encoder_weight = weights[0] / total_weight
        self.retrieval_prior_weight = weights[1] / total_weight
        self.answer_support_weight = weights[2] / total_weight
        self.max_rank_drop = max(0, int(max_rank_drop))
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
                if item.answer_support_score is None and item.fused_rank is None:
                    # 兼容旧调用方：没有辅助信号时保持原始 Cross-Encoder 排序。
                    item.rerank_score_combined = float(score)
                else:
                    item.rerank_score_combined = (
                        self.cross_encoder_weight * float(score)
                        + self.retrieval_prior_weight * retrieval_prior(item)
                        + self.answer_support_weight * float(item.answer_support_score or 0.0)
                    )
                scored.append(item)
        scored.sort(
            key=lambda item: (
                -float(item.rerank_score_combined if item.rerank_score_combined is not None
                       else item.rerank_score or 0.0),
                -float(item.fused_score),
                item.chunk.stable_chunk_id,
            )
        )
        scored = self._apply_rank_guard(scored)
        # 在同一次问答中保留完整重排顺序，供界面展示最终截断前的候选。
        self.last_scored_candidates = tuple(scored)
        return self._apply_document_diversity(scored, max(1, int(top_n)))

    def _apply_rank_guard(self, candidates: Sequence[EvidenceChunk]) -> list[EvidenceChunk]:
        """限制重排相对融合名次的最大后移，保持前置检索证据可追溯。"""

        if self.max_rank_drop <= 0 or not candidates:
            return list(candidates)
        remaining = list(candidates)
        output: list[EvidenceChunk] = []
        for position in range(1, len(candidates) + 1):
            due = [
                item for item in remaining
                if item.fused_rank is not None
                and int(item.fused_rank) + self.max_rank_drop <= position
            ]
            if due:
                pool = due
            else:
                pool = [
                    item for item in remaining
                    if item.fused_rank is None
                    or int(item.fused_rank) + self.max_rank_drop >= position
                ] or remaining
            selected = pool[0]
            output.append(selected)
            remaining.remove(selected)
        return output

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
