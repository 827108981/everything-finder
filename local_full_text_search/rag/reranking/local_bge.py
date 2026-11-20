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
        cross_encoder_weight: float = 1.0,
        retrieval_prior_weight: float = 0.0,
        answer_support_weight: float = 0.0,
        max_rank_drop: int = 0,
        ordering_policy=None,
        input_mode: str = "window_max",
        window_overlap: int = 64,
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
        # Legacy keyword arguments remain accepted for saved configurations and
        # older callers. They cannot re-enable mixed scoring or rank protection.
        self.cross_encoder_weight = 1.0
        self.retrieval_prior_weight = 0.0
        self.answer_support_weight = 0.0
        self.max_rank_drop = 0
        self.ordering_policy = ordering_policy
        if input_mode not in {"truncate", "window_max", "window_top2_mean"}:
            raise ValueError(f"Unknown BGE input mode: {input_mode}")
        if window_overlap < 0:
            raise ValueError("window_overlap must be nonnegative")
        self.input_mode = input_mode
        self.window_overlap = int(window_overlap)
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
        if self.input_mode != "truncate":
            return self._rerank_windows(question, candidates, top_n, token)
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
                raw_logits = logits.cpu().tolist()
            for item, score, raw_logit in zip(batch, scores, raw_logits, strict=True):
                item.rerank_raw_logit = float(raw_logit)
                item.rerank_score = float(score)
                item.rerank_diagnostics = {"input_mode": "truncate", "max_length": self.max_length}
                scored.append(item)
        scored = self._order_scored(question, scored, token)
        # Preserve the complete order for diagnostics and downstream selection.
        self.last_scored_candidates = tuple(scored)
        return scored[:max(1, int(top_n))]

    def _windows(self, question: str, text: str):
        """Token-aligned overlapping windows; every window repeats the query.

        Offsets refer to the unchanged source text. No filename, Gold, query
        expansion or business rules are involved. An overlong query is reported
        explicitly rather than silently shortening it or dropping body tokens.
        """
        limit = min(self.max_length, int(self.tokenizer.model_max_length))
        query_tokens = len(self.tokenizer(question, add_special_tokens=False)["input_ids"])
        budget = limit - query_tokens - self.tokenizer.num_special_tokens_to_add(pair=True)
        if budget <= 0:
            raise RagModelValidationError("问题超过 BGE 输入预算，无法保留问题并读取证据")
        encoded = self.tokenizer(
            question, text, truncation="only_second", max_length=limit,
            stride=min(self.window_overlap, budget - 1),
            return_overflowing_tokens=True, return_offsets_mapping=True,
        )
        windows = []
        for i, ids in enumerate(encoded["input_ids"]):
            spans = [span for span, side in zip(encoded["offset_mapping"][i], encoded.sequence_ids(i))
                     if side == 1 and span[1] > span[0]]
            windows.append((
                {key: encoded[key][i] for key in self.tokenizer.model_input_names if key in encoded},
                {"start_char": min((a for a, _ in spans), default=0),
                 "end_char": max((b for _, b in spans), default=0), "tokens": len(ids)},
            ))
        return windows

    def _score_window_inputs(self, inputs, token):
        results = []
        for start in range(0, len(inputs), self.batch_size):
            token.throw_if_cancelled()
            encoded = self.tokenizer.pad(inputs[start:start+self.batch_size], padding=True, return_tensors="pt")
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                logits = self.model(**encoded).logits.reshape(-1)
                scores = self._torch.sigmoid(logits).cpu().tolist()
                raw = logits.cpu().tolist()
            results.extend(zip(scores, raw, strict=True))
        return results

    @staticmethod
    def _aggregate_windows(scores, mode):
        if not scores:
            raise ValueError("Cannot aggregate empty BGE windows")
        ordered = sorted(scores, reverse=True)
        if mode == "window_max":
            return ordered[0]
        if mode == "window_top2_mean":
            top = ordered[:2]
            return sum(top) / len(top)
        raise ValueError(f"Unknown window aggregation: {mode}")

    def _rerank_windows(self, question, candidates, top_n, token):
        import math
        inputs, owners, windows = [], [], [[] for _ in candidates]
        for index, item in enumerate(candidates):
            token.throw_if_cancelled()
            for encoded, metadata in self._windows(question, item.chunk.text):
                inputs.append(encoded)
                owners.append(index)
                windows[index].append(metadata)
        scored = self._score_window_inputs(inputs, token)
        assert len(scored) == len(inputs)
        offsets = [0] * len(candidates)
        for owner, (score, logit) in zip(owners, scored, strict=True):
            windows[owner][offsets[owner]].update(score=float(score), raw_logit=float(logit))
            offsets[owner] += 1
        for item, pieces in zip(candidates, windows, strict=True):
            item.rerank_score = self._aggregate_windows([p["score"] for p in pieces], self.input_mode)
            # A mean probability has no single model logit; record that distinction.
            if self.input_mode == "window_max" or len(pieces) == 1:
                item.rerank_raw_logit = max(pieces, key=lambda p: p["score"])["raw_logit"]
            else:
                p = min(1-1e-12, max(1e-12, item.rerank_score))
                item.rerank_raw_logit = math.log(p / (1-p))
            item.rerank_diagnostics = {
                "input_mode": self.input_mode, "window_overlap": self.window_overlap,
                "window_count": len(pieces), "body_chars": len(item.chunk.text),
                "windows": pieces,
                "logit_kind": "aggregate_probability_logit" if self.input_mode == "window_top2_mean" and len(pieces)>1 else "model_logit",
            }
        ordered = self._order_scored(question, candidates, token)
        self.last_scored_candidates = tuple(ordered)
        return ordered[:max(1, int(top_n))]

    def _order_scored(self, question, candidates, token) -> list[EvidenceChunk]:
        """BGE-only default; fusion is a deterministic tie breaker, not a bonus."""
        scored = list(candidates)
        for item in scored:
            item.rerank_score_combined = item.rerank_score
            item.rerank_diagnostics = {
                **(item.rerank_diagnostics or {}),
                'strategy': 'bge_base',
                'reason': 'BGE score descending; no support bonus or fused-rank guard',
            }
        scored.sort(
            key=lambda item: (
                -float(item.rerank_score or 0.0),
                -float(item.fused_score),
                item.chunk.stable_chunk_id,
            )
        )
        if getattr(self, "ordering_policy", None) is not None:
            scored = self.ordering_policy.order(question, scored, token)
        return scored
