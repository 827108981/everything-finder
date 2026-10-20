from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from local_full_text_search.rag.models import CandidateDecision, EvidenceChunk, RagQuery


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    accepted: tuple[EvidenceChunk, ...]
    insufficient: bool
    reason: str
    rejected: tuple[tuple[str, str], ...] = ()
    needs_clarification: bool = False
    clarification_question: str = ""


class EvidencePolicy:
    """先执行可解释的硬门禁，再按重排分数和文档多样性选最终证据。"""

    def __init__(
        self,
        *,
        min_vector_score: float = 0.35,
        min_rerank_score: float = 0.10,
        max_per_document: int = 3,
    ):
        self.min_vector_score = float(min_vector_score)
        self.min_rerank_score = float(min_rerank_score)
        self.max_per_document = max(1, int(max_per_document))

    def evaluate(
        self, query: RagQuery, candidates: Sequence[EvidenceChunk], *, plan=None
    ) -> EvidenceDecision:
        if not candidates:
            return EvidenceDecision((), True, "no_candidates")
        accepted: list[EvidenceChunk] = []
        rejected: list[tuple[str, str]] = []
        for item in candidates:
            reason = self.rejection_reason(item)
            if reason:
                rejected.append((item.chunk.stable_chunk_id, reason))
            else:
                accepted.append(item)

        if not accepted:
            reason = (
                "all_candidates_conflict_with_query"
                if any("冲突" in value for _, value in rejected)
                else "below_retrieval_or_rerank_threshold"
            )
            return EvidenceDecision((), True, reason, tuple(rejected))

        # 条件明确匹配的候选优先于未说明条件的候选，二者内部沿用 BGE 次序。
        accepted.sort(
            key=lambda item: (
                bool(item.constraint_decision and item.constraint_decision.required_unknown_count),
                -float(item.rerank_score or 0.0),
                item.chunk.stable_chunk_id,
            )
        )
        selected: list[EvidenceChunk] = []
        counts: dict[str, int] = {}
        for item in accepted:
            document_key = item.chunk.document_key
            if counts.get(document_key, 0) >= self.max_per_document:
                rejected.append((item.chunk.stable_chunk_id, "同一文档候选数量达到上限"))
                continue
            selected.append(item)
            counts[document_key] = counts.get(document_key, 0) + 1
            if len(selected) >= max(1, int(query.final_top_k)):
                break
        if not selected:
            return EvidenceDecision((), True, "no_diverse_evidence", tuple(rejected))
        if plan is not None:
            # 明确限定了范围/阶段/方向时，至少要有一条最终证据直接说明该条件。
            critical_kinds = {"scope", "stage", "direction"}
            for constraint in plan.constraints:
                if not constraint.required or constraint.kind.value not in critical_kinds:
                    continue
                covered = any(
                    item.constraint_decision
                    and any(
                        judgement.constraint.kind is constraint.kind
                        and judgement.constraint.value == constraint.value
                        and judgement.status.value == "match"
                        for judgement in item.constraint_decision.judgements
                    )
                    for item in selected
                )
                if not covered:
                    return EvidenceDecision(
                        (),
                        True,
                        f"required_constraint_not_confirmed:{constraint.kind.value}={constraint.value}",
                        tuple(rejected),
                    )
        return EvidenceDecision(tuple(selected), False, "accepted", tuple(rejected))

    def rejection_reason(self, item: EvidenceChunk) -> str | None:
        constraint = item.constraint_decision
        if constraint is not None and constraint.required_conflict_count:
            return f"与问题中的必需业务条件冲突：{constraint.reason}"
        if item.rerank_score is None or item.rerank_score < self.min_rerank_score:
            return f"重排分数低于 {self.min_rerank_score:g}"
        if (
            item.lexical_rank is None
            and (item.vector_score is None or item.vector_score < self.min_vector_score)
        ):
            return f"未命中关键词且向量相似度低于 {self.min_vector_score:g}"
        return None
