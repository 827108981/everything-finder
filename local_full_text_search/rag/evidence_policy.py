from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from local_full_text_search.rag.models import ConstraintStatus, EvidenceChunk, RagQuery


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
            reason = self.rejection_reason(item, check_constraints=plan is not None)
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

        # 约束阶段已经完成“适用条件优先”的稳定排序；证据门禁只过滤，
        # 不再按另一套分数重新排序，保证最终证据与送入上下文的顺序可追溯。
        selected: list[EvidenceChunk] = []
        counts: dict[str, int] = {}
        final_limit = max(1, int(query.final_top_k))
        for item in accepted:
            if len(selected) >= final_limit:
                rejected.append((item.chunk.stable_chunk_id, "最终 TopK 截断"))
                continue
            document_key = item.chunk.document_key
            if counts.get(document_key, 0) >= self.max_per_document:
                rejected.append((item.chunk.stable_chunk_id, "同一文档候选数量达到上限"))
                continue
            selected.append(item)
            counts[document_key] = counts.get(document_key, 0) + 1
        if not selected:
            return EvidenceDecision((), True, "no_diverse_evidence", tuple(rejected))
        if plan is not None:
            # 明确限定了范围/阶段/方向时，至少要有一条最终证据直接说明该条件。
            critical_kinds = {"scope", "stage", "direction", "device", "topic"}
            for constraint in plan.constraints:
                if not constraint.is_hard or constraint.kind.value not in critical_kinds:
                    continue
                covered = any(
                    item.constraint_decision
                    and any(
                        judgement.constraint == constraint
                        and judgement.status is ConstraintStatus.MATCH
                        for judgement in item.constraint_decision.judgements
                    )
                    for item in selected
                )
                if not covered:
                    missing = f"{constraint.kind.value}={constraint.value}"
                    return EvidenceDecision(
                        (),
                        True,
                        f"required_constraint_not_confirmed:{missing}",
                        (*rejected, *(
                            (item.chunk.stable_chunk_id, f"关键硬条件未获证据确认：{missing}")
                            for item in selected
                        )),
                    )
        return EvidenceDecision(tuple(selected), False, "accepted", tuple(rejected))

    def rejection_reason(
        self, item: EvidenceChunk, *, check_constraints: bool = True
    ) -> str | None:
        constraint = item.constraint_decision
        hard_conflicts = (
            [judgement for judgement in constraint.judgements
             if judgement.constraint.is_hard and judgement.status is ConstraintStatus.CONFLICT]
            if check_constraints and constraint is not None else []
        )
        if hard_conflicts:
            return f"与问题中的硬条件冲突：{constraint.reason}"
        if item.rerank_score is None or item.rerank_score < self.min_rerank_score:
            return f"重排分数低于 {self.min_rerank_score:g}"
        if (
            item.lexical_rank is None
            and (item.vector_score is None or item.vector_score < self.min_vector_score)
        ):
            return f"未命中关键词且向量相似度低于 {self.min_vector_score:g}"
        return None
