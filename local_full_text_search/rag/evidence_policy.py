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
    """执行证据门禁并保留上游顺序；None 表示不设单文档配额。"""

    # Retained for explicit historical experiments, not the default answer path.
    uses_constraints = True
    preserve_complete_chunks = False
    selection_mode = "legacy_gates"

    def __init__(
        self,
        *,
        min_vector_score: float = 0.35,
        min_rerank_score: float = 0.10,
        max_per_document: int | None = 3,
        require_confirmed_conditions: bool = True,
    ):
        self.min_vector_score = float(min_vector_score)
        self.min_rerank_score = float(min_rerank_score)
        self.require_confirmed_conditions = bool(require_confirmed_conditions)
        # 显式整数保留历史实验/注入策略的兼容性；正式入口采用 None。
        self.max_per_document = None if max_per_document is None else max(1, int(max_per_document))

    def evaluate(
        self, query: RagQuery, candidates: Sequence[EvidenceChunk], *, plan=None
    ) -> EvidenceDecision:
        if not candidates:
            return EvidenceDecision((), True, "no_candidates")
        accepted: list[EvidenceChunk] = []
        rejected: list[tuple[str, str]] = []
        for item in candidates:
            reason = self.rejection_reason(item, check_constraints=plan is not None)
            if plan is not None and plan.adapter_active:
                from local_full_text_search.rag.identifier_evidence import annotate
                diagnostics = annotate(plan, item)
                if diagnostics.get('identifier_conflict'):
                    reason = 'explicit_identifier_conflict: 正文指向其他报警，不能作为当前报警处理依据'
                if diagnostics.get('explicit_scope_conflicts'):
                    reason = 'explicit_scope_conflict: 原文明确限定的型号或版本与用户条件冲突'
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

        # 条件阶段可能沿用BGE顺序，也可能使用旧排序。证据门禁只过滤，
        # 不另加分，保证最终证据与送入上下文的顺序可追溯。
        selected: list[EvidenceChunk] = []
        from local_full_text_search.rag.identifier_evidence import prioritize
        accepted = prioritize(plan, accepted)
        if plan is not None and plan.adapter_active:
            from local_full_text_search.rag.source_consensus import reserve
            accepted = reserve(accepted)
        counts: dict[str, int] = {}
        final_limit = max(1, int(query.final_top_k))
        for item in accepted:
            if len(selected) >= final_limit:
                rejected.append((item.chunk.stable_chunk_id, "最终 TopK 截断"))
                continue
            document_key = item.chunk.document_key
            exact_protected = bool(item.query_plan_diagnostics.get('exact_representative') or item.query_plan_diagnostics.get('source_representative'))
            quota_reached = self.max_per_document is not None and counts.get(document_key, 0) >= self.max_per_document
            if quota_reached and not exact_protected:
                reason = ('标识已有保留代表；其余同文档候选达到常规上限' if item.query_plan_diagnostics.get('exact_matches')
                          else '同一文档候选数量达到上限')
                rejected.append((item.chunk.stable_chunk_id, reason))
                continue
            if exact_protected:
                item.query_plan_diagnostics['protected_from_document_quota'] = self.max_per_document is not None
                item.query_plan_diagnostics['quota_bypassed'] = quota_reached
            selected.append(item)
            if not exact_protected:
                counts[document_key] = counts.get(document_key, 0) + 1
        if not selected:
            return EvidenceDecision((), True, "no_diverse_evidence", tuple(rejected))
        if plan is not None:
            # 明确限定了范围/阶段/方向时，至少要有一条最终证据直接说明该条件。
            critical_kinds = {"scope", "stage", "direction", "device", "topic"}
            for constraint in plan.constraints:
                if not constraint.is_hard or constraint.kind.value not in critical_kinds:
                    continue
                # Local checks distinguish lack of evidence from proved conflict.
                # Unknown must not silently discard all retained context. This is
                # only active for explicitly injected local-condition experiments.
                if constraint.kind.value in {"scope", "stage", "direction", "model"} and all(
                    item.query_plan_diagnostics.get('local_conditions') for item in selected
                ):
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
                    if not self.require_confirmed_conditions:
                        for item in selected:
                            notes = item.query_plan_diagnostics.setdefault('unconfirmed_conditions', [])
                            if missing not in notes:
                                notes.append(missing)
                        continue
                    return EvidenceDecision(
                        (),
                        True,
                        f"required_constraint_not_confirmed:{missing}",
                        (*rejected, *(
                            (item.chunk.stable_chunk_id, f"关键硬条件未获证据确认：{missing}")
                            for item in selected
                        )),
                    )
        uncertain = any(item.query_plan_diagnostics.get('unconfirmed_conditions') for item in selected)
        return EvidenceDecision(tuple(selected), False,
                                "accepted_with_unconfirmed_conditions" if uncertain else "accepted", tuple(rejected))

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


class SimpleEvidencePolicy:
    """Pass ranked candidates to source-bound, budgeted Context assembly.

    Relevance is owned by the reranker. There are no inferred business-condition
    filters, score cutoffs, per-document quotas, or fixed final fragment count.
    Exact duplicate bodies and all their sources are merged by ContextBuilder.
    The legacy class above remains available for reproducible comparisons.
    """

    uses_constraints = False
    preserve_complete_chunks = True
    selection_mode = "bge_order_complete_chunks_budget"
    min_vector_score = None
    min_rerank_score = None
    max_per_document = None
    require_confirmed_conditions = False

    def evaluate(self, query: RagQuery, candidates: Sequence[EvidenceChunk], *, plan=None) -> EvidenceDecision:
        if not candidates:
            return EvidenceDecision((), True, "no_candidates")
        return EvidenceDecision(tuple(candidates), False, "ranked_candidates_for_context_budget")
