"""Isolated applicability experiment over already reranked evidence."""
from __future__ import annotations

from dataclasses import replace

from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import ConstraintStatus, EvidenceChunk, QueryPlan
from local_full_text_search.rag.query_interpreter import QueryInterpreter


class ScopedApplicability:
    """Evaluate each explicit condition separately; retain the reranker order."""

    def __init__(self, branch_queries: tuple[str, ...] = (), *, mode: str = "C"):
        if mode not in {"O", "S", "C"}:
            raise ValueError(mode)
        self.branch_queries = branch_queries
        self.mode = mode
        self.matcher = ConstraintMatcher()
        self.interpreter = QueryInterpreter()
        self.last_bindings: list[dict[str, object]] = []

    def evaluate(self, plan: QueryPlan, candidates: list[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
        # The native matcher must first compute exactly the same per-candidate
        # judgement. Only the explicit branch scope and ordering are experimental.
        baseline = list(self.matcher.evaluate(plan, candidates))
        self.last_bindings = []
        if self.mode == "O" or not self.branch_queries:
            for item in candidates:
                self.last_bindings.append({"chunk_id": item.chunk.stable_chunk_id,
                                           "status": "original_scope", "branch": None,
                                           "reason": item.constraint_decision.reason if item.constraint_decision else ""})
            return self._order(candidates) if self.mode == "O" else tuple(baseline)

        # Evidence can support one of several requested conditional paths. A
        # conflict in another path cannot invalidate the entire source chunk.
        branch_plans = [self.interpreter.interpret(query) for query in self.branch_queries]
        branch_rows = []
        for branch_id, branch_plan in enumerate(branch_plans, 1):
            clones = [replace(item, constraint_decision=None, constraint_score=None) for item in candidates]
            self.matcher.evaluate(branch_plan, clones)
            branch_rows.append((branch_id, branch_plan, clones))
        for position, item in enumerate(candidates):
            options = []
            for branch_id, branch_plan, clones in branch_rows:
                copy = clones[position]
                decision = copy.constraint_decision
                assert decision is not None
                options.append((branch_id, branch_plan, decision, copy.constraint_score or 0.0))
            usable = [row for row in options if row[2].accepted]
            chosen = sorted(usable or options, key=lambda row: (
                not row[2].accepted, row[2].required_unknown_count,
                -row[2].required_match_count, row[0]))[0]
            item.constraint_decision = chosen[2]
            item.constraint_score = chosen[3]
            self.last_bindings.append({
                "chunk_id": item.chunk.stable_chunk_id,
                "branch": chosen[0], "status": "matched_or_unknown" if usable else "all_branches_conflict",
                "reason": chosen[2].reason,
                "branches": [{"branch": b, "query": p.original_text,
                              "accepted": d.accepted,
                              "judgements": [{"kind": j.constraint.kind.value, "value": j.constraint.value,
                                              "status": j.status.value, "reason": j.reason,
                                              "matched_text": j.matched_text}
                                             for j in d.judgements]}
                             for b, p, d, _ in options],
            })
        if self.mode == "S":
            return tuple(sorted(candidates, key=lambda item: (
                not item.constraint_decision.accepted,
                bool(item.constraint_decision.required_unknown_count),
                -(item.constraint_score or 0.0))))
        return self._order(candidates)

    def policy_plan(self, plan: QueryPlan) -> QueryPlan:
        """Final coverage gate may require only conditions shared by every branch."""
        if self.mode not in {"S", "C"} or not self.branch_queries:
            return plan
        global_constraints = tuple(
            condition for condition in plan.constraints
            if condition.source_text and all(condition.source_text in branch for branch in self.branch_queries)
        )
        return replace(plan, constraints=global_constraints)

    @staticmethod
    def _order(candidates: list[EvidenceChunk]) -> tuple[EvidenceChunk, ...]:
        return tuple(sorted(candidates, key=lambda item: not item.constraint_decision.accepted))
