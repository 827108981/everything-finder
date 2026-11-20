"""Opt-in, offline evidence-selection experiments; never imported by runtime.

Inputs are the frozen upstream order, judgments and document vectors. No Gold,
question identifiers, file-name rules or page rules enter selection.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

import numpy as np

from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidenceDecision, EvidencePolicy


@dataclass(frozen=True)
class SelectionFactors:
    remove_document_cap: bool = False
    redundancy_order: bool = False
    budget_pack: bool = False
    relevance_weight: float = .8
    safety_chunk_limit: int = 64

    def __post_init__(self):
        if not .5 <= self.relevance_weight <= 1:
            raise ValueError('relevance_weight must be in [0.5, 1]')
        if self.safety_chunk_limit < 1:
            raise ValueError('safety_chunk_limit must be positive')


ARMS = {
    'baseline': SelectionFactors(),
    'A': SelectionFactors(True),
    'B': SelectionFactors(False, True),
    'C': SelectionFactors(False, False, True),
    'AB': SelectionFactors(True, True),
    'AC': SelectionFactors(True, False, True),
    'BC': SelectionFactors(False, True, True),
    'ABC': SelectionFactors(True, True, True),
}


def select_context(query, candidates, plan, policy: EvidencePolicy, max_chars: int,
                   factors: SelectionFactors = SelectionFactors(),
                   vectors: Mapping[str, Sequence[float]] | None = None):
    """Return decision, actual Context and each candidate's selection decision.

MMR relevance is the normalized frozen eligible rank, not a newly calculated
business score. Redundancy is cosine to *actually selected* candidates. Packing
uses native ContextBuilder serialization, including headers and source metadata.
The all-off path delegates byte-for-byte to the production implementation.
"""
    builder = ContextBuilder()
    if not any((factors.remove_document_cap, factors.redundancy_order, factors.budget_pack)):
        decision = policy.evaluate(query, candidates, plan=plan)
        return decision, builder.build(decision.accepted, max_chars=max_chars), []
    if plan is not None and plan.adapter_active:
        raise ValueError('This experiment freezes the adapter-disabled pipeline')
    selected, rejected, trace, eligible = [], [], [], []
    for item in candidates:
        reason = policy.rejection_reason(item, check_constraints=plan is not None)
        if reason:
            rejected.append((item.chunk.stable_chunk_id, reason))
            trace.append({'chunk_id': item.chunk.stable_chunk_id, 'action': 'quality_or_condition_gate', 'reason': reason})
        else:
            eligible.append(item)
    original_rank = {x.chunk.stable_chunk_id: n for n, x in enumerate(eligible)}
    normalized_vectors = {}
    if factors.redundancy_order:
        for item in eligible:
            cid = item.chunk.stable_chunk_id
            v = np.asarray((vectors or {}).get(cid, []), dtype=np.float64)
            if v.ndim != 1 or not v.size or not np.isfinite(v).all() or np.linalg.norm(v) <= 0:
                raise ValueError(f'missing or invalid local vector: {cid}')
            normalized_vectors[cid] = v / np.linalg.norm(v)
        if len({len(v) for v in normalized_vectors.values()}) > 1:
            raise ValueError('inconsistent embedding dimensions')
    counts = {}
    limit = factors.safety_chunk_limit if factors.budget_pack else max(1, query.final_top_k)
    pool = list(eligible)
    while pool:
        if len(selected) >= limit:
            for item in pool:
                reason = 'safety_chunk_limit' if factors.budget_pack else '最终 TopK 截断'
                rejected.append((item.chunk.stable_chunk_id, reason))
                trace.append({'chunk_id': item.chunk.stable_chunk_id, 'action': reason})
            break
        def components(item):
            cid = item.chunk.stable_chunk_id
            relevance = 1 - original_rank[cid] / max(1, len(eligible) - 1)
            redundancy = max((float(normalized_vectors[cid] @ normalized_vectors[x.chunk.stable_chunk_id])
                              for x in selected), default=0.) if factors.redundancy_order else 0.
            redundancy = max(0., min(1., redundancy))
            score = factors.relevance_weight * relevance - (1 - factors.relevance_weight) * redundancy
            assert isfinite(score)
            return score, relevance, redundancy
        item = max(pool, key=lambda x: (components(x)[0], -original_rank[x.chunk.stable_chunk_id])) if factors.redundancy_order else pool[0]
        pool.remove(item)
        cid, doc = item.chunk.stable_chunk_id, item.chunk.document_key
        score, relevance, redundancy = components(item)
        event = {'chunk_id': cid, 'upstream_eligible_rank': original_rank[cid] + 1,
                 'mmr_score': score if factors.redundancy_order else None,
                 'rank_relevance': relevance, 'redundancy': redundancy,
                 'compared_to': [x.chunk.stable_chunk_id for x in selected] if factors.redundancy_order else []}
        if not factors.remove_document_cap and counts.get(doc, 0) >= policy.max_per_document:
            reason = '同一文档候选数量达到上限'
            rejected.append((cid, reason)); trace.append({**event, 'action': 'document_cap', 'reason': reason})
            continue
        if factors.budget_pack:
            # No rewriting, extra neighbors or partial-body truncation. Render
            # exactly the existing units; include all source headers in cost.
            trial = builder.build([*selected, item], max_chars=2**31 - 1)
            if len(trial.text) > max(256, max_chars):
                reason = 'complete_chunk_exceeds_remaining_budget'
                rejected.append((cid, reason))
                trace.append({**event, 'action': reason, 'trial_context_chars': len(trial.text)})
                continue
        selected.append(item); counts[doc] = counts.get(doc, 0) + 1
        trace.append({**event, 'action': 'selected', 'selection_rank': len(selected)})
    # Preserve the current final hard-condition confirmation policy. A changed
    # selection cannot evade that gate, including after budget packing.
    final_policy = EvidencePolicy(min_vector_score=policy.min_vector_score,
                                  min_rerank_score=policy.min_rerank_score,
                                  max_per_document=max(1, len(selected)))
    final = final_policy.evaluate(replace(query, final_top_k=max(1, len(selected))), selected, plan=plan)
    decision = replace(final, rejected=tuple(rejected) + final.rejected)
    context = builder.build(decision.accepted, max_chars=max_chars)
    if factors.budget_pack:
        assert len(context.text) <= max(256, max_chars)
        assert not any(e['truncated'] for e in context.entries)
    return decision, context, trace
