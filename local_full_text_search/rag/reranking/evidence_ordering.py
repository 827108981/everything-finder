"""Opt-in source-backed ordering; no Gold, domain answers, or remote calls."""
from __future__ import annotations

from dataclasses import asdict
from local_full_text_search.rag.reranking.evidence_features import evidence_features


class EvidenceOrdering:
    def __init__(self, *, rules: bool = False, judge=None, semantic_ids: set[str] | None = None,
                 uncertainty_top_k: int | None = None):
        self.rules = rules
        self.judge = judge
        self.semantic_ids = semantic_ids
        self.uncertainty_top_k = uncertainty_top_k

    def order(self, question, candidates, token):
        base = sorted(candidates, key=lambda x: (-float(x.rerank_score or 0), -x.fused_score, x.chunk.stable_chunk_id))
        routed = self.semantic_ids
        if routed is None and self.uncertainty_top_k is not None:
            k = max(1, self.uncertainty_top_k)
            routed = ({x.chunk.stable_chunk_id for x in candidates[:k]}
                      ^ {x.chunk.stable_chunk_id for x in base[:k]})
        representatives = set()
        seen = set()
        for item in base:
            features = evidence_features(question, item.chunk.text)
            if self.rules:
                for row in features['verified_rows']:
                    identifier = row['identifier'].casefold()
                    if identifier not in seen:
                        representatives.add(item.chunk.stable_chunk_id); seen.add(identifier)
            item.rerank_diagnostics = {'strategy': 'source_bound_v1', 'features': features,
                'rule_representative': item.chunk.stable_chunk_id in representatives,
                'source_binding': {'chunk_id':item.chunk.stable_chunk_id,
                    'member_chunk_ids':list(item.all_member_chunk_ids),
                    'offset_basis':'original candidate body; Python Unicode character offsets',
                    'source_locations':[asdict(s) for s in item.all_sources]}}
        for rank, item in enumerate(base, 1):
            item.rerank_diagnostics['base_rank'] = rank
            token.throw_if_cancelled()
            if self.judge is not None and (routed is None or item.chunk.stable_chunk_id in routed):
                judgement = self.judge.judge(question, item.chunk.text)
                item.rerank_diagnostics['semantic'] = asdict(judgement)
            else:
                item.rerank_diagnostics['semantic'] = {'valid': False, 'error': 'not_routed', 'support': 'unknown', 'applicability': 'unknown'}
            semantic = item.rerank_diagnostics['semantic']
            if semantic.get('valid') and semantic['applicability'] == 'conflict':
                tier = 3; reason = 'Source-backed explicit conflict; not deleted'
            elif semantic.get('valid') and semantic['support'] == 'direct':
                tier = 0; reason = 'Quoted direct support'
            elif item.chunk.stable_chunk_id in representatives:
                tier = 0; reason = 'Complete labelled record for explicit identifier'
            elif semantic.get('valid') and semantic['support'] in {'related', 'irrelevant'}:
                tier = 2; reason = 'Semantic judgement found no requested answer content'
            else:
                # Partial/unknown/invalid remain neutral; no blanket unknown penalty.
                tier = 1; reason = 'Partial or unknown: retain base relative order'
            item.rerank_diagnostics.update(tier=tier, reason=reason)
            # Keep existing EvidencePolicy thresholds on the original CE scale.
            item.rerank_score_combined = item.rerank_score
        ordered = sorted(base, key=lambda x: (x.rerank_diagnostics['tier'], x.rerank_diagnostics['base_rank']))
        for rank, item in enumerate(ordered, 1): item.rerank_diagnostics['final_rank'] = rank
        return ordered
