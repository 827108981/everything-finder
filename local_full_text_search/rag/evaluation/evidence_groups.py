"""Explicitly approved OR evidence groups; source/quote matching stays strict.

Each group is one requirement. Each alternative is a list of evidence mappings
that must ALL match. Only an explicitly approved group can enter the result.
This opt-in adapter does not change legacy Gold or default report scoring.
"""
from .stage_scoring import score_stage


def score_evidence_groups(stage, candidates, groups):
    results = []
    for group in groups:
        if group.get('approved') is not True:
            raise ValueError('Evidence group requires explicit approval')
        alternatives = []
        for mappings in group['alternatives']:
            if not mappings or any(m.kind != 'positive' for m in mappings):
                raise ValueError('An alternative needs positive source-quote mappings')
            score = score_stage(stage, candidates, mappings)
            alternatives.append(score['gold_evidence_found'] == len(mappings))
        if not alternatives:
            raise ValueError('Empty evidence group')
        results.append({'id': group['id'], 'found': any(alternatives),
                        'matched_alternatives': [i for i, hit in enumerate(alternatives) if hit]})
    return {'groups': results, 'found': sum(r['found'] for r in results),
            'total': len(results), 'recall': sum(r['found'] for r in results) / len(results) if results else None}
