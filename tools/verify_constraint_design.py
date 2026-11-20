"""Isolated, frozen-candidate validation of the proposed condition contract.

No production mutation, model inference, answer generation, or Gold mutation.
The local/joint checks are conservative diagnostic prototypes, not entailment.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from local_full_text_search.rag.models import (
    SourceRef, RagChunk, EvidenceChunk, RagQuery, ConstraintKind,
    ConstraintStatus, ConstraintJudgement,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.clarification import ClarificationPolicy
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.index_store import RagIndexStore
from tools.verify_rerank_scope_bonus import mappings

ROOT = Path('reports/constraint_design_20261102_v2')
SOURCE = Path('reports/rerank_dev_holdout_20261101')
ARMS = ('baseline', 'generic_only', 'dedup_only', 'generic_dedup',
        'branch_only', 'local_only', 'branch_local', 'joint_gate_diagnostic')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def normalize_plan(text, *, neutral=True, dedup=True):
    plan = QueryInterpreter().interpret(text)
    unique = {}
    for c in plan.constraints:
        # Only the semantically empty generic noun is neutralized here. Sample,
        # reagent and named entities retain their original applicability checks.
        if neutral and c.kind is ConstraintKind.OBJECT and c.value == '项目':
            continue
        key = (c.kind, c.value, c.required, c.polarity, c.strength)
        unique.setdefault(key if dedup else (len(unique), *key), c)
    return replace(plan, constraints=tuple(unique.values()))


def branches(text):
    # Deliberately limited parser: explicit sentence-boundary hypothetical only.
    # No inherited subject/model is guessed. Its limitations are reported.
    parts = re.split(r'(?<=[？?。；;])\s*(?=(?:如果|假如|若))', text)
    return tuple(normalize_plan(p) for p in parts if p.strip())


class LocalMatcher(ConstraintMatcher):
    def _judge(self, constraint, text, body='', filenames='', lexical_terms=()):
        atoms = [s.strip() for s in re.split(r'[。；;\n]', body) if s.strip()]
        judged = [super(LocalMatcher, self)._judge(constraint, a, a, '', lexical_terms)
                  for a in atoms]
        states = {j.status for j in judged}
        # Mixed applicability cannot be resolved without the row/subject relation.
        if ConstraintStatus.MATCH in states and ConstraintStatus.CONFLICT not in states:
            return next(j for j in judged if j.status is ConstraintStatus.MATCH)
        if states == {ConstraintStatus.CONFLICT}:
            return judged[0]
        return ConstraintJudgement(constraint, ConstraintStatus.UNKNOWN,
                                   '局部条件不确定或存在多个场景，不能推断适用', '')


def make_candidate(c):
    source = SourceRef(1, c['chunk_id'], None, 1, 1, c['file_path'],
                       c['filename'], Path(c['file_path']).suffix, (), c['location'],
                       page_number=c.get('page_number'), slide_number=c.get('slide_number'))
    refs = tuple(replace(source, stable_chunk_id=s['chunk_id'],
                         filename=s['filename'], location_text=s['location'])
                 for s in c.get('sources', []))
    chunk = RagChunk(c['chunk_id'], c['document_key'], 0, c['text'], c['text'],
                     hashlib.sha256(c['text'].encode()).hexdigest(), source)
    return EvidenceChunk(chunk, lexical_rank=c.get('lexical_rank'),
                         vector_score=c.get('vector_score'),
                         fused_score=c.get('fused_score', 0),
                         rerank_score=c['rerank_score'],
                         rerank_score_combined=c['rerank_score_combined'],
                         sources=refs, member_chunk_ids=tuple(c.get('member_chunk_ids', [])))


def joint_supported(plan, item):
    """Strict locality proxy. Intentionally cannot authorize cross-chunk joins."""
    required = [c for c in plan.constraints if c.is_hard and c.kind.value in
                {'scope', 'stage', 'direction', 'device', 'topic'}]
    if not required:
        return True
    for atom in re.split(r'[。；;\n]', item.chunk.text):
        if atom.strip() and all(ConstraintMatcher()._judge(c, atom, atom, '', plan.lexical_terms).status
                                is ConstraintStatus.MATCH for c in required):
            return True
    return False


def evaluate(text, candidates, arm):
    plans = ((QueryInterpreter().interpret(text),) if arm == 'baseline' else
             (normalize_plan(text, dedup=False),) if arm == 'generic_only' else
             (normalize_plan(text, neutral=False),) if arm == 'dedup_only' else
             branches(text) if arm in ('branch_only','branch_local','joint_gate_diagnostic') else
             (normalize_plan(text),))
    outputs, diagnostics = [], []
    for plan in plans:
        items = copy.deepcopy(candidates)
        matcher = LocalMatcher() if arm in ('local_only','branch_local','joint_gate_diagnostic') else ConstraintMatcher()
        ordered = matcher.evaluate(plan, items)
        competing = ClarificationPolicy().evaluate_competing_stages(plan, ordered)
        decision = EvidencePolicy().evaluate(RagQuery(text=text, final_top_k=8), ordered, plan=plan)
        selected = list(decision.accepted) if not (competing and competing.required) else []
        rejected_joint = []
        if arm == 'joint_gate_diagnostic':
            rejected_joint = [c.chunk.stable_chunk_id for c in selected if not joint_supported(plan, c)]
            selected = [c for c in selected if c.chunk.stable_chunk_id not in rejected_joint]
        outputs.append(selected)
        diagnostics.append({'question': plan.original_text,
            'constraints': [(c.kind.value, c.value, c.source_text) for c in plan.constraints],
            'reason': decision.reason, 'clarify': bool(competing and competing.required),
            'joint_rejected': rejected_joint,
            'selected': [c.chunk.stable_chunk_id for c in selected],
            'ordered': [candidate_snapshot(c) for c in ordered]})
    # Fixed total budget, round-robin between explicit branches. Never 8 per branch.
    merged, seen = [], set()
    for rank in range(8):
        for output in outputs:
            if rank < len(output) and output[rank].chunk.stable_chunk_id not in seen:
                item = output[rank]
                merged.append(item)
                seen.add(item.chunk.stable_chunk_id)
            if len(merged) == 8:
                break
        if len(merged) == 8:
            break
    context = ContextBuilder().build(merged, max_chars=12000)
    by_id = {candidate_snapshot(c)['chunk_id']: candidate_snapshot(c) for c in merged}
    visible = [{**by_id[e['chunk_id']], 'text': e['included_text']} for e in context.entries]
    return visible, diagnostics


def controls():
    # Synthetic logical counterexamples; not clinical truth or new reviewed Gold.
    fixtures = [
        ('generic_absence', '这个项目需要进行哪些处理步骤？', ['先复测两次。', '项目的背景介绍。'], ['c0']),
        ('negation', '发光值不是偏低，而是偏高', ['发光值偏低，应查底物。', '发光值偏高，应查曲线。'], ['c1']),
        ('two_branches', '单个项目发光值偏低怎么办？如果所有项目偏低呢？',
         ['单个项目发光值偏低，应检查甲。', '所有项目发光值偏低，应检查乙。'], ['c0', 'c1']),
        ('hypothetical_leak', '为什么磁分离泵不转？如果所有项目发光值偏低呢？',
         ['磁分离泵不转，应检查甲。', '所有项目发光值偏低，应检查乙。'], ['c0','c1']),
        ('mixed_rows', '所有项目发光值偏低怎么办？',
         ['单个项目发光值偏低，应检查甲。\n所有项目发光值偏高，应检查乙。'], []),
        ('disconnected_chunks', '所有项目定标偏低怎么办？',
         ['所有项目的概述。', '定标偏低，应检查甲。'], []),
        ('coherent', '所有项目定标偏低怎么办？', ['所有项目定标偏低，应检查甲。'], ['c0']),
        ('unknown', '所有项目定标偏低怎么办？', ['应检查甲。'], []),
        ('distinct_measurements_same_sentence', '所有项目发光值偏低怎么办？',
         ['所有项目温度偏低，同时单个项目发光值偏高，应检查乙。'], []),
        ('header_inheritance', '所有项目定标偏低怎么办？',
         ['所有项目\n定标\n发光值偏低：应检查甲。'], ['c0']),
        ('equivalent_phrasing', '所有项目校准偏低怎么办？', ['所有项目定标偏低，应检查甲。'], ['c0']),
    ]
    results = []
    for name, q, texts, expected in fixtures:
        candidates = [make_candidate({'chunk_id': f'c{i}', 'document_key': str(i),
            'file_path': f'E:/synthetic/c{i}.txt', 'filename': f'c{i}.txt', 'location': 'line 1',
            'text': t, 'lexical_rank': 1, 'vector_score': .8,
            'rerank_score': .9-i*.01, 'rerank_score_combined': .9-i*.01}) for i,t in enumerate(texts)]
        outcomes = {}
        for arm in ARMS:
            visible, diag = evaluate(q, candidates, arm)
            ids = [c['chunk_id'] for c in visible]
            if name == 'generic_absence':
                ids = ids[:1]  # This control tests order, not complete top8 membership.
            outcomes[arm] = {'actual': ids, 'passed': ids == expected, 'diagnostics': diag}
        results.append({'name': name, 'question': q, 'texts': texts,
                        'expected': expected, 'arms': outcomes})
    return results


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    active = RagIndexStore(Path('E:/everythingfinder-db/rag')).active()
    assert active.bundle_id == protocol['bundle_id']
    gold = {}
    for ds in protocol['datasets'].values():
        assert sha(Path(ds['path'])) == ds['sha256']
        gold.update({c['id']: c for c in map(json.loads, Path(ds['path']).read_text(encoding='utf-8').splitlines())})
    dump(ROOT/'protocol.json', {'source': str(SOURCE), 'source_sha256': sha(SOURCE/'cases.jsonl'),
        'bundle_id': active.bundle_id, 'datasets': protocol['datasets'], 'arms': ARMS,
        'script_sha256': sha(Path(__file__)), 'budget': {'topk':8, 'chars':12000, 'per_document':3},
        'limitations': ['Frozen candidate replay, no new retrieval/model inference/answers.',
            'Local sentence and joint gate are diagnostic proxies, not semantic entailment.',
            'Known 50 cases and previously inspected pending 10; not independent acceptance.',
            'Synthetic controls verify logic only, not clinical facts.']})
    rows = [r for r in map(json.loads, (SOURCE/'cases.jsonl').read_text(encoding='utf-8').splitlines()) if r['arm']=='pure']
    with sqlite3.connect(f'{active.metadata_path.as_uri()}?mode=ro', uri=True) as db:
        indexed = dict(db.execute('SELECT stable_chunk_id, text FROM rag_chunks'))
    verified_pairs = 0
    for row in rows:
        for c in row['candidate_stages']['rerank']:
            assert indexed[c['chunk_id']] == c['text']
            verified_pairs += 1
    results = []
    for r in rows:
        candidates = [make_candidate(c) for c in r['candidate_stages']['rerank']]
        maps = mappings(r, gold[r['id']])
        outcomes = {}
        for arm in ARMS:
            visible, diag = evaluate(r['question'], candidates, arm)
            score = score_stage('context', visible, maps)
            outcomes[arm] = {'score': score, 'context': visible, 'diagnostics': diag}
        results.append({'id':r['id'], 'question':r['question'], 'group_id':r['group_id'],
                        'cohort':r['cohort'], 'expected_behavior':r['expected_behavior'],
                        'arms':outcomes,
                        'baseline_reproduced': outcomes['baseline']['score'] == r['stages']['context']})
    assert len(results) == 60
    dump(ROOT/'cases.json', results)
    baseline_failures = [r['id'] for r in results if not r['baseline_reproduced']]
    assert not baseline_failures, baseline_failures
    summary = {}
    for cohort in ('development', 'original18', 'added32', 'holdout_pending'):
        selected = [r for r in results if r['expected_behavior'] != 'refuse' and
                    (r['cohort'] == cohort if cohort in ('development','holdout_pending') else
                     r['cohort']=='development' and ((int(r['id'][-3:])<=18)==(cohort=='original18')))]
        summary[cohort] = {}
        for arm in ARMS:
            scores = [r['arms'][arm]['score'] for r in selected]
            groups = defaultdict(list)
            for r in selected:
                groups[r['group_id']].append(r['arms'][arm]['score']['gold_evidence_recall'])
            summary[cohort][arm] = {'n':len(scores), 'hit':sum(s['gold_hit'] for s in scores),
                'coverage':sum(s['gold_evidence_recall'] for s in scores)/len(scores),
                'negatives':sum(s['hard_negative_found'] for s in scores),
                'group_coverage':sum(sum(v)/len(v) for v in groups.values())/len(groups),
                'regressions':[r['id'] for r in selected if r['arms'][arm]['score']['gold_evidence_recall']
                               < r['arms']['baseline']['score']['gold_evidence_recall']],
                'gains':[r['id'] for r in selected if r['arms'][arm]['score']['gold_evidence_recall']
                         > r['arms']['baseline']['score']['gold_evidence_recall']]}
    checks = controls()
    dump(ROOT/'controls.json', checks)
    for ds in protocol['datasets'].values():
        assert sha(Path(ds['path'])) == ds['sha256']
    assert RagIndexStore(Path('E:/everythingfinder-db/rag')).active().bundle_id == active.bundle_id
    dump(ROOT/'summary.json', {'baseline_reproduction':60, 'indexed_body_pairs_verified':verified_pairs,
        'datasets_unchanged':True,
        'summary':summary, 'controls':{a: {'passed':sum(c['arms'][a]['passed'] for c in checks),
        'total':len(checks), 'failures':[c['name'] for c in checks if not c['arms'][a]['passed']]} for a in ARMS}})
    print(json.dumps({'summary':summary,'controls':{a:sum(c['arms'][a]['passed'] for c in checks) for a in ARMS}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
