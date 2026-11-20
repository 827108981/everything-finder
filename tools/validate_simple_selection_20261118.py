"""All active suites: frozen selection controls plus real no-API runtime run."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from tools import validate_bge_windows_20261118 as exp
from tools.validate_selection_factorial import reconstruct, no_network, sha, write
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import SimpleEvidencePolicy
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.trace import candidate_snapshot

ROOT = exp.ROOT
OUT = ROOT/'reports/rag_simple_selection_20261118'
HISTORY = ROOT/'reports/rag_condition_check_20261118'
ARMS = ('baseline', 'A', 'simple_legacy_budget', 'simple')
STAGES = ('lexical', 'vector', 'fusion', 'deduplicated', 'rerank', 'constraints', 'evidence', 'context')


def check_frozen():
    p = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    drift = [f for f, h in p['protected'].items() if sha(f) != h]
    assert not drift, drift
    assert sha(HISTORY/'baseline_native.jsonl') == p['baseline_sha256']
    assert sha(HISTORY/'A_native.jsonl') == p['A_sha256']
    return p


def finish(row, plan, items, arm, mapping, settings):
    # No condition judgements or condition-derived scores survive into this arm.
    for item in items:
        item.constraint_decision = None
        item.constraint_score = None
    conditions = [candidate_snapshot(x, selected=True, reason='条件过滤已移除，保留 BGE 顺序') for x in items]
    decision = SimpleEvidencePolicy().evaluate(RagQuery(row['question']), items)
    evidence = [{**candidate_snapshot(x, selected=True, reason='待 Context 长度预算选入'), 'final_rank':i}
                for i, x in enumerate(decision.accepted, 1)]
    ctx = ContextBuilder().build(decision.accepted, max_chars=settings.rag_context_max_chars,
                                preserve_complete_chunks=arm == 'simple')
    stages = {**{s:row['candidate_stages'][s] for s in STAGES[:5]},
              'constraints':conditions, 'evidence':evidence, 'context':list(ctx.entries)}
    scores = {s:score_stage(s, stages[s], mapping) for s in STAGES}
    old, now = row['scores']['context'], scores['context']
    return {**{k:row[k] for k in ('id','group_id','question','review_status','cohort')},
            'arm':arm, 'query_plan':asdict(plan), 'candidate_stages':stages, 'scores':scores,
            'context':ctx.text, 'entries':list(ctx.entries), 'excluded_entries':list(ctx.excluded_entries),
            'decision':asdict(decision)|{'accepted':[x.chunk.stable_chunk_id for x in decision.accepted]},
            'gained_gold':sorted(set(now['gold_completion_ranks']) - set(old['gold_completion_ranks'])),
            'lost_gold':sorted(set(old['gold_completion_ranks']) - set(now['gold_completion_ranks'])),
            'added_hard_negative':sorted(set(now['hard_negative_ids']) - set(old['hard_negative_ids'])),
            'removed_hard_negative':sorted(set(old['hard_negative_ids']) - set(now['hard_negative_ids'])),
            'context_changed':ctx.text != row['context'], 'api_calls':0, 'answer_status':'not_run_no_api',
            'selection_config':dict(conditions=False, score_thresholds=False, final_top_k_applied=False,
                                   document_quota=None, whole_chunks=arm == 'simple',
                                   context_max_chars=settings.rag_context_max_chars)}


def replay():
    from unittest.mock import patch
    from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
    check_frozen()
    store, bundle, settings = exp.inputs()
    rows = exp.read_rows(HISTORY/'baseline_native.jsonl')
    cases = exp.load_cases(True)
    assert len(rows) == 111 and [c.case_id for c in cases] == [r['id'] for r in rows]
    result = {a:[] for a in ARMS}
    checks = []
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        chunkrows = load_chunk_rows(conn)
        mappings = {c.case_id:map_case_evidence(c, chunkrows) for c in cases}
        for row in rows:
            plan, reconstructed = reconstruct(row, store, conn)
            byid = {x.chunk.stable_chunk_id:x for x in reconstructed}
            items = [byid[c['chunk_id']] for c in row['candidate_stages']['rerank']]
            for item, saved in zip(items, row['candidate_stages']['rerank'], strict=True):
                item.rerank_diagnostics = deepcopy(saved.get('rerank_diagnostics', {}))
                item.rerank_raw_logit = saved.get('rerank_raw_logit')
                item.constraint_decision = None
                item.constraint_score = saved.get('constraint_score')
            # Reproduce both old controls using the retained legacy implementation.
            for arm in ('baseline', 'A'):
                with patch.object(exp, 'ConstraintMatcher', lambda:ConstraintMatcher(preserve_order=arm == 'A')):
                    r = exp.finish(row, plan, deepcopy(items), arm, mappings[row['id']], settings)
                result[arm].append(r)
                old = row if arm == 'baseline' else next(x for x in exp.read_rows(HISTORY/'A_native.jsonl') if x['id'] == row['id'])
                assert r['context'] == old['context'] and r['entries'] == old['entries'], (arm, row['id'])
                assert all(r['scores'][s] == old['scores'][s] for s in STAGES), (arm, row['id'], 'scores')
                checks.append(dict(id=row['id'], arm=arm, context_entries_scores_equal=True))
            for arm in ('simple_legacy_budget', 'simple'):
                result[arm].append(finish(row, plan, deepcopy(items), arm, mappings[row['id']], settings))
    for arm, records in result.items():
        (OUT/(arm+'_native.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, default=str)+'\n' for r in records), encoding='utf8')
    check_frozen()
    write(OUT/'replay_audit.json', dict(cases=111, arms=ARMS, control_checks=checks,
          api_calls=0, network_attempts=attempts, fixed_upstream_stages=list(STAGES[:5])))
    summarize()


def summarize():
    allrows = {a:exp.read_rows(OUT/(a+'_native.jsonl')) for a in ARMS}
    result = {}
    for arm, rows in allrows.items():
        assert len(rows) == 111
        result[arm] = {}
        for cohort in ('formal', 'diagnostic', 'source_holdout'):
            group = [r for r in rows if r['cohort'] == cohort]
            rr = [r for r in group if r['scores']['context']['gold_evidence_total_declared']]
            stages = {}
            for stage in STAGES:
                ss = [r['scores'][stage] for r in rr]
                stages[stage] = dict(positive_cases=len(rr), complete=sum(s['gold_evidence_found'] == s['gold_evidence_total_declared'] for s in ss),
                    mrr=sum(s['reciprocal_rank'] for s in ss)/max(1, len(ss)), negative_quotes=sum(s['hard_negative_found'] for s in ss),
                    **{f'hit_at_{k}':sum(bool(s['first_gold_rank'] and s['first_gold_rank'] <= k) for s in ss) for k in (1,3,5)})
            result[arm][cohort] = dict(cases=len(group), stages=stages,
                gains=[r['id'] for r in group if r['gained_gold']], losses=[r['id'] for r in group if r['lost_gold']],
                new_negatives=[r['id'] for r in group if r['added_hard_negative']],
                changed=[r['id'] for r in group if r['context_changed']],
                no_gold=[dict(id=r['id'], context_entries=len(r['entries']), chars=len(r['context']), decision=r['decision']['reason'])
                         for r in group if not r['scores']['context']['gold_evidence_total_declared']])
    write(OUT/'metrics.json', result)
    baseline = {r['id']:r for r in allrows['baseline']}
    audit = {}
    for arm, rows in allrows.items():
        changes = []
        for r in rows:
            old = baseline[r['id']]
            before = {e['chunk_id']:e for e in old['entries']}
            after = {e['chunk_id']:e for e in r['entries']}
            changes.append(dict(id=r['id'], question=r['question'], cohort=r['cohort'], review_status=r['review_status'],
                gained_gold=r['gained_gold'], lost_gold=r['lost_gold'], new_negatives=r['added_hard_negative'],
                added=[e for cid,e in after.items() if cid not in before],
                removed=[e for cid,e in before.items() if cid not in after],
                changed=[dict(before=before[cid], after=e) for cid,e in after.items() if cid in before and before[cid] != e],
                context_chars=len(r['context']), context_entries=len(r['entries']), exclusions=r['excluded_entries']))
        audit[arm] = changes
    write(OUT/'selection_audit.json', audit)
    for arm, cohorts in result.items():
        print(arm, json.dumps({c:dict(context=m['stages']['context'], gains=m['gains'], losses=m['losses'], new_negatives=m['new_negatives'], no_gold=m['no_gold']) for c,m in cohorts.items()}, ensure_ascii=False), flush=True)


def native():
    """Real local inference for all 111, default policy, no generation or secrets."""
    import torch
    from unittest.mock import patch
    from tools.validate_vector_packaging_20261116 import model
    from tools.validate_query_plan_adapter import STAGES as EVENT_STAGES
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from local_full_text_search.rag.vector_index_store import VectorIndexStore
    from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
    from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
    from local_full_text_search.core.task_manager import CancelToken
    check_frozen()
    torch.set_num_threads(6)
    store, bundle, settings = exp.inputs()
    index = VectorIndexStore(store.root).active(bundle.bundle_id, embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision, embedding_dimension=bundle.embedding_dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    expected = exp.read_rows(OUT/'simple_native.jsonl')
    output = OUT/'real_entry_simple.jsonl'
    if output.exists():
        raise ValueError('Refusing to overwrite real run; preserve it and use a new report directory')
    class Forbidden:
        def __getattr__(self, name):
            raise AssertionError('API/credentials forbidden: '+name)
    class BoundStore:
        def __init__(self, *args, **kwargs): pass
        def attached(self, source_bundle_id, *args, **kwargs):
            assert source_bundle_id == index.source_bundle_id
            return index
        active = attached
    checks = []
    with no_network() as attempts, patch('local_full_text_search.rag.answer_service.VectorIndexStore', BoundStore):
        emb = model(bundle)
        path = LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id, settings.rag_reranker_revision, 'reranker'))
        ranker = LocalBgeReranker(path, repo_id=settings.rag_reranker_repo_id, revision=settings.rag_reranker_revision,
                                  batch_size=4, max_length=settings.rag_reranker_max_length)
        assert ranker.input_mode == 'window_max'
        for e in expected:
            events, stages = [], {}
            def progress(event):
                events.append(event)
                for field, stage in EVENT_STAGES.items():
                    if field in event: stages[stage] = event[field]
            service = AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')), store,
                emb, ranker, Forbidden(), Forbidden(), settings, progress_callback=progress)
            result = service.retrieve(RagQuery(e['question'], include_ocr=True, min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=50, vector_top_k=50, fusion_top_k=30, reranker_top_k=20, final_top_k=8), CancelToken())
            ctx = next((v for v in reversed(events) if 'llm_context' in v), {})
            checks.append(dict(id=e['id'], context_equal=ctx.get('llm_context', '') == e['context'],
                entries_equal=ctx.get('context_entries', []) == e['entries'],
                upstream_orders_equal=all([c['chunk_id'] for c in stages.get(s, [])] == [c['chunk_id'] for c in e['candidate_stages'][s]] for s in STAGES[:5]),
                condition_check_disabled=all(not ev.get('condition_check_enabled', False) for ev in events if ev['stage'] == 'checking_constraints')))
            with output.open('a', encoding='utf8') as f:
                f.write(json.dumps(dict(id=e['id'], events=events, context=ctx.get('llm_context', ''), entries=ctx.get('context_entries', []),
                     result=asdict(result), checks=checks[-1], api_calls=0), ensure_ascii=False, default=str)+'\n')
            print('REAL_ENTRY', len(checks), checks[-1], flush=True)
    check_frozen()
    write(OUT/'real_entry_audit.json', dict(cases=len(checks), checks=checks, api_calls=0, network_attempts=attempts,
        real_model_inference=True, score_cache_used=False, vector_dependency='read-only published W0 index handle; no pointer changes'))
    assert not attempts and all(all(v for k,v in c.items() if k != 'id') for c in checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--native', action='store_true')
    p.add_argument('--summary', action='store_true')
    args = p.parse_args()
    native() if args.native else summarize() if args.summary else replay()
