"""Frozen downstream factorial and real pre-API validation. Never calls an API."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from tools import validate_bge_windows_20261118 as exp
from tools.validate_selection_factorial import reconstruct, no_network, write, sha
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence

ROOT = exp.ROOT
OUT = ROOT/'reports/rag_condition_check_20261118'
BASE = exp.OUT/'window_max_native.jsonl'
ARMS = {'baseline': (False, 'legacy'), 'A': (True, 'legacy'),
        'B': (False, 'source_bound'), 'AB': (True, 'source_bound'),
        'ABC': (True, 'source_bound'), 'AC': (True, 'legacy')}


def check_frozen():
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    drift = [p for p, h in protocol['protected'].items() if sha(p) != h]
    assert not drift, drift
    return protocol


def replay(arms):
    check_frozen()
    store, bundle, settings = exp.inputs()
    rows = exp.read_rows(BASE)
    assert len(rows) == 111
    checks = []
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        chunkrows = load_chunk_rows(conn)
        mappings = {c.case_id: map_case_evidence(c, chunkrows) for c in exp.load_cases(True)}
        for arm in arms:
            preserve, mode = ARMS[arm]
            output = []
            for row in rows:
                plan, items = reconstruct(row, store, conn)
                byid = {x.chunk.stable_chunk_id: x for x in items}
                items = [byid[c['chunk_id']] for c in row['candidate_stages']['rerank']]
                for item, saved in zip(items, row['candidate_stages']['rerank'], strict=True):
                    item.constraint_decision = None
                    item.constraint_score = saved.get('constraint_score')
                    item.rerank_diagnostics = deepcopy(saved.get('rerank_diagnostics', {}))
                    item.rerank_raw_logit = saved.get('rerank_raw_logit')
                matcher = ConstraintMatcher(preserve_order=preserve, judgement_mode=mode)
                policy_class = exp.EvidencePolicy
                def make_policy(**kwargs):
                    return policy_class(**kwargs, require_confirmed_conditions=arm not in {'AC','ABC'})
                with patch.object(exp, 'ConstraintMatcher', lambda: matcher), patch.object(exp, 'EvidencePolicy', make_policy):
                    record = exp.finish(row, plan, items, arm, mappings[row['id']], settings)
                # Snapshot scores and all upstream content must be frozen.
                assert record['candidate_stages']['rerank'] == row['candidate_stages']['rerank'], row['id']
                if arm == 'baseline':
                    check = dict(id=row['id'], context_equal=record['context'] == row['context'],
                                 entries_equal=record['entries'] == row['entries'], scores_equal=record['scores'] == row['scores'])
                    checks.append(check)
                    assert all(v for k, v in check.items() if k != 'id'), check
                output.append(record)
            (OUT/(arm+'_native.jsonl')).write_text(''.join(json.dumps(r, ensure_ascii=False, default=str)+'\n' for r in output), encoding='utf8')
            print('REPLAY', arm, len(output), flush=True)
    assert not attempts
    if checks:
        write(OUT/'baseline_reproduction.json', dict(cases=len(checks), checks=checks, api_calls=0, network_attempts=attempts))
    check_frozen()
    summarize()


def summarize():
    result = {}
    for arm in ARMS:
        path = OUT/(arm+'_native.jsonl')
        if not path.exists():
            continue
        rows = exp.read_rows(path)
        result[arm] = {}
        for cohort in ('formal', 'diagnostic', 'source_holdout'):
            rr = [r for r in rows if r['cohort'] == cohort and r['scores']['context']['gold_evidence_total_declared']]
            stats = {}
            for stage in ('rerank', 'constraints', 'evidence', 'context'):
                ss = [r['scores'][stage] for r in rr]
                stats[stage] = dict(positive_cases=len(rr), complete=sum(s['gold_evidence_found'] == s['gold_evidence_total_declared'] for s in ss),
                    mrr=sum(s['reciprocal_rank'] for s in ss)/len(ss), negative_quotes=sum(s['hard_negative_found'] for s in ss),
                    **{f'hit_at_{k}': sum(bool(s['first_gold_rank'] and s['first_gold_rank'] <= k) for s in ss) for k in (1, 3, 5)})
            result[arm][cohort] = dict(stages=stats, gains=[r['id'] for r in rr if r['gained_gold']],
                losses=[r['id'] for r in rr if r['lost_gold']], new_negatives=[r['id'] for r in rr if r['added_hard_negative']],
                changed=[r['id'] for r in rr if r['context_changed']],
                cleared=[r['id'] for r in rr if str(r['decision']['reason']).startswith('required_constraint_not_confirmed')])
    write(OUT/'metrics.json', result)
    for a, cohorts in result.items():
        print(a, json.dumps({c:{'context':m['stages']['context'], 'gains':m['gains'], 'losses':m['losses'], 'negatives':m['new_negatives'], 'cleared':m['cleared']} for c,m in cohorts.items()},ensure_ascii=False),flush=True)


def native(arm='A'):
    """All 111 real query/embedding/Chroma/BGE calls, no score-cache reuse."""
    import torch
    from tools.validate_vector_packaging_20261116 import model
    from tools.validate_query_plan_adapter import STAGES
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from local_full_text_search.rag.vector_index_store import VectorIndexStore
    from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
    from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
    from local_full_text_search.rag.models import RagQuery
    from local_full_text_search.core.task_manager import CancelToken
    check_frozen()
    torch.set_num_threads(6)
    store,bundle,settings=exp.inputs()
    index=VectorIndexStore(store.root).active(bundle.bundle_id,embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision,embedding_dimension=bundle.embedding_dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    expected=exp.read_rows(OUT/(arm+'_native.jsonl'))
    checks=[]
    class Forbidden:
        def retrieve(self,*a,**k):raise AssertionError('Credentials forbidden')
        def generate(self,*a,**k):raise AssertionError('API forbidden')
    class BoundStore:
        def __init__(self,*a,**k):pass
        def attached(self,source_bundle_id,*a,**k):
            assert source_bundle_id==index.source_bundle_id
            return index
        active=attached
    output=OUT/('real_entry_'+arm+'.jsonl')
    if output.exists():
        raise ValueError('Do not overwrite completed or partial real entry results')
    with no_network() as attempts,patch('local_full_text_search.rag.answer_service.VectorIndexStore',BoundStore):
        emb=model(bundle)
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker'))
        ranker=LocalBgeReranker(path,repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,
            batch_size=4,max_length=settings.rag_reranker_max_length)
        for e in expected:
            events=[];stages={}
            def progress(event):
                events.append(event)
                for key,stage in STAGES.items():
                    if key in event:stages[stage]=event[key]
            preserve,mode=ARMS[arm]
            service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,emb,ranker,
                Forbidden(),Forbidden(),settings,progress_callback=progress,
                constraint_matcher=ConstraintMatcher(preserve_order=preserve,judgement_mode=mode))
            result=service.retrieve(RagQuery(e['question'],include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=50,vector_top_k=50,fusion_top_k=30,reranker_top_k=20,final_top_k=8),CancelToken())
            ctx=next((v for v in reversed(events) if 'llm_context' in v),{})
            check=dict(id=e['id'],context_equal=ctx.get('llm_context','')==e['context'],
                entries_equal=ctx.get('context_entries',[])==e['entries'],
                upstream_orders_equal=all([c['chunk_id'] for c in stages.get(s,[])]==[c['chunk_id'] for c in e['candidate_stages'][s]]
                    for s in ('lexical','vector','fusion','deduplicated','rerank')),
                condition_order_equal=[c['chunk_id'] for c in stages.get('constraints',[])]==[c['chunk_id'] for c in e['candidate_stages']['constraints']])
            checks.append(check)
            with output.open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(id=e['id'],events=events,context=ctx.get('llm_context',''),entries=ctx.get('context_entries',[]),
                    result=asdict(result),checks=check,api_calls=0),ensure_ascii=False,default=str)+'\n')
            print('REAL_ENTRY',len(checks),check,flush=True)
    check_frozen()
    write(OUT/('real_entry_'+arm+'_audit.json'),dict(cases=len(checks),checks=checks,
        api_calls=0,network_attempts=attempts,score_cache_used=False,real_model_inference=True,
        vector_dependency='read-only active index handle; no attached-pointer or index modification'))
    assert not attempts and all(all(v for k,v in c.items() if k!='id') for c in checks)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--arms', nargs='+', choices=tuple(ARMS), default=['baseline', 'A'])
    p.add_argument('--native', choices=tuple(ARMS))
    args = p.parse_args()
    native(args.native) if args.native else replay(args.arms)
