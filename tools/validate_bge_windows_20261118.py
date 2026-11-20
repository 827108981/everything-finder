"""Isolated BGE input experiment on all 111 frozen, real retrieval traces.

Only token-identical FP32 model scores can be reused. Gold is evaluated after
production ordering, applicability, evidence selection and Context assembly.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from tools.validate_vector_packaging_20261116 import inputs, load_cases
from tools.validate_selection_factorial import reconstruct, no_network, sha, write
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_bge_windows_20261118'
BASE=ROOT/'reports/rag_vector_packaging_20261116/W0_native.jsonl'
ARMS=('truncate','window_max','window_top2_mean')

def read_rows(path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines()]

def token_key(encoded):
    return hashlib.sha256(json.dumps(encoded,sort_keys=True,separators=(',',':')).encode()).hexdigest()

class CachedRanker(LocalBgeReranker):
    def _score_window_inputs(self, encodings, token):
        missing={token_key(e):e for e in encodings if token_key(e) not in self.cache}
        if missing:
            values=super()._score_window_inputs(list(missing.values()),token)
            self.cache.update(zip(missing,values,strict=True))
            self.new_pairs+=len(values)
            write(OUT/'token_scores.json',self.cache)
        self.cache_hits+=len(encodings)-len(missing)
        return [self.cache[token_key(e)] for e in encodings]

def finish(row, plan, items, arm, mappings, settings):
    rerank=[candidate_snapshot(c,selected=True,reason='进入条件检查') for c in items]
    constrained=ConstraintMatcher().evaluate(plan,items)
    conditions=[candidate_snapshot(c,selected=c.constraint_decision.accepted,reason=c.constraint_decision.reason) for c in constrained]
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=None)
    decision=policy.evaluate(RagQuery(row['question'],final_top_k=settings.rag_final_top_k),constrained,plan=plan)
    accepted={x.chunk.stable_chunk_id:i for i,x in enumerate(decision.accepted,1)}
    reasons=dict(decision.rejected)
    evidence=[{**candidate_snapshot(c,selected=c.chunk.stable_chunk_id in accepted,
                reason='通过证据门禁' if c.chunk.stable_chunk_id in accepted else reasons.get(c.chunk.stable_chunk_id,'未进入最终 TopK')),
                'final_rank':accepted.get(c.chunk.stable_chunk_id)}
              for c in [*decision.accepted,*(x for x in constrained if x.chunk.stable_chunk_id not in accepted)]]
    ctx=ContextBuilder().build(decision.accepted,max_chars=settings.rag_context_max_chars)
    stages={**{s:row['candidate_stages'][s] for s in ('lexical','vector','fusion','deduplicated')},
            'rerank':rerank,'constraints':conditions,'evidence':evidence,'context':list(ctx.entries)}
    scores={**row['scores'],**{s:score_stage(s,stages[s],mappings) for s in ('rerank','constraints','evidence','context')}}
    old=row['scores']['context'];now=scores['context']
    return {**{k:row[k] for k in ('id','group_id','question','review_status','cohort')},
            'arm':arm,'query_plan':asdict(plan),'candidate_stages':stages,'scores':scores,
            'context':ctx.text,'entries':list(ctx.entries),'excluded_entries':list(ctx.excluded_entries),
            'decision':asdict(decision)|{'accepted':[x.chunk.stable_chunk_id for x in decision.accepted]},
            'gained_gold':sorted(set(now['gold_completion_ranks'])-set(old['gold_completion_ranks'])),
            'lost_gold':sorted(set(old['gold_completion_ranks'])-set(now['gold_completion_ranks'])),
            'added_hard_negative':sorted(set(now['hard_negative_ids'])-set(old['hard_negative_ids'])),
            'removed_hard_negative':sorted(set(old['hard_negative_ids'])-set(now['hard_negative_ids'])),
            'context_changed':ctx.text!=row['context'],
            'api_calls':0,'answer_status':'not_run_no_api'}

def run():
    OUT.mkdir(exist_ok=True)
    store,bundle,settings=inputs();cases=load_cases(True);rows=read_rows(BASE)
    assert len(rows)==111 and [c.case_id for c in cases]==[r['id'] for r in rows]
    protected=json.loads((ROOT/'reports/rag_vector_packaging_20261116/protocol.json').read_text(encoding='utf8'))['protected']
    protocol=dict(baseline_sha256=sha(BASE),bundle=bundle.bundle_id,model=settings.rag_reranker_repo_id,
        revision=settings.rag_reranker_revision,max_length=settings.rag_reranker_max_length,overlap=64,
        arms=ARMS,entry_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k,
        context_max_chars=settings.rag_context_max_chars,min_score=settings.rag_min_rerank_score,
        cases=111,method='Frozen W0 upstream candidates; actual local BGE on new token windows; current production downstream. No full retrieval rerun.',
        protected=protected)
    if (OUT/'protocol.json').exists():assert json.loads((OUT/'protocol.json').read_text(encoding='utf8'))==json.loads(json.dumps(protocol))
    write(OUT/'protocol.json',protocol)
    torch.set_num_threads(6)
    with no_network() as attempts,store.connect_read_only(bundle.metadata_path) as conn:
        chunkrows=load_chunk_rows(conn)
        mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker'))
        ranker=CachedRanker(path,repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,
                           batch_size=4,max_length=settings.rag_reranker_max_length,input_mode='window_max')
        ranker.cache=json.loads((OUT/'token_scores.json').read_text(encoding='utf8')) if (OUT/'token_scores.json').exists() else {}
        ranker.new_pairs=ranker.cache_hits=0
        seeded=0;checks=[];sample=[]
        for row in rows:
            for c in row['candidate_stages']['rerank']:
                e=ranker.tokenizer(row['question'],c['text'],truncation=True,max_length=ranker.max_length)
                encoded={k:e[k] for k in ranker.tokenizer.model_input_names if k in e}
                pair=[c['rerank_score'],c['rerank_raw_logit']]
                assert pair[1] is not None
                ranker.cache[token_key(encoded)]=pair;seeded+=1
                if len(sample)<8:sample.append((encoded,pair))
        # Real inference validates the reused FP32 score scale on a fixed prefix.
        actual=LocalBgeReranker._score_window_inputs(ranker,[e for e,_ in sample],CancelToken())
        max_error=max(abs(x[0]-old[0]) for x,(_,old) in zip(actual,sample))
        assert max_error<1e-5,max_error
        completed={a:read_rows(OUT/(a+'_native.jsonl')) if (OUT/(a+'_native.jsonl')).exists() else [] for a in ARMS}
        for a,rr in completed.items():assert [x['id'] for x in rr]==[x['id'] for x in rows[:len(rr)]]
        for n,row in enumerate(rows,1):
            plan,original=reconstruct(row,store,conn)
            for arm in ARMS:
                if n<=len(completed[arm]):continue
                items=deepcopy(original)
                for x in items:x.constraint_decision=None;x.constraint_score=0;x.rerank_diagnostics={}
                if arm=='truncate':ordered=ranker._order_scored(row['question'],items,CancelToken())
                else:
                    ranker.input_mode=arm
                    ordered=ranker.rerank(row['question'],items,settings.rag_reranker_top_k,CancelToken())
                record=finish(row,plan,ordered,arm,mappings[row['id']],settings)
                if arm=='truncate':
                    assert record['context']==row['context'],(row['id'],'baseline context')
                    assert record['entries']==row['entries'],(row['id'],'baseline entries')
                    for stage in ('rerank','constraints','evidence','context'):
                        assert record['scores'][stage]==row['scores'][stage],(row['id'],stage)
                with (OUT/(arm+'_native.jsonl')).open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
                completed[arm].append(record)
            print('CASE',n,row['id'],'new_model_windows',ranker.new_pairs,flush=True)
        drift=[p for p,h in protected.items() if sha(p)!=h]
        assert not attempts and not drift
        write(OUT/'execution_audit.json',dict(cases=111,arms=ARMS,api_calls=0,network_attempts=attempts,
            protected_drift=drift,baseline_context_entries_scores_equal=True,seeded_original_pairs=seeded,
            new_windows_this_execution=ranker.new_pairs,cache_hits=ranker.cache_hits,
            fixed_prefix_inference_pairs=len(sample),max_probability_error=max_error))
    summarize()

def summarize():
    result={}
    for arm in ARMS:
        rows=read_rows(OUT/(arm+'_native.jsonl'));assert len(rows)==111
        result[arm]={}
        for cohort in ('formal','diagnostic','source_holdout'):
            rr=[r for r in rows if r['cohort']==cohort and r['scores']['context']['gold_evidence_total_declared']]
            stats={}
            for stage in ('rerank','constraints','evidence','context'):
                ss=[r['scores'][stage] for r in rr]
                stats[stage]=dict(positive_cases=len(rr),complete=sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in ss),
                    mrr=sum(s['reciprocal_rank'] for s in ss)/len(ss),negative_quotes=sum(s['hard_negative_found'] for s in ss),
                    **{f'hit_at_{k}':sum(bool(s['first_gold_rank'] and s['first_gold_rank']<=k) for s in ss) for k in (1,3,5)})
            result[arm][cohort]=dict(stages=stats,gains=[r['id'] for r in rr if r['gained_gold']],losses=[r['id'] for r in rr if r['lost_gold']],
                new_negatives=[r['id'] for r in rr if r['added_hard_negative']],changed=[r['id'] for r in rr if r['context_changed']])
    write(OUT/'metrics.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

def native_probe(*, use_default=False, expected_path=None):
    """Check four actual AnswerService.retrieve entries, not just replay wiring."""
    from unittest.mock import patch
    from tools.validate_vector_packaging_20261116 import model
    from tools.validate_query_plan_adapter import STAGES
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from local_full_text_search.rag.vector_index_store import VectorIndexStore
    from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
    store,bundle,settings=inputs()
    index=VectorIndexStore(store.root).active(bundle.bundle_id,embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision,embedding_dimension=bundle.embedding_dimension,
        policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    expected={r['id']:r for r in read_rows(expected_path or OUT/'window_max_native.jsonl')}
    wanted=['rag-biz-004','rag-biz-019','rag-row-holdout-008','vector-source-independent-010']
    class Forbidden:
        def retrieve(self,*a,**k):raise AssertionError('Credentials forbidden')
        def generate(self,*a,**k):raise AssertionError('API forbidden')
    class BoundStore:
        def __init__(self,*a,**k):pass
        def attached(self,source_bundle_id,*a,**k):
            assert source_bundle_id==index.source_bundle_id
            return index
        active=attached
    output=[];checks=[]
    with no_network() as attempts,patch('local_full_text_search.rag.answer_service.VectorIndexStore',BoundStore):
        emb=model(bundle)
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker'))
        ranker=LocalBgeReranker(path,repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,
            batch_size=4,max_length=settings.rag_reranker_max_length,**({} if use_default else {'input_mode':'window_max'}))
        assert ranker.input_mode=='window_max'
        for cid in wanted:
            e=expected[cid];events=[];stages={}
            def progress(event):
                events.append(event)
                for key,stage in STAGES.items():
                    if key in event:stages[stage]=event[key]
            service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,emb,ranker,
                Forbidden(),Forbidden(),settings,progress_callback=progress)
            service.retrieve(RagQuery(e['question'],include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=50,vector_top_k=50,fusion_top_k=30,reranker_top_k=20,final_top_k=8),CancelToken())
            ctx=next(v for v in reversed(events) if 'llm_context' in v)
            check=dict(id=cid,context_equal=ctx['llm_context']==e['context'],entries_equal=ctx['context_entries']==e['entries'],
                ranks_equal=all([c['chunk_id'] for c in stages[s]]==[c['chunk_id'] for c in e['candidate_stages'][s]] for s in ('lexical','vector','fusion','deduplicated','rerank','constraints')))
            checks.append(check);output.append(dict(id=cid,events=events,context=ctx['llm_context'],api_calls=0))
            print('NATIVE_PROBE',check,flush=True)
    (OUT/'entry_probes.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in output),encoding='utf8')
    write(OUT/'entry_probe_audit.json',dict(checks=checks,cases=4,real_model_inference=True,cache_reuse=False,api_calls=0,network_attempts=attempts,constructor_default_used=use_default,
        vector_dependency='explicit read-only active W0 index handle; no attached pointer modification'))
    assert all(all(v for k,v in r.items() if k!='id') for r in checks) and not attempts

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summary',action='store_true');parser.add_argument('--native',action='store_true');args=parser.parse_args()
    native_probe() if args.native else summarize() if args.summary else run()
