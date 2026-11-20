"""Real retrieval entry, forbidden credentials/generation, then frozen arm replay."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
from dataclasses import asdict
import json
import argparse
from pathlib import Path
from tools.validate_rerank_semantic import OUT,read_jsonl,route,SavedJudge
from tools.validate_selection_factorial import load_inputs,no_network,write,ROOT
from tools.run_rag_experiment import (RAG_MODEL_CACHE_DIR,AppSettings,IndexGenerationStore,
    CancelToken,AnswerService,LocalBgeEmbeddingProvider,RagIndexStore,
    LocalModelManager,ModelSpec,LocalBgeReranker)
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.evidence_ordering import EvidenceOrdering
from tools.validate_query_plan_adapter import STAGES


class Forbidden:
    def retrieve(self,*a,**k):raise AssertionError('Credentials forbidden')
    def generate(self,*a,**k):raise AssertionError('Generation forbidden')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--semantic',action='store_true');args=parser.parse_args()
    import torch
    torch.set_num_threads(3)
    rows,s,_,_=load_inputs();settings=AppSettings.from_dict(s)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    selected=[r for r in rows if r['id'] in {'rag-biz-004','rag-biz-006','rag-biz-020','rag-biz-021','rag-biz-023'}]
    fresh=read_jsonl(OUT/'new_holdout_pending.jsonl')
    if not args.semantic:selected += [dict(r,cohort='new_pending') for r in fresh]
    semantic_records=read_jsonl(OUT/'model_progress.jsonl') if args.semantic else []
    output_stem='native_semantic' if args.semantic else 'native_retrieval'
    records=[];checks=[]
    with no_network() as attempts:
        embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
            repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
        ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
            repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,
            max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,
            retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,
            max_rank_drop=settings.rag_rerank_max_rank_drop)
        for row in selected:
            for arm in (['M','RM'] if args.semantic else ['baseline','R'] if row['cohort']!='new_pending' else ['baseline']):
                events=[];stages={}
                def progress(e):
                    events.append(e)
                    for field,stage in STAGES.items():
                        if field in e:stages[stage]=e[field]
                if args.semantic:
                    judge=SavedJudge(semantic_records,row['id']);judge.body_to_ids={}
                    for c in row['candidate_stages']['rerank']:
                        judge.body_to_ids.setdefault(c['text'],[]).append(c['chunk_id'])
                    assert route(row)<=judge.values.keys()
                    ranker.ordering_policy=EvidenceOrdering(rules=arm=='RM',judge=judge,semantic_ids=route(row))
                else:ranker.ordering_policy=EvidenceOrdering(rules=True) if arm=='R' else None
                service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),RagIndexStore('E:/everythingfinder-db/rag'),
                    embedding,ranker,Forbidden(),Forbidden(),settings,progress_callback=progress)
                q=RagQuery(row['question'],include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                    lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                    fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k)
                value={k:row[k] for k in ('id','question','group_id','cohort','review_status')}
                try:value['result']=asdict(service.retrieve(q,CancelToken()))
                except Exception as exc:value['error']=repr(exc)
                context=next((e for e in reversed(events) if 'llm_context' in e),{})
                value.update(arm=arm,events=events,candidate_stages=stages,context=context.get('llm_context',''),entries=context.get('context_entries',[]))
                records.append(value)
                (OUT/(output_stem+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in records),encoding='utf8')
                if row['cohort']!='new_pending' and (OUT/(arm+'.jsonl')).exists():
                    expected=next(r for r in read_jsonl(OUT/(arm+'.jsonl')) if r['id']==row['id'])
                    checks.append(dict(id=row['id'],arm=arm,context_equal=value['context']==expected['context'],
                        rerank_order_equal=[c['chunk_id'] for c in stages.get('rerank',[])]==[c['chunk_id'] for c in expected['rerank_candidates']],
                        rerank_input_equal={c['chunk_id'] for c in stages.get('rerank',[])}=={c['chunk_id'] for c in expected['rerank_candidates']}))
                print('native',row['id'],arm,'entries',len(value['entries']),value.get('error',''),flush=True)
        write(OUT/(output_stem+'_audit.json'),dict(network_attempts=attempts,api_calls=0,records=len(records),
            checks=checks,errors=[{'id':r['id'],'error':r['error']} for r in records if r.get('error')],
            full_real_retrieval=True,semantic_output_reused=args.semantic,
            note='Cached same-input semantic judgements; not an independent inference repeat' if args.semantic else 'No semantic inference',
            default_adapter_enabled=settings.rag_query_plan_enabled))
        assert not attempts and all('error' not in r for r in records)

if __name__=='__main__':main()
