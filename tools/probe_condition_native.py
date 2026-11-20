"""Real local retrieval through AnswerService.retrieve; no credentials or LLM."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
from dataclasses import asdict
import json
from pathlib import Path
from tools.validate_condition_binding import OUT
from tools.validate_selection_factorial import load_inputs, no_network, write, ROOT
from tools.run_rag_experiment import (RAG_MODEL_CACHE_DIR, AppSettings, IndexGenerationStore,
    CancelToken, AnswerService, LocalBgeEmbeddingProvider, RagIndexStore,
    LocalModelManager, ModelSpec, LocalBgeReranker)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from tools.validate_query_plan_adapter import STAGES


class Forbidden:
    def retrieve(self,*a,**k): raise AssertionError('Credentials forbidden')
    def generate(self,*a,**k): raise AssertionError('LLM forbidden')


def main():
    rows,s,_,maps=load_inputs(); settings=AppSettings.from_dict(s)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
        repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
    ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
        repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,
        max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,
        retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,
        max_rank_drop=settings.rag_rerank_max_rank_drop)
    selected=[r for r in rows if r['id'] in {'rag-biz-019','rag-biz-022','rag-biz-021','rag-biz-023'}]
    result=[]
    with no_network() as attempts:
        for r in selected:
            for arm in ('baseline','combined'):
                events=[]; stages={}
                def progress(e):
                    events.append(e)
                    for field,stage in STAGES.items():
                        if field in e:stages[stage]=e[field]
                service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),RagIndexStore('E:/everythingfinder-db/rag'),
                    embedding,ranker,Forbidden(),Forbidden(),settings,
                    query_interpreter=QueryInterpreter(bound_conditions=arm=='combined'),
                    constraint_matcher=ConstraintMatcher(local_conditions=arm=='combined'),progress_callback=progress)
                q=RagQuery(r['question'],include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                    lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                    fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k)
                record={'id':r['id'],'question':r['question'],'arm':arm}
                try: record['result']=asdict(service.retrieve(q,CancelToken()))
                except Exception as exc:record['error']=repr(exc)
                context=next((e for e in reversed(events) if 'llm_context' in e),{})
                record.update(events=events,candidate_stages=stages,context=context.get('llm_context',''),entries=context.get('context_entries',[]))
                record['scores']={k:score_stage(k,v,maps[r['id']]) for k,v in {**stages,'context':record['entries']}.items()}
                result.append(record)
                (OUT/'native.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in result),encoding='utf8')
                print(r['id'],arm,record['scores']['context']['gold_evidence_found'],record.get('error',''),flush=True)
    write(OUT/'native_audit.json',{'cases':len(selected),'runs':len(result),'network_attempts':attempts,'api_calls':0,
        'errors':[r for r in result if r.get('error')],'full_local_retrieval':True})


if __name__=='__main__':main()
