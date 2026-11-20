"""Full native AnswerService A/B with isolated context adapter; no production writes."""
from tools.validate_row_holdout import *
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
from dataclasses import asdict
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
import sys
OUT=ROOT/'reports/rag_evidence_trace_019_050_20261105'
def main():
 OUT.mkdir(exist_ok=True);freeze=json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/freeze.json').read_text(encoding='utf8'));assert all(sha(p)==h for p,h in freeze['hashes'].items())
 settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')))
 store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==freeze['bundle_id']
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf8')
 cases=[]
 for fn,cohort in [('rag_gold_v4_1_review.json','formal48'),('rag_acceptance_v1_review.json','regression10_pending'),('rag_row_holdout_v1_review.json','used_holdout10_pending'),('rag_row_holdout_v2_review.json','new_holdout10_pending')]:
  for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/fn,reviewed_only=False).cases:
   if cohort=='formal48' and c.review_status!='reviewed':continue
   cases.append((c,cohort))
 conn=store.connect_read_only(bundle.metadata_path);index=load_chunk_rows(conn);conn.close();maps={c.case_id:map_case_evidence(c,index) for c,_ in cases}
 units=json.loads((ROOT/'reports/rag_row_integration_20261105/row_index.json').read_text(encoding='utf8'));byuid={u['id']:u for u in units}
 analyzer=get_lexical_analyzer();tok=lambda t:[x.normalized for x in analyzer.analyze_text(t)]
 counts=[Counter(tok(' '.join(c['header']+' '+(c['value'] or '') for c in u['bindings']))) for u in units];df=Counter(t for c in counts for t in c);avg=sum(sum(c.values()) for c in counts)/len(counts)
 def search(q):
  terms=set(tok(q));scored=[]
  for n,c in enumerate(counts):
   length=sum(c.values());v=sum(math.log(1+(len(counts)-df[t]+.5)/(df[t]+.5))*c[t]*2.2/(c[t]+1.2*(.25+.75*length/avg)) for t in terms if c[t])
   if v:scored.append((v,n))
  return [units[n] for v,n in sorted(scored,reverse=True)[:10]]
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
 embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
 ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)

 cases=[(c,co) for c,co in cases if c.case_id in ('rag-biz-019','rag-biz-050')]
 dump('protocol.json',{'bundle_id':bundle.bundle_id,'settings_hash':sha(SETTINGS_PATH),'business_confirmation_hash':sha(ROOT/'reports/rag_row_ab_confirm_20261105/business_confirmation_019_050.json'),'frozen_hashes':freeze['hashes'],'harness_sha256':sha(__file__),'scope':'Fresh native retrieval twice; original sources/index/Gold/config/algorithms unchanged. No hosted generation in this phase.'})
 for case,cohort in cases:
  events=[]
  service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,embedding,ranker,OpenAICompatibleGateway(),WindowsCredentialSecretStore(),settings,progress_callback=events.append)
  q=RagQuery(text=case.question,lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=8,min_ocr_confidence=settings.ocr_min_confidence)
  result=service.retrieve(q,CancelToken())
  dump(case.case_id+'.json',{'id':case.case_id,'question':case.question,'status':str(result.status),'events':events})
  print('traced',case.case_id,flush=True)
 assert all(sha(p)==h for p,h in freeze['hashes'].items())
 dump('completion.json',{'cases':2,'hashes_unchanged':True})
if __name__=='__main__':main()
