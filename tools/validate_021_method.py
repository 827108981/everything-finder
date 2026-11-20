"""021 user-approved method-question Gold correction; no algorithm changes."""
from tools.validate_row_holdout import *
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import asdict
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import _load_rows
import copy, re
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.citation_validator import CitationValidator

OUT=ROOT/'reports/rag_021_method_20261106'
FIX=ROOT/'tests/rag/fixtures'

def main():
 OUT.mkdir(exist_ok=True)
 assert not (OUT/'answers.jsonl').exists(), 'Do not overwrite or quality-retry an executed run'
 settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')))
 store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active()
 oldfile=FIX/'rag_gold_v4_1_review.json';old=json.loads(oldfile.read_text(encoding='utf8'));document=copy.deepcopy(old)
 oldcase=next(c for c in load_gold_dataset(oldfile,reviewed_only=False).cases if c.case_id=='rag-biz-021')
 row=next(x for x in document['cases'] if x['id']=='rag-biz-021');assert row['corpus_snapshot']==bundle.bundle_id
 row['expected_behavior']='answer';row['missing_fields']=[]
 core=[('F1','LVC表示本次与前次成功定标数据偏差超出允许范围；报警本身不能确定哪一次定标异常。'),('F2','若怀疑上次定标所用试剂或校准品存在问题，按上次定标异常分支，延期上次定标数据后重新定标；不能理解为直接无条件沿用旧结果。'),('F3','若怀疑本次试剂或校准品存在问题，分别排查并更换有问题的试剂或校准品后重新定标；不要求无条件同时更换二者。'),('F4','先分析哪一次定标可能有问题，再按对应条件分支处理；不能仅凭LVC直接选定延期或更换路径。')]
 row['required_facts']=[t for _,t in core]
 row['answer_checks']=[{'id':i,'text':t,'role':'core'} for i,t in core]+[{'id':'C1','text':'若需判断某个具体项目当前应走哪条路径，还需进一步核对两次定标数据或核查结果；本题解释方法不要求强制先追问。','role':'context'}]
 row['forbidden_claims']=['仅凭LVC报警就认定上次定标错误，并无条件要求延期上次定标。','仅凭LVC报警就认定本次试剂或校准品已损坏，并无条件要求更换。','LVC一出现就必须同时更换本次试剂和校准品，不需区分原因。','未提供两次定标数据就完全不能解释LVC后的条件分支，必须先追问才能回答本题。']
 path='E:/生化免疫临床知识/化学发光定标处理思路分析.pptx'
 row['gold_evidence'].append({'file':path,'page':30,'position':'LVC排查思路／上次异常的延期分支','quote':'延期操作：\n把上次定标延期，重新定标（上次试剂有瓶间差或校准品有批间和瓶间差）','relevance':2,'why_sufficient':'用户已确认本题为解释方法；明确给出上次定标相关问题的延期后重定标路径。'})
 row['gold_evidence'].append({'file':path,'page':30,'position':'LVC排查思路／试剂问题分支','quote':'2、试剂问题：\n①.更换新瓶试剂重新测试；','relevance':2,'why_sufficient':'支持本次试剂有问题时更换新瓶试剂后重新测试。'})
 row['gold_evidence'].append({'file':path,'page':30,'position':'LVC排查思路／校准品问题分支','quote':'3、校准品问题：\n①.确认校准品量是否足够，是否放错；\n②.更换校准品重新测试。','relevance':2,'why_sufficient':'支持校准品问题需先核对并按对应分支更换重测；与试剂分支分开。'})
 row['annotation_provenance']={'source_version':'v4.1','status':'reviewed','review_date':'2026-11-06','review_basis':'用户明确确认021为方法型问题，expected_behavior由clarify改answer；两个条件分支必答，不要求计算或强制追问。','note':'问题原话、group_id、索引包和其他49题保持不变；新增原文仅补足分支操作的可核对依据。'}
 document['_阅读说明']['用途']='Gold v4.2：仅按2026-11-06用户明确确认修订021的方法型问题口径；v4.1及更早版本保留。'
 document['_阅读说明']['历史对照']='021标签修订为answer不代表算法改善。UI默认数据集不切换。007仍disputed、024仍pending。'
 document['_阅读说明']['状态变更记录']+=' 2026-11-06：021由clarify改answer，保留reviewed；必答两个条件分支，不强制追问。'
 newfile=FIX/'rag_gold_v4_2_review.json';newl=FIX/'rag_gold_v4_2.jsonl';assert not newfile.exists() and not newl.exists()
 newfile.write_text(json.dumps(document,ensure_ascii=False,indent=2)+'\n',encoding='utf8');newl.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in document['cases']),encoding='utf8')
 changed=[a['id'] for a,b in zip(old['cases'],document['cases']) if a!=b];assert changed==['rag-biz-021']
 assert _load_rows(newfile)[0]==_load_rows(newl)[0]
 newcase=next(c for c in load_gold_dataset(newfile,reviewed_only=False).cases if c.case_id=='rag-biz-021')
 conn=store.connect_read_only(bundle.metadata_path);index=load_chunk_rows(conn);maps=map_case_evidence(newcase,index);conn.close()
 assert all(m.alternatives and any(a.confirmed for a in m.alternatives) for m in maps if m.kind=='positive')
 paths=list((ROOT/'local_full_text_search/rag').rglob('*.py'))+[SETTINGS_PATH,oldfile,FIX/'rag_gold_v4_1.jsonl',store.manifest_path,Path(path)]
 hashes={str(p):sha(p) for p in paths}
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf8')
 dump('gold_delta.json',{'old':oldcase.raw,'new':newcase.raw,'changed_case_ids':changed,'other_49_exact_equal':True,'json_jsonl_equal':True,'positive_mappings':[asdict(m) for m in maps if m.kind=='positive']})
 dump('protocol.json',{'bundle_id':bundle.bundle_id,'model':settings.rag_api_model,'hashes':hashes,'scope':'Only Gold021 changed; original question retained. 3 actual AnswerService.answer runs;3 fresh generations on saved11/5 native context;2 historical11/5 outputs rejudged. Every answer judged with both old and user-approved new Gold. No domain hint or Gold given to answer model. No algorithm/prompt/tool changes, no H8 activation.','native_timeout_seconds':settings.rag_api_timeout_seconds,'diagnostic_and_judge_timeout_seconds':180,'quality_retries':False,'new_gold':str(newfile),'review_status':'021 remains reviewed by explicit user approval; other statuses unchanged'})
 import torch
 torch.set_num_threads(4);torch.set_num_interop_threads(1)
 manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
 embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
 ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)
 saved=[r for r in json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/resolved_cases.json').read_text(encoding='utf8')) if r['id']=='rag-biz-021'];base=next(r for r in saved if r['arm']=='baseline')
 local=Lock();results=[]
 def run(arm,rep):
  r={'id':'rag-biz-021','arm':arm,'repeat':rep,'question':newcase.question};events=[];held=[False]
  def release():
   if held[0]:held[0]=False;local.release()
  g=OpenAICompatibleGateway(max_transient_retries=2)
  class Gateway:
   def generate(self,*a,**k):release();return g.generate(*a,**k)
   def assess_clarification(self,*a,**k):release();return g.assess_clarification(*a,**k)
   def review_answer(self,*a,**k):release();return g.review_answer(*a,**k)
  try:
   if arm=='native_current':
    local.acquire();held[0]=True
    service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,embedding,ranker,Gateway(),WindowsCredentialSecretStore(),settings,progress_callback=events.append)
    q=RagQuery(text=newcase.question,lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k,min_ocr_confidence=settings.ocr_min_confidence)
    ans=service.answer(q,CancelToken());release();ce=next((e for e in reversed(events) if 'llm_context' in e),{})
    r.update(text=ans.answer_text,status=str(ans.status),context=ce.get('llm_context'),entries=ce.get('context_entries',[]),events=events,audit=asdict(ans.audit) if ans.audit else None)
   else:
    r['context']=base['context'];plan=AnswerService._plan_summary(QueryInterpreter().interpret(newcase.question))
    ans=g.generate(GenerationRequest(question=newcase.question,context=r['context'],query_plan_summary=plan,model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=180),CancelToken());r['text']=ans.text;r['status']='diagnostic_generated'
    r['citations']=list(CitationValidator().validate(ans.text,{s:() for s in re.findall(r'(?m)^\[(S\d+)\]',r['context'])})) if ans.text.strip()!='资料中没有足够依据。' else []
  except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:500];r['events']=events
  finally:release()
  return r
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,arm,rep) for rep in (1,2,3) for arm in ('native_current','saved_context')]):
   r=f.result();results.append(r)
   with (OUT/'answers.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
   print('answer',r['arm'],r['repeat'],r.get('error'),flush=True)
 for x in saved:results.append({'id':x['id'],'arm':'historical_'+x['arm'],'repeat':1,'question':x['question'],'text':x['answer']['text'],'status':x['answer']['status'],'context':x['context'],'source':'reports/rag_row_ab_confirm_20261105/resolved_cases.json'})
 judge_settings=replace(settings,rag_api_timeout_seconds=180)
 def judge(r,version):
  out={'arm':r['arm'],'repeat':r['repeat'],'gold_version':version}
  try:out['judgement']=GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=2),judge_settings,key).evaluate(oldcase if version=='v4.1_clarify' else newcase,{'text':r.get('text',''),'status':r.get('status',''),'evidence_context':r.get('context')},CancelToken())
  except Exception as e:out['error']=str(e).replace(key,'[REDACTED]')[:500]
  return out
 judgments=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(judge,r,v) for r in results if not r.get('error') for v in ('v4.1_clarify','v4.2_answer')]):
   r=f.result();judgments.append(r)
   with (OUT/'judgements.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
   print('judged',r['arm'],r['repeat'],r['gold_version'],r.get('judgement',{}).get('verdict'),r.get('error'),flush=True)
 dump('resolved_answers.json',results)
 dump('completion.json',{'fresh_answers':6,'historical_answers':2,'answer_errors':sum(bool(r.get('error')) for r in results),'judge_outputs':len(judgments),'judge_errors':sum(bool(r.get('error')) for r in judgments),'original_inputs_unchanged':all(sha(p)==h for p,h in hashes.items()),'json_jsonl_equal':_load_rows(newfile)[0]==_load_rows(newl)[0]})
 assert all(sha(p)==h for p,h in hashes.items())

if __name__=='__main__':main()
