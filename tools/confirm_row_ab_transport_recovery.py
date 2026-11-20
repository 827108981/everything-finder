"""Full native AnswerService A/B with isolated context adapter; no production writes."""
from tools.validate_row_holdout import *
from concurrent.futures import ThreadPoolExecutor,as_completed
from threading import Lock
from dataclasses import asdict
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
import sys
OUT=ROOT/'reports/rag_row_ab_confirm_20261105/transport_recovery'
def main():
 freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf8'));assert all(sha(p)==h for p,h in freeze['hashes'].items())
 settings=replace(AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8'))),rag_api_timeout_seconds=180)
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
 local_lock=Lock();policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=3)
 dump('protocol.json',{'started':str(datetime.now()),'bundle_id':bundle.bundle_id,'model':settings.rag_api_model,'settings_hash':sha(SETTINGS_PATH),'semantic_clarification':settings.rag_semantic_clarification_enabled,'answer_review':settings.rag_answer_review_enabled,'cases':len(cases),'arms':['baseline','H8'],'execution':'Actual AnswerService.answer for both arms; A native builder, B native builder wrapped by frozen row adapter. Serialized local inference,4remote workers. All configured generation/clarification/review and final citation validation retained. No quality retries.','harness_sha256':sha(__file__),'freeze_sha256':sha(OUT/'freeze.json')})
 def run(case,cohort,arm,repeat=1):
  r={'id':case.case_id,'cohort':cohort,'arm':arm,'repeat':repeat,'question':case.question,'review_status':case.review_status};events=[];held=[False];extra={}
  g=OpenAICompatibleGateway(max_transient_retries=2)
  def release():
   if held[0]:held[0]=False;local_lock.release()
  class Gateway:
   def generate(self,*a,**k):release();return g.generate(*a,**k)
   def assess_clarification(self,*a,**k):release();return g.assess_clarification(*a,**k)
   def review_answer(self,*a,**k):release();return g.review_answer(*a,**k)
  plan=QueryInterpreter().interpret(case.question)
  class RowBuilder:
   def build(self,original,*,max_chars):
    before=ContextBuilder().build(original,max_chars=max_chars);items=[]
    conn=store.connect_read_only(bundle.metadata_path)
    try:
     for n,u in enumerate(search(case.question),1):
      parent=store.load_chunk(conn,u['parents'][0]);ch=replace(parent,stable_chunk_id=u['id'],text=u['text'],normalized_text=u['text'],text_hash=hashlib.sha256(u['text'].encode()).hexdigest());i=EvidenceChunk(ch,lexical_rank=n,fused_rank=20+n,member_chunk_ids=tuple(u['parents']));i.answer_support_score,i.answer_support_reasons=compute_answer_support(plan,i);items.append(i)
    finally:conn.close()
    if items:ranker.rerank(case.question,items,len(items),CancelToken());ordered=ConstraintMatcher().evaluate(plan,ranker.last_scored_candidates)
    else:ordered=[]
    selected,trace=select_preserving(original,ordered,policy,min(8,len(original)+1));ctx=ContextBuilder().build(selected,max_chars=max_chars)
    extra.update(original_context=before.text,trace=trace,original_retained=all(canonical_content(e['included_text']) in canonical_content(ctx.text) for e in before.entries),selected_snapshots={i.chunk.stable_chunk_id:candidate_snapshot(i) for i in selected},row_scores=[candidate_snapshot(i) for i in ordered])
    return ctx
  try:
   local_lock.acquire();held[0]=True
   s=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,embedding,ranker,Gateway(),WindowsCredentialSecretStore(),settings,progress_callback=events.append)
   if arm=='H8':s.context_builder=RowBuilder()
   q=RagQuery(text=case.question,lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=8,min_ocr_confidence=settings.ocr_min_confidence)
   ans=s.answer(q,CancelToken());release()
   context_event=next((e for e in reversed(events) if 'llm_context' in e),{})
   context=context_event.get('llm_context');entries=context_event.get('context_entries',[])
   r.update(answer={'text':ans.answer_text,'status':str(ans.status)},context=context,entries=entries,audit=asdict(ans.audit) if ans.audit else None,extra=extra)
   snaps={c['chunk_id']:c for e in events for c in e.get('evidence_candidates',[])};snaps.update(extra.get('selected_snapshots',{}))
   stage=[{**snaps.get(e['chunk_id'],{'chunk_id':e['chunk_id']}),'included_text':e['included_text']} for e in entries]
   r['score']=score_stage('context',stage,maps[case.case_id]);r['context_sha256']=hashlib.sha256((context or '').encode()).hexdigest()
   r['judgement']=GoldAnswerJudge(g,settings,key).evaluate(case,{**r['answer'],'evidence_context':context},CancelToken())
   r['clarification_events']=[e for e in events if e.get('stage') in ('clarification_assessed','clarification_required')]
  except Exception as exc:r['error']=str(exc).replace(key,'[REDACTED]')[:500]
  finally:release()
  return r
 existing=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf8').splitlines()] if (OUT/'cases.jsonl').exists() else []
 done={(r['id'],r['arm'],r.get('repeat',1)) for r in existing}
 jobs=[(c,cohort,arm) for c,cohort in cases for arm in ('baseline','H8') if (c.case_id,arm,1) not in done and (c.case_id,arm) in {('rag-biz-019', 'H8')}];random.Random(2026110517).shuffle(jobs)
 results=list(existing)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,*j) for j in jobs]):
   r=f.result();results.append(r)
   with (OUT/'cases.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
   print(len(results),'/',len(cases)*2,r['id'],r['arm'],(r.get('judgement') or {}).get('verdict'),r.get('error'),flush=True)
 assert all(sha(p)==h for p,h in freeze['hashes'].items())
 dump('completion.json',{'finished':str(datetime.now()),'rows':len(results),'errors':sum(bool(r.get('error')) for r in results),'frozen_inputs_unchanged':True})
if __name__=='__main__':main()
