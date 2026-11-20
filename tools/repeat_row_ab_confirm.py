"""Predetermined fixed-context answer repetitions; not fresh retrieval repetitions."""
import json,time,hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from tools.prepare_row_ab_confirm import ROOT,OUT,sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.core.task_manager import CancelToken

def main():
 settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')));assert not settings.rag_answer_review_enabled and not settings.rag_semantic_clarification_enabled
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 cases={c.case_id:c for fn in ['rag_gold_v4_1_review.json','rag_acceptance_v1_review.json','rag_row_holdout_v1_review.json','rag_row_holdout_v2_review.json'] for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/fn,reviewed_only=False).cases}
 (OUT/'repeat_execution.json').write_text(json.dumps({'method':'Two fresh answer+judge calls per arm on each arm own frozen actual context. Not a repeated retrieval or latency trial. Same generation gateway, plan, max tokens and final citation check as current service; configured review/semantic clarification are false.','harness_sha256':sha(__file__),'selection_policy_sha256':sha(OUT/'repeat_policy.json')},indent=2),encoding='utf8')
 def run(row,rep):
  r={'id':row['id'],'cohort':row['cohort'],'arm':row['arm'],'repeat':rep,'method':'fixed_context_answer_repeat','context_sha256':row['context_sha256']}
  try:
   g=OpenAICompatibleGateway(max_transient_retries=2);ctx=row.get('context');case=cases[row['id']]
   if ctx:
    ans=g.generate(GenerationRequest(question=case.question,context=ctx,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question)),model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds),CancelToken())
    text=ans.text.strip()
    if text!='资料中没有足够依据。':CitationValidator().validate(text,{e['source_id']:() for e in row['entries']})
   else:text=row['answer']['text']
   r['answer']={'text':text,'status':row['answer']['status']};r['judgement']=GoldAnswerJudge(g,settings,key).evaluate(case,{**r['answer'],'evidence_context':ctx},CancelToken())
  except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:500]
  return r
 seen=set();pending=set();results=[]
 if (OUT/'repeats.jsonl').exists():
  results=[json.loads(l) for l in (OUT/'repeats.jsonl').read_text(encoding='utf8').splitlines()];seen={(r['id'],r['arm'],r['repeat']) for r in results}
 with ThreadPoolExecutor(max_workers=2) as pool:
  while True:
   by=defaultdict(dict)
   for l in (OUT/'cases.jsonl').read_text(encoding='utf8').splitlines():
    try:r=json.loads(l)
    except json.JSONDecodeError:continue
    by[r['id']][r['arm']]=r
   for cid,p in by.items():
    if len(p)!=2 or any(r.get('error') for r in p.values()):continue
    a,b=p['baseline'],p['H8'];va=a.get('judgement',{}).get('verdict');vb=b.get('judgement',{}).get('verdict')
    if a.get('context')==b.get('context'):continue
    losses=set(a['score']['gold_completion_ranks'])-set(b['score']['gold_completion_ranks']);neg=set(b['score']['hard_negative_ids'])-set(a['score']['hard_negative_ids'])
    eligible=(va=='consistent')!=(vb=='consistent') or vb=='incorrect' and va!='incorrect' or losses or neg
    if not eligible:continue
    for arm in ('baseline','H8'):
     for rep in (2,3):
      k=(cid,arm,rep)
      if k not in seen:seen.add(k);pending.add(pool.submit(run,p[arm],rep))
   if pending:
    done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
    for f in done:
     r=f.result();pending.remove(f);results.append(r)
     with (OUT/'repeats.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
     print(r['id'],r['arm'],r['repeat'],r.get('judgement',{}).get('verdict'),r.get('error'),flush=True)
   elif (OUT/'completion.json').exists():break
   else:time.sleep(3)
 (OUT/'repeat_completion.json').write_text(json.dumps({'rows':len(results),'errors':sum(bool(r.get('error')) for r in results)},indent=2),encoding='utf8')
if __name__=='__main__':main()
