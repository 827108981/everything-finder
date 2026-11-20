"""Recover only the three pre-request harness import errors; preserve originals."""
from tools.validate_021_method import *

def main():
 assert not (OUT/'saved_context_recovery_answers.jsonl').exists()
 settings=replace(AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8'))),rag_api_timeout_seconds=180)
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 source=json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/resolved_cases.json').read_text(encoding='utf8'))
 base=next(r for r in source if r['id']=='rag-biz-021' and r['arm']=='baseline')
 cases={v:next(c for c in load_gold_dataset(FIX/f,reviewed_only=False).cases if c.case_id=='rag-biz-021') for v,f in [('v4.1_clarify','rag_gold_v4_1_review.json'),('v4.2_answer','rag_gold_v4_2_review.json')]}
 original=[json.loads(l) for l in (OUT/'answers.jsonl').read_text(encoding='utf8').splitlines()]
 assert all(any(r['arm']=='saved_context' and r['repeat']==rep and 'GenerationRequest' in r.get('error','') for r in original) for rep in (1,2,3))
 (OUT/'recovery_protocol.json').write_text(json.dumps({'reason':'Three saved-context jobs failed on missing GenerationRequest import before invoking gateway.generate. No model response was discarded. Native jobs and old failures retained.','harness_sha256':sha(__file__),'corrected_main_sha256':sha(ROOT/'tools/validate_021_method.py'),'recover_repeats':[1,2,3]},ensure_ascii=False,indent=2),encoding='utf8')
 def run(rep):
  r={'id':'rag-biz-021','arm':'saved_context','repeat':rep,'question':cases['v4.2_answer'].question,'context':base['context'],'execution_recovery_of':'pre-request GenerationRequest NameError'}
  try:
   g=OpenAICompatibleGateway(max_transient_retries=2);plan=AnswerService._plan_summary(QueryInterpreter().interpret(r['question']))
   ans=g.generate(GenerationRequest(question=r['question'],context=r['context'],query_plan_summary=plan,model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=180),CancelToken());r['text']=ans.text;r['status']='diagnostic_generated'
   r['citations']=list(CitationValidator().validate(ans.text,{s:() for s in re.findall(r'(?m)^\[(S\d+)\]',r['context'])})) if ans.text.strip()!='资料中没有足够依据。' else []
  except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:500]
  return r
 results=[]
 with ThreadPoolExecutor(max_workers=3) as pool:
  for f in as_completed([pool.submit(run,rep) for rep in (1,2,3)]):
   r=f.result();results.append(r)
   with (OUT/'saved_context_recovery_answers.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
   print('recovered',r['repeat'],r.get('error'),flush=True)
 def judge(r,v):
  out={'arm':r['arm'],'repeat':r['repeat'],'gold_version':v}
  try:out['judgement']=GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=2),settings,key).evaluate(cases[v],{'text':r.get('text',''),'status':r.get('status',''),'evidence_context':r['context']},CancelToken())
  except Exception as e:out['error']=str(e).replace(key,'[REDACTED]')[:500]
  return out
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(judge,r,v) for r in results if not r.get('error') for v in cases]):
   r=f.result()
   with (OUT/'saved_context_recovery_judgements.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
   print('judge',r['repeat'],r['gold_version'],r.get('judgement',{}).get('verdict'),r.get('error'),flush=True)

if __name__=='__main__':main()
