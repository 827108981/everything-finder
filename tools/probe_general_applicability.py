"""Fixed-context generic applicability rule; diagnostic only, no deployment."""
from tools.probe_019_050_evidence import *
OUT=ROOT/'reports/rag_scene_entry_20261105'
RULE='证据适用性判断：先区分通用原理或排查方向与特定机型的结构、数值、操作参数。资料中的示例名称不同，不单独构成拒绝通用检查方向的理由；涉及专属参数时须核实明确适用对象，不得仅凭机型相近移用。将可能原因作为待检查项，不说成已经确认的故障。无法核实适用范围时明确说明不确定性。'

def main():
 settings=replace(AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8'))),rag_api_timeout_seconds=180)
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf8')
 base=json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/resolved_cases.json').read_text(encoding='utf8'))
 selected=[x for x in base if x['arm']=='baseline' and x['id'] in ('rag-biz-050','rag-row-holdout2-007')]
 jobs=[{'id':x['id'],'question':x['question'],'context':x['context'],'arm':arm,'repeat':rep} for x in selected for arm in ('baseline','generic_scope') for rep in (1,2,3)]
 freeze=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))['hashes'];assert all(sha(p)==h for p,h in freeze.items())
 dump('applicability_protocol.json',{'jobs':len(jobs),'generic_rule':RULE,'model':settings.rag_api_model,'selection':'050 observed over-restriction; holdout2-007 observed wrong-model numbers as a boundary check. Both previously seen; diagnostic, not independent generalization. Same saved native contexts, 3 fresh generations each arm. No quality retries. No per-question facts in rule.','frozen_hashes':freeze})
 dump('applicability_inputs.json',jobs)
 def run(x):
  r={k:v for k,v in x.items() if k!='context'}
  try:
   plan=AnswerService._plan_summary(QueryInterpreter().interpret(x['question']))
   if x['arm']=='generic_scope':plan+='\n'+RULE
   ans=OpenAICompatibleGateway(max_transient_retries=2).generate(GenerationRequest(question=x['question'],context=x['context'],query_plan_summary=plan,model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=180),CancelToken())
   r['text']=ans.text;refs={s:() for s in re.findall(r'(?m)^\[(S\d+)\]',x['context'])};r['citations']=list(CitationValidator().validate(ans.text,refs)) if ans.text.strip()!='资料中没有足够依据。' else []
  except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:500]
  return r
 results=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,x) for x in jobs]):
   r=f.result();results.append(r)
   with (OUT/'applicability_outputs.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
   print(r['id'],r['arm'],r['repeat'],r.get('error'),flush=True)
 dump('applicability_completion.json',{'jobs':len(results),'errors':sum(bool(x.get('error')) for x in results),'frozen_inputs_unchanged':all(sha(p)==h for p,h in freeze.items())})
if __name__=='__main__':main()
