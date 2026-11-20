"""Frozen-context diagnostic factorial, not production retrieval or deployment."""
import json,hashlib,re
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from tools.rejudge_scoring_v3 import ROOT,sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.core.task_manager import CancelToken
OUT=ROOT/'reports/rag_evidence_trace_019_050_20261105'
def main():
 settings=replace(AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8'))),rag_api_timeout_seconds=180)
 key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
 def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf8')
 cases={c.case_id:c for c in load_gold_dataset(ROOT/'tests/rag/fixtures/rag_gold_v4_1_review.json',reviewed_only=False).cases}
 source=json.loads((OUT/'original_source_verification.json').read_text(encoding='utf8'))
 hint='业务适用性确认：本题询问免疫暗计数诊断的通用检查方向。免疫场景化资料中该诊断的通用失败原因可用于先行检查，不因示例机型名称不同而一概拒绝采用；不据此确定故障部件或迁移机型专属参数。'
 freeze=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))['frozen_hashes'];assert all(sha(p)==h for p,h in freeze.items())
 jobs=[]
 for cid in ('rag-biz-019','rag-biz-050'):
  trace=json.loads((OUT/(cid+'.json')).read_text(encoding='utf8'));e=next(e for e in reversed(trace['events']) if 'llm_context' in e);ctx=e['llm_context'];last=max(int(x) for x in re.findall(r'(?m)^\[S(\d+)\]',ctx))
  parts=[x for x in source if x['file'].endswith('.pptx')] if cid.endswith('019') else [x for x in source if x['file'].endswith('.pdf')]
  extra='\n\n'+f'[S{last+1}]\n来源：'+('化学发光定标处理思路分析.pptx · 第13–14张（相邻连续页）' if cid.endswith('019') else '免疫场景化培训教材-光学V1.1.pdf · 第17页')+'\n内容：'+'\n'.join(x['text'] for x in parts)
  arms=[('baseline',ctx,''),('correct_source',ctx+extra,'')]
  if cid.endswith('050'):arms += [('business_scope_only',ctx,hint),('source_and_business_scope',ctx+extra,hint)]
  for arm,context,scope in arms:
   for rep in (1,2,3):jobs.append({'id':cid,'arm':arm,'repeat':rep,'context':context,'business_scope':scope})
 dump('probe_protocol.json',{'started':str(datetime.now()),'model':settings.rag_api_model,'harness_sha256':sha(__file__),'jobs':18,'selection':'2 scenarios;019baseline vs correct adjacent source;0502x2 source availability x business applicability hint;3 fresh generations per cell. No quality retries.','hint':hint,'limits':'Source injection uses a diagnostic extra source; may exceed production final_top_k. This is an oracle intervention, not a retrieval improvement, a formal score, or proof of generalization. No Gold/frozen algorithms/index/config file edits.','timeout_seconds':180,'saved_settings_unchanged':True,'fixed_input_hashes':freeze})
 dump('probe_inputs.json',jobs)
 def run(x):
  r={k:v for k,v in x.items() if k!='context'};r['context_sha256']=hashlib.sha256(x['context'].encode()).hexdigest()
  try:
   c=cases[x['id']];plan=AnswerService._plan_summary(QueryInterpreter().interpret(c.question));plan+='\n'+x['business_scope'] if x['business_scope'] else ''
   ans=OpenAICompatibleGateway(max_transient_retries=2).generate(GenerationRequest(question=c.question,context=x['context'],query_plan_summary=plan,model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds),CancelToken())
   r['text']=ans.text;refs={s:() for s in re.findall(r'(?m)^\[(S\d+)\]',x['context'])};r['citation_ids']=list(CitationValidator().validate(ans.text,refs)) if ans.text.strip()!='资料中没有足够依据。' else []
  except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:500]
  return r
 previous=[json.loads(l) for l in (OUT/'probe_outputs.jsonl').read_text(encoding='utf8').splitlines()] if (OUT/'probe_outputs.jsonl').exists() else []
 seen={(x['id'],x['arm'],x['repeat']) for x in previous};results=list(previous)
 with ThreadPoolExecutor(max_workers=4) as pool:
  for f in as_completed([pool.submit(run,x) for x in jobs if (x['id'],x['arm'],x['repeat']) not in seen]):
   r=f.result();results.append(r)
   with (OUT/'probe_outputs.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
   print(len(results),r['id'],r['arm'],r['repeat'],r.get('error'),flush=True)
 assert all(sha(p)==h for p,h in freeze.items())
 dump('probe_completion.json',{'outputs':len(results),'errors':sum(bool(x.get('error')) for x in results),'frozen_inputs_unchanged':True})
if __name__=='__main__':main()
