"""Frozen-context, randomized paired experiment; production stays untouched."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import random
import time
from tools.validate_context_ceiling import ROOT,FIX,AB,sha,read,lines,dump,cases
from tools.answer_decision_candidate import generate
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

OUT=ROOT/'reports/rag_answer_decision_20261105'
TARGETS=['rag-biz-014','rag-biz-015','rag-biz-021','rag-biz-049']
CONTROLS=['rag-biz-002','rag-biz-008','rag-biz-016','rag-biz-027','rag-biz-045']
def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    prior=read(ROOT/'reports/rag_retrieval_ab_results_20261103/protocol.json')
    assert all(sha(p)==h for p,h in prior['hashes'].items())
    assert RagIndexStore(Path('E:/everythingfinder-db/rag')).active().bundle_id==prior['bundle_id']
    cs=cases();base={r['id']:r for r in lines(AB) if r['arm']=='baseline'}
    data=[]
    for cid in TARGETS+CONTROLS:
        r=base[cid];data.append({'id':cid,'variant':'current','question':cs[cid].question,'context':r['context'],'context_entries':r['context_entries'],'role':'target' if cid in TARGETS else 'control'})
    prior_data=read(ROOT/'reports/rag_context_ceiling_20261104/prepared.json')
    r=next(r for r in prior_data if r['id']=='rag-biz-015')
    data.append({'id':r['id'],'variant':'reordered_stress','question':cs[r['id']].question,**r['source_completed'],'role':'target'})
    jobs=[{'id':r['id'],'variant':r['variant'],'arm':arm,'repeat':rep} for r in data for arm in ['baseline','candidate'] for rep in range(1,4)]
    random.Random(20261105).shuffle(jobs)
    paths=list(prior['hashes'])+[str(ROOT/'tools/answer_decision_candidate.py'),str(ROOT/'tools/validate_answer_decision.py')]
    dump(OUT/'protocol.json',{'started':str(datetime.now()),'phase':'target_and_controls_gate','bundle_id':prior['bundle_id'],
        'hashes':{p:sha(p) for p in paths},'targets':TARGETS,'controls':CONTROLS,'jobs':jobs,'seed':20261105,
        'gates':'Targets each 3/3 judge-consistent; 015 current AND reordered each 3/3 with no whole refusal. Controls no verified new omission/false clarification/unsupported claim. Failed gate stops full expansion. Gate verdicts require actual answer review, not labels alone.',
        'isolation':'Same context bytes/order within each pair, same model/output budget=1000. Generator sees question/context/normal query plan only, no Gold labels or facts. Reordered stress is prior data, no new manually completed evidence.',
        'calls':'One normal generation call per arm. Candidate maximum one format-only recovery; SDK transient retries unchanged. No quality-based retries. Execution failures separate.',
        'production':'No application default/settings/Gold/index modifications. Candidate is standalone adapter; app integration is a later gate.'})
    dump(OUT/'inputs.json',data);print('prepared',len(data),'contexts',len(jobs),'jobs',flush=True)

def run():
    p=read(OUT/'protocol.json');assert all(sha(f)==h for f,h in p['hashes'].items())
    s=AppSettings.from_dict(read(SETTINGS_PATH));key=WindowsCredentialSecretStore().retrieve(s.rag_api_secret_ref)or'';assert key
    cs=cases();inputs={(r['id'],r['variant']):r for r in read(OUT/'inputs.json')}
    dump(OUT/'execution.json',{'started':str(datetime.now()),'model':s.rag_api_model,'workers':6,'max_output_tokens':1000})
    def job(j):
        r=dict(j);data=inputs[(j['id'],j['variant'])];c=cs[j['id']];r['role']=data['role'];r['error']=None
        gateway=OpenAICompatibleGateway(max_transient_retries=2)
        request=GenerationRequest(question=data['question'],context=data['context'],model=s.rag_api_model,
          api_base_url=s.rag_api_base_url,api_style=s.rag_api_style,api_key=key,timeout_seconds=s.rag_api_timeout_seconds,
          query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(data['question'])))
        started=time.monotonic()
        try:
            if j['arm']=='candidate':out=generate(gateway,request,data['context_entries'],CancelToken())
            else:
                a=gateway.generate(request,CancelToken());refuse=a.text.strip()=='资料中没有足够依据。'
                out={'text':a.text,'status':'insufficient' if refuse else 'answered','behavior':'refuse' if refuse else 'not_classified','format_recoveries':0}
            r['generation_ms']=round((time.monotonic()-started)*1000);r['output']=out
            r['citation_check']=asdict(CitationValidator().inspect(out['text'],{e['source_id']:() for e in data['context_entries']}))
            r['answer']={'mode':'full','text':out['text'],'status':out['status']}
            t=time.monotonic();r['gold_judgement']=GoldAnswerJudge(gateway,s,key).evaluate(c,r['answer'],CancelToken());r['judge_ms']=round((time.monotonic()-t)*1000)
        except Exception as ex:r['error']={'type':type(ex).__name__,'message':str(ex)[:400].replace(key,'[REDACTED]')}
        r['total_ms']=round((time.monotonic()-started)*1000);return r
    path=OUT/'results.jsonl';done={(r['id'],r['variant'],r['arm'],r['repeat']) for r in lines(path)} if path.exists() else set()
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(job,j) for j in p['jobs'] if (j['id'],j['variant'],j['arm'],j['repeat']) not in done]
        for f in as_completed(futures):
            r=f.result()
            with path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
            print(r['id'],r['variant'],r['arm'],r['repeat'],(r.get('gold_judgement')or{}).get('verdict'),r['error'],flush=True)
    assert all(sha(f)==h for f,h in p['hashes'].items())
    rs=lines(path);dump(OUT/'completion.json',{'finished':str(datetime.now()),'results':len(rs),'expected':len(p['jobs']),'errors':sum(bool(r['error']) for r in rs),'hashes_unchanged':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);v=a.parse_args();{'prepare':prepare,'run':run}[v.action]()
