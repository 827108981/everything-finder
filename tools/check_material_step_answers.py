"""Paired repeated diagnostic for changed material contexts; no production writes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from tools.validate_material_steps import ROOT, sha

def main():
    out=ROOT/'reports/rag_material_steps_20261105'
    assert not (out/'answer_results.jsonl').exists()
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    hashes={**protocol['hashes'],**protocol['production_and_gold_hashes'],str(out/'cases.jsonl'):sha(out/'cases.jsonl'),__file__:sha(__file__)}
    assert all(sha(p)==h for p,h in hashes.items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    old={(r['id'],r['arm']):r for r in (json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8'))}
    replay={(r['id'],r['step']):r for r in (json.loads(l) for l in (out/'cases.jsonl').open(encoding='utf-8'))}
    cases={c.case_id:c for c in load_gold_dataset(ROOT/'tests/rag/fixtures/rag_gold_v4_1_review.json',reviewed_only=False).cases}
    jobs=[]
    # Changed target S1/023; S3/050 equivalent-source target; S3/003 regression control.
    for cid,step in (('rag-biz-023','S1'),('rag-biz-050','S3'),('rag-biz-003','S3')):
        for arm in ('baseline','candidate'):
            r=old[cid,'baseline'] if arm=='baseline' else replay[cid,step]
            for repeat in range(1,4):
                jobs.append({'id':cid,'step':step,'arm':arm,'repeat':repeat,'context':r['context'],'entries':r.get('context_entries',r.get('entries')),'status':old[cid,'baseline']['retrieval_status']})
    random.Random(20261105).shuffle(jobs)
    def dump(name,v):
        (out/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    dump('answer_protocol.json',{'started':str(datetime.now()),'hashes':hashes,'jobs':[{k:v for k,v in j.items() if k not in ('context','entries')} for j in jobs],
        'model':settings.rag_api_model,'repeats':3,'workers':6,'seed':20261105,
        'scope':'18 fresh generations plus Gold judgements, diagnostic only, no full-set answer score. S2 identical context skipped.',
        'S1_gate_exception':'Strict negative flag from DCF row sharing the exact CSDB positive table. Paired diagnostic checks misuse; does not clear rollout gate or modify Gold.'})
    def run(j):
        start=time.monotonic();g=OpenAICompatibleGateway(max_transient_retries=2)
        result={k:v for k,v in j.items() if k not in ('context','entries')}
        result['context_sha256']=hashlib.sha256(j['context'].encode()).hexdigest()
        try:
            generated=g.generate(GenerationRequest(question=cases[j['id']].question,context=j['context'],model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(cases[j['id']].question))),CancelToken())
            answer={'mode':'full','text':generated.text,'status':j['status'],'source_ids':[e['source_id'] for e in j['entries']]}
            result['answer']=answer
            result['citation_check']=asdict(CitationValidator().inspect(generated.text,{e['source_id']:() for e in j['entries']}))
            result['judgement']=GoldAnswerJudge(g,settings,key).evaluate(cases[j['id']],answer,CancelToken())
        except Exception as e:
            result['error']={'type':type(e).__name__,'message':str(e).replace(key,'[REDACTED]')[:400]}
        result['elapsed_ms']=round((time.monotonic()-start)*1000)
        return result
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            r=future.result();results.append(r)
            with (out/'answer_results.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            print(len(results),r['id'],r['arm'],r['repeat'],(r.get('judgement') or {}).get('verdict'),bool(r.get('error')),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump('answer_completion.json',{'finished':str(datetime.now()),'count':len(results),'execution_errors':sum(bool(r.get('error')) for r in results),'hashes_unchanged':True})

if __name__=='__main__':main()
