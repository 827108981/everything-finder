import argparse,hashlib,json,random,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from dataclasses import asdict
from tools.validate_row_integration import OUT,ROOT,sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['variation','row'],required=True);a=parser.parse_args()
    dest=OUT/(a.phase+'_answers.jsonl');assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in p['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')));key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    old=[json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')];base={r['id']:r for r in old if r['arm']=='baseline'}
    cases={c.case_id:c for c in load_gold_dataset(ROOT/'tests/rag/fixtures/rag_gold_v4_1_review.json',reviewed_only=False).cases}
    cid='rag-biz-020' if a.phase=='variation' else 'rag-biz-023';repeats=5 if a.phase=='variation' else 3
    if a.phase=='variation':candidate=next(r for r in json.loads((ROOT/'reports/rag_material_repairs_20261105/selection_cases.json').read_text(encoding='utf-8')) if r['id']==cid and r['arm']=='S_native')
    else:
        candidate=next(r for r in (json.loads(l) for l in (OUT/'cases.jsonl').open(encoding='utf-8')) if r['id']==cid and r['arm']=='H8')
        assert candidate['gained'] and not candidate['lost'] and not candidate['new_negative_quotes'],'No qualifying H8 evidence gain'
    jobs=[]
    for arm in ('baseline','candidate'):
        row=base[cid] if arm=='baseline' else candidate
        for repeat in range(1,repeats+1):jobs.append({'id':cid,'arm':arm,'repeat':repeat,'context':row['context'],'entries':row.get('entries',row.get('context_entries'))})
    random.Random(20261105).shuffle(jobs)
    def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    dump(a.phase+'_answer_protocol.json',{'started':str(datetime.now()),'repeats':repeats,'model':settings.rag_api_model,'phase':a.phase,'purpose':'020 identical-input stochasticity diagnostic; no generation prompt change. H8/023 targeted source-row integration check, not full-set answer scores.','inputs':[{'arm':j['arm'],'repeat':j['repeat'],'context_sha256':hashlib.sha256(j['context'].encode()).hexdigest()} for j in jobs]})
    def run(j):
        gateway=OpenAICompatibleGateway(max_transient_retries=2);r={k:v for k,v in j.items() if k not in ('context','entries')};start=time.monotonic()
        try:
            result=gateway.generate(GenerationRequest(question=cases[cid].question,context=j['context'],model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(cases[cid].question))),CancelToken())
            answer={'mode':'full','text':result.text,'status':base[cid]['retrieval_status'],'source_ids':[e['source_id'] for e in j['entries']]};r['answer']=answer;r['citation_check']=asdict(CitationValidator().inspect(result.text,{e['source_id']:() for e in j['entries']}));r['judgement']=GoldAnswerJudge(gateway,settings,key).evaluate(cases[cid],answer,CancelToken())
        except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:300]
        r['elapsed_ms']=round((time.monotonic()-start)*1000);return r
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(run,j) for j in jobs]):
            r=future.result();results.append(r)
            with dest.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            print(a.phase,len(results),r['arm'],r['repeat'],(r.get('judgement') or {}).get('verdict'),bool(r.get('error')),flush=True)
    assert all(sha(k)==h for k,h in p['hashes'].items())
    dump(a.phase+'_answer_completion.json',{'finished':str(datetime.now()),'rows':len(results),'errors':sum(bool(r.get('error')) for r in results),'hashes_unchanged':True})

if __name__=='__main__':main()
