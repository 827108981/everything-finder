"""Stream remote answer diagnostics; local retrieval timing remains a separate metric."""
from tools.check_row_integration_answers import *
from concurrent.futures import wait,FIRST_COMPLETED
OUT=ROOT/'reports/rag_row_holdout_20261105'
def main():
    assert not (OUT/'answers.jsonl').exists()
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')));key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    cases={c.case_id:c for c in load_gold_dataset(ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json',reviewed_only=False).cases}
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    dump('answer_protocol.json',{'started':str(datetime.now()),'harness_sha256':sha(__file__),'generation_workers':4,'selection':'Only repetition1 local results;10pending questions x2arms. Other timing repetitions are not answer trials.','latency':'Generation duration includes network; judge duration separate. Remote four-way concurrency means these are not single-user end-to-end latency. No judge included in response-time claims.','no_local_model_competition':'No other local torch inference launched. Hosted API requests may add minor client CPU/network load, normal desktop noise not controlled.'})
    def run(row):
        r={'id':row['id'],'arm':row['arm'],'context_sha256':hashlib.sha256(row['context'].encode()).hexdigest()};g=OpenAICompatibleGateway(max_transient_retries=2)
        try:
            start=time.perf_counter()
            if row['context']:
                v=g.generate(GenerationRequest(question=cases[row['id']].question,context=row['context'],model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(cases[row['id']].question))),CancelToken());text=v.text
            else:text=row['local_text']
            r['generation_ms']=(time.perf_counter()-start)*1000
            answer={'mode':'full','text':text,'status':row['status'],'source_ids':[e['source_id'] for e in row['entries']]};r['answer']=answer;r['citation_check']=asdict(CitationValidator().inspect(text,{e['source_id']:() for e in row['entries']}))
            start=time.perf_counter();r['judgement']=GoldAnswerJudge(g,settings,key).evaluate(cases[row['id']],answer,CancelToken());r['judge_ms']=(time.perf_counter()-start)*1000
        except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:300]
        return r
    seen=set();pending=set();results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        while len(results)<20:
            path=OUT/'timing_cases.jsonl'
            if path.exists():
                for l in path.read_text(encoding='utf-8').splitlines():
                    try:row=json.loads(l)
                    except json.JSONDecodeError:continue
                    pair=row['id'],row['arm']
                    if row['repeat']==1 and pair not in seen:seen.add(pair);pending.add(pool.submit(run,row))
            if pending:
                done,_=wait(pending,timeout=2,return_when=FIRST_COMPLETED)
                for f in done:
                    r=f.result();pending.remove(f);results.append(r)
                    with (OUT/'answers.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
                    print('answer',len(results),r['id'],r['arm'],(r.get('judgement') or {}).get('verdict'),r.get('error'),flush=True)
            else:time.sleep(2)
    dump('answer_completion.json',{'finished':str(datetime.now()),'rows':len(results),'errors':sum(bool(r.get('error')) for r in results)})
if __name__=='__main__':main()
