"""Complete all changed H8 contexts; same frozen prompts and no quality retries."""
from tools.check_row_integration_answers import *
def main():
    dest=OUT/'remaining_row_answers.jsonl';assert not dest.exists()
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in p['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')));key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    old=[json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')];base={r['id']:r for r in old if r['arm']=='baseline'}
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    changed=[r for r in (json.loads(l) for l in (OUT/'cases.jsonl').open(encoding='utf-8')) if r['arm']=='H8' and r['changed'] and r['id']!='rag-biz-023']
    jobs=[]
    for row in changed:
        for arm in ('baseline','candidate'):
            r=base[row['id']] if arm=='baseline' else row
            for repeat in range(1,4):jobs.append({'id':row['id'],'arm':arm,'repeat':repeat,'context':r['context'],'entries':r.get('entries',r.get('context_entries'))})
    random.Random(20261105).shuffle(jobs)
    dump('remaining_answer_protocol.json',{'started':str(datetime.now()),'harness_sha256':sha(__file__),'jobs':[{k:v for k,v in j.items() if k not in ('context','entries')} for j in jobs],'purpose':'Complete the other five changed H8 contexts; 023 already has six fresh paired answers. No prompt/algorithm changes, no quality retries.'})
    def run(j):
        gateway=OpenAICompatibleGateway(max_transient_retries=2);r={k:v for k,v in j.items() if k not in ('context','entries')};start=time.monotonic();cid=j['id']
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
            print(len(results),r['id'],r['arm'],r['repeat'],(r.get('judgement') or {}).get('verdict'),bool(r.get('error')),flush=True)
    assert all(sha(k)==h for k,h in p['hashes'].items())
    dump('remaining_answer_completion.json',{'finished':str(datetime.now()),'rows':len(results),'errors':sum(bool(r.get('error')) for r in results),'hashes_unchanged':True})
if __name__=='__main__':main()
