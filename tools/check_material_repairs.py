from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import asdict
from datetime import datetime
import json,random,re,time
from tools.validate_material_repairs import OUT,ROOT,sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import GenerationRequest,TextCompletionRequest
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator

def main():
    assert not (OUT/'answer_results.jsonl').exists()
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));hashes={**p['hashes'],__file__:sha(__file__)}
    assert all(sha(k)==h for k,h in hashes.items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    cases={c.case_id:c for n in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/n,reviewed_only=False).cases}
    audit=json.loads((OUT/'header_audit.json').read_text(encoding='utf-8'));controls=[];seen=set()
    for t in audit:
        if t.get('reason')!='header_not_marked_in_source':continue
        rows=[r for r in t['raw_cells'][1:6] if any(r)]
        s=json.dumps(rows,ensure_ascii=False)
        if not rows or s in seen:continue
        seen.add(s);controls.append({'id':str(len(controls)+1),'rows':rows,'source_file':t['file'],'page':t['page'],'expected':False})
        if len(controls)==12:break
    dump('header_controls_input.json',controls)
    g=OpenAICompatibleGateway(max_transient_retries=2)
    try:
        res=g.complete_text(TextCompletionRequest(system_prompt='你只判断表格结构。输入是多个表格的前六行原始单元格。判断第一行是否确实为列名，后续行为数据。全是数据、上下层复杂表头或不确定时返回false。不要修改任何单元格，不使用领域答案。仅返回JSON {"tables":[{"id":"原id","is_header":true或false,"reason":"简短结构依据"}]}，每个id一次。',user_prompt=json.dumps([{'id':r['id'],'rows':r['rows']} for r in controls],ensure_ascii=False),model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=2000),CancelToken())
        raw=res if isinstance(res,str) else res.text
        vals=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))['tables'];assert {r['id'] for r in vals}=={r['id'] for r in controls}
        dump('header_controls_result.json',{'tables':vals,'false_positive_count':sum(r['is_header'] for r in vals),'count':len(controls),'scope':'Source-derived header-removal controls, not blind real-world accuracy.'})
    except Exception as e:dump('header_controls_result.json',{'error':str(e).replace(key,'[REDACTED]')[:300]})
    previous=[json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')]
    base={r['id']:r for r in previous if r['arm']=='baseline'}
    selected=[r for r in json.loads((OUT/'selection_cases.json').read_text(encoding='utf-8')) if r['arm']=='S_native' and r['changed']]
    jobs=[]
    for r in selected:
        for arm in ('baseline','candidate'):
            source=base[r['id']] if arm=='baseline' else r
            for repeat in range(1,4):jobs.append({'id':r['id'],'arm':arm,'repeat':repeat,'context':source['context'],'entries':source.get('entries',source.get('context_entries')),'status':base[r['id']]['retrieval_status']})
    random.Random(20261105).shuffle(jobs)
    dump('answer_protocol.json',{'started':str(datetime.now()),'hashes':hashes,'jobs':[{k:v for k,v in j.items() if k not in ('context','entries')} for j in jobs],'workers':6,'repeats':3,'model':settings.rag_api_model,'scope':'All seven changed contexts: six formal and one pending regression. 42 paired fresh answers, not a full-set score. Other selection variants identical, not redundantly regenerated. No H table bindings in generator.',
      'table_flag':'023 full CSDB/DCF table remains flagged. Diagnostic exception only: test row misuse without clearing rollout gate or modifying Gold.'})
    def run(j):
        g=OpenAICompatibleGateway(max_transient_retries=2);r={k:v for k,v in j.items() if k not in ('context','entries')};t=time.monotonic()
        try:
            generated=g.generate(GenerationRequest(question=cases[j['id']].question,context=j['context'],model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(cases[j['id']].question))),CancelToken())
            answer={'mode':'full','text':generated.text,'status':j['status'],'source_ids':[e['source_id'] for e in j['entries']]}
            r['answer']=answer;r['citation_check']=asdict(CitationValidator().inspect(generated.text,{e['source_id']:() for e in j['entries']}));r['judgement']=GoldAnswerJudge(g,settings,key).evaluate(cases[j['id']],answer,CancelToken())
        except Exception as e:r['error']=str(e).replace(key,'[REDACTED]')[:300]
        r['elapsed_ms']=round((time.monotonic()-t)*1000);return r
    results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            r=f.result();results.append(r)
            with (OUT/'answer_results.jsonl').open('a',encoding='utf-8') as s:s.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            print(len(results),r['id'],r['arm'],r['repeat'],(r.get('judgement') or {}).get('verdict'),bool(r.get('error')),flush=True)
    assert all(sha(k)==h for k,h in hashes.items())
    dump('answer_completion.json',{'finished':str(datetime.now()),'rows':len(results),'errors':sum(bool(r.get('error')) for r in results),'hashes_unchanged':True})

if __name__=='__main__':main()
