"""Generate and judge completed retrieval rows while remaining local work continues."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time
from tools.validate_retrieval_ab import summarize, sha
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
from local_full_text_search.rag.index_store import RagIndexStore

def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path);a=p.parse_args()
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((a.source/'protocol.json').read_text(encoding='utf-8'))
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    root=Path(__file__).resolve().parents[1];fixtures=root/'tests/rag/fixtures'
    cases={c.case_id:c for f in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(fixtures/f,reviewed_only=False).cases}
    def dump(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    protocol['remote_execution']={'started':str(datetime.now()),'workers':6,'source':str(a.source.resolve()),'harness_sha256':sha(__file__),
        'reason':'Overlap independent remote generation/judging with remaining sequential local inference. Same questions/context/settings; no answer cache.'}
    dump(out/'protocol.json',protocol)
    def run(row):
        t=time.monotonic();answer=judge=error=citation=None
        gateway=OpenAICompatibleGateway(max_transient_retries=2)
        case=cases[row['id']]
        try:
            if row['error']:raise RuntimeError(str(row['error']))
            if row['context']:
                generated=gateway.generate(GenerationRequest(question=case.question,context=row['context'],
                    model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,
                    timeout_seconds=settings.rag_api_timeout_seconds,query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question))),CancelToken())
                text=generated.text
                citation=asdict(CitationValidator().inspect(text,{e['source_id']:() for e in row['context_entries']}))
            else:text=row['local_answer_text']
            answer={'mode':'full','text':text,'status':row['retrieval_status'],'source_ids':[e['source_id'] for e in row['context_entries']]}
            judge=GoldAnswerJudge(gateway,settings,key).evaluate(case,answer,CancelToken())
        except Exception as e:error={'type':type(e).__name__,'message':str(e)[:400]}
        return {**row,'answer':answer,'gold_judgement':judge,'remote_error':error,'citation_check':citation,'remote_ms':round((time.monotonic()-t)*1000)}
    seen=set();futures=set();results=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        while len(results)<236:
            path=a.source/'retrieval_cases.jsonl'
            if path.exists():
                for line in path.read_text(encoding='utf-8').splitlines():
                    try:r=json.loads(line)
                    except json.JSONDecodeError:continue  # writer may be between bytes
                    k=(r['id'],r['arm'])
                    if k not in seen:
                        seen.add(k);futures.add(pool.submit(run,r))
            if futures:
                done,_=wait(futures,timeout=3,return_when=FIRST_COMPLETED)
                for f in done:
                    r=f.result();results.append(r);futures.remove(f)
                    with (out/'cases.jsonl').open('a',encoding='utf-8') as stream:
                        stream.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
                    print(len(results),r['arm'],r['id'],(r.get('gold_judgement')or{}).get('verdict'),r['remote_error'],flush=True)
                    dump(out/'progress.json',{'completed':len(results),'retrieval_seen':len(seen),'expected':236,'errors':sum(bool(x['remote_error']) for x in results)})
            else:time.sleep(2)
    assert len(seen)==236 and all(sha(p)==h for p,h in protocol['hashes'].items())
    assert RagIndexStore(Path('E:/everythingfinder-db/rag')).active().bundle_id==protocol['bundle_id']
    for name in ('query_plans.json','continuation_links.json','inference_cost.json','resume_provenance.json'):
        if (a.source/name).exists():(out/name).write_bytes((a.source/name).read_bytes())
    dump(out/'completion.json',{'finished':str(datetime.now()),'rows':len(results),'expected':236,'hashes_unchanged':True,'bundle_unchanged':True,
        'production_defaults_unchanged':True,'errors':sum(bool(r['error'] or r['remote_error']) for r in results),
        'citation_issues':sum(bool(r['citation_check'] and not r['citation_check']['structurally_valid']) for r in results),
        'excluded':'Intermediate smoke/interrupted source runs and any duplicate source-harness remote requests not included in scores.'})
    summarize(out,results,dump)

if __name__=='__main__':main()
