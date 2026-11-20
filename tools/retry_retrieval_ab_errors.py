"""Retry execution exceptions once; never retry an answer because of its verdict."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
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
from tools.validate_retrieval_ab import summarize

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);a=p.parse_args();out=a.out
    rows=[json.loads(l) for l in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    retry_path=out/'execution_retries.jsonl'
    retries=[json.loads(l) for l in retry_path.read_text(encoding='utf-8').splitlines()] if retry_path.exists() else []
    tried={(r['id'],r['arm']) for r in retries}
    s=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(s.rag_api_secret_ref) or '';assert key
    fixtures=Path(__file__).resolve().parents[1]/'tests/rag/fixtures'
    cases={c.case_id:c for f in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(fixtures/f,reviewed_only=False).cases}
    gateway=OpenAICompatibleGateway(max_transient_retries=2)
    for row in rows:
        if not row.get('remote_error') or (row['id'],row['arm']) in tried:continue
        r=dict(row);r['initial_remote_error']=row['remote_error'];r['retry_scope']='judge_only' if row.get('answer') else 'generation_and_judge'
        t=time.monotonic();r['remote_error']=None
        try:
            if r.get('answer') is None:
                c=cases[r['id']]
                generated=gateway.generate(GenerationRequest(question=c.question,context=r['context'],model=s.rag_api_model,
                    api_base_url=s.rag_api_base_url,api_style=s.rag_api_style,api_key=key,timeout_seconds=s.rag_api_timeout_seconds,
                    query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(c.question))),CancelToken())
                r['answer']={'mode':'full','text':generated.text,'status':r['retrieval_status'],'source_ids':[e['source_id'] for e in r['context_entries']]}
                r['citation_check']=asdict(CitationValidator().inspect(generated.text,{e['source_id']:() for e in r['context_entries']}))
            r['gold_judgement']=GoldAnswerJudge(gateway,s,key).evaluate(cases[r['id']],r['answer'],CancelToken())
        except Exception as e:r['remote_error']={'type':type(e).__name__,'message':str(e)[:400]}
        r['retry_ms']=round((time.monotonic()-t)*1000)
        with retry_path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
        retries.append(r)
        print(r['id'],r['arm'],r['retry_scope'],(r.get('gold_judgement')or{}).get('verdict'),r['remote_error'],flush=True)
    if len(rows)==236:
        lookup={(r['id'],r['arm']):r for r in retries}
        resolved=[lookup.get((r['id'],r['arm']),r) for r in rows]
        (out/'resolved_cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n' for r in resolved),encoding='utf-8')
        def dump(path,v):path.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
        dump(out/'retry_summary.json',{'initial_errors':sum(bool(r['remote_error']) for r in rows),'retried':len(retries),
            'remaining_errors':sum(bool(r['remote_error']) for r in resolved),'policy':'Only execution exceptions, once per row. Keep existing generated answer when only judging failed. No quality-based retry. Original cases.jsonl unchanged.'})
        summarize(out,resolved,dump)

if __name__=='__main__':main()
