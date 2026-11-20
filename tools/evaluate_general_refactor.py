"""Frozen Context comparisons, unchanged Gold judge, and planned-answer ablation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.context_builder import ContextBundle, organize_context
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway, SYSTEM_PROMPT
from local_full_text_search.rag.generation.structured import generate_structured, PLAN_PROMPT, REVIEW_PROMPT
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rag_general_refactor_20261113'


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def capture_rows(path):
    result=[]
    with path.open(encoding='utf8') as stream:
        for line in stream:
            if not line.strip():continue
            if not line.endswith('\n'):break
            raw=json.loads(line)
            digest=hashlib.sha256(json.dumps(raw,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
            result.append({**{k:raw[k] for k in ('id','question','context','entries','result','cohort') if k in raw},'_capture_sha256':digest})
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm', choices=['A','B'], required=True)
    parser.add_argument('--structured', action='store_true')
    parser.add_argument('--stability', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--follow', action='store_true')
    parser.add_argument('--revision', default='')
    parser.add_argument('--output-dir', type=Path)
    args=parser.parse_args()
    if args.output_dir: OUT=args.output_dir.resolve()
    mode='structured' if args.structured else 'fixed_prompt'
    if args.revision and not args.revision.replace('_','').isalnum(): raise ValueError('Invalid revision')
    target=OUT/(args.arm+'_'+mode+('_'+args.revision if args.revision else '')+('_stability' if args.stability else '')+'.jsonl')
    if target.exists() and not args.resume: raise FileExistsError(target)
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases}
    captures=capture_rows(OUT/(args.arm+'.jsonl'))
    expected_count=sum(d['count'] for d in manifest['datasets'])
    if len(captures)!=expected_count and not args.follow: raise ValueError('Incomplete frozen capture')
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')))
    secret=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not secret: raise RuntimeError('No saved credential')
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    if bundle.bundle_id!=manifest['bundle_id']: raise ValueError('Published bundle changed')
    prompts={'base':SYSTEM_PROMPT,'plan':PLAN_PROMPT,'review':REVIEW_PROMPT}
    protocol={'mode':mode,'arm':args.arm,'context_source':args.arm+'.jsonl','model':settings.rag_api_model,
        'repeat':3 if args.stability else 1,'quality_retry':False,'labels_runtime_access':False,
        'prompt_hashes':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in prompts.items()},
        'limit':settings.rag_api_timeout_seconds,'context_hash':'per-row capture_sha256; append-only streaming' if args.follow else hashlib.sha256((OUT/(args.arm+'.jsonl')).read_bytes()).hexdigest(),
        'note':'A/B fixed_prompt uses identical system prompt, no dynamic plan instructions; structured is a separate generation ablation. Native retrieval gates retained. Not a fresh end-to-end run.'}
    protocol_path=target.with_suffix('.protocol.json')
    if args.revision:
        protocol['revision']=args.revision
        protocol['structured_code_sha256']=hashlib.sha256((ROOT/'local_full_text_search/rag/generation/structured.py').read_bytes()).hexdigest()
    if args.resume:
        if json.loads(protocol_path.read_text(encoding='utf8'))!=protocol: raise ValueError('Resume protocol drift')
    else: protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    done={(r['id'],r['repeat']) for r in rows(target)} if target.exists() else set()
    interpreter=QueryInterpreter(enhanced=True,domain_config=settings.rag_domain_config)

    def work(capture, repeat):
        cid=capture['id'];case=cases[cid]
        assert case.question==capture['question']
        row={'id':cid,'arm':args.arm,'repeat':repeat,'mode':mode,'question':case.question,
             'review_status':case.review_status,'cohort':capture['cohort'],'context':capture['context'],
             'context_sha256':hashlib.sha256(capture['context'].encode()).hexdigest()}
        row['capture_sha256']=capture['_capture_sha256']
        gateway=OpenAICompatibleGateway(max_transient_retries=2);token=CancelToken()
        request=GenerationRequest(case.question,capture['context'],settings.rag_api_model,settings.rag_api_base_url,
                                  settings.rag_api_style,secret,settings.rag_api_timeout_seconds,1000, '')
        try:
            if not capture['context']:
                result=capture.get('result',{})
                answer=result.get('clarification_question') or '资料中没有足够依据。'
                row['gate_reason']=result.get('reason','empty_context')
            elif args.structured:
                plan=interpreter.interpret(case.question)
                refs={}
                with store.connect_read_only(bundle.metadata_path) as conn:
                    for e in capture['entries']:
                        refs[e['source_id']]=tuple(store.load_chunk(conn,cid).source for cid in e.get('member_chunk_ids',[e['chunk_id']]))
                context=ContextBundle(capture['context'],tuple(s for values in refs.values() for s in values),refs,tuple(capture['entries']))
                context=organize_context(context,plan)
                logs=[]
                answer=generate_structured(gateway,request,plan,context,token,logs.append).text
                row['structured_diagnostics']=logs
            else:
                answer=gateway.generate(request,token).text
            row['answer']={'text':answer,'status':'answered' if capture['context'] else capture.get('result',{}).get('status','insufficient'),
                'source_ids':list(dict.fromkeys(__import__('re').findall(r'\[(S\d+)\]',answer)))}
            if not args.stability:
                row['judgement']=GoldAnswerJudge(gateway,settings,secret).evaluate(case,{**row['answer'],'evidence_context':capture['context']},token)
            row['signature']={'citations':sorted(row['answer']['source_ids']),
                'numbers':sorted(set(__import__('re').findall(r'\d+(?:\.\d+)?',answer))),
                'scope_models':sorted(set(__import__('re').findall(r'[A-Za-z]{1,6}-\d{3,6}[A-Za-z]?',answer))),
                'certainty_markers':sorted(set(__import__('re').findall(r'可能|已确认|一定|确定|不能确定',answer)))}
        except Exception as exc: row['error']=str(exc).replace(secret,'[REDACTED]')
        return row

    tasks=[(cap,repeat) for cap in captures if not args.stability or cap['id'] in freeze['focus_ids']
           for repeat in range(1,4 if args.stability else 2) if (cap['id'],repeat) not in done]
    def save(f):
            r=f.result()
            with target.open('a',encoding='utf8') as out:out.write(json.dumps(r,ensure_ascii=False)+'\n')
            print(r['id'],r['repeat'],r.get('judgement',{}).get('verdict'),r.get('error',''),flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(work,*task) for task in tasks}
        submitted=done | {(cap['id'],repeat) for cap,repeat in tasks}
        if args.follow:
            import time
            current=captures
            last_size=(OUT/(args.arm+'.jsonl')).stat().st_size
            while True:
                try:
                    current_size=(OUT/(args.arm+'.jsonl')).stat().st_size
                    if current_size!=last_size:
                        current=capture_rows(OUT/(args.arm+'.jsonl'));last_size=current_size
                except json.JSONDecodeError:
                    time.sleep(1);continue
                for cap in current:
                    if args.stability and cap['id'] not in freeze['focus_ids']:continue
                    for repeat in range(1,4 if args.stability else 2):
                        if (cap['id'],repeat) not in submitted:
                            submitted.add((cap['id'],repeat));futures.add(pool.submit(work,cap,repeat))
                completed,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED) if futures else (set(),set())
                for f in completed: save(f);futures.remove(f)
                if (OUT/(args.arm+'_complete.json')).exists() and len(current)==expected_count and not futures:break
                if not futures:time.sleep(2)
        else:
            for f in as_completed(futures):save(f)


if __name__=='__main__':main()
