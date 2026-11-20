"""Paired prompt-only experiment; fixed existing Context, unchanged judge and Gold."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import threading

from tools.validate_prompt_factorial_20261119 import inputs, TARGETS, ROOT, HISTORY, write, sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway, generation_system_prompt, provider_request_payload
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

OUT=ROOT/'reports/rag_unified_prompt_20261119'
ARMS=('baseline','unified')
EXPECTED=250
LOCK=threading.Lock()


def append(path, row):
    with LOCK, path.open('a',encoding='utf8') as f:
        f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')


def monitored_gateway(metadata):
    from openai import OpenAI
    class Completion:
        def __init__(self, create): self.create_real=create
        def create(self, **payload):
            r=self.create_real(**payload)
            metadata.append(dict(model=getattr(r,'model',None),system_fingerprint=getattr(r,'system_fingerprint',None),
                finish_reasons=[getattr(c,'finish_reason',None) for c in getattr(r,'choices',[])],
                status=getattr(r,'status',None),incomplete_details=str(getattr(r,'incomplete_details',None)),
                usage=r.usage.model_dump() if getattr(r,'usage',None) else None))
            return r
    def factory(**kwargs):
        from types import SimpleNamespace
        c=OpenAI(**kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=Completion(c.chat.completions.create)),
            responses=Completion(c.responses.create),close=c.close)
    return OpenAICompatibleGateway(client_factory=factory,max_transient_retries=2)


def initialize():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():return check_frozen()
    cases,contexts,settings=inputs()
    bundle=RagIndexStore(Path('E:/everythingfinder-db/rag')).active()
    assert bundle.bundle_id=='g000033-71ec3c52f3'
    previous=json.loads((ROOT/'reports/rag_prompt_factorial_20261119/protocol.json').read_text(encoding='utf8'))
    assert generation_system_prompt('baseline')==previous['prompts']['baseline']
    assert not settings.rag_query_plan_enabled and not settings.rag_answer_review_enabled and not settings.rag_semantic_clarification_enabled
    protected=[Path(SETTINGS_PATH),HISTORY/'real_entry_simple.jsonl',HISTORY/'simple_native.jsonl',HISTORY/'report.html',
        ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json',bundle.metadata_path,Path('E:/everythingfinder-db/rag/rag.manifest.json')]
    protected+=list((ROOT/'local_full_text_search/rag').rglob('*.py'))
    protected+=list((ROOT/'tests/rag/fixtures').glob('*.jsonl'))+list((ROOT/'tests/rag/fixtures').glob('*_review.json'))
    source_paths=set()
    for r in contexts.values():
        for e in r['entries']:
            for s in e['sources']:
                if Path(s['file_path']).is_file():source_paths.add(Path(s['file_path']))
    protected+=sorted(source_paths)
    protocol=dict(bundle_id=bundle.bundle_id,cases=111,arms=list(ARMS),targets=list(TARGETS),expected_generation_tasks=EXPECTED,
        first_round_cases=111,targeted_rounds=3,workers=3,shuffle_seed=2026111902,model=settings.rag_api_model,
        api_style=settings.rag_api_style,max_output_tokens=1000,temperature=0,
        prompts={a:generation_system_prompt(a) for a in ARMS},protected={str(p):sha(p) for p in protected},
        source_files_frozen=len(source_paths),frozen_context_sha256={k:sha256(v['context'].encode()).hexdigest() for k,v in contexts.items()},
        changed_factors=['generation system prompt only; unified prompt exactly as reviewed; no examples'],
        unchanged=['Gold','judge prompt/parser','retrieval','BGE','selection','Context','persistent settings'],
        adapter=False,semantic_clarification=False,answer_review=False,excluded_repair_target='rag-biz-019 frozen, retained in denominator',
        acceptance='Explicit source-supported gains; no confirmed new material regression; target repairs stable over 3 runs and real service. Judge disputes separately.',
        recovery='One sequential pass only for failed transport/judge execution; never retry answer quality; reuse generated answer on judge failure.',
        stop='Complete fixed 250 tasks and 14 service calls; no prompt tuning after freeze; reject candidate if material regression.')
    write(OUT/'protocol.json',protocol)
    write(OUT/'step_status.json',dict(freeze='通过',prompt='通过（与用户确认文本一致，无示例）',unit='待核实',paired='待核实',real_entry='待核实',adoption='待核实'))
    print('FROZEN',len(cases),EXPECTED,settings.rag_api_model,flush=True)
    return protocol


def check_frozen():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    drift=[f for f,d in p['protected'].items() if sha(f)!=d]
    assert not drift,drift
    return p


def run(recovery=False):
    protocol=initialize()
    check_frozen()
    cases,contexts,settings=inputs()
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key,'Missing configured API credential'
    main_file=OUT/'answers.jsonl'
    prior=[json.loads(l) for l in main_file.read_text(encoding='utf8').splitlines()] if main_file.exists() else []
    tasks=[(k,a,1) for k in cases for a in ARMS]+[(k,a,n) for k in TARGETS for a in ARMS for n in (2,3)]
    random.Random(protocol['shuffle_seed']).shuffle(tasks)
    assert len(tasks)==EXPECTED
    output=OUT/('transport_recovery.jsonl' if recovery else 'answers.jsonl')
    if recovery:
        assert len(prior)==EXPECTED
        done={(r['id'],r['arm'],r['round']) for r in (json.loads(l) for l in output.read_text(encoding='utf8').splitlines())} if output.exists() else set()
        tasks=[r for r in prior if 'error'in r and (r['id'],r['arm'],r['round']) not in done]
    else:
        done={(r['id'],r['arm'],r['round']) for r in prior};tasks=[t for t in tasks if t not in done]
    def worker(t):
        if recovery:
            r={k:v for k,v in t.items() if k!='error'};r.update(original_error=t['error'],recovery_attempt=1)
            k,a,n=r['id'],r['arm'],r['round']
        else:
            k,a,n=t;r=dict(id=k,arm=a,round=n,question=cases[k].question,review_status=cases[k].review_status,
                expected_behavior=cases[k].expected_behavior,model=settings.rag_api_model,generator_only=True,gold_in_generation=False)
        case,ctx=cases[k],contexts[k]
        request=GenerationRequest(question=case.question,context=ctx['context'],model=settings.rag_api_model,
            api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,
            timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=1000,
            query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question)),prompt_variant=a)
        payload=provider_request_payload(request)
        if recovery:assert payload==r['api_request']
        r.update(api_request=payload,context_sha256=sha256(ctx['context'].encode()).hexdigest(),
            context_chars=len(ctx['context']),system_prompt_sha256=sha256(generation_system_prompt(a).encode()).hexdigest())
        assert r['context_sha256']==protocol['frozen_context_sha256'][k]
        append(OUT/'request_journal.jsonl',dict(id=k,arm=a,round=n,recovery=recovery,api_request=payload))
        meta=r.setdefault('generation_response_metadata',[])
        try:
            if 'answer' not in r:r['answer']=monitored_gateway(meta).generate(request,CancelToken()).text
            c=CitationValidator().inspect(r['answer'],{e['source_id']:() for e in ctx['entries']})
            r['citation_check']=dict(structurally_valid=c.structurally_valid,cited_ids=list(c.cited_ids),unknown_ids=list(c.unknown_ids),reason=c.reason)
            # A successfully generated answer is saved before judging; judge cannot overwrite it.
            append(OUT/'generation_journal.jsonl',{k:v for k,v in r.items() if k!='judgement'})
            r['judgement']=GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=2),settings,key).evaluate(
                case,dict(status='answered',text=r['answer'],evidence_context=ctx['context']),CancelToken())
        except Exception as exc:r['error']=str(exc).replace(key,'[REDACTED]')[:600]
        return r
    with ThreadPoolExecutor(max_workers=1 if recovery else protocol['workers']) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            r=f.result();append(output,r)
            print('RECOVERY' if recovery else 'RESULT',r['id'],r['arm'],r['round'],r.get('judgement',{}).get('verdict','error'),r.get('error',''),flush=True)
    check_frozen()
    print('COMPLETE','recovery' if recovery else 'main',len(tasks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['initialize','check','run','recover']);a=p.parse_args().action
    {'initialize':initialize,'check':check_frozen,'run':run,'recover':lambda:run(True)}[a]()
