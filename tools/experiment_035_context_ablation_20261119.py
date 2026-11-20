"""Diagnostic only: four frozen Context interventions, no runtime/Gold edits."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import argparse
import json
from pathlib import Path
import random
import re
import threading

from tools.validate_minimal_four_prompt_20261119 import (
    check_frozen, inputs, monitored_gateway, append as locked_append,
)
from tools.validate_prompt_factorial_20261119 import ROOT, HISTORY, sha, write
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    generation_system_prompt, provider_request_payload, OpenAICompatibleGateway,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

OUT = ROOT / 'reports/rag_035_context_ablation_20261119'
CASE = 'rag-biz-035'
ARMS = {'A': '原完整Context', 'B': '仅S1、S2直接证据',
        'C': 'S1、S2加S3室内失控资料', 'D': '原完整Context仅移除S3'}
REPEATS = 5


def prepare():
    prior_protocol = check_frozen()
    cases, contexts, settings = inputs()
    original = contexts[CASE]['context']
    parts = list(re.finditer(r'(?m)^\[(S\d+)\]\s*\n', original))
    sections = {m[1]: original[m.start():parts[i+1].start() if i+1<len(parts) else len(original)]
                for i,m in enumerate(parts)}
    assert len(sections) == 20 and ''.join(sections.values()) == original
    selected = {'A': list(sections), 'B': ['S1','S2'], 'C': ['S1','S2','S3'],
                'D': [s for s in sections if s != 'S3']}
    variants = {a: ''.join(sections[s] for s in ids) for a,ids in selected.items()}
    assert variants['A'] == original
    assert variants['A'].replace(sections['S3'], '', 1) == variants['D']
    assert variants['C'] == variants['B'] + sections['S3']
    assert settings.rag_api_model == prior_protocol['model']
    assert generation_system_prompt('minimal_four') == prior_protocol['prompts']['minimal_four']
    return cases[CASE], contexts[CASE], settings, variants, selected


def freeze():
    case, ctx, settings, variants, selected = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
        assert all(sha(p)==d for p,d in protocol['protected'].items()), 'Frozen input changed'
        assert protocol['context_sha256'] == {a:sha256(v.encode()).hexdigest() for a,v in variants.items()}
        return protocol
    write(OUT/'contexts.json', {a:dict(context=v,source_ids=selected[a],chars=len(v)) for a,v in variants.items()})
    paths = [Path(p) for p in check_frozen()['protected']]
    paths += [Path(__file__), OUT/'contexts.json']
    protocol = dict(case=CASE, question=case.question, date='2026-11-19',
        bundle_id=check_frozen()['bundle_id'], arms=ARMS, repetitions=REPEATS,
        expected_tasks=20, workers=2, model=settings.rag_api_model,
        prompt_variant='minimal_four', system_prompt=generation_system_prompt('minimal_four'),
        prompt_sha256=sha256(generation_system_prompt('minimal_four').encode()).hexdigest(),
        max_output_tokens=1000,temperature=0,shuffle_seed=2026111935,
        source_ids=selected,context_sha256={a:sha256(v.encode()).hexdigest() for a,v in variants.items()},
        protected={str(p):sha(p) for p in paths},
        changed='Only evidence blocks in frozen Context; preserve original wording, order, IDs, headers. No summary or relabeling.',
        hypotheses=['S3 induces unsupported masked-out-of-control explanation',
            'Model can add the same mechanism without S3 (background knowledge or other distractors)',
            'Deleting context may lose necessary facts despite reducing expansion'],
        primary_outcome='Unsupported claim about QC rules/target values masking real out-of-control, with quoted answer and actual citation.',
        secondary_outcomes=['Gold core coverage','other unsupported claims','invalid citations'],
        causal_limit='5 runs per arm are a pilot. Short/long differences also change length and distractors. B vs C and A vs D isolate S3 addition/removal; no padded length control. No fine-grained semantic causal claim.',
        retries='Gateway transient transport retry only. Save generated answer before judging. No quality-based regeneration; at most one recovery of execution failure.',
        stop='All 20 fresh tasks plus source-bound review. Do not extend repeats after seeing results.',
        service_scope='Generator-only cause isolation; not end-to-end acceptance or general selection policy.',
        gold_in_generation=False, default_settings_changed=False)
    write(OUT/'protocol.json',protocol)
    write(OUT/'step_status.json',dict(freeze='通过',source_binding='通过（复用相同源文件哈希及原资料核验）',
        paired_generation='待核实',manual_review='待核实',report='待核实',adoption='不适用：诊断实验不修改正式链路'))
    return protocol


def run():
    protocol = freeze()
    case, ctx, settings, variants, selected = prepare()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key, 'Missing configured credential'
    plan_summary = AnswerService._plan_summary(QueryInterpreter().interpret(case.question))
    def request_for(a):
        return GenerationRequest(question=case.question, context=variants[a], model=settings.rag_api_model,
            api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,
            timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=1000,
            query_plan_summary=plan_summary,prompt_variant='minimal_four')
    previous = next(json.loads(s) for s in (ROOT/'reports/rag_minimal_four_prompt_20261119/answers.jsonl').read_text(encoding='utf8').splitlines()
                    if json.loads(s)['id']==CASE and json.loads(s)['arm']=='minimal_four' and json.loads(s)['round']==1)
    assert provider_request_payload(request_for('A')) == previous['api_request'], 'A request drift'
    output = OUT/'answers.jsonl'
    prior = [json.loads(l) for l in output.read_text(encoding='utf8').splitlines()] if output.exists() else []
    done = {(r['arm'],r['round']) for r in prior}
    tasks=[]
    rng=random.Random(protocol['shuffle_seed'])
    for n in range(1,REPEATS+1):
        arms=list(ARMS);rng.shuffle(arms);tasks.extend((a,n) for a in arms)
    assert len(tasks)==20
    tasks=[t for t in tasks if t not in done]
    def worker(task):
        a,n=task; req=request_for(a)
        r=dict(id=CASE,arm=a,round=n,question=case.question,
            context_sha256=protocol['context_sha256'][a],context_chars=len(variants[a]),
            source_ids=selected[a],generator_only=True,gold_in_generation=False,
            api_request=provider_request_payload(req),generation_response_metadata=[])
        locked_append(OUT/'request_journal.jsonl',dict(arm=a,round=n,api_request=r['api_request']))
        try:
            r['answer']=monitored_gateway(r['generation_response_metadata']).generate(req,CancelToken()).text
            ci=CitationValidator().inspect(r['answer'],{s:() for s in selected[a]})
            r['citation_check']=dict(structurally_valid=ci.structurally_valid,cited_ids=list(ci.cited_ids),unknown_ids=list(ci.unknown_ids))
            locked_append(OUT/'generation_journal.jsonl',r)
            r['judgement']=GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=2),settings,key).evaluate(
                case,dict(status='answered',text=r['answer'],evidence_context=variants[a]),CancelToken())
        except Exception as exc:
            r['error']=str(exc).replace(key,'[REDACTED]')[:600]
        return r
    with ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            r=f.result();locked_append(output,r)
            print('RESULT',r['arm'],r['round'],r.get('judgement',{}).get('verdict','error'),r.get('error',''),flush=True)
    freeze()
    print('COMPLETE',len(prior)+len(tasks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','run'])
    {'freeze':freeze,'run':run}[parser.parse_args().action]()
