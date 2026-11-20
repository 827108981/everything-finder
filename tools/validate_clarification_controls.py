"""Fixed controls for specific conditions, comparisons and missing evidence.

Real-document controls reuse frozen context; synthetic controls test logic only.
These are diagnostic pending cases, not added to reviewed Gold.
"""
import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def main():
    out=Path(sys.argv[1]) if len(sys.argv)>1 else Path('reports/rubric_clarification_20261102')
    assert '--retry-failed' in sys.argv or not (out/'controls.json').exists()
    originals=[r for r in map(json.loads,Path('reports/project_constraint_fix_20261102/cases.jsonl').read_text(encoding='utf-8').splitlines()) if r['arm']=='after']
    baseline={r['id']:r for r in originals}
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    cases={c.case_id:c for c in load_gold_dataset(Path('tests/rag/fixtures/rag_gold_v3.jsonl')).cases}
    controls=[
        ('known_unmarked','rag-biz-014','CL-2000i原机型，用的是无标识磁分离盘，吸液流量按多少调？','answer'),
        ('known_bm10','rag-biz-014','CL-2000i原机型，用的是BM10盘，吸液流量按多少调？','answer'),
        ('compare_discs','rag-biz-014','请分别列出CL-2000i原机型无标识盘和BM10盘的吸液流量标准。','answer'),
        ('known_network','rag-biz-049','CL-6000i SDM3.0网口模块的非运动参数能一键保存吗？','answer'),
        ('known_serial','rag-biz-049','CL-6000i SDM1.0串口模块的非运动参数怎样保存？','answer'),
        ('compare_versions','rag-biz-049','分别比较SDM1.0串口和SDM3.0网口模块保存非运动参数的方法。','answer'),
        ('known_current_fault','rag-biz-021','单项目LVC，已核查确认上次定标正常、本次定标有问题，现在先排查什么？','answer'),
        ('compare_lvc_paths','rag-biz-021','LVC在上次定标异常与本次定标异常这两种情况下，处理思路分别是什么？','answer'),
        ('synthetic_unknown_version',None,'这个模块应该怎么保存参数？','clarify'),
        ('synthetic_known_version',None,'版本B的模块应该怎么保存参数？','answer'),
        ('synthetic_compare',None,'版本A和版本B保存参数的方式分别是什么？','answer'),
        ('synthetic_corpus_gap',None,'这个模块断电后要等待多久才能恢复？','refuse'),
        ('synthetic_unique_capability',None,'我现在这个模块能不能一键保存所有参数？','clarify'),
    ]
    tasks=[]
    for name,cid,q,behavior in controls:
        if cid:
            context=next(e['llm_context'] for e in baseline[cid]['events'] if e.get('llm_context'))
            template=cases[cid]
        else:
            context='[S1]\n来源：构造的逻辑测试材料（非真实业务资料）\n版本A逐项保存参数，版本B支持一键保存所有参数。本文没有记载断电恢复等待时长。'
            template=cases['rag-biz-049']
        raw={**template.raw,'question':q,'expected_behavior':behavior,'answer_checks':[],
             'required_facts':[],'forbidden_claims':[],'review_status':'pending'}
        case=dataclasses.replace(template,question=q,expected_behavior=behavior,
            answer_checks=(),required_facts=(),forbidden_claims=(),missing_fields=(),raw=raw)
        tasks.append((name,q,context,case,'real_frozen_context' if cid else 'synthetic_logic'))
    (out/'controls_protocol.json').write_text(json.dumps({'note':'Fixed controls; no retrieval rerun for modified queries; judge behavior only, not all answer facts.',
        'controls':[{'id':n,'question':q,'expected':c.expected_behavior,'source':s} for n,q,ctx,c,s in tasks]},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def run(task):
        name,q,context,case,source=task
        gateway=OpenAICompatibleGateway(max_transient_retries=1);token=CancelToken()
        request=GenerationRequest(question=q,context=context,
            model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,
            api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,
            query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(q)))
        assessment=gateway.assess_clarification(request,token)
        needs_clarification=bool(assessment and assessment.needs_clarification)
        text=assessment.question if needs_clarification else gateway.generate(request,token).text
        answer={'text':text,'status':'needs_clarification' if needs_clarification else 'answered'}
        judgement=GoldAnswerJudge(gateway,settings,key).evaluate(case,answer,token)
        return {'id':name,'question':q,'expected_behavior':case.expected_behavior,'source':source,
            'needs_clarification':needs_clarification,
            'assessment':dataclasses.asdict(assessment) if assessment else None,
            'answer':answer,'judgement':judgement,'review_status':'pending'}
    attempts_path=out/'controls_attempts.jsonl'
    latest={}
    if attempts_path.exists():
        latest={r['id']:r for r in map(json.loads,attempts_path.read_text(encoding='utf-8').splitlines())}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(run,task):task[0] for task in tasks if task[0] not in latest or 'error_type'in latest[task[0]]}
        for f in as_completed(futures):
            name=futures[f]
            try:
                value=f.result()
            except Exception as exc:
                value={'id':name,'error_type':type(exc).__name__,'error_message':str(exc).replace(key,'[REDACTED]')[:200]}
            latest[name]=value
            with attempts_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(value,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
    results=[latest[t[0]] for t in tasks]
    (out/'controls.json').write_text(json.dumps(results,ensure_ascii=False,indent=2).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    print([(r['id'],r.get('error_type') or r['judgement']['behavior'],r.get('needs_clarification')) for r in results])


if __name__=='__main__':main()
