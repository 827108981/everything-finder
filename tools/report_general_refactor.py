"""Report all frozen cases, losses, limitations and original-source navigation."""
from collections import Counter
from difflib import unified_diff
import hashlib
import html as html_module
import json
from pathlib import Path
import sqlite3
import xml.etree.ElementTree as ET

from local_full_text_search.rag.evaluation.reporting.comparison import write_comparison_report
from tools.evaluate_general_refactor import ROOT, OUT, rows

STAGES = ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')


def read(name):
    path=OUT/name
    return json.loads(path.read_text(encoding='utf8')) if path.exists() else {}


def main():
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases=[c for d in manifest['datasets'] for c in rows(ROOT/d['path'])]
    captures={arm:{r['id']:r for r in rows(OUT/(name+'.jsonl'))} for arm,name in [('off','A'),('on','B')]}
    assert all(set(c)=={r['id'] for r in cases} for c in captures.values())
    answers={(arm,r['id']):r for arm in ('A','B') for r in rows(OUT/(arm+'_fixed_prompt.jsonl'))}
    summary={'total':len(cases),'bundle':manifest['bundle_id'],'comparison':'Current enabled adapter versus refactored enabled adapter; identical fixed generation prompt',
             'cohorts':{},'gains':[],'losses':[],'answer_gains':[],'answer_losses':[],
             'retrieval_gain_answer_not_improved':[],'per_case':[],'inactive_equivalence':{},'metric_limitations':[]}
    for name in ('main_reviewed','main_diagnostic','regression_reviewed','regression_diagnostic'):
        ids=[c['id'] for c in cases if (c['id'].startswith('rag-biz') == name.startswith('main')) and ((c['review_status']=='reviewed') == name.endswith('reviewed'))]
        group={'count':len(ids),'arms':{}}
        for arm,answer_arm in [('off','A'),('on','B')]:
            metrics={}
            for stage in STAGES:
                eligible=[captures[arm][cid]['scores'][stage] for cid in ids if captures[arm][cid]['scores'].get(stage,{}).get('gold_evidence_total_declared')]
                n=len(eligible)
                metrics[stage]={'eligible_cases':n,'all_gold_retained':sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in eligible),
                    'MRR':sum(s.get('reciprocal_rank') or 0 for s in eligible)/n if n else None,
                    'mean_evidence_recall':sum(s.get('gold_evidence_recall') or 0 for s in eligible)/n if n else None}
                for k in (1,3,5):
                    metrics[stage]['recall@'+str(k)]=sum(sum(rank<=k for rank in s.get('gold_completion_ranks',{}).values())/s['gold_evidence_total_declared'] for s in eligible)/n if n else None
            group['arms'][arm]={'stages':metrics,'answers':dict(Counter(answers.get((answer_arm,cid),{}).get('judgement',{}).get('verdict','error_or_unfinished') for cid in ids))}
        summary['cohorts'][name]=group
    for case in cases:
        cid=case['id'];a=captures['off'][cid];b=captures['on'][cid]
        lost=[stage for stage in STAGES if b['scores'].get(stage,{}).get('gold_evidence_found',0)<a['scores'].get(stage,{}).get('gold_evidence_found',0)]
        af=a['scores']['context']['gold_evidence_found'];bf=b['scores']['context']['gold_evidence_found']
        av=answers.get(('A',cid),{}).get('judgement',{}).get('verdict');bv=answers.get(('B',cid),{}).get('judgement',{}).get('verdict')
        if bf>af:summary['gains'].append(cid)
        if bf<af:summary['losses'].append(cid)
        if av!='consistent' and bv=='consistent':summary['answer_gains'].append(cid)
        if av=='consistent' and bv!='consistent':summary['answer_losses'].append(cid)
        if bf>af and bv!='consistent':summary['retrieval_gain_answer_not_improved'].append(cid)
        first=lost[0] if lost else 'generation_or_judge' if av=='consistent' and bv!='consistent' else None
        summary['per_case'].append({'id':cid,'review_status':case['review_status'],'before_context_gold':af,'after_context_gold':bf,
            'before_verdict':av,'after_verdict':bv,'first_observed_evidence_loss':lost[0] if lost else None,
            'failure_layer':first,'context_identical':a['context']==b['context'],
            'reason':'Earliest observed declared-quote count decrease; not a proven causal explanation' if lost else 'No upstream declared-evidence count loss; inspect actual answer and judge' if first else 'No measured loss',
            'B_findings':answers.get(('B',cid),{}).get('judgement',{}).get('findings',[])})
    audit=read('architecture_audit.json');summary['architecture_audit']=audit
    summary['metric_limitations']=['Gold verdict is automatic consistency, not independent human correctness/business usability.',
        'Scope retention is reported as evidence diagnostics, not a gold-labelled semantic rate.',
        'No fair single scalar for unsupported claim rate when the unchanged judge does not enumerate all claims.',
        'The same prompt comparison isolates Context changes; structured generation is reported separately.']
    structured=rows(OUT/'B_structured.jsonl') if (OUT/'B_structured.jsonl').exists() else []
    summary['structured_verdicts']=dict(Counter(r.get('judgement',{}).get('verdict','error') for r in structured))
    summary['structured_errors']=[{'id':r['id'],'error':r.get('error') or log.get('error')} for r in structured for log in r.get('structured_diagnostics',[{}]) if r.get('error') or log.get('error')]
    v2=rows(OUT/'B_structured_v2.jsonl') if (OUT/'B_structured_v2.jsonl').exists() else []
    summary['structured_v2']={'count':len(v2),'verdicts':dict(Counter(r.get('judgement',{}).get('verdict','error') for r in v2)),
        'cohorts':{name:dict(Counter(r.get('judgement',{}).get('verdict','error') for r in v2 if (r['review_status']=='reviewed')==formal)) for name,formal in [('reviewed',True),('diagnostic',False)]},
        'rejection_reasons':dict(Counter(c['reason'] for r in v2 for log in r.get('structured_diagnostics',[]) for c in log.get('structural_rejections',[]))),
        'generation_errors':[{'id':r['id'],'error':log['error']} for r in v2 for log in r.get('structured_diagnostics',[]) if log.get('error')]}
    native=rows(OUT/'native_structured_v2.jsonl') if (OUT/'native_structured_v2.jsonl').exists() else []
    summary['native_structured_v2']={'count':len(native),'verdicts':dict(Counter(r.get('judgement',{}).get('verdict','error') for r in native)),
        'per_case':[{'id':r['id'],'verdict':r.get('judgement',{}).get('verdict'),'findings':r.get('judgement',{}).get('findings',[]),
                     'structured_event_present':any(e.get('stage')=='structured_answer_plan' for e in r.get('events',[]))} for r in native]}
    summary['inactive_equivalence']=read('disabled_compatibility.json')
    summary['unchanged_context_answer_changes']=[r['id'] for r in summary['per_case'] if r['context_identical'] and r['before_verdict']!=r['after_verdict']]
    summary['answer_binding_proxies']={}
    for name,items in [('A',list(r for (arm,_),r in answers.items() if arm=='A')),('B',list(r for (arm,_),r in answers.items() if arm=='B')),('structured_v2',v2)]:
        summary['answer_binding_proxies'][name]={key:dict(Counter((r.get('judgement',{}).get(key) or {}).get('status','unavailable') for r in items)) for key in ('source_binding','object_binding','identifier_check')}
    summary['metric_limitations']+=['Source/object binding are unchanged model-judge proxies, not audited correctness, scope accuracy, citation accuracy or business usability rates.',
        'No new scope-labelled Gold was created. Scope retention rate and unsupported claim rate remain not measured; raw facts and rejection logs are available.']
    summary['stability']={}
    for name in ('B_fixed_prompt_stability','B_structured_v2_stability'):
        path=OUT/(name+'.jsonl')
        if not path.exists():continue
        grouped={}
        for row in rows(path):grouped.setdefault(row['id'],[]).append(row)
        summary['stability'][name]={cid:{'repeats':len(rr),'errors':sum(bool(r.get('error')) for r in rr),
            'signature_count':len({json.dumps(r.get('signature'),sort_keys=True,ensure_ascii=False) for r in rr}),
            'accepted_claim_counts':[sum(len(log.get('answer_plan',{}).get('claims',[])) for log in r.get('structured_diagnostics',[])) for r in rr] if 'structured' in name else None,
            'claim_sets':[[(c.get('text'),c.get('scope'),c.get('certainty'),c.get('source_refs')) for log in r.get('structured_diagnostics',[]) for c in log.get('answer_plan',{}).get('claims',[])] for r in rr],
            'note':'Signature agreement is not proof of semantic equivalence; zero accepted claims in every run is stable failure, not quality success'} for cid,rr in grouped.items()}
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    index=read('index_freeze.json')
    with sqlite3.connect(Path(index['metadata_path']).resolve().as_uri()+'?mode=ro',uri=True) as db:
        sources={json.loads(r[0])['file_path'] for r in db.execute('SELECT source_json FROM rag_chunks')}
    write_comparison_report(OUT,cases,captures,answers,summary,sources)
    page=OUT/'report.html';html=page.read_text(encoding='utf8')
    html=html.replace('修改前 · Adapter 关闭','修改前 · 当前 Adapter 开启').replace('修改后 · Adapter 开启','修改后 · 通用重构 Adapter 开启')
    html=html.replace('关闭：答案判分','修改前：答案判分').replace('开启：答案判分','修改后：答案判分')
    html=html.replace('Adapter 关闭','当前 Adapter 开启')
    html=html.replace('自动判分：<b>未完成</b>','自动判分：<b>裁判报错（执行已结束）</b>')
    def pretty(value):
        return '<pre>'+html_module.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
    banner='<section class="panel"><h2>本轮结论：未通过开启验收</h2><p>两个开关继续默认关闭。主对照固定同一生成 Prompt，只比较修改前后 Context；结构化答案另做固定 Context 实验，不合并归因。</p><p>所有测试题、Gold、分组、状态及判分逻辑保持冻结；改动前工作区已有未提交修改，哈希比较以本轮开始时为准。</p><h3>两步生成 v2（与 B 使用同一 Context）</h3>'+pretty(summary['structured_v2'])+'<details><summary>关闭兼容与稳定性</summary>'+pretty({'disabled':summary['inactive_equivalence'],'stability':summary['stability']})+'</details></section>'
    html=html.replace('<main>','<main>'+banner,1)
    for r in v2:
        anchor='id="'+r['id']+'"';start=html.find(anchor)
        if start<0:continue
        pos=html.find('<div class="body">',start)+len('<div class="body">')
        extra='<details><summary>同一 B Context：结构化生成 v2、原文绑定和拒绝原因</summary>'+pretty(r.get('answer',{}).get('text',r.get('error','')))+pretty(r.get('judgement',{}))+pretty(r.get('structured_diagnostics',[]))+'</details>'
        html=html[:pos]+extra+html[pos:]
    unit=ET.parse(OUT/'unit_results.xml').getroot()
    unit_counts={key:sum(int(s.attrib.get(key,0)) for s in unit.iter('testsuite')) for key in ('tests','failures','errors','skipped')}
    html=html.replace('<main>','<main><section class="panel"><h2>单元验证与原生入口</h2>'+pretty({'unit':unit_counts,'native':summary['native_structured_v2']})+'<p>旧断言未删除：10 项失败仍是验收阻塞项。详见结论文件；不能把新增测试通过称为全套测试通过。</p></section>',1)
    formal_before=sum(g['arms']['off']['answers'].get('consistent',0) for n,g in summary['cohorts'].items() if n.endswith('reviewed'))
    formal_after=sum(g['arms']['on']['answers'].get('consistent',0) for n,g in summary['cohorts'].items() if n.endswith('reviewed'))
    formal_count=sum(g['count'] for n,g in summary['cohorts'].items() if n.endswith('reviewed'))
    concise=f'正式集合 {formal_count} 题：相同 Prompt 自动一致性 {formal_before} → {formal_after}；结构化生成 v2 一致 {summary["structured_v2"]["cohorts"]["reviewed"].get("consistent",0)}。裁判错误单列，不当成已确认错误答案。未审核题不进入正式分母。'
    explanation=(OUT/'failure_analysis.md').read_text(encoding='utf8') if (OUT/'failure_analysis.md').exists() else ''
    html=html.replace('<main>','<main><section class="panel"><h2>先看结果：未通过开启验收</h2><p>'+concise+'</p><details><summary>通俗解释：具体在哪里丢了，为什么</summary>'+pretty(explanation)+'</details></section>',1)
    page.write_text(html,encoding='utf8')
    baseline=Path(read('freeze.json')['snapshot']);changes=[];diff=[]
    for p in (ROOT/'local_full_text_search').rglob('*'):
        if p.suffix not in {'.py','.json'}:continue
        rel=p.relative_to(ROOT);old=baseline/rel
        before=old.read_text(encoding='utf8') if old.exists() else '';after=p.read_text(encoding='utf8')
        if before!=after:
            changes.append(str(rel));diff.extend(unified_diff(before.splitlines(True),after.splitlines(True),fromfile='before/'+str(rel),tofile='after/'+str(rel)))
    (OUT/'runtime_changes.diff').write_text(''.join(diff),encoding='utf8')
    unit=ET.parse(OUT/'unit_results.xml').getroot()
    failures=[{'test':c.attrib.get('name'),'detail':c.find('failure').attrib.get('message','')} for c in unit.iter('testcase') if c.find('failure') is not None]
    notes=['# 通用架构重构验证报告','', '## 结论','保持 Adapter 和两步生成默认关闭。删除特殊规则不等于已经实现业务质量提升；任何证据丢失、范围错误或稳定性问题均阻止默认采用。',
        '本报告使用冻结 91 题。同一 Prompt 全量 A/B 与两步生成分开。未修改 Gold 或判分规则。',
        '## 输入核验',json.dumps(audit,ensure_ascii=False,indent=2),'## 代码变更',*['- '+p for p in changes],
        '## 全量前后对照',json.dumps(summary['cohorts'],ensure_ascii=False,indent=2),
        '## 收益与退化',f"Context 完整引文增加：{summary['gains']}",f"Context 完整引文减少：{summary['losses']}",
        f"自动答案判分改善：{summary['answer_gains']}",f"自动答案判分退化：{summary['answer_losses']}",
        '## 退化题最早观察环节','|题目|首个引文丢失阶段|前后答案判分|Context 相同|','|---|---|---|---|']
    for r in summary['per_case']:
        if r['id'] in summary['losses'] or r['id'] in summary['answer_losses']:
            notes.append(f"|{r['id']}|{r['failure_layer']}|{r['before_verdict']} → {r['after_verdict']}|{r['context_identical']}|")
    notes+=['## 两步生成（独立实验）',json.dumps(summary['structured_verdicts'],ensure_ascii=False),
        '### 真实 AnswerService 入口',json.dumps(summary['native_structured_v2'],ensure_ascii=False,indent=2),
        '### 修正通用误伤后的 v2',json.dumps(summary['structured_v2'],ensure_ascii=False,indent=2),
        'v1 将条件概括误当成逐字引文核验；v2 保持原文引文硬核验，把条件和范围的忠实概括交给独立语义审查。两轮全部原始结果均保留，没有挑好结果替换。',
        '### 仍有确定的生成缺陷',
        '- 短引文未带入所在完整行的标识符时，硬校验会删掉正确动作。例如处理动作在同一行，但答案写出行首代码、引文只截动作。需要用已验证的结构化行跨度绑定，而不是仅放宽字符串判断。',
        '- 方法问题的请求字段识别仍可能漏掉“何时、更换还是延期”等表达，导致把所问的条件分支误判为未要求的更换结论。',
        '- 语义审核可能把原资料列表顺序误判成新增先后条件，或删除部分流程。下一轮须验证通用步骤关系提取，不能加某道题的豁免。',
        '## 关闭兼容性',json.dumps(summary['inactive_equivalence'],ensure_ascii=False,indent=2),
        '## 相同 Context 下判分变化',str(summary['unchanged_context_answer_changes']),
        '这些题的变化不能归因于检索改进；可能由生成或裁判波动造成。',
        '## 单元测试',f'失败 {len(failures)} 项，原始 JUnit 结果保留。没有删除或改写旧断言使其通过。',json.dumps(failures,ensure_ascii=False,indent=2),
        '## 尚未解决／无法通用化的部分',
        '- 缺少明确章节和场景关系时，不能仅凭相邻页连接；需要通用文档结构元数据支持，当前不猜测。',
        '- 仅文件名或目录名提供平台线索时，不能据此认定每个字段覆盖全部机型；多来源字段仍可分别引用，未证明全平台共识。',
        '- 通用语义审核仍依赖模型，字符串核验不能证明所有对象、条件和单位关系。最终正确性与业务可用性需要逐题人工复核。',
        '- 条件、分支和步骤仅保持原文关系；缺少结构信息时不推断唯一操作流程。',
        '- 当前测试集不是新的独立验收集；泛化证据为不同实体/字段的合成机制测试，不冒充业务泛化已经通过。',
        '## 测试入口',
        '```powershell\n.venv/Scripts/python.exe -m tools.audit_general_refactor\n.venv/Scripts/python.exe -m tools.check_general_disabled_compatibility\n.venv/Scripts/python.exe -m pytest tests/rag tests/test_settings_service.py -q\n.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm A\n.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm B\n.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm B --structured --revision v2\n.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm B --stability\n.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm B --structured --revision v2 --stability\n```',
        '稳定性全部重复保留在 JSONL，metrics.json 包含结构化事实、引用、数值、范围和确定性集合。没有用表面文字相似度代替事实一致性。']
    notes[3:3]=[concise,explanation]
    (OUT/'conclusion.md').write_text('\n\n'.join(notes),encoding='utf8')
    (OUT/'final_acceptance.json').write_text(json.dumps({'protected_unchanged':not audit.get('protected_drift'),
        'default_enable_recommendation':False,'runtime_changes':changes,'unit_failures':failures,
        'unit_counts':unit_counts,'context_losses':summary['losses'],'answer_losses':summary['answer_losses'],
        'native_structured_v2':summary['native_structured_v2'],'structured_v2':summary['structured_v2'],
        'status':'not_accepted','reason':'Evidence loss, generation false rejections, unstable claims and unresolved unit failures'},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:summary[k] for k in ('total','gains','losses','answer_gains','answer_losses','structured_verdicts')},ensure_ascii=True))
    from tools.render_general_native_report import main as render_native
    render_native()


if __name__=='__main__':main()
