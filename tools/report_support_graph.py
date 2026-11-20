"""Summarize frozen native runs without changing the application scorer."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_support_graph_20261113'
STAGES=('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')


def read_rows(name):
    p=OUT/name
    return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines() if l.strip()] if p.exists() else []


def main():
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases=[json.loads(line) for d in manifest['datasets'] for line in (ROOT/d['path']).read_text(encoding='utf8').splitlines() if line.strip()]
    runs={a:{r['id']:r for r in read_rows(a+'.jsonl')} for a in ('A','B')}
    raw_b_verdicts=dict(Counter(r.get('judgement',{}).get('verdict','error') for r in runs['B'].values()))
    if (OUT/'B_effective.jsonl').exists():
        runs['B']={r['id']:r for r in read_rows('B_effective.jsonl')}
    fixed={a:{r['id']:r for r in read_rows(a+'_fixed_prompt.jsonl')} for a in ('A','B')}
    assert all(set(r)=={c['id'] for c in cases} for r in [*runs.values(),*fixed.values()])
    def verdict(row):return row.get('judgement',{}).get('verdict','error')
    def stage_metrics(records,stage):
        scores=[r['scores'].get(stage,r['scores']['context']) for r in records if r['scores']['context'].get('gold_evidence_total_declared')]
        n=len(scores)
        return {'cases':n,'all_gold_retained':sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in scores),
                'evidence_recall':sum(s.get('gold_evidence_recall') or 0 for s in scores)/n if n else None,
                'MRR':sum(s.get('reciprocal_rank') or 0 for s in scores)/n if n else None,
                **{'recall@'+str(k):sum(sum(rank<=k for rank in s.get('gold_completion_ranks',{}).values())/s['gold_evidence_total_declared'] for s in scores)/n if n else None for k in (1,3,5)}}
    summary={'bundle_id':manifest['bundle_id'],'case_count':len(cases),'cohorts':{},'per_case':[],
        'B_answer_artifact':'B_effective.jsonl' if (OUT/'B_effective.jsonl').exists() else 'B.jsonl',
        'B_original_native_verdicts':raw_b_verdicts,
        'answer_gains':[],'answer_losses':[],'evidence_gains':[],'evidence_losses':[],
        'evidence_improved_answer_not_improved':[], 'answer_changed_uncertain':[],
        'protocol':'Native A/B: same frozen data/model and primary SYSTEM_PROMPT; B activates additional planning/review calls. Fixed-prompt A/B separately isolates selected Context. No algorithm-only claim for the combined result.'}
    cohorts={'reviewed':[c['id'] for c in cases if c['review_status']=='reviewed'],
             'diagnostic':[c['id'] for c in cases if c['review_status']!='reviewed'],
             'main_reviewed':[c['id'] for c in cases if c['review_status']=='reviewed' and c['id'].startswith('rag-biz')],
             'regression_reviewed':[c['id'] for c in cases if c['review_status']=='reviewed' and not c['id'].startswith('rag-biz')]}
    for name,ids in cohorts.items():
        summary['cohorts'][name]={'count':len(ids),'arms':{}}
        for arm in ('A','B'):
            rr=[runs[arm][cid] for cid in ids]
            group=summary['cohorts'][name]['arms'][arm]={
                'native_verdicts':dict(Counter(verdict(r) for r in rr)),
                'fixed_prompt_verdicts':dict(Counter(verdict(fixed[arm][cid]) for cid in ids)),
                'stages':{s:stage_metrics(rr,s) for s in STAGES},
                'answer_statuses':dict(Counter(r.get('result',{}).get('status','error') for r in rr)),
                'behavior':dict(Counter(r.get('judgement',{}).get('behavior','unavailable') for r in rr)),
                'binding_proxies':{k:dict(Counter((r.get('judgement',{}).get(k) or {}).get('status','unavailable') for r in rr)) for k in ('source_binding','object_binding','identifier_check')}}
            grouped=defaultdict(list)
            for r in rr:grouped[r['group_id']].append(verdict(r)=='consistent')
            group['group_macro_consistency']=sum(sum(v)/len(v) for v in grouped.values())/len(grouped) if grouped else None
    for case in cases:
        cid=case['id'];a=runs['A'][cid];b=runs['B'][cid]
        ag=a['scores']['context']['gold_evidence_found'];bg=b['scores']['context']['gold_evidence_found']
        av,bv=verdict(a),verdict(b)
        losses=[s for s in STAGES if b['scores'].get(s,{}).get('gold_evidence_found',0)<a['scores'].get(s,{}).get('gold_evidence_found',0)]
        if ag<bg:summary['evidence_gains'].append(cid)
        if ag>bg:summary['evidence_losses'].append(cid)
        if av!='consistent' and bv=='consistent':summary['answer_gains'].append(cid)
        if av=='consistent' and bv!='consistent':summary['answer_losses'].append(cid)
        if ag<bg and bv!='consistent':summary['evidence_improved_answer_not_improved'].append(cid)
        if av!=bv and (av=='error' or bv=='error' or case['review_status']!='reviewed'):summary['answer_changed_uncertain'].append(cid)
        events=b['events'];logs=[e for e in events if e.get('stage')=='structured_answer_plan']
        graph=next((e['fact_support_graph'] for e in events if 'fact_support_graph' in e),{})
        r={'id':cid,'question':case['question'],'review_status':case['review_status'],
           'before_verdict':av,'after_verdict':bv,'context_gold_before':ag,'context_gold_after':bg,
           'first_evidence_count_loss':losses[0] if losses else None,
           'first_failure_layer':losses[0] if ag>bg and losses else ('generation_or_review_or_judge' if bv!='consistent' and bg==b['scores']['context']['gold_evidence_total_declared'] else 'evidence_incomplete' if bv!='consistent' else None),
           'context_identical':a['context']==b['context'],'findings':b.get('judgement',{}).get('findings',[]),
           'missing_checks':[x for x in b.get('judgement',{}).get('rubrics',[]) if x.get('status') in ('missing','incorrect','contradicted','uncertain','present')],
           'fallback':[e.get('fallback',{}) for e in logs],
           'clarification':b.get('result',{}).get('status') in ('clarify','needs_clarification'),
           'graph_nodes':len(graph.get('nodes',[])),'graph_edges':len(graph.get('edges',[])),
           'field_support':graph.get('field_support',[]),
           'answer_plans':[e.get('answer_plan',{}) for e in logs],
           'structural_rejections':[x for e in logs for x in e.get('structural_rejections',[])],
           'semantic_rejections':[x for e in logs for x in e.get('semantic_checks',[]) if x.get('supported') is not True],
           'result_error':b.get('error'),'judge_error':b.get('judge_error')}
        summary['per_case'].append(r)
    rows=summary['per_case']
    summary['fallback']={'cases':sum(any(x.get('used') for x in r['fallback']) for r in rows),
        'reasons':dict(Counter(x.get('reason') for r in rows for x in r['fallback'] if x.get('used'))),
        'verified_claims_added':sum(x.get('verified_claims_added',0) for r in rows for x in r['fallback'])}
    summary['graph']={'cases':sum(r['graph_nodes']>0 for r in rows),
        'field_statuses':dict(Counter(f['status'] for r in rows for f in r['field_support'])),
        'conflict_edges':sum(len(p.get('conflicts',[])) for r in rows for p in r['answer_plans']),
        'note':'Field detection is partial support, not independent semantic correctness.'}
    claims=[c for r in runs['B'].values() for c in r.get('result',{}).get('review_claims',[])]
    summary['review_statuses']=dict(Counter(c['status'] for c in claims))
    summary['clarification_ids']=[r['id'] for r in runs['B'].values() if r.get('result',{}).get('status') in ('clarify','needs_clarification')]
    summary['stability']={}
    for name in ('B_fixed_prompt_stability','B_structured_stability'):
        group=defaultdict(list)
        for r in read_rows(name+'.jsonl'):group[r['id']].append(r)
        summary['stability'][name]={cid:{'repeats':len(rr),'errors':sum(bool(r.get('error')) for r in rr),
            'signature_variants':len({json.dumps(r.get('signature'),sort_keys=True) for r in rr}),
            'claims':[[{'text':c['text'],'scope':c['scope'],'certainty':c['certainty'],'source_refs':c['source_refs'],'fact_ids':c.get('fact_ids',[])} for e in r.get('structured_diagnostics',[]) for c in e.get('answer_plan',{}).get('claims',[])] for r in rr],
            'note':'Numbers/scope/citation/claim variability is diagnostic; paraphrase variation alone is not semantic failure.'} for cid,rr in group.items()}
    summary['unmeasured_metrics']={'business_usability':'Requires human business review; no gold change authorized',
        'scope_accuracy':'Only unchanged judge source/object binding proxies; no complete scope-labelled denominator',
        'fact_support_rate':'LLM support assessment recorded, not independently ground-truthed',
        'conflict_detection_accuracy':'No exhaustive conflict Gold; synthetic conflict tests and graph events only',
        'unsupported_evidence_filter_rate':'No exhaustive candidate relevance labels; all selection reasons preserved'}
    stability=defaultdict(list)
    for row in read_rows('stability_judgements.jsonl'):
        stability[(row['mode'],row['id'])].append(row)
    summary['stability_gold_checks']={mode:{cid:{
        'repeats':len(rr),'verdicts':dict(Counter(verdict(r) for r in rr)),
        'context_variants':len({r['context_sha256'] for r in rr}),
        'core_rubrics':{str(r['repeat']):[(x['id'],x['status']) for x in r.get('judgement',{}).get('rubrics',[]) if x.get('role')=='core'] for r in rr},
        'fact_id_set_variants':len({tuple(r.get('fact_ids',[])) for r in rr}),
        'value_set_variants':len({tuple(r.get('answer_values_without_citations',[])) for r in rr})}
        for (m,cid),rr in stability.items() if m==mode} for mode in ('fixed_prompt','structured')}
    unit=ET.parse(OUT/'unit_results.xml')
    failures=[c.attrib['name'] for c in unit.findall('.//testcase') if c.find('failure') is not None]
    summary['unit_tests']={'total':len(unit.findall('.//testcase')),'failures':failures,'skipped':sum(c.find('skipped') is not None for c in unit.findall('.//testcase'))}
    (OUT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    focus=json.loads((OUT/'freeze.json').read_text(encoding='utf8'))['focus_ids']
    lines=['# 重点问题：原始答案与逐阶段记录','这里只展示冻结运行产物，不修改或替代 Gold。']
    for cid in focus:
        r=next(x for x in rows if x['id']==cid)
        lines.extend(['\n## '+cid, r['question'], '\n### A 实际回答',runs['A'][cid].get('result',{}).get('text','运行失败'),
            '\n### B 实际回答',runs['B'][cid].get('result',{}).get('text','运行失败'),
            '\n### 逐阶段完整引文计数', '\n'.join(f"- {s}: {runs['A'][cid]['scores'].get(s,{}).get('gold_evidence_found',0)} → {runs['B'][cid]['scores'].get(s,{}).get('gold_evidence_found',0)}" for s in STAGES),
            '\n### 缺失、回退和首个观察到的损失', '```json',json.dumps({k:r[k] for k in ['before_verdict','after_verdict','missing_checks','fallback','first_evidence_count_loss','first_failure_layer']},ensure_ascii=False,indent=2),'```'])
    (OUT/'focus_answers.md').write_text('\n'.join(lines),encoding='utf8')
    print(json.dumps({k:summary[k] for k in ['case_count','answer_gains','answer_losses','evidence_gains','evidence_losses','fallback']},ensure_ascii=False))


if __name__=='__main__':main()
