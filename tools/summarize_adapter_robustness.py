"""Three-way, source-bound metrics and explicit changes; no Gold mutation."""
from collections import Counter
import re
import argparse
from tools.rag_eval_common import ROOT,read,lines,write,sha
from tools.validate_query_plan_adapter import FILES

OUT=ROOT/'reports/rag_adapter_robustness_20261111'
OLD=ROOT/'reports/rag_query_plan_adapter_20261111'
STAGES=('lexical','vector','fusion','rerank','evidence','context')

def aggregate(rows,stage):
    valid=[r['scores'][stage] for r in rows if r['scores'].get(stage,{}).get('gold_evidence_total_declared',0)>0]
    denom=sum(s['gold_evidence_total_declared'] for s in valid)
    result={'questions_with_gold':len(valid),'gold_quotes':denom,
            'complete_quotes_retained':sum(s['gold_evidence_found'] for s in valid),
            'MRR':sum(s['reciprocal_rank'] or 0 for s in valid)/len(valid) if valid else None,
            'all_gold_retained_questions':sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in valid)}
    for k in (1,3,5):
        result[f'Recall@{k}']=sum(sum(rank<=k for rank in s['gold_completion_ranks'].values()) for s in valid)/denom if denom else None
    result['full_quote_retention']=result['complete_quotes_retained']/denom if denom else None
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',action='store_true');args=ap.parse_args()
    suffix='_release' if args.release else ''
    candidate_name='native_release' if args.release else 'native_v2'
    runs={'disabled':{r['id']:r for r in lines(OLD/'baseline.jsonl')},
          'previous_adapter':{r['id']:r for r in lines(OLD/'candidate.jsonl')},
          'robust_adapter':{r['id']:r for r in lines(OUT/(candidate_name+'.jsonl'))}}
    assert all(len(r)==92 for r in runs.values())
    answers={arm:{} for arm in 'AB'};errors=[]
    for path in ([OUT/'final_answers_AB.jsonl'] if args.release else sorted(OUT.glob('answers_AB_part*.jsonl'))):
        for r in lines(path):
            if r.get('error'): errors.append({'file':path.name,**r});continue
            if r.get('judgement') and r.get('context_equal'):
                assert r['id'] not in answers[r['arm']],r['id']
                answers[r['arm']][r['id']]={**r,'file':path.name}
    previous={}
    for path in sorted(OLD.glob('answers_B*.jsonl')):
        for r in lines(path):
            if r.get('judgement') and not r.get('error'):previous[r['id']]=r
    comparisons=[]
    for cid,b in runs['robust_adapter'].items():
        changes={}
        for name in ('disabled','previous_adapter'):
            a=runs[name][cid]
            old=set(a['scores']['context']['gold_completion_ranks']);new=set(b['scores']['context']['gold_completion_ranks'])
            changes[name]={'gained':sorted(new-old),'lost':sorted(old-new),'same_context':a['context']==b['context']}
        comparisons.append({'id':cid,'question':b['question'],'cohort':b['cohort'],'review_status':b['review_status'],
            'changes':changes,'answer_disabled':answers['A'].get(cid,{}).get('judgement',{}).get('verdict'),
            'answer_previous_adapter':previous.get(cid,{}).get('judgement',{}).get('verdict'),
            'answer_robust_adapter':answers['B'].get(cid,{}).get('judgement',{}).get('verdict'),
            'active':next((e['query_plan']['adapter_active'] for e in b['events'] if 'query_plan' in e),False),
            'answer_text_disabled':answers['A'].get(cid,{}).get('answer',{}).get('text'),
            'answer_text_robust_adapter':answers['B'].get(cid,{}).get('answer',{}).get('text')})
    cohorts={}
    for cohort in dict.fromkeys(r['cohort'] for r in comparisons):
        selected=[r for r in comparisons if r['cohort']==cohort];ids=[r['id'] for r in selected]
        cohorts[cohort]={'count':len(ids),'answers':{arm:dict(Counter(answers[arm].get(cid,{}).get('judgement',{}).get('verdict','not_completed') for cid in ids)) for arm in 'AB'},
            'metrics':{arm:{stage:aggregate([rows[cid] for cid in ids],stage) for stage in STAGES} for arm,rows in runs.items()}}
        cohorts[cohort]['answer_checks']={}
        for arm in 'AB':
            js=[answers[arm][cid]['judgement'] for cid in ids if cid in answers[arm]]
            cohorts[cohort]['answer_checks'][arm]={
                'source_binding':dict(Counter(j.get('source_binding',{}).get('status','unknown') for j in js)),
                'object_binding':dict(Counter(j.get('object_binding',{}).get('status','unknown') for j in js)),
                'behavior_match':dict(Counter(j.get('behavior','unknown') for j in js)),
                'core_fact_checks':dict(Counter(r['status'] for j in js for r in j.get('rubrics',[]) if r.get('role')=='core' and r.get('kind')=='required_facts')),
                'interpretation':'Automatic rubric/binding signals only. Object binding is a relevance proxy; no independent numeric relevance or semantic scope score is invented.'}
    scopes=[];structured=[]
    for cid,row in runs['robust_adapter'].items():
        candidate_map={x['chunk_id']:x for x in row['candidate_stages'].get('evidence',[])}
        for e in row['entries']:
            diagnostics=e.get('query_plan_diagnostics',{})
            # Denominator comes from the pre-budget source observations, limited
            # to the intended projected source spans. Not the filtered output list.
            intended=candidate_map.get(e['chunk_id'],{}).get('query_plan_diagnostics',{}).get('scope_observations',[])
            if e.get('source_spans'):
                intended=[o for o in intended if any(s['start']<=o['start'] and o['end']<=s['end'] for s in e['source_spans'])]
            for obs in intended:
                scopes.append({'id':cid,'source_id':e['source_id'],**obs,'retained':obs['text'] in e['included_text']})
            for r in diagnostics.get('structured_rows',[]):
                structured.append({'id':cid,'source_id':e['source_id'],'identifier':r['identifier'],
                    'complete':all(cell['quote'] in e['included_text'] for cell in r['slots'].values()) and r['identifier'] in e['included_text']})
    manifest=read(OUT/(candidate_name+'_manifest.json'))
    summary={'bundle':manifest['bundle'],'retrieval_cases':92,'valid_answers':{a:len(r) for a,r in answers.items()},
        'errors':[{'id':r['id'],'arm':r['arm'],'error':r['error'],'file':r['file']} for r in errors],
        'cohorts':cohorts,'active_count':sum(r['active'] for r in comparisons),
        'scope_mentions':len(scopes),'scope_mentions_retained':sum(s['retained'] for s in scopes),
        'complete_structured_rows':sum(r['complete'] for r in structured),'selected_structured_rows':len(structured),
        'code_drift':[p for p,h in manifest['code_hashes'].items() if sha(p)!=h],
        'gold_drift':[p for p,h in manifest['dataset_hashes'].items() if sha(p)!=h],
        'answer_regressions_vs_disabled':[r['id'] for r in comparisons if r['answer_disabled']=='consistent' and r['answer_robust_adapter'] not in ('consistent',None)],
        'answer_gains_vs_disabled':[r['id'] for r in comparisons if r['answer_robust_adapter']=='consistent' and r['answer_disabled'] not in ('consistent',None)],
        'evidence_losses_vs_previous':[r['id'] for r in comparisons if r['changes']['previous_adapter']['lost']],
        'evidence_gains_vs_previous':[r['id'] for r in comparisons if r['changes']['previous_adapter']['gained']],
        'metric_limits':'Recall@k is micro complete-Gold-quote recall, MRR is macro first complete Gold rank. Refusal cases without positive Gold excluded. Scope counts measure retained mentions, not semantic applicability. Automatic judge labels are not human approval.'}
    write(OUT/f'summary{suffix}.json',summary);write(OUT/f'comparison{suffix}.json',comparisons)
    write(OUT/f'scope_structure_audit{suffix}.json',{'scopes':scopes,'structured_rows':structured})
    write(OUT/f'case_023_full_trace{suffix}.json',{name:rows['rag-biz-023'] for name,rows in runs.items()})
    md=['# 逐题真实回答与变化','', 'A=默认关闭，B=本次增强。历史上一版另在 comparison.json 保留。自动标签不能代替对象/条件审核。','']
    for r in comparisons:
        a=answers['A'].get(r['id'],{});b=answers['B'].get(r['id'],{})
        md += [f"## {r['id']} {r['question']}",'',f"集合：{r['cohort']}；标签：{r['review_status']}；A→B：{r['answer_disabled']} → {r['answer_robust_adapter']}",
            f"与上一版比较的证据变化：{r['changes']['previous_adapter']}",'','### A','',r['answer_text_disabled'] or '未完成','','### B','',r['answer_text_robust_adapter'] or '未完成',
            '', '### B引用和对象检查','',str(b.get('judgement',{}).get('source_binding',{})),str(b.get('judgement',{}).get('object_binding',{})),'']
    (OUT/f'answer_review{suffix}.md').write_text('\n'.join(md),'utf-8')
    print({k:v for k,v in summary.items() if k not in ('cohorts','metric_limits')})

if __name__=='__main__':main()
