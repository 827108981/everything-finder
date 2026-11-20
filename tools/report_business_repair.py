"""Source-quote metrics and paired-answer reports; excludes retired cases by manifest."""
import json, argparse, sqlite3
from collections import Counter
from pathlib import Path
from hashlib import sha256
from tools.validate_business_repair import native
from tools.rag_eval_common import lines,write,sha
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

OUT=native.OUT
FOCUS=['rag-biz-019','rag-biz-050','rag-row-holdout2-001','rag-row-holdout-001','evidence-flow-independent-07','rag-biz-020','rag-biz-021','rag-biz-023']

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='on_all');ap.add_argument('--answers',default='answers_AB');ap.add_argument('--output',default='report');args=ap.parse_args()
    cases={c.case_id:c for p in native.FILES for c in load_gold_dataset(p,reviewed_only=False).cases}
    cap={a:{r['id']:r for r in lines(OUT/((args.candidate if a=='on' else 'off_all')+'.jsonl'))} for a in ('off','on')}
    assert len(cases)==91 and all(set(rows)==set(cases) for rows in cap.values())
    answers=lines(OUT/(args.answers+'.jsonl')) if (OUT/(args.answers+'.jsonl')).exists() else []
    if args.answers!='answers_AB': answers=[r for r in lines(OUT/'answers_AB.jsonl') if r['arm']=='A']+answers
    by={(r['arm'],r['id']):r for r in answers}
    summary={'total':91,'main_total':49,'retired_ids':['rag-biz-040'],'bundle':'g000033-71ec3c52f3','cohorts':{},'gains':[],'losses':[],'unchanged_context_ids':[],'inactive_equivalence':{},'answer_gains':[],'answer_losses':[],'uncertain_changes':[],'retrieval_gain_answer_not_improved':[],'same_context_answer_variation':[]}
    groups={'main_reviewed':[i for i,c in cases.items() if i.startswith('rag-biz') and c.review_status=='reviewed'],
        'main_diagnostic':[i for i,c in cases.items() if i.startswith('rag-biz') and c.review_status!='reviewed'],
        'regression_reviewed':[i for i,c in cases.items() if not i.startswith('rag-biz') and c.review_status=='reviewed'],
        'regression_diagnostic':[i for i,c in cases.items() if not i.startswith('rag-biz') and c.review_status!='reviewed']}
    for group,ids in groups.items():
        summary['cohorts'][group]={'count':len(ids),'arms':{}}
        for arm,rows in cap.items():
            stats={}
            for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context'):
                scores=[rows[i]['scores'][stage] for i in ids];scores=[s for s in scores if s['gold_evidence_total_declared']]
                metrics={'eligible_cases':len(scores)}
                for k in (1,3,5): metrics['recall@'+str(k)]=sum(sum(rank<=k for rank in s['gold_completion_ranks'].values())/s['gold_evidence_total_declared'] for s in scores)/len(scores) if scores else None
                metrics['MRR']=sum(s['reciprocal_rank'] or 0 for s in scores)/len(scores) if scores else None
                metrics['all_gold_retained']=sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in scores)
                metrics['mean_evidence_recall']=sum(s['gold_evidence_recall'] or 0 for s in scores)/len(scores) if scores else None
                stats[stage]=metrics
            key='A' if arm=='off' else 'B'
            scope_spans=[x for i in ids for e in rows[i]['entries'] for x in e.get('query_plan_diagnostics',{}).get('scope_observations',[])]
            structured=[e for i in ids for e in rows[i]['entries'] if e.get('source_spans') or e.get('query_plan_diagnostics',{}).get('matched_list_fields')]
            judgments=[by.get((key,i),{}).get('judgement',{}) for i in ids]
            summary['cohorts'][group]['arms'][arm]={'stages':stats,'answers':dict(Counter(j.get('verdict','not_run') for j in judgments)),
                'structured_context_entries':len(structured),'scope_spans_in_context':len(scope_spans),
                'scope_retention_interpretation':'Source-local text presence only; adapter-off lacks these annotations, not zero applicability. Semantic scope assessed independently by judge.',
                'object_binding':dict(Counter(j.get('object_binding',{}).get('status','not_run') for j in judgments)),
                'source_binding':dict(Counter(j.get('source_binding',{}).get('status','not_run') for j in judgments)),
                'core_checks':dict(Counter(r['status'] for j in judgments for r in j.get('rubrics',[]) if r.get('kind')=='required_facts' and r.get('role')=='core')),
                'forbidden_claims':dict(Counter(r['status'] for j in judgments for r in j.get('rubrics',[]) if r.get('kind')=='forbidden_claims')),
                'behavior':dict(Counter(j.get('behavior','not_run') for j in judgments))}
    inactive=[];fail=[]
    for cid,c in cases.items():
        a,b=cap['off'][cid],cap['on'][cid]
        da=a['scores']['context']['gold_evidence_found'];db=b['scores']['context']['gold_evidence_found']
        if db>da:summary['gains'].append(cid)
        if db<da:summary['losses'].append(cid)
        if a['context']==b['context']:summary['unchanged_context_ids'].append(cid)
        active=next((e['query_plan']['adapter_active'] for e in b['events'] if 'query_plan' in e),False)
        if not active:
            inactive.append(cid)
            for stage in a['candidate_stages']:
                sig=lambda row:[(x['chunk_id'],x.get('lexical_score'),x.get('vector_score'),x.get('fused_score'),x.get('rerank_score'),x.get('selected')) for x in row['candidate_stages'].get(stage,[])]
                if sig(a)!=sig(b):fail.append({'id':cid,'stage':stage})
            if a['context']!=b['context']:fail.append({'id':cid,'stage':'context'})
        av=by.get(('A',cid),{}).get('judgement',{}).get('verdict');bv=by.get(('B',cid),{}).get('judgement',{}).get('verdict')
        if av and bv:
            if av!=bv and not active and a['context']==b['context']:
                summary['same_context_answer_variation'].append({'id':cid,'off':av,'on':bv,'reason':'Inactive adapter, same context/plan; cannot attribute to algorithm change'})
            if av!='consistent' and bv=='consistent':summary['answer_gains'].append(cid)
            elif av=='consistent' and bv!='consistent':summary['answer_losses'].append(cid)
            elif av!=bv:summary['uncertain_changes'].append({'id':cid,'off':av,'on':bv})
            if db>da and bv!='consistent':summary['retrieval_gain_answer_not_improved'].append(cid)
    summary['inactive_equivalence']={'count':len(inactive),'ids':inactive,'failures':fail}
    summary['answer_count']=len(answers);summary['answer_errors']=[{'id':r['id'],'arm':r['arm'],'error':r['error']} for r in answers if r.get('error')]
    write(OUT/(args.output+'_metrics.json'),summary)
    text=['# 本轮验证报告\n','同一新版91题开关配对；主集49、回归42；退休040不参与任何本轮分母。旧版本总分不可直接比较。\n',
        '## 检索与Context\n',f"完整Gold证据增加：{', '.join(summary['gains']) or '无'}。减少：{', '.join(summary['losses']) or '无'}。\n",
        f"未激活Adapter的{len(inactive)}题，逐阶段候选、分数及Context不变检查：{len(fail)}项差异。\n",'|集合|题数|关闭答案分布|开启答案分布|\n|---|---:|---|---|\n']
    for group,data in summary['cohorts'].items():text.append(f"|{group}|{data['count']}|{data['arms']['off']['answers']}|{data['arms']['on']['answers']}|\n")
    text.extend(['\n## 全部题目的前后指标\n','逐阶段Recall@1/3/5、MRR、完整证据保留率见 metrics.json。这里Recall的分母是该题声明的正向证据数，不是仅命中任意一条；拒答等无正向证据题不纳入检索Recall分母。\n',f"答案改善：{summary['answer_gains']}\n",f"答案退化：{summary['answer_losses']}\n",f"其他变化待审：{summary['uncertain_changes']}\n",f"证据增加但答案尚未通过：{summary['retrieval_gain_answer_not_improved']}\n",'\n## 重点题实际答案与轨迹\n'])
    focus=[]
    for cid in FOCUS:
        c=cases[cid];item={'id':cid,'question':c.question,'gold_checks':list(c.answer_checks),'arms':{}}
        text.extend(['### '+cid+'\n',c.question+'\n'])
        for arm,key in [('off','A'),('on','B')]:
            capture=cap[arm][cid];answer=by.get((key,cid),{})
            item['arms'][arm]={'capture':capture,'answer':answer}
            text.extend([f"**{arm}**：完整Gold引文 {capture['scores']['context']['gold_evidence_found']}/{capture['scores']['context']['gold_evidence_total_declared']}；判分 {answer.get('judgement',{}).get('verdict','未完成')}\n",answer.get('answer',{}).get('text','未完成')+'\n'])
        focus.append(item)
    write(OUT/(args.output+'_focus_full_traces.json'),focus)
    (OUT/(args.output+'.md')).write_text('\n'.join(text),encoding='utf8')
    from local_full_text_search.rag.evaluation.reporting.comparison import write_comparison_report
    # Resolve Gold basenames against actual published source metadata, not guessed roots.
    metadata = Path('E:/everythingfinder-db/rag') / ('rag.' + summary['bundle']) / 'metadata.db'
    with sqlite3.connect(metadata.resolve().as_uri() + '?mode=ro', uri=True) as db:
        source_files = {json.loads(r[0])['file_path'] for r in db.execute('SELECT source_json FROM rag_chunks')}
    raw_cases = [json.loads(line) for path in native.FILES for line in path.read_text(encoding='utf8').splitlines() if line.strip()]
    write_comparison_report(OUT, raw_cases, cap, by, summary, source_files)
    print(json.dumps({k:summary[k] for k in ('gains','losses','answer_count','answer_gains','answer_losses','answer_errors')},ensure_ascii=False))

if __name__=='__main__':main()
