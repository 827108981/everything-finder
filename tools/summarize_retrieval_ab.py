"""Readable paired comparison, preserving pending and historical cohort boundaries."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path
import statistics

ARMS=('baseline','A','B','A+B')

def main():
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);args=p.parse_args();out=args.out
    input_file=out/'resolved_cases.jsonl' if (out/'resolved_cases.jsonl').exists() else out/'cases.jsonl'
    rows=[json.loads(l) for l in input_file.read_text(encoding='utf-8').splitlines()]
    by={}
    for r in rows:by.setdefault(r['id'],{})[r['arm']]=r
    assert len(rows)==236 and len(by)==59 and all(set(g)==set(ARMS) for g in by.values())
    cohorts={
        '正式48题':[cid for cid,g in by.items() if g['baseline']['cohort']=='formal48'],
        '原18题固定集合（排除007，共17题）':[cid for cid,g in by.items() if g['baseline']['cohort']=='formal48' and int(cid[-3:])<=18],
        '新增32题集合（024待审，共31题正式）':[cid for cid,g in by.items() if g['baseline']['cohort']=='formal48' and int(cid[-3:])>18],
        '回归10题（全部待审，仅诊断）':[cid for cid in by if cid.startswith('rag-acc')],
        '024待审（仅诊断）':['rag-biz-024']}
    lines=['# A、B、A+B配对验证报告','',
        'A为保守的场景页/续页关联补取；B为完整问题＋子问题检索、分诉求相关性评分。每组原有重排候选20条保留，最多增加20条；最终证据上限及条件规则沿用原配置。',
        '', '本报告使用v4.1标注：005已移除非必要安全检查；007排除、024待审，回归10题待审。四组固定同一AI索引和模型，完成236条检索与回答核查。模型判分仅作辅助，不能视为人工准确率。',
        '', '原始v4历史报告与本次标注不同，本文只比较本次四组，不把标注变化计算为算法收益。','']
    all_stats={}
    identity={label:sum(g[left]['context']==g[right]['context'] for g in by.values())
              for label,left,right in (('A_vs_baseline','A','baseline'),('B_vs_baseline','B','baseline'),('AB_vs_B','A+B','B'))}
    lines.extend(['## 生成输入是否真的改变','',
        f"全部59题：A与基线上下文完全相同{identity['A_vs_baseline']}题；B与基线相同{identity['B_vs_baseline']}题；A+B与B相同{identity['AB_vs_B']}题。相同上下文、相同问题的独立生成/判分波动不能归因于检索。",''])
    for name,ids in cohorts.items():
        lines.extend(['## '+name,'','|方案|检查通过|Gold上下文命中|平均证据覆盖|指定反例次数|引用异常|错误|','|---|---:|---:|---:|---:|---:|---:|'])
        for arm in ARMS:
            rs=[by[cid][arm] for cid in ids]
            positive=[r for r in rs if r['expected_behavior']!='refuse' and r['index_coverage']['positive_total']]
            st=[r['stages']['context'] for r in positive]
            # Use the exact same metrics implementation as the app's evaluation.
            from local_full_text_search.rag.evaluation.metrics import aggregate
            m=aggregate(rs)['stages']['context']
            good=sum((r.get('gold_judgement')or{}).get('verdict')=='consistent' for r in rs)
            errs=sum(bool(r['error'] or r['remote_error']) for r in rs)
            citation_issues=sum(bool(r.get('citation_check') and not r['citation_check']['structurally_valid']) for r in rs)
            data={'passed':good,'total':len(rs),'context':m,'errors':errs,'citation_issues':citation_issues}
            all_stats.setdefault(name,{})[arm]=data
            lines.append(f"|{arm}|{good}/{len(rs)}|{m['gold_hit_cases']}/{m['cases_with_gold']}|{m['mean_evidence_recall']:.1%}|{m['hard_negative_candidate_count']}|{citation_issues}|{errs}|")
        lines.append('')
    paired={}
    for arm in ARMS[1:]:
        changes=[]; samectx=0; improved=[];regressed=[];coverage_gain=[];coverage_loss=[]
        for cid,g in sorted(by.items()):
            b,v=g['baseline'],g[arm]
            bs,vs=b['stages']['context'],v['stages']['context']
            bv=(b.get('gold_judgement')or{}).get('verdict');vv=(v.get('gold_judgement')or{}).get('verdict')
            identical=b['context']==v['context']
            if identical and bv!=vv:samectx+=1
            if b['cohort']=='formal48':
                if bv!='consistent' and vv=='consistent':improved.append(cid)
                if bv=='consistent' and vv!='consistent':regressed.append(cid)
                if (vs['gold_evidence_recall'] or 0)>(bs['gold_evidence_recall'] or 0):coverage_gain.append(cid)
                if (vs['gold_evidence_recall'] or 0)<(bs['gold_evidence_recall'] or 0):coverage_loss.append(cid)
            if bv!=vv or bs['gold_evidence_recall']!=vs['gold_evidence_recall'] or bs['hard_negative_found']!=vs['hard_negative_found']:
                changes.append({'id':cid,'cohort':b['cohort'],'same_context':identical,'before':bv,'after':vv,
                                'before_coverage':bs['gold_evidence_recall'],'after_coverage':vs['gold_evidence_recall']})
        paired[arm]={'formal_pass_gains':improved,'formal_pass_losses':regressed,'formal_evidence_gains':coverage_gain,'formal_evidence_losses':coverage_loss,
                     'same_context_verdict_changes_all59':samectx,'changes':changes}
        lines.extend(['## '+arm+'相对基线逐题变化','',
            '正式题检查转为通过：'+('、'.join(improved) or '无')+'。',
            '正式题由通过转为未通过：'+('、'.join(regressed) or '无')+'。',
            '正式题证据覆盖增加：'+('、'.join(coverage_gain) or '无')+'。',
            '正式题证据覆盖减少：'+('、'.join(coverage_loss) or '无')+'。',
            f'全部59题中，上下文完全相同但模型判分发生变化：{samectx}题；这些变化不作为检索效果的证据。','',
            '|题号|集合|检查结果变化|证据覆盖变化|上下文相同|','|---|---|---|---|---|'])
        for c in changes:
            bc='—' if c['before_coverage'] is None else f"{c['before_coverage']:.0%}"
            ac='—' if c['after_coverage'] is None else f"{c['after_coverage']:.0%}"
            lines.append(f"|{c['id']}|{c['cohort']}|{c['before']} → {c['after']}|{bc} → {ac}|{'是' if c['same_context'] else '否'}|")
        lines.append('')
    lines.extend(['## 019、050的证据去向',''])
    if (out/'050_alternative_source_audit.json').exists():
        alt=json.loads((out/'050_alternative_source_audit.json').read_text(encoding='utf-8'))
        lines.extend(['**050存在固定Gold指标漏记的替代来源。** B和A+B取回《高速免疫场景化培训教材-光学V1.1.pptx》第15张，基线和A没有取回。原始PPT、其索引正文和原Gold PDF第17页均能找到同一段四类原因引文；本次逐项核验均通过。',
            '因此不能仅凭原Gold页未命中，判定B没有补回支持答案的资料。这个局部收益单列为证据核验结果；正式Gold与计分不变，候选等价证据的补标仍需业务审核。',
            '替代PPT地址：'+alt['original_paths']['ppt']+'。详见050_alternative_source_audit.json。',''])
    target={'rag-biz-019':'47a3663acd65d8fc404bea85455f55b7a4bc1e0b30454dc97e777de9e6350d2e',
            'rag-biz-050':'e9b8c2a0c6ad65ebb77179fa8fb7101173b58dec5e274ceaadfe737258a8d2cf'}
    targets={}
    for cid,chid in target.items():
        lines.extend(['### '+cid,'','|方案|重排位置|证据去向|回答检查|','|---|---:|---|---|'])
        targets[cid]={}
        for arm,g in by[cid].items():
            rank=next((i+1 for i,r in enumerate(g['candidate_stages'].get('rerank',[])) if r['chunk_id']==chid),None)
            e=next((r for r in g['candidate_stages'].get('evidence',[]) if r['chunk_id']==chid),{})
            state='进入上下文' if any(r['chunk_id']==chid for r in g['candidate_stages'].get('context',[])) else e.get('reason','未进入重排')
            verdict=(g.get('gold_judgement')or{}).get('verdict')
            targets[cid][arm]={'rerank_rank':rank,'state':state,'verdict':verdict,'routes':g.get('subquery_routes')}
            lines.append(f'|{arm}|{rank or "—"}|{state}|{verdict}|')
        lines.append('')
        if cid=='rag-biz-050':
            for route in by[cid]['B'].get('subquery_routes',[]):
                hit=next((x for x in route['candidates'] if x['chunk_id']==chid),None)
                if hit:
                    lines.append(f"B子查询“{route['query']}”：原Gold页关键词排名{hit['lexical_rank']}，向量排名{hit['vector_rank']}，融合排名{hit['fused_rank']}；子查询前10候选限制仍会排除它。以上不否定B另行召回了经原文核验的PPT替代来源。")
            lines.append('')
    plans=json.loads((out/'query_plans.json').read_text(encoding='utf-8'))
    lines.extend(['## 边界与耗时','',
        f"59题中{sum(bool(p['queries']) for p in plans.values())}题增加了子查询；{sum('error' in p for p in plans.values())}题规划报错并回退原问。",
        '为保证配对输入完全相同，本次实验在四组之间复用同一查询的确定性本地检索，以及相同问题/正文对的Cross-Encoder原始分数；模型回答与模型判分分别重新请求。实验不复用历史答案。',
        '因此各组运行墙钟耗时受缓存命中与执行顺序影响，不能据此比较线上延迟；inference_cost.json记录实际缓存未命中的推理开销，query_plans.json记录查询规划耗时。尚未完成独立冷缓存的端到端延迟验收。',
        'A当前只覆盖能保守识别的短故障场景页＋原因/步骤续页，不代表已支持所有跨页表格和章节结构；没有直接检索分数的关联候选仍受现有证据门槛约束。',
        'B保留原问，但子问题自然语言条件保持主要依赖模型；已做明确型号保留校验，其他条件仍需检查原始query_plans.json。单次结果不能证明查询规划与生成完全稳定。',
        '没有新增独立盲测集合，也没有在本次同时测试独立扩大候选预算的对照，因此不能将全部变化归因于某一个内部步骤，更不能宣称对其他业务零影响。',
        '生产配置、业务代码、索引与Gold状态未改动。试跑/中断记录保留，仅本目录完整236条进入汇总。',''])
    if (out/'retry_summary.json').exists():
        retries=json.loads((out/'retry_summary.json').read_text(encoding='utf-8'))
        lines.extend([f"初轮执行异常{retries['initial_errors']}条；只对异常统一重试一次，最终仍有{retries['remaining_errors']}条执行异常。裁判失败时保留原始生成答案，仅重试判分。初始cases.jsonl、重试execution_retries.jsonl、最终resolved_cases.jsonl分别保存。",''])
    result={'cohorts':all_stats,'paired':paired,'targets':targets,'context_identity':identity}
    (out/'paired_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    raw=(out/'report.html').read_text(encoding='utf-8')
    # Place readable outcome tables ahead of the detailed JSON appendix.
    try:
        import markdown
        introduction=markdown.markdown('\n'.join(lines),extensions=['tables'])
    except ImportError:
        introduction='<pre>'+html.escape('\n'.join(lines))+'</pre>'
    (out/'report.html').write_text(raw.replace('<h1>A、B、A+B验证报告</h1>','<h1>A、B、A+B验证报告</h1>'+introduction),encoding='utf-8')
    print(json.dumps({'cohorts':all_stats,'paired':{k:{kk:vv for kk,vv in v.items() if kk!='changes'} for k,v in paired.items()}},ensure_ascii=False,indent=2))

if __name__=='__main__':main()

