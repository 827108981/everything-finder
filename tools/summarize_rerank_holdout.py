"""Summarize the frozen experiment without selecting new parameters."""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();out=args.run
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    completion=json.loads((out/'completion.json').read_text(encoding='utf-8'))
    assert completion['n']==120 and completion['status']=='completed'
    rows=[json.loads(l) for l in (out/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==120
    paired=defaultdict(dict)
    for r in rows:paired[(r['cohort'],r['id'])][r['arm']]=r
    assert len(paired)==60 and all(set(v)=={'current','pure'} for v in paired.values())
    upstream=['lexical','vector','fusion','deduplicated']
    for v in paired.values():
        a,b=v['current'],v['pure']
        for s in upstream:assert a['candidate_stages'].get(s,[])==b['candidate_stages'].get(s,[]),(a['id'],s)
        # Sorting/combined scores may differ, but every candidate's raw model score and judgement must be identical.
        a_scores={c['chunk_id']:c['rerank_score'] for c in a['candidate_stages'].get('rerank',[])}
        b_scores={c['chunk_id']:c['rerank_score'] for c in b['candidate_stages'].get('rerank',[])}
        assert a_scores==b_scores
        a_dec={c['chunk_id']:c.get('constraint_decision') for c in a['candidate_stages'].get('constraints',[])}
        b_dec={c['chunk_id']:c.get('constraint_decision') for c in b['candidate_stages'].get('constraints',[])}
        assert a_dec==b_dec
    for d in protocol['datasets'].values():assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    def score(row,key):
        return row['stages']['context'] if key=='context' else row['rerank_topk'][key]
    def subset(cohort,arm):
        return [r for r in rows if r['arm']==arm and r['expected_behavior']!='refuse'
                and (r['cohort']==cohort if cohort in ['development','holdout_pending'] else
                    r['cohort']=='development' and ((int(r['id'][-3:])<=18)==(cohort=='original18')))]
    summary={'protocol':protocol,'checks':{'pairs':60,'matched_upstream_candidates':True,'matched_raw_scores':True,
                'matched_constraint_judgements':True,'gold_files_unchanged':True},'cohorts':{},'changes':{}}
    for cohort in ['development','original18','added32','holdout_pending']:
        summary['cohorts'][cohort]={}
        for arm in ['current','pure']:
            selected=subset(cohort,arm);values={}
            for key in ['1','3','5','8','20','context']:
                scores=[score(r,key) for r in selected]
                groups=defaultdict(list)
                for r in selected:groups[r['group_id']].append(score(r,key)['gold_evidence_recall'])
                values[key]={'n':len(scores),'hit':sum(s['gold_hit'] for s in scores),
                    'coverage':sum(s['gold_evidence_recall'] for s in scores)/len(scores),
                    'mrr':sum(s['reciprocal_rank'] for s in scores)/len(scores),
                    'evidence_found':sum(s['gold_evidence_found'] for s in scores),
                    'evidence_total':sum(s['gold_evidence_total_declared'] for s in scores),
                    'negative_occurrences':sum(s['hard_negative_found'] for s in scores),
                    'group_coverage':sum(sum(v)/len(v) for v in groups.values())/len(groups)}
            summary['cohorts'][cohort][arm]=values
    for cohort in ['development','holdout_pending']:
        for key in ['8','context']:
            changes=[]
            for (group,cid),v in paired.items():
                if group!=cohort:continue
                a,b=score(v['current'],key),score(v['pure'],key)
                changes.append({'id':cid,'question':v['current']['question'],'current':a,'pure':b,
                                'coverage_delta':None if a['gold_evidence_recall'] is None else b['gold_evidence_recall']-a['gold_evidence_recall'],
                                'new_negative_ids':sorted(set(b['hard_negative_ids'])-set(a['hard_negative_ids'])),
                                'removed_negative_ids':sorted(set(a['hard_negative_ids'])-set(b['hard_negative_ids']))})
            summary['changes'][cohort+'_'+key]=changes
    dump=lambda p,v:p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump(out/'comparison.json',summary)
    labels={'development':'开发集50题（46题有正向证据）','original18':'开发集原18题（16题有正向证据）',
            'added32':'开发集新增32题（30题有正向证据）','holdout_pending':'验收候选10题（pending，仅预验收）'}
    lines=['# 只改重排：开发集与新主题预验收对照','','本实验仅比较当前混合重排与纯模型重排。条件检查、同文件限额3、最终8条及上下文12000字均固定。判定方案和参数在查看验收题运行结果之前冻结，未根据新题调参。','',
        '**资格边界：10道验收候选仍为pending。此处为明确标记的诊断预验收，未进入正式计分，也没有修改审核状态或批量评测加载门禁。**','',
        '## 重排前8条（主指标）','','| 集合 | 当前命中 | 纯模型命中 | 当前平均证据覆盖 | 纯模型平均证据覆盖 | 当前MRR@8 | 纯模型MRR@8 | 当前/纯模型指定反例次数 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort in labels:
        a=summary['cohorts'][cohort]['current']['8'];b=summary['cohorts'][cohort]['pure']['8']
        lines.append(f"| {labels[cohort]} | {a['hit']}/{a['n']} | {b['hit']}/{b['n']} | {a['coverage']:.1%} | {b['coverage']:.1%} | {a['mrr']:.4f} | {b['mrr']:.4f} | {a['negative_occurrences']}/{b['negative_occurrences']} |")
    lines+=['','平均证据覆盖按题平均，包含未进入候选的Gold；MRR@8按前8条中首条完整正确证据的倒数排名平均。所有评分都需要完整引文覆盖，不把部分引文计为完整命中。','',
        '## 最终上下文（后续策略固定的次指标）','','| 集合 | 当前命中 | 纯模型命中 | 当前平均证据覆盖 | 纯模型平均证据覆盖 | 当前/纯模型指定反例次数 |','|---|---:|---:|---:|---:|---:|---:|']
    for cohort in labels:
        a=summary['cohorts'][cohort]['current']['context'];b=summary['cohorts'][cohort]['pure']['context']
        lines.append(f"| {labels[cohort]} | {a['hit']}/{a['n']} | {b['hit']}/{b['n']} | {a['coverage']:.1%} | {b['coverage']:.1%} | {a['negative_occurrences']}/{b['negative_occurrences']} |")
    for cohort in ['development','holdout_pending']:
        lines+=['',f'## {labels[cohort]}：逐题变化','', '| 题号 | 当前/纯模型重排首个Gold排名 | 当前/纯模型前8覆盖 | 当前/纯模型最终覆盖 | 新增前8反例 | 新增最终反例 |','|---|---:|---:|---:|---|---|']
        for (group,cid),v in paired.items():
            if group!=cohort:continue
            c,p=v['current'],v['pure'];a,b=score(c,'8'),score(p,'8');x,y=score(c,'context'),score(p,'context')
            fmt=lambda z:'—' if z is None else f'{z:.1%}'
            ranks=[v[k]['rerank_topk']['20']['first_gold_rank'] for k in ['current','pure']]
            lines.append(f"| {cid} | {ranks[0] or '—'}/{ranks[1] or '—'} | {fmt(a['gold_evidence_recall'])}/{fmt(b['gold_evidence_recall'])} | {fmt(x['gold_evidence_recall'])}/{fmt(y['gold_evidence_recall'])} | {','.join(sorted(set(b['hard_negative_ids'])-set(a['hard_negative_ids']))) or '—'} | {','.join(sorted(set(y['hard_negative_ids'])-set(x['hard_negative_ids']))) or '—'} |")
    lines+=['','## 冻结与校验','',f"- AI包：`{protocol['bundle_id']}`；模型与设置见同目录protocol.json。",f"- 参数冻结时间：`{protocol['frozen_at']}`；脚本SHA256：`{protocol['script_sha256']}`。",'- 所有60对问题的重排前4阶段候选和正文一致、原模型分一致、条件判定一致。唯一实验变量是混合评分和名次保护。',
        '- 开发集复用此前核实的模型原始分；新题实际首次推理200对，另一个排序组复用相同原始分；不以缓存运行时间宣称性能改善。',
        '- 验收题未改写为reviewed，未调用正式批量运行器将pending计入正式成绩；结果只在本实验诊断文件中存储。',
        '- 120条流程均完成，API调用0次；未生成答案，不能宣称答案准确率、拒答质量或完整上线效果已被验证。',
        '- 原50题已参与调优，开发集表现不代表独立效果；新10题虽未参与本轮规则选择，题量小、全为answer且集中于医检培训资料，不能作为普适保证。',
        '- 新增指定反例意味着需看正文适用性，不直接等于错误回答；未标注候选仍是未判定。',
        '- Gold及审核文件保持原始哈希；同一业务问题按group_id平均的覆盖率见comparison.json。',
        '', '## 原始产物','',f'- [冻结协议]({(out/"protocol.json").resolve().as_posix()})',f'- [逐题原始结果]({(out/"cases.jsonl").resolve().as_posix()})',f'- [比较统计]({(out/"comparison.json").resolve().as_posix()})']
    report=Path('docs/rag_rerank_dev_holdout_validation_20261101.md');report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Small standalone HTML to make the per-case comparison easy to inspect.
    h=['<!doctype html><meta charset="utf-8"><title>重排开发集与预验收对照</title><style>body{font:16px system-ui;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}details{margin:15px 0}pre{white-space:pre-wrap}</style>',
       '<h1>仅重排A/B：开发集与pending预验收</h1><p>当前：0.65/0.05/0.30，名次保护3。纯模型：1/0/0，关闭名次保护。其余策略固定。验收候选未审核，非正式计分；未生成答案。</p>']
    for (cohort,cid),v in paired.items():
        h.append('<details><summary>'+html.escape(cid+' '+v['current']['question'])+'</summary><p>'+html.escape(labels[cohort])+'</p><table><tr><th>方式</th><th>前8命中/覆盖</th><th>最终命中/覆盖</th><th>前8资料（名次、来源、正文）</th></tr>')
        for arm in ['current','pure']:
            r=v[arm];a,b=score(r,'8'),score(r,'context')
            sources=''.join('<details><summary>'+html.escape(str(i)+' '+x['filename']+' '+str(x.get('location','')))+'</summary><pre>'+html.escape(x.get('text',''))+'</pre></details>' for i,x in enumerate(r['candidate_stages'].get('rerank',[])[:8],1))
            h.append('<tr><td>'+arm+'</td><td>'+html.escape(str((a['gold_hit'],a['gold_evidence_recall'])))+'</td><td>'+html.escape(str((b['gold_hit'],b['gold_evidence_recall'])))+'</td><td>'+sources+'</td></tr>')
        h.append('</table></details>')
    (out/'comparison.html').write_text('\n'.join(h),encoding='utf-8')
    print(json.dumps({k:{a:v['8'] for a,v in d.items()} for k,d in summary['cohorts'].items()},ensure_ascii=False))


if __name__=='__main__':main()
