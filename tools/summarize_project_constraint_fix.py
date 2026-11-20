"""Audit paired retrieval and actual answers without modifying Gold."""
from __future__ import annotations
import collections
import hashlib
import html
import json
from pathlib import Path

OUT=Path('reports/project_constraint_fix_20261102')


def main():
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    pairs=collections.defaultdict(dict)
    for r in rows:
        pairs[(r['cohort'],r['id'])][r['arm']]=r
    assert len(pairs)==60 and all(set(p)=={'before','after'} for p in pairs.values())
    assert not any('error_type'in r for p in pairs.values() for r in p.values())
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for ds in protocol['datasets'].values():
        assert hashlib.sha256(Path(ds['path']).read_bytes()).hexdigest()==ds['sha256']
    for (_,cid),p in pairs.items():
        for stage in ('lexical','vector','fusion','deduplicated','rerank'):
            assert p['before']['candidate_stages'].get(stage)==p['after']['candidate_stages'].get(stage),(cid,stage)
    def score(r):
        return r['stages']['context']
    summary={}
    for group in ('development','original18','added32','holdout_pending'):
        selected=[p for (cohort,cid),p in pairs.items() if
            (cohort==group if group in ('development','holdout_pending') else
             cohort=='development' and ((int(cid[-3:])<=18)==(group=='original18')))]
        summary[group]={}
        for arm in ('before','after'):
            positive=[p[arm] for p in selected if p[arm]['expected_behavior']!='refuse']
            groups=collections.defaultdict(list)
            for r in positive:
                groups[r['group_id']].append(score(r)['gold_evidence_recall'])
            judged=[p[arm] for p in selected if p[arm].get('gold_judgement')]
            core=[c for r in judged for c in r['gold_judgement'].get('rubrics',[]) if c['role']=='core']
            negatives=[c for r in judged for c in r['gold_judgement'].get('rubrics',[]) if c['kind']=='forbidden_claims']
            summary[group][arm]={'positive_n':len(positive),
                'hit':sum(score(r)['gold_hit'] for r in positive),
                'coverage':sum(score(r)['gold_evidence_recall'] for r in positive)/len(positive),
                'group_coverage':sum(sum(v)/len(v) for v in groups.values())/len(groups),
                'negatives':sum(score(r)['hard_negative_found'] for r in positive),
                'judge_verdicts':dict(collections.Counter(r['gold_judgement']['verdict'] for r in judged)),
                'behavior_judge':dict(collections.Counter(r['gold_judgement'].get('behavior') for r in judged)),
                'core_checks':dict(collections.Counter(c['status'] for c in core)),
                'forbidden_checks':dict(collections.Counter(c['status'] for c in negatives))}
    changes=[]
    for (cohort,cid),p in sorted(pairs.items()):
        a,b=p['before'],p['after']
        ca,cb=score(a),score(b)
        answer_a=a.get('answer') or {}
        answer_b=b.get('answer') or {}
        ja,jb=(r.get('gold_judgement') or {} for r in (a,b))
        ar={r['id']:r for r in ja.get('rubrics',[])}
        br={r['id']:r for r in jb.get('rubrics',[])}
        rubric_changes=[{'id':key,'role':ar[key]['role'],'text':ar[key]['gold_text'],
            'before':ar[key]['status'],'after':br[key]['status'],
            'before_reason':ar[key]['reason'],'after_reason':br[key]['reason']}
            for key in ar.keys()&br.keys() if ar[key]['status']!=br[key]['status']]
        changes.append({'id':cid,'cohort':cohort,'question':a['question'],
            'expected_behavior':a['expected_behavior'],
            'before_coverage':ca['gold_evidence_recall'],'after_coverage':cb['gold_evidence_recall'],
            'new_negatives':sorted(set(cb['hard_negative_ids'])-set(ca['hard_negative_ids'])),
            'before_verdict':ja.get('verdict'),'after_verdict':jb.get('verdict'),
            'answer_changed':answer_a.get('text')!=answer_b.get('text'),
            'rubric_changes':rubric_changes})
    result={'summary':summary,'checks':{'paired_questions':60,'identical_upstream_and_rerank':True,
        'gold_hashes_unchanged':True,'latest_errors':0,'raw_rows':len(rows),
        'initial_harness_error_rows':sum('error_type' in r for r in rows)},'changes':changes}
    variation_path=OUT/'variation_038.json'
    if variation_path.exists():
        variation=json.loads(variation_path.read_text(encoding='utf-8'))
        result['variation_038']={'same_context':variation['same_context'],
            'same_first_evidence':variation['same_first_evidence'],
            'original':[(r['arm'],r['judgement']['verdict']) for r in variation['original']],
            'repeats':[(r['arm'],r['repeat'],r['judgement']['verdict']) for r in variation['repeats']],
            'note':'Retain original main comparison; repeats do not replace any result.'}
    (OUT/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=html.escape
    lines=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>泛称项目条件误伤修复验证</title>',
        '<style>body{font:16px system-ui;margin:30px;max-width:1500px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}pre{white-space:pre-wrap}summary{cursor:pointer;padding:10px}.note{padding:15px;background:#fff4db}</style>',
        '<h1>当前应用配置：泛称“项目”条件误伤修复</h1>',
        '<p class="note">50题真实问答前后对照，10题pending仅检索诊断。使用当前混合评分及名次保护。完全相同的请求复用结果以控制随机差异。模型裁判仅辅助审核，不能替代人工确认。</p>',
        '<table><tr><th>集合</th><th>修改前命中/覆盖</th><th>修改后命中/覆盖</th><th>前/后指定干扰次数</th></tr>']
    for group,v in summary.items():
        a,b=v['before'],v['after']
        lines.append(f'<tr><td>{esc(group)}</td><td>{a["hit"]}/{a["positive_n"]}，{a["coverage"]:.1%}</td><td>{b["hit"]}/{b["positive_n"]}，{b["coverage"]:.1%}</td><td>{a["negatives"]}/{b["negatives"]}</td></tr>')
    a,b=summary['development']['before'],summary['development']['after']
    lines.append('</table><p>主轮模型裁判consistent：'+str(a['judge_verdicts'].get('consistent',0))+'/50 → '
        +str(b['judge_verdicts'].get('consistent',0))+'/50。模型裁判仅辅助审核。</p>')
    if variation_path.exists():
        lines.append('<p class="note">完整回答尚未全部通过：021仍未请求补充条件，027仍遗漏预处理要求。038主轮出现新增遗漏，固定追加2次/版本后，修改前后均有遗漏；直接首条证据相同，背景不同，不能排除输入变化的影响。'
                     '<a href="variation_038.json">查看038全部复查结果</a>。原主轮结果保持原样。</p>')
    for c in changes:
        p=pairs[(c['cohort'],c['id'])]
        lines.append('<details><summary>'+esc(f'{c["id"]} {c["question"]} | 覆盖 {c["before_coverage"]}→{c["after_coverage"]} | 裁判 {c["before_verdict"]}→{c["after_verdict"]}')+'</summary>')
        lines.append('<pre>'+esc(json.dumps(c,ensure_ascii=False,indent=2))+'</pre><table><tr><th>修改前</th><th>修改后</th></tr><tr>')
        for arm in ('before','after'):
            r=p[arm]
            lines.append('<td><pre>'+esc((r.get('answer') or {}).get('text','未生成答案'))+'</pre><details><summary>裁判检查项</summary><pre>'+esc(json.dumps(r.get('gold_judgement'),ensure_ascii=False,indent=2))+'</pre></details>')
            for x in r['candidate_stages'].get('context',[]):
                lines.append('<details><summary>'+esc(x['filename']+' '+x['location'])+'</summary><pre>'+esc(x.get('included_text') or x['text'])+'</pre></details>')
            lines.append('</td>')
        lines.append('</tr></table></details>')
    lines.append('</html>')
    (OUT/'report.html').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
