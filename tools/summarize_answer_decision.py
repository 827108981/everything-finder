import html
import json
import math
from collections import Counter
from statistics import median
from tools.validate_answer_decision import OUT,read,lines,dump

def main():
    rs=lines(OUT/'results.jsonl');inputs=read(OUT/'inputs.json');table=[]
    for p in inputs:
        row={'id':p['id'],'variant':p['variant'],'role':p['role'],'arms':{}}
        for arm in ['baseline','candidate']:
            rr=[r for r in rs if r['id']==p['id'] and r['variant']==p['variant'] and r['arm']==arm]
            row['arms'][arm]={'completed':len(rr),'consistent':sum((r.get('gold_judgement')or{}).get('verdict')=='consistent' for r in rr),
                'errors':sum(bool(r['error']) for r in rr),'verdicts':dict(Counter((r.get('gold_judgement')or{}).get('verdict','execution_error') for r in rr)),
                'behaviors':dict(Counter((r.get('output')or{}).get('behavior','unavailable') for r in rr))}
        table.append(row)
    timing={}
    for arm in ['baseline','candidate']:
        rr=[r for r in rs if r['arm']==arm];tt=sorted(r['generation_ms'] for r in rr if r.get('generation_ms') is not None)
        successful_recoveries=sum((r.get('output')or{}).get('format_recoveries',0) for r in rr)
        exhausted=sum(bool(r['error'] and r['error']['type'] in {'ValueError','TypeError','JSONDecodeError'} and not r.get('output')) for r in rr) if arm=='candidate' else 0
        timing[arm]={'completed_generation':len(tt),'generation_median_ms':median(tt) if tt else None,'generation_p95_ms':tt[max(0,math.ceil(.95*len(tt))-1)] if tt else None,
            'format_recoveries_successful_output':successful_recoveries,'format_recoveries_exhausted':exhausted,
            'minimum_generation_requests':len(rr)+successful_recoveries+exhausted,'errors':sum(bool(r['error']) for r in rr),
            'note':'Latency only for completed generation outputs; includes format recovery time, excludes judge. Failed generation latency excluded; not an unbiased latency superiority claim. Request lower bound excludes SDK transport retries.'}
    failed_targets=[r for r in table if r['role']=='target' and (r['arms']['candidate']['completed']!=3 or r['arms']['candidate']['consistent']!=3)]
    control_alerts=[r for r in table if r['role']=='control' and (r['arms']['candidate']['errors'] or r['arms']['candidate']['consistent']<r['arms']['baseline']['consistent'] or r['arms']['candidate']['behaviors'].get('clarify',0))]
    summary={'rows':len(rs),'expected':60,'cases':table,'timing':timing,'automatic_target_gate_passed':len(rs)==60 and not failed_targets,
        'failed_targets':[r['id']+':'+r['variant'] for r in failed_targets],'control_alerts':[r['id'] for r in control_alerts],
        'note':'Automatic gate only; verify question/answer semantics and extra claims before adoption. Failed first gate stops full-set expansion.'}
    dump(OUT/'summary.json',summary)
    md=['# 单次请求回答决策隔离实验','',f'已完成 {len(rs)}/60 组，10种固定上下文，基线/候选各3次。资料逐字相同，当前模型与生产请求预算1000不变；015另含既有重排压力上下文。',
        '候选在单次请求中返回诉求、证据状态及文本，服务适配器映射现有返回状态；无人工补证据、无Gold输入生成、无生产接入。',
        '首阶段通过后才扩展主集/回归；本阶段失败则停止。控制题虽来自现有主集，不代表完整回归集已验证。','',
        '|题号|资料|角色|基线通过/3|候选通过/3|候选执行错误|','|---|---|---|---:|---:|---:|']
    for r in table:md.append(f'|{r["id"]}|{r["variant"]}|{r["role"]}|{r["arms"]["baseline"]["consistent"]}|{r["arms"]["candidate"]["consistent"]}|{r["arms"]["candidate"]["errors"]}|')
    md+=['','自动目标门槛：'+str(summary['automatic_target_gate_passed']),'控制项警报：'+str(summary['control_alerts']),'','生成耗时/恢复：','```json',json.dumps(timing,ensure_ascii=False,indent=2),'```']
    esc=lambda v:html.escape(str(v))
    body='<h1>回答决策隔离实验</h1><p>10种固定上下文，基线/候选各运行3次。无生产接入。模型裁判不等于人工审核，未把本阶段子集成绩当全量成绩。</p>'
    if (OUT/'conclusion.md').exists():body+='<p><strong>结论：首阶段未通过，本版不采用，不扩大到全量主集/回归，也不接入生产。</strong> <a href="conclusion.md">完整结论与限制</a> · <a href="manual_audit.json">逐项审核说明</a></p>'
    body+='<table><tr><th>题号</th><th>资料</th><th>角色</th><th>基线通过/3</th><th>候选通过/3</th><th>候选错误</th></tr>'
    for r in table:body+=f'<tr><td>{esc(r["id"])}</td><td>{esc(r["variant"])}</td><td>{esc(r["role"])}</td><td>{r["arms"]["baseline"]["consistent"]}</td><td>{r["arms"]["candidate"]["consistent"]}</td><td>{r["arms"]["candidate"]["errors"]}</td></tr>'
    body+='</table><h2>生成耗时与格式恢复</h2><pre>'+esc(json.dumps(timing,ensure_ascii=False,indent=2))+'</pre>'
    for p in inputs:
        md+=['','## '+p['id']+' '+p['variant'],'',p['question']]
        body+='<details><summary>'+esc(p['id']+' / '+p['variant']+' / '+p['question'])+'</summary>'
        for r in sorted([r for r in rs if r['id']==p['id'] and r['variant']==p['variant']],key=lambda r:(r['repeat'],r['arm'])):
            j=r.get('gold_judgement')or{};out=r.get('output')or{}
            label=r['arm']+' / '+str(r['repeat'])+' / '+str(j.get('verdict','execution_error'))
            answer=(r.get('answer')or{}).get('text',str(r['error']))
            md+=['','### '+label,'',answer,'','行为：'+str(out.get('behavior'))+'；状态：'+str(out.get('status'))]
            body+='<h3>'+esc(label)+'</h3><pre>'+esc(answer)+'</pre><p>'+esc('行为：'+str(out.get('behavior'))+'；状态：'+str(out.get('status')))+'</p>'
            for q in j.get('rubrics',[]):
                if q['status'] in ['missing','contradicted','present','uncertain']:
                    text=q['id']+' '+q['status']+'：'+q['gold_text']+'；'+q['reason'];md+=['- '+text];body+='<p>'+esc(text)+'</p>'
            if out.get('parts'):body+='<details><summary>结构化决策与引文定位</summary><pre>'+esc(json.dumps(out['parts'],ensure_ascii=False,indent=2))+'</pre></details>'
        body+='</details>'
    (OUT/'report.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>回答决策隔离实验</title><style>body{font:16px/1.7 system-ui;max-width:1150px;margin:32px auto;padding:0 24px}td,th{border:1px solid #ccc;padding:8px}table{border-collapse:collapse}pre{white-space:pre-wrap;background:#f4f6f8;padding:15px}details{margin:15px 0}summary{cursor:pointer}</style>'+body,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
