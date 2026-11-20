"""Source-bound four-arm diagnostic report; never changes Gold or runtime."""
from collections import Counter
from hashlib import sha256
import html
import json
from pathlib import Path

from tools.experiment_035_context_ablation_20261119 import OUT, ARMS, ROOT, HISTORY, CASE
from tools.report_vector_packaging_20261116 import SourceRegistry


def main():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf8'))
    protocol=read('protocol.json');contexts=read('contexts.json');review=read('manual_review.json')
    rows=[json.loads(l) for l in (OUT/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==20 and len({(r['arm'],r['round']) for r in rows})==20
    annotations={(r['arm'],r['round']):r for r in review['records']}
    assert len(annotations)==20
    registry=SourceRegistry([])
    original=next(json.loads(l) for l in (HISTORY/'real_entry_simple.jsonl').read_text(encoding='utf8').splitlines() if json.loads(l)['id']==CASE)
    entries={e['source_id']:e for e in original['entries']}
    links={s:[dict(source=source,ids=registry.register(source)) for source in e['sources']] for s,e in entries.items()}
    assert not registry.misses
    write=lambda name,data:(OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    write('report_sources.json',registry.sources)
    summary={a:dict(tasks=len(rs:=[r for r in rows if r['arm']==a]),
        auto_verdicts=dict(Counter(r.get('judgement',{}).get('verdict','error') for r in rs)),
        target_claim=sum(annotations[(a,r['round'])]['target_claim'] for r in rs),
        rule_check_expansion=sum(annotations[(a,r['round'])]['rule_check_expansion'] for r in rs),
        core_complete=sum(annotations[(a,r['round'])]['core_complete'] for r in rs),
        other_unsupported=sum(bool(annotations[(a,r['round'])]['other_unsupported']) for r in rs),
        context_chars=contexts[a]['chars'],source_count=len(contexts[a]['source_ids'])) for a in ARMS}
    write('summary.json',dict(arms=summary,conclusion=review['conclusion'],limits=review['limits']))
    esc=lambda x:html.escape(str(x))
    def source_buttons(s):
        return ''.join(f'<span>{esc(p["source"]["filename"])} · {esc(p["source"]["location"])} '+
            ''.join(f'<button data-original-source="{esc(i)}">打开原资料</button>' for i in p['ids'])+'</span>' for p in links[s])
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>035 Context四组对照实验</title><style>
body{margin:0;background:#f5f7fb;color:#26364d;font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif}header{background:white;border-bottom:1px solid #dce3ed;padding:22px 32px}main{max-width:1600px;margin:20px auto;padding:0 20px}.panel{background:white;border:1px solid #dce3ed;border-radius:10px;padding:22px;margin:18px 0}h1{margin:0;color:#173e67;font-size:25px}h2{color:#173e67;font-size:21px}table{border-collapse:collapse;width:100%;table-layout:fixed}th,td{border-bottom:1px solid #dce3ed;padding:12px;vertical-align:top;overflow-wrap:anywhere}th{background:#eff4fb;text-align:left}.scroll{overflow:auto}.badge{display:inline-block;padding:3px 9px;border-radius:6px;background:#e7f4e9}.bad{background:#ffeaee;color:#a12635}.warn{background:#fff6cc}.answers th{width:25%}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fc;padding:14px;border-radius:6px;font:14px/1.8 system-ui,"Microsoft YaHei"}summary{cursor:pointer;color:#26578b}button{cursor:pointer;border:1px solid #cbd8e8;background:white;border-radius:5px;padding:5px 9px;color:#26578b;margin:4px}a{color:#26578b}small{color:#687991}details{margin:10px 0}#sourceStatus{position:sticky;bottom:0;background:#edf4fc;padding:10px;border:1px solid #cbd8e8}.answer{min-height:120px}</style><header><h1>035 · Context 四组因果诊断</h1><p>相同问题、模型和四项精简提示；仅改变证据组合。每组5次，共20次全新生成与独立Gold判分。</p></header><main>'''
    page+=f'<section class="panel"><h2>结论</h2><p>{esc(review["conclusion"])}</p><p>{esc(review["limits"])}</p><p>问题：{esc(protocol["question"])}</p><p>模型：{esc(protocol["model"])}；AI bundle：{esc(protocol["bundle_id"])}；temperature=0；max_output_tokens=1000。</p><p>这是生成器隔离实验，不是完整链路验收；自动Gold结果与逐条来源复核分别保留。没有修改运行代码、正式提示、Gold或持久化设置。</p></section>'
    page+='<section class="panel"><h2>四组结果</h2><table id="summary"><thead><tr><th>方案</th><th>实际输入</th><th>原具体机制完整复现</th><th>规则/设置检查扩展（探索性）</th><th>必答完整</th><th>任一无对应依据扩展</th><th>自动Gold原判</th></tr></thead><tbody>'
    for a,s in summary.items():
        page+=f'<tr><td>{esc(a+" · "+ARMS[a])}</td><td>{s["source_count"]}段 / {s["context_chars"]}字符</td><td>{s["target_claim"]}/5</td><td>{s["rule_check_expansion"]}/5</td><td>{s["core_complete"]}/5</td><td>{s["other_unsupported"]}/5</td><td>{esc(json.dumps(s["auto_verdicts"],ensure_ascii=False))}</td></tr>'
    page+='</tbody></table><p>原具体机制未完整复现，但A4有最接近的“确认并非规则或状态判断不当，排除实际失控”，已作为同类扩展标出；不能因未出现原句就说风险消失。规则/设置检查统计为运行后的探索性复核，未替换预设主要指标。一般复核质控状态/记录有S1/S2支持，不计为该错误。合理改写不要求逐字照抄，但不能新增资料未建立的因果关系。</p></section>'
    page+='<section class="panel"><h2>每次实际回答与来源复核</h2><p>同一行的“第N次”是重复编号，不表示四组具有相同随机种子或可以抵消输出随机性。所有回答均保留，没有择优。</p>'
    for n in range(1,6):
        page+=f'<h3>第{n}次</h3><table class="answers"><thead><tr>'+''.join(f'<th>{esc(a+" · "+ARMS[a])}</th>' for a in ARMS)+'</tr></thead><tbody><tr>'
        for a in ARMS:
            r=next(r for r in rows if r['arm']==a and r['round']==n);m=annotations[(a,n)]
            page+=f'<td data-arm="{a}" data-round="{n}"><span class="badge {"bad" if m["rule_check_expansion"] else ""}">规则/设置扩展：{"有" if m["rule_check_expansion"] else "无"}</span><p>{esc(m["reason"])}</p><pre class="answer">{esc(r.get("answer",r.get("error","")))}</pre><details><summary>代理来源复核（不覆盖自动原判）</summary><pre>{esc(json.dumps(m,ensure_ascii=False,indent=2))}</pre></details><details><summary>自动判分及逐项检查</summary><pre>{esc(json.dumps(r.get("judgement",{}),ensure_ascii=False,indent=2))}</pre></details><details><summary>本次实际引用来源</summary>'
            for s in r.get('citation_check',{}).get('cited_ids',[]):
                if s in entries:page+=f'<p><b>{esc(s)}</b> {source_buttons(s)}</p>'
            page+='</details></td>'
        page+='</tr></tbody></table>'
    page+='</section><section class="panel"><h2>实际Context（原文、原编号）</h2>'
    for a,c in contexts.items():page+=f'<details><summary>{esc(a+" · "+ARMS[a])} · {c["chars"]}字符</summary><pre>{esc(c["context"])}</pre></details>'
    page+='</section><section class="panel"><h2>直接证据与疑似干扰资料</h2>'
    for s in ['S1','S2','S3','S4']:page+=f'<details><summary>{esc(s)} · {esc(entries[s]["filename"])} · {esc(entries[s]["location"])}</summary>{source_buttons(s)}<pre>{esc(entries[s]["included_text"])}</pre></details>'
    safe_protocol={k:v for k,v in protocol.items() if k!='protected'}
    page+=f'</section><details class="panel"><summary>冻结协议与复核说明</summary><pre>{esc(json.dumps(safe_protocol,ensure_ascii=False,indent=2))}</pre></details><p id="sourceStatus">请双击同目录 open_report.cmd，然后点击资料按钮定位原页。</p></main>'
    page+='''<script>document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;const status=document.querySelector('#sourceStatus');const token=new URLSearchParams(location.search).get('token');if(!token){status.textContent='请双击同目录open_report.cmd后使用原资料定位。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();status.textContent=d.message||d.error;}catch(err){status.textContent=String(err)}});</script></html>'''
    (OUT/'report.html').write_text(page,encoding='utf8')
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n',encoding='utf8')
    write('report_build_audit.json',dict(rows=20,arms=4,source_entries=len(registry.sources),report_sha256=sha256(page.encode()).hexdigest()))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
