"""Application report template + paired answers; all upstream stages are frozen."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import base64
import gzip
import json
from pathlib import Path
import re

from tools.validate_prompt_factorial_20261119 import OUT, ROOT, HISTORY, ARMS, TARGETS, check_frozen
from tools.render_general_native_report import packed
from tools.report_vector_packaging_20261116 import SourceRegistry
from local_full_text_search.rag.evaluation.reporting import service

LABELS = {"baseline": "旧提示（本轮重新生成）", "behavior": "仅回答行为", "completeness": "仅完整性与引用", "combined": "两者组合"}


def summarize(rows):
    first = {(r["id"], r["arm"]): r for r in rows if r["round"] == 1}
    summary = {}
    for arm in ARMS:
        summary[arm] = {}
        for status in ("reviewed", "pending", "disputed"):
            selected = [r for r in rows if r["arm"] == arm and r["round"] == 1 and r["review_status"] == status]
            base = {r["id"]: r for r in rows if r["arm"] == "baseline" and r["round"] == 1 and r["review_status"] == status}
            gains = [r["id"] for r in selected if r["id"] in base and r.get("judgement", {}).get("verdict") == "consistent"
                     and base[r["id"]].get("judgement", {}).get("verdict") != "consistent"]
            losses = [r["id"] for r in selected if r["id"] in base and r.get("judgement", {}).get("verdict") != "consistent"
                      and base[r["id"]].get("judgement", {}).get("verdict") == "consistent"]
            summary[arm][status] = dict(cases=len(selected), verdicts=dict(Counter(r.get("judgement", {}).get("verdict", "error") for r in selected)),
                legacy_guard_verdicts=dict(Counter(r.get("legacy_guard_judgement", {}).get("verdict", "error") for r in selected)), gains=gains, losses=losses)
    repeated = {case_id: {arm: [r.get("judgement", {}).get("verdict", "error") for r in sorted(rows, key=lambda r:r["round"])
                               if r["id"] == case_id and r["arm"] == arm] for arm in ARMS} for case_id in TARGETS}
    return dict(tasks=len(rows), expected_tasks=500, complete=len(rows)==500,
        by_arm=summary, target_repeats=repeated, errors=[{k:r[k] for k in ("id","arm","round","error")} for r in rows if "error" in r])


def render():
    check_frozen()
    rows = [json.loads(l) for l in (OUT / "answers.jsonl").read_text(encoding="utf8").splitlines()]
    if (OUT / "transport_recovery.jsonl").exists():
        recovered = {(r['id'],r['arm'],r['round']):r for r in
            (json.loads(l) for l in (OUT / 'transport_recovery.jsonl').read_text(encoding='utf8').splitlines())}
        rows = [recovered.get((r['id'],r['arm'],r['round']),r) for r in rows]
    summary = summarize(rows)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf8")
    # Decode the historical template directly; no second large cache is needed.
    historical_html = (HISTORY / 'report.html').read_text(encoding='utf8')
    offset = historical_html.index('const PACKED=') + len('const PACKED=')
    historical_arms, _ = json.JSONDecoder().raw_decode(historical_html[offset:])
    data = json.loads(gzip.decompress(base64.b64decode(historical_arms['simple'])))
    registry = SourceRegistry([])
    registry.sources.update(json.loads((HISTORY / "report_sources.json").read_text(encoding="utf8")))
    contexts = {}
    for line in (HISTORY / "real_entry_simple.jsonl").open(encoding="utf8"):
        c = json.loads(line)
        contexts[c["id"]] = {k:c[k] for k in ("context", "entries")}
    for r in rows:
        r["citation_sources"] = []
        cited = r.get("citation_check", {}).get("cited_ids", [])
        for e in contexts[r["id"]]["entries"]:
            if e["source_id"] not in cited: continue
            for source in e["sources"]:
                ids = registry.register(source)
                r["citation_sources"].append(dict(source_id=e["source_id"], filename=source["filename"], location=source["location"], source_ids=ids))
    assert not registry.misses, registry.misses
    # UI checks use this exact rendering snapshot, never a moving experiment log.
    (OUT / 'report_answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
    for c in data["cases"]:
        c["answer"] = {}
        c["gold_judgement"] = {}
        c["generation"] = {}
        c["actual_behavior"] = "not_run"
        c.pop("api_validation", None)
        c["timing"] = {}
    data["run"] = dict(run_id="prompt_factorial_20261119", ai_bundle_id="g000033-71ec3c52f3",
        mode="固定Context提示2×2对照；旧提示也在本轮重新生成；评测格式修正单列；019不修复")
    data["summary"]["answer_judgement_counts"] = {}
    page = service.render_html({})
    page = page.replace('<script>\nconst DATA={};', '<script>\n(async()=>{\n' +
        "async function unpack(s){const a=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([a]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}\n" +
        "let DATA=await unpack('" + packed(data) + "');\n" +
        "const PROMPT_RESULTS=await unpack('" + packed(rows) + "');\n" +
        "const PROMPT_CONTEXTS=await unpack('" + packed(contexts) + "');\n" +
        "const PROMPT_SUMMARY=" + json.dumps(summary, ensure_ascii=False).replace("</", "<\\/") + ";\n" +
        "const PROMPT_LABELS=" + json.dumps(LABELS, ensure_ascii=False) + ";let activeArm='baseline';\n", 1)
    page = page.replace('</script></body>', "})().catch(e=>{document.getElementById('banner').textContent='加载失败：'+e;});\n</script></body>", 1)
    header = '''<section class="panel" id="promptExperiment"><h2>旧提示与新提示 · 固定Context四组对照</h2>
<p>111题上游候选、BGE分窗口最高分、简单选段和实际正文全部固定。四组只改变生成提示；旧提示本轮重新生成，避免把不同批次波动算成提升。019保留观察、不修复。50道reviewed、60道pending、1道disputed分别统计。</p>
<p>004的型号格式核验修正单列；050缺乏引文适用型号的直接证明，仍保留不确定。自动一致不等于人工确认。下方阶段排名相同是实验控制，不是重新检索后的收益。</p>
<div id="experimentSummary"></div><div class="toolbar"><label>原程序详细诊断方案 <select id="promptArm"></select></label><label>逐题前后比较 <select id="promptCase"></select></label></div>
<div id="promptCompare"></div><p id="sourceStatus">双击同目录open_report.cmd后，点击按钮可打开原资料对应页。</p>
<details><summary>实验提示全文与冻结协议</summary><pre id="promptProtocol"></pre></details></section>'''
    page = page.replace('<main>', '<main>' + header, 1)
    page = page.replace('</head>', '<style>#promptExperiment{min-width:0}#experimentSummary table{width:100%;table-layout:fixed}#experimentSummary th,#experimentSummary td{white-space:normal;overflow-wrap:anywhere;vertical-align:top}#experimentSummary th:first-child{width:14%}#experimentSummary th:last-child{width:30%}#promptCompare .panel{min-width:0}#supplementalAudit table{width:100%;table-layout:fixed}#supplementalAudit td{overflow-wrap:anywhere}</style></head>', 1)
    page = page.replace('<b>${esc(e.evidence_id)}</b>', '<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page = page.replace('<pre>${content}</pre>', '${sourceLinks(c)}<pre>${content}</pre>')
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf8"))
    service_rows = [json.loads(l) for l in (OUT / 'service_entry.jsonl').read_text(encoding='utf8').splitlines()] if (OUT / 'service_entry.jsonl').exists() else []
    if (OUT / 'service_entry_judge_recovery.jsonl').exists():
        repairs = {(r['id'],r['arm']):r for r in (json.loads(l) for l in (OUT / 'service_entry_judge_recovery.jsonl').read_text(encoding='utf8').splitlines())}
        service_rows = [repairs.get((r['id'],r['arm']),r) for r in service_rows]
    manual = json.loads((OUT / 'manual_review.json').read_text(encoding='utf8')) if (OUT / 'manual_review.json').exists() else {'records':[]}
    service_view = [dict(id=r['id'],arm=r['arm'],context_equal_frozen=r.get('context_equal_frozen'),judgement=r.get('judgement'),answer=r.get('result'),error=r.get('error')) for r in service_rows]
    acceptance = json.loads((OUT/'final_acceptance.json').read_text(encoding='utf8')) if (OUT/'final_acceptance.json').exists() else None
    supplemental = "const FINAL_ACCEPTANCE="+json.dumps(acceptance,ensure_ascii=False).replace('</','<\\/')+";const SERVICE_RESULTS="+json.dumps(service_view,ensure_ascii=False).replace('</','<\\/')+";const MANUAL_REVIEW="+json.dumps(manual,ensure_ascii=False).replace('</','<\\/')+";\n"
    script = "const PROMPT_PROTOCOL=" + json.dumps({k:v for k,v in protocol.items() if k!='protected'},ensure_ascii=False).replace("</", "<\\/") + ";\n" + r'''
function sourceLinks(r){return (r.source_ids||[]).map(id=>`<button data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
function activatePrompt(arm){activeArm=arm;for(const c of DATA.cases){const r=PROMPT_RESULTS.find(r=>r.id===c.id&&r.arm===arm&&r.round===1);c.answer=r?{text:r.answer||'',status:r.error?'error':'answered',model:r.model,source_ids:r.citation_check?.cited_ids||[],warnings:['冻结Context生成器隔离实验']}:{text:'',status:'not_run'};c.gold_judgement=r?.judgement||{};c.actual_behavior=r?.judgement?.behavior||'not_run';c.generation={draft_answer:r?.answer||''};}DATA.run.run_id='prompt_factorial_'+arm;DATA.summary.answer_judgement_counts=Object.fromEntries(Object.entries(PROMPT_SUMMARY.by_arm[arm]).flatMap(([s,v])=>Object.entries(v.verdicts)).reduce((m,[k,n])=>m.set(k,(m.get(k)||0)+n),new Map()));}
function renderPromptSummary(){document.getElementById('experimentSummary').innerHTML=`<p><b>完成 ${PROMPT_SUMMARY.tasks}/500 次生成与独立判分任务</b>；每次任务分别请求生成和裁判，重点7题每组3次。${PROMPT_SUMMARY.complete?'已全部完成':'仍在运行，当前结果不能作为最终结论'}</p><table><thead><tr><th>方案</th><th>正式50题</th><th>待审60题</th><th>争议1题</th><th>正式变化（相对本轮旧提示）</th></tr></thead><tbody>${Object.keys(PROMPT_LABELS).map(a=>`<tr><td>${esc(PROMPT_LABELS[a])}</td>${['reviewed','pending','disputed'].map(s=>`<td>${esc(JSON.stringify(PROMPT_SUMMARY.by_arm[a][s].verdicts))}</td>`).join('')}<td>变好：${esc(PROMPT_SUMMARY.by_arm[a].reviewed.gains.join('、')||'无')}<br>变差：${esc(PROMPT_SUMMARY.by_arm[a].reviewed.losses.join('、')||'无')}</td></tr>`).join('')}</tbody></table>`;}
function renderPromptCompare(){const id=document.getElementById('promptCase').value;const c=DATA.cases.find(c=>c.id===id);if(!c)return;const ctx=PROMPT_CONTEXTS[id];document.getElementById('promptCompare').innerHTML=`<h3>${esc(id)} · ${esc(c.question)}${id==='rag-biz-019'?'（本轮冻结，不修复）':''}</h3><p>审核状态：${esc(c.review_status)}；预期行为：${esc(c.expected_behavior)}；实际Context ${ctx.context.length}字符，四组SHA256一致。</p><div class="grid2">${Object.keys(PROMPT_LABELS).map(a=>{const rs=PROMPT_RESULTS.filter(r=>r.id===id&&r.arm===a).sort((a,b)=>a.round-b.round);return `<section class="panel prompt-answer" data-arm="${a}"><h3>${esc(PROMPT_LABELS[a])}</h3>${rs.map(r=>`<details ${r.round===1?'open':''}><summary>第${r.round}次 · ${esc(r.judgement?.verdict||r.error||'未完成')}</summary><pre class="answer-text">${esc(r.answer||r.error||'')}</pre><p>${esc(r.judgement?.behavior_reason||'')}</p><details><summary>逐项必答、禁止主张、实际引用核验</summary><pre>${esc(JSON.stringify(r.judgement,null,2))}</pre></details><details><summary>答案引用的原资料</summary>${(r.citation_sources||[]).map(s=>`<p>${esc(s.source_id+' · '+s.filename+' · '+s.location)} ${sourceLinks(s)}</p>`).join('')||'无事实引用'}</details></details>`).join('')||'未完成'}</section>`;}).join('')}</div><details><summary>四组实际发送的同一Context</summary><pre>${esc(ctx.context)}</pre></details><details><summary>本题Gold必答项（仅评测器使用）</summary><pre>${esc(JSON.stringify({answer_checks:c.answer_checks,forbidden_claims:c.forbidden_claims,missing_fields:c.missing_fields},null,2))}</pre></details>`;}
document.getElementById('promptArm').innerHTML=Object.entries(PROMPT_LABELS).map(([a,n])=>`<option value="${a}">${esc(n)}</option>`).join('');
document.getElementById('promptCase').innerHTML=DATA.cases.map(c=>`<option value="${esc(c.id)}">${esc(c.id+' · '+c.question)}</option>`).join('');
document.getElementById('promptProtocol').textContent=JSON.stringify(PROMPT_PROTOCOL,null,2);
document.getElementById('promptCase').addEventListener('change',renderPromptCompare);
document.getElementById('promptArm').addEventListener('change',e=>{activatePrompt(e.target.value);renderBanner();renderSummary();renderPriority();renderCharts();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();});
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const status=document.getElementById('sourceStatus');const token=new URLSearchParams(location.search).get('token');if(!token){status.textContent='请双击同目录open_report.cmd后打开原资料。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();status.textContent=d.message||d.error;}catch(error){status.textContent=String(error);}});
activatePrompt('baseline');renderPromptSummary();renderPromptCompare();
'''
    script = supplemental + script + r'''
const verdictLabels={consistent:'一致',incomplete:'不完整',needs_clarification:'需澄清',incorrect:'错误',uncertain:'不确定',behavior_mismatch:'行为不符',error:'执行错误'};
document.querySelectorAll('#experimentSummary tbody tr').forEach((tr,i)=>{const arm=Object.keys(PROMPT_LABELS)[i];['reviewed','pending','disputed'].forEach((status,j)=>{const d=PROMPT_SUMMARY.by_arm[arm][status];tr.cells[j+1].innerHTML=`<b>一致 ${d.verdicts.consistent||0} / ${d.cases}</b><br>`+Object.entries(d.verdicts).filter(([k])=>k!=='consistent').map(([k,n])=>`${esc(verdictLabels[k]||k)} ${n}`).join('；');});});
if(FINAL_ACCEPTANCE){const conclusion=document.createElement('section');conclusion.className='panel';conclusion.innerHTML=`<h3>最终结论：${esc(FINAL_ACCEPTANCE.recommendation)}</h3><p>本轮Context及非system请求参数核对 ${FINAL_ACCEPTANCE.frozen_context_and_non_system_payload_checks}/500；真实入口Context一致 ${FINAL_ACCEPTANCE.real_service_context_equal}/14。初始执行故障 ${FINAL_ACCEPTANCE.initial_execution_errors} 次，补跑后剩余 ${FINAL_ACCEPTANCE.remaining_errors.length} 次。默认提示仍为baseline；Adapter关闭；用户持久化设置未改。</p><p>组合提示对部分漏答和引用有收益，但021条件分支连续失败，035重复结果波动；不满足替换默认提示的稳定性要求。022、030及020存在裁判复核问题，见下方原回答与独立复核，不将自动分当人工准确率。</p><p>重点测试67通过。全RAG测试579通过、10项既有跨文档共识测试失败、1跳过、1预期失败；不能报告全库测试通过。</p>`;document.getElementById('promptExperiment').prepend(conclusion);}
const extra=document.createElement('details');extra.id='supplementalAudit';extra.innerHTML=`<summary>真实问答入口14次与人工复核记录</summary><p>以下为真实本地召回、BGE和AnswerService入口；每题旧提示与组合提示各1次，不计入冻结输入的500次结果。</p><table><thead><tr><th>题目</th><th>方案</th><th>Context一致</th><th>自动结果</th><th>回答和逐项检查</th></tr></thead><tbody>${SERVICE_RESULTS.map(r=>`<tr><td>${esc(r.id)}</td><td>${esc(PROMPT_LABELS[r.arm])}</td><td>${esc(String(r.context_equal_frozen))}</td><td>${esc(r.judgement?.verdict||r.error||'未完成')}</td><td><details><summary>展开</summary><pre>${esc(JSON.stringify(r.answer,null,2))}</pre><pre>${esc(JSON.stringify(r.judgement,null,2))}</pre></details></td></tr>`).join('')}</tbody></table><p>以下复核保留原自动分，不覆盖Gold或自动判分；只复核列出的记录，不代表人工审完500条。</p><pre>${esc(JSON.stringify(MANUAL_REVIEW,null,2))}</pre>`;document.getElementById('promptExperiment').append(extra);
'''
    assert page.count('\ninit();\n') == 1
    page = page.replace('\ninit();\n', '\n' + script + '\ninit();\n', 1)
    (OUT / "report.html").write_text(page, encoding="utf8")
    (OUT / "report_sources.json").write_text(json.dumps(registry.sources, ensure_ascii=False, indent=2), encoding="utf8")
    (OUT / "open_report.cmd").write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n', encoding="utf8")
    print("REPORT", OUT / "report.html", summary["tasks"], flush=True)


if __name__ == "__main__":
    render()
