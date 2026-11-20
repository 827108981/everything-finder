"""Offline paired reports. Gold and frozen observations never enter retrieval."""
from __future__ import annotations

import hashlib
import html
import json
import re
import sys
import zipfile
from pathlib import Path


def write_comparison_report(output_dir, cases, captures, answers, metrics, source_files):
    output_dir = Path(output_dir)
    sources = {}
    unresolved = []
    by_name = {}
    for filename in source_files:
        path = Path(filename)
        by_name.setdefault(path.name.casefold(), set()).add(str(path))

    def evidence_link(evidence):
        raw = evidence.get('file_path') or evidence.get('file', '')
        path = Path(raw)
        archive = None
        member = None
        if ' > ' in raw:
            archive_text, member = raw.split(' > ', 1)
            archive = Path(archive_text)
            member = member.replace('\\', '/')
            if archive.is_file() and zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as z:
                    if member not in z.namelist():
                        decoded_matches = []
                        for name in z.namelist():
                            try:
                                if name.encode('cp437').decode('gbk') == member:
                                    decoded_matches.append(name)
                            except (UnicodeEncodeError, UnicodeDecodeError):
                                continue
                        if len(decoded_matches) == 1:
                            member = decoded_matches[0]
                        else:
                            archive = None
            else:
                archive = None
        matches = [str(path)] if path.is_file() else sorted(by_name.get(path.name.casefold(), []))
        if archive:
            matches = [raw]
        # Relative corpus paths disambiguate identical basenames; never choose arbitrarily.
        if len(matches) > 1 and len(path.parts) > 1:
            matches = [p for p in matches if p.replace('\\', '/').casefold().endswith(raw.replace('\\', '/').casefold())]
        loc = evidence.get('location') or evidence.get('position') or evidence.get('location_text', '')
        page = evidence.get('page_number') or evidence.get('page')
        slide = evidence.get('slide_number')
        if len(matches) != 1 or (not archive and not Path(matches[0]).is_file()):
            unresolved.append({'file': raw, 'location': loc, 'matches': matches})
            return '<span class="bad">原资料未唯一定位：' + html.escape(raw) + '</span>'
        path = Path(matches[0]).resolve()
        if path.suffix.lower() in {'.ppt', '.pptx'} and not slide:
            slide, page = page, None
        if not page and not slide:
            m = re.search(r'第\s*(\d+)\s*(页|张)', loc)
            if m:
                if m[2] == '张': slide = int(m[1])
                else: page = int(m[1])
        location = {'location_text': loc, 'page_number': page, 'slide_number': slide}
        key = hashlib.sha256((str(path) + json.dumps(location, ensure_ascii=False)).encode()).hexdigest()[:24]
        sources[key] = {'path': str(path), 'location': location}
        if archive:
            sources[key].update(archive=str(archive.resolve()), member=member)
        place = f'第 {slide} 张' if slide else f'第 {page} 页' if page else loc
        uri = archive.resolve().as_uri() if archive else path.as_uri() + (f'#page={page}' if page else '')
        return f'<button class="source" data-source="{key}">打开原资料 · {html.escape(path.name)} · {html.escape(place)}</button><a class="fallback" href="{html.escape(uri, quote=True)}">文件</a>'

    def pretty(value):
        return '<pre>' + html.escape(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'

    def candidate_overview(stages):
        fields = ('chunk_id', 'filename', 'location', 'lexical_rank', 'lexical_score', 'vector_rank', 'vector_score', 'fused_rank', 'fused_score', 'rerank_score', 'final_rank', 'selected', 'reason', 'constraint_decision', 'duplicate_of')
        return {stage: [{key: row[key] for key in fields if key in row and row[key] is not None} for row in candidates] for stage, candidates in stages.items()}

    rows = []
    for case in cases:
        cid = case['id']
        gold = ''.join('<div class="evidence">' + evidence_link(e) + pretty(e.get('quote', '')) + '</div>' for e in case.get('gold_evidence', []))
        arms = []
        verdicts = []
        for arm, title in [('off', '修改前 · Adapter 关闭'), ('on', '修改后 · Adapter 开启')]:
            capture = captures[arm][cid]
            answer = answers.get(('A' if arm == 'off' else 'B', cid), {})
            judgement = answer.get('judgement', {})
            verdict = judgement.get('verdict') or '未完成'
            verdicts.append(verdict)
            stage_rows = ''.join(f'<tr><td>{html.escape(stage)}</td><td>{s.get("gold_evidence_found", 0)}/{s.get("gold_evidence_total_declared", 0)}</td><td>{s.get("first_gold_rank", "—")}</td></tr>' for stage, s in capture.get('scores', {}).items())
            context_sources = ''.join('<div class="evidence">' + ''.join(evidence_link(s) for s in entry.get('sources', [])) + pretty(entry.get('included_text', '')) + '</div>' for entry in capture.get('entries', []))
            plan = next((e['query_plan'] for e in capture.get('events', []) if 'query_plan' in e), {})
            arms.append('<section><h3>' + title + '</h3><p>自动判分：<b>' + html.escape(verdict) + '</b></p>' + pretty(answer.get('answer', {}).get('text') or answer.get('error') or '未完成') + '<details><summary>缺失项、额外主张与判分理由</summary>' + pretty(judgement) + '</details><table><tr><th>阶段</th><th>完整引文</th><th>首条排名</th></tr>' + stage_rows + '</table><details><summary>实际 Context 与来源</summary>' + context_sources + '<details><summary>发送正文（完整）</summary>' + pretty(capture.get('context', '')) + '</details></details><details><summary>QueryPlan 与激活状态</summary>' + pretty(plan) + '</details><details><summary>各阶段候选、分数与排除原因</summary><p>全部候选排名与排除信息；未入选候选的完整正文保留在原始 trace JSONL 中。</p>' + pretty(candidate_overview(capture.get('candidate_stages', {}))) + '</details><details><summary>答案复用与运行来源</summary>' + pretty(answer.get('verification_provenance', {'note': '见原始运行记录；不表示独立重复实验'})) + '</details></section>')
        state = '退化' if verdicts[0] == 'consistent' and verdicts[1] != 'consistent' else '改善' if verdicts[0] != 'consistent' and verdicts[1] == 'consistent' else '相同或待审'
        label = '正式计分' if case.get('review_status') == 'reviewed' else '仅诊断，不计分'
        rows.append(f'<details class="case" id="{html.escape(cid)}" data-state="{state}"><summary>{html.escape(cid)} · {html.escape(case["question"])} <span class="pill">{state} · {label}</span></summary><div class="body"><p>预期行为：{html.escape(case.get("expected_behavior", ""))}；审核状态：{html.escape(case.get("review_status", ""))}</p><details><summary>Gold：必答项、禁答项和原文证据</summary>' + pretty({k: case.get(k) for k in ('answer_checks', 'required_facts', 'forbidden_claims', 'constraints', 'missing_fields', 'corpus_snapshot')}) + gold + '</details><div class="grid">' + ''.join(arms) + '</div></div></details>')
    css = '''body{margin:0;background:#f5f7fb;color:#172033;font:14px/1.6 "Microsoft YaHei",sans-serif}header{background:#172033;color:white;padding:22px}main{max-width:1550px;margin:auto;padding:22px}.panel,.case{background:white;border:1px solid #e1e6ef;border-radius:12px;margin:14px 0;padding:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}.grid section{min-width:0}summary{cursor:pointer;font-weight:bold;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f8fafc;padding:12px;max-height:560px;overflow:auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #e1e6ef;text-align:left}.pill{color:#2563eb;font-size:12px}.evidence{border-left:4px solid #94a3b8;padding:8px;margin:8px 0}button,input,select{padding:8px;border:1px solid #cbd5e1;border-radius:6px;background:white;margin:4px;font:inherit}button{cursor:pointer;color:#2563eb}.bad{color:#b42318}.fallback{margin:8px}.body{padding:8px}#status{white-space:pre-wrap}@media(max-width:900px){.grid{grid-template-columns:1fr}}'''
    script = '''const token=new URLSearchParams(location.search).get('token');
document.addEventListener('click',async e=>{const b=e.target.closest('[data-source]');if(!b)return;const s=document.querySelector('#status');if(!token){s.textContent='请双击同目录 open_report.cmd 打开报告，随后点击即可定位原资料页。直接双击 HTML 可阅读全部对照，但浏览器不能直接调用 PowerPoint。';return;}b.disabled=true;try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.source})});const d=await r.json();s.textContent=d.message||d.error;}catch(err){s.textContent='原资料定位失败：'+err;}finally{b.disabled=false;}});
function filter(){let q=document.querySelector('#search').value.toLowerCase(),s=document.querySelector('#state').value;document.querySelectorAll('.case').forEach(c=>c.hidden=!(c.textContent.toLowerCase().includes(q)&&(!s||c.dataset.state===s)));}document.querySelector('#search').oninput=filter;document.querySelector('#state').onchange=filter;'''
    labels = {'main_reviewed': '主集 · 已审核', 'main_diagnostic': '主集 · 仅诊断', 'regression_reviewed': '回归集 · 已审核', 'regression_diagnostic': '回归集 · 仅诊断'}
    overview = '<section class="panel"><h2>分组前后对照</h2><table><tr><th>集合</th><th>题数</th><th>关闭：答案判分</th><th>开启：答案判分</th><th>Context 完整 Gold 题数：前 → 后</th></tr>'
    for group, data in metrics.get('cohorts', {}).items():
        a, b = data['arms']['off'], data['arms']['on']
        overview += '<tr><td>' + labels.get(group, html.escape(group)) + '</td><td>' + str(data['count']) + '</td><td>' + html.escape(str(a['answers'])) + '</td><td>' + html.escape(str(b['answers'])) + '</td><td>' + str(a['stages']['context']['all_gold_retained']) + ' → ' + str(b['stages']['context']['all_gold_retained']) + '</td></tr>'
    overview += '</table><p>一致性是自动裁判结果，不等于业务人员确认的正确率。相同 Context 下的回答变化不能归因于检索算法。</p></section>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG 前后对照报告</title><style>' + css + '</style><header><h1>RAG 前后对照报告</h1>同一测试集逐题配对 · 实际引文、Context 和答案 · 自动裁判结论仍需业务审核</header><main><div class="panel">共 ' + str(len(cases)) + ' 题。未审核题仅诊断，旧版本不同题数的总分不可直接比较。默认 Adapter 仍关闭。<p id="status">双击 open_report.cmd 可启用原资料页定位；HTML 可独立离线阅读。</p><a href="conclusion.md">完整结论</a> · <a href="final_acceptance.json">验收记录</a></div>' + overview + '<details class="panel"><summary>各阶段完整指标、改善与退化明细</summary>' + pretty(metrics) + '</details><div class="panel"><input id="search" placeholder="题号、问题、资料名"><select id="state"><option value="">全部</option><option>改善</option><option>退化</option><option>相同或待审</option></select></div>' + ''.join(rows) + '<script>' + script + '</script></main></html>'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'report.html').write_text(document, encoding='utf8')
    (output_dir / 'report_sources.json').write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding='utf8')
    (output_dir / 'report_link_audit.json').write_text(json.dumps({'resolved_locations': len(sources), 'unresolved': unresolved}, ensure_ascii=False, indent=2), encoding='utf8')
    root = Path(__file__).resolve().parents[4]
    launcher = f'@echo off\r\ncd /d "{root}"\r\n"{sys.executable}" -m tools.serve_rag_report --report "%~dp0report.html"\r\n'
    (output_dir / 'open_report.cmd').write_text(launcher, encoding='utf8')
    return {'cases': len(cases), 'resolved_locations': len(sources), 'unresolved': len(unresolved)}
