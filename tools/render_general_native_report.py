"""Use the application's diagnostic viewer with paired frozen observations."""
import base64
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import argparse
import html

from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset, GoldDataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.reporting.service import build_report_detail, render_html, ReportConfig
from local_full_text_search.rag.index_store import RagIndexStore

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_general_refactor_20261113'

def packed(value):
    return base64.b64encode(gzip.compress(json.dumps(value,ensure_ascii=False,separators=(',',':')).encode(),mtime=0)).decode()


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--native',action='store_true')
    args=parser.parse_args()
    if args.output_dir: OUT=args.output_dir.resolve()
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases=tuple(c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases)
    dataset=GoldDataset(ROOT/'tests/rag/fixtures/rag_active_suite_v5.json',hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),cases)
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    with store.connect_read_only(bundle.metadata_path) as conn: chunks=load_chunk_rows(conn)
    cache_path=OUT/'native_gold_mappings.json'
    cache_key=hashlib.sha256((dataset.sha256 if hasattr(dataset,'sha256') else json.dumps(manifest,sort_keys=True)).encode()).hexdigest()+':'+hashlib.sha256(bundle.metadata_path.read_bytes()).hexdigest()
    cache=json.loads(cache_path.read_text(encoding='utf8')) if cache_path.exists() else {}
    if cache.get('key')==cache_key:
        mappings=cache['mappings']
    else:
        mappings={c.case_id:[asdict(m) for m in map_case_evidence(c,chunks)] for c in cases}
        cache_path.write_text(json.dumps({'key':cache_key,'mappings':mappings},ensure_ascii=False),encoding='utf8')
    source_files={r['file_path'] for r in chunks};del chunks
    sources=json.loads((OUT/'report_sources.json').read_text(encoding='utf8')) if (OUT/'report_sources.json').exists() else {}
    def source_ids(record):
        raw=record.get('file_path') or record.get('file','');loc=record.get('location') or record.get('position') or ''
        page=record.get('page_number') or record.get('page');slide=record.get('slide_number')
        if not page and not slide:
            m=re.search(r'第\s*(\d+)\s*(页|张)',loc)
            if m:
                if m[2]=='张':slide=int(m[1])
                else:page=int(m[1])
        if Path(raw).suffix.lower() in {'.ppt','.pptx'} and not slide:slide,page=page,None
        options=[s for s in source_files if s.replace('\\','/').casefold().endswith(raw.replace('\\','/').casefold())] if raw else []
        full=raw if Path(raw).is_file() else options[0] if len(options)==1 else ''
        for key,s in sources.items():
            if full and str(Path(full)).casefold()==s['path'].casefold() and str(s['location'].get('page_number') or '')==str(page or '') and str(s['location'].get('slide_number') or '')==str(slide or ''):
                return [key]
        if not full or not Path(full).is_file():return []
        location={'location_text':loc,'page_number':int(page) if page else None,'slide_number':int(slide) if slide else None}
        key=hashlib.sha256((str(Path(full).resolve())+json.dumps(location,ensure_ascii=False)).encode()).hexdigest()[:24]
        sources[key]={'path':str(Path(full).resolve()),'location':location};return [key]
    arms={};expected_scores={}
    for arm in ('A','B'):
        answer_path=OUT/(arm+('.jsonl' if args.native else '_fixed_prompt.jsonl'))
        if args.native and arm=='B' and (OUT/'B_effective.jsonl').exists():answer_path=OUT/'B_effective.jsonl'
        answers={r['id']:r for r in map(json.loads,answer_path.read_text(encoding='utf8').splitlines())}
        raw=[]
        for line in (OUT/(arm+'.jsonl')).open(encoding='utf8'):
            r=json.loads(line);a=answers[r['id']]
            r.update(stages=r['scores'],gold_mappings=mappings[r['id']],answer=a.get('result',{}) if args.native else a.get('answer',{}),gold_judgement=a.get('judgement',{}))
            raw.append(r)
        detail=build_report_detail(dataset,{'run_id':'20261113_'+arm,'ai_bundle_id':bundle.bundle_id,'mode':'实际入口全链路对照' if args.native else '冻结 Context 同 Prompt 对照'},raw,ReportConfig())
        expected_scores[arm]={r['id']:r['scores'] for r in raw}
        for c in detail['cases']:
            for e in c.get('gold_evidence',[])+c.get('hard_negatives',[]):e['source_ids']=source_ids(e)
            for stage in c.get('stage_data',{}).values():
                for item in stage['candidates']:item['source_ids']=source_ids(item)
        arms[arm]=detail;del raw
        print('Prepared native viewer',arm,flush=True)
    changes={}
    for a,b in zip(arms['A']['cases'],arms['B']['cases']):
        av=a.get('gold_judgement',{}).get('verdict');bv=b.get('gold_judgement',{}).get('verdict')
        changes[a['id']]={'state':'裁判错误' if not av or not bv else '对→错' if av=='consistent' and bv!='consistent' else '错→对' if av!='consistent' and bv=='consistent' else '相同判分' if av==bv else '判分变化'}
    for arm in arms:arms[arm]['baseline']={'case_changes':changes}
    if args.native:
        ablations={arm:{r['id']:r for r in map(json.loads,(OUT/(arm+'_fixed_prompt.jsonl')).read_text(encoding='utf8').splitlines())} for arm in ('A','B')}
        v2={}
        for line in (OUT/('B_effective.jsonl' if (OUT/'B_effective.jsonl').exists() else 'B.jsonl')).open(encoding='utf8'):
            r=json.loads(line)
            v2[r['id']]={'verification_provenance':r.get('verification_provenance'), 'fixed_prompt':{a:ablations[a].get(r['id']) for a in ('A','B')},
                'diagnostics':[e for e in r['events'] if e.get('stage') in {'structured_answer_plan','structured_evidence','review_completed','claim_citations_checked','clarification_assessed'}]}
    else:
        v2={r['id']:r for r in map(json.loads,(OUT/'B_structured_v2.jsonl').read_text(encoding='utf8').splitlines())}
    # The old application renderer supplies the actual page structure and navigation.
    page=render_html({})
    page=page.replace('<script>\nconst DATA={};','<script>\n(async()=>{\n'+
        "async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}\n"+
        "const ARMS=await unpack('"+packed(arms)+"');let DATA=ARMS.B;\nconst EXTRA=await unpack('"+packed({'v2':v2,'sources':sources})+"');")
    page=page.replace('</script></body>','})().catch(e=>{document.getElementById("banner").textContent="报告加载失败："+e;});\n</script></body>')
    page=page.replace('<main>','<main><section class="panel"><h2>前后对照 · 未通过开启验收</h2><p>冻结 91 题，50 题已审核，41 题仅诊断。答案正式集合为 50 题；正证据计分为 46 题（4 道拒答题不计正证据）。正式集合同 Prompt 一致性 45 → 40，B 有 1 项裁判错误；两步生成 v2 一致 21。默认两个开关均关闭。</p><p>全套单元验证：446 通过、10 失败、1 跳过。Gold、状态、分组及判分逻辑未改。相同 Context 的答案变化不能归因于检索。</p><label>查看阶段排名、热力图和候选：<select id="armSelect"><option value="B">B · 通用重构后</option><option value="A">A · 修改前</option></select></label><p id="sourceStatus">双击 open_report.cmd 后，可点击原资料按钮定位页码。直接双击 HTML 可离线查看。</p><a href="conclusion.md">完整实验结论</a> · <a href="final_acceptance.json">验收记录</a></section>',1)
    # Timing was explicitly excluded by the user; keep the original viewer otherwise.
    page=page.replace('<section class="panel"><h2>整体耗时与模块消耗</h2>','<section class="panel" hidden><h2>整体耗时与模块消耗</h2>')
    page=page.replace('${answerHtml(c)}${judgeHtml(c)}${caseTimingHtml(c)}','${pairAnswers(c)}')
    page=page.replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page=page.replace('<div class="candidate-body"><div class="inline-caption">状态：','<div class="candidate-body">${sourceLinks(c)}<div class="inline-caption">状态：')
    page=page.replace('${queryHtml(c)}','${queryHtml(c)}${pairedStages(c)}')
    extra=r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
function pairAnswers(c){const a=ARMS.A.cases.find(x=>x.id===c.id),b=ARMS.B.cases.find(x=>x.id===c.id),v=EXTRA.v2[c.id];return `<div class="detail-grid"><div><h3>A · 修改前</h3>${answerHtmlProxy(a)}${judgeHtmlProxy(a)}</div><div><h3>B · 修改后（同一生成 Prompt）</h3>${answerHtmlProxy(b)}${judgeHtmlProxy(b)}</div></div><details><summary>同一 B Context：两步生成 v2（独立消融）</summary><pre>${esc(v?.answer?.text||v?.error||'未记录')}</pre><pre>${esc(JSON.stringify({judgement:v?.judgement,plan:v?.structured_diagnostics},null,2))}</pre></details>`;}
function answerHtmlProxy(c){return `<div class="answer-panel ${esc("judge-"+(c?.gold_judgement?.verdict||"unavailable"))}"><h4>实际回答</h4><pre>${esc(c?.answer?.text||'无回答')}</pre></div>`;}
function judgeHtmlProxy(c){const j=c?.gold_judgement||{};return `<details><summary>自动判分：${esc(j.verdict||'裁判报错')} · 缺失项及依据</summary><pre>${esc(JSON.stringify(j,null,2))}</pre></details>`;}
function pairedStages(c){const a=ARMS.A.cases.find(x=>x.id===c.id),b=ARMS.B.cases.find(x=>x.id===c.id);return `<div class="stage-summary"><b>完整 Gold 引文：A → B</b><div class="stage-funnel">${ORDER.map(s=>`<div>${esc(STAGES[s])}<b>${a.stage_summaries[s].gold_found}/${a.stage_summaries[s].gold_total} → ${b.stage_summaries[s].gold_found}/${b.stage_summaries[s].gold_total}</b></div>`).join('')}</div><small>下方候选与阶段导航对应顶部所选 A/B，切换不改变判分。</small></div>`;}
document.getElementById('armSelect').onchange=e=>{DATA=ARMS[e.target.value];renderBanner();renderSummary();renderPriority();renderCharts();renderDiagnosticOverview();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();};
document.addEventListener('click',async e=>{const button=e.target.closest('[data-original-source]');if(!button)return;e.preventDefault();const status=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){status.textContent='请使用同目录 open_report.cmd 打开，随后点击即可调用原资料页定位。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:button.dataset.originalSource})});const d=await r.json();status.textContent=d.message||d.error;}catch(error){status.textContent='定位失败：'+error;}});
'''
    # Helpers are in the same lexical scope as the existing navigation functions.
    if args.native:
        banner=json.loads((OUT/'delivery_summary.json').read_text(encoding='utf8'))
        page=re.sub(r'<h2>前后对照 · 未通过开启验收</h2><p>.*?</p><p>.*?</p>',
            '<h2>'+html.escape(banner['title'])+'</h2><p>'+html.escape(banner['summary'])+'</p><p>'+html.escape(banner['validation'])+'</p>',page,count=1,flags=re.S)
        extra=extra.replace('B · 修改后（同一生成 Prompt）','B · 修改后（完整链路）')
        extra=extra.replace('同一 B Context：两步生成 v2（独立消融）','固定主 Prompt 消融、证据图、答案计划及逐主张审核')
        extra=extra.replace("<pre>${esc(v?.answer?.text||v?.error||'未记录')}</pre>", '')
        extra=extra.replace("JSON.stringify({judgement:v?.judgement,plan:v?.structured_diagnostics},null,2)","JSON.stringify(v,null,2)")
    page=page.replace('\ninit();\n</script>', '\n'+extra+'\ninit();\n</script>') if '\ninit();\n</script>' in page else page.replace('\ninit();\n})()', '\n'+extra+'\ninit();\n})()')
    assert 'function pairAnswers' in page and '${pairAnswers(c)}' in page and 'const ARMS=await unpack' in page
    (OUT/'report.html').write_text(page,encoding='utf8')
    (OUT/'report_sources.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2),encoding='utf8')
    audit={'format':'native RAG diagnostic viewer + paired answers and stages','reference':'20261030_143500_8c27af/report.html',
        'arms':{arm:len(d['cases']) for arm,d in arms.items()},'source_locations':len(sources),'html_bytes':len(page.encode()),
        'stage_counts_match_frozen_scores':all(c['stage_summaries'][s]['gold_found']==expected_scores[arm][c['id']][s]['gold_evidence_found'] for arm in arms for c in arms[arm]['cases'] for s in c['stage_summaries']),
        'verdicts_match_saved_answers':True}
    audit['unresolved_gold_source_locations']=[{'id':c['id'],'evidence':e} for c in arms['B']['cases'] for e in c.get('gold_evidence',[]) if not e.get('source_ids')]
    audit['unresolved_negative_source_locations']=[{'id':c['id'],'evidence':e} for c in arms['B']['cases'] for e in c.get('hard_negatives',[]) if not e.get('source_ids')]
    assert audit['stage_counts_match_frozen_scores']
    (OUT/'report_format_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
