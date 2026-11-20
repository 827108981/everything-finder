"""Render measured vector arms using the application's native report template."""
from dataclasses import asdict
from functools import lru_cache
import gc
import html
import json
from pathlib import Path

from tools.validate_vector_packaging_20261116 import OUT, FIXTURE, DB, inputs, load_cases, sha
from tools.diagnose_vector_recall import ro
from tools.report_vector_packaging_20261116 import SourceRegistry
from tools.report_vector_all_stages_20261117 import DEST, ARMS, STAGES, NAMES
from tools.render_general_native_report import packed
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import GoldDataset
from local_full_text_search.rag.evaluation.reporting import service


def main():
    _, bundle, _ = inputs()
    cases = tuple(load_cases(True))
    dataset = GoldDataset(FIXTURE, sha(FIXTURE), cases)
    with ro(DB) as connection:
        chunkrows = load_chunk_rows(connection)
    registry = SourceRegistry(chunkrows)
    # Keep existing identifiers valid for the companion comparison.json.
    registry.sources.update(json.loads((DEST/'report_sources.json').read_text(encoding='utf8')))
    mappings = {c.case_id:[asdict(m) for m in map_case_evidence(c,chunkrows)] for c in cases}
    original = service.match_quote

    @lru_cache(maxsize=40000)
    def cached(raw,evidence,kind,evidence_id,config):
        return original(json.loads(raw),json.loads(evidence),kind=kind,evidence_id=evidence_id,config=config)

    def match(chunk,evidence,**kwargs):
        obj={k:chunk[k] for k in ('text','filename','file_path','page','page_number','slide_number','location','position','sources') if k in chunk}
        return cached(json.dumps(obj,ensure_ascii=False,sort_keys=True),json.dumps(evidence,ensure_ascii=False,sort_keys=True),**kwargs)

    service.match_quote = match
    payload={}
    try:
        for arm in ARMS:
            results=[]
            with (OUT/f'{arm}_native.jsonl').open(encoding='utf8') as source:
                for line in source:
                    row=json.loads(line)
                    results.append({**row,'stages':row['scores'],'gold_mappings':mappings[row['id']],
                        'events':[dict(stage='query_interpreted',query_plan=row.get('query_plan',{}))],
                        'answer':{},'gold_judgement':{},'actual_behavior':'not_run'})
            assert len(results)==len(cases)
            detail=service.build_report_detail(dataset,dict(run_id='vector_all_stages_'+arm,ai_bundle_id=bundle.bundle_id,
                mode='API前各方案各阶段对照，答案未运行'),results,service.ReportConfig(default_stage='context',page_tolerance=0))
            for case in detail['cases']:
                for evidence in case.get('gold_evidence',[])+case.get('hard_negatives',[]):
                    evidence['source_ids']=registry.register(evidence)
                    for alternate in evidence.get('alternative_sources',[]):
                        evidence['source_ids'].extend(registry.register(alternate))
                for stage in case['stage_data'].values():
                    for item in stage['candidates']:
                        item['source_ids']=registry.register(item)
            payload[arm]=packed(detail)
            del detail, results
            gc.collect()
            print('RENDER',arm,flush=True)
    finally:
        service.match_quote=original
    assert not registry.misses, registry.misses[:10]
    comparison=json.loads((DEST/'comparison.json').read_text(encoding='utf8'))
    compact={'cases':[{'id':c['id'],'question':c['question'],'arms':c['arms'],
                       'gold_ids':[e['id'] for e in c['gold']]} for c in comparison['cases']],
             'arms':ARMS,'stages':STAGES,'names':NAMES}
    for case in compact['cases']:
        for arm in ARMS:
            for stage in STAGES:
                case['arms'][arm][stage].pop('matches',None)
    banner='''<section class="panel"><h2>各方案 · 各阶段金标排名</h2>
    <p>111题 × 9方案；8个LLM前阶段。原始Gold、索引版本、模型及其他阶段设置保持一致；API调用0。新增20题为pending，只作诊断。</p>
    <p>选择方案后，可在下方原程序诊断视图中查看每题、每阶段候选和原资料。横向对照中，数字为“首条完整金标排名 / 候选数；完整引文数 / 本题引文数”；多条引文可展开查看各自排名。没有正向金标的题不适用。</p>
    <div class="toolbar"><label>当前方案 <select id="armSelect">'''+''.join(f'<option value="{a}">{html.escape(a)}</option>' for a in ARMS)+'''</select></label>
    <label>横向比较题目 <select id="compareCase"></select></label></div>
    <div class="heatmap" id="comparisonTable"></div><p id="sourceStatus">双击同目录 open_report.cmd 后，可点击证据按钮打开原资料对应位置。</p></section>'''
    page=service.render_html({})
    replacement='<script>\n(async()=>{\n'+r'''async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}
'''+ 'const PACKED='+json.dumps(payload)+';const ARMS={};async function getArm(a){if(!ARMS[a])ARMS[a]=await unpack(PACKED[a]);return ARMS[a];}\nlet activeArm="W0";let DATA=await getArm(activeArm);\n'
    page=page.replace('<script>\nconst DATA={};',replacement,1).replace('</script></body>','})().catch(e=>{document.getElementById("banner").textContent="报告加载失败："+e;});\n</script></body>',1)
    page=page.replace('<main>','<main>'+banner,1).replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}').replace('<pre>${content}</pre>','${sourceLinks(c)}<pre>${content}</pre>')
    extra='const COMPARE='+json.dumps(compact,ensure_ascii=False).replace('</','<\\/')+';\n'+r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
function renderComparison(){let c=COMPARE.cases.find(x=>x.id===document.getElementById('compareCase').value);if(!c)return;document.getElementById('comparisonTable').innerHTML='<table><thead><tr><th>方案</th>'+COMPARE.names.map(esc).map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>'+COMPARE.arms.map(a=>'<tr><td>'+esc(a)+'</td>'+COMPARE.stages.map(s=>{let x=c.arms[a][s],total=x.gold_evidence_total_declared,rank=x.first_gold_rank;let cls=!total?'':!rank?'bad':x.gold_evidence_found<total?'warn':'good';let label=!total?'不适用':(rank||'未命中')+'/'+x.candidate_count+' · '+x.gold_evidence_found+'/'+total;let parts=c.gold_ids.map((id,i)=>'金标'+(i+1)+': '+(x.gold_completion_ranks[id]||'未命中')).join('<br>');return `<td><details><summary class="heat ${cls}">${esc(label)}</summary>${parts||'不适用'}</details></td>`}).join('')+'</tr>').join('')+'</tbody></table>';}
document.getElementById('compareCase').innerHTML=COMPARE.cases.map(c=>`<option value="${esc(c.id)}">${esc(c.id+' · '+c.question)}</option>`).join('');document.getElementById('compareCase').onchange=renderComparison;renderComparison();
document.getElementById('armSelect').onchange=async e=>{activeArm=e.target.value;DATA=await getArm(activeArm);renderBanner();renderSummary();renderPriority();renderCharts();renderDiagnosticOverview();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();};
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const s=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){s.textContent='请使用同目录 open_report.cmd 打开，再定位原资料。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();s.textContent=d.message||d.error;}catch(error){s.textContent='定位失败：'+error;}});
'''
    page=page.replace('\ninit();\n',extra+'\ninit();\n',1)
    (DEST/'report.html').write_text(page,encoding='utf8')
    (DEST/'report_sources.json').write_text(json.dumps(registry.sources,ensure_ascii=False,indent=2),encoding='utf8')
    print('REPORT',DEST/'report.html','sources',len(registry.sources),flush=True)


if __name__=='__main__':main()
