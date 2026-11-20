"""Native report with explicit completion state, comparisons and source buttons."""
from __future__ import annotations
from dataclasses import asdict
from functools import lru_cache
import hashlib
import html
import json
from pathlib import Path
import re
from zipfile import ZipFile
from tools.validate_vector_packaging_20261116 import *
from tools.render_general_native_report import packed
from local_full_text_search.rag.evaluation.gold_loader import GoldDataset
from local_full_text_search.rag.evaluation.reporting import service

class SourceRegistry:
    def __init__(self,chunkrows):
        self.rows=chunkrows;self.sources={};self.misses=[];self.known={}
        previous=BASE/'report_sources.json'
        if previous.exists():
            for key,s in json.loads(previous.read_text(encoding='utf8')).items():
                loc=s['location'];self.known[(s['path'].replace('\\','/').casefold(),str(loc.get('page_number') or ''),str(loc.get('slide_number') or ''))]=s
    def register(self,r):
        raw=r.get('file_path') or r.get('file') or ''
        if not raw:return []
        page=r.get('page_number') or r.get('page');slide=r.get('slide_number')
        loc=r.get('location') or r.get('position') or ''
        if Path(raw).suffix.lower() in {'.ppt','.pptx'}:slide,page=slide or page,None
        if not page and not slide:
            m=re.search(r'第\s*(\d+)\s*(页|张)',loc)
            if m:
                if m[2]=='张':slide=int(m[1])
                else:page=int(m[1])
        key=hashlib.sha256(json.dumps([raw.replace('\\','/').casefold(),page,slide,loc],ensure_ascii=False).encode()).hexdigest()[:24]
        if key in self.sources:return [key]
        prior=self.known.get((raw.replace('\\','/').casefold(),str(page or ''),str(slide or '')))
        if prior:
            source=dict(prior)
        elif ' > ' in raw:
            archive,member=raw.split(' > ',1)
            if not Path(archive).is_file():self.misses.append(r);return []
            with ZipFile(archive) as z:
                members=[]
                for info in z.infolist():
                    names=[info.filename]
                    if not info.flag_bits&0x800:
                        try:names.append(info.filename.encode('cp437').decode('gbk'))
                        except (UnicodeError,LookupError):pass
                    if any(n.replace('\\','/')==member.replace('\\','/') for n in names):members.append(info.filename)
                if len(members)!=1:self.misses.append(r);return []
            source=dict(path=raw,archive=archive,member=members[0])
        else:
            path=Path(raw)
            if not path.is_file():
                matches={x['file_path'] for x in self.rows if Path(x['file_path']).name==path.name and Path(x['file_path']).is_file()}
                if len(matches)!=1:self.misses.append(r);return []
                path=Path(next(iter(matches)))
            source=dict(path=str(path.resolve()))
        source['location']=dict(page_number=page,slide_number=slide,location_text=loc)
        self.sources[key]=source
        return [key]

def main():
    cases=tuple(load_cases(True));byid={c.case_id:c for c in cases}
    dataset=GoldDataset(FIXTURE,sha(FIXTURE),cases)
    with ro(DB) as c:chunkrows=load_chunk_rows(c)
    mappings={c.case_id:[asdict(m) for m in map_case_evidence(c,chunkrows)] for c in cases}
    registry=SourceRegistry(chunkrows)
    arms={}
    # Final native runs take precedence over vector-only controls with same name.
    for suffix in ('_exact.jsonl','_ann.jsonl','_exact_holdout.jsonl','_ann_holdout.jsonl','_native.jsonl'):
        for path in sorted(OUT.glob('*'+suffix)):
            name=path.name[:-6]
            arms[name]=lines(path)
    if not arms:
        prior=lines(BASE/'baseline.jsonl')
        assert all(byid[r['id']].question==r['question'] for r in prior)
        arms['W0_frozen_previous']=prior
    original=service.match_quote
    @lru_cache(maxsize=40000)
    def cached(raw,evidence,kind,evidence_id,config):return original(json.loads(raw),json.loads(evidence),kind=kind,evidence_id=evidence_id,config=config)
    def match(chunk,evidence,**kwargs):
        obj={k:chunk[k] for k in ('text','filename','file_path','page','page_number','slide_number','location','position','sources') if k in chunk}
        return cached(json.dumps(obj,ensure_ascii=False,sort_keys=True),json.dumps(evidence,ensure_ascii=False,sort_keys=True),**kwargs)
    service.match_quote=match
    payload={};summaries={}
    try:
        for arm,records in arms.items():
            existing={r['id']:r for r in records};results=[]
            for case in cases:
                r=existing.get(case.case_id,dict(id=case.case_id,question=case.question,candidate_stages={},scores={}))
                results.append({**r,'stages':r['scores'],'gold_mappings':mappings[case.case_id],
                    'events':[dict(stage='query_interpreted',query_plan=r.get('query_plan',{}))],
                    'answer':{},'gold_judgement':{},'actual_behavior':'not_run'})
            detail=service.build_report_detail(dataset,dict(run_id='vector_packaging_'+arm,ai_bundle_id=read('protocol.json')['bundle'],
                mode='API前资料表示对照；答案未执行'),results,service.ReportConfig(default_stage='context' if arm.endswith('_native') else 'vector',page_tolerance=0))
            for case in detail['cases']:
                for e in case.get('gold_evidence',[])+case.get('hard_negatives',[]):
                    e['source_ids']=registry.register(e)
                    for a in e.get('alternative_sources',[]):e['source_ids'].extend(registry.register(a))
                for stage in case['stage_data'].values():
                    for item in stage['candidates']:item['source_ids']=registry.register(item)
            payload[arm]=packed(detail)
            if 'vector' in records[0].get('scores',{}):summaries[arm]=metrics([r for r in records if 'vector' in r.get('scores',{})])
            print('RENDER',arm,len(records),flush=True)
    finally:service.match_quote=original
    status=read('execution_status.json') if (OUT/'execution_status.json').exists() else dict(status='in_progress',current='full_corpus_encoding')
    finished=status.get('status')=='complete'
    evidenceaudit=read('holdout_evidence_audit.json')
    registry.register(evidenceaudit['cases'][0])
    audit_html=''.join('<details><summary>'+html.escape(r['id']+' · '+Path(r['file']).name+' · '+str(r['page'] or r['position']))+'</summary>'+
        ''.join(f'<button data-original-source="{k}">打开原资料对应页</button>' for k in registry.register(r))+
        '<pre>'+html.escape(r['quote'])+'</pre></details>' for r in evidenceaudit['cases'])
    rows=''
    for arm,m in summaries.items():
        rows+='<tr><td>'+html.escape(arm)+'</td>'+''.join(f'<td>{m[g]["complete"]}/{m[g]["positive_cases"]}</td><td>{m[g]["mrr"]:.4f}</td>' for g in ('formal','diagnostic','source_holdout'))+'</tr>'
    tests='未完成'
    if (OUT/'pytest.log').exists():tests=(OUT/'pytest.log').read_text(encoding='utf8').splitlines()[-1]
    conclusion=(OUT/'conclusion.md').read_text(encoding='utf8') if (OUT/'conclusion.md').exists() else '本轮尚未完成全库对照，不得推断新包装有效；默认与生产索引保持不变。'
    banner=f'''<section class="panel" id="packagingAudit"><h2>向量包装与结构化表示：前后对照</h2>
    <p><b>执行状态：{'已完成全部批次' if finished else '验证尚未全部完成，不能宣布采用'}。</b> 当前：{html.escape(str(status.get('current','')))}</p>
    <p>固定91题：50 reviewed / 40 pending / 1 disputed；新增20题、10个来源、3种格式，全部pending，仅诊断不计正式分。API调用0。默认Adapter关闭，未替换生产索引。</p>
    <p>兼容/单元测试（以实际日志为准）：{html.escape(tests)}。另有解析/切分测试记录parser_pytest.xml。旧52282窗口正文、跨度、编码及ID逐条一致。</p>
    <table><tr><th>方案</th><th>正式完整@50</th><th>正式MRR</th><th>原诊断完整@50</th><th>诊断MRR</th><th>新来源完整@50</th><th>新来源MRR</th></tr>{rows}</table>
    <p>未运行的集合为0/0，不代表0分。MRR/Hit以第一条完整Gold计算；完整覆盖须满足全部正向引文。返回父块命中与真正编码窗口覆盖分开保存；最终Context只核对actual included_text。</p>
    <label>选择方案：<select id="armSelect">{''.join(f'<option value="{a}">{html.escape(a)}</option>' for a in arms)}</select></label>
    <p id="sourceStatus">请用同目录open_report.cmd打开，按钮可定位原资料页。Word使用真实段落位置，不编造物理页码。</p>
    <h3>当前结论与限制</h3><pre>{html.escape(conclusion)}</pre>
    <details id="holdoutSources"><summary>20道待审题原文与索引核验 / 点击打开原资料</summary>{audit_html}</details>
    <p><a href="conclusion.md">结论全文</a> · <a href="final_acceptance.json">最终核验记录（完成后生成）</a></p></section>'''
    page=service.render_html({})
    replacement='<script>\n(async()=>{\n'+r'''async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}
'''+ 'const PACKED='+json.dumps(payload)+';const ARMS={};async function getArm(a){if(!ARMS[a])ARMS[a]=await unpack(PACKED[a]);return ARMS[a];}\nlet activeArm='+json.dumps(next(iter(arms)))+';let DATA=await getArm(activeArm);\n'
    page=page.replace('<script>\nconst DATA={};',replacement,1).replace('</script></body>','})().catch(e=>{document.getElementById("banner").textContent="报告加载失败："+e;});\n</script></body>',1)
    page=page.replace('<main>','<main>'+banner,1).replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}').replace('<pre>${content}</pre>','${sourceLinks(c)}<pre>${content}</pre>${encodingDetails(c)}')
    extra=r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
function encodingDetails(record){return (record.vector_unit_hits||[]).map(h=>`<details><summary>实际命中窗口与编码来源 · ${esc(h.kind||'')} · ${esc(h.unit_id||'')}</summary><p>窗口正文</p><pre>${esc(h.text||'')}</pre><p>实际编码文字（未保存时不推测）</p><pre>${esc(h.embedding_text||'本条旧索引轨迹未保存编码文字；精确对照JSONL与representations.db保留完整输入。')}</pre><p>背景与结构来源</p><pre>${esc(JSON.stringify(h.representation_metadata||h.representation||{},null,2))}</pre></details>`).join('');}
document.getElementById('armSelect').onchange=async e=>{activeArm=e.target.value;DATA=await getArm(activeArm);renderBanner();renderSummary();renderPriority();renderCharts();renderDiagnosticOverview();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();};
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const s=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){s.textContent='请使用同目录open_report.cmd打开，再定位原资料。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();s.textContent=d.message||d.error;}catch(error){s.textContent='定位失败：'+error;}});
'''
    page=page.replace('\ninit();\n',extra+'\ninit();\n',1)
    (OUT/'report.html').write_text(page,encoding='utf8')
    write(OUT/'report_sources.json',registry.sources)
    write(OUT/'report_format_audit.json',dict(arms=list(arms),cases=111,source_misses=registry.misses,complete=finished,api_calls=0))
    assert not registry.misses
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n',encoding='utf8')
    print('REPORT',OUT/'report.html',len(registry.sources),'sources',flush=True)

if __name__=='__main__':main()
