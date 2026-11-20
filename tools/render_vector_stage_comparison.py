"""Publish the frozen five-arm vector-only comparison in the native RAG format."""
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile
from tools.compare_vector_stage_20261116 import OUT,LABELS
from tools.validate_selection_factorial import ROOT,write,sha
from tools.render_general_native_report import packed
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.evaluation.gold_loader import GoldDataset,load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence,load_chunk_rows
from local_full_text_search.rag.evaluation.reporting import service


def main():
    manifestpath=ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json'
    manifest=json.loads(manifestpath.read_text(encoding='utf8'))
    cases=tuple(c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases)
    dataset=GoldDataset(manifestpath,sha(manifestpath),cases)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    diffs=json.loads((OUT/'differences.json').read_text(encoding='utf8'))
    integrity=json.loads((OUT/'execution_audit.json').read_text(encoding='utf8'))
    filteraudit=json.loads((OUT/'filter_metadata_audit.json').read_text(encoding='utf8'))
    diagnostics=json.loads((OUT/'case_diagnostics.json').read_text(encoding='utf8'))
    controls=json.loads((OUT/'representation_controls.json').read_text(encoding='utf8'))
    arms={a:[json.loads(x) for x in (OUT/(a+'.jsonl')).read_text(encoding='utf8').splitlines()] for a in LABELS}
    with ro('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db') as c:chunkrows=load_chunk_rows(c)
    mappings={c.case_id:[asdict(m) for m in map_case_evidence(c,chunkrows)] for c in cases}
    rowsbyid={r['id']:r for r in arms['baseline']}
    src=json.loads((ROOT/'reports/rag_bge_base_adoption_20261116/report_sources.json').read_text(encoding='utf8'))
    sourcemisses=[];used={};sourceindex={}
    for key,s in src.items():
        loc=s['location'];sourceindex[(s['path'].replace('\\','/').casefold(),str(loc.get('page_number') or ''),str(loc.get('slide_number') or ''))]=key
    def register(record):
        raw=record.get('file_path') or record.get('file') or ''
        if not raw:return []
        loc=record.get('location') or record.get('location_text') or record.get('position') or ''
        page=record.get('page_number') or record.get('page');slide=record.get('slide_number')
        if Path(raw).suffix.lower() in {'.ppt','.pptx'}:slide,page=slide or page,None
        if not page and not slide:
            m=re.search(r'第\s*(\d+)\s*(页|张)',loc)
            if m:
                if m[2]=='张':slide=int(m[1])
                else:page=int(m[1])
        norm=raw.replace('\\','/').casefold()
        key=sourceindex.get((norm,str(page or ''),str(slide or '')))
        if key:used[key]=src[key];return [key]
        if ' > ' in raw:
            archive,member=raw.split(' > ',1)
            if not Path(archive).is_file():sourcemisses.append(record);return []
            with ZipFile(archive) as z:
                options=[]
                for info in z.infolist():
                    variants=[info.filename]
                    if not info.flag_bits&0x800:
                        try:variants.append(info.filename.encode('cp437').decode('gbk'))
                        except (UnicodeError,LookupError):pass
                    if any(s.replace('\\','/')==member.replace('\\','/') for s in variants):options.append(info.filename)
                if len(options)!=1:sourcemisses.append(record);return []
            source=dict(path=raw,archive=str(Path(archive)),member=options[0])
        else:
            path=Path(raw)
            if not path.is_file():
                candidate=Path('E:/免疫资料')/raw
                paths={r['file_path'] for r in chunkrows if Path(r['file_path']).name==Path(raw).name}
                if candidate.is_file():path=candidate
                elif len(paths)==1 and Path(next(iter(paths))).is_file():path=Path(next(iter(paths)))
                else:sourcemisses.append(record);return []
            source=dict(path=str(path.resolve()))
        source['location']=dict(location_text=loc,page_number=int(page) if page else None,slide_number=int(slide) if slide else None)
        key=hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]
        used[key]=source
        return [key]
    def links(record):return ''.join(f'<button type="button" data-original-source="{k}">打开原资料对应页</button>' for k in register(record))
    def pre(x):return '<pre>'+html.escape(x if isinstance(x,str) else json.dumps(x,ensure_ascii=False,indent=2))+'</pre>'
    # Render-only cache omits rank/score fields that the quote matcher never reads.
    original=service.match_quote
    @lru_cache(maxsize=30000)
    def cached_match(raw,evidence,kind,evidence_id,config):return original(json.loads(raw),json.loads(evidence),kind=kind,evidence_id=evidence_id,config=config)
    def match(chunk,evidence,**kwargs):
        selected={k:chunk[k] for k in ('text','filename','file_path','page','page_number','slide_number','location','position','sources') if k in chunk}
        return cached_match(json.dumps(selected,ensure_ascii=False,sort_keys=True),json.dumps(evidence,ensure_ascii=False,sort_keys=True),**kwargs)
    service.match_quote=match
    payload={}
    try:
        for arm,records in arms.items():
            results=[{**r,'stages':r['scores'],'gold_mappings':mappings[r['id']],
                'events':[dict(stage='query_interpreted',query_plan=r['query_plan'])],
                'answer':{},'gold_judgement':{},'actual_behavior':'not_run'} for r in records]
            detail=service.build_report_detail(dataset,dict(run_id='vector_stage_'+arm,ai_bundle_id=manifest['bundle_id'],
                mode=LABELS[arm]+'；本轮只比较向量阶段；其他阶段和答案未执行'),results,
                service.ReportConfig(default_stage='vector',page_tolerance=0))
            for c in detail['cases']:
                expected=next(r for r in records if r['id']==c['id'])['scores']['vector']
                assert c['stage_summaries']['vector']['gold_found']==expected['gold_evidence_found']
                assert c['stage_summaries']['vector']['first_gold_rank']==expected['first_gold_rank']
                for e in c.get('gold_evidence',[])+c.get('hard_negatives',[]):
                    e['source_ids']=register(e)
                    for a in e.get('alternative_sources',[]):e['source_ids'].extend(register(a))
                for stage in c['stage_data'].values():
                    for item in stage['candidates']:item['source_ids']=register(item)
            payload[arm]=packed(detail)
            print('Prepared native vector report',arm,flush=True)
    finally:service.match_quote=original
    local=defaultdict(list)
    for r in controls['results']:local[(r['case'],r['parent'])].append(r)
    local_summary=[]
    for (cid,parent),reps in local.items():
        best={kind:min((r for r in reps if r['kind']==kind),key=lambda x:x['replacement_rank']) for kind in {r['kind'] for r in reps}}
        local_summary.append(dict(case=cid,parent=parent,file=reps[0]['file'],page=reps[0]['page'],baseline_rank=reps[0]['baseline_rank'],
            controls={k:{x:r[x] for x in ['replacement_rank','token_count','truncated','text']} for k,r in best.items()}))
    write(OUT/'representation_summary.json',local_summary)
    mrows=''.join(f'<tr><td>{LABELS[a]}</td><td>{metrics[a]["formal"]["complete"]}/46</td><td>{metrics[a]["diagnostic"]["complete"]}/40</td><td>{metrics[a]["formal"]["mrr"]:.4f}</td><td>{metrics[a]["diagnostic"]["mrr"]:.4f}</td><td>{html.escape("、".join(diffs[a]["losses"]) or "无")}</td></tr>' for a in LABELS)
    missing=[]
    for d in diagnostics:
        r=rowsbyid[d['id']]
        if r['scores']['vector']['gold_evidence_total_declared'] and r['scores']['vector']['gold_evidence_recall']<1:
            missing.append(dict(id=d['id'],question=r['question'],review_status=r['review_status'],targets=d['target_ranks']))
    localhtml=''
    for r in local_summary:
        localhtml+=f'<details><summary>{html.escape(r["case"])} · {html.escape(Path(r["file"]).name)} · 第{r["page"]}页/张 · 原名次{r["baseline_rank"]}</summary>'+links(r)
        for kind,v in r['controls'].items():
            localhtml+=f'<h4>{html.escape(kind)}：局部替换名次{v["replacement_rank"]}，{v["token_count"]}token，截断{v["truncated"]}</h4>'+pre(v['text'])
        localhtml+='</details>'
    missinghtml=''.join('<details><summary>'+html.escape(r['id']+'：'+r['question'])+'</summary>'+pre(r['targets'])+'</details>' for r in missing)
    banner=f'''<section class="panel" id="vectorAudit"><h2>整个向量召回阶段：91题 × 5组对照</h2>
    <p>相同新版Gold v5.1、相同发布AI索引、相同本地BGE模型；原始问题未改。API调用0，生产算法与默认设置未改。</p>
    <p><b>结论：</b>当前已审核有正向证据46题全部覆盖；诊断40题中37题完整，3题缺失。精确搜索、扩大搜索深度均没有补回；不能把所有排名靠后都当成向量漏召回。</p>
    <p>下面五组只运行向量阶段。新组没有运行融合、重排、Context或答案；这些列显示未执行是本轮范围，不是系统故障。</p>
    <p>A↔B检查近似搜索；A↔C检查搜索深度；B↔D检查查询编码指令；B↔E检查父块汇总。D/E是诊断消融，不能因为个别题变好就直接采用。</p>
    <table><tr><th>方案</th><th>正式完整Gold@50</th><th>诊断完整Gold@50</th><th>正式MRR</th><th>诊断MRR</th><th>比A新增引文丢失题</th></tr>{mrows}</table>
    <p>46和40只包含有正向证据的问题；诊断40题包括39 pending和1 disputed。全部50 reviewed、40 pending、1 disputed合计91题，pending/disputed只作诊断。覆盖按向量召回返回的父块正文、来源和原页判定，不代表每条Gold都在参与编码的最佳小窗口里。Gold覆盖不是答案正确率，也不是整个语料所有相关资料的召回率。</p>
    <label>选择方案：<select id="armSelect">{''.join(f'<option value="{a}">{html.escape(label)}</option>' for a,label in LABELS.items())}</select></label>
    <p id="sourceStatus">通过同目录open_report.cmd打开，点击按钮可定位原资料页。</p>
    <p><a href="conclusion.md">详细结论和下一步</a> · <a href="final_acceptance.json">执行验收记录</a></p>
    <h3>仍然真正缺失的3题</h3>{missinghtml}
    <h3>核查了哪些其他向量环节</h3><p>实际Chroma的52282条向量与缓存逐条相符；父块/子块及过滤元数据逐条核对通过；32条重新编码抽查通过。91题查询38–77token，窗口24–192token，未发现512上限截断。QueryPlan送入向量的文字91/91等于原问题。</p>
    <p>ANN与全库精确Top50有少量成员差异，但未改变任何题的Gold覆盖。扩大搜索深度同样无新增覆盖。查询指令移除导致006、row_holdout2-007丢引文；前两窗均分导致006、050丢引文，均不建议采用。</p>
    <h3>资料表示的单因素核查：15题、192个局部控制</h3><p>局部控制只替换一个被核查来源的表示，其余全库竞争向量固定；不是完整索引重建，也不是独立验收收益。整父块超过512token的控制已标明截断。原表第一行结构仅用于诊断，未作为生产表头识别器上线。</p>
    <p>新发现行级表示有明显影响的已有题：row-006 ISE速度11→3，row-009样本针清洗50→1，row_holdout2-003反应体积19→1，row_holdout2-009测量波长14→1。它们原来均在Top50；这4题来源集中同一套BS-5000资料，不能算独立泛化验证。</p>
    <p>independent-05原模块表仅去掉文件名/位置包装，局部352→1，表格整行表示反而955；原理说明页仍在Top50之外。这说明除了表头拆散，还有编码文字的来源包装及问题与资料语义匹配问题。</p>
    <details id="localControls"><summary>展开各来源原名次、新表示与局部对照</summary>{localhtml}</details>
    <h3>下一步边界</h3><p>继续处理表格对象绑定；将来源包装列为独立实验因素，不能全局删除。006连续说明和ISE原理页仍需针对表示/语义匹配做因果验证。019、IMFH1、条码的已知最终丢失发生在向量之后，不作为本轮向量缺失。</p>
    </section>'''
    page=service.render_html({})
    assert '<script>\nconst DATA={};' in page
    replacement='<script>\n(async()=>{\n'+r'''async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}
'''+ 'const PACKED='+json.dumps(payload)+';const ARMS={};async function getArm(a){if(!ARMS[a])ARMS[a]=await unpack(PACKED[a]);return ARMS[a];}\nlet activeArm="baseline";let DATA=await getArm(activeArm);\n'
    page=page.replace('<script>\nconst DATA={};',replacement,1)
    page=page.replace('</script></body>','})().catch(e=>{document.getElementById("banner").textContent="报告加载失败："+e;});\n</script></body>',1)
    page=page.replace('<main>','<main>'+banner,1)
    page=page.replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page=page.replace('<pre>${content}</pre>','${sourceLinks(c)}<pre>${content}</pre>')
    extra=r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
document.getElementById('armSelect').onchange=async e=>{activeArm=e.target.value;DATA=await getArm(activeArm);renderBanner();renderSummary();renderPriority();renderCharts();renderDiagnosticOverview();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();};
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const s=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){s.textContent='请使用同目录open_report.cmd打开，再定位原资料。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();s.textContent=d.message||d.error;}catch(error){s.textContent='定位失败：'+error;}});
'''
    page=page.replace('\ninit();\n',extra+'\ninit();\n',1)
    (OUT/'report.html').write_text(page,encoding='utf8')
    write(OUT/'report_sources.json',used)
    write(OUT/'report_format_audit.json',dict(arms=list(LABELS),cases=91,stages='vector only',source_misses=sourcemisses,
        quote_match_cache=cached_match.cache_info()._asdict()))
    assert not sourcemisses,'Missing report source paths'
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n',encoding='utf8')
    write(OUT/'missing_cases.json',missing)
    print('Native vector report written',len(used),'sources',flush=True)


if __name__=='__main__':main()
