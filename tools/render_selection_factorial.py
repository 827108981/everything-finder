"""Native diagnostic report with eight selectable arms and paired actual Context."""
from copy import deepcopy
import base64
import gzip
import hashlib
import html
import json
from pathlib import Path
import re
from zipfile import ZipFile

from tools.validate_selection_factorial import ROOT, OUT, HISTORY, load_inputs, write
from tools.analyze_selection_factorial import read_arm
from tools.selection_factorial import ARMS
from tools.render_general_native_report import packed
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset, GoldDataset
from local_full_text_search.rag.evaluation.reporting.service import build_report_detail, render_html, ReportConfig
from local_full_text_search.rag.evaluation.reporting import service as report_service

LABELS={'baseline':'基线','A':'A · 取消单文档配额','B':'B · MMR冗余排序','C':'C · 正文预算选段',
        'AB':'A+B','AC':'A+C','BC':'B+C','ABC':'A+B+C'}


def main(*, output=OUT, arms=ARMS, labels=LABELS, reader=read_arm,
         mutable_stages=('evidence','context'), initial_arm='AC', banner_override=None):
    OUT, ARMS, LABELS, read_arm = output, arms, labels, reader
    # Reuse only fixed upstream presentation records from the verified A report.
    # New evidence/Context matching, all official scores and diagnoses are still
    # rebuilt below. Historical answers are never copied into the new report.
    historical_html=(HISTORY/'report.html').read_text(encoding='utf8')
    encoded=re.search(r"const ARMS=await unpack\('([^']+)'\)",historical_html)
    if encoded is None: raise ValueError('Historical native upstream report missing')
    frozen_details=json.loads(gzip.decompress(base64.b64decode(encoded[1])))['A']
    upstream={c['id']:c['stage_data'] for c in frozen_details['cases']}
    frozen_queries={c['id']:deepcopy(c.get('query',{})) for c in frozen_details['cases']}
    del historical_html,encoded,frozen_details
    original_stage_match=report_service._match_stage_candidates
    def cached_stage_match(case,stage,candidates,config,formal_mappings=None):
        if stage not in mutable_stages:
            cached=upstream[case.case_id][stage]['candidates']
            assert [x['chunk_id'] for x in cached]==[x['chunk_id'] for x in candidates[:config.topk]], 'upstream report cache drift'
            assert all(x.get('text')==y.get('text') for x,y in zip(cached,candidates)), 'upstream body drift'
            return deepcopy(cached),{}
        return original_stage_match(case,stage,candidates,config,formal_mappings)
    report_service._match_stage_candidates=cached_stage_match
    # Render-only memoization: all eight arms share frozen upstream candidates.
    # The original matcher still calculates each distinct result, unchanged.
    # Cache entire original arguments, so provenance or text differences cannot
    # be mistaken for identical evidence. No evaluation score is overwritten.
    original_match = report_service.match_quote
    match_cache = {}
    def cached_match(chunk, evidence, **kwargs):
        # match_quote reads body and source locations, never rank or score.
        selected = {k:chunk.get(k) for k in ('text','filename','file_path','page','page_number',
            'slide_number','location','position','sources')}
        key = json.dumps([selected,evidence,{k:str(v) for k,v in kwargs.items()}],ensure_ascii=False,sort_keys=True)
        if key not in match_cache: match_cache[key] = original_match(chunk,evidence,**kwargs)
        return match_cache[key]
    report_service.match_quote = cached_match
    originals, settings, mappings, _=load_inputs()
    originals={r['id']:r for r in originals}
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases=tuple(c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases)
    dataset=GoldDataset(ROOT/'tests/rag/fixtures/rag_active_suite_v5.json',hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),cases)
    sources=json.loads((HISTORY/'report_sources.json').read_text(encoding='utf8'))
    used=set(); source_misses=[]; location_cache={}
    source_index={(str(Path(s['path'])).casefold(),str(s['location'].get('page_number') or ''),str(s['location'].get('slide_number') or '')):key for key,s in sources.items()}
    def source_ids(record):
        raw=record.get('file_path') or record.get('file',''); loc=record.get('location') or record.get('position') or ''
        page=record.get('page_number') or record.get('page');slide=record.get('slide_number')
        if not page and not slide:
            m=re.search(r'第\s*(\d+)\s*(页|张)',loc)
            if m:
                if m[2]=='张':slide=int(m[1])
                else:page=int(m[1])
        if Path(raw).suffix.lower() in {'.ppt','.pptx'} and not slide:slide,page=page,None
        cache_key=(raw,loc,str(page or ''),str(slide or ''))
        if cache_key in location_cache:return location_cache[cache_key]
        if ' > ' in raw:
            archive,member=raw.split(' > ',1)
            if Path(archive).is_file():
                with ZipFile(archive) as z:
                    names=[]
                    for info in z.infolist():
                        variants=[info.filename]
                        if not info.flag_bits & 0x800:
                            try:variants.append(info.filename.encode('cp437').decode('gbk'))
                            except (UnicodeError,LookupError):pass
                        if any(v.replace('\\','/')==member.replace('\\','/') for v in variants):names.append(info.filename)
                    if len(names)!=1: raise ValueError('Ambiguous/missing archive source: '+raw)
                    member=names[0];info=z.getinfo(member)
                    assert not info.is_dir()
                location={'location_text':loc,'page_number':int(page) if page else None,'slide_number':int(slide) if slide else None}
                key=hashlib.sha256((raw+json.dumps(location,ensure_ascii=False)).encode()).hexdigest()[:24]
                sources[key]={'path':raw,'archive':str(Path(archive).resolve()),'member':member,'location':location}
                used.add(key);location_cache[cache_key]=[key];return [key]
        norm=str(Path(raw)).casefold()
        options={s['path'] for s in sources.values() if s['path'].replace('\\','/').casefold().endswith(norm.replace('\\','/'))} if raw else set()
        full=str(Path(raw).resolve()) if Path(raw).is_file() else next(iter(options)) if len(options)==1 else ''
        key=source_index.get((str(Path(full)).casefold(),str(page or ''),str(slide or ''))) if full else None
        if key:
            used.add(key);location_cache[cache_key]=[key];return [key]
        if not full or not Path(full).is_file():
            source_misses.append({'file':raw,'location':loc,'page':page,'slide':slide});location_cache[cache_key]=[];return []
        location={'location_text':loc,'page_number':int(page) if page else None,'slide_number':int(slide) if slide else None}
        key=hashlib.sha256((full+json.dumps(location,ensure_ascii=False)).encode()).hexdigest()[:24]
        sources[key]={'path':full,'location':location};used.add(key)
        source_index[(str(Path(full)).casefold(),str(page or ''),str(slide or ''))]=key
        location_cache[cache_key]=[key];return [key]
    payload={};pairs={};scores_valid=True; shared_upstream={}
    for arm in ARMS:
        rows=read_arm(arm); raw=[]
        for r in rows.values():
            original=originals[r['id']]
            stages={**original['candidate_stages'],'evidence':r['evidence']}
            if 'rerank' in mutable_stages:
                stages['rerank']=r['rerank_candidates']
            if 'constraints' in mutable_stages:
                stages['constraints']=r['condition_candidates']
            raw.append({k:r[k] for k in ('id','question','group_id','review_status','cohort') } | {
                'scores':r['scores'],'stages':r['scores'],'candidate_stages':stages,
                'gold_mappings':mappings[r['id']], 'events':[{'query_plan':r['query_plan'],'stage':'query_interpreted'},
                    {'stage':'context_built','context_entries':r['entries'],'llm_context':r['context']}],
                'answer':{},'gold_judgement':{},'actual_behavior':'not_run'})
        detail=build_report_detail(dataset,{'run_id':'selection_factorial_'+arm,'ai_bundle_id':manifest['bundle_id'],
                    'mode':'API前冻结候选重放；答案未运行'},raw,ReportConfig())
        for c in detail['cases']:
            r=rows[c['id']]
            c['query']=deepcopy(frozen_queries[c['id']])
            if 'constraints' in mutable_stages:
                c['query']['constraints']=deepcopy(r['query_plan']['constraints'])
            c['query'].setdefault('limits',{})['experimental_selection']={
                'arm':arm,'document_cap':None if ARMS[arm].remove_document_cap else 3,
                'total_cap':64 if ARMS[arm].budget_pack else 8,'max_chars':12000}
            scores_valid &= all(c['stage_summaries'][s]['gold_found']==r['scores'][s]['gold_evidence_found'] for s in c['stage_summaries'])
            for e in c.get('gold_evidence',[])+c.get('hard_negatives',[]):e['source_ids']=source_ids(e)
            for stage in c.get('stage_data',{}).values():
                for item in stage['candidates']:item['source_ids']=source_ids(item)
            state='正确引文丢失' if r['lost_gold'] else '恢复引文/需核对负例' if r['gained_gold'] and r['added_hard_negative'] else '恢复正确引文' if r['gained_gold'] else '新增负例正文' if r['added_hard_negative'] else '仅Context变化' if r['context_changed'] else '无变化'
            detail.setdefault('baseline',{}).setdefault('case_changes',{})[c['id']]={'state':state}
        if arm=='baseline':
            shared_upstream={c['id']:{stage:deepcopy(data) for stage,data in c['stage_data'].items()
                            if stage not in mutable_stages} for c in detail['cases']}
        else:
            # Share unchanged upstream bodies; retain each arm's own complete
            # rank history. The browser restores this losslessly on selection.
            for c in detail['cases']:
                c['_rank_histories']={}
                for stage,data in c['stage_data'].items():
                    if stage in mutable_stages:continue
                    base=shared_upstream[c['id']][stage]
                    assert len(base['candidates'])==len(data['candidates'])
                    for before,after in zip(base['candidates'],data['candidates']):
                        before_core={k:v for k,v in before.items() if k!='rank_history'}
                        after_core={k:v for k,v in after.items() if k!='rank_history'}
                        assert before_core==after_core, 'upstream presentation drift'
                        if before.get('rank_history')!=after.get('rank_history'):
                            c['_rank_histories'][after.get('content_group_id') or after['chunk_id']]=after['rank_history']
                    c['stage_data'][stage]={'shared_from_baseline':True}
        payload[arm]=packed(detail)
        pairs[arm]={cid:{k:r[k] for k in ('context','entries','gained_gold','lost_gold','added_hard_negative','removed_chunks','added_chunks','selection_trace','decision')} for cid,r in rows.items()}
        for rr in pairs[arm].values():
            for e in rr['entries']:
                e['source_ids']=list(dict.fromkeys(sid for src in e['sources'] for sid in source_ids(src)))
        del detail,raw
        print('Prepared native report',arm,flush=True)
    assert scores_valid
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    negative_reviews=json.loads((OUT/'negative_review.json').read_text(encoding='utf8'))
    rows_html=''.join(f'<tr><td>{LABELS[a]}</td><td>{m["formal"]["complete"]}/46</td><td>{m["diagnostic"]["complete"]}/40</td><td>{len(m["formal"]["loss_ids"])}</td><td>{m["formal"]["new_negative_quote_count"]}+{m["diagnostic"]["new_negative_quote_count"]}</td><td>{m["formal"]["mean_evidence_recall"]:.1%}</td></tr>' for a,m in metrics.items())
    banner=f'''<section class="panel"><h2>最终选段八组对照：暂不采用，不改变应用默认行为</h2>
    <p>91题 × 8组 = 728次离线选段。基线逐字复现91/91；API调用0。50题已审核，其中46题有正向引文；其余41题仅诊断，其中40题有正向引文。</p>
    <p>A取消单文档配额；B固定MMR（0.8相关性、0.2冗余）；C取消8段截断，按相同12000字符预算装入完整片段，安全上限64段。上游候选、条件判断和阈值不变。</p>
    <table><thead><tr><th>方案</th><th>正式完整证据</th><th>诊断完整证据</th><th>正式新增Gold丢失题</th><th>新增负例引文（正式+诊断）</th><th>正式平均引文召回</th></tr></thead><tbody>{rows_html}</tbody></table>
    <p>负例正文出现≠答案已经误用：CSDB同表DCF与TBIL现象背景单列核查；真正跨对象/范围内容见下方复核记录。MMR仅改变顺序，未改变入选集合。所有组合均未满足采用门槛。</p>
    <p>单元/兼容回归：459通过，10个既有失败，1跳过；新增失败0。Gold及运行默认配置未改。没有合格候选进入独立验收，不能宣称泛化已通过。</p>
    <label>选择对照方案：<select id="armSelect">{''.join(f'<option value="{a}"'+(' selected' if a=='AC' else '')+f'>{LABELS[a]}</option>' for a in ARMS)}</select></label>
    <p id="sourceStatus">通过同目录 open_report.cmd 打开后，点击按钮即可定位原资料页；直接打开HTML可离线查看。</p>
    <a href="conclusion.md">详细结论</a> · <a href="final_acceptance.json">验收记录</a>
    <details><summary>8条新增负例的逐条复核</summary><pre>{html.escape(json.dumps(negative_reviews,ensure_ascii=False,indent=2))}</pre></details></section>'''
    if banner_override is not None:
        banner=banner_override
    page=render_html({})
    page=page.replace('<script>\nconst DATA={};','<script>\n(async()=>{\n'+
        "async function unpack(s){const bytes=Uint8Array.from(atob(s),c=>c.charCodeAt(0));return JSON.parse(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).text());}\n"+
        'const PACKED='+json.dumps(payload)+';const ARMS={};async function getArm(a){if(!ARMS[a]){const d=await unpack(PACKED[a]);if(a!=="baseline"){const base=await getArm("baseline");for(const c of d.cases){const b=base.cases.find(x=>x.id===c.id);for(const [stage,data] of Object.entries(c.stage_data)){if(data.shared_from_baseline)c.stage_data[stage]={...b.stage_data[stage],candidates:b.stage_data[stage].candidates.map(x=>({...x,rank_history:c._rank_histories[x.content_group_id||x.chunk_id]||x.rank_history}))};}}}ARMS[a]=d;}return ARMS[a];}\n'+
        "let activeArm='AC';let DATA=await getArm(activeArm);await getArm('baseline');\nconst PAIRS=await unpack('"+packed(pairs)+"');\n")
    page=page.replace('</script></body>','})().catch(e=>{document.getElementById("banner").textContent="报告加载失败："+e;});\n</script></body>')
    page=page.replace('<main>','<main>'+banner,1)
    page=page.replace("let activeArm='AC';", "let activeArm="+json.dumps(initial_arm)+";")
    page=page.replace('<section class="panel"><h2>整体耗时与模块消耗</h2>','<section class="panel" hidden><h2>整体耗时与模块消耗</h2>')
    page=page.replace('${answerHtml(c)}${judgeHtml(c)}${caseTimingHtml(c)}','${pairedContext(c)}')
    page=page.replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page=page.replace('<div class="candidate-body"><div class="inline-caption">状态：','<div class="candidate-body">${sourceLinks(c)}<div class="inline-caption">状态：')
    extra=r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
function includedRows(r){return r.entries.map(e=>`<details><summary>${esc(e.source_id)} · ${esc(e.filename)} · ${esc(e.location)}${e.truncated?' · 已截断':''}</summary>${sourceLinks(e)}<pre>${esc(e.included_text)}</pre></details>`).join('')||'<p>没有实际Context</p>';}
function pairedContext(c){const a=PAIRS.baseline[c.id],b=PAIRS[activeArm][c.id];return `<h3>实际 Context 前后对照</h3><p>答案：本轮未运行（不调用API）；没有答案正确率或模型裁判评分。</p><div class="detail-grid"><div><h4>基线 · ${a.context.length}字符 / ${a.entries.length}段</h4>${includedRows(a)}<details><summary>完整原始Context</summary><pre>${esc(a.context)}</pre></details></div><div><h4>${esc(activeArm)} · ${b.context.length}字符 / ${b.entries.length}段</h4>${includedRows(b)}<details><summary>完整原始Context</summary><pre>${esc(b.context)}</pre></details></div></div><details><summary>本方案新增/丢失引文与候选排除理由</summary><pre>${esc(JSON.stringify({gained_gold:b.gained_gold,lost_gold:b.lost_gold,added_hard_negative:b.added_hard_negative,added_chunks:b.added_chunks,removed_chunks:b.removed_chunks,decision:b.decision,selection_trace:b.selection_trace},null,2))}</pre></details>`;}
document.getElementById('armSelect').onchange=async e=>{activeArm=e.target.value;DATA=await getArm(activeArm);renderBanner();renderSummary();renderPriority();renderCharts();renderDiagnosticOverview();renderHeatmap();renderCaseTable();renderDetails();renderExcluded();};
document.addEventListener('click',async e=>{const button=e.target.closest('[data-original-source]');if(!button)return;e.preventDefault();const status=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){status.textContent='请使用同目录 open_report.cmd 打开，再点击原资料按钮。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:button.dataset.originalSource})});const d=await r.json();status.textContent=d.message||d.error;}catch(error){status.textContent='定位失败：'+error;}});
'''
    page=page.replace('\ninit();\n</script>','\n'+extra+'\ninit();\n</script>') if '\ninit();\n</script>' in page else page.replace('\ninit();\n})()', '\n'+extra+'\ninit();\n})()')
    assert '${pairedContext(c)}' in page and 'function pairedContext' in page
    (OUT/'report.html').write_text(page,encoding='utf8')
    write(OUT/'report_sources.json',{k:sources[k] for k in sorted(used)})
    write(OUT/'report_format_audit.json',{'format':'native application viewer with selectable arms and paired actual Context',
        'arms':list(ARMS),'case_count':len(cases),'stage_counts_match_frozen_scores':scores_valid,
        'source_locations':len(used),'unresolved_source_locations':source_misses,'html_bytes':len(page.encode()),
        'old_answers_imported':False,'answer_status':'not_run_no_api'})
    cmd='@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n'
    (OUT/'open_report.cmd').write_text(cmd,encoding='utf8')
    print('Report bytes',len(page.encode()),'source locations',len(used),'unresolved',len(source_misses))


if __name__=='__main__':main()
