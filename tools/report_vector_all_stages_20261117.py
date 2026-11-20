"""Build a source-linked, per-case comparison of all measured retrieval stages."""
from collections import defaultdict
from dataclasses import asdict
import html
import json
from pathlib import Path

from tools.validate_vector_packaging_20261116 import OUT, ROOT, DB, inputs, load_cases
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows
from tools.report_vector_packaging_20261116 import SourceRegistry

ARMS = ('W0','W1','W2','W3','W4','W5','F10','F11','F10_count_matched')
STAGES = ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')
NAMES = ('关键词召回','向量召回','融合排序','去重','重排','业务条件检查','最终证据筛选','实际 Context')
DEST = OUT/'all_stages_20261117'


def main():
    inputs()
    cases = load_cases(True)
    with ro(DB) as connection:
        registry = SourceRegistry(load_chunk_rows(connection))
    result = {c.case_id: {'id':c.case_id,'question':c.question,'status':c.review_status,
        'expected_behavior':c.expected_behavior, 'gold':[], 'arms':{}} for c in cases}
    for case in cases:
        for i, evidence in enumerate(case.gold_evidence, 1):
            item = dict(evidence)
            item['id'] = f'{case.case_id}:positive:{i}'
            item['source_ids'] = registry.register(evidence)
            for alternate in evidence.get('alternative_sources', []):
                item['source_ids'].extend(registry.register(alternate))
            result[case.case_id]['gold'].append(item)
    summary = defaultdict(dict)
    for arm in ARMS:
        path = OUT/f'{arm}_native.jsonl'
        with path.open(encoding='utf8') as source:
            rows = 0
            for line in source:
                row = json.loads(line)
                case = result[row['id']]
                assert case['question'] == row['question'] and set(STAGES) <= set(row['scores'])
                compact = {}
                for stage in STAGES:
                    score = row['scores'][stage]
                    compact[stage] = {k:score[k] for k in ('candidate_count','gold_evidence_found',
                        'gold_evidence_total_declared','first_gold_rank','gold_completion_ranks',
                        'hard_negative_found')}
                    compact[stage]['matches'] = {}
                    candidates = row['candidate_stages'].get(stage, [])
                    if stage == 'evidence' and any('selected' in item for item in candidates):
                        candidates = [item for item in candidates if item.get('selected') is True]
                    for evidence_id, rank in score['gold_completion_ranks'].items():
                        if not rank or rank > len(candidates):
                            continue
                        candidate = candidates[rank-1]
                        compact[stage]['matches'][evidence_id] = {
                            'filename':candidate.get('filename',''),
                            'location':candidate.get('location',''),
                            'text':candidate.get('included_text',candidate.get('text','')),
                            'source_ids':registry.register(candidate)}
                case['arms'][arm] = compact
                case['cohort'] = row['cohort']
                rows += 1
        assert rows == len(cases), (arm, rows, len(cases))
    assert not registry.misses, registry.misses
    for arm in ARMS:
        for cohort in ('formal','diagnostic','source_holdout'):
            selected = [c for c in result.values() if c['cohort']==cohort and c['gold']]
            summary[arm][cohort] = {}
            for stage in STAGES:
                scores = [c['arms'][arm][stage] for c in selected]
                summary[arm][cohort][stage] = {
                    'cases':len(scores),
                    'complete':sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in scores),
                    'mrr':sum(1/s['first_gold_rank'] if s['first_gold_rank'] else 0 for s in scores)/max(1,len(scores))}
    DEST.mkdir(exist_ok=True)
    data = {'cases':list(result.values()),'summary':summary,'arms':ARMS,'stages':STAGES,'names':NAMES}
    (DEST/'comparison.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    (DEST/'report_sources.json').write_text(json.dumps(registry.sources,ensure_ascii=False,indent=2),encoding='utf8')
    embedded = json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    page = PAGE.replace('__DATA__',embedded)
    (DEST/'report.html').write_text(page,encoding='utf8')
    (DEST/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(DEST/'report.html')+'"\r\n',encoding='utf8')
    print('REPORT',DEST/'report.html','cases',len(cases),'arms',len(ARMS),'sources',len(registry.sources))


PAGE = '''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG 各方案各阶段金标排名对照</title>
<style>body{margin:0;background:#f3f5f8;color:#172b4d;font:14px system-ui,"Microsoft YaHei",sans-serif}header{background:#172539;color:white;padding:22px 30px}header h1{margin:0;font-size:22px}header p{margin:8px 0 0;color:#bdcce0}main{max-width:1600px;margin:auto;padding:20px}.panel{background:white;border:1px solid #d9e0ea;border-radius:6px;padding:18px;margin-bottom:16px}h2{font-size:18px;margin:0 0 12px}p{line-height:1.55}label{display:inline-block;margin:5px 16px 5px 0}select,input{font:inherit;padding:7px;border:1px solid #aab8ca;border-radius:4px}table{width:100%;border-collapse:collapse}th,td{text-align:left;border-bottom:1px solid #e3e8ef;padding:8px;vertical-align:top}th{background:#f4f7fb;white-space:nowrap}td button{border:0;background:transparent;color:#1254a6;cursor:pointer;text-align:left;font:inherit;padding:0}.complete{color:#087b45}.partial{color:#a55b00}.miss{color:#bd2d34}.na{color:#60738c}.tablewrap{overflow:auto}.detail{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f8fa;padding:12px;border-radius:4px}.source{color:#1254a6;background:white;border:1px solid #aab8ca;border-radius:4px;padding:4px 8px;cursor:pointer}small{color:#66768a}#message{color:#1254a6}</style></head><body><header><h1>RAG 各方案各阶段金标排名对照</h1><p>真实本地检索链路 · LLM之前 · 不调用生成或裁判 API</p></header><main>
<section class="panel"><h2>阅读方法</h2><p>每格显示“第一条完整金标的排名 / 当前阶段候选数；覆盖的引文数/本题总引文数”。点击格子可看每条金标排名和完成匹配的片段。没有正向金标的拒答题显示“不适用”。</p><p>一条引文若分布在多个片段，排名表示最后一个必要片段出现的位置，详情展示该位置的片段，不能单凭它判断整条引文。排名是该阶段的候选排名，不是准确率。“未命中”只表示当前候选列表中未匹配到。Context只核对实际 included_text。新增20题仍为pending，不计正式分。</p><p>W0–W5改变向量编码文本；F10保留W0并补充PPT原生表格行；F11本轮与F10表示和索引相同；等数量替换仅为敏感性实验。</p></section>
<section class="panel"><h2>阶段汇总</h2><label>题集 <select id="cohort"><option value="formal">正式正向46题</option><option value="diagnostic">原诊断正向40题</option><option value="source_holdout">新来源待审核20题</option></select></label><label>指标 <select id="metric"><option value="complete">全部金标引文覆盖题数</option><option value="mrr">第一条金标 MRR</option></select></label><div class="tablewrap"><table id="summary"></table></div></section>
<section class="panel"><h2>逐题方案对照</h2><label>题目 <select id="case"></select></label><label>搜索题号或问题 <input id="search" placeholder="如 rag-biz-019"></label><p id="question"></p><div id="gold"></div><div class="tablewrap"><table id="comparison"></table></div></section>
<section class="panel" id="detailPanel"><h2>单元格详情</h2><p>点击上表中的任意阶段查看。</p></section><p id="message"></p></main><script>
const DATA=__DATA__;
const $=id=>document.getElementById(id);const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function sources(ids){return (ids||[]).map(id=>`<button class="source" data-source="${esc(id)}">打开原资料对应位置</button>`).join(' ')}
function summary(){let co=$('cohort').value,m=$('metric').value; $('summary').innerHTML='<thead><tr><th>方案</th>'+DATA.names.map(n=>'<th>'+esc(n)+'</th>').join('')+'</tr></thead><tbody>'+DATA.arms.map(a=>'<tr><td>'+esc(a)+'</td>'+DATA.stages.map(s=>{let x=DATA.summary[a][co][s];return '<td>'+(m==='complete'?x.complete+'/'+x.cases:x.mrr.toFixed(4))+'</td>'}).join('')+'</tr>').join('')+'</tbody>'}
function fill(){let filter=$('search').value.toLowerCase(),old=$('case').value;let cases=DATA.cases.filter(c=>(c.id+' '+c.question).toLowerCase().includes(filter));$('case').innerHTML=cases.map(c=>`<option value="${esc(c.id)}">${esc(c.id+' · '+c.question)}</option>`).join('');if(cases.some(c=>c.id===old))$('case').value=old;render()}
function cell(x){if(!x.gold_evidence_total_declared)return '<span class="na">不适用</span>';let r=x.first_gold_rank?'第'+x.first_gold_rank+' / '+x.candidate_count:'未命中 / '+x.candidate_count;let c=x.gold_evidence_found+'/'+x.gold_evidence_total_declared;let cls=x.gold_evidence_found===x.gold_evidence_total_declared?'complete':x.gold_evidence_found?'partial':'miss';return `<span class="${cls}">${r}<br>引文 ${c}</span>`}
function render(){let c=DATA.cases.find(x=>x.id===$('case').value);if(!c){$('question').textContent='没有匹配题目';$('comparison').innerHTML='';$('gold').innerHTML='';return}$('question').textContent=c.question+' · '+c.status+' · '+c.cohort;$('gold').innerHTML=c.gold.map((g,i)=>`<details><summary>金标 ${i+1}：${esc(g.file||'')} ${esc(g.position||g.location||g.page||'')}</summary><p>${esc(g.quote)}</p>${sources(g.source_ids)}</details>`).join('');$('comparison').innerHTML='<thead><tr><th>方案</th>'+DATA.names.map(n=>'<th>'+esc(n)+'</th>').join('')+'</tr></thead><tbody>'+DATA.arms.map(a=>'<tr><td>'+esc(a)+'</td>'+DATA.stages.map(s=>`<td><button data-arm="${esc(a)}" data-stage="${esc(s)}">${cell(c.arms[a][s])}</button></td>`).join('')+'</tr>').join('')+'</tbody>'; $('detailPanel').innerHTML='<h2>单元格详情</h2><p>点击上表中的任意阶段查看。</p>'}
document.addEventListener('click',async e=>{let b=e.target.closest('[data-arm]');if(b){let c=DATA.cases.find(x=>x.id===$('case').value),x=c.arms[b.dataset.arm][b.dataset.stage];$('detailPanel').innerHTML='<h2>'+esc(c.id+' · '+b.dataset.arm+' · '+DATA.names[DATA.stages.indexOf(b.dataset.stage)])+'</h2>'+c.gold.map((g,i)=>{let rank=x.gold_completion_ranks[g.id],match=x.matches[g.id];return `<h3>金标 ${i+1}：${rank?'第 '+rank+' 名':'未命中'}</h3><p>${esc(g.quote)}</p>${sources(g.source_ids)}${match?`<p>实际匹配：${esc(match.filename)} ${esc(match.location)}</p>${sources(match.source_ids)}<pre class="detail">${esc(match.text)}</pre>`:''}`}).join('');return}b=e.target.closest('[data-source]');if(!b)return;let token=new URLSearchParams(location.search).get('token');if(!token){$('message').textContent='请运行同目录 open_report.cmd 后打开报告。';return}try{let r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.source})});let d=await r.json();$('message').textContent=d.message||d.error}catch(err){$('message').textContent='打开失败：'+err}});
$('case').onchange=render;$('search').oninput=fill;$('cohort').onchange=summary;$('metric').onchange=summary;fill();summary();</script></body></html>'''

if __name__ == '__main__':
    main()
