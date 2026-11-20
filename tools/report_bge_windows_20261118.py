"""Render BGE window controls with the existing native RAG report viewer."""
from dataclasses import asdict
from collections import Counter
import html
import json
from pathlib import Path
from tools.validate_bge_windows_20261118 import OUT, ROOT, ARMS, read_rows
from tools.render_general_native_report import packed

LABELS={'truncate':'原方案：512-token截断','window_max':'分窗口：最高分','window_top2_mean':'分窗口：前两窗口平均分'}

def main():
    from tools import report_vector_all_stages_20261117 as compare
    compare.OUT=compare.DEST=OUT;compare.ARMS=ARMS
    compare.main()
    from tools import render_vector_all_stages_native_20261117 as native
    native.OUT=native.DEST=OUT;native.ARMS=ARMS
    # Restore actual execution metadata omitted by the older presentation helper.
    frozen={r['id']:r for r in read_rows(ROOT/'reports/rag_vector_packaging_20261116/W0_native.jsonl')}
    original_builder=native.service.build_report_detail
    def build(dataset,metadata,results,config):
        metadata['mode']='BGE输入隔离实验；冻结上游候选；LLM未运行'
        for r in results:
            keys={'stage','query_limits','query_plan','lexical_fts_expression','lexical_terms','lexical_route_diagnostics','vector_route_diagnostics'}
            r['events']=[{k:v for k,v in e.items() if k in keys} for e in frozen[r['id']]['events'] if any(k in e for k in keys-{'stage'})]
        detail=original_builder(dataset,metadata,results,config)
        byid={r['id']:r for r in results}
        for case in detail['cases']:
            r=byid[case['id']]
            state='丢失金标' if r['lost_gold'] else '恢复金标' if r['gained_gold'] else '新增标注负例' if r['added_hard_negative'] else 'Context变化' if r['context_changed'] else '无变化'
            detail.setdefault('baseline',{}).setdefault('case_changes',{})[case['id']]={'state':state}
            for stage in ('rerank','constraints','evidence','context'):
                assert case['stage_summaries'][stage]['gold_found']==r['scores'][stage]['gold_evidence_found'],(case['id'],stage)
        return detail
    native.service.build_report_detail=build
    try:native.main()
    finally:native.service.build_report_detail=original_builder
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    variants={a:read_rows(OUT/(a+'_native.jsonl')) for a in ARMS}
    summary='<section class="panel"><h2>仅修改 BGE 输入：全量对照</h2><p>111题 × 3方案。原始召回、去重与重排入口20条冻结；条件检查、Top8、0.1门槛及12000字符预算不变。新窗口使用真实本地BGE推理；token完全相同的输入复用既有FP32分数。未重跑上游召回，未调用API，未生成答案。最高分为主要候选，前两窗口平均分为预设对照，未按题目调参。</p><table><tr><th>方案</th><th>集合</th><th>重排完整证据</th><th>实际Context完整证据</th><th>Context MRR</th><th>Context负例引文</th><th>恢复 / 丢失证据题</th></tr>'
    cohort_names={'formal':'已审核正向题','diagnostic':'诊断正向题（待审核）','source_holdout':'来源保留题（待审核）'}
    for a in ARMS:
        for cohort,m in metrics[a].items():
            s=m['stages'];c=s['context'];r=s['rerank']
            summary+=f'<tr><td>{LABELS[a]}</td><td>{cohort_names[cohort]}</td><td>{r["complete"]}/{r["positive_cases"]}</td><td>{c["complete"]}/{c["positive_cases"]}</td><td>{c["mrr"]:.4f}</td><td>{c["negative_quotes"]}</td><td>{html.escape("、".join(m["gains"]) or "无")} / {html.escape("、".join(m["losses"]) or "无")}</td></tr>'
    summary+='</table><p>完整证据表示该题全部Gold引文保留；不等于答案准确率。负例数仅覆盖已标注负例，不能证明其他资料全部适用。所有Gold审核状态不变。</p><p><b>结论：</b>最高分修复目标行不可见，恢复1道诊断题；8题首条金标排名下降但未丢失已有Gold。前两窗口平均分丢失2题，并新增3条正式标注负例，不采用。窗口能力已实现为可注入选项，本轮保持应用默认truncate及用户设置不变。新增其他型号/报警片段见结论，不能宣称所有题变好。</p><a href="conclusion.md">本轮结论</a> · <a href="final_acceptance.json">执行与验收记录</a></section>'
    adoption=ROOT/'reports/rag_bge_window_adoption_20261118/adoption.json'
    if adoption.exists():
        summary=summary.replace('窗口能力已实现为可注入选项，本轮保持应用默认truncate及用户设置不变。',
            '2026-11-18已按用户要求正式采用分窗口最高分，普通问答和批量评测默认window_max。接入后111题回放与4题真实入口均与实验一致，未修改用户持久化设置。此表保留原三组实测对照。')
        summary=summary.replace('<h2>仅修改 BGE 输入：全量对照</h2>', '<h2>已采用分窗口最高分 · BGE输入全量对照</h2>')
        summary=summary.replace('</section>', '<details><summary>当前默认接入核验</summary><pre>'+html.escape(adoption.read_text(encoding='utf8'))+'</pre></details></section>',1)
    details={}
    for a,rows in variants.items():
        details[a]={}
        for r in rows:
            window_items=[]
            for c in r['candidate_stages']['rerank']:
                d=c.get('rerank_diagnostics') or {}
                if d.get('window_count',0)>1:
                    window_items.append({'file':c['filename'],'location':c['location'],'chunk_id':c['chunk_id'],
                        'score':c['rerank_score'],'windows':[{**w,'actual_source_text':c['text'][w['start_char']:w['end_char']]} for w in d['windows']]})
            details[a][r['id']]={k:r[k] for k in ('context','entries','gained_gold','lost_gold','added_hard_negative','decision')}
            details[a][r['id']]['windows']=window_items
    page=(OUT/'report.html').read_text(encoding='utf8')
    page=page.replace('111题 × 9方案','111题 × 3方案').replace('各方案 · 各阶段金标排名','BGE输入方案 · 各阶段金标排名')
    page=page.replace('原始Gold、索引版本、模型及其他阶段设置保持一致','原始Gold、索引版本、BGE模型、上游候选及其他阶段设置保持一致')
    page=page.replace('API前各方案各阶段对照，答案未运行','BGE输入隔离对照，冻结上游候选，答案未运行')
    page=page.replace('let activeArm="W0"','let activeArm="truncate"')
    for a,label in LABELS.items():page=page.replace(f'<option value="{a}">{a}</option>',f'<option value="{a}">{label}</option>')
    page=page.replace('<main>','<main>'+summary,1)
    page=page.replace('<p id="sourceStatus">','<div id="windowDetails"></div><p id="sourceStatus">',1)
    extra="const WINDOW_COMPARE=await unpack('"+packed(details)+"');\n"+r'''
function renderWindowDetails(){const id=document.getElementById('compareCase').value;const baseline=WINDOW_COMPARE.truncate[id];if(!baseline)return;document.getElementById('windowDetails').innerHTML='<h3>该题实际Context与窗口评分对照</h3><div class="detail-grid">'+Object.keys(WINDOW_COMPARE).map(a=>{const r=WINDOW_COMPARE[a][id];return `<div><h4>${esc(a)}</h4><p>恢复引文：${esc(r.gained_gold.join(', ')||'无')}；丢失：${esc(r.lost_gold.join(', ')||'无')}；新增标注负例：${esc(r.added_hard_negative.join(', ')||'无')}</p><details><summary>实际Context（${r.context.length}字符，${r.entries.length}段）</summary><pre>${esc(r.context)}</pre></details><details><summary>长片段各窗口实际输入、位置与分数</summary><pre>${esc(JSON.stringify(r.windows,null,2))}</pre></details><details><summary>最终排除原因</summary><pre>${esc(JSON.stringify(r.decision,null,2))}</pre></details></div>`}).join('')+'</div>';}
document.getElementById('compareCase').addEventListener('change',renderWindowDetails);renderWindowDetails();
'''
    page=page.replace('\ninit();\n',extra+'\ninit();\n',1)
    (OUT/'report.html').write_text(page,encoding='utf8')
    (OUT/'open_report.cmd').write_text('@echo off\ncd /d "'+str(ROOT)+'"\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\n',encoding='utf8')
    print(OUT/'report.html',flush=True)

if __name__=='__main__':main()
