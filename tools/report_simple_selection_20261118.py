"""Application-native eight-stage comparison, actual Context and source links."""
import html
import json
from tools.validate_simple_selection_20261118 import OUT, ROOT, ARMS, exp
from tools.render_general_native_report import packed

LABELS = {'baseline':'修改前：旧条件排序与门禁', 'A':'A：仅保留BGE顺序，仍有条件门禁',
          'simple_legacy_budget':'消融：无条件门禁／阈值／Top8，沿用截断预算',
          'simple':'修改后：无条件过滤，完整片段按预算选入'}


def main():
    from tools import report_vector_all_stages_20261117 as compare
    compare.OUT = compare.DEST = OUT
    compare.ARMS = ARMS
    compare.main()
    from tools import render_vector_all_stages_native_20261117 as native
    native.OUT = native.DEST = OUT
    native.ARMS = ARMS
    original_builder = native.service.build_report_detail
    frozen = {r['id']:r for r in exp.read_rows(ROOT/'reports/rag_vector_packaging_20261116/W0_native.jsonl')}
    def build(dataset, metadata, results, config):
        metadata['mode'] = 'LLM前阶段固定；简单方案Context已进行111题生成与独立Gold裁判；自动判分不等于人工确认'
        for r in results:
            keys = {'stage','query_limits','query_plan','lexical_fts_expression','lexical_terms','lexical_route_diagnostics','vector_route_diagnostics'}
            r['events'] = [{k:v for k,v in e.items() if k in keys} for e in frozen[r['id']]['events'] if any(k in e for k in keys-{'stage'})]
        detail = original_builder(dataset, metadata, results, config)
        byid = {r['id']:r for r in results}
        for c in detail['cases']:
            r = byid[c['id']]
            state = '丢失金标' if r['lost_gold'] else '恢复金标' if r['gained_gold'] else '新增标注负例' if r['added_hard_negative'] else 'Context变化' if r['context_changed'] else '无变化'
            detail.setdefault('baseline',{}).setdefault('case_changes',{})[c['id']] = {'state':state}
            for s in ('rerank','constraints','evidence','context'):
                assert c['stage_summaries'][s]['gold_found'] == r['scores'][s]['gold_evidence_found'], (c['id'],s)
        return detail
    native.service.build_report_detail = build
    try:
        native.main()
    finally:
        native.service.build_report_detail = original_builder
    metrics = json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    conclusion = (OUT/'conclusion.md').read_text(encoding='utf8') if (OUT/'conclusion.md').exists() else '验证尚未完成。'
    banner = '<section class="panel"><h2>删除条件过滤 · 最简证据选择 · 修改前后对照</h2><p>111题 × 4方案，8个LLM前阶段固定；另对简单方案冻结Context调用已配置生成模型111次，并用独立Gold裁判判分111次。索引、Gold、原始问题、召回、去重、BGE窗口最高分和12000字符预算固定。修改后不使用条件删除、条件排序、分数阈值、Top8或单文件配额；只按BGE顺序与完整片段预算装入。此轮答案判分由LLM自动完成，不等于人工确认。待审、争议题不进入正式分数。</p><p>条件检查列为兼容历史横向对照而保留；修改后该列只透传，不执行判断。最终筛选列是预算前候选；<b>实际Context列才表示真正装入的正文</b>。旧配置里的Top8和阈值在简化方案中不生效。</p><table><tr><th>方案</th><th>正式完整引文</th><th>诊断完整引文</th><th>来源待审完整引文</th><th>新增标注负例题</th></tr>'
    for arm in ARMS:
        banner += '<tr><td>'+LABELS[arm]+'</td>'+''.join('<td>'+str(metrics[arm][c]['stages']['context']['complete'])+'/'+str(metrics[arm][c]['stages']['context']['positive_cases'])+'</td>' for c in ('formal','diagnostic','source_holdout'))
        banner += '<td>'+html.escape('、'.join(i for c in metrics[arm].values() for i in c['new_negatives']) or '无')+'</td></tr>'
    banner += '</table><p><b>结论：没有丢失原Gold；相对A仅新增1题待审来源证据。扩大选段也让更多低相关和不同场景资料进入Context，不能据此宣布答案质量提升。</b></p><p>完整引文是来源与原文匹配，不是最终答案正确率。无正向Gold题单列观察；未标注的错对象／错产线不会自动被负例计数覆盖。</p><details><summary>展开完整实验结论、归因和剩余问题</summary><pre>'+html.escape(conclusion)+'</pre></details><p><a href="conclusion.md">结论全文</a> · <a href="final_acceptance.json">完整执行与验收记录</a>。逐题实际正文、预算排除和来源在下方展开查看。</p></section>'
    details = {}
    for arm in ARMS:
        details[arm] = {}
        for r in exp.read_rows(OUT/(arm+'_native.jsonl')):
            details[arm][r['id']] = {k:r[k] for k in ('context','entries','excluded_entries','gained_gold','lost_gold','added_hard_negative','decision')}
            details[arm][r['id']]['selection_config'] = r.get('selection_config', dict(legacy=True, final_top_k_applied=True))
    page = (OUT/'report.html').read_text(encoding='utf8')
    page = page.replace('111题 × 9方案', '111题 × 4方案').replace('let activeArm="W0"', 'let activeArm="simple"')
    page = page.replace('业务条件检查', '条件检查（修改后已移除）')
    page = page.replace('<main>', '<main>'+banner, 1)
    page = page.replace('<p id="sourceStatus">', '<div id="simpleDetails"></div><div id="apiAnswerPanel"></div><p id="sourceStatus">', 1)
    for arm, label in LABELS.items():
        page = page.replace(f'<option value="{arm}">{arm}</option>', f'<option value="{arm}">{label}</option>')
    extra = "document.getElementById('armSelect').value=activeArm;\nconst SIMPLE_COMPARE=await unpack('"+packed(details)+"');\n"+r'''
function renderSimpleDetails(){const id=document.getElementById('compareCase').value;if(!SIMPLE_COMPARE.simple[id])return;document.getElementById('simpleDetails').innerHTML='<h3>修改前后实际Context与选段原因</h3><div class="detail-grid">'+Object.keys(SIMPLE_COMPARE).map(a=>{const r=SIMPLE_COMPARE[a][id];return `<div><h4>${esc(a)}</h4><p>恢复：${esc(r.gained_gold.join(', ')||'无')}；丢失：${esc(r.lost_gold.join(', ')||'无')}；新增标注负例：${esc(r.added_hard_negative.join(', ')||'无')}</p><details><summary>实际Context（${r.entries.length}段，${r.context.length}字符）</summary><pre>${esc(r.context)}</pre></details><details><summary>实际选段策略、排除与预算原因</summary><pre>${esc(JSON.stringify({config:r.selection_config,decision:r.decision,context_exclusions:r.excluded_entries},null,2))}</pre></details></div>`}).join('')+'</div>';}
document.getElementById('compareCase').addEventListener('change',renderSimpleDetails);renderSimpleDetails();
'''
    page = page.replace('\ninit();\n', extra+'\ninit();\n', 1)
    (OUT/'report.html').write_text(page, encoding='utf8')
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n', encoding='utf8')
    if (OUT/'api_answers.jsonl').is_file() and (OUT/'api_summary.json').is_file():
        from tools.update_simple_selection_api_report_20261118 import main as attach_api_answers
        attach_api_answers()
    print('REPORT', OUT/'report.html', flush=True)


if __name__ == '__main__':
    main()
