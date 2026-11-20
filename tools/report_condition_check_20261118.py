"""Use the existing native report template for condition factorial controls."""
import html
import json
from tools.validate_condition_check_20261118 import OUT, ROOT, ARMS, exp
from tools.render_general_native_report import packed

LABELS={'baseline':'基线：旧判定＋旧排序','A':'A：旧判定＋保持BGE顺序',
        'B':'B：局部关系判定＋旧排序','AB':'A+B：局部关系判定＋保持BGE顺序',
        'ABC':'A+B+C：另放开未知条件门禁（未采用）','AC':'A+C：旧判定＋保序＋放开门禁（未采用）'}


def main():
    from tools import report_vector_all_stages_20261117 as compare
    compare.OUT=compare.DEST=OUT;compare.ARMS=tuple(ARMS)
    compare.main()
    from tools import render_vector_all_stages_native_20261117 as native
    native.OUT=native.DEST=OUT;native.ARMS=tuple(ARMS)
    original_builder=native.service.build_report_detail
    frozen={r['id']:r for r in exp.read_rows(exp.BASE)}
    def build(dataset,metadata,results,config):
        metadata['mode']='条件检查六组隔离实验；冻结BGE窗口最高分及上游；API 0'
        for r in results:
            keys={'stage','query_limits','query_plan','lexical_fts_expression','lexical_terms','lexical_route_diagnostics','vector_route_diagnostics'}
            r['events']=[{k:v for k,v in e.items() if k in keys} for e in frozen[r['id']]['events'] if any(k in e for k in keys-{'stage'})]
        detail=original_builder(dataset,metadata,results,config)
        byid={r['id']:r for r in results}
        for c in detail['cases']:
            r=byid[c['id']]
            state='丢失金标' if r['lost_gold'] else '恢复金标' if r['gained_gold'] else 'Context变化' if r['context_changed'] else '无变化'
            detail.setdefault('baseline',{}).setdefault('case_changes',{})[c['id']]={'state':state}
            for s in ('rerank','constraints','evidence','context'):
                assert c['stage_summaries'][s]['gold_found']==r['scores'][s]['gold_evidence_found'],(c['id'],s)
        return detail
    native.service.build_report_detail=build
    try:native.main()
    finally:native.service.build_report_detail=original_builder
    metrics=json.loads((OUT/'metrics.json').read_text(encoding='utf8'))
    audit=json.loads((OUT/'selection_audit.json').read_text(encoding='utf8'))
    conclusion=(OUT/'conclusion.md').read_text(encoding='utf8') if (OUT/'conclusion.md').exists() else '验证仍在进行，不能宣布采用。'
    banner='<section class="panel"><h2>条件检查：排序、关系判定、门禁的独立对照</h2><p>111题 × 6方案；实际Context逐字匹配Gold，API调用0。索引、Gold、BGE分窗口最高分、Top20入口、Top8及12000字符预算保持一致。正式正向46题、诊断正向40题、来源保留pending20题分别统计；无正向Gold题另审，不能从统计中消失。</p><table><tr><th>方案</th><th>正式完整引文</th><th>诊断完整引文</th><th>来源保留完整引文</th><th>新增标注负例</th><th>无正向Gold题的Context变化</th></tr>'
    for a in ARMS:
        banner+='<tr><td>'+LABELS[a]+'</td>'+''.join('<td>'+str(metrics[a][c]['stages']['context']['complete'])+'/'+str(metrics[a][c]['stages']['context']['positive_cases'])+'</td>' for c in ('formal','diagnostic','source_holdout'))
        banner+='<td>'+html.escape('、'.join(i for c in metrics[a].values() for i in c['new_negatives']) or '无')+'</td><td>'+html.escape('、'.join(x['id'] for x in audit['no_gold_behavior_changes'][a]) or '无')+'</td></tr>'
    banner+='</table><p>“完整引文”不是答案准确率；标注负例不覆盖所有错对象和错场景。C使017/018跨产线题由空Context变为非空，属于不采用的阻断项。</p><details open><summary>结论与采用状态</summary><pre>'+html.escape(conclusion)+'</pre></details><p><a href="conclusion.md">结论全文</a> · <a href="selection_audit.json">全部选段变化与引文核验</a> · <a href="source_review_reading.md">新增选段原文</a> · <a href="independent_relations_results.json">新增待审关系探针</a></p></section>'
    details={}
    for a in ARMS:
        details[a]={}
        for r in exp.read_rows(OUT/(a+'_native.jsonl')):
            before={c['chunk_id']:i for i,c in enumerate(r['candidate_stages']['rerank'],1)}
            details[a][r['id']]={k:r[k] for k in ('context','entries','gained_gold','lost_gold','added_hard_negative','decision')}
            details[a][r['id']]['condition_audit']=[dict(chunk_id=c['chunk_id'],filename=c['filename'],location=c['location'],
                bge_rank=before[c['chunk_id']],condition_rank=i,decision=c['constraint_decision'],
                relation_proofs=c.get('query_plan_diagnostics',{}).get('condition_relations',{}),
                final_selected=any(e['chunk_id']==c['chunk_id'] for e in r['entries'])) for i,c in enumerate(r['candidate_stages']['constraints'],1)]
    page=(OUT/'report.html').read_text(encoding='utf8')
    page=page.replace('111题 × 9方案','111题 × 6方案').replace('let activeArm="W0"','let activeArm="baseline"')
    page=page.replace('<main>','<main>'+banner,1)
    page=page.replace('<p id="sourceStatus">','<div id="conditionDetails"></div><p id="sourceStatus">',1)
    for a,label in LABELS.items():page=page.replace(f'<option value="{a}">{a}</option>',f'<option value="{a}">{label}</option>')
    extra="const CONDITION_COMPARE=await unpack('"+packed(details)+"');\n"+r'''
function renderConditionDetails(){const id=document.getElementById('compareCase').value;if(!CONDITION_COMPARE.baseline[id])return;document.getElementById('conditionDetails').innerHTML='<h3>实际Context与条件判断对照</h3><div class="detail-grid">'+Object.keys(CONDITION_COMPARE).map(a=>{const r=CONDITION_COMPARE[a][id];return `<div><h4>${esc(a)}</h4><p>恢复：${esc(r.gained_gold.join(', ')||'无')}；丢失：${esc(r.lost_gold.join(', ')||'无')}；门禁：${esc(r.decision.reason)}</p><details><summary>实际Context（${r.entries.length}段）</summary><pre>${esc(r.context)}</pre></details><details><summary>BGE/条件排名、判定原文与排除依据</summary><pre>${esc(JSON.stringify(r.condition_audit,null,2))}</pre></details><details><summary>最终门禁与所有排除原因</summary><pre>${esc(JSON.stringify(r.decision,null,2))}</pre></details></div>`}).join('')+'</div>';}
document.getElementById('compareCase').addEventListener('change',renderConditionDetails);renderConditionDetails();
'''
    page=page.replace('\ninit();\n',extra+'\ninit();\n',1)
    (OUT/'report.html').write_text(page,encoding='utf8')
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n',encoding='utf8')
    print('CONDITION_REPORT',OUT/'report.html',flush=True)


if __name__=='__main__':main()
