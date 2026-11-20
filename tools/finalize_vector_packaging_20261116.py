"""Fail-closed completion audit and source-bound comparison, no answer claims."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from tools.validate_vector_packaging_20261116 import *

def main():
    store,bundle,settings=inputs()
    selection=read('selection.json');factorial=read('factorial_protocol.json');native=read('native_audit.json')
    arms=list(dict.fromkeys(['W0',selection['packaging_candidate'],'F10','F11']))
    expected=['representation_audit.json','background_structure_audit.json','holdout_evidence_audit.json','encoding_complete.json',
        'ann_audit.json','dual_audit.json','factorial_protocol.json','selection.json','native_audit.json','all_pytest.xml']
    expected += [f'W{i}_exact.jsonl' for i in range(6)]
    expected += [f'{a}_ann.jsonl' for a in ('W0','W4','W5','F10','F11','F10_count_matched')]
    expected += [f'{a}_exact.jsonl' for a in ('F10','F11','F10_count_matched')]
    expected += [f'{a}_exact_holdout.jsonl' for a in arms]
    expected += [f'{a}_ann_holdout.jsonl' for a in arms]+[f'{a}_native.jsonl' for a in arms]
    assert all((OUT/f).is_file() for f in expected),[f for f in expected if not (OUT/f).is_file()]
    tests=ET.parse(OUT/'all_pytest.xml').findall('.//testcase')
    failed=[c.attrib for c in tests if c.find('failure') is not None or c.find('error') is not None]
    assert not failed,failed
    comparisons={};summaries={};stagechanges={};reviews=[]
    baseline=lines(OUT/'W0_ann_holdout.jsonl');nativebase=lines(OUT/'W0_native.jsonl')
    for arm in arms:
        records=lines(OUT/f'{arm}_ann_holdout.jsonl');run=lines(OUT/f'{arm}_native.jsonl')
        assert len(records)==len(run)==111
        summaries[arm]=metrics(records);comparisons[arm]=deltas(baseline,records)
        stagechanges[arm]={}
        for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context'):
            common=[(b,a) for b,a in zip(nativebase,run) if stage in b.get('scores',{}) and stage in a.get('scores',{})]
            stagechanges[arm][stage]=deltas([b for b,a in common],[a for b,a in common],stage)
        for b,a in zip(baseline,records):
            before=b['candidate_stages']['vector'][:5];after=a['candidate_stages']['vector'][:5]
            if [x['chunk_id'] for x in before]!=[x['chunk_id'] for x in after]:
                reviews.append(dict(arm=arm,id=a['id'],question=a['question'],review_status=a['review_status'],
                    before=before,after=after,known_negative_counts=[b['scores']['vector']['hard_negative_found'],a['scores']['vector']['hard_negative_found']],
                    review_status_for_new_top5='pending_source_applicability_review'))
    write(OUT/'comparisons.json',dict(vector=comparisons,stages=stagechanges,metrics=summaries))
    write(OUT/'top5_applicability_review.json',reviews)
    # Actual Context is measured separately from vector acquisition; existing
    # downstream limitations cannot erase measured vector gains.
    gate={}
    for arm in arms:
        changes=comparisons[arm];ctx=stagechanges[arm]['context']
        improved=set(changes['gains']+changes['rank_gains'])
        holdout=[r for r in lines(OUT/f'{arm}_ann_holdout.jsonl') if r['cohort']=='source_holdout' and r['id'] in improved]
        heldcases={c.case_id:c for c in load_gold_dataset(FIXTURE,reviewed_only=False).cases}
        raw=[json.loads(x) for x in FIXTURE.read_text(encoding='utf8').splitlines()]
        groups={r['id']:r['annotation_provenance']['source_group'] for r in raw}
        improvedgroups={groups[r['id']] for r in holdout}
        gate[arm]=dict(formal_complete=summaries[arm]['formal']['complete'],no_new_gold_loss=not changes['losses'],
            no_new_known_negative=not changes['new_negatives'],independent_improved_source_groups=sorted(improvedgroups),
            no_new_actual_context_loss=not ctx['losses'],
            full_top5_applicability_human_review='pending' if any(r['arm']==arm for r in reviews) else 'no_changes',
            vector_hard_gates_pass=not changes['losses'] and not changes['new_negatives'] and len(improvedgroups)>=2)
    write(OUT/'adoption_gates.json',gate)
    # Layout screenshots / business applicability are not silently replaced by
    # an automatic word-overlap score. No unreviewed candidate is activated.
    conclusion='暂不采用新的默认包装或结构策略。当前代码提供可隔离验证的通用能力；生产仍使用原包装与原索引。'
    rows=[]
    for arm,m in summaries.items():
        rows.append(f'| {arm} | {m["formal"]["complete"]}/46 | {m["diagnostic"]["complete"]}/40 | {m["source_holdout"]["complete"]}/20 | {m["formal"]["mrr"]:.4f} |')
    text='# 向量包装与结构化表示：开发验证结果\n\n'+conclusion+'\n\n'
    text+='| 方案 | 正式完整Gold@50 | 原诊断完整Gold@50 | 新来源诊断完整Gold@50 | 正式MRR |\n|---|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n'
    for arm in arms:
        text+=f'## {arm}\n\n向量新增完整引文：{comparisons[arm]["gains"]}；新增丢失：{comparisons[arm]["losses"]}。\n\n'
        text+=f'实际Context新增完整引文：{stagechanges[arm]["context"]["gains"]}；丢失：{stagechanges[arm]["context"]["losses"]}。\n\n'
        text+=f'独立来源收益组：{gate[arm]["independent_improved_source_groups"]}；源对象适用性复核：{gate[arm]["full_top5_applicability_human_review"]}。\n\n'
    text+='## 因素与边界\n\nW0–W5保持52282个原逻辑窗口及原跨度；只改变编码文字。F10/F11加入同一批原生PPT表格行；增加向量数量与结构本身的收益不能混算。F10_count_matched提供固定数量替换敏感性。\n\n'
    text+='W6：'+read('dual_audit.json')['reason']+'\n\n'
    text+='PDF、图片表格恢复未实现；没有可信标题的Word/PDF使用完整文件名hint，不假装标题提取已覆盖所有格式。复杂合并表格保留原表示并记录未覆盖。\n\n'
    text+='新增20题全部pending；资料出处和引文已核对，不代表业务人员正式验收。新Top5对象/范围复核清单见top5_applicability_review.json。原已审核测试集及007/040状态未改变。\n\n'
    text+='未调用生成或裁判API，不报告答案正确率。019、IMFH1、条码等如果证据在向量之后丢失，按实际最早环节记录，不能用下游失误否认本轮向量层收益。\n\n'
    text+='## 可复现命令\n\n` .venv/Scripts/python.exe -X utf8 -m tools.run_vector_packaging_20261116 --resume `\n\n'
    text+=f'单元/兼容测试：{len(tests)}项，失败{len(failed)}。API调用0。主报告report.html；open_report.cmd支持原资料位置按钮。\n'
    (OUT/'conclusion.md').write_text(text,encoding='utf8')
    (ROOT/'docs/rag_vector_packaging_development_validation_result_20261116.md').write_text(text,encoding='utf8')
    immutable=read('protocol.json')['protected'];drift=[p for p,h in immutable.items() if sha(p)!=h]
    assert not drift,drift
    acceptance=dict(completed=False,scope='API-before only; no generation or answer-correctness claim',cases=111,
        formal_positive=46,diagnostic_positive=40,source_holdout_pending=20,api_calls=0,network_api_attempts=0,
        immutable_drift=drift,source_bundle=bundle.bundle_id,default_adapter_enabled=settings.rag_query_plan_enabled,
        tests=dict(total=len(tests),failed=len(failed)),gates=gate,production_activated=False,
        pending_business_review=20,source_applicability_review_count=len(reviews))
    write(OUT/'final_acceptance.json',acceptance)
    write(OUT/'execution_status.json',dict(status='validating_report',current='all experimental batches executed; report verification pending',updated=datetime.now(timezone.utc).isoformat()))
    from tools.report_vector_packaging_20261116 import main as report
    report()
    import subprocess,sys
    subprocess.run([sys.executable,'-X','utf8','-m','tools.check_packaging_report_ui'],cwd=ROOT,check=True)
    ui=read('report_ui_audit.json')
    assert ui['cases']==111 and not ui['page_errors'] and ui['source_request_resolves']
    write(OUT/'execution_status.json',dict(status='complete',current='all experimental batches executed; no production activation',updated=datetime.now(timezone.utc).isoformat()))
    report()
    subprocess.run([sys.executable,'-X','utf8','-m','tools.check_packaging_report_ui'],cwd=ROOT,check=True)
    assert read('report_ui_audit.json')['complete']
    acceptance['completed']=True
    write(OUT/'final_acceptance.json',acceptance)

if __name__=='__main__':main()
