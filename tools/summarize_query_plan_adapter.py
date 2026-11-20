"""Source-quote and answer comparisons; preserves original automatic judgements."""
from collections import Counter
from pathlib import Path
import json

from tools.rag_eval_common import lines, read, write, sha
from tools.validate_query_plan_adapter import OUT, FILES

def answer_rows(prefix):
    result={};errors=[]
    for path in sorted(OUT.glob(prefix+'*.jsonl')):
        for row in lines(path):
            if row.get('error'): errors.append({'file':path.name,'id':row['id'],'error':row['error']});continue
            if row.get('judgement') and row.get('context_equal'):
                assert row['id'] not in result, (path,row['id'])
                result[row['id']]={**row,'run_file':path.name}
    return result,errors

def main():
    old={r['id']:r for r in lines(OUT/'baseline.jsonl')}
    new={r['id']:r for r in lines(OUT/'candidate.jsonl')}
    a,ae=answer_rows('answers_A');b,be=answer_rows('answers_B')
    assert len(old)==len(new)==len(a)==len(b)==92, (len(old),len(new),len(a),len(b))
    comparison=[]
    for key,r in old.items():
        n=new[key]
        prev=set(r['scores']['context']['gold_completion_ranks']);curr=set(n['scores']['context']['gold_completion_ranks'])
        comparison.append({'id':key,'question':r['question'],'cohort':r['cohort'],'review_status':r['review_status'],
            'gained':sorted(curr-prev),'lost':sorted(prev-curr),'context_changed':r['context']!=n['context'],
            'stages':{stage:{'A':r['scores'].get(stage,{}),'B':n['scores'].get(stage,{})} for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')},
            'answer_A':a[key]['judgement']['verdict'],'answer_B':b[key]['judgement']['verdict'],
            'answer_text_A':a[key]['answer']['text'],'answer_text_B':b[key]['answer']['text'],
            'answer_files':[a[key]['run_file'],b[key]['run_file']]})
    cohorts={}
    for name in dict.fromkeys(r['cohort'] for r in comparison):
        rows=[r for r in comparison if r['cohort']==name]
        cohorts[name]={'count':len(rows),'A':dict(Counter(r['answer_A'] for r in rows)),
                       'B':dict(Counter(r['answer_B'] for r in rows)),
                       'context_changes':sum(r['context_changed'] for r in rows),
                       'gained_questions':[r['id'] for r in rows if r['gained']],
                       'lost_questions':[r['id'] for r in rows if r['lost']]}
    manifest=read(OUT/'candidate_manifest.json')
    drift=[p for p,h in manifest['code_hashes'].items() if sha(p)!=h]
    drift += [p for p,h in manifest['dataset_hashes'].items() if sha(p)!=h]
    summary={'cohorts':cohorts,'input_drift':drift,'transport_or_harness_errors':ae+be,
             'retrieval_runs_per_arm':92,'paired_downstream_answers':184,
             'changed_answer_ids':[r['id'] for r in comparison if r['answer_A']!=r['answer_B']],
             'regressions':[r['id'] for r in comparison if r['answer_A']=='consistent' and r['answer_B']!='consistent'],
             'improvements':[r['id'] for r in comparison if r['answer_A']!='consistent' and r['answer_B']=='consistent'],
             'adoption':'暂不默认采用；保留可配置实现。存在方法型问题回答回退，需要解决证据简表与详细分支的优先级。'}
    summary['manual_review']=read(OUT/'manual_review.json')
    stage_totals={}
    for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context'):
        stage_totals[stage]={}
        for cohort in ('formal','all'):
            subset=[r for r in comparison if cohort=='all' or r['cohort']=='formal']
            stage_totals[stage][cohort]={arm:sum(r['stages'][stage][arm].get('gold_evidence_found',0) for r in subset) for arm in ('A','B')}
    summary['stage_gold_evidence_counts']=stage_totals
    write(OUT/'comparison.json',comparison);write(OUT/'summary.json',summary)
    trace={'question':new['rag-biz-023']['question'],'baseline':old['rag-biz-023'],
           'candidate':new['rag-biz-023'],'native_A':lines(OUT/'native_023_A.jsonl'),
           'native_B':lines(OUT/'native_023_B.jsonl'),'native_B_repeat2':lines(OUT/'native_023_B_repeat2.jsonl')}
    write(OUT/'case_023_full_trace.json',trace)
    md=['# QueryPlan 与领域适配层验证报告','',
        '结论：已实现并完成本轮验证，暂不默认开启。精确标识符与行证据保护解决了第23题，但方法型问题仍有回答回退，不能宣称其他问题完全不受影响。','',
        '## 冻结输入和运行方式','',
        f"- AI bundle：`{manifest['bundle']}`；普通全文索引 generation 不作为版本。",
        '- 50道现有主集：48道正式 reviewed，007 disputed、024 pending 仅诊断；另42道历史回归/待审问题分集合报告。Gold 未修改。',
        '- 各92次真实原生检索/重排/证据选择；184次真实模型回答+裁判，回答测试复用这两组冻结的上游候选和分数，走原生下游并逐题确认实际上下文相同。不是184次独立重新检索。',
        '- 第23题另跑完整 AnswerService 问答：A一次、B两次。不得把回放当独立全链路重复。',
        '- 第一次回放脚本并发初始化 Chroma 失败6次；已修正为串行本地准备，仅补未完成项。失败保留在原日志，不计质量失败。运行中断后只续跑未完成项。',
        '- 第一候选在36题后停止：所有代码命中统一提权导致020的必要证据被挤走。修订为每标识保护一个代表，其余保序。另修正否定和非互斥对象条件。',
        '- 本轮未调整算法去迎合后续答案分数；开关默认false。测试使用gpt-5.6-sol、现有本地BGE与发布索引。',
        f'- 结束哈希漂移：{drift or "无（运行代码、Gold）"}。','',
        '## 第23题轨迹','', '| 阶段 | 修改前正确证据名次 | 修改后正确证据名次 |','|---|---:|---:|']
    row=next(r for r in comparison if r['id']=='rag-biz-023')
    for stage,score in row['stages'].items():
        md.append(f"| {stage} | {score['A'].get('first_gold_rank') or '未入选'} | {score['B'].get('first_gold_rank') or '未入选'} |")
    md += ['', '原文：`E:/生化免疫临床知识/化学发光定标处理思路分析.pptx` 第11张，CSDB行。表格实际单元格及索引原文保存于 source_verification_023.json。',
           '实际发送行：CSDB／校准结果错误，检测到气泡／底物注入过程中，检测到气泡／底物灌注后重新定标。原父块偏移76–117，实际引文绑定S1。',
           '旧链路将其以“同一文档候选数量达到上限”排除；新链路提前保留精确代表。其他明确报警被标为 explicit_identifier_conflict；同页其他行不进入该投影。',
           '向量名次2→6，未改善；关键词21→1、融合8→1，证据选择未入选→1。不能把最终收益全部归因于向量或重排模型。',
           '', '完整原生回答对照：','', '**A：** '+trace['native_A'][0]['result']['text'],'', '**B：** '+trace['native_B'][0]['result']['text'],'',
           '## 各集合自动裁判结果','', '| 集合 | 题数 | A判consistent | B判consistent | 新增完整引文题 | 丢失完整引文题 |','|---|---:|---:|---:|---|---|']
    for name,c in cohorts.items():md.append(f"| {name} | {c['count']} | {c['A'].get('consistent',0)} | {c['B'].get('consistent',0)} | {', '.join(c['gained_questions']) or '无'} | {', '.join(c['lost_questions']) or '无'} |")
    md += ['', '这是自动裁判结果，不是人工确认正确率。same-context变化也可能受到新增QueryPlan摘要或生成/裁判波动影响，不能归因于召回改进。历史回归不是全新盲测。',
           '', '## 各阶段完整引文条数（不是答案正确率）','',
           '| 阶段 | 正式48题 A | 正式48题 B | 全92题 A（含诊断） | 全92题 B（含诊断） |',
           '|---|---:|---:|---:|---:|']
    for stage,c in stage_totals.items():md.append(f"| {stage} | {c['formal']['A']} | {c['formal']['B']} | {c['all']['A']} | {c['all']['B']} |")
    md += ['', '第23题固定候选/分数隔离：旧上游+旧选择=未入选；旧上游+新选择=入选；新上游+旧选择=未入选；新上游+新选择=入选。见case_023_selection_ablation.json。说明本题关键是最后的证据选择，而非只靠新增召回。',
           '', '## 逐题变化与审核清单','', '| 题号 | 集合 | 引文新增/丢失 | 上下文变化 | A → B |','|---|---|---|---|---|']
    for r in comparison:md.append(f"| {r['id']} | {r['cohort']} | +{len(r['gained'])}/-{len(r['lost'])} | {'是' if r['context_changed'] else '否'} | {r['answer_A']} → {r['answer_B']} |")
    md += ['', '## 采用门槛与剩余问题','',
           '- 023原文与索引行绑定、真实入口回答：通过；006/030新增引文也核对原资料与实际上下文，见 gained_evidence_source_audit.json。006另查看PDF第7页，确认CL-8000i图示与低位启动/高位停止/空瓶切换的归属。',
           '- 单元/兼容性：362 passed、1 skipped；关闭开关真实入口抽样也保持原上下文。见 pytest.xml 与 baseline_final.jsonl。',
           '- 方法型问题：021自动判consistent→incomplete，原五条完整引文仍在；新增LVC简表成为前排来源，回答侧重改变，未清楚表达超出允许范围。裁判对F3的covered较宽松，应结合详细分支审核，不能仅看一个标签。',
           '- 独立07自动判consistent→incorrect，但A/B均回答密封圈寿命66000次和同一物料编码。A判对象一致、B判整支注射器被换成密封圈，裁判口径不一致，且该题仍pending。保留原判并列为Gold/裁判争议，不将其直接算成已确认算法退化。原资料第12/27张明确提到密封圈。',
           '- 回归row_holdout2-001由拒答变为回答，但把CL-2600i资料泛化成所有免疫仪，属于实际适用范围风险。两组上下文相同，问题计划摘要有变化；不能归因于召回，也不能忽略该错误。',
           '- 034两组答案意思基本相同，自动行为评分不同，不能将该题分数改善当作明确业务收益。030两组都能回答，新增引文只记证据改善。',
           '- 019、040、050仍有原来的材料召回/完整回答缺口；本次不是它们的通用解法。',
           '- 结构化槽位只覆盖严格的规则表格。叙述、复杂表格与跨页关系仍是unknown，不能将其自动判为没有答案。',
           '- 总体是否有退化，以summary.json的regressions和逐题真实回答为准；不因总体consistent数上升忽略局部退化。',
           '- 采用结论：保留可插拔实现，暂不切换应用默认开关。下一轮应限定研究“答案目标如何决定简表与详细条件分支的取舍”，保持本轮数据冻结，不继续追加报警码例外。',
           '', '## 复核入口','',
           '- comparison.json：每题各阶段计分、真实回答、集合及变化。',
           '- case_023_full_trace.json：023完整解析、全部候选/排除原因、实际上下文和原生回答。',
           '- candidate.jsonl：所有候选的query_plan_diagnostics、selected/reason、source_spans、answer_slot_coverage。',
           '- baseline/candidate_manifest.json：索引、配置、代码与Gold哈希。',
           '- answers_A*.jsonl / answers_B*.jsonl：真实模型回答、原裁判、失败记录与上下文一致性。',
           '- manual_review.json：关键改善、回退、对象适用风险与裁判争议的逐项复核；未覆盖全部184条断言的人工复核。',
           '- ../../docs/rag_query_plan_domain_adapter_20261111.md：开发说明、兼容字段、开关和能力边界。','']
    (OUT/'report.md').write_text('\n'.join(md),encoding='utf-8')
    # A compact, readable audit file without rerank traces mixed into Gold labels.
    detail=['# 实际回答变化复核','']
    for r in comparison:
        if r['answer_A']==r['answer_B'] and not r['gained'] and not r['lost']:continue
        detail += [f"## {r['id']}：{r['question']}",'',f"自动裁判：{r['answer_A']} → {r['answer_B']}；上下文变化：{r['context_changed']}",'', '### 修改前','',r['answer_text_A'],'','### 修改后','',r['answer_text_B'],'']
    (OUT/'answer_changes.md').write_text('\n'.join(detail),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
