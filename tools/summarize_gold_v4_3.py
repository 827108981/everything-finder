"""Summarize immutable rerun records; keep advisory judge and review separate."""
from collections import Counter
from pathlib import Path
import html
import json
from tools.rag_eval_common import ROOT,read,write,sha

OUT=ROOT/'reports/rag_gold_v4_3_retest_20261110'
CHANGED={'rag-biz-'+x for x in ('019','033','040','049')}
NOTES={
'rag-biz-001':('回答漏项','已收到完整交换机步骤，但只说“考虑交换机故障”，没有说后续可考虑更换。属于操作结尾漏答。'),
'rag-biz-006':('最终选段丢失','PDF第7页词法4、融合15、重排/条件后18，因最终TopK未送达。等价片段已支持低/高位和切瓶，真实缺口是空瓶警告。'),
'rag-biz-014':('澄清口径','回答按无标识/BM10列出两种正确流量，没有明确询问盘型。按当前Gold仍不通过；不将它说成数值答错。'),
'rag-biz-019':('重排入口前丢失；附带引用风险','第14张步骤页向量41、融合60，未入去重后30候选；模型未收到40次以上建议。引用单项目DEV片段支撑多项目结论也有适用性风险。优先认定证据缺失，不能将整题概括为知识错误。'),
'rag-biz-023':('单文档配额排除','第11张CSDB表格向量2、融合8、重排/条件后11，最终reason=同一文档候选数量达到上限。原文和索引完整，不能称为解析缺失。'),
'rag-biz-028':('Gold细节及引用归属待复核','回答已列试剂周期、底物、校准品、维护并明确不能锁定板卡；裁判因未说“结合质控”扣F2，存在必答范围过严疑点。另将BM80A文件名当机型，而正文标CL-8000i/8200i，需独立核对引用归属。保留原判，不修改Gold。'),
'rag-biz-029':('Gold细节过严疑点','回答已解释零靶值、随机波动、SD不适用及厂家上限。裁判仅因未说“通常在检测下限以下”扣分；该细节是否必要需业务复核，不应直接记为系统业务答错。'),
'rag-biz-035':('已有证据未完整使用','第50张主要原因三项已送达；回答列样本与校准但漏“质控失控”。按现有用户确认Gold属漏答。'),
'rag-biz-040':('融合丢第二分支；生成又遗漏已有分支','新问法下第16张向量44但未进入融合返回候选；第14张实际已送达。仍只回答资料不足，说明既有第二分支缺证据，也有第一分支未使用问题。不能把它全归为Gold。'),
'rag-biz-045':('超出所问且适用范围不明','答案正确否定仅凭钠低即可定件，但额外给出BS-2800M判断值和备件编码；问题未限定机型，额外内容增加误用风险。自动原判uncertain，不强行记为确定错误。'),
'rag-biz-050':('必要事实未答；等价来源适用性待核','原目标第17页词法39但未入融合。实际S5有相似四类原因却明确举例CL-2600i；S1/S8支持CL-8000i诊断和遮光/污染。不能仅因指定页缺失认定所有事实未送达，也不能直接把CL-2600i专属结构搬到CL-8000i。当前实际答案确实漏供电/零地等方向。'),
'rag-acc-006':('待审Gold范围与术语疑点','问Fab/Fc和结合位点，答案已答主要定义；Gold另要求不同抗体位点差异，并用“同一种属”而引用正文写“同一种类”。保持pending，建议先业务审核。'),
'rag-row-holdout-008':('待审题号别名争议及裁判问题','问题写IMFH1，正文写MFH1。答案明确提示二者不同再介绍MFH1；不能把这段澄清直接当成偷换对象。保留原判incorrect，复核列待核实，不直接归为模型硬错。'),
'rag-row-holdout2-001':('范围不明','问题泛指免疫仪，回答依据明确来自CL-2600i系列；需界定适用范围，pending不并入正式分数。'),
'rag-row-holdout2-007':('来源适用性待核','当前未回答阻值；历史审计已发现来源的机型归属不足，不能为提高通过率直接搬用1115Ω。保持pending。'),
'evidence-flow-independent-04':('额外引用适用性待核','S1直接支持A&T两种条件分支，答案核心正确；同时引用S2/S3的BS系列资料，裁判对其适用性不确定。不能说成两分支答错。'),
}

def main():
    done=read(OUT/'completion.json'); assert done['total']==99
    rows=[json.loads(l) for l in (OUT/'runs.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==99 and len({(r['id'],r['repeat']) for r in rows})==99
    first=[r for r in rows if r['repeat']==1]
    prior=[json.loads(l) for l in (ROOT/'reports/rag_evidence_answer_native_recheck_20261109/e2e_runs.jsonl').read_text(encoding='utf8').splitlines()]
    old={r['id']:r['arms']['A'] for r in prior if r['repeat']==1 and r['cohort']=='formal'}
    def verdict(r):return r.get('judgement',{}).get('verdict','execution_error')
    cohorts={c:dict(Counter(verdict(r) for r in first if r['cohort']==c)) for c in sorted({r['cohort'] for r in first})}
    modified={i:[{'repeat':r['repeat'],'verdict':verdict(r),'findings':r.get('judgement',{}).get('findings',[])} for r in sorted(rows,key=lambda x:x['repeat']) if r['id']==i] for i in sorted(CHANGED)}
    fixed=[r for r in first if r['cohort']=='formal' and r['id'] not in CHANGED]
    delta=[{'id':r['id'],'old':old[r['id']]['judgement']['verdict'],'new':verdict(r),
        'same_context':old[r['id']]['context']==r['context']} for r in fixed if old[r['id']]['judgement']['verdict']!=verdict(r)]
    summary={'cohorts_first_round':cohorts,'modified_three_runs':modified,
        'unchanged44':{'count':len(fixed),'old_consistent':sum(old[r['id']]['judgement']['verdict']=='consistent' for r in fixed),
            'new_consistent':sum(verdict(r)=='consistent' for r in fixed),'changed_verdicts':delta,
            'same_context_count':sum(old[r['id']]['context']==r['context'] for r in fixed)},
        'formal48_first_consistent':sum(verdict(r)=='consistent' for r in first if r['cohort']=='formal'),
        'formal_excluded':['rag-biz-007','rag-biz-024'],'total_runs':99,'pending_diagnostic_first_cases':43,
        'source_index_verified_quotes':5,'unit_checks_passed':44,'completion':done,
        'warning':'自动判分不是人工正确率；本轮仅改Gold，不能归因为算法收益。'}
    write(OUT/'summary.json',summary)
    reviews=[]
    for r in first:
        if verdict(r)!='consistent':
            layer,reason=NOTES.get(r['id'],('待核实','自动未通过，尚需业务或引文复核。'))
            reviews.append({'id':r['id'],'raw_verdict':verdict(r),'layer':layer,'review':reason,'review_status':'待核实' if '待核' in layer or '疑点' in layer or '口径' in layer else '失败',
                'source':'runs.jsonl','repeat':1,'override_applied':False})
    write(OUT/'failure_review.json',reviews)
    lines=['# Gold v4.3 修订与完整重测报告\n\n',
        '本轮完成019、033、040、049修订以及99次真实问答。只改Gold与实验工具，未采用冻结的算法候选。正式集50题中48题reviewed计分，007 disputed与024 pending排除；024另作诊断。三份历史回归集各10题、历史独立集12题均pending，分别报告。\n\n',
        '## 输入与校验\n\n',
        '- 发布AI索引：g000033-71ec3c52f3；模型gpt-5.6-sol。设置和共同生产代码哈希与11月9日原生基线一致。\n',
        '- 新JSON与JSONL逐条完全一致，46题逐字段不变；新旧所有review_status一致。正式48题加载器校验通过。\n',
        '- 四题共5条引文在原资料和当前索引中均定位成功。049第35页重新查看原始PDF画面；019/033/040重读原生幻灯片，040结合用户已提供的截图核对。\n',
        '- 44项现有加载、裁判和来源绑定测试通过。原始判分完整保留，不用复核意见覆盖自动结果。\n',
        '- 进程在已保存69条结果后中断；检查无存活进程并核对冻结输入后，只补跑30条未完成任务。resume_manifest记录了只增加断点续跑能力的工具变更。已保存结果不重跑、不择优。\n',
        '- 首轮91题；四个修订题各额外两轮，共99次。未按耗时评价。原文/索引核验是标注有效性检查，实际引用支持另外核查。\n\n',
        '## 首轮自动判分\n\n|集合|数量|一致|其余原判|\n|---|---:|---:|---|\n']
    for c,counts in cohorts.items():
        lines.append(f"|{c}|{sum(counts.values())}|{counts.get('consistent',0)}|{json.dumps({k:v for k,v in counts.items() if k!='consistent'},ensure_ascii=False)}|\n")
    lines+=['\n以上为自动裁判一致率，不等于人工确认正确率。四题重复不扩充首轮正式分母。\n\n## 四题修订结果\n\n|题|修订|三轮原判|结论|\n|---|---|---|---|\n']
    desc={'rag-biz-019':'四项core，保留建议40次以上','rag-biz-033':'按范围、方法学、复测趋势、磁珠检查分步','rag-biz-040':'只问两种现象各优先检查哪个部件','rag-biz-049':'只问备份，按串口/网口分支answer'}
    for i,rr in modified.items():
        lines.append(f"|{i}|{desc[i]}|{'、'.join(x['verdict'] for x in rr)}|{'三轮均通过；反映题面/口径修正后的表现，不是算法收益' if all(x['verdict']=='consistent' for x in rr) else NOTES.get(i,('','仍有缺口'))[1]}|\n")
    u=summary['unchanged44']
    lines += [f"\n## 同口径的44题对照\n\n11月9日A臂首轮一致 {u['old_consistent']}/44；本轮 {u['new_consistent']}/44。{u['same_context_count']}/44 的实际上下文逐字相同。四个修订题不在此对照中。没有算法变动，差异属于重复生成/判分波动，不能称为系统进步或退步。\n\n|题|旧原判|本轮原判|上下文相同|\n|---|---|---|---|\n"]
    for x in delta:lines.append(f"|{x['id']}|{x['old']}|{x['new']}|{x['same_context']}|\n")
    lines+=['\n## 当前问题逐题复核\n\n|题|自动原判|定位与复核|\n|---|---|---|\n']
    for r in reviews: lines.append(f"|{r['id']}|{r['raw_verdict']}|{r['layer']}：{r['review']}|\n")
    lines+=['\n## 优先级和结论\n\n',
        '1. 先修证据送达：006和023已有正确候选却被最终选段丢弃；019和040在融合/重排入口前丢失必要步骤或条件分支。应分别定位，不能笼统归为切片或重排模型不行。\n',
        '2. 同时保留生成问题：040已收到样本针方向仍整体拒答；035收到主要原因仍漏答。资料送达是必要条件，但不保证完整回答。\n',
        '3. 评分仍需复核：028/029的附带细节，回归题的IMFH1/MFH1、机型范围和术语问题。保持历史标签与自动原判，待业务确认后再修订，不能当场放宽Gold制造涨分。\n',
        '4. 建议采用本轮用户授权的Gold v4.3修订作为后续实验口径；暂不据此采用任何算法候选。应用默认测试集保持原设置。新结果没有证明生产算法变好。\n\n',
        '## 交付索引\n\n',
        '- 新Gold：tests/rag/fixtures/rag_gold_v4_3_review.json、rag_gold_v4_3.jsonl、rag_gold_v4_3.md。\n',
        '- runs.jsonl：99次实际回答、完整候选链、实际上下文、原判和来源核验。\n',
        '- report.html：可展开逐题回答、Gold要点、裁判检查项和实际证据。\n',
        '- changes.json、dataset_validation.json、source_audit.json：版本差异和标注核验。\n',
        '- protocol.json、resume_manifest.json、completion.json、pytest.xml：冻结输入、恢复记录、结束校验和测试。\n',
        '- failure_review.json：逐题复核意见，与Gold和自动裁判分开保存。\n']
    (OUT/'report.md').write_text(''.join(lines),encoding='utf8')
    gold={c['id']:c for c in read(ROOT/'tests/rag/fixtures/rag_gold_v4_3_review.json')['cases']}
    esc=lambda s:html.escape(str(s))
    h=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>Gold v4.3重测</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;color:#192536}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:16px}summary{cursor:pointer;padding:12px}details{border:1px solid #ccd4dd;margin:12px 0;padding:8px}h1{font-size:26px}</style><h1>Gold v4.3 修订后重测</h1><p>99次执行。正式48题、其他pending诊断分开。自动原判不是人工正确率；本轮未修改算法。</p><pre>',esc(json.dumps(summary,ensure_ascii=False,indent=2)),'</pre>']
    for r in sorted(rows,key=lambda r:(r['cohort'],r['id'],r['repeat'])):
        h += [f"<details><summary>{esc(r['id'])} · 第{r['repeat']}次 · {esc(verdict(r))} · {esc(r['question'])}</summary>",'<p>实际回答</p><pre>',esc(r.get('answer',{}).get('text','无回答')),'</pre>']
        if r['id'] in gold:h+=['<p>Gold必答要点</p><pre>',esc('\n'.join(gold[r['id']]['required_facts'])),'</pre>']
        if r['id'] in NOTES:h+=['<p>复核意见：',esc(NOTES[r['id']][1]),'</p>']
        for title,key in [('自动裁判原判','judgement'),('实际送达上下文','context'),('来源核验','source_audit')]:
            value=r.get(key); h+=['<details><summary>',title,'</summary><pre>',esc(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)),'</pre></details>']
        h+=['</details>']
    h+=['</html>'];(OUT/'report.html').write_text(''.join(h),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
