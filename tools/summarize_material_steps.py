"""Produce a human-readable report without modifying inputs or Gold."""
from collections import Counter
import html
import json
from pathlib import Path
from tools.validate_material_steps import ROOT, sha

def main():
    out=ROOT/'reports/rag_material_steps_20261105'
    summary=json.loads((out/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((out/'answer_protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in protocol['hashes'].items())
    answers=[json.loads(l) for l in (out/'answer_results.jsonl').open(encoding='utf-8')]
    assert len(answers)==18 and (out/'answer_completion.json').exists()
    cases=[json.loads(l) for l in (out/'cases.jsonl').open(encoding='utf-8')]
    grouped={}
    for r in answers:grouped.setdefault((r['step'],r['id'],r['arm']),[]).append(r)
    counts={str(k):dict(Counter((r.get('judgement') or {}).get('verdict','execution_error') for r in rs)) for k,rs in grouped.items()}
    manual={
        '023_table_flag':{'strict_gate':'flagged; not silently cleared','reason':'CSDB正证据和DCF负例位于同一页报警表，新增整页同时命中两条标注。',
                          'answer_review':'逐条阅读三次候选回答，均解释底物注入检测到气泡，并要求底物灌注后重新定标；未采用DCF或只重启软件。',
                          'conclusion':'本次未发现跨行误用；同页整块标注不能直接代表回答错误。保留原Gold和严格统计，推广前需要表格行结构验证。'},
        '040_routes':{'correction':'不能把040未改善归因于未拆问。冻结规划实际拆成两个查询。',
                      'finding':'第二分支正确页841cbed3...在两个补充查询各自前60个融合结果中均未出现，不能靠最终选材恢复。'},
        '050_equivalent_source':{'strict_gold':'指定PDF第17页仍未进入上下文，严格来源指标无增益。',
                                  'verified_source':'高速免疫场景化培训教材-光学V1.1.pptx第15张提供等价四项排查原因；已有原文/索引核验，本次源文件哈希未变，完整引文检查由缺失变为存在。'},
        'S2_scope':'仅跨页标题关联原型，5组关系从当前索引重新构建一致。未开展表格按行重建或完整章节解析，不把零收益外推到整个资料结构化方向。'}
    (out/'manual_audit.json').write_text(json.dumps(manual,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 资料环节三步独立实验（2026-11-05）','','## 结论与范围','',
        '第一步保守选材对023有稳定局部收益；第二步已有跨页关联原型单独使用无最终收益；第三步已有多诉求检索对050有稳定局部收益，但040仍未找到正确分支。三个方案均未切入生产。',
        'S1为本轮新执行的选择逻辑；S2/S3复用11月3日冻结的独立A/B检索和评分结果，本轮逐字复现上下文，没有重新运行嵌入/重排模型。另新跑18次回答及其Gold判定。不能称为三个全流程新跑，也没有新的全50题回答通过率。','',
        '当前AI bundle_id：`g000033-71ec3c52f3`。正式48题=原集合17题+新增31题；024仍pending，007仍disputed且排除；回归10题pending仅作诊断。回归题已反复使用，不是盲测验收。','',
        '## 三个独立变量','',
        '| 实验 | 唯一改变 | 本轮执行方式 |',
        '|---|---|---|',
        '| S1 最终选材 | 同文件相同正文去重；原问题明确出现且当前上下文缺失的字母标识，可从已合格候选补入空位；所有硬条件必须匹配 | 新执行59题，保持原有内容和8段/12000字预算，不替换任何非重复原文 |',
        '| S2 跨页关联 | 短故障标题与下一页原因/步骤建立有依据的关联，补候选并提供关联场景进行评分 | 独立A对baseline；从当前索引重新构建5组关系，再复现59题上下文 |',
        '| S3 多诉求检索 | 问题拆成2–3个查询，保留原查询和候选，有限补召回与子查询评分 | 独立B对baseline；冻结28/59题的拆问计划，复现59题上下文 |','',
        'S1规则没有题号、文件名或CSDB特例，也没有访问Gold；它只测试“明确标识缺失+空位+硬条件符合”这一充分条件，不能代表已实现语义覆盖检查。S2/S3均未叠加S1。','',
        '## 证据结果','',
        '| 集合（有正证据题数） | 基线命中 | S1 | S2 | S3 |',
        '|---|---:|---:|---:|---:|']
    for label,key in [('原18题固定集合，可计分17题（正证据15）','original17'),('新增集合，可计分31题（正证据29）','added31'),('回归10题，pending','regression10_pending'),('024，pending','pending024')]:
        cs=[summary[s]['cohorts'][key] for s in ('S1','S2','S3')]
        lines.append(f"| {label} | {cs[0]['before_hit']}/{cs[0]['positive_count']} | "+' | '.join(f"{c['after_hit']}/{c['positive_count']}" for c in cs)+' |')
    lines+=['','正式正证据命中：S1由39/44变为40/44；S2/S3仍39/44。平均完整证据覆盖率：基线83.3%，S1为85.6%，S2/S3仍83.3%。仅命中一段与完整覆盖不同。',
            'S1只改变023，其余58题上下文逐字不变，所有59题原有正文均保留；023从4段变5段，1395字符。S2全部59题上下文不变。S3改变23/59题，050新增经核验的等价引文，严格指定文件指标没有计入这项收益。',
            'S1新增一个标注负例页命中：023同页DCF行；没有丢失任何原正证据。S2/S3没有新增严格正证据丢失或新增标注负例。保留严格门槛：S1仍为flagged，S2/S3证据门槛通过不代表有足够推广价值。','',
            '## 本轮新跑的回答复测','',
            '| 实验/题目 | 基线答对 | 候选答对 |',
            '|---|---:|---:|']
    for step,cid in [('S1','rag-biz-023'),('S3','rag-biz-050'),('S3','rag-biz-003')]:
        vals=[sum((r.get('judgement') or {}).get('verdict')=='consistent' for r in grouped[step,cid,a]) for a in ('baseline','candidate')]
        lines.append(f'| {step}/{cid} | {vals[0]}/3 | {vals[1]}/3 |')
    errors=sum(bool(r.get('error')) for r in answers)
    citations=sum(not r.get('citation_check',{}).get('structurally_valid',False) for r in answers if r.get('answer'))
    lines+=['',f'18次新回答，执行错误{errors}次，引用结构异常{citations}次。固定上下文、同模型、每组3次，随机交错执行；Gold只传给裁判，未传给回答模型。该诊断子集不是全量指标。',
            '023候选3次均未混用DCF；050候选恢复四类排查原因并保留不能直接判坏的边界。003用于观察旧扣板例外是否因上下文变化而遗漏，本轮结果见表，不能用历史单次差异断言稳定回退。',
            '## 未解决的问题与下一步','',
            '006没有可用的缺失字母标识，S1不能保护“空瓶切换并告警”的事实；019的DEV标识已在标题中出现，S1不能识别处理步骤仍缺失。S2虽然找回019处理页，最终配额仍截掉它。040已拆问但正确第二分支页仍没有召回。',
            '建议下一轮集中验证资料结构：先让表格每行携带列名、单位、条件；再让同一场景标题与后续处理步骤成为可共同纳入的证据单元。各自先检查原文/索引一致性，再在固定预算下检索和选材。避免仅取消配额或盲目增加候选数量。',
            'S1保留为低风险候选，收益目前仅覆盖一题，尚不足以证明广泛泛化；S3不作为所有问题的默认额外调用。下一轮只对明确有多个诉求的输入测试，额外记录每个分支是否获得证据、代价及原正确答案是否受损。',
            '验收要求：逐项核实必要证据保留；相似机型/版本不得混用；原17题、增31题、待审回归分别报告；回答重复验证、代价测量及更多未参与调试的真实问题通过后再考虑上线。',
            '', '## 核验与文件','',
            '177组上下文逐字复现通过；当前生产代码、设置、Gold、源输入与元数据库哈希一致；24条已有人工可复查引文的原文件哈希及既有原文/索引核验标记通过。4个新增选择逻辑测试通过；git diff --check通过（既有CRLF提示）。没有修改生产、设置、已发布索引或Gold。',
            '`protocol.json`、`reproduction.json`、`cases.jsonl`、`summary.json`为证据回放；`answer_protocol.json`、`answer_results.jsonl`、`answer_completion.json`为18次新回答；`manual_audit.json`记录同页负例、等价来源与040拆问核查。','']
    text='\n'.join(lines)
    (out/'report.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/rag_material_steps_validation_20261105.md').write_text(text,encoding='utf-8')
    path=out/'report.html';old=path.read_text(encoding='utf-8')
    panel='<section><h2>本轮新回答诊断</h2><pre>'+html.escape(text)+'</pre>'
    for r in answers:panel+='<details><summary>'+html.escape(f"{r['step']} {r['id']} {r['arm']} repeat={r['repeat']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    panel+='</section>'
    assert '本轮新回答诊断' not in old
    path.write_text(old+panel,encoding='utf-8')
    assert path.read_text(encoding='utf-8').count('<details>')==195
    (out/'delivery_verification.json').write_text(json.dumps({'evidence_rows':len(cases),'answer_rows':len(answers),'report_case_sections':195,'hashes_unchanged':True,'execution_errors':errors,'citation_issues':citations,'answer_counts':counts},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
