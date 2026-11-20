"""Readable release report with provenance, regression attribution and limits."""
from collections import Counter
from tools.rag_eval_common import ROOT,read,lines,write
from tools.summarize_adapter_robustness import aggregate,STAGES
OUT=ROOT/'reports/rag_adapter_robustness_20261111'
OLD=ROOT/'reports/rag_query_plan_adapter_20261111'

def main():
    summary=read(OUT/'summary_release.json');comparison=read(OUT/'comparison_release.json')
    release=read(OUT/'release_validation_manifest.json')
    native={r['id']:r for r in lines(OUT/'native_release.jsonl')}
    old={r['id']:r for r in lines(OLD/'candidate.jsonl')}
    baseline={r['id']:r for r in lines(OLD/'baseline.jsonl')}
    answers={(r['id'],r['arm']):r for r in lines(OUT/'final_answers_AB.jsonl')}
    ablation=read(OUT/'selection_ablation_release.json')
    stages=[]
    for stage in STAGES:
        for name,rows in [('关闭',baseline),('上一版',old),('本版',native)]:
            s=aggregate([r for r in rows.values() if r['cohort']=='formal'],stage)
            stages.append((stage,name,s))
    md=['# QueryPlan Domain Adapter 稳健性开发与验证报告','',
        '**结论：本轮有局部收益，但未达到默认开启条件。保留可选实现，默认仍关闭。**',
        'CSDB完整证据、LVC条件分支、CL-2600i范围限定得到改善；跨型号的附加参数、Gold/裁判对象歧义及既有证据缺口仍在。总体自动一致数不是通过标准。','',
        '## 1. 验证范围与冻结输入','',
        f"- AI bundle：`{summary['bundle']}`；未把普通全文索引generation当版本。",
        '- 现有主集50题，48 reviewed正式题；007 disputed、024 pending只诊断。另10验收、10行级回归、10第二行级回归、12独立待审题，共92题。不同集合分开计分，历史回归不称新盲测。',
        '- Gold、索引、生成模型、原重排模型、原检索阈值、应用默认设置不改。工作区原有未提交改动保留；没有提交或发布。',
        '- 上一版92题真实检索及历史答案保留；本轮v2跑92题真实检索、184次配对实际生成+裁判，另做5题完整原生问答。',
        '- 代码复核后修正通用方法识别和投影范围提示泄漏；中间native_final在12题后停止并保留，不与正式结果拼接。最终native_release再次完整跑92题真实检索。',
        f"- 最终答案集合：A/B各92条。B有{len(release['fresh_generation_ids'])}题新调用；{release['reused_identical_B']}题经实际Context和生成摘要完全一致核验后复用本轮已保存答案。不是184次额外全链路调用。",
        f"- 最终改变生成输入的题：{', '.join(release['changed_input_ids']) or '无'}；额外复核题也保留原始结果。",
        '- 没有为了质量变好重试同一题；有固定Context消融和独立原生复核，其样本全部保留。没有将不同运行混成一次随机重复的平均值。',
        f"- 最终运行代码漂移：{summary['code_drift']}；Gold漂移：{summary['gold_drift']}。默认与用户持久化开关检查见defaults_check.json。",'',
        '## 2. 修改了哪一层','',
        '| 层 | 具体变化 | 不解决什么 |','|---|---|---|',
        '| QueryPlan/配置 | 激活与识别分开，低置信度仅保留；范围/实体/方法问项可诊断 | 不把解析字段当事实或完整业务知识库 |',
        '| 检索/融合 | 仅激活后追加既有精确/子问项路由；最多2个高置信度精确代表 | 没有更换BGE模型或重写BM25/RRF |',
        '| 证据选择 | 方法题保持原序，完整流程代表优先于简表；显式型号/版本限定才允许冲突 | 不能保证TopK内总有完整业务流程 |',
        '| Context | 完整表头、范围前缀和当前行，范围提示也按保留区间过滤；不截半行 | 不规则表格、跨页箭头仍不猜列/关系 |',
        '| 生成 | 来源范围、部件层级、条件分支约束；不强制方法题追问 | 提示约束不是逐条断言的语义硬校验 |',
        '| 日志/验证 | 新字段、候选排除、保护、实际槽位和范围观察、配对请求等价性 | 日志出现某词不代表语义答案正确 |','',
        '完整逐文件清单及新增字段见 [开发说明](../../docs/rag_adapter_robustness_development_20261111.md)，开发前八项分析见 [分析文档](../../docs/rag_adapter_robustness_analysis_20261111.md)。',
        '通用代码不包含测试ID、正确来源页或报警答案目录；化学免疫词汇和可调整门槛集中在domain_configs/chemiluminescence.json。','',
        '## 3. 各集合自动裁判结果','',
        '下表A=关闭、B=最终本版。它是实际输出的自动标签，不是业务审核通过率。','',
        '| 集合 | 数量 | A consistent | B consistent | B其他标签 |','|---|---:|---:|---:|---|']
    for name,c in summary['cohorts'].items():
        md.append(f"| {name} | {c['count']} | {c['answers']['A'].get('consistent',0)} | {c['answers']['B'].get('consistent',0)} | { {k:v for k,v in c['answers']['B'].items() if k!='consistent'} } |")
    md+=['','**43题没有激活Adapter，词法/向量/融合/重排顺序、Context和实际生成请求全部与关闭组相同。** 因而其中的012/034分数改善及028/043/独立05分数下降，不能归因于这次代码。见inactive_equivalence.json。',
        '历史上一版正式集自动consistent为42/48，其当时关闭基线为36/48；本轮关闭基线自身也有波动。因此不能把历史36→42与本轮数字直接解释成算法增益下降或上升，必须看配对请求、具体断言和回退原因。',
        '关闭开关另做10题真实入口复查，实际Context全部与历史关闭基线一致。单元测试也覆盖无配置、坏路径、普通索引问题、未知标识符。','',
        '## 4. 正式48题逐阶段指标','',
        'Recall@k为完整Gold引文的微平均召回率；MRR为有正向Gold题目的首个完整引文名次倒数宏平均。没有正向证据的拒答题不计召回分母。完整引文严格绑定来源和实际文本，不按同页即命中。','',
        '| 阶段 | 版本 | Recall@1 | Recall@3 | Recall@5 | MRR | 完整引文 |','|---|---|---:|---:|---:|---:|---:|']
    for stage,name,s in stages:
        md.append(f"| {stage} | {name} | {s['Recall@1']:.3f} | {s['Recall@3']:.3f} | {s['Recall@5']:.3f} | {s['MRR']:.3f} | {s['complete_quotes_retained']}/{s['gold_quotes']} |")
    md+=['','其他集合的同样指标、全部Gold保留题数见summary_release.json。不能用只增加问题数后的总分判断算法进步。','',
        f"本版范围观察词保留 {summary['scope_mentions_retained']}/{summary['scope_mentions']}；被选中的严格结构化行完整保留 {summary['complete_structured_rows']}/{summary['selected_structured_rows']}。这不是所有复杂表格命中率，也不是语义范围正确率。未找到适用范围的正文仍是unknown。",'',
        '## 5. 第23题完整轨迹','', '| 阶段 | 关闭 | 上一版Adapter | 本版 |','|---|---:|---:|---:|']
    for stage in STAGES:
        values=[rows['rag-biz-023']['scores'][stage]['first_gold_rank'] for rows in (baseline,old,native)]
        md.append('| '+stage+' | '+' | '.join(str(v) if v else '未入选' for v in values)+' |')
    md+=['','原文件 `E:/生化免疫临床知识/化学发光定标处理思路分析.pptx` 第11张。原表单元格：CSDB／校准结果错误，检测到气泡／底物注入过程中，检测到气泡／底物灌注后重新定标。源PPT单元格、索引正文和Context区间核验保留在original_source_audit.json、case_023_full_trace_release.json。',
        '', '最终实际答案：','',answers[('rag-biz-023','B')]['answer']['text'],'',
        '候选排除与保护的逐条记录在native_release.jsonl的candidate_stages.evidence及events；不是只展示通过的候选。', '',
        '## 6. 重点结果与原因','',
        '| 题目 | 本轮证据/回答结论 | 原因与采用限制 |','|---|---|---|',
        '| 023 | 保留CSDB完整行，含义、原因、灌注和重定标可答 | 通用标识符/行保护奏效；不把DEV/DUP处理移植到CSDB |',
        '| 021 | 超出允许范围、先区分前次/本次、延期与更换分支均可答 | 方法与简表取舍改善；固定Context契约消融也观察到补齐条件，但样本小 |',
        '| 020 | 仍能回答试剂/校准品检查与后续处理 | 最终补上纯处理问法的method识别；没有只保留DEV术语 |',
        '| row_holdout2-001 | 实际答案限定CL-2600i系列，不推广所有免疫仪 | 原文件范围有据；裁判有时反而把范围限定判成对象替换，判分待裁决 |',
        '| independent-07 | 明确密封圈66000次和物料号，不声称整套注射器寿命 | 原PPT12/27张支持；pending Gold写成注射器，保留冲突待业务审核 |',
        '| 030 | 比上一版少一条Gold引文；观察到的答案仍正确 | SD是低置信度未知标识，退回原检索；不为此题单独加白名单 |',
        '| row_holdout-001 | 仍存在错误附加范围 | 正确DC24V之外，引用BS-2600M/BS-2800M的22.8–25.2V来回答BS-2800；范围提示没拦住 |',
        '| 019/040 | 原有证据或回答完整性问题仍在 | 本轮没把缺失材料流程变成可用答案；不宣称通用解决全部资料问题 |','',
        f"最终自动回退清单（相对关闭）：{', '.join(summary['answer_regressions_vs_disabled']) or '无'}。其中未激活题属于相同请求波动，不能算新增算法造成的确定退化。",
        f"最终自动新增consistent：{', '.join(summary['answer_gains_vs_disabled']) or '无'}。其中012/034等相同请求变化不算确定算法收益。",
        f"完整引文相对上一版：新增题{summary['evidence_gains_vs_previous']}；丢失题{summary['evidence_losses_vs_previous']}。逐条见comparison_release.json。",'',
        '## 7. 隔离实验与归因','',
        '| 冻结约束后候选/分数 | 完整Gold引文条数（92题含待审） |','|---|---:|']
    for arm in ('old_selection','new_selection','new_context','combined'):
        md.append(f"| {arm} | {sum(r['arms'][arm]['scores']['gold_evidence_found'] for r in ablation)} |")
    md+=['','旧Context在92/92题完全复现。单独Context改动没有增加Gold引文；选择改动因门槛收紧丢030一条。最终真实全链路021多一条来自上游排序/候选组合改变，不能全部归为Context补充。',
        '另对020/021/023/条码/独立07/电压/028共7题，固定实际Context，只切换生成约束，共14次实际生成+判分。保留with/without两侧原结果在contract_ablation.jsonl。021从漏条件到完整；电压题两侧都跨型号；条码题实际已限定范围但裁判不稳定。028未激活，两个请求相同，标签不同说明生成/裁判波动。',
        '该消融基于v2冻结Context，最终投影修正后的关键题另做实际回答复核。不将两个不同Context的分数拼成同一个隔离实验。','',
        '## 8. 上线建议与下一轮最小任务','',
        '- **可以上线的部分**：新增诊断字段、配置校验、未激活兼容回退和测试工具可随默认关闭版本交付；不表示允许全量打开业务增强。',
        '- **需要灰度的部分**：完整规则表格行保护、有限精确代表、方法条件保留。仅在显式开关下对真实问题观察，继续记录引用和局部范围错误。',
        '- **暂时不能上线的部分**：全量默认启用；将当前范围提示当成可靠的跨型号硬保证；将自动consistent数提升直接作为发布依据。',
        '- **必须由业务人员确认的部分**：独立07的更换对象Gold；允许“用户没给型号时给出来源限定答案”的裁判标准；争议范围关系。审核结果不得由开发自行改成reviewed。',
        '', '下一轮只处理一个共同缺陷：**回答中的附加断言没有逐条绑定来源适用范围**。先从“正确电压+错误跨型号范围”这样的已定位错误入手，建立来源标题+正文局部条件记录，再核对每个实际引用的主张；只删除/限定不受支持的附加结论，保留已支持的核心答案。不要再扩大词典或按报警码加规则。',
        '验证必须含：跨型号同数值、版本边界、子件/整件、合法限定范围、普通问题不介入；用固定Context验证该层后，再检查原集合与独立集合。通过门槛是关键范围错误消失且无条件/分支损失，不是仅看总分。','',
        '## 9. 检查清单与复核入口','',
        '| 检查 | 结论 |','|---|---|',
        '| 单元/兼容 | 383 passed，1 skipped；见pytest_release.xml。跳过项是需环境变量开启的模型下载集成测试；真实发布模型已在92题原生检索中运行 |',
        '| 默认关闭与普通问题 | 10题关闭入口Context一致；43题未激活全部请求一致 |',
        '| 最终真实检索 | 92题完成，代码/Gold无漂移 |',
        '| 最终实际答案 | A/B各92条有来源记录；改变请求重新调用，完全一致才复用 |',
        '| 同页误命中 | 计分实际引文；范围提示仅来自实际保留区间 |',
        '| 跨型号语义保证 | 未通过，电压题仍有反例 |',
        '| 业务Gold争议 | 未决，独立07与条码裁判需确认 |','',
        '- `answer_review_release.md`：92题问题、A/B完整真实回答及对象/引用判分。',
        '- `comparison_release.json`：逐题证据损益、激活情况和三方标签。',
        '- `summary_release.json`：所有分集合/分阶段指标，定义和限制。',
        '- `native_release.jsonl`：QueryPlan、每层候选、排除原因、实际Context、来源区间。',
        '- `manual_findings.json`：代码与源文审阅判断，明确不是业务审批。',
        '- `release_validation_manifest.json`、`release_input_delta.json`：最终答案复用与重新运行依据。',
        '- `revision_log.json`：全部阶段、停止原因与未混合历史记录。','']
    (OUT/'report.md').write_text('\n'.join(md),'utf-8')
    print('report written',OUT/'report.md')

if __name__=='__main__':main()
