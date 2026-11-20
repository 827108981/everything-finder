# 条件过滤移除与最简证据选段：开发验证结论（2026-11-18）

**确定结论：已按用户要求切换默认链路，取消条件删除与二次排序，最终证据只保留BGE顺序、合并重复正文并按完整片段和字符预算装入。原有Gold没有丢失，但相对A没有新增正式题收益，扩大选段带来了更多相关但不能直接作答的背景，以及明确不同场景的资料。不能宣布整体答案更正确。**

本轮全部工作为API前验证；生成API、裁判API、密钥读取和检索模型执行过程的网络尝试均为0。报告另使用本机HTTP服务验证资料导航。结果不等于答案正确率，也没有验证LLM能否正确忽略不适用背景。

## 1. 实际实现与边界

- AnswerService默认使用SimpleEvidencePolicy。持久化旧开关即使为true也不能重新启用条件过滤；不修改用户settings.json。
- 不执行ConstraintMatcher，不按固定业务词删除或重排，不执行条件未确认清空及基于候选阶段词的追问。原始问题和解析词仍供召回、BGE及诊断使用。
- 不使用0.1重排门槛、0.35向量门槛、最终Top8或单文件配额。上游重排入口Top20保持不变，并非无限全库装入。
- ContextBuilder合并相同正文并保留来源；按原BGE顺序选入完整片段，来源标题和分隔符也计入12000字符预算。超长完整片段跳过并继续看后续片段，不截断操作、数值或表格。
- 保留已有父块正文及其表头/条件，没有另加邻页召回、表格推理、摘要模型或Gold驱动补文。原块已缺表头的情况不会凭此修复。
- 已关闭Domain Adapter保持关闭；未修改索引、Gold、源资料、召回模型和BGE窗口最高分。旧EvidencePolicy/ConstraintMatcher仅保留显式注入，以复现历史实验。

## 2. 固定输入与规模

- 已发布AI索引：`g000033-71ec3c52f3`。67个冻结输入哈希无漂移。
- 采用当前active_suite_v5_1中的49+10+10+10+12=91题，以及20题来源待审集，总计111；不混入v1/v2/v3历史修订重复计数，rag-biz-040仍删除。
- 按审核状态：50题reviewed（其中46有正向Gold、4无正向Gold）、40题pending及1题disputed组成诊断集合（40有正向Gold、1无正向Gold），另20题source_holdout全部pending；正式与待审分别报告。
- 先复现旧默认与A，222次逐题Context、entries和八阶段分数与历史完全一致；新方案与预算消融固定相同五个上游阶段。
- 简化方案111/111真实AnswerService.retrieve运行：真实本地编码、Chroma与BGE，不复用重排分数缓存；只绑定同一索引的只读向量句柄，未修改active/attached指针。实际Context与entries逐题等于冻结候选实验。

## 3. 完整Gold引文进入实际Context

|方案|正式正向46题|诊断正向40题|来源待审20题|
|---|---:|---:|---:|
|修改前旧默认|44/46|38/40|19/20|
|A：仅取消条件二次排序|45/46|38/40|19/20|
|移除过滤/阈值/Top8，旧截断预算|45/46|38/40|20/20|
|修改后：完整片段预算|45/46|38/40|20/20|

正式Context MRR：旧默认0.7943→A及简化0.8051；来源集0.6267→A0.6242→简化0.6292。没有正向Gold的5题不进入Recall/MRR分母，但仍审查实际输入资料。

## 4. 具体改善、未改善与变化归因

- 免疫条码rag-row-holdout2-001：恢复两条Gold，BGE第2/3→Context第2/3。这项收益已经由A获得，不能再归因于删除条件过滤。
- vector-source-independent-003：CL-8000i超声空化电压高报警机制，Gold在BGE第10。取消Top8后进入Context第10，待审来源集19/20→20/20。原PDF第1页可定位引文，尚不计正式分。
- 相对旧默认和A均无Gold引文丢失。完整片段预算相对旧截断预算未增减任何Gold，46处部分截断降为0；它保住的是原片段完整性，不是新增答案事实。

|顺序隔离步骤|新增Gold题|丢失Gold题|新增已标注负例引文|
|---|---|---|---:|
|A → remove_conditions_only|无|无|0|
|remove_conditions_only → remove_conditions_and_top8|vector-source-independent-003|无|6|
|remove_conditions_and_top8 → remove_conditions_top8_thresholds|无|无|3|
|remove_conditions_top8_thresholds → complete_chunk_budget|无|无|0|

这组链式对照是固定候选的顺序消融，不能将每一步效应当成所有组合下都独立成立。尤其“无新增标注负例”不代表没有新增错场景：仅删条件时，017/018原门禁仍会失效。

## 5. 新增背景与错误场景，不能只看正向分

全部111题的已标注负例引文：旧默认13→A 14→简化23。相对旧默认新增10条、涉及9题；相对A新增9条。负例被召回/纳入不等于已经在答案中误用。

|题目|新增资料的具体问题|原来未纳入的直接原因|
|---|---|---|
|002|主控板D107，不能代替功率驱动板D63/D64|BGE第9，Top8|
|011|CL-6000i入口，没有CL-8000i的AD阈值|BGE第11，Top8|
|018|免疫仪超声混匀板，不能回答影像探头功率|低于0.1；整组设备未确认门禁也被移除|
|029|普通Westgard规则不能机械套到零靶值阴性质控|BGE第10，Top8|
|031|L-J图失控限讨论，可作背景但不是所问规则的唯一依据|BGE第9，Top8|
|034|复测操作不能作为感染确诊、治疗依据|低于0.1|
|045|ISE斜率超限多原因、特定模块钠电极编码，不能直接确诊故障件|一条旧方向冲突；另一条已被A放入|
|acc-005|白帽识别分支不能替代已知白板血清污染分支|BGE第18，Top8|
|acc-006|抗原表位不是抗体Fab/Fc定义|低于0.1|

- 拒答预期017/018的Context由0段变为20段，内容来自免疫设备，说明默认链路已失去原先的设备门禁。不是“已证实LLM答错”，而是“API前确实获得不同产线资料”。
- independent-12的影像设备题旧默认已有8段不适用内容，新方案17段；属于旧问题暴露/扩展，不能算全部新增错误。
- 025的校准质控品保存与患者标本保存、条码题中的试剂/样本条码等既有范围混入仍然存在；没有通过修改Gold消除这些风险。
- 2043条实际Context正文全部与当前索引代表块逐字一致，346条附加来源均为相同或排版等价正文（68处仅排版不同）；没有将完整父块冒充预算后的引文。新增负例及恢复Gold共13条原文件位置另存original_source_checks.json。PPTX仅作原XML位置检查，不冒充所有表格布局均已人工审核。

## 6. 仍缺完整证据的三题，最早失败位置

|题目|缺失证据与位置|结论|
|---|---|---|
|019 多项目DEV原因与解决|场景Gold在Context第7，原因处理Gold去重第30，在重排入口20之外|重排入口前截断；不是本轮条件删除导致|
|row-holdout2-007 BS-800温度传感器|阻值证据Context第8；另一条Gold去重第24，未进入BGE20|上游候选入口不足|
|independent-05 A&T/Medica ISE原理与识别|已保留2/3条Gold；缺失第一条没有进入关键词/向量候选|召回或来源定位缺口，需要另立实验核实，不能靠末端选段补回|

## 7. 测试与执行核对

- 修改前：557 passed，10 failed，1 skipped，1 xfailed。修改后：568 passed，仍是相同10 failed，1 skipped，1 xfailed；新增失败0。
- 10个已有失败全部来自tests/rag/test_source_consensus.py，名称逐项相同，不宣称全套单元测试全绿。
- 简化选段、入口及历史策略显式注入专项31 passed；设置兼容8 passed。旧低分拒答、条件排除测试改为显式注入旧策略，保留历史行为测试；新增默认入口测试证明旧开关不能启用条件删除。未改业务Gold。
- 诊断与评测界面说明同步删除旧默认条件门禁、分数门槛和TopK描述，明确历史运行仍按当次记录解释；相关界面测试5 passed。
- 实际入口111/111上游顺序、实际Context、entries一致，0 API、0网络尝试、0密钥读取。所有实际Context≤12000字符、所有条目未截断。
- report.html沿用应用原报告格式，四方案、八阶段、逐条Gold排名及实际Context前后对照；双击open_report.cmd可点按钮打开原文件对应页/幻灯片。

- 原资料导航另经真实本机HTTP服务验证：PDF第9页与PPT第13张的文件、页码/幻灯片载荷绑定一致；测试替换桌面打开动作，未自动启动用户的PDF/PPT应用。

## 8. 采用状态与下一步

**按用户要求已采用简化默认链路；不自动恢复固定条件规则。架构变简单并保留原Gold，但本轮没有证据证明所有候选尽量填满预算能提升答案正确性。**
下一步最小问题是“怎样控制低相关尾部进入Context”，应与条件解析分开：固定当前原始候选与完整片段预算，仅对比保留/取消经过校验的相关性门槛，报告正确引文保留与错对象背景，不再追加报警、项目、阶段词例外。019和BS-800另在上游入口诊断，不把它们混入末端选段收益。
本轮没有必须修改Gold的业务裁决；031等负例属于相关背景而非绝对禁入资料，应继续区分“出现”与“误用”。如需判定最终回答是否更好，需要另行生成和引用核验；本轮遵循API前范围未执行。

## 9. 文件与复现命令

运行时代码：answer_service.py、evidence_policy.py、context_builder.py、config/defaults.py。历史条件模块保留作对照，不再被默认执行。诊断说明同步修改ui/rag_diagnostic_dialog.py与ui/rag_eval_page.py。
测试：test_simple_evidence_selection.py新增；test_rag_end_to_end.py和test_selection_a_adoption.py同步默认行为及显式历史策略。

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_simple_selection_20261118
.venv/Scripts/python.exe -X utf8 -m tools.attribute_simple_selection_20261118
.venv/Scripts/python.exe -X utf8 -m tools.validate_simple_selection_20261118 --native
.venv/Scripts/python.exe -X utf8 -m tools.audit_simple_selection_20261118
.venv/Scripts/python.exe -X utf8 -m tools.review_simple_selection_sources_20261118
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag -q
.venv/Scripts/python.exe -X utf8 -m tools.finalize_simple_selection_20261118
.venv/Scripts/python.exe -X utf8 -m tools.report_simple_selection_20261118
```

真实入口脚本拒绝覆盖历史运行；复跑请使用新的报告目录并重新冻结输入。before_code保存本次任务开始前的四个运行时文件，仅用于复核，非Git回退。
