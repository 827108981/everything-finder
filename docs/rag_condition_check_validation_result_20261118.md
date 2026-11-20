# 条件检查开发验证结论（2026-11-18）

结论：A的排序收益已被隔离实验证实，可以独立保留为候选；B/C暂不采用，整套方案不默认开启。A不是失败的排序方案，但尚未满足“新增不适用资料不进入Context”的整体验收条件。
本轮完成实现、六组111题冻结候选对照、A全部111题真实API前入口、原资料核验、兼容测试。没有调用生成或裁判API，因此不声明答案正确率改善。
源代码默认仍为旧条件判定与旧条件排序；BGE继续使用已采用的window_max。没有改变Gold、索引、用户持久化设置、Domain Adapter默认关闭状态。

## 1. 冻结输入与顺序
AI索引：`g000033-71ec3c52f3`。67个冻结输入文件哈希无漂移。重排入口20、最终Top8、0.1门槛、12000字符预算不变。
先复现基线111/111，再A，再B/A+B，再因门禁实际阻断而运行A+B+C及A+C。初版B已知多分支阻断用例失败及其首轮结果保留在initial_B_before_unit_fix，修复短量词语法后重跑，未隐去首轮失败。

## 2. 实际结果（全部正向Gold引文进入实际Context）
|方案|正式正向46题|诊断正向40题|来源保留pending20题|
|---|---:|---:|---:|
|baseline|44/46|38/40|19/20|
|A|45/46|38/40|19/20|
|B|34/46|38/40|16/20|
|AB|35/46|37/40|16/20|
|ABC|45/46|38/40|19/20|
|AC|45/46|38/40|19/20|

完整证据覆盖不是答案准确率。没有正向Gold的题仍检查门禁/Context行为，不因无法计算Recall而忽略。正式、诊断、pending来源集不混算。

## 3. A：取消条件二次排序
- 免疫条码rag-row-holdout2-001：两条Gold的BGE排名2/3，旧条件检查降到15/16；A保留2/3，实际Context恢复两条Gold，正式完整覆盖44/46→45/46。
- 生化条码rag-acc-001：首条Gold实际Context排名7→2。
- vector-source-independent-011：首条Gold排名4→5，仍完整；来源集MRR由0.6267降到0.6242，不宣称所有排名提高。
- 没有新增Gold丢失，没有新增已标注hard_negative；19题Context文本或顺序改变，17题入选成员改变，36条新增＋36条移出。
- 72条增删都核实索引文字；原资料核验区分整段文字与逐单元格文字，不能用文字存在冒充关系正确。Word/Excel记录真实段落/单元格；旧Office使用源文件指纹对应的现有转换缓存，没有重新转换或改资料。

## 4. 不能隐瞒的A新增选段风险
|题目|新增资料|为什么需要限制|
|---|---|---|
|025：TSH/CEA标本保存|临床基础第79张校准质控品保存|校准质控品的8小时不能当成患者标本的保存期限；原文与缓存均核实|
|免疫条码/生化样本条码|BS-2600M&BS-2800M维修手册PDF第70页|页面标题明确是试剂条码参数，不能直接当样本条码或免疫平台依据；已渲染查看原页|
|independent-12：ACUSON X700|CL-8000i超声频率PDF第5页、空化电压DOCX|免疫混匀超声不是超声影像探头；这道无正向Gold的诊断拒答题基线已非空，A又更换两段错误产线资料|
|022|LVC第29张、方法总览第38张|与目标低发光现象不能自动等同；第49张多分支总结被恢复是正收益，LVC页只能按局部范围参考|
这些候选在原始召回/BGE中已存在；A改变了最终入选状态，旧条件检查没有识别正确的对象归属。不能归咎于BGE改变，也不能因为Gold未退步就称无风险；同样不能抹去A恢复正确证据的独立收益。其他泛化、无关背景及被移除片段见selection_audit.json，不能当作业务人员全部审核通过。

## 5. B：局部关系判定为什么不能采用
新增source_bound策略记录候选原文、真实偏移、关系依据、三态与原因。可避免不同属性方向冲突、重新定标动作误作故障阶段、多分支整段误删，且没有改检索查询。
但规则缺乏足够的关系覆盖：真实111题相对旧判定新增硬排除为0，不能证明已经解决复杂错对象/错机型。多数变化是MATCH→UNKNOWN。旧排序仍奖励MATCH时，020/021等必要证据会被挤出；旧最终门禁要求至少一条MATCH时，UNKNOWN又导致整批清空。
B丢失引文题（按正式/诊断/来源分组）如下：
- B / formal：rag-biz-005、rag-biz-019、rag-biz-020、rag-biz-021、rag-biz-026、rag-biz-028、rag-biz-032、rag-biz-033、rag-biz-036、rag-biz-041、rag-biz-050
- B / diagnostic：无
- B / source_holdout：vector-source-independent-001、vector-source-independent-002、vector-source-independent-012
- AB / formal：rag-biz-005、rag-biz-019、rag-biz-026、rag-biz-028、rag-biz-032、rag-biz-033、rag-biz-035、rag-biz-036、rag-biz-041、rag-biz-043、rag-biz-050
- AB / diagnostic：evidence-flow-independent-01
- AB / source_holdout：vector-source-independent-001、vector-source-independent-002、vector-source-independent-012

数量说明：019基线就缺一条Gold，B使它连已保留的场景引文也丢失；因此丢失引文题数与完整题数下降量不一定相等。

## 6. C：最终门禁消融
A+B+C与A+C恢复到A的正向覆盖，没有增加新的正向证据收益。但017/018由“超声影像诊断仪条件未确认、Context为空”变成8段/5段Context。只看46道正向题会漏掉这两项退步。
这证明未知不能一律清空，也不能一律放行所有设备/主题；本轮全面放开门禁的C不采用。没有生成答案，不能断言017/018最终必然幻觉，但已确定API前门禁行为改变且输入资料不适用。

## 7. 独立关系探针与限制
新增待审探针在运行前冻结，来源页未出现在111题重排候选中。严格筛选只找到1个可核实独立事实页，形成3条同事实对照，不凑成18题。旧判断2/3、新判断3/3，只证明该关系可判，不证明跨文档泛化。
来源为BM80A服务手册PDF第561页“整机温度偏高”；已渲染检查该文本位于维护故障表左列。相同属性相反方向、不同属性方向分别测试；全部pending，不计正式分。原定6个来源页的独立覆盖不足，明确标为coverage_incomplete。

## 8. 真实入口与测试
- A：111/111真实AnswerService.retrieve入口，真实本地编码、Chroma和BGE推理，不复用重排分数缓存。上游五阶段顺序、条件顺序、实际Context和entries全部与冻结实验一致。
- 只注入同一已发布索引对应的只读向量句柄，不修改attached指针、不重建索引。网络尝试0、API调用0。
- 全部tests/rag：修改前530通过、10失败、1跳过、1预期失败；修改后557通过、10失败、1跳过、1预期失败。10个失败名称逐项一致，均为既有source_consensus失败；不宣称全绿。
- 最终条件专项84通过、1个历史预期失败；入口/诊断相关78通过。旧local_conditions实验仍有历史xfail，没有冒充新策略通过。
- 最终Context按实际included_text核验。B若证明引文被截断，诊断降为UNKNOWN，不沿用完整父块的适用性证明。

## 9. 本轮未解决与下一步
019处理页仍在去重第30、重排入口20之外；本轮不改候选入口，不能修复。BS-800温度系数及A&T/Medica仍有上游缺口；来源003仍需核实PDF/DOCX替代证据，不能改Gold掩盖。
下一轮最小方向：保留A为固定对照，只验证“事实对象/资料范围”是否能可靠绑定；先用已核实的不同对象和多分支负例检验本地关系判断，再决定是否接入。不得继续通过多加阶段词、报警码例外或一律UNKNOWN来制造通过。没有新证据前，不继续调本轮参数。

## 10. 采用决定与代码
- A：独立排序收益成立，保留显式注入选项；未达到零新增错场景的整体验收，不自动替换默认。
- B：暂不采用。关系规则覆盖不足，且与旧排序/门禁耦合导致退步。
- C、A+B+C：暂不采用。不能用正向覆盖恢复抵消跨产线门禁放行。
- 默认业务路径：保持本轮开始时状态。无需人工补全词典；现有Gold无需修改。
生产代码修改：constraint_matcher.py（可注入保序及source_bound），condition_relations.py（局部关系和原文证明），evidence_policy.py（独立可注入门禁消融），context_builder.py（未知范围说明及截断证明校验），answer_service.py（记录实际策略开关）。所有新行为默认关闭。

## 11. 复现命令
```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_condition_check_20261118 --arms baseline A B AB ABC AC
.venv/Scripts/python.exe -X utf8 -m tools.validate_condition_check_20261118 --native A
.venv/Scripts/python.exe -X utf8 -m tools.audit_condition_check_20261118
.venv/Scripts/python.exe -X utf8 -m tools.probe_condition_relations_20261118
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag -q
.venv/Scripts/python.exe -X utf8 -m tools.report_condition_check_20261118
```
真实入口脚本拒绝覆盖已有结果；复跑应使用新报告目录或明确保存历史结果后执行，不按答案好坏择优覆盖。
