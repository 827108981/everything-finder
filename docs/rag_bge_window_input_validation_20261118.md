# BGE 长片段分窗口输入开发与验证（2026-11-18）

## 结论

已实现通用分窗口输入，并完成111题×3组对照。每次模型输入仍不超过512 token，完整保留问题，正文按token分窗，重叠64 token；原文、切片、索引、Gold及候选池均不改变。

**最高分汇总修复了“答案行没有进入模型输入”的问题，但不是所有题的排序都改善。** 前两窗口平均分有明确退化，不建议采用。最高分方案可作为下一步采用候选；本次是隔离实验，生产构造器默认仍为`truncate`，没有修改用户持久化设置，不能误认为桌面已启用分窗口。

## 严格边界

- 原包装W0；已发布AI包`g000033-71ec3c52f3`。
- 使用11月17日完整真实轨迹冻结的关键词、向量、融合、去重候选，重排入口仍为20条。
- 当前BGE模型、问题原文、条件检查、最终Top8、0.1最低分、12000字符预算保持一致。
- 新窗口实际调用本地FP32 BGE；仅输入token完全相同的模型结果复用缓存。缓存键包含模型输入ID与attention mask，协议冻结模型和revision。
- 共2220个原问题/片段对，新增推理709个窗口；取最高分与平均分使用同一批窗口分数。
- 固定前8对重新推理确认缓存分数，最大概率差0。
- 另外004、019、IMFH1、VPP共4题通过真实AnswerService.retrieve重新执行检索与全部窗口推理，不复用分数缓存；候选顺序、实际Context及entries全部与回放一致。
- 全部111题：正式已审核50题（46题有正向Gold）；诊断41题（40题有正向Gold，007 disputed）；来源保留20题pending。本轮不修改审核状态。
- API调用0、网络连接尝试0。不把Context证据覆盖当成答案正确率。

## 实现

`local_full_text_search/rag/reranking/local_bge.py`新增两个可注入输入模式：

- `window_max`：完整正文分窗，每个窗口重复原问题，取最大相关性分数作为原片段分数。
- `window_top2_mean`：相同窗口取最高两个分数的平均；只有一个窗口时保留其原分数。
- `truncate`：原路径，仍为默认；保留原有调用兼容性。

窗口预算扣除问题和特殊token。使用fast tokenizer的overflow/offset功能保留原文位置，不以字符数估算token，不写题号、机型或表格专用优先规则。问题本身超过输入预算时，窗口模式显式报错，不能静默截掉问题。当前111题未触发该错误。

每个候选记录输入模式、窗口数量、各窗口起止字符位置、token数、模型分数和logit。平均分的logit明确标为聚合概率换算值，不冒充某次模型输出。模型评分后继续使用原BGE基础排序规则，融合分仅作同分排序依据。

Context依然使用原候选全文及原来源；本轮没有把模型看到的窗口直接替换为Context，也没有修改最终证据选择规则。

## 三组结果

下表是**实际Context中完整Gold引文覆盖题数**。

|方案|正式46题|诊断40题|来源保留20题|正式Context标注负例数|
|---|---:|---:|---:|---:|
|原512-token截断|44/46|37/40|19/20|11|
|分窗口最高分|44/46|38/40|19/20|11|
|分窗口前两平均分|44/46|37/40|19/20|14|

三个集合分开统计，不能用最后一行两个集合的总数不变掩盖“有的恢复、有的丢失”。最高分方案没有新增已标注Gold丢失、没有新增已标注负例；不代表新增片段都已经得到业务审核。

|方案|正式Context MRR|诊断Context MRR|来源保留Context MRR|
|---|---:|---:|---:|
|原截断|0.7961|0.7107|0.6446|
|最高分|0.7943|0.7144|0.6267|
|前两平均分|0.7924|0.7411|0.5685|

MRR是每题第一条完整Gold名次倒数的平均；它不衡量完整业务流程是否齐全。最高分方案恢复一题，但多个已命中题的排名后移，因此某些集合MRR下降。

## 重点题逐条结论

|题目|原方案|最高分|前两平均分|解释|
|---|---|---|---|---|
|IMFH1，row-holdout-008|重排14、Context未命中|重排1、Context1|重排1、Context1|模型现在能读取原来位于字符876附近的目标电机行；不是提高固定题目的分数|
|VPP，source-independent-010|重排7、Context7|重排4、Context4|重排4、Context4|模型能读到原来在字符819附近的VPP行；原方案本来已保留Gold，本轮主要是排序改善|
|019，多项目DEV|Context1/2|Context1/2|Context1/2|第14张在去重后30位，仍未进入前20；本轮未改入口，不能解决该缺口|
|免疫条码，row-holdout2-001|重排2/3，条件后15/16，Context0/2|相同|相同|旧条件排序覆盖模型顺序，本轮未改条件检查|
|BS-360E/BS-430加热方式，row-holdout2-005|Context2/2、首条第2|保持|Context0/2|两个窗口分数0.99529、0.00281，平均后0.49905，正确片段掉到重排16并被Top8截掉|
|清洗水温度范围，source-independent-009|Context1/1、第1|保持|Context0/1|两个窗口0.99871、约0.00004，平均后约0.49937，正确片段掉到重排13|

前两平均分还恢复source-independent-003的指定PDF，但不能抵消它丢失source-independent-009。002、011、031新增已标注负例引文。故前两平均分明确不采用。

## 最高分方案的全部首条Gold降位题

以下为**实际Context名次**，不是丢失；具体候选可在HTML中展开。

|题号|原名次→新名次|
|---|---|
|rag-biz-014|3→4|
|rag-biz-007（disputed）|2→3|
|rag-row-holdout-010|1→2|
|rag-row-holdout2-007|7→8|
|evidence-flow-independent-03|2→3|
|vector-source-independent-001|4→5|
|vector-source-independent-004|2→3|
|vector-source-independent-011|2→4|

41题Context发生内容或顺序变化，新增40个片段、移出35个片段。完整清单为`candidate_changes.json`。

需要区分两类原因：

1. 新窗口看到正文后半段的相关内容，其他候选因此上升，原来靠前的Gold相对后移。不能把所有名次下降直接等同答案证据失效。
2. 同一机制也可能提高长片段中的局部相关内容。例如007新增CL-2000i维护片段；source-independent-001新增302340200、302340400其他报警片段，source-independent-004新增302340300片段。它们不是目标型号/报警的直接证明。候选原本已在上游池中，本轮改变排序使其进入最终Context；排序变化与原有范围检查不足要分别记录。

这里没有生成答案，不能声称模型已经误用了这些资料，也不能因为已标注负例数不变就声称没有范围风险。最高分不是“读完整后所有资料都会排对”，也没有解决模型精确标识符和适用范围判别能力。

## 验证与限制

- 原基线111/111实际Context、entries、阶段评分复现。
- 三组均111题，前4个阶段候选和评分冻结，Gold和设置等保护哈希不变。
- 4道真实入口核验全部一致；未将4题说成额外独立验收题。
- 聚焦单元/兼容测试15通过，覆盖尾部目标行、窗口重叠、短输入token一致、问题预算、取消和同文档多段兼容。
- 全量543通过、10失败、1跳过、1预期失败。10项与11月16日BGE接入记录中的source_consensus失败名称逐项一致，没有新增失败，不宣称全绿。
- HTML复用程序原生格式，三个方案、全部题、八个阶段、窗口原文及分数、实际Context对照。原资料位置注册4046项，按钮请求已核对清单；UI检查模拟打开请求，没有逐个启动4046个原资料阅读器。
- 所有新增字段只来自原问题/候选和模型输出，运行链路不读取Gold。实验中的Gold只用于运行后的匹配统计。
- 没有调整512模型位置编码，没有扩大候选入口，没有改变条件排序或分数门槛，没有重建索引。

## 本轮采用判断

1. **保留分窗口读取能力**：已经证实能解决模型看不到答案行的输入缺陷。
2. **优先考虑最高分方案**：本轮增加一题完整证据、改善VPP排序，无新增Gold丢失；仍需接受并核对8题降位和范围干扰。
3. **不采用前两窗口平均分**：其平均操作稀释了已有完整答案，产生可复现的证据丢失。
4. 本轮先交付隔离结果，默认仍为truncate。后续若采用最高分，应明确这是输入完整性的改进，不是条件检查或跨型号适用性的修复；没有为特定题补额外筛选规则。

## 文件与复现

- 实现：`local_full_text_search/rag/reranking/local_bge.py`
- 测试：`tests/rag/test_bge_window_input.py`
- 实验：`tools/validate_bge_windows_20261118.py`
- 原生报告：`tools/report_bge_windows_20261118.py`
- UI核验：`tools/check_bge_windows_report_20261118.cjs`
- 结果目录：`reports/rag_bge_windows_20261118/`

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_windows_20261118
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_windows_20261118 --native
.venv/Scripts/python.exe -X utf8 -m tools.report_bge_windows_20261118
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag/test_bge_window_input.py tests/rag/test_rerank_support.py tests/rag/test_selection_a_adoption.py -q
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag tests/test_settings_service.py tests/test_open_location.py -q
```

实验缓存和输出按冻结协议复用；若改动输入/参数，应使用新实验目录，不覆盖历史对照。`open_report.cmd`启动带原资料定位的本地报告服务。
