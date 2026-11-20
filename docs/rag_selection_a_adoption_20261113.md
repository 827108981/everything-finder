# A选段方案正式接入与剩余问题（2026-11-13）

## 当前决定

按用户授权采用A：取消同一文档最多3段的硬限制，保留原始候选顺序、相关性门槛、条件门禁、总TopK8及12000字符Context预算。不启用B的MMR，也不启用C的取消总段数限制。Domain Adapter和其他已回退增强仍默认关闭；用户持久化设置及Gold未修改。

应用需要重新启动或重新创建AnswerService才能使用更新后的实例。

## 实现

- `local_full_text_search/rag/evidence_policy.py`：允许`max_per_document=None`表示不限单文档段数；显式整数仍支持旧基线复现。
- `local_full_text_search/rag/answer_service.py`：默认构造不限单文档段数的EvidencePolicy。旧持久化`rag_max_chunks_per_document`字段保留读取/序列化兼容，但不再控制普通应用选段；显式注入的历史策略仍有效。
- `tests/rag/test_selection_a_adoption.py`：覆盖旧配置不恢复配额、原顺序/门槛/TopK、显式历史策略兼容及设置不变。
- `tools/validate_selection_a_adoption.py`：从真实索引恢复冻结候选，以应用实际策略重放91题，与历史A逐字比对。Gold只用于选择结束后的判分。

## 验证

索引包：`g000033-71ec3c52f3`。当前活动集合91题，其中50题已审核（46题有正向证据），其余41题仅诊断（40题有正向证据）。

|检查|结果|
|---|---|
|索引正文及已冻结元数据一致|通过|
|Gold与持久化设置哈希不变|通过|
|正式策略与实验A实际Context、条目、评分一致|91/91通过|
|已审核正向题的全部Gold引文覆盖|37/46 → 41/46|
|诊断正向题的全部Gold引文覆盖|35/40 → 35/40|
|新增Gold引文丢失|0题|
|RAG及设置回归|463通过、10个既有失败、1跳过；新增失败0|
|远程API/答案生成|未调用；无答案正确率结论|

收益涉及019、021、023、030及row-holdout-001。019只恢复场景页，处理页仍缺，因此不计入“全部引文完整”的新增4题。

10个失败均位于`test_source_consensus.py`，与上次回退后基线相同，属于旧共识增强行为的测试期待，不能宣称全套测试全绿。此次运行tests/rag为455通过，另外settings_service为8通过；与前次用例名比较，仅新增4项采用测试。

命令：

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_selection_a_adoption
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag -q --junitxml=reports/rag_selection_a_adoption_20261113/unit_results.xml
.venv/Scripts/python.exe -X utf8 -m pytest tests/test_settings_service.py -q --junitxml=reports/rag_selection_a_adoption_20261113/settings_unit_results.xml
```

本轮冻结了上游候选，未重新跑BM25、向量或重排；这证明正式代码接入与已验证A一致，不能称为独立的新一轮端到端验证。HTML保留八组历史对照，默认展示A，并明确标注当前采用状态。

## 最严重的剩余风险：明确的适用条件没有进入判断

022问“几乎所有发光项目定标失败，发光值都低于1000”。QueryPlan只识别了阶段、方向、对象，未提取“几乎所有项目”的范围。因此，`化学发光定标处理思路分析.pptx`第17张明确标注“个别项目定标失败，报DEV”的片段没有被识别为范围不符，并进入A的Context。第28张个别项目LVC资料也在Context内。

这证明问题理解到条件检查之间存在缺口：资料并非没写条件，而是问题条件未被结构化，后面没有进行该条件的比较。其风险是把另一种情况的处理方法交给最终LLM。原文与索引核验沿用上一轮source_audit及页面图像，本轮再次核对实际A QueryPlan与Context。

这属于修复优先级判断，不代表它造成最多缺失，也不代表最终答案已错误。A前上游就存在这些候选；旧配额只是偶然挡住其中一部分。不能因该问题否定A已证明的收益。

下一步保留A不动，只修复“明确条件提取及候选冲突判断”这一边界：条件需有原文依据和对应对象；明确冲突与未说明分开；方法问题中的条件分支不能误删；不同问法与未参与调试问题一起验证。不得只补022一句话，也不得把unknown全部过滤。验证仍止于API前，分别记录识别正确率、冲突漏检/误杀和正确引文保留情况。

## 仍然缺证据的其他题不能归为同一原因

|题目|已观察到的缺口|
|---|---|
|004|正确片段被当前总TopK截断，C能恢复；本轮没有采用C|
|006|正确片段重排后第18位，进不了Top8；不能直接归因为条件检查|
|019|场景13已恢复；处理14在去重阶段仍有、重排阶段记录已无，需区分重排入口截断与排序问题|
|028|其中一条Gold在已有关键词和向量候选记录中均缺失，属于更早的召回缺口|
|row-holdout2-001|多来源条码证据仍受最终总段数限制，C能恢复|

023恢复CSDB表格行时带入同表DCF，042带入相关现象背景，均不能只因负例计数增加就认定答案退化。022则确有独立片段的明确范围不符，应优先检查。

## 交付

- `reports/rag_selection_a_adoption_20261113/report.html`：应用原生格式的前后对照。
- 同目录`open_report.cmd`：启动本机查看服务，使用“打开原资料对应页”按钮定位原文。
- `production_replay.json`：91题逐项接入核验。
- `production_A.jsonl`：实际生产策略重放结果。
- 历史`rag_selection_factorial_20261113`不覆盖；其当时“不采用”结论已被用户本轮“采用A、分层归因”决定替代，B/C及组合未采用。
