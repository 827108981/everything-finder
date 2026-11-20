# 分窗口最高分正式接入（2026-11-18）

按用户明确要求，将`LocalBgeReranker`的`input_mode`默认值由`truncate`改为`window_max`。普通问答Worker和批量评测Worker均使用该构造器默认值，因此两条路径同步生效。窗口最大512 token，重叠64 token，不修改重排入口20、最终Top8和条件检查。

问答进度日志补充实际输入模式、重叠及窗口上限；批量评测元数据读取实际服务中的reranker属性，不使用猜测值。旧截断模式保留用于历史对照；历史向量包装实验显式指定truncate，避免重跑旧实验时无意使用新默认。

## 验证

- 使用不传input_mode的默认构造器回放111题；沿用已验证、token完全一致的本地模型分数，重新执行生产排序、条件检查、最终证据门禁和Context组装。
- 111/111的候选顺序、实际Context、entries及阶段分数与上一轮window_max结果一致。
- 004、019、IMFH1、VPP共4题重新执行真实检索及本地BGE窗口推理，不复用分数缓存；全部与回放一致。并非新增独立验收题。
- 聚焦测试16通过，包含默认构造器实际读取尾部目标行的测试。
- 全量544通过、10个既有source_consensus失败、1跳过、1预期失败；失败名称与之前10项一致，无新增失败。
- API调用0；Gold、发布索引及用户持久化设置保护哈希不变。
- 当前源码默认已采用。未重建已打包的exe，也未替换已经运行的进程；已打开的旧版本不会自动热更新。

## 当前最优先的问题

基于111题API前轨迹，下一步优先处理条件检查的适用性判断与排序：高相关正确证据可能因为缺少一个显式对象词被标UNKNOWN，然后整体排在低相关MATCH之后；同时其他型号/报警的相关片段又未必被确认适用性。

最明确的正式案例：免疫条码题两份正确资料在BGE第2/3位，条件后第15/16位，最终Top8丢失。生化条码尺寸题也由第2降至第14，最终经分数过滤仍保留第7；它是排序缺陷但不是当前Gold丢失。

不能把其他缺口都归因于条件检查：

|题目|当前缺口|所在阶段|
|---|---|---|
|019 多项目DEV|处理页在去重后第30，未进入重排前20|场景/处理页关联、上游排序及重排入口|
|BS-800温度传感器，row-holdout2-007|另一条必要证据在去重后第24，未进入重排|重排入口|
|A&T/Medica ISE，independent-05|仍为2/3完整引文|此前已存在的上游证据覆盖缺口，本轮未修复|
|超声空化电压，source-independent-003|指定PDF缺失，但Context存在同名DOCX故障机制|等价证据与版本需要复核，不能直接算无答案|

最高分模式恢复IMFH1、改善VPP，但8题首条Gold名次下降，部分问题新增其他型号/报警片段。没有生成答案，不能说答案已经误用，也不能仅凭Gold覆盖不降就宣称适用范围问题已解决。

接入没有新增任何按题号、报警码、机型或页码加分的规则。下一步应独立验证UNKNOWN不整体后移、明确冲突依据同一对象/属性，并保留未知标记；不把所有UNKNOWN当成正确，也不全部删除。

## 产物

- `reports/rag_bge_window_adoption_20261118/production_replay.json`
- `reports/rag_bge_window_adoption_20261118/entry_probe_audit.json`
- `reports/rag_bge_window_adoption_20261118/adoption.json`
- `reports/rag_bge_window_adoption_20261118/pytest.xml`
- 三组对照HTML沿用`reports/rag_bge_windows_20261118/report.html`，顶部增加当前已采用状态。原三组实验分数不变。

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_window_adoption_20261118
.venv/Scripts/python.exe -X utf8 -m tools.report_bge_windows_20261118
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag tests/test_settings_service.py tests/test_open_location.py -q
```
