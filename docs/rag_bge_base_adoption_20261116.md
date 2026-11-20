# BGE基础排序接入记录（2026-11-16）

本轮按用户“现在将代码改成BGE基础排序”的明确指令，将应用默认重排切换为此前五组实验中的B。没有启用结构规则、本地4B模型或QueryPlan Adapter；没有调用远程API，也没有运行答案生成。

## 实际变更

- `local_full_text_search/rag/reranking/local_bge.py`：BGE sigmoid分数降序；同分仅用融合分数、稳定chunk ID确定顺序。`rerank_score_combined`与BGE分数相同，移除辅助混合及融合旧名次保护。保留全量重排顺序，并直接返回前N；删除重排返回值里的旧单文档最多3段逻辑，与已采用A保持一致。
- 旧权重参数仍接受，以兼容历史脚本及配置；不论旧配置写0.65/0.05/0.30或max_rank_drop=3，均不能恢复混合排序。有效参数为1/0/0/0。
- `workers/rag_query_worker.py`和`rag_eval_worker.py`：普通问答与批量评测使用相同BGE默认，不再传旧辅助权重。
- `config/defaults.py`：兼容字段默认改成1/0/0/0。没有写入用户持久化设置文件。
- `rag/evaluation/runner.py`：记录实际`bge_base`及有效权重，避免报告错误显示旧配置权重。
- `answer_service.py`：明确辅助分仅为只读历史诊断，不再影响BGE排序。召回、解析、重排入口20条、条件检查、最终Top8/12000字符均未改变。
- 可选`ordering_policy`接口仍保留，普通问答和评测均不注入；结构规则与本地语义判别不参与默认流程。

## 验证范围和结果

1. 冻结91题回放，直接执行生产`_order_scored`与生产条件检查、证据筛选、ContextBuilder：重排顺序、条件顺序、实际Context、条目与评分 **91/91等同此前B**。
2. 实际本地模型及`AnswerService.retrieve`重新执行002、004、006、021、row_holdout2-001五题：重排及Context均与冻结B一致。没有凭据访问或答案生成，网络连接被阻断，调用API为0。
3. 聚焦重排/策略/工作线程测试：29 passed。测试覆盖旧权重无法重新启用混合、融合旧排名无法覆盖模型顺序、同文档多段不被重排配额删除、同分排序稳定。
4. 全量：520 passed、10 failed、1 skipped、1 xfailed。10项均为已记录的source_consensus历史失败；最终验收脚本逐项对比失败名称，不把它们算作新增失败，也不宣称全绿。
5. HTML使用原生报告样式，旧P与当前B逐题对照，附原资料页定位按钮；通过`open_report.cmd`启动本地报告服务。
6. Gold五文件、活动suite、AI metadata、用户持久化settings共8项保护哈希不变。原资料及发布AI包`g000033-71ec3c52f3`没有重建。

这次是**接入已验证的B，不是再次提出一个新算法**。接入确认使用冻结候选回放和5道真实入口；没有声称重新真实召回全部91题。完整证据数字沿用同一冻结输入验证：正式41/46→43/46，诊断35/40→35/40，004与006恢复。未生成业务答案，不报告答案正确率。

## 已知问题继续保留

- 019的处理页在重排入口20段以外；028有一条所需证据未召回；B不会凭空补出它们。
- 免疫条码正确证据仍被条件排序推到15/16，最终前八截断；本轮没有修改条件检查来掩盖此问题。
- Medica题中新增BS-5000参数等范围干扰仍存在。用户本轮明确采用B，保留收益，同时单列该缺口；不把采用等同于这些风险已经解决。
- M/RM新增002错主控板片段是历史实验问题，当前B不启用模型排序，不继承这一处模型额外改序。
- 全部历史Gold审核状态不变；新增12题pending不加入正式计分。

## 产物与复现

目录：`reports/rag_bge_base_adoption_20261116/`。

- `production_replay.json`、`B.jsonl`：生产顺序及实际Context的91题核验。
- `native.jsonl`、`native_audit.json`：5道真实检索全轨迹，API0。
- `pytest.xml`、`pytest.log`、`focused_tests.log`：真实测试结果。
- `report.html`、`report_sources.json`、`open_report.cmd`：旧P/当前B对照和原资料定位。
- `final_acceptance.json`：保护哈希、既有失败清单、实际入口与UI核验。

```powershell
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_base_adoption --phase replay
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_base_adoption --phase native
.venv/Scripts/python.exe -X utf8 -m tools.validate_bge_base_adoption --phase report
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag tests/test_settings_service.py tests/test_open_location.py -q
```

下一步如需继续修复，单独处理条件/对象适用性，不在BGE排序里继续增加题目词表或表格特例。
