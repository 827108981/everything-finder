# 答案判分项 v2

`rag_gold_v2_review.json` 与 `rag_gold_v2.jsonl` 的 18 道题内容相同。v2 保留 v1 的问题、检索金标、负例、索引版本和 `required_facts`，新增 `answer_checks`。`required_facts` 仅作历史参考；新版答案裁判按 `answer_checks` 判分。

每条检查项有唯一 `id`、`text` 和 `role`：

- `core`：当前问题必须回答的事实。缺失时为“必答内容缺失”。
- `safety`：维修操作必须提示的安全事项。缺失时为“安全提示缺失”。
- `context`：可帮助理解的背景。缺失会记录，但不把答案判错。

答案中的矛盾事实或出现 `forbidden_claims` 会判为“明确错误”。回答方式由裁判独立检查，不因某一事实缺失而自动改变：`clarify` 需要明确请用户确认决定答案的条件；`refuse` 的固定短句“资料中没有足够依据”可以符合拒答要求。无法可靠判断时标为“待人工复核”。独立 LLM 裁判的逐项判断仍须人工核查，尤其是安全项和业务语义。

重点修订：007 的流量数值为必答，BM80 与蠕动泵为背景；009 的阀门与 SR09 顺序为必答，遮光、预热和反应杯说明为背景；003/005 单列断电、防静电要求；008 单列软件试转检查；014 单列“请确认磁盘类型”的澄清动作；017/018 的扩展拒答理由改为可选背景。

复核已保存的一次运行，只重新调用答案裁判，不重新检索或生成：

```powershell
.venv\Scripts\python.exe rejudge_report.py --cases tests\rag\fixtures\rag_gold_v2_review.json --run <原运行目录> --out <新的输出目录>
```

新目录生成 `report.html`、`report_detail.json`、`report_cases.csv` 和 `judgement_comparison.csv`。原运行目录保持不变；报告写明复核来源与旧→新判定。使用不同 Gold 版本的答案判分数不可直接作为检索质量变化。
