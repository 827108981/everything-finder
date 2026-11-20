# 可选领域适配层：本轮开发说明

## 使用和回退
`rag_query_plan_enabled` 默认仍为 false。仅在隔离验证设置中开启，`rag_domain_config="chemiluminescence"` 加载内置 JSON，也可传配置路径。配置不可读或校验失败时回退 generic；没有达到高置信度门槛时，原有查询字段、路由和生成摘要不变。未修改用户持久化设置、索引包、Gold 或默认测试集。

## QueryPlan 与配置
旧字段兼容，新增均有默认值：adapter_active、strategies、scope_constraints、protected_evidence、method_branches、ambiguities、adapter_policy。entity_details 增加 is_strong_constraint、needs_scope_check；各实体/信号仍记录 confidence/source/reason。

protected_evidence 是“应保护何种证据”的要求，不存来源ID或正确页码；method_branches 是原问题的分支问项，不是模型编造的业务答案。实际找到的范围和行在候选 query_plan_diagnostics 记录。

通用层负责：激活门槛、未知原词保留、多问项拆解、原始问题继承、有限代表、原文区间、范围限制、诊断和回退。化学免疫配置负责：少量稳定的阶段/对象/产品线/部件别名、型号和数值模式、范围文本模式、部件层级提示。没有报警码答案表，没有测试ID、Gold、页码或来源路径参与运行。

配置文件采用 JSON（与现有加载器、打包方式一致），位置 `local_full_text_search/rag/domain_configs/chemiluminescence.json`。新增 activation_confidence=0.9、max_exact_representatives=2（允许0–3）、scope_patterns、scope_entity_types、part_whole_groups。配置不需要穷举所有报警码、项目或试剂。版本标识是原文件内容哈希。

## 每层修改

| 文件 | 改动 | 解决的问题 |
|---|---|---|
| models.py | 兼容字段与强约束/范围标记 | 可追踪增强是否真正触发 |
| domain_adapter.py / domain_configs/chemiluminescence.json | 门槛、上限、稳定别名、条件模式 | 不把完整业务知识写进代码 |
| query_plan_enrichment.py | 识别后再决定激活；保留版本/时间/否定/范围；方法/数值/解释/范围策略 | 普通大写词不触发强保护；方法题不只看缩写 |
| retrieval/lexical.py / vector.py | 只有 adapter_active 才追加路由 | 普通问题保留原始检索 |
| identifier_evidence.py | 高置信度精确代表总上限；方法题不提词条到首位，有流程/条件结构时选流程代表，否则保留完整行代表；完整表头/前缀区间；显式范围冲突诊断 | 保护有边界，防止方法流程被简表抢占 |
| constraint_matcher.py | 未激活走旧路径；不同型号仅在明确限定时冲突 | “提到另一个型号”不等于资料不适用 |
| evidence_policy.py | 显式限定型号/版本冲突拒绝；沿用原阈值和配额 | 精确命中不越过已证实冲突 |
| evidence_contract.py | 来源文本的范围观察、方法/部件边界生成约束、实际Context诊断 | 防止范围扩大、整件/子件混用 |
| context_builder.py | 表头+范围前缀+完整行；范围提示只取保留区间；整行不足预算则不截半行；记录排除原因 | 防止只留代码/原因，丢处理或适用范围；防止被删行的词通过提示泄漏 |
| answer_service.py | 激活后的生成约束、全部新增计划字段与实际覆盖日志 | 对每条回答追溯问项、来源和缺口 |

## 明确的边界
- 范围观察是原文中出现的词，不是自动证明该段所有事实属于该对象。多个型号、跨页关系、流程图箭头仍需语义判断。
- 强冲突只对明确限定的型号/软件版本以及已有阶段/范围条件实施；试剂等未给出唯一身份时不做硬过滤。
- 方法策略保持原顺序，不能保证有限TopK内含全部业务分支。完整性仍需实际回答验证。
- 结构保护只适用于严格规则四列表格；不规则/跨页表格保留原块，不猜列。
- 行投影把表头之前的原文一起保留；源块本身没带适用范围时不能凭空补出。
- Context 的范围提示及生成约束不是语义正确性硬保证；当前引用ID校验不等于逐条事实适用性证明。
- 本轮没有默认强制二次LLM复核，避免改变普通问题和引入未验证的全链路行为。未解决的生成泛化会在验证报告列为上线阻碍。

## 验证入口
1. `python -m pytest tests/rag -q`：全套回归及新增边界测试。
2. `python -m tools.validate_adapter_robustness --arm <唯一运行名>`：已发布包上92题真实检索。加 `--answers --ids ...` 运行重点题完整问答。
3. `python -m tools.ablate_adapter_robustness`：冻结上一版约束后候选和分数，分离选择与Context修改；92条旧Context必须全部复现。
4. `python -m tools.answer_adapter_robustness --candidate native_v2 --output <唯一名> --ids ...`：冻结上游、实际下游生成与判分。每条必须确认Context完全相同；不是新的全链路重跑。
5. `python -m tools.summarize_adapter_robustness`：分集合生成三方对照、逐阶段Recall/MRR、引文损益和真实回答。

测试脚本运行Gold判分在正常链路之后。默认关闭与开启使用同一索引、LLM、设置和标签；不做质量重试。历史上一版答案作为历史对照，不冒充同期配对。独立/回归集合已参与过历史调试，不称全新盲测。
