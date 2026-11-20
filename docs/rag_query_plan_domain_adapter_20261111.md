# 通用 QueryPlan 与领域适配层开发说明

## 本次范围

新增的是现有链路上的可选增强，不重建索引，不更换向量/重排/生成模型，不修改 Gold。以问题、已发布索引正文及原始位置为运行输入。测试标签只交给事后评测器。

开关：`AppSettings.rag_query_plan_enabled`。默认 `false`，须通过回归采用门槛才能默认开启。
领域选择：`rag_domain_config="chemiluminescence"`；空字符串运行通用解析；也可填写自定义 JSON 配置绝对路径。配置缺失或无效时退回通用模式并记录原因。

## 结构兼容

原 `QueryPlan.entities: tuple[str, ...]` 被现有支持度计算、诊断和测试使用，保留原类型。
新的 `entity_details` 是结构化 entities：`text/type/normalized_value/confidence/source/strength/reason`。
这样不要求旧调用方突然接受对象，也不丢掉用户要求的实体信息。`unknown_entities` 同样使用结构化记录；报警语境能够确认“它是一个代码”，但不表示系统已知该代码含义，因此未查词典的报警代码也在未知清单中保留。

新增字段：

| 字段 | 内容与用途 |
|---|---|
| entity_details | 原文实体、类型、规范值、置信度、来源、强度、理由 |
| answer_requirements | definition/cause/handling/procedure/comparison/evidence_requirement；每项含 confidence/source/reason |
| sub_queries | 原始问题+关注问项、focus 原文、继承条件及 confidence/source/reason |
| retrieval_hints | 精确匹配、表格优先、证据保护、槽位覆盖等策略信号及解释 |
| unknown_entities | 不认识的标识符和引号术语，仍参加召回，不变成硬过滤 |
| domain_config | 配置文件名和内容摘要；通用模式为 generic |
| plan_diagnostics | 配置加载/降级等解释记录 |

原问题永远保留。未明确提到型号或阶段时不补写；否定/推测表达不能变为全局硬条件。多个条件分支不能同时变为互斥的硬条件。

## 哪些通用，哪些是领域配置

通用代码：结构化实体模型、问题目标检测、保守分句、原条件继承、未知词保留、精确文本边界匹配、原始检索+补充路线、RRF、候选去向、配额保护、实际上下文覆盖核对。

领域配置：`local_full_text_search/rag/domain_configs/chemiluminescence.json` 中的少量 stage/scope/metric/direction 别名，以及大写代码、型号、数值单位正则。采用标准库 JSON，避免新引入 YAML 依赖；其作用等同用户建议的 YAML 示例。没有 CSDB/DEV/DUP 的答案词典，没有页码、正确文件名或 Gold ID。

原有 QueryAnalyzer/ConstraintMatcher 中的历史词项未强行迁移，以免改变关闭开关时的行为。新领域配置仅通过 DomainAdapter 注入；新代码不增加逐报警处理规则。将来可分批迁移旧词项，不能声称本次已经完全消除了历史领域耦合。

## 检索与证据选择

1. 原问题正常执行原有关键词与向量检索。
2. 对用户原文中的明确标识，追加精确词检索；旧关键词后端与 word_v2 都使用当前发布包的 metadata FTS。root、扩展名、OCR 条件保持一致。词边界避免把 XYZ1 当作 XYZ。
3. 只有可识别的多目标/分支问题追加关注问项向量路线。每条均包含完整原问题，不把继承条件裁掉。每路对父块只投一次票，沿用现有 RRF；同一标识重复出现不会创建重复的精确路线。
4. 为避免“热门型号/报警码占满候选”，每个标识仅选择一个优先代表：完整、规则的结构化行优先，否则使用原排序最前的精确候选。其他候选维持相对顺序。RRF/重排原分数不伪装成新模型分数。
5. 硬条件冲突仍淘汰，未知条件仍未知；不因未知词淘汰候选。问题明确说“报 X”，候选却明确处理“报 Y”且不含 X 时，记录 explicit_identifier_conflict。
6. 最终证据中，精确代表独立于普通同文档配额保留；重复提到同一标识的其他段落仍受预算限制，记录“标识已有保留代表”。总段数、字符预算保持原设置，保护不保证无限保留或绕过硬冲突/分数门槛。
7. 能严格识别为完整四列“标识/描述/原因/解决方法”的表格，才按用户标识抽取连续原文行。保存 source_spans，引用仍绑定原 chunk、原文件、原页。列数不规则、多行单元格、缺表头等情况不猜行归属，保留原块。
8. 对实际发送的 included_text 核对 definition/cause/handling 槽位。只有目标标识和对应非空单元格都实际存在才是 covered；其他为 unknown，不把词语共现当成答案正确。

## 边界

- 这是保守的确定性解析器，不是任意复杂中文的语义解析器。无法可靠拆分时保留完整原问题；最多四个关注问项。单句复杂比较可能只保留 comparison 目标而不拆检索路线。
- “原因”可能是处理报警时的推断性回答目标，其置信度低于明确提问；它不是预先编造的原因，更不能要求模型补造来源没有的事实。
- 结构化覆盖仅验证规则表格。一般叙述、任意表头、跨页/合并单元格、图像中的对象归属仍需后续能力；unknown 不等于答不出。
- 型号精确共现不证明整段适用；表格行优先也不证明简表比详细排查流程更适合问题。LVC 等方法型问题特别需要回归。
- 只做覆盖诊断，未把缺少某槽位自动变成全盘拒答。生成时要求分别回答有依据问项并说明缺口。

## 日志与测试

QueryPlan 全字段进入 query_interpreted 事件。每个候选增加 query_plan_diagnostics：exact_matches、structured_rows 原文偏移、identifier_conflict、exact_representative、配额保护状态、slot_coverage。原有 selected/reason 继续记录阈值、冲突、去重、TopK、配额、字符预算等去向。

上下文条目增加 source_spans/projection 和实际 included_text。answer_slot_coverage 事件核对截断后的实际文本，不能用完整父块代替。

运行入口：

```powershell
.venv\Scripts\python.exe -m pytest tests/rag -q
.venv\Scripts\python.exe -m tools.validate_query_plan_adapter --arm <新名称> --disabled
.venv\Scripts\python.exe -m tools.validate_query_plan_adapter --arm <新名称>
.venv\Scripts\python.exe -m tools.validate_query_plan_adapter --arm <新名称> --ids rag-biz-023 --answers
```

`tools/answer_query_plan_replay.py` 用冻结的真实上游候选与分数运行原生下游、真实模型和独立裁判，必须核对最终上下文完全一致；它不是新的全链路检索，也不计为独立检索重复。第23题另用真实完整问答入口核验。

最终数据与采用判断见 `reports/rag_query_plan_adapter_20261111/report.md`，不凭单次分数决定默认开启。
