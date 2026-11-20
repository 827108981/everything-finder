# Domain Adapter 稳健性开发前分析

## 1. 当前数据结构
QueryPlan 保留原问题、规范化问题、意图、字符串实体、条件、词法/语义查询及查询变体；可选层增加 entity_details、answer_requirements、sub_queries、retrieval_hints、unknown_entities、domain_config、plan_diagnostics。QueryEntity 有文本、类型、规范值、置信度、来源、强度、原因。目前缺少统一激活门槛、来源范围和方法分支要求。

## 2. 入口与调用链
AnswerService 按 rag_query_plan_enabled 构造 QueryInterpreter；原 QueryAnalyzer 先工作，可选 enrich_plan 在其后补充。LexicalRetriever → VectorRetriever → HybridRetriever 的 RRF → 去重 → 重排 → ConstraintMatcher → EvidencePolicy → ContextBuilder → GenerationGateway。默认开关为 false，本次不得修改默认值或用户持久化设置。

## 3. 识别方式
配置中的别名/正则识别型号、阶段、范围、方向、指标、数值；通用标识符模式保留未知代码。只有明确的“报/报警/错误码”语境才赋予 alarm_code 类型，代码含义不从词典猜测。已有规则识别的条件被镜像为实体。目前所有丰富实体都会触发增强，低置信度标识符也参与保护，这是门槛缺陷。

## 4. 精确匹配影响
词法增加精确标识符查询并前移代表；向量增加完整原问题+关注问项的子查询；融合后再次前移；本地重排仍按原模型执行；EvidencePolicy 再次选择精确代表；ContextBuilder 对规则四列表格提取当前行。不是改变 embedding 模型，也不是把人工答案放进提示。

## 5. 配额和去重
默认词法/向量各50，融合30，重排20，最终8段，每文档3段，12000字符。去重保留来源关联。原增强每个标识符选一代表，代表不占常规文档配额；没有全局代表上限，也未区分方法问题和解释问题。表格投影可能丢失表头/范围前缀，字符预算可能截断处理字段。

## 6. 风险位置
- query_plan_enrichment.py、retrieval/lexical.py、retrieval/vector.py：entity_details 非空即介入，普通问题也会改变检索/提示。
- identifier_evidence.py:prioritize：一律优先表格；方法流程可能降到后面。Q021 前次不是事实召回丢失，而是上下文组织/答案完整性退化。
- evidence_policy.py：精确代表可能挤占正常 TopK，多个实体没有总上限。
- context_builder.py：行投影不保留范围前缀；原有 device/topic 检查不足以保证型号、版本、部件层级。
- answer_service.py:_plan_summary：提示不绑定具体来源范围；仅凭增加实体提示可能使相同 Context 产生更宽泛答案。
- 现有引用合法性校验只核对引用 ID，不能证明该引用支持答案对象和适用范围。

## 7. 测试入口与指标
tests/rag/test_query_plan_adapter.py、test_query_plan_native.py 和全套 RAG 单测；tools/validate_query_plan_adapter.py 为真实索引入口。冻结历史 baseline/candidate 的92题、各阶段正文、真实答案和 manifest，进行新旧对照。主集50题中48 reviewed；007 disputed、024 pending 单列。42条历史回归另列，其中未审核题不进入正式分数。Gold 字段包含问题、行为、约束、原文证据、禁答事实和 answer_checks。
检索按来源+实际引文核对，不按同页即命中；报告 Recall@1/3/5、MRR、完整引文保留。答案检查正确性/分支/范围/引用，自动判分不能替代业务审核。默认关闭的行为等价性单独检验。

## 8. 层级职责与实施顺序
QueryPlan 只记录用户明确说了什么、想回答什么、缺什么；不写入来源页码、正确答案或伪造业务分支。配置维护少量稳定名词、实体模式、激活/代表配额。证据层从正文记录范围、完整行和方法结构；生成层使用这些来源约束，不能从 QueryPlan 推导业务事实。
先修触发门槛，再限制代表数量与方法顺序，再保留来源范围/部件边界；分别记录独立对照，最后验证组合。方法题保留条件回答，不强制追问；真正要求数值计算而缺数据才说明缺项。结构检测仅为启发式，不声称语义完整。全部步骤必须记录通过/失败/未决，出现退化如实列出，不能只报总一致率。
