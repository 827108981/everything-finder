# 2026-11-12 业务题目、证据选择与评测修订

## 本轮边界

先按用户确认修订业务问题与 Gold，再用同一份新版数据比较 Adapter 开关。保留历史版本和历史报告，历史文件中的040不是当前活动题。当前入口为 `tests/rag/fixtures/rag_active_suite_v5.json`；主集49题，回归集42题，总计91题。主集47题 reviewed，007 disputed、024 pending；回归集中本轮明确确认的条码、电压、密封圈题为 reviewed，其他状态不变。诊断题不得混入正式得分。

`rag_query_plan_enabled` 默认仍为 false。验证仅修改进程内配置，不写用户 settings.json。应用评测页的默认文件更新为 v5 已审核47题导出；主集49题中的007/024仅供诊断。用户仍可主动加载历史版本用于历史复核。

## Gold 修改及原文核验

|题目|新版要求|依据|
|---|---|---|
|019|大部分发光项目定标报DEV，问可能原因和解决思路|化学发光定标处理思路分析，第13张场景、第14张四类原因、排除故障污染后灌注、建议40次以上、重新定标|
|050|CL-8000i暗计数失败，问四类可能原因|免疫场景化培训教材-光学 PDF第17页；不要求更换件或故障确诊|
|independent-07|明确问注射器密封圈寿命与编码|生化场景化培训教材-电解质第12、27张；66000次，040-000765-00|
|电压题|准确型号改为BS-2800M；同时回答DC24V标称值与22.8～25.2V正常范围|BS-2600M/BS-2800M维修手册封面、PDF第205/277页（印刷页195/267）；没有设立BS-2800别名|
|条码题|高速与中低速免疫资料共同支持的码制|高速条码PPT第17张、中低速条码PPT第14张；相同值并非推断自单型号|
|040|从新版主集、活动清单及本轮全部分母删除|保留旧版本，不把删除前后的总分当算法收益|

9条新增/调整引文均用原资料和 g000033-71ec3c52f3 索引正文核验。`reports/rag_business_repair_20261112/source_audit.json` 保存原文、页码、chunk ID、匹配结果和原文件哈希。各新版 review JSON、JSONL、可阅读 Markdown 同内容。

## 通用代码改动

1. `source_consensus.py`：从当前已发布索引读取与问题明确字段相符的完整列表行，保留原检索过滤条件。比较同名字段的规范化集合；相同值只回答一次；不同值记录冲突，不取并集。平台分组来自配置中的集合/资料名称线索，不来自题号、Gold或答案表。
2. 只有至少两个已识别平台组的实际 Context 来源、相同字段及相同值才形成 `platform_consensus`。同平台的PDF/PPT副本不会构成跨平台共识。正文明确型号或版本限制时降为 scoped_agreement，不能扩大范围。候选阶段发现冲突，即使部分冲突材料没有进入Context，也不能宣称平台共识。
3. 共识字段记录 fact、values、supporting_sources、supporting_source_count、platform_groups、conflicts、consensus_level、scope_basis。实际 Context 重新核验字段是否完整保留，不把候选命中当成实际引用。
4. 限量保留代表证据。枚举字段、场景和相邻说明共用有限前置位置；普通排序、重排分数门禁及硬冲突检查继续生效。没有高置信领域激活时，不执行这些路线。
5. 相邻页连接只用于短的明确报警场景页，紧邻下一页必须以原因/处理标题开始且没有另一个明确报警。保留两个真实源块及各自页码，不拼造原文，不连接任意相邻页。
6. 数值题中的未知部件编号可作精确检索提示；仍是软提示，不升级为硬过滤条件。只限量优先保留正文同时包含该编号和数值单位的相关材料。
7. 明确复合部件名不自动触发整件/子件歧义。具体部件词放在领域配置中；同时独立提到整件和子件时仍保留对象核对要求。
8. 范围表达允许限定词和“项目”之间有有限长度的修饰语，例如“大部分发光项目”；规则配置化，并保留原问题文本和置信来源。
9. 第一轮91题发现050仍退化：完整原因已到Context，但生成误把通用培训资料当作型号冲突，转而展开其他故障的维修流程。追加通用修复：问题中的诊断主题与正文标题精确对应且有原因段时限量保护；只问可能原因时不扩展成更换流程；未知型号范围仍是未知，不自动判冲突，具体阈值和专属参数不据此推广。首次失败保留在 answers_AB.jsonl，不用后续成功覆盖失败记录。
10. 实际Context的共识结果另作为生成范围约束传递，明确不能把共同字段缩回单型号；不给模型预写答案。纯原因题不再额外附带方法型问题的阈值/操作分支要求。最终生成结果与此前失败分别存档。
11. 报警标题“XYZ排查思路”也属于显式报警归属；不必出现“报XYZ”字样才能识别，避免其他报警的方法串入。相邻页继承还要求父场景的显式阶段/范围已匹配、无硬冲突；父场景未通过证据门禁，后页也不能以其名义获得支持。

## 评测修订

- Context 记录合并来源的完整 chunk ID 和文件路径。计分仍要求该来源对应的完整引文出现在实际 Context；只合并来源ID不能使另一行内容得分。
- 裁判区分标称值与正常范围、可能原因与确定故障、跨来源共同事实与单型号特有条件。裁判不能用 Gold 补齐实际引用的缺口。
- 活动集合集中声明版本、题数、哈希和退休题号；本轮运行器只从此清单读取，不使用全目录扫描来拼凑用例。
- 原50题与现49题不是同一分母。开关对比均使用相同v5题面、事实和来源；正式题、pending/disputed诊断题分别统计。

## 验证协议

1. 全部 RAG 单元与兼容测试，验证通用问题不激活、完整列表、冲突、明确范围、重复来源、部件对象、活动集合和来源ID。
2. 关闭/开启 Adapter，各自完整运行91题真实BM25、向量、融合、重排、条件、证据和Context链路；保存每阶段正文、来源、排名、淘汰原因。
3. 从冻结检索轨迹重放原生下游并调用真实LLM，必须与原轨迹Context逐字相等；不相等则记错误，不能冒充配对结果。Gold只送独立裁判。
4. 重点八题：019、050、条码、电压、密封圈，以及020/021/023回归，做固定提示下旧/新Context消融；再固定新Context比较通用/领域提示。生成消融不冒充完整端到端门禁测试。
5. 对额外主张另存逐条来源支持核查。单次LLM答案不能证明统计稳定；任何退化、无法判清或调用失败单列。
6. 验证末核对持久化设置哈希、索引bundle、数据哈希，报告改善、退化、未解决问题及是否可开启。结果见同目录报告，不能以整体一致数作为唯一结论。

## 目前能力边界

共识比较是保守的结构化字段比较，不是开放域语义证明。未识别的字段、跨行不完整表格和同义值不会擅自合并。平台组说明覆盖了哪些资料，不能证明所有型号、版本或现场设置一致。若后续要支持更多字段格式，应新增与题号无关的解析及反例测试，而不是加入具体题目例外。

## 本轮文件清单

|文件|用途|
|---|---|
|local_full_text_search/rag/source_consensus.py|新增：字段比较、共识、真实引用复核、相邻场景、限量保护、生成共识约束|
|local_full_text_search/rag/domain_configs/chemiluminescence.json|平台/产品线名称线索、复合部件名、范围修饰语、诊断主题模式；不包含答案值或题号|
|local_full_text_search/rag/domain_adapter.py|校验新增配置|
|local_full_text_search/rag/query_plan_enrichment.py|明确部件和范围解析|
|local_full_text_search/rag/retrieval/lexical.py|结构化共识及未知参数编号检索提示|
|local_full_text_search/rag/retrieval/hybrid.py|保留来源诊断及有限保护|
|local_full_text_search/rag/identifier_evidence.py|注释重算时保留已核实的来源关联诊断|
|local_full_text_search/rag/evidence_policy.py|有限代表证据通过正常分数/冲突门禁后保留|
|local_full_text_search/rag/context_builder.py|合并来源ID、文件路径、原因主题适用性说明|
|local_full_text_search/rag/evidence_contract.py|只问原因时不扩成维修流程；保留未知与冲突区别|
|local_full_text_search/rag/answer_service.py|真实Context共识日志和生成约束|
|local_full_text_search/rag/evaluation/answer_judge.py|标称值/范围、跨资料支持、可能/确定性结论的通用裁判要求|
|local_full_text_search/ui/rag_eval_page.py|当前默认评测入口改为v5已审核47题；不修改用户持久化设置|
|tests/rag/test_source_consensus.py|新增通用正反例、实际引文计分、活动集合检查|
|tests/rag/test_adapter_robustness.py|修正单一明确对象不应自动歧义的旧测试|
|tools/revise_gold_v5.py|可复查的Gold修订、原文/索引验证、版本及哈希输出|
|tools/validate_business_repair.py|活动集合的真实检索/完整问答入口|
|tools/answer_business_repair.py|冻结上游、原生下游与真实LLM的配对验证|
|tools/ablate_business_repair.py|新旧Context/固定Context提示消融及额外主张核查|
|tools/report_business_repair.py、tools/finalize_business_repair.py|分集合、逐题报告及最终验收|
|tools/validate_query_plan_adapter.py|运行协议题数由实际集合计算，不再写死50+42|

新增数据文件组（每组含review JSON、JSONL、Markdown）：rag_gold_v5、rag_acceptance_v1_1、rag_row_holdout_v1_1、rag_row_holdout_v2_1、rag_independent_v2。另有rag_gold_v5_reviewed.jsonl及rag_active_suite_v5.json。未覆盖任何旧版本文件。acceptance题目本身未修改，只发布同版快照便于活动集合固定。
