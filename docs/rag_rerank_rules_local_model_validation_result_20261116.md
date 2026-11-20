# 重排规则与本地模型开发验证结论（2026-11-16）

## 1. 决定与适用边界

已完成可注入代码、五组固定候选实验、真实检索入口核验、12道新增pending题及HTML前后对照。远程推理API调用0，没有生成最终业务答案。

**确定结论：旧混合分和旧排名保护确实压低了004、006的有效证据；本轮B/R/M/RM均恢复这两题，但新增结构规则和离线模型均未比B多恢复一题完整Gold。当前4B语义判别器不值得启用。B/R也暂不全量替换默认排序：收益真实，但新增错平台参数正文尚未解决，独立pending预验收没有额外收益。**

不以“仍有旧问题”否定新收益。以下分别列旧问题、新增问题、排序下降和完整证据丢失。原默认、持久化设置、Gold状态、QueryPlan Adapter均未改变；本轮不恢复已回退的回答后处理体系。

## 2. 冻结范围与指标

- AI包：`g000033-71ec3c52f3`，实际metadata位于`E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db`。完整SHA见`protocol.json`，不使用普通全文generation冒充AI版本。
- 91道现有题：主集合49、回归42；50 reviewed、40 pending、1 disputed。reviewed中46题有正向引文，其余4题不充当正向召回分母；诊断41题中40题有正向引文。007 disputed、024 pending不变，040不恢复。
- 同一原问题、同一20个候选及正文；原查询、BM25、向量、融合、去重、条件检查不变。最终A冻结：无单文档段落配额、Top8、12000字符。
- “完整”=实际阶段正文按来源绑定覆盖本题全部已冻结正向引文；同页其它文字、未入Context父块、正确关键词出现都不能替代。它不等于答案正确率，也不证明Gold涵盖全部业务知识。
- 表中R@K是逐题引文召回率均值，不是“有一条就算整题成功”。MRR是首条有效正例名次倒数的均值；正式表取最终Context（至多8条）。按group_id等权指标见`metrics.json`；原18题、主集合其余题、四回归集分别见`subset_metrics.json`，不混改题数量比较。
- 旧91题均已用于过调试，不能称独立验收。另12道新题仅预验收，全部pending，另表展示，不计正式分。

## 3. 五组API前对照

| 方案 | 正式完整证据 | 诊断完整证据 | 正式Context MRR | 正式R@1 / R@3 / R@5 | 新增已标负例 |
|---|---:|---:|---:|---|---:|
| P：当前混合分+旧保护 | 41/46 | 35/40 | 0.6665 | 0.4587 / 0.7572 / 0.8696 | 0 |
| B：BGE基础排序 | 43/46 | 35/40 | 0.7689 | 0.5667 / 0.8094 / 0.8855 | 0 |
| R：B+结构规则 | 43/46 | 35/40 | 0.7658 | 0.5630 / 0.8312 / 0.9072 | 0 |
| M：B+离线判别 | 43/46 | 35/40 | 0.7575 | 0.5580 / 0.8094 / 0.8855 | 1 |
| RM：规则+离线判别 | 43/46 | 35/40 | 0.7538 | 0.5543 / 0.8312 / 0.9072 | 1 |

所有候选组相对P均净增004、006两个业务组，无已标正向Gold丢失；M/RM新增002的一条已标负例。诊断集完整数不变，但MRR下降，不能说没有任何排序退化。

正式各阶段完整证据数：关键词45/46、向量44/46、融合45/46、去重45/46，五组完全相同；重排入口20条导致44/46，条件后仍44/46；最终证据及Context，P为41/46，其余43/46。这次没有修复召回入口缺失。

## 4. 收益与损失逐项归因

### 4.1 004：温度采集对象

正确PDF第31页明确给出三块板与采集位置的对应关系。原资料、索引正文、实际入选文本已核对。去重6→BGE3→旧混合9→旧保护9；B/R恢复第3，进入Context。

原机制把“安装位置/更换操作”等词面相关片段排在真正的“采集对象”说明前。正确片段CE约0.99345，旧辅助分约0.38806，混合约0.7684。收益来自撤销这套混合/旧保护的排序干预，不是离线模型理解力的新增收益。

### 4.2 006：清洗液吸入、停止与换瓶

正确PDF第7页直接说明低液位启动P01及SV01/SV02、高液位停泵阀、空瓶自动切换。两条Gold均在原文及索引核验通过。去重15→BGE4→旧混合10→旧保护18；B/R恢复第4并进入Context。

旧保护参照融合旧名次，反而把模型认为相关的证据继续推后。已有错误CL-2000i阀号干扰不应算成本轮才出现；新增CL-2000i片段同样不能给CL-8000i借用阀编号。

### 4.3 023、020、021：规则的实际作用与局限

结构规则只对能够可靠解析的完整四列表格行生效，且逐字绑定原问题标识符，每标识符最多一个代表块；不是所有表格通吃。91题实际只触发020、021、023，来自同一报警表文档。R相对B只这3题Context顺序改变，只有020入选集合发生变化，没有新增完整Gold。

023：P重排11→B7→R1；P实际Context已完整，R只是把完整报警表提前。不能把提前名次记成新答全一题。

021：含义及两条处理分支在五组实际Context中都保留，没有强制计算或追问。R把缩写表提前，相对B会把方法分支后推；未造成Gold丢失，但说明“结构完整”不能等同“整题方法完整”。模型组件还把仅含含义的表判为direct，存在完整性误判。

020：规则/模型没有新增Gold收益，必须查看前后实际Context而非只数代码命中。未另加覆盖贪心、MMR或特定题配额。

### 4.4 新增退化：002错部件资料

问功率驱动板D63/D64，M/RM新增硬件PDF第22页主控板D107/D108。原文页面已渲染核对、索引引文校验通过，确实是另一块板，不是同义证据。B中第9，M中第8进入Context。

该块未路由给模型；模型降序其它片段后，中性候选被间接顶入前八。不能只审查“被模型直接提权的证据”。这是一处本轮M/RM新增的明确负例，而非基线已有问题。没有调用API，不能进一步声称最终答案已经误用。

### 4.5 其它新增与既有干扰

- Medica样本量（rag-row-holdout-002）：正确70/140μL仍保留；基线已有A&T干扰，但B/R新加入BS-5000明确15μL参数，属于新增错平台正文。此新增来自B基座调整，不是结构规则专门造成；没有Gold丢失不等于无风险。
- 生化条码回归rag-acc-001：新免疫来源与Gold所问宽55/高10/留白5mm相同，基线已含多份免疫来源。不能把跨平台来源当作全平台证明，但本轮未确认新增数值冲突。
- 049：CL-6000i Gold保留，新CL-8000i备份资料扩大既有干扰；不得据此给CL-6000i借用版本规则。
- 050：四项可能原因完整保留；新故障资料350CPS与培训300CPS存在可选阈值差异。题目只问原因，不能扩展为阈值或光度计已损坏。基线已含同类其它格式资料。
- 更完整的新旧来源清单、实际入选文本及人工检查依据见`context_delta_review.jsonl`、`negative_review.json`、`source_visual_review.json`。

## 5. 离线模型：可运行，但判断不可靠

固定Qwen3-4B-Instruct-2507的Q4_K_M量化，仓库bartowski，revision `ae44f08e1392f39c0e474af10c3ff8355c8b6688`，权重SHA `2fde00ce69dd4899c70d020845e2638353015bba0fdf161b3eb965f2bca4464e`。`llama-cpp-python==0.3.36`，CPU四线程、ChatML、seed42、temperature0、4096上下文、输出上限768。锁定上游revision、模型路径、运行参数见protocol/model manifest。

模型只输入原问题+本候选正文，不传Gold、题号、正确来源或人工标签。引用必须逐字存在且位置唯一；非法JSON、错误枚举、无效引文、超长输入、缓存损坏均弃权，不按质量重试。

- 12个真实来源组件对各独立推理两次，共24次。12/12输出标签与引文稳定；稳定不等于正确。
- 组件08连续两次把LVC试剂/校准品处理判为CSDB的partial compatible。引用真实，报警对象错了，采用安全门槛失败。
- 组件07含义表被当作方法题direct；组件10完整CSDB表反而判partial。仅在这些小样本标签上，direct精确率2/4、召回2/10；6次预期冲突中没有判出conflict。不能将这些比例当成全语料质量估计。
- 组件4次引文校验失败回退中性。业务路由282个候选（76题，另15题不需要模型），220有效、62无效。无效主要42条引文不存在、10条applicability枚举非法，其余10条为缺引文/引文格式或重复歧义。
- M/RM共享完全相同的首次判别结果，不能算两次独立模型验证。仅组件重复是独立生成。没有按好坏重试。
- 路由在业务推理前冻结为P和B前8集合的对称差，不用Gold。该路由会遗漏排序交界处的中性干扰，002已证实；不能声称全候选语义校验已完成。
- BGE512token，完整1820候选中512个正文被截断；模型路由282中73个也存在该差异。逐条可见范围见`model_visibility.json`。组件12段均全部在BGE可见范围，错误不能归因于BGE看不全。本轮模型没有比B新增完整Gold，不做“更强语义能力”主张；模型组件门槛已失败，等可见正文收益对照及独立模型采用验收不再推进，不以继续换模型/提示寻找好分数。

没有模型-only新收益，已有确证错误，因此不值得增加这套模型作为排序决策器。保留显式注入实验接口与错误诊断，不接默认配置。

## 6. 下游剩余问题：不能全算在重排

R正式仍未完整的3题：

1. 019：处理页14去重第30（融合31），根本不在20个重排候选；场景页13在。换比较器不能补出未进入输入的解决步骤。应后续单独验证重排入口召回保留。
2. 028：所需总述页49未在固定关键词/向量候选找到；临床原因页58已出现。属于上游入口/匹配范围问题，不能报作本轮修复。
3. row_holdout2-001条码：正确表格重排2、3，条件后15、16，被Top8排除。硬对象“样本”的表述在正确表格未明说→unknown；安装资料写“样本”→得分2被排前。问题发生在条件排序，不能把这一损失归因给重排，也不能解决成删除全部unknown。

12道新题中08、11有正确候选，却在原topic关键门禁被清空；解析硬topic含“水质检测传感器是”“隔膜泵是”。这是解析与门禁的组合误伤，当前冻结未改。

## 7. 新12题与业务审核缺口

另取参数单位、功能对象、条件步骤、多目标分支各3题，全部pending。26条引文在原PPT和本次索引均核对；实际slide关系按presentation顺序读取，不直接把slide文件名当页码。12张原页已视觉复核。新目标chunk未出现在旧91题候选中，但源文件可以重叠，不能称整文档完全独立。

P/B/R均10/12实际Context完整，无新增Gold得失；不是泛化有效的充分证据。M/RM组件未过门槛，不在新集继续调试。JSON/JSONL一致、加载器和来源映射通过。具体结果见fresh_*及new_holdout_visual_review.json。

新03问排空流程，但冻结标注只核对两周与界面路径，未覆盖原文拔出底物/磁分离注液针前提；新09应限定所述单向阀结构，而非所有阀型。它们是检索哨兵，不是可直接用于答案完整性评分的业务批准Gold。不得因10/12而声称12道标准答案已审核完毕。后续业务审核补齐流程必答项前不转reviewed。

## 8. 开发内容与通用边界

本轮新增/修改的运行代码：

| 文件 | 本轮作用 |
|---|---|
| local_full_text_search/rag/reranking/evidence_features.py | 通用ASCII标识符、复用完整结构行核验；无法绑定保持未知 |
| local_full_text_search/rag/reranking/local_semantic_judge.py | 可选进程内模型、逐字唯一引文、预算、版本化缓存与失败回退 |
| local_full_text_search/rag/reranking/evidence_ordering.py | 显式注入、分档稳定排序、原文offset/SourceRef、每次改序理由 |
| local_full_text_search/rag/reranking/local_bge.py | 可选ordering_policy，保存原logit；原默认混合/保护不动 |
| local_full_text_search/rag/models.py、trace.py | 新增默认诊断字段，不破坏旧构造调用 |
| requirements-rag-semantic.txt | 独立可选CPU运行时，不强制通用RAG安装模型 |
| tests/rag/test_rerank_evidence_ordering.py | 原文绑定、未知、部分支持、对象行、错误缓存、无配置兼容等 |
| tools/validate_rerank_semantic.py、probe_rerank_semantic_native.py | 五组冻结回放与真实retrieve入口、断网断生成核验 |
| tools/analyze_rerank_semantic.py、prepare_rerank_semantic_holdout.py、audit_rerank_model_visibility.py | 来源判分、pending新题、实际token可见性审计 |
| tools/render_rerank_semantic.py、render_selection_factorial.py、check_selection_report_ui.cjs | 原生HTML五组、可变重排阶段、原资料页按钮及UI测试 |
| tools/finalize_rerank_semantic.py、write_rerank_semantic_conclusion.py | 保护文件、候选/条件恒定性、验收与本文复现 |

没有新增题号/文件路径/页码业务分支，所有题号只在评测与诊断中。没有修改原Gold或现有业务规则来适配输出；工作区原有其它未提交改动保留，不算本轮产出。

计划中的逐答案目标/关系/方法分支完整输出schema本轮未实现成可靠能力：现实现仅片段级support、applicability、最多2句原文与reason。对象判别未过组件门槛，因此不继续堆覆盖规划器，明确限为实验实现，不能称已完成生产级方法理解。

## 9. 验证执行、故障与复现命令

完成：91题P基线Context/条目/分数逐字复现；五组各91题；五组候选20条、raw CE、条件判定保持相同；原8个保护文件哈希无漂移。

真实AnswerService.retrieve：P/R重点5题各一次、12新题P共22次；M/RM重点5题各一次共10次。与对应回放的候选、顺序、Context核验通过。M/RM真实入口复用冻结判别，不冒充再次独立推理。所有网络审计attempts为空，生成器与API凭据入口被阻断。

测试：聚焦30 passed；全量517 passed、10 failed、1 skipped、1 xfailed。10项均为历史source_consensus失败且模块哈希未变，未新增确定测试失败，**并非全绿**。聚焦包含全量运行后新增的两个缓存校验测试，不把两次结果相加当总用例数。原资料定位/设置/报告兼容另跑24 passed。最初一次Windows临时目录rename权限异常已隔离重跑及最终全量通过，原失败日志仍保留。

UI使用捆绑Node/无头Edge，五组切换、91题导航、Context前后、未调用API标识、来源按钮allowlist及pending页码分别检查。系统D:/js/node.exe本次启动异常，改用本机已有捆绑Node，不涉及业务推理。原资料外部窗口启动在UI批测中mock，真实本地定位器另有兼容测试；不声称2958个位置都逐个打开过。

主要命令（仓库根目录）：

```powershell
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag/test_rerank_evidence_ordering.py tests/rag/test_rerank_support.py tests/rag/test_selection_a_adoption.py -q
.venv/Scripts/python.exe -X utf8 -m pytest tests/rag tests/test_settings_service.py tests/test_open_location.py -q --junitxml=reports/rag_rerank_semantic_20261116/pytest_final.xml
.venv/Scripts/python.exe -X utf8 -m tools.validate_rerank_semantic --phase model
.venv/Scripts/python.exe -X utf8 -m tools.validate_rerank_semantic --phase replay
.venv/Scripts/python.exe -X utf8 -m tools.probe_rerank_semantic_native
.venv/Scripts/python.exe -X utf8 -m tools.probe_rerank_semantic_native --semantic
.venv/Scripts/python.exe -X utf8 -m tools.analyze_rerank_semantic --fresh
.venv/Scripts/python.exe -X utf8 -m tools.audit_rerank_model_visibility
.venv/Scripts/python.exe -X utf8 -m tools.finalize_rerank_semantic
.venv/Scripts/python.exe -X utf8 -m tools.write_rerank_semantic_conclusion
.venv/Scripts/python.exe -X utf8 -m tools.render_rerank_semantic
```

`--phase model`会重新跑组件两次；正常复查无需重复模型生成，用已保存判别replay即可。不要把缓存命中当重复可靠性验证。

## 10. 计划完成状态与采用

| 步骤 | 结果 |
|---|---|
| 冻结/复现 | 通过，91/91，无保护文件漂移 |
| 规则/兼容实现 | 完成，严格结构只触发3题，独立增益不足 |
| 本地模型可运行性 | 通过CPU执行，未调用API |
| 本地模型可靠性 | 失败：重复错报警对象、含义误当完整方法、62/282无效 |
| 五组固定候选比较 | 完成；B基座净增2，M/RM新增负例 |
| 真实API前链路 | 完成32次，重点与回放一致 |
| 新12题 | P/B/R预验收完成10/12，pending；模型采用分支因门槛失败不推进 |
| 生成答案/答案质量 | 用户本轮要求不调用API，明确未运行 |
| 报告/配置 | 完成HTML前后对照和页定位，默认未切换 |

可合入但默认关闭：原始分数、来源绑定、可见性、改序原因、无效输出回退与对照报告基础设施。无需更改检索/生成默认。

可采用的规则：作为只读结构诊断可靠；作为影响排序的R目前没有超过B的独立完整证据收益，暂不作为全量默认策略。

暂不采用：当前M/RM模型排序；B/R全量替换也不在本轮自动落地。不是说BGE无用，而是已确认两题收益与新增错平台片段需同时保留证据，不能扩大结论。

需要业务确认：新12题完整答案检查项（尤其新03前提、新09结构范围），以及050可选阈值的版本适用差异。已确认004/006恢复无需再让用户裁决是否是真增益。

下一轮最小改进：**只修条件/对象绑定的排序误伤与错场景放行**，固定本轮B基座作实验对照，不改召回和生成。先同时覆盖“正确表格条件未知却被推后”和“明确另一平台数值混入”两个相反边界；不对全部未知扣分，不加型号词典穷举。验收须保住004/006、恢复条码正确表，并避免新Medica15μL/002错板片段。019另立重排入口保留实验，不能混在条件实验报功。本轮不继续换模型或叠新层。
