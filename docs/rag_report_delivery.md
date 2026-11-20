# RAG 验证报告交付约定

**最新验证（2026-11-15）**：`reports/rag_condition_binding_20261115/report.html`。条件绑定四组实验已完成；组合方案存在方法分支误删，暂不采用。正式实现仍为下述已采用A。详情见`docs/rag_condition_binding_validation_20261115.md`。

**当前状态（2026-11-13）：多层回答重构仍已回退；随后按用户明确授权采用最终选段实验A（取消单文档配额，保留TopK8及12000字符预算）。最新报告为 `reports/rag_selection_a_adoption_20261113/report.html`，详见 `docs/rag_selection_a_adoption_20261113.md`。注意“选段A”与更早支持图实验的“A基线”不是同一个实验编号。下文其他入口为历史记录。**

每轮验证默认生成 `report.html`。2026-11-13 用户指定以 `C:/Users/50263194/AppData/Local/LocalFullTextSearch/rag_eval_runs/20261030_143500_8c27af/report.html` 为格式基准：复用应用原有指标卡片、阶段趋势、热力图、题目总表以及左侧题目/阶段/排名导航。在此基础上增加前后回答并列、阶段引文数量对照和 A/B 查看切换。Markdown 用作说明附件，不替代 HTML。

## 必须呈现的内容

- 同一问题、相同 Gold 版本的前后回答、判分和缺失项。
- 各阶段完整引文覆盖、排名和候选排除理由。
- 实际发送的 Context、来源与 QueryPlan 激活状态。
- Gold 原文、必答项、禁答项、审核状态及原资料定位按钮。
- 正式集合与 pending/disputed 诊断集合分开，题量变化不得解释成算法改善。
- 冻结输入复用答案时保留来源，不计为独立重复实验；不同运行失败不能用成功覆盖。

## 本轮入口

最新入口：`reports/rag_support_graph_20261113/report.html`。

本轮为通用证据支持图、字段级回退、审核与澄清重构。报告展示冻结 A 与最终 B 对照；B 的原生完整运行另存于 `B.jsonl`，最终页面使用对全部保存审核结果应用确定性渲染修复的 `B_effective.jsonl`，未重新抽取生成答案。两者及重放后的裁判变化均可追溯。整体业务验收未通过，不能把配置已启用解释为已通过上线验收。

历史通用化报告仍保留在 `reports/rag_general_refactor_20261113/report.html`。

双击同目录 `open_report.cmd`，即可从浏览器报告点击打开原 PDF 指定页或 PowerPoint 指定幻灯片。该启动器启动只监听本机的临时服务，使用随机令牌及报告来源清单，不能通过请求传入任意文件路径。关闭启动器窗口即停止服务。直接双击 HTML 仍能离线阅读；浏览器不能直接调用 PowerPoint，需要通过启动器使用定位按钮。

页码从 1 开始。PPT 的 Gold `page` 按幻灯片解释，不作为打印页码。无法唯一确认的同名文件显示失败，不自动选择。ZIP 内 PDF 解压到系统临时缓存后打开准确页，原包不改变。打开失败或阅读器不支持定位时会显示实际返回信息，不把“仅打开文件”视为定位成功。

## 生成与复现

```powershell
.venv/Scripts/python.exe -m tools.report_business_repair --candidate verified_all --answers verified_answers_B --output final_report
# 只重建页面，不重新运行检索、生成和裁判：
.venv/Scripts/python.exe -m tools.render_business_report
# 手动启动原资料定位服务：
.venv/Scripts/python.exe -m tools.serve_rag_report --report reports/rag_business_repair_20261112/report.html
```

本轮生成命令为 `.venv/Scripts/python.exe -m tools.report_general_refactor`，只重建原生格式可运行 `.venv/Scripts/python.exe -m tools.render_general_native_report`。原生布局、导航和证据高亮复用 `evaluation/reporting/service.py`，由离线工具追加对照与原资料按钮；不修改被冻结的判分模块。`comparison.py` 提供来源清单准备，但长折叠列表不再作为最终格式。不能将报告中的固定文件路径或题号用于业务运行。

最新支持图报告重建命令（不重新生成模型答案）：

```powershell
.venv/Scripts/python.exe -m tools.report_support_graph
.venv/Scripts/python.exe -m tools.finalize_support_graph_report
.venv/Scripts/python.exe -m tools.render_general_native_report --native --output-dir reports/rag_support_graph_20261113
```

完整实现、验收失败项和复现步骤见 `reports/rag_support_graph_20261113/conclusion.md`。

## 清理边界

只删除能重新生成的缓存、且不影响追溯的临时产物，以及无依赖的字节完全相同副本。删除前核对 SHA-256、引用和路径边界，保留删除清单及副本保留位置。

2026-11-13 清理记录见 `reports/cleanup_20261113.json`。没有删除历史 Gold、有效回归数据、原始回答、失败运行、各轮实验清单或原始业务资料。即使两轮实验清单字节相同，只要仍是各轮加载或审计入口，也保留各自文件。
