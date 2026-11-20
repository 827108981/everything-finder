# 向量包装与结构化表示：开发验证结果

暂不采用新的默认包装或结构策略。当前代码提供可隔离验证的通用能力；生产仍使用原包装与原索引。

| 方案 | 正式完整Gold@50 | 原诊断完整Gold@50 | 新来源诊断完整Gold@50 | 正式MRR |
|---|---:|---:|---:|---:|
| W0 | 46/46 | 37/40 | 20/20 | 0.6967 |
| F10 | 46/46 | 39/40 | 20/20 | 0.7047 |
| F11 | 46/46 | 39/40 | 20/20 | 0.7047 |

## W0

向量新增完整引文：[]；新增丢失：[]。

实际Context新增完整引文：[]；丢失：[]。

独立来源收益组：[]；源对象适用性复核：no_changes。

## F10

向量新增完整引文：['rag-row-holdout-003', 'evidence-flow-independent-05', 'evidence-flow-independent-06']；新增丢失：[]。

实际Context新增完整引文：[]；丢失：[]。

独立来源收益组：[]；源对象适用性复核：pending。

## F11

向量新增完整引文：['rag-row-holdout-003', 'evidence-flow-independent-05', 'evidence-flow-independent-06']；新增丢失：[]。

实际Context新增完整引文：[]；丢失：[]。

独立来源收益组：[]；源对象适用性复核：pending。

## 因素与边界

W0–W5保持52282个原逻辑窗口及原跨度；只改变编码文字。F10/F11加入同一批原生PPT表格行；增加向量数量与结构本身的收益不能混算。F10_count_matched提供固定数量替换敏感性。

W6：No two non-regressing single views have complementary complete-Gold gains; W6 not applicable.

PDF、图片表格恢复未实现；没有可信标题的Word/PDF使用完整文件名hint，不假装标题提取已覆盖所有格式。复杂合并表格保留原表示并记录未覆盖。

新增20题全部pending；资料出处和引文已核对，不代表业务人员正式验收。新Top5对象/范围复核清单见top5_applicability_review.json。原已审核测试集及007/040状态未改变。

未调用生成或裁判API，不报告答案正确率。019、IMFH1、条码等如果证据在向量之后丢失，按实际最早环节记录，不能用下游失误否认本轮向量层收益。

## 可复现命令

` .venv/Scripts/python.exe -X utf8 -m tools.run_vector_packaging_20261116 --resume `

单元/兼容测试：69项，失败0。API调用0。主报告report.html；open_report.cmd支持原资料位置按钮。
