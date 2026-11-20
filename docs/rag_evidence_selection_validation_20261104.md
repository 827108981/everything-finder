# 最终证据选择独立验证（2026-11-04）

结论：本次独立验证未通过，保留现有默认规则。019的正确原文可以补入上下文，但新选择器会丢失015中CL-6000i的诊断入口，同时引入022的错场景分支；仅取消文档数量上限也未通过预设保留检查。因此按计划停止新答案生成，本轮不报告答案通过率提升。

**计分边界：严格Gold丢失不等于同等数量事实丢失。** 003、046、047、049的引文在其他来源中完整存在；001、014、032、048也出现同内容或排版变化的候选替代来源，需复核适用性后才可补标。它们不能笼统当作真实事实缺失。本轮不修改Gold、不自动接受这些替代来源，也不据此把34/44直接说成真实事实准确率。

**仍存在足以否决该方案的真实缺口：** 015问两款机型的诊断入口，新上下文仍有CL-8000i相关资料，却没有CL-6000i“应用→诊断→诊断→M1”的路径，无法完整回答原问。022新增的个别项目DEV说明不能直接回答几乎全部项目同时偏低的问题。

原因分析：新选择器重新按冻结的综合分数及文字相似度取舍，改变了原有“业务条件优先及名次保护”形成的顺序。文字相似度不是业务重复度，通用高分背景也不等于某个机型或子问题的专属依据。原硬冲突门槛虽然保留，仍不足以保护不同子问题各自必需的证据。这个结果否定的是本次具体的分组＋MMR实现，不能推广为所有证据组方法都无效。

本轮没有调整权重反复尝试，也没有因019改善而忽略回退。quota_free是消融诊断，不能作为已验证通过的替代方案。023新增负例与正向内容同在一张报警表中，042新增负例是相关但不足的背景，负例次数不等于错误回答次数；022的场景不符与015的入口缺失单独核验。


固定上一轮四条上游路径的59题，共236组候选、分数及条件判定；先逐字复现旧上下文，再比较最终选择。正式48题，024与回归10题单列待审，007排除。
规则预先固定：连续页成组、精确正文去重、MMR相关性权重0.7、来源只作平分时偏好；总段数8、字数12000、机型冲突及分数门槛保持不变。无题号/答案/Gold访问。quota_free仅作“去掉文档配额”的消融诊断。

结论：**证据保留门槛未通过，停止生成阶段，暂不启用。** 原因是至少一道题丢失原有Gold或增加已标注负例。

|路径|方法|正式题Gold命中|平均覆盖|指定反例次数|全部59题上下文变化|
|---|---|---|---|---|---|
|baseline|quota_free|39 → 40/44|83.3% → 87.5%|11 → 13|19|
|baseline|grouped_mmr|39 → 33/44|83.3% → 69.5%|11 → 12|53|
|A|quota_free|39 → 41/44|83.3% → 89.8%|11 → 13|19|
|A|grouped_mmr|39 → 34/44|83.3% → 71.8%|11 → 12|53|
|B|quota_free|39 → 40/44|83.3% → 87.5%|10 → 12|18|
|B|grouped_mmr|39 → 33/44|83.3% → 69.5%|10 → 12|53|
|A+B|quota_free|39 → 40/44|83.3% → 87.5%|10 → 12|18|
|A+B|grouped_mmr|39 → 34/44|83.3% → 71.8%|10 → 12|53|

## baseline 主方案逐题检查

Gold丢失：rag-biz-001、rag-biz-003、rag-biz-014、rag-biz-015、rag-biz-032、rag-biz-047、rag-biz-048、rag-biz-046、rag-biz-049。
Gold补回：rag-biz-021、rag-biz-023。
新增指定负例：rag-biz-022、rag-biz-023、rag-biz-042。

- rag-biz-001（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-001:positive:1']；补回[]；新增负例[]。
- rag-biz-003（formal48）：覆盖 1.0 → 0.6；丢失['rag-biz-003:positive:1', 'rag-biz-003:positive:2']；补回[]；新增负例[]。
- rag-biz-014（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-014:positive:1']；补回[]；新增负例[]。
- rag-biz-015（formal48）：覆盖 1.0 → 0.5；丢失['rag-biz-015:positive:2']；补回[]；新增负例[]。
- rag-biz-019（formal48）：覆盖 0.0 → 0.0；丢失[]；补回[]；新增负例[]。
- rag-biz-021（formal48）：覆盖 0.6666666666666666 → 1.0；丢失[]；补回['rag-biz-021:positive:2']；新增负例[]。
- rag-biz-022（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-022:hard_negative:1']。
- rag-biz-023（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-023:positive:1']；新增负例['rag-biz-023:hard_negative:1']。
- rag-biz-032（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-032:positive:1']；补回[]；新增负例[]。
- rag-biz-042（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-042:hard_negative:1']。
- rag-biz-047（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-047:positive:1']；补回[]；新增负例[]。
- rag-biz-048（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-048:positive:1', 'rag-biz-048:positive:2', 'rag-biz-048:positive:3', 'rag-biz-048:positive:4', 'rag-biz-048:positive:5']；补回[]；新增负例[]。
- rag-biz-046（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-046:positive:1']；补回[]；新增负例[]。
- rag-biz-049（formal48）：覆盖 0.5 → 0.0；丢失['rag-biz-049:positive:1']；补回[]；新增负例[]。

## A 主方案逐题检查

Gold丢失：rag-biz-001、rag-biz-003、rag-biz-014、rag-biz-015、rag-biz-032、rag-biz-046、rag-biz-048、rag-biz-047、rag-biz-049。
Gold补回：rag-biz-019、rag-biz-021、rag-biz-023。
新增指定负例：rag-biz-022、rag-biz-023、rag-biz-042。

- rag-biz-001（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-001:positive:1']；补回[]；新增负例[]。
- rag-biz-003（formal48）：覆盖 1.0 → 0.6；丢失['rag-biz-003:positive:1', 'rag-biz-003:positive:2']；补回[]；新增负例[]。
- rag-biz-014（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-014:positive:1']；补回[]；新增负例[]。
- rag-biz-015（formal48）：覆盖 1.0 → 0.5；丢失['rag-biz-015:positive:2']；补回[]；新增负例[]。
- rag-biz-019（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-019:positive:1']；新增负例[]。
- rag-biz-021（formal48）：覆盖 0.6666666666666666 → 1.0；丢失[]；补回['rag-biz-021:positive:2']；新增负例[]。
- rag-biz-022（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-022:hard_negative:1']。
- rag-biz-023（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-023:positive:1']；新增负例['rag-biz-023:hard_negative:1']。
- rag-biz-032（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-032:positive:1']；补回[]；新增负例[]。
- rag-biz-042（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-042:hard_negative:1']。
- rag-biz-046（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-046:positive:1']；补回[]；新增负例[]。
- rag-biz-048（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-048:positive:1', 'rag-biz-048:positive:2', 'rag-biz-048:positive:3', 'rag-biz-048:positive:4', 'rag-biz-048:positive:5']；补回[]；新增负例[]。
- rag-biz-047（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-047:positive:1']；补回[]；新增负例[]。
- rag-biz-049（formal48）：覆盖 0.5 → 0.0；丢失['rag-biz-049:positive:1']；补回[]；新增负例[]。

## B 主方案逐题检查

Gold丢失：rag-biz-003、rag-biz-001、rag-biz-014、rag-biz-015、rag-biz-032、rag-biz-046、rag-biz-047、rag-biz-049、rag-biz-048。
Gold补回：rag-biz-021、rag-biz-023。
新增指定负例：rag-biz-022、rag-biz-023、rag-biz-025、rag-biz-042。

- rag-biz-003（formal48）：覆盖 1.0 → 0.6；丢失['rag-biz-003:positive:1', 'rag-biz-003:positive:2']；补回[]；新增负例[]。
- rag-biz-001（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-001:positive:1']；补回[]；新增负例[]。
- rag-biz-014（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-014:positive:1']；补回[]；新增负例[]。
- rag-biz-015（formal48）：覆盖 1.0 → 0.5；丢失['rag-biz-015:positive:2']；补回[]；新增负例[]。
- rag-biz-019（formal48）：覆盖 0.0 → 0.0；丢失[]；补回[]；新增负例[]。
- rag-biz-021（formal48）：覆盖 0.6666666666666666 → 1.0；丢失[]；补回['rag-biz-021:positive:2']；新增负例[]。
- rag-biz-022（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-022:hard_negative:1']。
- rag-biz-023（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-023:positive:1']；新增负例['rag-biz-023:hard_negative:1']。
- rag-biz-025（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-025:hard_negative:1']。
- rag-biz-032（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-032:positive:1']；补回[]；新增负例[]。
- rag-biz-042（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-042:hard_negative:1']。
- rag-biz-046（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-046:positive:1']；补回[]；新增负例[]。
- rag-biz-047（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-047:positive:1']；补回[]；新增负例[]。
- rag-biz-049（formal48）：覆盖 0.5 → 0.0；丢失['rag-biz-049:positive:1']；补回[]；新增负例[]。
- rag-biz-048（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-048:positive:1', 'rag-biz-048:positive:2', 'rag-biz-048:positive:3', 'rag-biz-048:positive:4', 'rag-biz-048:positive:5']；补回[]；新增负例[]。

## A+B 主方案逐题检查

Gold丢失：rag-biz-001、rag-biz-003、rag-biz-014、rag-biz-015、rag-biz-032、rag-biz-046、rag-biz-047、rag-biz-048、rag-biz-049。
Gold补回：rag-biz-019、rag-biz-021、rag-biz-023。
新增指定负例：rag-biz-022、rag-biz-023、rag-biz-025、rag-biz-042。

- rag-biz-001（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-001:positive:1']；补回[]；新增负例[]。
- rag-biz-003（formal48）：覆盖 1.0 → 0.6；丢失['rag-biz-003:positive:1', 'rag-biz-003:positive:2']；补回[]；新增负例[]。
- rag-biz-014（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-014:positive:1']；补回[]；新增负例[]。
- rag-biz-015（formal48）：覆盖 1.0 → 0.5；丢失['rag-biz-015:positive:2']；补回[]；新增负例[]。
- rag-biz-019（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-019:positive:1']；新增负例[]。
- rag-biz-021（formal48）：覆盖 0.6666666666666666 → 1.0；丢失[]；补回['rag-biz-021:positive:2']；新增负例[]。
- rag-biz-022（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-022:hard_negative:1']。
- rag-biz-023（formal48）：覆盖 0.0 → 1.0；丢失[]；补回['rag-biz-023:positive:1']；新增负例['rag-biz-023:hard_negative:1']。
- rag-biz-025（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-025:hard_negative:1']。
- rag-biz-032（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-032:positive:1']；补回[]；新增负例[]。
- rag-biz-042（formal48）：覆盖 1.0 → 1.0；丢失[]；补回[]；新增负例['rag-biz-042:hard_negative:1']。
- rag-biz-046（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-046:positive:1']；补回[]；新增负例[]。
- rag-biz-047（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-047:positive:1']；补回[]；新增负例[]。
- rag-biz-048（formal48）：覆盖 1.0 → 0.0；丢失['rag-biz-048:positive:1', 'rag-biz-048:positive:2', 'rag-biz-048:positive:3', 'rag-biz-048:positive:4', 'rag-biz-048:positive:5']；补回[]；新增负例[]。
- rag-biz-049（formal48）：覆盖 0.5 → 0.0；丢失['rag-biz-049:positive:1']；补回[]；新增负例[]。

## 边界

本轮验证最终选择机制；baseline/B路径原本没有019答案页，不可能由选择器凭空补出。A/A+B已有答案页，才可检验它能否被保留。
050的PPT替代来源仍单列参考，未改正式Gold。真实事实覆盖不能仅靠固定Gold命中率判断。候选相关性和条件识别错误不会被本轮修复。
若保留检查失败，按事先约定不继续批量生成；没有新生成结果时不得报告答案准确率提升。

## 原18题与新增题固定集合（A路径，主方案）

|集合|正向证据题数|严格Gold命中变化|平均覆盖变化|
|---|---:|---|---|
|original17|15|13 → 11|86.7% → 67.3%|
|added31|29|26 → 23|81.6% → 74.1%|
|regression10_pending|10|10 → 10|100.0% → 100.0%|

5项选择器保护测试通过；全部236组旧上下文逐字复现；472组选择结果均遵守8段/12000字符预算。索引、模型分数、Gold、业务代码及默认设置未变化。
