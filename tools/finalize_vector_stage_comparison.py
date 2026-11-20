"""Summarize measured vector-only experiments without changing runtime or Gold."""
import json
from collections import Counter
from pathlib import Path
from tools.compare_vector_stage_20261116 import OUT, LABELS, summarize
from tools.validate_selection_factorial import ROOT, write, sha


def main():
    read=lambda name:json.loads((OUT/name).read_text(encoding='utf8'))
    p=read('protocol.json');metrics=read('metrics.json');audit=read('execution_audit.json')
    controls=read('representation_controls.json');ui=read('report_ui_audit.json')
    diagnostics=read('case_diagnostics.json');diff=read('differences.json')
    arms={a:[json.loads(x) for x in (OUT/(a+'.jsonl')).read_text(encoding='utf8').splitlines()] for a in LABELS}
    ids=p['case_ids']
    for arm,rows in arms.items():
        assert [r['id'] for r in rows]==ids
        assert summarize(rows)==metrics[arm]
        assert all(r['api_calls']==0 and set(r['candidate_stages'])=={'vector'} for r in rows)
        assert all(len(r['candidate_stages']['vector'])==50 for r in rows)
    assert len(ids)==len(set(ids))==91
    assert len(controls['original_quote_verification'])==22
    assert all(q['original_verified'] and q['index_verified'] for q in controls['original_quote_verification'])
    assert not read('filter_metadata_audit.json')['errors']
    assert not audit['network_attempts'] and not controls['network_attempts']
    assert not read('report_format_audit.json')['source_misses'] and not ui['page_errors']
    drift=[path for path,expected in p['protected'].items() if sha(path)!=expected]
    assert not drift
    testlog=(ROOT/'reports/vector_stage_tests_20261116.log').read_text(encoding='utf8')
    assert '36 passed' in testlog and 'failed' not in testlog
    write(OUT/'final_acceptance.json',dict(
        status='diagnostic experiments complete; no production fix adopted',bundle=p['bundle'],cases=91,
        statuses=dict(Counter(r['review_status'] for r in arms['baseline'])),
        positive_cohorts={'formal_reviewed':46,'diagnostic_pending':39,'diagnostic_disputed':1},
        arms=list(LABELS),metric_recomputed_equal=True,case_sets_equal=True,candidates_per_query=50,
        actual_chroma_vectors_compared=52282,fresh_model_recomputed_sample=32,
        source_local_cases=15,source_local_representations=192,original_and_index_quotes_verified=22,
        tests={'command':'.venv/Scripts/python.exe -m pytest tests/rag/test_vector_unit_v2.py tests/rag/test_vector_query_variants.py tests/rag/test_chroma_vector_store.py tests/rag/test_stage_scoring.py tests/rag/test_approved_gold_sources.py tests/rag/test_query_plan_native.py tests/rag/test_report_generation.py tests/rag/test_comparison_report.py -q','passed':36},
        ui=ui,api_calls=0,network_attempts=[],protected_drift=drift,
        production_changed=False,default_changed=False,gold_changed=False,index_published=False,
        limitations=['local source replacement is not a whole-index rebuild or held-out validation',
                     'no alternative embedding model compared',
                     'arbitrary root/extension-filtered queries not comprehensively tested',
                     'new arms do not execute fusion/rerank/Context/answers']))
    lines=[
        '# 整个向量召回阶段对照实验与结论（2026-11-16）',
        '',
        '**结论：不只有切片问题。确认有表格关系表示不足、来源包装改变语义排名两个因素；另有语义匹配不足尚未分清具体原因。当前3道真正缺失题，不是向量库未收录或ANN近似搜索漏搜造成。**',
        '',
        '已连续完成91题真实向量基线、4组全量单因素控制、全库完整性核查、15题192个局部表示控制、22条引文原文与索引核对、36项测试，以及HTML报告检查。本轮API调用0；不修改生产算法、持久化设置、Gold或已发布索引。这里交付的是诊断结论，不宣称已统一修好向量召回。',
        '',
        '## 1. 入口、冻结范围与指标',
        '',
        '- [交互式报告](../reports/rag_vector_stage_comparison_20261116/report.html)：使用程序原生诊断报告格式，支持切换A/B/C/D/E、查看91题候选和Gold、打开对应原资料页。',
        '- [启动报告及原资料定位](../reports/rag_vector_stage_comparison_20261116/open_report.cmd)：直接打开HTML可阅读；运行此命令后，页内按钮可通过本地服务定位PDF页/PPT幻灯片。',
        '- 当前Gold清单：`tests/rag/fixtures/rag_active_suite_v5_1.json`。50 reviewed、40 pending、1 disputed，共91题。',
        '- 正式有正向证据46题；诊断有正向证据40题（39 pending、1 disputed）。007争议题只参与诊断，不计入正式结果。4道正式拒答题和1道诊断拒答题没有正向Gold，不纳入正向召回分母。',
        '- AI bundle：`g000033-71ec3c52f3`；11769个父块；unit_v2共52282个子窗口。',
        '- 模型：`BAAI/bge-large-zh-v1.5`，revision `79e7739b6ab944e86d6171e44d24c997fc1e0116`，1024维，上限512token。不是重排模型。',
        '- 当前窗口：目标128token、最大192、重叠32。向量返回Top50父块，通常先查询400个子窗口，以最佳子窗口分数作为父块分数。',
        '- 本轮全部使用同一新版Gold，不拿之前44/46与本次46/46作算法提升对比。此前修复来源登记改变了计分依据，本轮没有再次修改Gold。',
        '',
        '**完整Gold@50**：返回的前50父块正文按来源、原页和原文引文匹配，覆盖该题每一条Gold事实；已批准替代来源按同一事实的OR关系匹配。不把同页其他内容或其他文件中仅相关文字算作覆盖。它是候选父块证据覆盖，不代表所有事实都在最佳编码小窗口中，也不代表答案正确。',
        '',
        '**Hit@K**：前K父块至少找到一条完整Gold；**MRR**：第一条完整Gold排名的倒数均值，没找到记0。MRR只看第一条，因此可能在部分题丢失其他事实时仍然升高。报告保留完整覆盖和逐题增损，避免只看MRR。',
        '',
        '## 2. 五组全量对照',
        '',
        'A调用当前生产VectorRetriever、本地真实编码器和实际Chroma。B同一查询/资料向量做52282条穷举余弦搜索。C只把ANN取窗深度从400增加到至少1600，最终仍返回50父块。D相对B只移除查询编码指令。E相对B只把父块最佳窗口得分换成前两个窗口均分，单窗口沿用自身得分。',
        '',
        'A↔B判断近似搜索损失，A↔C判断原取窗深度；B↔D、B↔E分别隔离编码指令和父块聚合。D/E不是可直接采用的新方案。',
        '',
        '| 组别 | 正式完整@50 | 诊断完整@50 | 正式MRR | 诊断MRR | 比A新增引文丢失题 |',
        '|---|---:|---:|---:|---:|---|',
    ]
    for a in LABELS:
        m=metrics[a]
        lines.append(f'| {LABELS[a]} | {m["formal"]["complete"]}/46 | {m["diagnostic"]["complete"]}/40 | {m["formal"]["mrr"]:.4f} | {m["diagnostic"]["mrr"]:.4f} | {"、".join(diff[a]["losses"]) or "无"} |')
    lines += [
        '',
        '| A当前基线集合 | Hit@1 | Hit@3 | Hit@5 | Hit@50 |',
        '|---|---:|---:|---:|---:|',
        '| 正式46题 | 27 | 36 | 38 | 46 |',
        '| 诊断40题 | 19 | 27 | 29 | 37 |',
        '',
        '- B、C没有补回任何缺失Gold；independent-11首条Gold由36变37，完整覆盖不变。ANN与精确Top50存在少量成员差异，不能说两种搜索逐条完全相同。',
        '- D移除查询指令丢失006、row_holdout2-007，且045多出一条已标负例。虽然诊断MRR升高，仍不值得采用。',
        '- E前两窗均分丢失006、050，没有补回3道缺失题；多窗口互相稀释，不能因为少数题排名升高就替换现有最佳窗口汇总。',
        '- 当前已审核46题全部覆盖，仅表示本次已登记Gold在Top50父块中齐全；其中006排43、025排13、050排26，仍有排名和后续保留问题。',
        '',
        '## 3. 沿向量链逐环核对',
        '',
        '| 环节 | 本轮证据 | 可下结论与限制 |',
        '|---|---|---|',
        '| QueryPlan送入向量的查询 | 91/91与用户当前原问题相同，Adapter保持关闭 | 本轮不是改写漏掉型号或对象；不代表所有开启Adapter的查询也如此 |',
        '| 查询编码/截断 | 38–77token，0条超过512；保留原查询指令更稳 | 未发现查询截断；不能据此宣称模型语义理解没有问题 |',
        '| 资料编码/截断 | 已存窗口24–192token；32条重新编码抽查与实际存储一致 | 当前窗口不触发512截断；切断关系、语义稀释仍可能发生 |',
        '| 数据库收录/关联 | 全部52282实际Chroma向量与缓存逐条比较；11769父块全有窗口 | 没有漏写、孤儿、错父块ID；原始资料解析正确与否是另一件事 |',
        '| 数学与索引距离 | 实存向量有限且单位化，norm约0.99999982–1.00000024；索引使用cosine | 未发现归一化或距离配置混用 |',
        '| 元数据过滤 | 全部52282窗口的根路径、扩展名、source_type、unit_kind与父块一致；低置信OCR父块0 | 本轮91题无根路径/扩展名筛选；任意筛选查询未完整验证。不能把潜在过滤风险写成本轮缺失根因 |',
        '| ANN近似搜索 | B精确搜索覆盖与A相同 | 当前3道缺失不靠提高ANN精度解决；不是所有可能查询的数学保证 |',
        '| 取窗深度 | C从400扩到至少1600，覆盖不变 | 缺失资料本身在精确父块排序50以外，不是前400窗被同页挤满造成 |',
        '| 父块聚合 | E前两窗均分没有补回，反而丢006/050 | 保留现有max聚合；其他聚合方式未穷举 |',
        '| 资料文字表示 | 行级/正文去包装局部控制存在明显影响，也有恶化 | 必须进一步做全库候选索引，不能按题临时选一种表示 |',
        '',
        '全库缓存与实存逐条一致，不等于52282条都重新跑了模型；重新编码是均匀抽取32条，已明确区分。完整数据在execution_audit.json和filter_metadata_audit.json。',
        '',
        '## 4. 仍缺失的3题',
        '',
        '| 题目 | 当前问题 | 精确排序下正确证据位置 | 判断 |',
        '|---|---|---|---|',
        '| rag-row-holdout-003 | 从待机开始检测首样本，两系列分别多久 | 第149 | 表格型号列与启动时间行的关系表示有问题；原表头+正确行局部149→1 |',
        '| evidence-flow-independent-05 | A&T和Medica原理区别及如何辨别类型 | 通用原理说明692；模块事实最佳批准来源141，原指定表352 | 来源包装对指定模块表有明确影响，但原理说明仍没补回；不是一套表格规则就能全解 |',
        '| evidence-flow-independent-06 | BS-2800M与BS-430分别什么接口通信 | 同页第571 | 多型号表语义混合；逐行表示两目标分别92和146，仍未进50，不能宣称表格行级化已解决 |',
        '',
        '这3题均为pending，只能作诊断。在现有向量下，最终候选Top100仍缺3题；Top200仅补回row-003；Top1000才覆盖全部诊断Gold。单纯把50改1000可用大量候选兜底，但不能证明语义排序改善，也会改变下游竞争，因此本轮未采用。',
        '',
        '## 5. 还有哪些题属于资料表示问题',
        '',
        '以下比较是固定其余全库竞争向量，仅替换一个已审计父块的表示。不是全库索引重建，也不是独立验收。生成全部可用原表行后，选出与本题目标一致的行审计；不是把Gold答案人工写进生产窗口。',
        '',
        '| 题目 | 事实/来源 | 原指定来源排名 | 原表完整行表示 | 结论 |',
        '|---|---|---:|---:|---|',
        '| row-003 | BS-2800M服务培训，第6张，启动时间 | 149 | 1 | 已确认该来源对结构恢复敏感 |',
        '| row-006 | BS-5000产品介绍，第6张，ISE速度 | 11 | 3 | 新发现排名问题，原来不漏召回 |',
        '| row-009 | 同文件第8张，样本针清洗方式 | 50 | 1 | 新发现临界排名问题 |',
        '| row_holdout2-003 | 同文件第5张，反应体积 | 19 | 1 | 新发现排名问题 |',
        '| row_holdout2-009 | 同文件第5张，测量波长 | 14 | 1 | 新发现排名问题 |',
        '| independent-11 | 电解质第6张，适用机型和备件号 | 37 | 20 | 局部有改善，但非当前缺失 |',
        '| independent-06 | 板卡第10张，两型号通信 | 571 | BS-2800M为92；BS-430为146 | 有改善但仍未补回 |',
        '| row_holdout2-001 | 高速/中低速条码第17/14张 | 21/18 | 25/26 | 行级化并不总改善；两份原证据均已在向量Top50 |',
        '| row_holdout2-005 | 热工第5张 | 16 | 18 | 略差，不列为修复收益 |',
        '',
        '新增4道BS-5000题同源，不能写成4个独立领域验证成功。之前row-004、row-005的局部表格证据仍在旧报告；新版已登记正确替代来源后两题已覆盖，本轮不把旧Gold漏计当算法缺陷。',
        '',
        '### 与表头继承不同的第二个因素：来源包装',
        '',
        'independent-05的电解质第6张，仅把现有小窗的来源/位置包装（包括正文标签）去掉，正文和模型不改，该来源局部352→1；同页完整行表示反而955，整父块2125。说明“缺表头”不是统一解释，编码文字中的来源包装会改变语义权重。',
        '',
        '但不能全库删文件名：006去包装43→80，050指定PDF67→163，row-003149→1016。型号、部件信息也可能只在标题里。应将纯正文与带范围标题的表示作为独立全库实验因素，而非针对某题改文字。',
        '',
        '### 连续说明文字也有匹配偏后：006、025、ISE原理页',
        '',
        '- 006清洗液启停/空瓶切换在PDF第7页，是连续说明，不是该页表格恢复问题；最佳窗口父块43。去包装80、整父块816，简单合并或增大窗口不成立。它当前可被关键词补回，最终Context此前完整。',
        '- 025标本室温保存时间在临床培训第18张，当前父块13；去包装17、整父块61。不能凭排名稍后归为表格列断裂。',
        '- independent-05通用原理第5张692；去包装386、整父块2588，仍缺失。原因可能涉及窗口表示与模型对“直接/间接法”和提问语言的语义对齐；本轮没有换embedding模型，不能直接宣布“模型不行”。',
        '',
        '这三项未获得完整因果修复，只能明确排除部分粗略方案；下一轮必须继续保留它们，不能只测有利的表格例子。',
        '',
        '## 6. 不应算成向量缺失的其他阶段问题',
        '',
        '下表向量列来自本轮真实检索；后续列引用此前BGE基础排序运行 `reports/rag_bge_base_adoption_20261116/B.jsonl`，仅用于区分归属，不是本轮新组的Context效果。新组完全没有调用融合、重排和生成。',
        '',
        '| 题目 | 本轮向量 | 既有后续轨迹 | 阶段归属 |',
        '|---|---|---|---|',
        '| 019 | 场景第1、原因处理第36，两条完整 | 原因页融合31、去重30，重排Top20后缺失 | 后续候选预算/重排截断，不能说向量没找着 |',
        '| row-008 IMFH1 | 正确来源第1 | 既有重排14、条件后10，最终前8未保留 | 条件排序/最终预算，非本轮向量漏召回 |',
        '| row_holdout2-001 条码 | 高速21、中低速18，都有 | 既有重排2/3，条件后15/16，最终未保留 | 条件处理后丢失，非缺少跨文件向量候选 |',
        '| 050 | 批准的PPT第26/31，四原因完整 | 指定PDF仍67，但不能忽略正确替代PPT | 来源登记后的计分已修正，不把PDF排名67当无答案 |',
        '',
        '## 7. 采用结论与下一轮实施边界',
        '',
        '本轮不采用任何生产检索参数变更。保留查询指令、max父块聚合、当前ANN设置；B/C无覆盖收益，D/E出现退化。默认Adapter仍关闭。',
        '',
        '下一轮应统一修改资料到向量之间的“表示构建”，而不是一题一条规则：',
        '',
        '1. **结构化表示因素**：PPT解析保留表格行列与合并格关系；向量行绑定实际列头、对象、必要标题、单位和原页。PDF按原始坐标/版式单独验证，不能把PPT首行启发式直接套过去。普通说明段沿用原处理。',
        '2. **来源包装因素**：与结构化表示独立，比较现有包装、正文主体表示、保留必要范围标题的双表示。元数据继续保存完整路径、型号和原页；不从查询删除型号，不依据Gold决定某题用哪种表示。',
        '3. **全库四格对照**：当前表示、仅结构化、仅包装策略、两者组合。分别创建隔离候选索引，资料覆盖和模型相同，所有候选都重新竞争。不得拿本次局部149→1当最终收益。',
        '4. **统一验收**：先检查原文/表格字段对齐和引文来源，再跑91题向量及未参与表示调试的资料/问题；同时列完整覆盖、Hit@1/3/5、MRR和逐题增损。不能将本轮15题再称独立验收，也不能只挑同一BS-5000文件。',
        '5. **仍缺失再比模型**：如结构关系正确且来源包装有对照后，006/ISE原理/通信表仍差，再做固定同一表示的另一离线embedding模型对照，把模型因素与切片因素分开。不是现在就盲目换模型。',
        '',
        '验收必须包含：46道正式正向题原覆盖不丢；3道诊断缺失逐题是否补回；006/050/条码/热工等反向控制不恶化；新资料独立验证。若有权衡，列具体失去的事实与来源，由业务判断，不用总体MRR掩盖。',
        '',
        '## 8. 命令、验收与可复查产物',
        '',
        '```powershell',
        '.venv/Scripts/python.exe -X utf8 -m tools.compare_vector_stage_20261116',
        '.venv/Scripts/python.exe -X utf8 -m tools.probe_vector_stage_representations',
        '.venv/Scripts/python.exe -X utf8 -m tools.render_vector_stage_comparison',
        '.venv/Scripts/python.exe -m pytest tests/rag/test_vector_unit_v2.py tests/rag/test_vector_query_variants.py tests/rag/test_chroma_vector_store.py tests/rag/test_stage_scoring.py tests/rag/test_approved_gold_sources.py tests/rag/test_query_plan_native.py tests/rag/test_report_generation.py tests/rag/test_comparison_report.py -q',
        '```',
        '',
        '- 测试：36 passed。五组均91题、同顺序、同问题，每题50个向量父块，汇总从JSONL重新计算一致。',
        '- 浏览器检查：Headless Edge切换五组、91题导航、搜索、原资料按钮请求、2539个来源路径及原页字段；页面脚本错误0。实际Office/PDF窗口启动被模拟，未逐个打开2539页。',
        '- 数据保护：冻结Gold/设置/父块DB/窗口DB及关键生产文件的哈希无漂移；网络请求尝试0；API调用0；没有发布新索引。',
        '- 本轮仅新增实验/报告脚本和文档。仓库其他未提交业务改动来自既有工作，没有清理、回退或混入此次修复。',
        '',
        '| 文件 | 内容 |',
        '|---|---|',
        '| protocol.json | 跑分前冻结输入、对照因素、停止条件、保护哈希 |',
        '| baseline/exact/deeper/no_prefix/top2.jsonl | 各组全部候选、问题、QueryPlan、排名与来源 |',
        '| metrics.json / differences.json | 汇总及每题引文增损、首条排名变化 |',
        '| case_diagnostics.json | 查询token、深度、ANN/精确重合与每条事实精确排名 |',
        '| execution_audit.json / filter_metadata_audit.json | 全库实存比对、抽查、过滤元数据 |',
        '| representation_controls.json | 192个局部控制全文、token、截断标记、分数、原文核对 |',
        '| topk_sensitivity.json | 最终父块TopK敏感性，仅诊断 |',
        '| report_format_audit.json / report_ui_audit.json | 原生格式计分一致、来源及界面核验 |',
        '| final_acceptance.json | 汇总重新核算、测试、保护哈希与未覆盖边界 |',
        '',
        '## 9. 引文原资料及当前索引核验（22条）',
        '',
        '逐条正文规范化匹配均通过；表格数值对象还核对原始表格行列。本轮核对的是选定15题的22条指定引文，不宣称91题全部原文重新人工审核。HTML对应证据和局部控制内有打开原资料页按钮。',
        '',
        '| 题目 | 真实文件与1起始页/张 | 原文核对 | 索引核对 | 引文 |',
        '|---|---|---|---|---|',
    ]
    for q in controls['original_quote_verification']:
        text=q['quote'].replace('|','\\|').replace('\n',' / ')
        lines.append(f'| {q["case"]} | [{Path(q["file"]).name}](<{q["file"].replace(chr(92),"/")}>) 第{q["page"]}页/张 | 通过 | 通过 | {text} |')
    lines += ['', '## 10. 全部91题逐题向量对照', '',
        '名次均为第一条Gold；“—”表示Top50没有任一Gold。完整数要求全部事实。先看完整覆盖，再看名次。无正向Gold题用“不适用”，不会因没召回Gold受到扣分。', '',
        '| ID | 状态 | 问题 | A完整数 | A名次 | B名次 | C名次 | D完整数/名次 | E完整数/名次 |',
        '|---|---|---|---|---:|---:|---:|---|---|']
    for i,r in enumerate(arms['baseline']):
        scores={a:arms[a][i]['scores']['vector'] for a in LABELS}
        rank=lambda a:str(scores[a]['first_gold_rank'] or '—')
        found=lambda a:(f'{scores[a]["gold_evidence_found"]}/{scores[a]["gold_evidence_total_declared"]}' if scores[a]['gold_evidence_total_declared'] else '不适用')
        lines.append(f'| {r["id"]} | {r["review_status"]} | {r["question"].replace("|","/")} | {found("baseline")} | {rank("baseline")} | {rank("exact")} | {rank("deeper")} | {found("no_prefix")} · {rank("no_prefix")} | {found("top2")} · {rank("top2")} |')
    doc='\n'.join(lines)+'\n'
    target=ROOT/'docs/rag_vector_stage_comparison_20261116.md'
    target.write_text(doc,encoding='utf8')
    # Avoid a second drifting narrative; both destinations are generated from one source.
    (OUT/'conclusion.md').write_text(doc.replace('../reports/rag_vector_stage_comparison_20261116/',''),encoding='utf8')
    print('Complete:',target,'; 91 x 5 verified; original/index quotes22; tests36; UI sources2539')


if __name__=='__main__':main()
