"""Write delivery records from frozen observations; never alter Gold or scoring."""
from pathlib import Path
from collections import Counter
import json
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_support_graph_20261113'


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf8'))


def rows(name):
    return [json.loads(l) for l in (OUT/name).read_text(encoding='utf8').splitlines() if l.strip()]


def save(name,data):
    (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')


def main():
    m=read('metrics.json');freeze=read('freeze.json');audit=read('architecture_audit.json')
    runs={a:{r['id']:r for r in rows(name)} for a,name in [('A','A.jsonl'),('B','B_effective.jsonl'),('B_raw','B.jsonl')]}
    fixed={a:{r['id']:r for r in rows(a+'_fixed_prompt.jsonl')} for a in ('A','B')}
    verdict=lambda r:r.get('judgement',{}).get('verdict','error')
    # These diagnoses are offline report annotations, never runtime routing.
    diagnoses={
      'rag-biz-003':('结构化语义审核 → 字段级回退合并','原计划包含旧扣板例外，但语义审核因遗漏“主控板故障”前提将其删除。回退答案补全了前提并通过审核，合并器却只允许补入 missing_fields 的类型（此题为 value），把 condition 例外再次丢弃。随后审核只检查已留下的三条主张，没有发现例外丢失。','generation/structured.py:187'),
      'rag-biz-014':('语义澄清校验 → AnswerService 降级','clarification_assessed 记录 decision_available=false、结构或引用未通过核验。随后给出两种盘型的条件值，但没有追问当前盘型。未把一个值错误套用所有盘型；仍未满足冻结 Gold 要求的追问。','answer_service.py:558'),
      'rag-biz-015':('答案计划/审核的缺失字段表达','两个机型的入口和一个机型的数值已答，但缺少另一机型数值的明确边界，仅输出“部分问项尚未核实”。修改前已 incomplete；原始 B 判 consistent，去掉笼统资料缺失声明后再判 incomplete。属于判分敏感性，不是重新生成带来的内容损失。','answer_review.py:95'),
      'rag-biz-016':('答案计划适用范围 → 审核未拦截','把清洗液耗尽报警条目的原因用于泛指持续气泡报警；最终裁判指出 S5 适用条件不同。原始 B 判 consistent、重放后判 incorrect，而实质性原因句未变，说明原判漏检。保留两次判分，不择优。','generation/structured.py:155; generation/openai_compatible.py'),
      'rag-biz-017':('评测返回结构校验','业务回答为“资料中没有足够依据”。已有 Context 时裁判返回 not_applicable 被冻结判分器拒绝，导致 judge_error。不是证明业务回答错误，也不能计为通过。未更改判分器或重试择优。','evaluation/answer_judge.py（未修改）'),
      'rag-biz-018':('评测返回结构校验','与017同类：实际拒答，但裁判来源绑定字段未满足既有结构要求。单列评测错误，不归因于检索或答案质量。','evaluation/answer_judge.py（未修改）'),
      'rag-biz-019':('重排保留上限','正确两段在 BM25、向量、融合及去重候选均存在；rerank 由2段降到1段，原因/处理页没有进入后续证据集合。软约束使场景页恢复到 Context，但不能挽回重排前已丢的处理页。缺少四原因、灌注与次数，不能用生成补写。','retrieval/hybrid.py / 重排候选截断；scores.rerank'),
      'rag-biz-020':('软条件排序 → EvidencePolicy 文档配额','重排有2段正确证据。条件排序将场景页从第1位移至第9位；随后同文档配额及字段覆盖选择只保留1段。场景细节没有被粗字段分类识别为独有信息。固定候选限幅试验能恢复该题但伤害006/019，已拒绝采用。','constraint_matcher.py; evidence_policy.py:75'),
      'rag-biz-034':('来源绑定判分争议','回答拒绝凭单次结果确诊/治疗，符合冻结预期行为；裁判认为所引复测资料不能支持这个临床限制句。保持 incorrect 原分，列为“知识断言还是证据能力边界”的评测语义争议，不擅自修 Gold 或判分。','evaluation/answer_judge.py（未修改）'),
      'rag-biz-035':('最终答案审核改写','结构化计划已经保留原文三类可能原因。最终审核重新推理“题设在控，因此不宜考虑质控失控”，并把该排除句标 supported，导致与同场景原文相反。属于审核器越权推断，而非召回不足。','answer_service.py:656; generation/openai_compatible.py'),
      'rag-biz-036':('字段级回退合并','回退草稿中的两点和3–6点均通过语义校验，合并器先加入一条 condition 后，仅继续接收 comparison 类型，因此丢掉已验证的点数事实。最终审核看见 missing_field，却只标缺失而未补入。','generation/structured.py:187; answer_review.py:95'),
      'rag-biz-050':('答案计划完整性','Context 中四类原因齐全，计划只生成三类；“cause 已有”被误当原因列表完整，未触发补全。原始渲染还错误声称其余资料未明确，已用全91条确定性重放修正；UPS/地线遗漏依然存在，判 incomplete。','generation/structured.py:160; support_graph.py; answer_review.py:92'),
      'rag-acc-002':('结构化校验与字段级回退','初始计划把通信分类标 confirmed，被确定性门禁删除；回退中操作医生指定项目、按LIS信息测试等细节被粗 comparison 类型合并/短引文审核遗漏。最终审核补了定义但没有补完整流程。待审题只诊断。','generation/structured.py:95; generation/structured.py:187'),
      'rag-row-holdout-001':('证据型号绑定 → 计划/审核','正确范围页和另一个型号的额定值同时入选。结构校验拒绝无法在所引正文绑定的 BS-2800M；最终审核却用 BS-2800 来源支持 BS-2800M。保留供电范围但未输出，存在型号来源错绑。A本来也 incorrect，不能称为新增一致性退化。','generation/structured.py:85; generation/openai_compatible.py'),
      'rag-row-holdout-008':('召回/索引编号证据不足 → 答案对象边界','目标 IMFH1 的完整 Gold 引文未进入 Context，出现 MFH1 相似编号。回答明确写“相似编号”，原始裁判判 incomplete，重放裁判判对象错配；两次判分相冲突。需要核对资料编号/OCR，但本轮不改题。','retrieval/; evaluation/answer_judge.py（未修改）'),
      'rag-row-holdout2-007':('来源型号范围不明确','实际来源给出1115Ω和1140Ω，缺少可确认的BS-800绑定。系统保留两种来源及范围不明，而未给唯一值。原始 uncertain、重放 behavior_mismatch；需要确认版本和型号适用，不能从文件名猜定。','context_builder.py / 来源范围元数据'),
      'evidence-flow-independent-06':('审核渲染未表达具体缺失信息','网口/串口事实正确，未明确“现场是否连通仍需现场状态或测试”。原始B consistent，确定性重放 incomplete；事实句未变，只替换笼统缺失提示。属于边界表述及判分波动。待审题不纳正式分母。','answer_review.py:95'),
      'evidence-flow-independent-11':('澄清结构/引用校验 → 最终输出','澄清评估未通过结构或引用核验。答案计划记录 clarification_needed=true 和缺少序列号/标签，但最终只保留型号及阈值事实，没有将这个计划中的追问落实。A同样未通过，不是新增退化。','answer_service.py:558; generation/structured.py:194'),
    }
    detailed=[]
    for r in m['per_case']:
        if r['after_verdict']=='consistent' and r['id'] not in m['evidence_losses']:continue
        cid=r['id'];module,reason,location=diagnoses.get(cid,('待核实','尚无足够事件定位因果。',''))
        b=runs['B'][cid]
        detailed.append({**r,'first_observed_module':module,'causal_diagnosis':reason,'code_location':location,
           'before_answer':runs['A'][cid].get('result',{}).get('text'), 'after_answer':b.get('result',{}).get('text'),
           'raw_B_verdict':verdict(runs['B_raw'][cid]),
           'fixed_prompt_A_verdict':verdict(fixed['A'][cid]),'fixed_prompt_B_verdict':verdict(fixed['B'][cid]),
           'diagnostic_events':[e for e in b['events'] if e.get('stage') in ('clarification_assessed','structured_answer_plan','review_completed')],
           'source_entries':b['entries']})
    save('failure_analysis.json',detailed)
    instability=[]
    for mode,cc in m['stability_gold_checks'].items():
        for cid,r in cc.items():
            if r['verdicts'].get('consistent',0)!=3:instability.append({'mode':mode,'id':cid,'verdicts':r['verdicts']})
    gates=[
      ('题库与判分文件不变','passed',f"{audit['protected_count']}个保护文件无漂移；活动91题不变"),
      ('索引与来源冻结','passed','bundle/metadata哈希不变；390个原文件复核无变化。原文件初始哈希在验证期间补采，不冒充任务开始时基线'),
      ('无题号/固定答案/Gold运行路径','passed','本轮静态审计未发现已知业务常量或运行时evaluation导入；静态扫描不是穷尽性语义证明'),
      ('显式关闭兼容','passed','91题关闭增强的计划与固定候选后续结果一致，旧配置端到端16项通过；未声称另跑91题新向量推理完全相同'),
      ('完整证据不明显下降','passed','正式正证据46题，全引文完整保留37→39；仍有020单题损失，不能用总数掩盖'),
      ('无新增范围越界/来源错绑','failed','003例外丢失形成无条件要求；035审核产生原文不支持的排除主张。电压题的型号错绑也未解决'),
      ('无新增确定故障语气','pending','原因语气结构约束与机制测试通过；没有覆盖全量人工标注的certainty真值，不宣称100%安全'),
      ('避免全题拒答','passed','部分支持机制测试通过，完整B未因单个缺字段整题拒答；这不等于部分答案已足够'),
      ('多证据/部分答案机制','passed','77项相关机制测试通过；真实样本仍有合并遗漏'),
      ('冲突拆分业务准确率','pending','显式同范围冲突合成测试通过；真实图中未形成已独立核验的冲突分母，不能报准确率'),
      ('澄清可靠性','failed','014及independent-11未落实必须询问的缺失条件；021可先条件回答但重复不稳定'),
      ('核心事实重复稳定','failed','固定Context每题3次，多个题核心项缺失/错绑仍变化；不按字面措辞变化判失败'),
      ('退化定位','passed','所有10条原一致→非一致记录均在failure_analysis.json定位；017/018为裁判错误，034为判分争议'),
      ('完整旧测试全绿','failed','RAG 446通过16失败1跳过；10项原有失败，6项与新软约束/默认开关契约不同。未修改旧断言'),
      ('严格所有Prompt不变的全链路单因素对照','pending','主生成Prompt及模型一致；新增结构化/审核Prompt改变完整调用链。另做91+91固定主Prompt隔离；不冒充全链路所有Prompt相同'),
    ]
    acceptance={'decision':'暂不采用为稳定生产默认版本','implementation_and_verification':'completed','business_acceptance':'failed',
       'configuration':'按本轮明确授权，四项能力已默认和本机启用，strict=false；这是已启用的验证状态，不代表通过验收。未擅自回滚用户要求的设置。',
       'rollback':'双击 rollback_flags.cmd 仅恢复5项旧配置；应用需重新载入配置。',
       'gates':[{'criterion':a,'status':b,'evidence':c} for a,b,c in gates],
       'remaining_failures':[r['id'] for r in detailed], 'answer_losses':m['answer_losses'],
       'stability_not_three_of_three':instability,
       'raw_B_verdicts':dict(Counter(verdict(r) for r in runs['B_raw'].values())),
       'effective_B_verdicts':dict(Counter(verdict(r) for r in runs['B'].values())),
       'effective_answer_protocol':read('review_render_replay.json'),
       'unmeasured_metrics':m['unmeasured_metrics']}
    save('final_acceptance.json',acceptance)
    reviewed=m['cohorts']['reviewed']['arms']
    save('delivery_summary.json',{'title':'通用证据支持重构 · 已完成验证，整体验收未通过',
       'summary':f"冻结91题：50题已审核、41题仅诊断。正式集合一致判分 {reviewed['A']['native_verdicts'].get('consistent',0)}/50 → {reviewed['B']['native_verdicts'].get('consistent',0)}/50；原始B为39/50。正证据46题完整保留37→39。右侧B使用保存的审核结果作确定性渲染修正，不重新生成模型答案；原始B完整保留。",
       'validation':'RAG测试446通过、16失败、1跳过；相关机制77通过；设置8通过；旧配置端到端16通过。固定Context 8题×3次×2路径已全部保存。四项能力按授权启用，strict=false，但不建议保持为稳定生产默认。题库、Gold和判分代码未修改。'})
    lines=['# 通用 RAG 重构：完整开发与验证结论',
      '\n本轮完成代码审计、13个运行文件修改、机制测试、冻结91题A/B、固定主Prompt隔离、重点8题各3次重复、配置启用和原生HTML报告。**业务验收未通过，不建议将组合方案作为稳定生产默认。**',
      '\n## 1. 最重要的结论',
      '证据选择整体有所改善，但结构化生成与审核又丢掉了已经有证据的例外、数值和步骤。当前最需要修的是“已验证事实如何完整保留”，而不是继续靠术语或单题关键词提升召回。',
      '- 正式50题原一致38题，原始完整B一致39题；通用渲染修复后对全部91条冻结审核结果重放，正式一致37题。这个变化包含裁判波动，不能把39当作最终结果或择优报告。',
      '- 正式正证据46题：完整引文保留37→39，平均Evidence Recall 85.51%→92.03%；020依然发生单题证据损失。',
      '- 固定主Prompt隔离：正式一致39→42，但A有2次裁判结构错误、B没有；不能把全部+3归因于检索。',
      '- Q023本次完整回答通过，重复结构化3/3通过；Q021本次完整回答通过，但固定Context结构化重复0/3通过，不能称为稳定修复。',
      '- 新能力按本轮要求已启用。启用事实与采用决定分开：验收失败，建议恢复兼容配置；已提供只恢复五个开关的rollback_flags.cmd，未擅自覆盖其他设置。',
      '\n## 2. 冻结输入与对照含义',
      f"AI索引：`{m['bundle_id']}`。50题reviewed、40题pending、1题disputed；主集49题，其中47题正式，回归42题，其中3题正式。所有91题都运行，只有50题计正式答案指标；正向证据分母46题，4道正式拒答题无正证据。",
      '基线为任务开始时工作区快照baseline_src，不能拿Git HEAD代替，因为已有历史未提交修改。70个保护文件（题库/清单/判分）哈希未变。390个原文件复核无变化；原文件哈希在验证期间补采，任务开始时仅有索引metadata哈希，不能声称原文件开始哈希已记录。',
      'A/B原生链路使用相同问题、索引、模型和主生成SYSTEM_PROMPT；B按要求新增计划/审核提示与开关，所以原生对照是组合方案比较，不是“所有Prompt和配置完全相同”的算法单因素实验。A_fixed_prompt/B_fixed_prompt另以各自真实Context、同一主Prompt和同一模型运行全部91题。',
      '所有运行产物保留。中间试验在intermediate_v2、pilot_B中，不纳入最终分数。最终B固定快照为validated_final_src；当前代码仅answer_review.py的确定性尾句渲染与之不同。B_effective对全91条保存的审核记录应用该差异，仅变化答案再次调用原判分器，没有重抽生成或审核模型。',
      '\n## 3. 实现和硬约束变化',
      '详见 docs/rag_support_graph_audit_20261113.md 与 docs/rag_support_graph_implementation_20261113.md。',
      '- 明确型号/标识符继续受约束；阶段、数量范围、方向、主题及泛设备类型变成排序/解释用软条件。推断别名不覆盖原问题。',
      '- Support Graph 保存事实、原文来源、范围、条件、显式冲突、明确父子部件和同段编号步骤；不凭相邻页或chunk顺序推断业务顺序。',
      '- 已通过基本质量门槛的候选允许最多两段新增字段补充，保留原顺序锚点、精确代表、多来源代表和总字符预算。不是取消全部配额。',
      '- 结构化主张支持多来源共同验证，绑定原文完整行；缺字段调用旧链路，再验证后按字段合并。',
      '- 最终审核保留supported主张，分别记录数字、编号、范围与引用；语义澄清有已知事实时先条件回答再补充问题。',
      '- 删除“某补充句未过审核 ⇒ 其余资料均未明确”的错误渲染推断；不能由审核拒绝倒推语料没有答案。',
      '\n### 运行文件清单',
      *['- `'+r['path']+'`' for r in read('changed_files.json')],
      '\n## 4. 全量分层结果',
      '|集合|题数|A完整链路一致|B最终一致|A固定主Prompt一致|B固定主Prompt一致|',
      '|---|---:|---:|---:|---:|---:|']
    for name,cc in m['cohorts'].items():
        aa,bb=cc['arms']['A'],cc['arms']['B']
        lines.append(f"|{name}|{cc['count']}|{aa['native_verdicts'].get('consistent',0)}|{bb['native_verdicts'].get('consistent',0)}|{aa['fixed_prompt_verdicts'].get('consistent',0)}|{bb['fixed_prompt_verdicts'].get('consistent',0)}|")
    lines+=['\n这些是既有LLM裁判的一致判分，不冒充人工业务准确率。diagnostic不进入正式分母。各组完整判分分布、group_macro_consistency及对象/来源绑定代理指标见metrics.json。',
      '\n### 正式46题正证据：各阶段',
      '|阶段|Evidence Recall A→B|Recall@1 A→B|Recall@3 A→B|Recall@5 A→B|MRR A→B|全引文题数 A→B|',
      '|---|---:|---:|---:|---:|---:|---:|']
    for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context'):
        a,b=reviewed['A']['stages'][stage],reviewed['B']['stages'][stage]
        cells=[stage]+[f"{a[k]:.4f} → {b[k]:.4f}" for k in ('evidence_recall','recall@1','recall@3','recall@5','MRR')]+[f"{a['all_gold_retained']} → {b['all_gold_retained']}"]
        lines.append('|'+ '|'.join(cells)+'|')
    lines+=['\nRecall@k为每题完整Gold引文在前k条的覆盖比例再平均，MRR为第一条完整Gold引文倒数排名；使用既有source-quote-v3，不把同页其他行算命中。',
      '\n## 5. 收益、退化与首个失败模块',
      '证据收益：'+', '.join(m['evidence_gains'])+'。证据损失：'+', '.join(m['evidence_losses'])+'。',
      '原不一致→最终一致：'+', '.join(m['answer_gains'])+'。其中待审题仅作诊断。',
      '原一致→最终非一致：'+', '.join(m['answer_losses'])+'。其中017/018是裁判错误；034含判分争议；不能统称算法退化。',
      '|题目|状态|A→B|首次定位|具体原因|', '|---|---|---|---|---|']
    for r in detailed:
        lines.append('|'+ '|'.join([r['id'],r['review_status'],r['before_verdict']+'→'+r['after_verdict'],r['first_observed_module'],r['causal_diagnosis']])+'|')
    lines+=['\n每题完整证据、计划、原始审核、actual Context、前后答案与代码位置见failure_analysis.json；重点题可读版见focus_answers.md；HTML可逐题查看原文位置与阶段候选。',
      '\n## 6. 重放造成的判分变化',
      'B.jsonl是原始完整入口输出；B_effective.jsonl是最终确定性渲染重放。所有变化一视同仁重新判分，没有依据分数选择重试。除050去除错误资料缺失声明外，以下变化多反映原判漏检/边界文字敏感性，应保留不确定性：',
      '|题目|原始B|最终B|', '|---|---|---|']
    for cid,r in runs['B'].items():
        if verdict(r)!=verdict(runs['B_raw'][cid]):lines.append(f"|{cid}|{verdict(runs['B_raw'][cid])}|{verdict(r)}|")
    lines+=['\n## 7. 固定Context稳定性（8题×3次×2路径）',
      '结构化重复只隔离内部计划生成/校验，不包含完整AnswerService外层审核；因此不能当成第二批原生端到端运行。每题每路径Context SHA256一致。数字、范围、引用、fact_ids和Gold核心项均留档；fact_id变化或措辞不同本身不自动判失败。',
      '|题目|普通生成一致/3|结构化生成一致/3|结构化判分分布|', '|---|---:|---:|---|']
    for cid in freeze['focus_ids']:
        a=m['stability_gold_checks']['fixed_prompt'][cid];b=m['stability_gold_checks']['structured'][cid]
        lines.append(f"|{cid}|{a['verdicts'].get('consistent',0)}|{b['verdicts'].get('consistent',0)}|{json.dumps(b['verdicts'],ensure_ascii=False)}|")
    lines+=['\nQ023已出现可重复收益；Q021、Q050、条码与电压的结构化路径仍有遗漏或绑定问题。单次完整入口通过不能抵消重复失败。',
      '\n## 8. 回退、澄清、审核与无法测量的指标',
      f"字段级回退触发{m['fallback']['cases']}/91题：{json.dumps(m['fallback']['reasons'],ensure_ascii=False)}；经校验补入{m['fallback']['verified_claims_added']}条事实。",
      f"事实图覆盖{m['graph']['cases']}/91题。字段状态{json.dumps(m['graph']['field_statuses'],ensure_ascii=False)}；partial_supported只是候选线索，不是完整语义证明。",
      '审核主张状态：'+json.dumps(m['review_statuses'],ensure_ascii=False)+'。这些是审核器自报，不是独立标注的Unsupported Claim Rate。',
      '实际没有以clarify状态提前结束的题目；部分追问通过partial_clarification_deferred追加。澄清模型“无须追问”、核验失败和真正已知条件不同，详见diagnostic_counts.json。',
      *['- '+k+'：'+v for k,v in m['unmeasured_metrics'].items()],
      '\n## 9. 测试与复现命令',
      '```powershell',
      '.venv/Scripts/python.exe -m pytest tests/rag -q --junitxml=reports/rag_support_graph_20261113/unit_results.xml',
      '.venv/Scripts/python.exe -m pytest tests/rag/test_support_graph_runtime.py tests/rag/test_general_architecture.py tests/rag/test_generation_gateway.py -q',
      '.venv/Scripts/python.exe -m pytest tests/test_settings_service.py -q',
      '.venv/Scripts/python.exe reports/rag_support_graph_20261113/run_legacy_config_tests.py',
      '.venv/Scripts/python.exe tools/validate_support_graph.py --runtime-root reports/rag_support_graph_20261113/baseline_src --output-dir reports/rag_support_graph_20261113 --suite-root . --arm A --workers 6',
      '.venv/Scripts/python.exe tools/validate_support_graph.py --runtime-root reports/rag_support_graph_20261113/validated_final_src --output-dir reports/rag_support_graph_20261113 --suite-root . --arm B --workers 6',
      '.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm A --output-dir reports/rag_support_graph_20261113 --workers 4',
      '.venv/Scripts/python.exe -m tools.evaluate_general_refactor --arm B --output-dir reports/rag_support_graph_20261113 --workers 4',
      '# 固定Context重复：B路径分别加 --stability 与 --structured --stability',
      '.venv/Scripts/python.exe -m tools.judge_support_stability',
      '.venv/Scripts/python.exe -m tools.replay_support_review_render',
      '.venv/Scripts/python.exe -m tools.report_support_graph',
      '.venv/Scripts/python.exe -m tools.finalize_support_graph_report',
      '.venv/Scripts/python.exe -m tools.render_general_native_report --native --output-dir reports/rag_support_graph_20261113',
      '```',
      '原始生成产物已存在时不覆盖；部分脚本拒绝覆盖以避免混合运行。新的独立重跑应另设输出目录并重新冻结，不能删掉失败行再跑。',
      '结果：RAG 446通过、16失败、1跳过；机制77通过；设置8通过；显式旧配置16通过。机制77是RAG集合子集，不与446相加。10项失败原已存在，6项是本轮软条件和默认开关与旧断言冲突；完整列表见unit_failure_audit.json。保留失败，没有修改测试断言。',
      '\n## 10. 配置与采用决定',
      'QueryPlan=true、StructuredAnswer=true、AnswerReview=true、strict=false、SemanticClarification=true。模型、API、持久化其他设置不变；引用校验仍执行。应用已启动时需重新载入设置。',
      '**不建议保持组合方案为稳定生产默认。** 本轮按明确要求完成启用，但验收不能宣称成功。回滚只需运行本目录rollback_flags.cmd，恢复五个旧开关；代码、Gold和其他用户设置不会被删除。',
      '|能力|决定|', '|---|---|',
      '|审计日志、原文绑定记录、可回滚开关、支持图诊断|可保留；机制与兼容性已验证|',
      '|有界字段覆盖、软条件排序|仅实验；020退化说明不能通用采用|',
      '|结构化计划＋字段回退＋最终审核的组合默认|暂不采用为稳定生产默认；事实完整性及审核越权尚未解决|',
      '|澄清自动判断|仅实验；字段丢失和无效评估的降级仍需修复|',
      '|型号别名、IMFH1/MFH1、BS-800温度版本、拒答边界判分|需要原文或业务裁决；本轮不改Gold|',
      '\n## 11. 下一轮最小改进与停止条件',
      '1. 先只修“已验证事实合并”，冻结当前检索和Context，不再调召回权重。以对象＋字段＋条件分支＋数值角色标识原子事实，不以cause/solution/value类型出现一次即判完整；回退已核验的条件例外、比较两侧数值应进入未覆盖事实集合。不得以题号或答案常量指定必答。',
      '2. 将“审核支持性”和“审核完整性”分开：审核器可以指出遗漏或建议修正，但新增排除/范围推断必须重新接受来源验证；不允许把已有正确计划自由改写成相反结论。',
      '3. 先用此次失败的冻结Context定位修复，再用未参与调试的资料问题验证。重点观察条件例外、比较两侧数值、多个原因、部分答案边界、来源别名。不能只看总一致数。',
      '4. 独立处理019重排截断与020选段丢失，不把它们混入生成修复归因。已反证的软条件加分限幅试验不再继续扫权重。',
      '停止条件：出现新来源错绑/新增范围扩展立即判该轮不采用；完整事实集合不再遗漏且固定Context3次核心事实稳定、旧正确题无新增实质错误，才进入组合全量验收。',
      '\n## 12. 文件导航',
      '- report.html：原生格式，A/B答案、阶段候选、热力图与原资料页按钮。',
      '- open_report.cmd：启动本地只读报告与受控原资料打开服务；可直接双击HTML离线查看，页定位按钮需该服务。',
      '- metrics.json / final_acceptance.json：完整分母、指标、门槛和失败状态。',
      '- A.jsonl / B.jsonl / B_effective.jsonl：原始真实轨迹及最终确定性重放，均保留。',
      '- failure_analysis.json / focus_answers.md / architecture_examples.json：逐题原因及计划、证据图、Context、答案计划实际样例。',
      '- source_file_integrity.json / source_hash_recheck.json / architecture_audit.json：语料与测试输入完整性。',
      '- runtime_changes.diff / changed_files.json：相对于本轮开始快照的运行代码差异。']
    (OUT/'conclusion.md').write_text('\n'.join(lines),encoding='utf8')
    examples={}
    for cid in freeze['focus_ids']:
        a,b=runs['A'][cid],runs['B'][cid]
        examples[cid]={'question':b['question'],'A_query':next((e['query_plan'] for e in a['events'] if 'query_plan' in e),None), 'B_query':next((e['query_plan'] for e in b['events'] if 'query_plan' in e),None),
          'A_context':a['context'],'B_context':b['context'], 'B_entries':b['entries'],
          'B_graphs':[e.get('fact_support_graph') for e in b['events'] if 'fact_support_graph' in e],
          'B_answer_plans':[e.get('answer_plan') for e in b['events'] if e.get('stage')=='structured_answer_plan'],
          'A_answer':a.get('result'), 'B_answer':b.get('result')}
    save('architecture_examples.json',examples)
    events=[(r['id'],e) for r in runs['B'].values() for e in r['events']]
    save('diagnostic_counts.json',{'structured_plan_events':sum(e.get('stage')=='structured_answer_plan' for _,e in events),
      'review_completed':sum(e.get('stage')=='review_completed' for _,e in events),
      'clarification_assessed':sum(e.get('stage')=='clarification_assessed' for _,e in events),
      'clarification_invalid_ids':[cid for cid,e in events if e.get('stage')=='clarification_assessed' and not e.get('decision_available')],
      'clarification_positive_ids':[cid for cid,e in events if e.get('stage')=='clarification_assessed' and e.get('needs_clarification')],
      'partial_clarification_deferred_ids':[cid for cid,e in events if e.get('stage')=='partial_clarification_deferred'],
      'review_failures':[{'id':cid,'event':e} for cid,e in events if e.get('stage')=='review_failed']})
    steps=read('validation_steps.json')
    for s in steps:
        if s['status']=='running':s['status']='passed';s['limit']='运行全部完成，不代表业务验收通过'
        if s['step']=='原生报告、原文定位和最终验收':s['status']='pending';s['limit']='报告及UI核验完成后更新；业务验收failed'
    save('validation_steps.json',steps)
    print('Wrote conclusion, acceptance, failure analysis, examples, diagnostic counts')


if __name__=='__main__':main()
