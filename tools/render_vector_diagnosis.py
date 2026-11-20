"""Publish vector attribution alongside the unchanged native 91-case viewer."""
import hashlib
import html
import json
import shutil
from pathlib import Path
from pptx import Presentation
from tools.diagnose_vector_recall import OUT
from tools.validate_selection_factorial import ROOT, write, sha
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.content_identity import canonical_content

BASE=ROOT/'reports/rag_bge_base_adoption_20261116'
DOC=ROOT/'docs/rag_vector_recall_diagnosis_20261116.md'

# Audit annotations only. These IDs, facts and controls never enter runtime RAG.
FINDINGS={
 'rag-biz-006':('来源竞争、短窗口与问法共同影响；不是漏收录', '指定PDF排43；向量前两名是两份CL-8000i装机指南，正文直接说明浮子及进液泵。目标PDF把开始/停止、空瓶切换分散在不同窗口。整父块替换反而43→816，去包装43→80，拆问也未改善，不能把原因单独归为切片太小。当前关键词补回，最终Context完整。'),
 'rag-biz-007':('有争议的参数题，不作为正式失败', '向量排26，关键词2，当前Context完整。140±5与另一来源145±5的争议仍保留；不以此题支持更换模型或改变参数答案。'),
 'rag-biz-019':('不是向量漏召回；重排入口之前的候选截断', '场景页和处理页向量分别1、36，均已召回；处理页融合31、去重30，未进入前20个重排输入。因此重排结果缺处理页，不能解释为BGE把它打低或向量没找到。'),
 'rag-biz-025':('参数表的语义排名偏后，已被混合检索补回', '指定保存时间表向量13，关键词1，当前Context1且完整。窗口包含多项目、多温度、反复冻融等内容；低排名是模型相似度结果，不是索引缺字。尚未对该题单独隔离包装、窗口和模型贡献。'),
 'rag-biz-028':('主答案已第1；缺的是另一份概括性原则', '临床系统偏高/偏低的直接原因与措施排1；检测系统综合考虑的概括页排56。概括页文字偏定标与系统故障，离“临床普遍偏低”更间接，答案文字在原文/索引/完整小窗口都存在。删板卡追问56→60，整父块56→71，不能归为表格切断。严格Gold覆盖不全不等于没有找到临床原因。'),
 'rag-biz-050':('指定PDF排名偏后；同正文PPT已被召回', 'PDF17排67，但高速PPT15排26、全线PPT17排31，均含四类可能原因且在当前Context。PDF中诊断标题和完整原因分成两窗口；整父块67→105，不能声称合并即修复。去掉型号查询67→7，仅说明机型词影响模型匹配，不能据此在生产删除型号约束。'),
 'rag-row-holdout-003':('表头/型号与数值行的表示问题，证据最明确', '实际答案行所在窗口的全库小窗口排名9699；父块依靠“测试速度”等窗口排149。用原PPT表头+原行+型号列生成105token表示，固定同一模型及竞争向量，局部替换149→1。整父块404且544token会截断；盲目加大不成立。当前关键词1已补回。'),
 'rag-row-holdout-004':('指定来源漏计；另一个同型号对照表已第4', 'Gold指定PPT55排70，但服务资质培训PPT7向量4，表头明确BS-2000与BS-2800M，两列样本类型逐项相同，且已进入Context首段。指定页面的表头+行局部替换70→28，拆两个型号11/38。原最佳窗口已含完整行和型号，不是答案被切没。'),
 'rag-row-holdout-005':('指定来源漏计，同时存在窗口表示问题', 'Gold指定PPT27排53；电解质教材PPT35向量1、实际Context首段已含同样凯特斜率。两页行序号8/7不同，不能用整段逐字匹配当等价事实缺失。指定页标题和斜率行分开，表头+行53→1。两页测试速度不同，替代审核仅限本题斜率，不扩展到速度参数。'),
 'rag-row-holdout-006':('对比参数行排名11，当前已补回', '同一参数页混有生化和ISE速度、多个模块配置；小窗口及多条件语义存在混杂，但本轮未做该题单因素干预，不能确认为模型或切片单一原因。最终Context完整。'),
 'rag-row-holdout-008':('不是向量问题；后续排序/选段丢失', 'IMFH1目标表向量1、融合1；BGE重排14，条件后10，最终未入Context。向量已完成任务。'),
 'rag-row-holdout-009':('比较问题排名靠边，混合检索已补回', '样本针清洗比较表向量50，关键词24，融合4，重排1，Context完整。窗口含清洗方式和相邻功能；尚未隔离查询、包装与表头贡献，不把表格题统一归因为切片损坏。'),
 'rag-row-holdout2-001':('不是向量漏召回；条件检查降序', '高速与中低速码制表向量21、18，BGE2、3；条件检查后15、16，未进最终8段。当前风险在后续适用性/条件排序，不在向量查不到条码资料。'),
 'rag-row-holdout2-003':('参数比较表向量19，当前Context完整', '范围数值包含在真实窗口，关键词6、融合3、重排1。未做单因素归因；不能凭排名19就认定向量失败。'),
 'rag-row-holdout2-005':('混合型号参数表向量16，当前Context完整', '空气浴/固体直热与多个型号和多个加热部件同表。词与参数均在索引，关键词3、融合5、重排2，未出现最终证据缺失。'),
 'rag-row-holdout2-007':('跨页范围页靠后；丢在重排输入之前', '参数页向量6，明确BS-800的前页44；两页都召回，型号页未进重排前20，最终只剩参数页。跨页不能凭相邻就认定适用，但当前Gold已有页间绑定；此为候选组合/入口问题。'),
 'rag-row-holdout2-009':('波长参数表向量14，当前Context完整', '混合参数页含反应体积、速度、型号等；参数本身未丢。关键词1、重排1。无证据支持“模型不行/需要换模型”的单一判断。'),
 'evidence-flow-independent-05':('查询多目标、来源包装影响；替代原理表亦未被计入', '通用原理页5412，指定模块对照页684；另两份A&T/Medica原理表实际向量37、41，其中一份在当前Context。去掉现场未知追问，对照页684→48（通用原理页仍1271）；仅去掉目标窗口来源/位置包装，对照页684→1。原表完整行重写反而770/1949，说明不是所有表都适合相同行级变换。静态资料不能判断现场安装模块。'),
 'evidence-flow-independent-06':('多型号通信表表示及语义偏好；单纯拆问无收益', '指定串口/网口表329；前排是确有BS-2800M网口信息的维修手册和现场连线资料，不能都算无关。完整表保留但多个型号/IP/COM共用一向量。原表行级局部替换BS-2800M行40、BS-430行54；整父块370，拆型号571/401，删现场追问630，不能宣称通用拆问可修复。关键词补回后当前Context完整。'),
 'evidence-flow-independent-11':('范围表长尾，关键词已补回', '指定模块与序列号表向量原回放36、精确37；近似/浮点排序有小差别，但均在50以内。问题是需型号/序列号才能判定具体模块；当前Context完整，不应把需澄清误算向量缺资料。'),
}


def main():
    rows=[json.loads(l) for l in (OUT/'diagnosis.jsonl').read_text(encoding='utf8').splitlines()]
    frozen={r['id']:r for r in map(json.loads,(ROOT/'reports/rag_support_graph_20261113/A.jsonl').read_text(encoding='utf8').splitlines())}
    production={r['id']:r for r in map(json.loads,(BASE/'B.jsonl').read_text(encoding='utf8').splitlines())}
    controls=json.loads((OUT/'representation_controls.json').read_text(encoding='utf8'))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    audit=json.loads((OUT/'execution_audit.json').read_text(encoding='utf8'))
    visibility=json.loads((OUT/'visibility_audit.json').read_text(encoding='utf8'))
    sources=json.loads((BASE/'report_sources.json').read_text(encoding='utf8'))
    def register(path,page,slide=False):
        loc=dict(location_text=f'第 {page} '+('张幻灯片' if slide else '页'),page_number=None if slide else int(page),slide_number=int(page) if slide else None)
        source=dict(path=str(Path(path)),location=loc)
        key=hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]
        sources[key]=source
        assert Path(path).is_file()
        return key
    def button(key):return f'<button type="button" data-original-source="{key}">打开原资料对应页</button>'
    alternatives=[]
    selected_alts={'rag-biz-050':[26,31],'rag-row-holdout-004':[4],'rag-row-holdout-005':[1],'evidence-flow-independent-05':[37,41]}
    scope_notes={
        'rag-biz-050':'同暗计数诊断、53杯位、四原因；属于用户此前认可的免疫场景资料范围。未扩展到中低速专属结构或更换件。未自行登记正式Gold替代。',
        'rag-row-holdout-004':'原PPT表格第0行明确两型号，第1行两列对应样本类型一致。只核对样本类型，不将整页全部参数视为等价。',
        'rag-row-holdout-005':'同为凯特模块；Na/K 45–70、Cl 40–70 mv/dec一致，序号不同。两页速度不同，禁止把速度也并为共识。',
        'evidence-flow-independent-05':'原PPT模块类型列、检测原理列明确对应A&T间接法、Medica直接法；不支持现场当前安装状态，也不能替代通用原理解释页。',
    }
    for cid,ranks in selected_alts.items():
        for rank in ranks:
            c=frozen[cid]['candidate_stages']['vector'][rank-1]
            path=Path(c['file_path']);page=c['slide_number'] or c['page_number']
            original=original_text(path,page)
            slide=Presentation(path).slides[int(page)-1]
            tables=[[[cell.text for cell in row.cells] for row in shape.table.rows] for shape in slide.shapes if shape.has_table]
            if cid=='rag-biz-050':quote=next(r for r in rows if r['id']==cid)['targets'][0]['quote']
            elif cid=='rag-row-holdout-004':quote=next(r for r in rows if r['id']==cid)['targets'][0]['quote']
            elif cid=='rag-row-holdout-005':quote='电极斜率范围\nNa+/K+: 45-70mv/dec, Cl-: 40-70mv/dec,'
            else:quote='A&T\n间接离子选择电极法'
            # Independent-05's Medica fact is checked separately, not inferred from A&T.
            required=[quote]+(['Medica\n直接离子选择电极法。'] if cid=='evidence-flow-independent-05' else [])
            assert all(canonical_content(q) in canonical_content(original) for q in required)
            assert all(canonical_content(q) in canonical_content(c['text']) for q in required)
            entries=[e for e in production[cid]['entries'] if e['filename']==c['filename'] and e['location']==c['location']]
            included=any(all(canonical_content(q) in canonical_content(e['included_text']) for q in required) for e in entries)
            alternatives.append(dict(id=cid,vector_rank=rank,file=str(path),page=page,quote=required,
                original_verified=True,index_verified=True,actual_context_verified=included,table_cells=tables,
                original_text=original,scope_review=scope_notes[cid],status='proposed_equivalent_not_added_to_gold',
                source_sha256=sha(path),source_id=register(path,page,True)))
    write(OUT/'alternative_evidence_review.json',alternatives)
    metrics={}
    for cohort in ('formal','diagnostic'):
        cases=[r for r in frozen.values() if r['cohort']==cohort and r['scores']['vector']['gold_evidence_total_declared']]
        ranks=[r['scores']['vector'].get('first_gold_rank') for r in cases]
        metrics[cohort]=dict(n=len(cases),complete=sum(r['scores']['vector']['gold_evidence_recall']==1 for r in cases),
            **{f'hit_at_{k}':sum(x is not None and x<=k for x in ranks) for k in (1,3,5,50)},
            mrr=sum(1/x if x else 0 for x in ranks)/len(ranks))
    assert len(rows)==20 and sum(len(r['targets']) for r in rows)==30
    assert all(t['original_verified'] for r in rows for t in r['targets'])
    assert all(r['ann_top50_overlap_with_frozen']==50 for r in rows)
    assert all(x['quote_present_in_one_parent'] for x in visibility['indexed_quote_checks'])
    assert not audit['network_attempts'] and not audit['protected_drift'] and not controls['protected_drift']
    findings=[dict(id=r['id'],question=r['question'],review_status=r['status'],cause=FINDINGS[r['id']][0],explanation=FINDINGS[r['id']][1]) for r in rows]
    write(OUT/'case_findings.json',findings)
    write(OUT/'vector_metrics.json',metrics)
    md=['# BGE基础排序接入后的向量召回诊断（2026-11-16）','',
        '## 结论','',
        'BGE基础排序已接入。本轮未修改向量检索、切片、模型、Gold或用户设置，API调用0。不能把报告里的低排名全部解释为向量失败；也不能把指定Gold文件漏召回等同于没有任何正确答案证据。','',
        '本轮证据支持的主要问题是：**资料的向量表示（表头、型号、参数行与来源包装）影响语义排序；多目标查询也会干扰匹配；另有替代正确来源未登记导致严格计分低估。近似索引漏搜不是本轮缺失的主要原因。**','',
        '在7道严格Gold证据不全的题中，050、row-004、row-005已找到覆盖本题事实的替代来源；independent-05也找到两种模块原理的替代对照表，但通用原理解释页仍缺。这里只单列人工可复查结果，不调整原始分数、不自行批准Gold或扩大来源适用范围。','',
        '## 冻结输入和检查','',
        f'- 已发布AI bundle：`{protocol["bundle"]}`；向量索引指纹：`{protocol["vector_index"]["index_fingerprint"]}`。',
        f'- Embedding：`{protocol["model"]}`，revision `{protocol["revision"]}`，1024维。它负责向量召回，与本轮已采用的BGE重排器是两个独立环节。',
        '- 当前unit_v2向量库：52282个细窗口、11769父块；目标128 token、最大192、重叠32；包含来源/位置包装；模型上限512。父块按最佳窗口相似度排序，非平均分。',
        '- 生产向量Top50；先查400个小窗口，再合并父块，必要时扩查。0.35属于后续证据阈值，不是本轮向量Top50的相似度过滤线。',
        '- 固定91题：50 reviewed、40 pending、1 disputed；正式有正向证据46题，诊断有正向证据40题。pending/disputed不当正式验收。',
        '- 选取20题：指定Gold召回不全、首个Gold晚于10、或当前Context不全。对每个原查询全库52282向量精确余弦计算，再与实际Chroma近似查询比较。',
        '- 52282缓存向量归一化/有限性检查通过；81个Chroma实存向量抽查与缓存相等。不能把抽查写成全部实存向量逐条核对。',
        '- 20题原查询ANN Top50与冻结运行20/20集合完全一致；ANN与全库精确Top50重合48～50条。所有7道不全题，指定目标在精确检索中仍在50之外。',
        '- 30条目标引文在原资料、索引父块均逐字规范化核验通过。PPT表格通过原始表格单元格核对型号列/参数行归属。查询38～71 token；当前已存窗口24～192 token，未发现本轮原查询或窗口因512上限被截断。',
        '- Gold、已发布父块DB、向量unit DB、持久化设置的冻结哈希无漂移；未发布新索引；Adapter默认状态未改。','',
        '## 严格Gold原口径（不是答案正确率）','',
        '| 集合 | 有正向题 | Gold全部覆盖@50 | 任一Gold Hit@1 | Hit@3 | Hit@5 | Hit@50 | MRR |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,label in [('formal','正式'),('diagnostic','诊断')]:
        m=metrics[key];md.append(f'| {label} | {m["n"]} | {m["complete"]} | {m["hit_at_1"]} | {m["hit_at_3"]} | {m["hit_at_5"]} | {m["hit_at_50"]} | {m["mrr"]:.4f} |')
    md += ['', 'Hit@K是前K条存在至少一条完整的指定Gold引文；MRR为第一条Gold排名倒数的均值，未命中记0；全部覆盖要求每条声明Gold均被找到。二者均保持来源绑定，不自动计算未批准替代证据。','',
        '## 7道严格覆盖不全的逐题原因','',
        '| 题目 | 指定Gold的精确排名 | 关键诊断 |', '|---|---|---|']
    for r in rows:
        if any(t['exact_completion_rank']>50 for t in r['targets']):
            md.append(f'| {r["id"]} | '+ '、'.join(str(t['exact_completion_rank']) for t in r['targets'])+' | '+FINDINGS[r['id']][1]+' |')
    md += ['', '## 固定模型的表示对照','',
        '以下仅替换被审计来源的表示，其余52282竞争向量固定。选定目标来源是事后诊断，运行编码器不输入Gold答案。它能说明局部表示是否影响排名，**不能当作全库重建、独立泛化或生产收益**，因为其他文档重建后也会重新竞争。','',
        '| 对照 | 原表示排名 | 原表头＋完整行 | 完整父块替换 |', '|---|---:|---|---:|',
        '| row-003 待机启动时间 | 149 | 1 | 404（544 token，截断） |',
        '| row-004 样本类型 | 70 | 28 | 715（578 token，截断） |',
        '| row-005 凯特斜率 | 53 | 1 | 25 |',
        '| independent-06 通信方式 | 329 | BS-2800M行40；BS-430行54 | 370 |',
        '| independent-05 模块原理表 | 684 | A&T行1949；Medica行770 | 2334 |',
        '| 050 暗计数PDF | 67 | 不适用（PDF文本页） | 105 |',
        '| 006 清洗液PDF | 43 | 不适用（PDF文本页） | 816 |','',
        '另一个关键干预：independent-05仅去掉原小窗口的来源/位置包装，对照页684→1，该窗口包含两种模块原理；但通用原理页仍很后。去包装会让050从67变为163、006从43变为80，所以不能一刀切去掉所有来源信息。原问题去现场状态追问让independent-05对照页684→48、原理页5412→1271；independent-06反而329→630。','',
        '003最初“首个样本结果”改写改变了“启动检测”的终点，已明确排除，不作为结论。另跑保持启动检测语义的两个型号问法，原表示排名311、185；表头＋行局部替换均排1。不能选择性只报告有利的改写。','',
        '## 替代来源核验（不改Gold）','']
    for a in alternatives:
        md += [f'### {a["id"]} · 向量第{a["vector_rank"]}名', '',
            f'- 原资料：`{a["file"]}`，第{a["page"]}张。',
            f'- 原文、索引正文：均核验通过；实际included_text：{"包含所核事实" if a["actual_context_verified"] else "本来源未进入或未完整包含；不能用父块代替实际发送正文"}。',
            '- 事实：'+'；'.join(q.replace('\n',' / ') for q in a['quote']),
            '- 适用核对：'+a['scope_review'],
            '- 状态：建议业务确认后登记替代；本轮未自行变更正式标签。','']
    md += ['## 全部20道关注题的阶段归属','']
    for r in rows:
        md += [f'### {r["id"]}（{r["status"]}）','',r['question'],'',FINDINGS[r['id']][0]+'。'+FINDINGS[r['id']][1],'',
            '各阶段指定Gold覆盖：'+' → '.join(f'{k} {s["gold_evidence_found"]}/{s["gold_evidence_total_declared"]}' for k,s in r['stage_scores'].items())+'。','']
    md += ['## 下一步与采用结论','',
        '1. 保留已采用的BGE基础重排。本轮没有证据要求回滚，也没有将向量诊断误当作新的算法改进。',
        '2. 优先核定同事实替代来源，逐项登记型号/参数/单位，不把整页一并批准。050、row-004、row-005的“向量0命中”不能直接用来支持换模型。历史严格分数保留，后续如批准新口径则同时报告旧口径与新口径。',
        '3. 向量层最有实验证据的下一步是可追溯的表示对照：现有窗口 vs 保留原表头/对象/完整行的补充表示。使用完整来源集合重建隔离索引，固定模型、查询、TopK、融合及下游，不默认采用，不将本报告局部替换名次当泛化收益。包含非表格回归，尤其050、006和independent-05反例。',
        '4. 来源/位置包装与子查询分别做单因素对照；保留原问与型号约束。不能靠删型号、将所有问题拆开或全部加大窗口来修。',
        '5. 只有同样表示和查询下仍有稳定缺失，再比较另一离线Embedding模型；本轮没有换模型对照，不能宣称BGE模型整体不适合。',
        '6. 019、IMFH1、条码题和BS-800型号前页的丢失属于向量之后，另列模块问题，不和向量改动一起修。','',
        '**确定结论：当前不能采用“全面加大片段”“全面删元数据”“全面拆查询”或“直接换模型”作为统一修复。表头/对象与参数行结合的表示有明确局部收益；来源绑定漏计和查询/包装影响同样真实，必须分开验证。**','',
        '## 执行与交付','',
        '```powershell',
        '.venv/Scripts/python.exe -X utf8 -m tools.diagnose_vector_recall',
        '.venv/Scripts/python.exe -X utf8 -m tools.probe_vector_representations',
        '.venv/Scripts/python.exe -X utf8 -m tools.render_vector_diagnosis',
        '```','',
        '前一轮BGE接入已跑520通过、10个既有source_consensus失败、1跳过、1预期失败；本轮未改运行算法，不把该历史结果冒充本轮新跑。当前新增检查为全库精确检索、ANN重放、源文引文/表格核验、表示局部替换、报告/来源定位与冻结哈希检查。',
        '报告沿用原生91题阶段表及历史旧排序/BGE前后Context对照；上方新增向量专门诊断。直接看report.html；要点击定位原资料页，使用同目录open_report.cmd。未运行LLM，不提供答案正确率。','']
    DOC.write_text('\n'.join(md),encoding='utf8')
    shutil.copyfile(DOC,OUT/'conclusion.md')
    # Reuse the tested native viewer without re-scoring or rewriting old traces.
    page=(BASE/'report.html').read_text(encoding='utf8')
    esc=lambda s:html.escape(str(s))
    parts=['<section class="panel" id="vectorDiagnosis"><h2>向量召回专项诊断 · 2026-11-16</h2>',
        '<p><b>BGE基础排序已接入；本轮向量算法未改，API调用0。</b>全库52282向量精确计算，20题ANN重放与冻结Top50均一致；30条Gold引文原资料/索引均核验通过。</p>',
        '<p>正式指定Gold全部覆盖44/46；诊断35/40。这是严格来源口径，不是答案正确率。050、row-004、row-005存在已召回的同事实替代资料，不能简单称为向量查不到答案。</p>',
        '<p>表头＋原行局部替换：149→1、70→28、53→1；整父块多题变坏。仅为固定竞争向量的事后表示诊断，不是全库重建/上线收益。</p>',
        '<p>逐题展开可看原文、真实编码窗口、源文件按钮、精确排名和排前面的竞争资料。下方91题旧排序/BGE对照为上一轮冻结结果，本轮没有重写得分。</p>',
        '<p><a href="conclusion.md">完整诊断说明</a></p>',
        '<h3>7道严格Gold覆盖不全题：先分清是否真的没有答案证据</h3><table><thead><tr><th>题目</th><th>指定来源精确排名</th><th>具体原因和对照证据</th></tr></thead><tbody>']
    for r in rows:
        if any(t['exact_completion_rank']>50 for t in r['targets']):
            parts.append(f'<tr><td>{esc(r["id"])}</td><td>{esc([t["exact_completion_rank"] for t in r["targets"]])}</td><td>{esc(FINDINGS[r["id"]][1])}</td></tr>')
    parts+=['</tbody></table><h3>替代来源：事实已核对，未自行加进Gold</h3>']
    for a in alternatives:
        parts.append(f'<details><summary>{esc(a["id"])} · 向量{a["vector_rank"]} · {esc(Path(a["file"]).name)} · 第{a["page"]}张</summary>{button(a["source_id"])}<p>{esc(a["scope_review"])}</p><p>原文/索引均通过；本来源实际Context事实覆盖：{a["actual_context_verified"]}</p><pre>{esc(a["original_text"])}</pre><pre>{esc(json.dumps(a["table_cells"],ensure_ascii=False,indent=2))}</pre></details>')
    parts+=['<h3>20道关注题的完整证据与原因</h3>']
    for r in rows:
        parts.append(f'<details class="vector-case" id="vector-{esc(r["id"])}"><summary>{esc(r["id"])} · {esc(r["question"])} · {esc(r["status"])}</summary><p>{esc(FINDINGS[r["id"]][0])}：{esc(FINDINGS[r["id"]][1])}</p>')
        parts.append('<p>当前BGE各阶段指定Gold覆盖：'+esc(' → '.join(f'{k} {s["gold_evidence_found"]}/{s["gold_evidence_total_declared"]}' for k,s in r['stage_scores'].items()))+'</p>')
        for t in r['targets']:
            sid=register(t['file'],t['page'],Path(t['file']).suffix.lower()=='.pptx')
            parts.append(f'<h4>{esc(t["evidence_id"])} · 精确完整排名{t["exact_completion_rank"]}</h4>{button(sid)}<p>{esc(t["file"])} · 第{esc(t["page"])}页/张 · 原资料核验：{t["original_verified"]} · 索引父块引文核验：通过</p><pre>{esc(t["quote"])}</pre>')
            for p in t['parents']:
                parts.append(f'<details><summary>实际向量小窗口（按相似度）与问题对照</summary><pre>{esc(json.dumps(dict(chunk_id=p["chunk_id"],query_control_ranks=p["query_control_ranks"],no_prefix_rank=p["no_prefix_rank"]),ensure_ascii=False,indent=2))}</pre>')
                for u in p['units']:
                    parts.append(f'<p>窗口{u["ordinal"]} · 全库小窗口排名{u["global_unit_rank"]} · {u["token_count"]} token · 包含完整Gold短引文：{u["contains_full_quote"]}</p><pre>{esc(u["embedding_text"])}</pre>')
                parts.append('</details>')
        reps=[x for x in controls['results'] if x['case']==r['id']]
        parts.append('<details><summary>来源局部表示替换（不是全库重建收益）</summary><pre>'+esc(json.dumps(reps,ensure_ascii=False,indent=2))+'</pre></details>')
        parts.append('<details><summary>精确向量前20竞争资料</summary><pre>'+esc(json.dumps(r['top20'],ensure_ascii=False,indent=2))+'</pre></details></details>')
    parts.append('</section>')
    page=page.replace('<main>','<main>'+''.join(parts),1)
    (OUT/'report.html').write_text(page,encoding='utf8')
    write(OUT/'report_sources.json',sources)
    report_audit=json.loads((BASE/'report_format_audit.json').read_text(encoding='utf8'))
    report_audit.update(vector_cases=len(rows),vector_quotes=30,alternative_sources=len(alternatives),source_locations=len(sources),html_bytes=len(page.encode()))
    write(OUT/'report_format_audit.json',report_audit)
    cmd='@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n'
    (OUT/'open_report.cmd').write_text(cmd,encoding='utf8')
    acceptance=dict(api_calls=0,network_attempts=[],source_quotes_original_verified=30,source_quotes_index_verified=30,
        replay_top50_equal_cases=20,exact_search_vector_count=52282,local_representation_controls=len(controls['results']),
        alternative_sources=len(alternatives),gold_modified=False,runtime_algorithm_modified=False,protected_drift=[p for p,h in protocol['protected'].items() if sha(ROOT/p)!=h])
    assert not acceptance['protected_drift']
    write(OUT/'final_acceptance.json',acceptance)
    print(json.dumps(acceptance,ensure_ascii=False))


if __name__=='__main__':main()
