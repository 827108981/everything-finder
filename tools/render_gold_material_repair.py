"""Native RAG report: annotation revision, fresh changed-query retrieval, lineage."""
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from tools.validate_selection_factorial import ROOT, write, sha
from tools.retrieve_repaired_material_20261116 import OUT
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset, GoldDataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence, load_chunk_rows
from local_full_text_search.rag.evaluation.reporting.service import build_report_detail, render_html, ReportConfig


def main():
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json').read_text(encoding='utf8'))
    allcases=tuple(c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases)
    byid={c.case_id:c for c in allcases}
    fresh=[json.loads(x) for x in (OUT/'fresh_retrieval.jsonl').read_text(encoding='utf8').splitlines()]
    cases=tuple(byid[r['id']] for r in fresh)
    audit=json.loads((OUT/'chunk_lineage_audit.json').read_text(encoding='utf8'))
    labels=json.loads((OUT/'validation.json').read_text(encoding='utf8'))
    changes=json.loads((OUT/'gold_changes.json').read_text(encoding='utf8'))
    old={r['id']:r for r in map(json.loads,(ROOT/'reports/rag_bge_base_adoption_20261116/B.jsonl').read_text(encoding='utf8').splitlines())}
    sources={}
    def register(record):
        path=record.get('file_path') or record.get('file') or record.get('path')
        if not path:return []
        loc=record.get('location') or record.get('location_text') or record.get('position') or ''
        page=record.get('page_number') or record.get('page');slide=record.get('slide_number') or record.get('slide')
        if Path(path).suffix.lower() in {'.pptx','.ppt'}:slide,page=slide or page,None
        if not page and not slide:
            m=re.search(r'第\s*(\d+)\s*(页|张)',loc)
            if m:
                if m[2]=='张':slide=int(m[1])
                else:page=int(m[1])
        source=dict(path=str(Path(path)),location=dict(location_text=loc,page_number=int(page) if page else None,slide_number=int(slide) if slide else None))
        assert Path(path).is_file(),path
        key=hashlib.sha256(json.dumps(source,ensure_ascii=False,sort_keys=True).encode()).hexdigest()[:24]
        sources[key]=source
        return [key]
    def links(record):
        return ''.join(f'<button type="button" data-original-source="{sid}">打开原资料对应页</button>' for sid in register(record))
    def pre(value):return '<pre>'+html.escape(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2))+'</pre>'
    with ro('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db') as c:chunks=load_chunk_rows(c)
    rows=[]
    for r in fresh:
        case=byid[r['id']]
        rows.append(dict(id=case.case_id,question=case.question,group_id=case.group_id,review_status=case.review_status,
            cohort='formal' if case.review_status=='reviewed' else 'diagnostic',scores=r['scores'],stages=r['scores'],
            candidate_stages=r['stages'],events=r['events'],answer={},gold_judgement={},actual_behavior='not_run',
            gold_mappings=[asdict(m) for m in map_case_evidence(case,chunks)]))
    dataset=GoldDataset(ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json',sha(ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json'),cases)
    detail=build_report_detail(dataset,dict(run_id='material_repair_20261116',ai_bundle_id=manifest['bundle_id'],
        mode='修订后3题真实本地检索；API前验证；不改变检索算法'),rows,ReportConfig(page_tolerance=0))
    for c in detail['cases']:
        for evidence in c.get('gold_evidence',[])+c.get('hard_negatives',[]):
            evidence['source_ids']=register(evidence)
            for alt in evidence.get('alternative_sources',[]):evidence['source_ids'].extend(register(alt))
        for s in c['stage_data'].values():
            for candidate in s['candidates']:candidate['source_ids']=register(candidate)
        assert all(c['stage_summaries'][s]['gold_found']==fresh[[r['id'] for r in fresh].index(c['id'])]['scores'][s]['gold_evidence_found'] for s in c['stage_summaries'])
    unitroot=ET.parse(OUT/'full_unit_results.xml')
    tests=unitroot.findall('.//testcase')
    failures=[(t.get('classname'),t.get('name')) for t in tests if t.find('failure') is not None or t.find('error') is not None]
    prior=json.loads((ROOT/'reports/rag_bge_base_adoption_20261116/final_acceptance.json').read_text(encoding='utf8'))
    assert sorted(failures)==sorted(tuple(x) for x in prior['existing_failures'])
    counts=dict(passed=sum(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests),failed=len(failures),
        skipped_or_xfail=sum(t.find('skipped') is not None for t in tests),new_failures=0)
    status=dict(Counter(c.review_status for c in allcases))
    stats=audit['stats']
    md=['# 资料标注修复与向量切片根因核查（2026-11-16）','',
        '## 完成的资料修订','',
        '新增业务 Gold v5.1，主集49题、全套91题。历史版本保留；Q040仍不在当前集合。审核状态保持原样：'+str(status)+'。',
        '- 050、row-004、row-005、independent-05加入已批准、逐项核实的替代来源：6个来源页、8个事实级来源登记。相同事实按OR计分，不能变成要求全部来源召回。',
        '- independent-05改为比较A&T与Medica原理及识别方法，不再要求判断未知现场安装状态。直接/间接法解释和模块对照是必答；第三品牌例子不额外设成必答。',
        '- independent-06只问BS-2800M、BS-430分析部与电脑的通信接口，不再问现场是否连通。',
        '- Q028只问临床结果普遍偏低的排查方向，去掉板卡推断及相关必答项。',
        '- 应用评测页的默认Gold路径指向新版已审核主集；未写入用户持久化设置，未改Adapter默认状态。','',
        '## 修订后的真实检索（无API）','',
        '以下使用三个新问题重新运行本地QueryPlan、关键词、向量、融合、BGE基础重排、条件检查、选段及Context。未调用生成API，也不判断答案正确率。','',
        '| 题目 | 向量完整证据 | 融合 | 重排 | 实际Context |','|---|---|---|---|---|']
    for r in fresh:
        def s(stage):
            v=r['scores'][stage];return f"{v['gold_evidence_found']}/{v['gold_evidence_total_declared']}，首条 {v['first_gold_rank'] or '未命中'}"
        md.append('|'+r['id']+'|'+'|'.join(s(k) for k in ('vector','fusion','rerank','context'))+'|')
    md += ['',
        '改问法/改Gold与算法收益分开：050、row-004、row-005在同一批旧向量候选上登记替代后，完整事实从漏计变为命中（名次26、4、1）；这是计分修正，不是向量算法提升。',
        'independent-05仍缺通用直接/间接原理说明的Gold引文，模块对照事实已经进入Context；不能说该题已经全部解决。independent-06由关键词补回后Context完整，向量本身仍漏。','',
        '## 切片问题究竟发生在哪里','',
        '1. 原资料：表头、型号、数据行在真实PPT表格单元格中有关系。',
        '2. PPT解析：当前快解析pptx_stream逐个提取a:p段落，单元格摊平为换行正文；未保存表格坐标、行列绑定、合并关系。16个所查表格页的流式解析重放均与发布的全文索引正文一致。常规PPT解析器会保留行内竖线，但也没有完整表格结构。',
        '3. 普通索引→RAG：BlockReader只读raw_text和页码等字段，SourceBlock不携带表格结构extra；父块按普通文字组装。26个所查父块在发布generation33原始块中可找到完整正文。',
        '4. 向量子块：VectorUnitBuilder按换行、句号及128目标/192最大/32重叠token切窗口。line_window是切完后的类型标签，并不继承表头。embedding_text只额外加文件名、位置，没有页标题和表头继承。',
        '5. 检索：每个父块用最相似的子窗口参与排名；只有先找到子窗口，才返回完整父块。父块里有完整表，不代表每个用于向量匹配的小窗口知道数值属于哪个型号。',
        '数据库本身未发现存错证据。26个目标父块重建出的子块ID、正文、编码正文、位置、token数与库中全部一致；只改SQLite正文或只调重排无法恢复上游已经丢掉的表格关系。','',
        '## 全量PPTX表格扫描边界与数量','',
        f'扫描当前AI索引的74份可访问PPTX，共3532个已索引幻灯片页，459页有原生表格、491个原生表格。404个非合并且首行标签可自动检查的表格包含2364个非空数据行。',
        f'- 2359行完整文字仍在某个父块中；其中68行不能在单个向量窗口中完整找到。',
        f'- 2291行可在单个窗口完整找到，其中924行所在窗口没有同时包含全部首行标签；1367行同时保留了这些标签。',
        '- 此处首行标签共现是结构风险统计，不等于924行都检索失败，也不等于首行必然是真正语义表头；87个含合并等复杂表格未自动判断。',
        '- PDF、扫描图、旧PPT不在原生PPTX表格统计内；不能据此声称所有资料均检查完。',
        '- 已核查74份PPTX的真实幻灯片顺序与流式XML顺序一致，本轮没有发现页码错位。','',
        '## 代表问题与归因','',
        '| 题目 | 具体问题 | 已有单因素证据及限制 |','|---|---|---|',
        '| row-003 待机首样本启动 | 型号表头只在窗口0；4–5分钟、1分钟在窗口1，窗口1没有BS-2000/BS-2800M列名 | 原查询全库精确父块149；仅将该来源改成原表头+对应行表示为1。证明该来源表示影响大，不代表全库重建保证同幅提升 |',
        '| independent-06 通信接口 | BS-430串口位于第二窗口，缺少通信表头；第一窗口混有多种机型/IP/COM，BS-2800M网口仍与表头同处 | 新问题全库精确571，关键词5补回、重排及Context1；不能把整题都归因于表头丢失 |',
        '| independent-05 ISE原理 | 模块对照表与通用原理说明分散；旧来源包装干扰排名，原理说明页仍漏 | 新问题精确原理页692、模块事实最佳141；旧查询去包装的局部控制684→1，而整行重写反而更差，不能统一套同一种行改写 |',
        '| row-004 样本类型 | 正确替代资料向量4已经召回；指定旧Gold表头/目标行并未必拆开 | 主要已确认漏计，不能冒充切片修复 |',
        '| row-005 电极斜率 | 正确替代资料向量1；原Gold行序号不同 | 只登记斜率事实，两个来源的测试速度不同，没有把整页参数合并 |',
        '| Q050 暗计数 | PDF标题与完整四原因分到不同窗口；同事实PPT已进向量26/31 | 替代登记修复漏计；PDF表示仍有问题，但不是四原因资料完全不存在 |',
        '| Q006 缓冲容器 | 启动、停止、空瓶分在多个窗口，需多条件组合 | 整父块替换和拆问未改善；不能断言合并切片一定修好 |',
        '| Q019 DEV | 场景页和处理页都在向量Top50，处理页在融合后约31、去重后30未进重排前20 | 主要是融合/重排入口候选截断，不是当前向量没有资料 |','',
        '## 验证及剩余工作','',
        '- 17条原文引文在原资料及当前已发布AI正文中核实通过；JSON/JSONL一致、GoldLoader加载通过；审核状态不变。',
        f'- 全部RAG测试：{counts["passed"]}通过，10个已存在的source_consensus失败，1跳过、1预期失败；失败名称与接入BGE前基线完全一致，新增失败0。替代证据针对测试6/6通过。',
        '- 三道新问法均跑真实本地检索；全库52282向量精确余弦核对证实independent-05/06的指定事实在Top50之外，不是仅ANN没搜到。',
        '- 已发布AI父块、向量子块、历史Gold和用户设置哈希无漂移；API调用0。',
        '- 本轮只完成资料标注修复及根因定位，尚未实施新的解析/切片策略，也没有发布重建索引。',
        '- 下一步应单独修复“原始表格结构提取及传递→表头/行/条件绑定的向量表示”，对非表格保留旧策略，先用隔离新索引验证全部受影响表格与独立问题；不要直接替换现有库。',
        '- 复杂合并表头、PDF表格恢复及来源包装干扰要分别验证；不能把行级表示在row-003的收益推广成所有表格收益。','']
    document='\n'.join(md)
    (ROOT/'docs/rag_gold_material_chunk_lineage_20261116.md').write_text(document,encoding='utf8')
    (OUT/'conclusion.md').write_text(document,encoding='utf8')
    (ROOT/'tests/rag/fixtures/rag_gold_v5_1.md').write_text('# Gold v5.1资料修订\n\n'+ '\n'.join(md[4:11])+'\n\n`gold_evidence`每项为一条必须覆盖的事实；其`alternative_sources`为用户已批准的同事实来源，按OR计分。同一事实任一完整来源即可，不增加分母。来源包含file/page/quote/approved/approval_basis，审核状态继承旧版。\n',encoding='utf8')
    changes_html=''
    for ch in changes:
        c=ch['after'];oldcase=ch['before']
        ev=''.join('<div class="evidence">'+html.escape(e['file'])+' · '+str(e['page'])+links(e)+pre(e['quote'])+
            ''.join('<p>可接受替代：'+html.escape(a['file'])+' · '+str(a['page'])+links(a)+'</p>' for a in e.get('alternative_sources',[]))+'</div>' for e in c['gold_evidence'])
        changes_html+=f'<details><summary>{html.escape(c["id"])} · {html.escape(c["question"])}</summary><p>原问题：{html.escape(oldcase["question"])}</p><p>新问题：{html.escape(c["question"])}</p><p>状态：{c["review_status"]}，未变</p><h4>本题必须回答</h4>{pre(c.get("required_facts",[]))}<h4>原资料及替代来源</h4>{ev}</details>'
    compare=''
    for r in fresh:
        before=old[r['id']]
        def entries_html(entries):return ''.join('<details><summary>'+html.escape(e['filename']+' · '+e['location'])+'</summary>'+''.join(links(s) for s in e.get('sources',[]))+pre(e['included_text'])+'</details>' for e in entries)
        compare+=f'<details><summary>{html.escape(r["id"])}：旧问题 / 新问题与实际Context</summary><div class="detail-grid"><div><h4>旧问题，历史BGE基础排序</h4>{pre(before["question"])}{entries_html(before["entries"])}</div><div><h4>修订问题，真实重新检索</h4>{pre(r["question"])}{entries_html(r["entries"])}</div></div><p>问题和标签已修订；这不是同问题、同标签的算法A/B结论。</p></details>'
    lineage_html=''
    for r in audit['selected_lineage']:
        lineage_html+='<details><summary>'+html.escape(Path(r['file']).name+f' · {r["slide"] or r["page"]}')+'</summary>'+links(r)+f'<p>发布解析器：{r["parser_name"]}；父块在原始块中：{r["parent_text_in_joined_blocks"]}</p>'+pre(r['parent_text'])+''.join(f'<h4>向量窗口 {u["ordinal"]} · {u["token_count"]} token</h4>'+pre(u['embedding_text']) for u in r['units'])+'</details>'
    annotation=json.loads((OUT/'annotation_only_scores.json').read_text(encoding='utf8'))
    table=''.join(f'<tr><td>{r["id"]}</td><td>{r["before"]["gold_evidence_found"]}/{r["before"]["gold_evidence_total_declared"]}</td><td>{r["after_on_frozen_old_query"]["gold_evidence_found"]}/{r["after_on_frozen_old_query"]["gold_evidence_total_declared"]}</td><td>{r["after_on_frozen_old_query"]["first_gold_rank"]}</td></tr>' for r in annotation if not r['question_changed'])
    banner=f'''<section class="panel" id="materialAudit"><h2>资料修订完成；解析与切片根因已定位</h2>
    <p>新版v5.1，全套91题审核状态不变；6题标注修订、3道新问题真实重跑。API调用0，BGE基础排序、发布AI索引及持久化设置未改。</p>
    <p>替代正确来源按同一事实OR计分。下面原生热力图及阶段详情仅为3道改问法题的新运行，不是91题新一轮完整检索报告。</p>
    <p><b>根因：</b>PPT流式解析摊平单元格；向量切片按token窗口切开，不继承表头。不是所有表都缺表头，也未发现数据库存错。</p>
    <p>74份PPTX、491个原生表格；可检查表格的924行与首行标签不共处向量窗口、68行跨窗口；不是检索失败率。26父块重建与已存片段一致，16表格页解析重放与发布块一致。</p>
    <p id="sourceStatus">双击同目录open_report.cmd后，可点击打开原资料页；直接打开HTML可离线查阅。</p>
    <p><a href="conclusion.md">完整核查结论</a> · <a href="final_acceptance.json">验收记录</a></p>
    <h3>六题资料与Gold前后审核</h3>{changes_html}
    <h3>仅登记替代来源的计分变化（同一批旧向量候选）</h3><table><tr><th>题号</th><th>原完整事实</th><th>修订完整事实</th><th>修订首条排名</th></tr>{table}</table><p>这是纠正漏计；不是召回算法改善。</p>
    <h3>三题新旧实际Context对照</h3>{compare}
    <details><summary>原始块 → 父块 → 向量子块：26父块逐条核对</summary>{lineage_html}</details>
    <p>RAG单元回归：{counts['passed']}通过、10个既有失败、1跳过、1预期失败；新增失败0。未实施新的解析/切片策略，未声称问题全部修好。</p></section>'''
    page=render_html(detail).replace('<main>','<main>'+banner,1)
    page=page.replace('<b>${esc(e.evidence_id)}</b>','<b>${esc(e.evidence_id)}</b> ${sourceLinks(e)}')
    page=page.replace('<pre>${content}</pre>','${sourceLinks(c)}<pre>${content}</pre>')
    extra=r'''
function sourceLinks(record){return (record.source_ids||[]).map(id=>`<button type="button" data-original-source="${esc(id)}">打开原资料对应页</button>`).join('');}
document.addEventListener('click',async e=>{const b=e.target.closest('[data-original-source]');if(!b)return;e.preventDefault();const s=document.getElementById('sourceStatus'),token=new URLSearchParams(location.search).get('token');if(!token){s.textContent='请使用同目录open_report.cmd打开，再定位原资料。';return;}try{const r=await fetch('/open-source',{method:'POST',headers:{'Content-Type':'application/json','X-Report-Token':token},body:JSON.stringify({id:b.dataset.originalSource})});const d=await r.json();s.textContent=d.message||d.error;}catch(error){s.textContent='定位失败：'+error;}});
'''
    page=page.replace('\ninit();\n</script>',extra+'\ninit();\n</script>')
    (OUT/'report.html').write_text(page,encoding='utf8')
    write(OUT/'report_sources.json',sources)
    (OUT/'open_report.cmd').write_text('@echo off\r\ncd /d "'+str(ROOT)+'"\r\n"'+str(ROOT/'.venv/Scripts/python.exe')+'" -X utf8 -m tools.serve_rag_report --report "'+str(OUT/'report.html')+'"\r\n',encoding='utf8')
    protected_drift=[p for p,h in labels['protected'].items() if sha(ROOT/p)!=h]
    assert not protected_drift and all(Path(s['path']).is_file() for s in sources.values())
    final=dict(material_repairs_completed=True,changed_cases=6,fresh_questions=3,total_cases=len(allcases),
        approved_alternative_fact_sources=8,verified_original_and_index_quotes=17,json_jsonl_equal=True,loader='pass',
        review_status_counts=status,review_status_unchanged=True,tests=counts,existing_failures=failures,new_failures=[],
        api_calls=0,persistent_settings_changed=False,published_index_changed=False,protected_drift=protected_drift,
        source_buttons=len(sources),chunk_lineage_reproduced=len(audit['rebuilt_selected_parents']),
        parsed_pages_reproduced=len(audit['selected_parser_replays']),table_scan=stats,
        source_materials_modified=False,production_slicer_modified=False,
        unresolved=['ISE通用原理引文未入Context','ISE及通信接口向量Top50仍缺指定事实','解析结构丢失与切片表头继承尚未修复'],
        scope='annotation repairs and upstream diagnosis; not algorithm improvement or answer correctness verification')
    write(OUT/'report_detail.json',detail)
    write(OUT/'final_acceptance.json',final)
    print('Report generated:',OUT/'report.html','native cases',len(detail['cases']),'sources',len(sources),flush=True)


if __name__=='__main__':main()
