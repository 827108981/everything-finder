"""Summarize source integrity and independent structure experiments."""
from collections import Counter
import html,json
from tools.validate_structured_material import OUT,ROOT
from tools.validate_material_steps import sha
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert (OUT/'completion.json').exists() and all(sha(k)==v for k,v in p['hashes'].items())
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').open(encoding='utf-8')]
    assert len(rows)==177 and len({(r['id'],r['arm']) for r in rows})==177
    audit=json.loads((OUT/'structure_audit.json').read_text(encoding='utf-8'))
    units=json.loads((OUT/'table_units.json').read_text(encoding='utf-8'))
    seq=json.loads((OUT/'row_sequence_audit.json').read_text(encoding='utf-8'));assert not seq
    cases={c.case_id:c for n in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/n,reviewed_only=False).cases}
    baseline={r['id']:r for r in (json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')) if r['arm']=='baseline'}
    # Entire quote presence supplements declared source scoring; not semantic grading.
    quote_audit=[]
    for r in rows:
        before=canonical_content(baseline[r['id']]['context']);after=canonical_content(r['context'])
        for n,e in enumerate(cases[r['id']].gold_evidence,1):
            q=canonical_content(e['quote'])
            if (q in before)!=(q in after):quote_audit.append({'id':r['id'],'arm':r['arm'],'evidence':n,'before':q in before,'after':q in after,'quote':e['quote']})
    unlinked=[r['id'] for r in rows if r['arm']=='C' and not r['parents'] and r['changed']]
    assert not unlinked,'Unlinked controls must reproduce baseline exactly'
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    compare=[]
    by={(r['id'],r['arm']):r for r in rows}
    for cid in baseline:
        control=by[cid,'P'];table=by[cid,'T']
        compare.append({'id':cid,'gains_vs_P':sorted(set(table['after']['gold_completion_ranks'])-set(control['after']['gold_completion_ranks'])),'losses_vs_P':sorted(set(control['after']['gold_completion_ranks'])-set(table['after']['gold_completion_ranks'])),'same_row_hits':control['row_hits']==table['row_hits']})
    assert all(r['same_row_hits'] for r in compare)
    (OUT/'quote_presence_audit.json').write_text(json.dumps(quote_audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'table_matched_pool_comparison.json').write_text(json.dumps(compare,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    notes={
      'structure':'256表按原生首行标记等保守规则接受，235表回退；接受行的单元格和顺序均可在索引中找到。自动文本一致性并不证明表头语义或表外条件完整。',
      'blank_cells':{'count':sum(any(not c for c in u['cells']) for u in units),'risk':'JSON单元格数组保留空位，但检索文本仅按列顺序换行呈现，空值可能不够清楚；不能据此宣称已解决所有列对齐问题。'},
      'notes_scope':'本原型未完整继承表外脚注、多级表头、跨表条件；PDF、Word、Excel不在本轮结构改造范围。',
      'header_sample':'抽查header_review_sample.json中的8表，包含软件路径、版本、样本现象、质控规则和机型比较；列名与原单元格可对应。属于小样本结构检查，不是业务人员逐表认证。',
      'count_budget':'C按原始页面数扣8段和单文档3段额度，文字预算12000；多页组合没有绕过预算。T单行仍按一个候选计额度，未实现同表多行合并。',
      'provenance':'P/T来自完全相同的行检索结果，P投影原页，T投影表头与行；原候选始终保留。P不新增表格表示，因此P提升不能直接归因于结构化。',
      'scores':'首次真实计算本轮新视图的Cross-Encoder分数；同一问句/正文对在本轮缓存复用。没有新增生成模型调用，若证据门槛不通过则不组合、不做全量回答测试。'}
    notes['023_coverage_gap']='报警表有清楚的标识/描述/原因/解决方法表头，但PPT内部没有firstRow=1，因此被回退，1520个行片段没有CSDB。023本轮未覆盖，不能据其无改善否定表格行表示。'
    notes['040_regression']='T最终上下文缺少原来已有的小样本量项目排查引文；也没有第二分支引文。进入上下文的新增表格行包括急诊进样等信息，不能替代重复性故障的原因。'
    notes['021_regression']='C补入27/28页定义和可能原因，但同文件3页额度随后用于29页，原有30页“分析哪次定标有问题／上次／当前”分支被排除；原句在最终上下文确实缺失。'
    coverage=[]
    for cid,c in cases.items():
        if c.review_status=='disputed':continue
        for n,e in enumerate(c.gold_evidence,1):
            name=str(e['file']).replace('\\','/').split('/')[-1].casefold()
            tables=[t for t in audit if t['file'].replace('\\','/').split('/')[-1].casefold()==name and t.get('page')==e.get('page')]
            if tables:coverage.append({'id':cid,'evidence':n,'file':e['file'],'page':e['page'],'accepted_tables':sum(t['status']=='accepted' for t in tables),'fallback_reasons':[t['reason'] for t in tables if t['status']=='fallback']})
    (OUT/'declared_table_coverage.json').write_text(json.dumps(coverage,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'limitations.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 资料结构验证（2026-11-05）','','## 范围与结论','',
      '当前AI索引 `g000033-71ec3c52f3`。本轮按结构保真、表格独立、跨页独立顺序执行。原18题固定集合可计分17题，新增集合可计分31题；007 disputed排除、024 pending和回归10题pending单列。三个独立分支各59题，共177组新本地检索/评分/选材实验，未修改生产或Gold。',
      '结构可提取不等于检索有效。表格实验另有相同行候选、取回整页的P对照，以辨别额外召回和表格表示的效果。C独立于P/T，关联单元按真实页面数计预算。','',
      '## 结构保真','',
      '74份当前索引对应PPTX，491个原生表格：接受256表、生成1520个行片段；235表回退，其中合并单元格74表、表头未被原文件明确标记128表、表头不明确33表。',
      '接受片段的单元格文字全部存在于原表与当前索引；1520个行片段对应原文行顺序核验无差异。单位来自表头或原单元格，未转换数值。3项新增测试覆盖单位/来源保留、合并回退、原文与索引数值不一致回退。',
      '仍有267行包含空单元格，当前文本表现形式不够明确；表外脚注、跨表条件、多级表头、PDF等尚未完整处理。因此保真结论仅限已核对的原生单元格与顺序，不能宣布资料结构问题整体解决。','',
      '023的报警表未设置PPT内部首行标记，被保守规则回退；本轮行库中没有CSDB行。它属于未覆盖，不能把023无改善解释为“拆行已经试过但无效”。覆盖明细见declared_table_coverage.json。','',
      '## 独立效果','',
      '| 分支 | 正式正证据命中（44题） | 平均完整证据覆盖 | 改变上下文题数（59题） | 新增命中题 | 丢失原证据题 | 新增负例原句题 |',
      '|---|---:|---:|---:|---|---|---|']
    for arm in ('P','T','C'):
        s=summary[arm]
        lines.append(f"| {arm} | {s['before_hit']}→{s['after_hit']} | {s['before_coverage']:.1%}→{s['after_coverage']:.1%} | {s['changed']} | {','.join(s['gains']) or '无'} | {','.join(s['losses']) or '无'} | {','.join(s['new_actual_negative_quotes']) or '无'} |")
    lines+=['','P=同样行检索取回原页；T=取回表头与单行；C=跨页场景与处理组合。新负例原句只表示上下文新增该原句，不代表模型采用它；旧报告按父页面ID标记的负例另保留在cases.jsonl中。完整引文检查也保存在quote_presence_audit.json，避免将等价来源误判成纯粹的内容丢失。','',
      '## 固定集合分开比较','',
      '| 集合 | P命中前→后 | T命中前→后 | C命中前→后 |',
      '|---|---:|---:|---:|']
    for label,fn in [('原18题固定集合（有效17题）',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])<=18),('新增集合（有效31题）',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])>18),('回归10题 pending',lambda r:r['cohort']=='regression10_pending'),('024 pending',lambda r:r['cohort']=='pending024')]:
        cells=[]
        for arm in ('P','T','C'):
            rs=[r for r in rows if r['arm']==arm and fn(r) and r['before']['gold_evidence_recall'] is not None]
            cells.append(f"{sum(r['before']['gold_hit'] for r in rs)}→{sum(r['after']['gold_hit'] for r in rs)} / {len(rs)}")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines+=['','## 是否进入组合与回答阶段','',
      '证据门槛：T、C必须各自保留原有完整证据，且不能新增已标注的错误场景原句。只有两者均通过，才考虑组合；没有收益也不因“门槛通过”而自动采用。',
      '实际门槛：'+json.dumps({k:summary[k]['gate'] for k in ('T','C')},ensure_ascii=False)+'。本轮回答生成调用为0；未给出新的回答通过率。','',
      '跨页C改善019，但021丢失原来已有的“先分析哪次定标有问题”分支，完整原句在上下文确实缺失。表格T使040丢失原有小样本量项目排查引文，也不能用其他格式等价正文解释。两者未通过，因此按预设停止组合与全量回答生成。','',
      '## 具体改变与核查','']
    for r in rows:
        if r['gained'] or r['lost'] or r['actual_new_negative_quotes']:
            lines.append(f"- {r['id']} / {r['arm']}：新增{r['gained']}；丢失{r['lost']}；新增负例原句{r['actual_new_negative_quotes']}。")
    lines+=['','## 边界与下一步','',
      '本轮是原生PPTX表格的影子关键词路线和既有跨页关系原型，未重新建立完整向量索引，不能外推为所有格式、所有切片策略均已验证。实际预算、场景条件、评分权重固定；P/T检索路线使用相同分词但不完全等同生产word_v2的完整匹配加成。',
      '优先依据P对T的相同候选比较判断表格表示是否有价值。若只是P有效，收益主要来自新增候选；若T丢失事实，需要先处理同行/同表内容合并、适用条件和最终配额。跨页方案若失败，应检查完整组合的条件判定与配额，而不是扩大相邻页窗口。',
      '当前不建议启动全量索引重建。下一步先解决结构覆盖和保留方式：表头识别不能只依赖PPT样式标记；行视图需明确空值与列绑定；跨页关系必须覆盖处理分支；同源片段合并及最终选择需避免以新增片段挤掉已有必要事实。这些仍需独立验证，不能保证零影响。P/T底层行命中相同，但P去重回父页、T保留多行，候选数量和配额交互也属于本轮干预，不能宣称纯文本表示的因果效果。',
      '所有抽查引文与页码在结构JSON和逐题上下文中可复查。正式Gold状态和原文文件、生产代码、设置、AI索引均通过前后哈希检查；未组合上线。','']
    report='\n'.join(lines);(OUT/'report.md').write_text(report,encoding='utf-8')
    (ROOT/'docs/rag_structured_material_validation_20261105.md').write_text(report,encoding='utf-8')
    body='<!doctype html><meta charset="utf-8"><title>资料结构独立验证</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;margin:8px;padding:12px}</style><h1>资料结构独立验证</h1><pre>'+html.escape(report)+'</pre>'
    for r in rows:body+='<details><summary>'+html.escape(f"{r['arm']} {r['id']} gained={r['gained']} lost={r['lost']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'delivery_verification.json').write_text(json.dumps({'evidence_rows':len(rows),'same_pool_table_comparisons':len(compare),'row_order_mismatches':len(seq),'unlinked_C_controls_changed':unlinked,'hashes_unchanged':True,'new_generation_calls':0,'tests_passed':3},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
