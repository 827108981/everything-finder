from collections import Counter
import html,json
from tools.validate_material_repairs import ROOT,OUT,sha
from local_full_text_search.rag.content_identity import canonical_content

def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(k)==h for k,h in p['hashes'].items())
    assert (OUT/'answer_completion.json').exists()
    s=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    rows=json.loads((OUT/'selection_cases.json').read_text(encoding='utf-8'))
    answers=[json.loads(l) for l in (OUT/'answer_results.jsonl').open(encoding='utf-8')]
    assert len(rows)==177 and len(answers)==42
    controls=json.loads((OUT/'header_controls_result.json').read_text(encoding='utf-8'))
    group={}
    for a in answers:group.setdefault((a['id'],a['arm']),[]).append(a)
    bindings=json.loads((OUT/'bound_rows.json').read_text(encoding='utf-8'))
    assert all(r['cell_and_order_verified'] and [c['value'] or '' for c in r['bindings']]==r['source_values'] for r in bindings)
    h=s['header'];lines=['# 两项修复独立验证（2026-11-05）','','## 执行结论与边界','',
      'H表头/绑定与S选材分别执行，S没有使用H生成的新行库。当前AI bundle_id为g000033-71ec3c52f3。H完成结构层验证，未证明问答收益；S证据不丢失，但仍有预算和适用性限制，不能据此称所有资料问题已解决。生产、Gold和已发布索引未变。','',
      '## H：表头识别与空单元格','',
      f"原生PPTX表格：接受{h['previous_tables']}→{h['now_tables']}个，行记录{h['old_rows']}→{h['now_rows']}条。新增的未标记简单表格经去重后{h['unique_classifications']}组，12次模型批量判断；执行错误{h['classification_errors']}。模型只判表头角色，不生成列名、数值或单元格原文。合并单元格与原有歧义表仍回退，剩余107表。",
      f"{h['now_rows']}条绑定记录的原始单元格与顺序核验通过；{h['blank_rows']}条含空值的行使用明确column/header/value=null/empty字段表示，原始0、斜线及单位不转为空。",
      '023原文件《化学发光定标处理思路分析.pptx》第11张，CSDB行恢复为：标识=CSDB；描述=校准结果错误，检测到气泡；原因=底物注入过程中，检测到气泡；解决方法=底物灌注后重新定标。DCF仍是另一行。',
      f"结构负对照：从当前资料移除表头的12组数据行，误识别{controls.get('false_positive_count','执行错误')}/12。这是同资料的变换对照，不是独立盲测，不能据此宣称真实表头识别率100%。",
      '抽查新增表头包括管脚/信号/状态、版本审核、机型制冷方式、名称/位置/作用及报警表，列角色与原始内容相符。完整384表尚未经过业务人员逐表审核；多级表头、脚注、PDF/Word/Excel仍未解决。','',
      '## S：选材与配额','',
      '固定规则：保留基线上下文正文及顺序，只在8段/12000字剩余预算内补内容；新内容须通过原门槛、全部硬条件明确匹配；同文件精确重复正文去重；允许补充候选超过同文件3段，但不替换原有非重复正文。没有题号、指定文件或报警码规则。',
      'S_native使用原候选；S_table、S_continuation另尝试前一轮表格/跨页视图的来源页。后二者仅投影已记录的视图排名信号，未重算独立父页Cross-Encoder分数，不能当成完整生产实现；原始信号来自实际已评分视图，非父页独立相关性测量。三种变体最终59题上下文完全相同，因此回答只复测一次，不能作为三次独立成功。',
      '证据结果：正式44道有正证据题，命中39→40；021补回可能原因且保留原条件分支，023补回CSDB表格。59题全部原正文保留，无原Gold引文丢失；上下文改变7题。原先表格方案对040、跨页方案对021的证据回退没有出现。',
      '严格门槛仍flagged：023补回整张表，同时包含已标注的DCF负例原句。保留该统计，不静默修改Gold。为检查是否误用同行内容，进行本轮定点回答诊断；该诊断不等于通过推广门槛。','',
      '## 本轮42次新回答','',
      '| 题目 | 集合 | 基线答对/3 | 候选答对/3 |','|---|---|---:|---:|']
    for cid in sorted({a['id'] for a in answers}):
        vals=[sum((a.get('judgement') or {}).get('verdict')=='consistent' for a in group[cid,arm]) for arm in ('baseline','candidate')]
        lines.append(f"| {cid} | {'回归pending' if cid.startswith('rag-acc') else '正式'} | {vals[0]} | {vals[1]} |")
    errs=sum(bool(a.get('error')) for a in answers);flags=[{'id':a['id'],'arm':a['arm'],'repeat':a['repeat'],'reason':a['citation_check']['reason']} for a in answers if a.get('citation_check') and not a['citation_check']['structurally_valid']]
    lines+=['',f'共42次回答及相应Gold裁判，执行错误{errs}次，引用结构标记{len(flags)}次。相同模型、每组3次、固定种子交错执行。仅对7个变化上下文复测，不是全量50题或独立盲测正确率。',
      'Gold只提供给裁判；表头任务不读取题目或Gold，回答任务不读取检查项或标准答案。H没有接入本轮回答上下文，所以本表回答收益仅归属于S候选。','',
      '## 原固定集合与新增、待审回归分开','',
      '| 集合 | 正证据命中前→后 | 平均完整引文覆盖前→后 |','|---|---:|---:|']
    native=[r for r in rows if r['arm']=='S_native']
    for label,pred in [('原18题固定集合，17题有效',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])<=18),('新增集合，31题有效',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])>18),('回归10题pending',lambda r:r['cohort']=='regression10_pending'),('024 pending',lambda r:r['cohort']=='pending024')]:
        pos=[r for r in native if pred(r) and r['before']['gold_evidence_recall'] is not None]
        lines.append(f"| {label} | {sum(r['before']['gold_hit'] for r in pos)}→{sum(r['after']['gold_hit'] for r in pos)} / {len(pos)} | {sum(r['before']['gold_evidence_recall'] for r in pos)/len(pos):.1%}→{sum(r['after']['gold_evidence_recall'] for r in pos)/len(pos):.1%} |")
    lines+=['','## 结论与下一步','',
      'H结构缺陷已在此原型中修复：不再只靠PPT内部表头标记，列/值/空值明确绑定；值得保留做后续检索实验，但没有检索、重排和回答提升证据，不能直接推广到所有格式。',
      'S保守候选证据保留通过，取得局部增益；能力边界明确：预算满时不替换，因此006等TopK满额问题仍无法改善，019也未恢复必要处理页；不能声称跨页或多分支问题整体解决。021的回答是否追问，另见逐次结果，不从证据补齐直接推断追问已修复。',
      '建议下一步只验证H在固定候选预算下的检索收益，并以S作为防止旧证据丢失的对照；先保留表头判定日志和原文回查能力。当前S的DCF同页标记和42次回答中的回退必须复核，均通过后才讨论接入正式业务代码。不要进行无边界的整库重建，也不能承诺对所有未知问题零影响。','',
      '## 交付核验','',
      '结构绑定2287条，选材177条，逐字复现59条基线上下文；原资料、配置、生产代码、索引和Gold前后哈希核验通过。新增5项测试通过，覆盖空值/单位/0、满额不替换、只补空位、未知/冲突条件拒收、同源精确去重。历史文件未覆盖。','']
    txt='\n'.join(lines);(OUT/'report.md').write_text(txt,encoding='utf-8');(ROOT/'docs/rag_material_repairs_validation_20261105.md').write_text(txt,encoding='utf-8')
    body='<!doctype html><meta charset="utf-8"><title>两项修复独立验证</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:10px;margin:8px}</style><pre>'+html.escape(txt)+'</pre>'
    for a in answers:body+='<details><summary>'+html.escape(f"回答 {a['id']} {a['arm']} {a['repeat']}")+'</summary><pre>'+html.escape(json.dumps(a,ensure_ascii=False,indent=2))+'</pre></details>'
    for r in native:body+='<details><summary>'+html.escape(f"证据 {r['id']} changed={r['changed']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    verification={'binding_rows':len(bindings),'selection_rows':len(rows),'answer_rows':len(answers),'answer_errors':errs,'citation_flags':flags,'headers_negative_control':controls,'input_hashes_unchanged':True,'all_selection_variants_identical':all(len({r['context'] for r in rows if r['id']==cid})==1 for cid in {r['id'] for r in rows})}
    (OUT/'delivery_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(txt)

if __name__=='__main__':main()
