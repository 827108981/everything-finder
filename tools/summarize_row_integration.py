from collections import Counter
import html,json
from tools.validate_row_integration import ROOT,OUT,sha
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in p['hashes'].items())
    for name in ('completion.json','row_answer_completion.json','variation_answer_completion.json','remaining_answer_completion.json'):assert (OUT/name).exists()
    rows=[json.loads(l) for l in (OUT/'cases.jsonl').open(encoding='utf-8')];assert len(rows)==236
    answers=[json.loads(l) for name in ('row_answers.jsonl','remaining_row_answers.jsonl') for l in (OUT/name).open(encoding='utf-8')];assert len(answers)==36
    variation=[json.loads(l) for l in (OUT/'variation_answers.jsonl').open(encoding='utf-8')];assert len(variation)==10
    summary=json.loads((OUT/'summary.json').read_text(encoding='utf-8'));groups={}
    for r in answers:groups.setdefault((r['id'],r['arm']),[]).append(r)
    for r in rows:
        assert r['chars']<=12000 and len(r['entries'])<=r['limit'] and r['original_bodies_retained'] and not r['lost']
        if r['arm']=='H8':assert len(r['trace'])<=1
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    neg=[]
    for r in rows:
        for eid in r['new_negative_quotes']:
            n=int(eid.rsplit(':',1)[1])-1;neg.append({'id':r['id'],'arm':r['arm'],'evidence_id':eid,'annotation':cases[r['id']].hard_negatives[n]})
    (OUT/'budget_negative_audit.json').write_text(json.dumps(neg,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 新行库接入与预算诊断（2026-11-05）','','## 结论','',
      '新行库H8在相同8段/12000字符预算下补回023，严格完整引文命中39/44→40/44，没有丢失已有引文或新增标注负例原句。它改变了6个上下文，已全部进行每组3次的新回答复测，结果见下表。局部收益不等于已证明全局零退化。',
      '020另使用上一轮S_native上下文做固定输入5次对照，只检查重复性，未改生成提示词或算法，不能声称修复了遗漏。8/12/16段预算只作诊断；后两者不属于同预算改进。生产、Gold及已发布索引未改动。','',
      '## 固定实验','',
      '当前AI bundle_id=g000033-71ec3c52f3。H8使用2287条纠正表头与明确列/值/空值绑定的行记录：现有分词器BM25取10行，真实重算行视图Cross-Encoder及原辅助权重，全部硬条件匹配后最多补1行，保留基线原文和顺序。未重建完整向量索引。',
      'S8/S12/S16固定基线和此前A分支真实评分的来源页候选，不移除旧证据，只补空位。段数上限分别8、12、16，字数上限都为12000；没有把多页隐蔽算作一页。候选顺序和保护基线规则与H8不同，因此S组是独立预算诊断，不能当成H8的严格消融。','',
      '## 59题证据回放','',
      '| 分支 | 正式正证据命中（44题） | 平均完整引文覆盖 | 改变上下文（59题） | 丢失原引文 | 新增标注负例原句的题数 |',
      '|---|---:|---:|---:|---:|---:|']
    for arm in ('H8','S8','S12','S16'):
        s=summary[arm];lines.append(f"| {arm} | {s['before_hit']}→{s['after_hit']} | {s['before_coverage']:.1%}→{s['after_coverage']:.1%} | {s['changed']} | {len(s['losses'])} | {len(s['new_negative_quotes'])} |")
    lines+=['','新增标注负例原句表示上下文出现了该片段，不能直接推断模型采用了错误结论；其中023为同页其他报警行，042为相关但不足的背景。原标注和统计均保留，详见budget_negative_audit.json。S12/S16均未通过门槛，因此未开展其全量回答测试。','',
      '## H8全部变化上下文的36次新回答','',
      '| 题目 | 基线一致/3 | H8一致/3 |','|---|---:|---:|']
    for cid in sorted({r['id'] for r in answers}):
        vals=[sum((r.get('judgement') or {}).get('verdict')=='consistent' for r in groups[cid,arm]) for arm in ('baseline','candidate')]
        lines.append(f"| {cid}{'（pending）' if cid.startswith('rag-acc') else ''} | {vals[0]} | {vals[1]} |")
    lines+=['','023的新上下文仅增加CSDB行及原列绑定，没有DCF原句。表格父页ID式负例计分可能仍报同页命中，实际原句审计单列，不静默改Gold。其余上下文变化需要看逐次回答，不能仅凭原证据不丢失宣称生成不受影响。','',
      '## 020固定输入重复复核','',
      '| 输入 | 一致/5 | 其他裁判结果 |','|---|---:|---|']
    for arm in ('baseline','candidate'):
        rs=[r for r in variation if r['arm']==arm];counts=Counter((r.get('judgement') or {}).get('verdict','execution_error') for r in rs)
        lines.append(f"| {arm} | {counts['consistent']} | {dict(counts)} |")
    lines+=['','候选有一次缺少“排查系统故障并维护后重新定标”的明确结束步骤，核心检查项F2仍被判missing；上一轮候选3次也有1次相同边界遗漏。保留两轮结果，不改Gold，不以小样本断言统计显著或确定因果，但不能忽略重复出现的遗漏。该输入与本轮H8/020不同，不能混合归因。','',
      '## 满额问题定位','',
      '006与019必要页均通过原证据门槛；在保留基线后继续补候选的固定顺序中，它们分别排到第18和第20个位置，因此扩至12或16仍没有纳入。扩大数量已先引入其他负例片段。这是当前顺序与预算的共同结果，不证明任何扩大预算方法都无效，也不证明扩大到18/20后回答必然正确。',
      '因此不建议全局提高TopK。后续应检查候选之间的重复内容和必要条件覆盖，或者条件明确的按需补检索；不能只靠增加段数。','',
      '## 原固定集合、新增集合与待审回归分开','',
      '| 集合 | 基线 | H8 | S12 | S16 |','|---|---:|---:|---:|---:|']
    for label,fn in [('原18题固定集合，有效17题',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])<=18),('新增有效31题',lambda r:r['cohort']=='formal48' and int(r['id'].split('-')[-1])>18),('回归10题pending',lambda r:r['cohort']=='regression10_pending'),('024 pending',lambda r:r['cohort']=='pending024')]:
        cells=[]
        for arm in ('H8','S12','S16'):
            rs=[r for r in rows if r['arm']==arm and fn(r) and r['before']['gold_evidence_recall'] is not None]
            if not cells:cells.append(f"{sum(r['before']['gold_hit'] for r in rs)}/{len(rs)}")
            cells.append(f"{sum(r['after']['gold_hit'] for r in rs)}/{len(rs)}")
        lines.append('| '+label+' | '+' | '.join(cells)+' |')
    errors=sum(bool(r.get('error')) for r in answers+variation)
    lines+=['','## 采用建议与限制','',
      'H8是有证据支持的局部候选，先根据上述全部变化题回答结果判断是否存在回退，再讨论接入；不能把023单题的3次稳定改善当成普遍泛化。现有回归集已反复使用，不是独立盲测。',
      '020旧选材候选的遗漏未解决，保留实验状态；本轮没有实施新的生成补救。12/16预算方案不采用。资料格式仅原生PPTX简单表，复杂合并表、脚注和其他文件格式仍有范围缺口。',
      f'交付：236组证据回放、H8全部6个变化题的36次回答、020额外10次固定输入复核，共46次新回答及相应Gold裁判；执行错误{errors}次。所有输入/原资料/生产代码/配置/Gold/索引哈希核验通过。所有回答使用原生产提示词，没有把Gold标签或答案提供给回答模型。','']
    txt='\n'.join(lines);(OUT/'report.md').write_text(txt,encoding='utf-8');(ROOT/'docs/rag_row_integration_validation_20261105.md').write_text(txt,encoding='utf-8')
    body='<!doctype html><meta charset="utf-8"><title>新行库接入验证</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:10px;margin:8px}</style><pre>'+html.escape(txt)+'</pre>'
    for r in answers+variation:body+='<details><summary>'+html.escape(f"回答 {r['id']} {r['arm']} {r['repeat']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    for r in rows:body+='<details><summary>'+html.escape(f"证据 {r['id']} {r['arm']} gained={r['gained']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'delivery_verification.json').write_text(json.dumps({'evidence_rows':len(rows),'new_answers':len(answers+variation),'generation_errors':errors,'all_original_bodies_retained':all(r['original_bodies_retained'] for r in rows),'budget_checks_passed':True,'hashes_unchanged':True,'html_sections':body.count('<details>')},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(txt)
if __name__=='__main__':main()
