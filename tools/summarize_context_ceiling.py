"""Render diagnostic context-ceiling replay, never change evaluation labels."""
from collections import Counter
import html
from tools.validate_context_ceiling import *

def main():
    prepared=read(OUT/'prepared.json'); results=lines(OUT/'results.jsonl'); audits=read(OUT/'source_audit.json')
    retry_path=OUT/'execution_retries.jsonl'
    retries=lines(retry_path) if retry_path.exists() else []
    retry_map={(r['id'],r['arm'],r['repeat']):r for r in retries}
    results=[retry_map.get((r['id'],r['arm'],r['repeat']),r) for r in results]
    (OUT/'resolved_results.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    report=['# 完整资料对照验证（诊断实验）','',
      '取两轮历史基线失败题并集：14道正式题、2道pending回归题。005已获批准的标注修订单独排除。每题两种资料各生成3次，并分别进行Gold裁判。',
      '当前资料使用已经冻结的生产检索结果；模型、配置、生产代码和索引哈希与该结果对应版本一致。本次不重跑本地召回。补全资料由已核实的原文所在索引整页置前，再尽量保留原资料。',
      '**这不是正式全量成绩，也不是候选算法收益或理论性能上限。补全组同时改变顺序、来源展示和部分干扰资料，结果只能用来诊断资料与回答环节的改善空间。三个重复不足以证明统计显著性。**','',
      '|题目|行为|当前通过/3|补全通过/3|新增片段|移除片段|原文引文当前完整存在|',
      '|---|---|---:|---:|---:|---:|---|']
    table=[]
    for p in prepared:
        arms={a:[r for r in results if r['id']==p['id'] and r['arm']==a] for a in ('current','source_completed')}
        counts={a:sum((r.get('gold_judgement')or{}).get('verdict')=='consistent' for r in rs) for a,rs in arms.items()}
        aa=[a for a in audits if a['id']==p['id']]
        row={'id':p['id'],'cohort':p['cohort'],'expected_behavior':p['expected_behavior'],'counts':counts,
          'verdicts':{a:[(r.get('gold_judgement')or{}).get('verdict','error') for r in sorted(rs,key=lambda r:r['repeat'])] for a,rs in arms.items()},
          'current_exact_quotes':sum(a['current_quote_present'] for a in aa),'total_quotes':len(aa),
          'added':len(p['added_ids']),'removed':len(p['removed_original_ids']),'same_context':p['same_context']}
        table.append(row)
        report.append(f"|{p['id']}|{p['expected_behavior']}|{counts['current']}|{counts['source_completed']}|{row['added']}|{row['removed']}|{row['current_exact_quotes']}/{row['total_quotes']}|")
    groups={}
    for group,pred in [('formal14',lambda r:r['cohort']=='formal'),('original18_subset3',lambda r:r['cohort']=='formal' and int(r['id'][-3:])<=18),
                       ('added_subset11',lambda r:r['cohort']=='formal' and int(r['id'][-3:])>18),('regression_pending2',lambda r:r['cohort']!='formal')]:
        groups[group]={}
        for arm in ('current','source_completed'):
            rs=[r for r in results if pred(r) and r['arm']==arm]
            groups[group][arm]={'attempts':len(rs),'verdicts':dict(Counter((r.get('gold_judgement')or{}).get('verdict','error') for r in rs))}
    summary={'cases':table,'groups':groups,'errors':sum(bool(r['error']) for r in results),
       'execution_recoveries':len(retries),
       'citation_issues':sum(bool(r.get('citation_check') and not r['citation_check']['structurally_valid']) for r in results)}
    dump(OUT/'summary.json',summary)
    report+=['','## 重复验证结论','']
    stable_gains=[r['id'] for r in table if r['cohort']=='formal' and r['counts']['current']==0 and r['counts']['source_completed']==3]
    stable_losses=[r['id'] for r in table if r['cohort']=='formal' and r['counts']['current']==3 and r['counts']['source_completed']==0]
    report+=['- 当前0/3、补全3/3：'+('、'.join(stable_gains)or'无')+'。',
      '- 当前3/3、补全0/3：'+('、'.join(stable_losses)or'无')+'。',
      '- 资料变化同时涉及内容、顺序和来源展示，不能把这些差异归因于某个未测试的检索修复。',
      '- 同一上下文对照项见same_context；这些题的判定变化不属于检索收益。','']
    report+=['','## 逐题原因复查','']
    notes=read(OUT/'manual_diagnostic_notes.json')
    for row in table:
        note=notes['cases'][row['id']]
        report+=['- **'+row['id']+'**：'+note['basis']]
    report+=['','## 已确认的资料丢失位置','',
      '|题目|冻结记录中的丢失位置|含义|','|---|---|---|',
      '|006|正确PDF页重排第18，最终TopK截断|部分操作事实由其他来源保留，但空瓶警告缺失|',
      '|019|正确步骤页融合第60，候选保留30时丢失|场景标题与后页处理步骤未关联|',
      '|023|正确表格融合第8、重排第11，通过条件检查；最终因同文件数量上限排除|已找到正确CSDB行，最终未提供给回答模型|',
      '|040|所有项目异常对应的第16张在词法/向量Top50均未定位|问题第二个分支缺证据|',
      '|050|正确PDF页词法第39，未进入融合前60输出|后续重排无法处理未进入候选的资料|',
      '', '上述排名取既有基线冻结记录；本轮只重跑生成与裁判。完整阶段数据见historical_loss_stages.json。']
    report+=['','### Gold之外的额外主张','',json.dumps(notes.get('extra_claim_review',[]),ensure_ascii=False,indent=2),'',
      '## 固定分组结果','', '```json',json.dumps(groups,ensure_ascii=False,indent=2),'```','',
      '## 核验范围与限制','',
      '- 24条正向引文均在原文件对应页和本次索引正文找到；补全上下文全部包含。原始路径、页码、整页文字、文件哈希见source_audit.json。',
      '- 019额外保留前页真实业务场景，避免把无标题原因页当作通用步骤。',
      '- 045拒答题没有伪造正向证据，两组上下文完全相同，只观察生成/裁判波动。',
      '- 引文未按字面命中，不等于必要事实不存在；等价来源、分散表达需语义复查。',
      '- Gold裁判与回答使用同一模型，自动一致不等于业务人员确认正确；需检查多余主张。',
      '- 未更改生产配置、索引、Gold内容和review_status，不把本诊断结果写入正式全量报告。','',
      '## 逐题答案与判定','']
    for p in prepared:
        report+=['### '+p['id'], '', p['question'],'']
        for r in sorted([r for r in results if r['id']==p['id']],key=lambda r:(r['repeat'],r['arm'])):
            j=r.get('gold_judgement')or{}
            report+=['**'+r['arm']+' / '+str(r['repeat'])+' / '+str(j.get('verdict','error'))+'**','',
               (r.get('answer')or{}).get('text',str(r['error'])),'',
               '判定：'+str(j.get('behavior_reason','')),'']
            for q in j.get('rubrics',[]):
                if q['status'] in ('missing','contradicted','present','uncertain'):
                    report+=['- '+q['id']+'（'+q['status']+'）：'+q['gold_text']+'；'+q['reason']]
            report+=['']
    md='\n'.join(report)+'\n';(OUT/'report.md').write_text(md,encoding='utf-8')
    esc=lambda v:html.escape(str(v))
    body='<h1>完整资料对照验证</h1><p>14道正式题、2道pending回归题，两种资料各回答3次。只用于诊断，不是正式全量成绩或候选算法收益。</p>'
    body+='<p>补全组改变内容、排序和部分来源展示，不是理论性能上限。裁判与回答使用同一模型，评分通过仍需核查额外主张。</p>'
    body+='<p><strong>结论：关键资料缺失有实际影响，追问与部分回答仍需独立解决。</strong> 完整结论见 <a href="conclusion.md">结论说明</a>。</p>'
    body+='<p>14道正式题各运行3次：当前资料通过 '+str(groups['formal14']['current']['verdicts'].get('consistent',0))+'/42，补全资料通过 '+str(groups['formal14']['source_completed']['verdicts'].get('consistent',0))+'/42。此数字不是50题总体通过率。</p>'
    body+='<p>稳定从0/3变3/3：'+esc('、'.join(stable_gains))+'。稳定从3/3变0/3：'+esc('、'.join(stable_losses))+'。014、021、049两组均未完成追问。</p>'
    body+='<table><tr><th>题号</th><th>行为</th><th>当前通过/3</th><th>补全通过/3</th><th>新增/移除片段</th></tr>'
    for p in table:body+=f'<tr><td>{esc(p["id"])}</td><td>{esc(p["expected_behavior"])}</td><td>{p["counts"]["current"]}</td><td>{p["counts"]["source_completed"]}</td><td>{p["added"]}/{p["removed"]}</td></tr>'
    body+='</table><h2>固定分组</h2><pre>'+esc(json.dumps(groups,ensure_ascii=False,indent=2))+'</pre>'
    body+='<p>24条引文均在原资料和当前索引对应页核验；详见 <a href="source_audit.json">引文核验</a>、<a href="manual_diagnostic_notes.json">原因与争议复查</a>、<a href="historical_loss_stages.json">资料丢失阶段</a>。</p>'
    body+='<h2>额外主张待核实</h2><pre>'+esc(json.dumps(notes.get('extra_claim_review',[]),ensure_ascii=False,indent=2))+'</pre>'
    for p in prepared:
        body+='<details><summary>'+esc(p['id'])+'：'+esc(p['question'])+'</summary><p>'+esc(notes['cases'][p['id']]['basis'])+'</p>'
        for r in sorted([r for r in results if r['id']==p['id']],key=lambda r:(r['repeat'],r['arm'])):
            j=r.get('gold_judgement')or{}
            body+='<h3>'+esc(r['arm'])+' / 第'+str(r['repeat'])+'次 / '+esc(j.get('verdict','error'))+'</h3><pre>'+esc((r.get('answer')or{}).get('text',str(r['error'])))+'</pre>'
            body+='<p>'+esc(j.get('behavior_reason',''))+'</p>'
            for q in j.get('rubrics',[]):
                if q['status'] in ('missing','contradicted','present','uncertain'):body+='<p>'+esc(q['id']+' '+q['status']+'：'+q['gold_text']+'；'+q['reason'])+'</p>'
        body+='</details>'
    (OUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>完整资料对照验证</title><style>body{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 24px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap;background:#f6f8fa;padding:16px}h3{border-top:2px solid #ddd;padding-top:24px}</style>'+body,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
