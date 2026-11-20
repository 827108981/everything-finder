"""Paired correctness report, preserving pending cohorts and no latency claims."""
import json,html
from pathlib import Path
from collections import Counter,defaultdict
from tools.prepare_row_ab_confirm import ROOT,OUT,sha
COHORT={'formal48':'正式有效48题','regression10_pending':'历史回归10题（待审）','used_holdout10_pending':'之前行级验收10题（已参与分析）','new_holdout10_pending':'本次独立问题10题（待审）','original17':'原18题有效17题（子集）','added31':'原新增有效31题（子集）','formal_without_disputed035':'正式集合排除035（敏感性子集）'}
LABEL={'consistent':'一致','incomplete':'事实缺漏','uncertain':'待核实','incorrect':'错误','needs_clarification':'未追问','behavior_mismatch':'回答方式不符','safety_gap':'安全要求缺漏','unavailable':'无法判分'}
def main():
 rows=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf8').splitlines()]
 raw_error_count=sum(bool(r.get('error')) for r in rows)
 recovery_path=OUT/'transport_recovery/cases.jsonl'
 recovered={(r['id'],r['arm']):r for r in [json.loads(l) for l in recovery_path.read_text(encoding='utf8').splitlines()] if not r.get('error')} if recovery_path.exists() else {}
 rows=[{**recovered[(r['id'],r['arm'])],'execution_recovery_of':r['error']} if r.get('error') and (r['id'],r['arm']) in recovered else r for r in rows]
 (OUT/'resolved_cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 def verdict(r):return (r.get('judgement') or {}).get('verdict','unavailable')
 pairs=defaultdict(dict)
 for r in rows:pairs[r['id']][r['arm']]=r
 audits=[]
 for cid,rs in pairs.items():
  if len(rs)!=2:continue
  a,b=rs['baseline'],rs['H8'];va,vb=verdict(a),verdict(b)
  ctxdiff=(a.get('context') or '')!=(b.get('context') or '')
  source_same=(a.get('context') or '')==b.get('extra',{}).get('original_context',b.get('context') or '')
  ga=set(a.get('score',{}).get('gold_completion_ranks',{}));gb=set(b.get('score',{}).get('gold_completion_ranks',{}))
  na=set(a.get('score',{}).get('hard_negative_ids',[]));nb=set(b.get('score',{}).get('hard_negative_ids',[]))
  category='一致性未变化'
  if va!='consistent' and vb=='consistent':category='观察到改善' if ctxdiff else '上下文相同但判定变好'
  if va=='consistent' and vb!='consistent':category='疑似退步' if ctxdiff else '上下文相同但判定变差'
  audits.append({'id':cid,'cohort':a['cohort'],'question':a['question'],'A':va,'B':vb,'context_changed':ctxdiff,'native_context_equal':source_same,'gold_gains':sorted(gb-ga),'gold_losses':sorted(ga-gb),'new_marked_negatives':sorted(nb-na),'original_bodies_retained':b.get('extra',{}).get('original_retained',True),'category':category})
 summary={}
 for cohort in sorted({r['cohort'] for r in rows}):
  rr=[r for r in rows if r['cohort']==cohort];pp=[p for p in audits if p['cohort']==cohort]
  summary[cohort]={'rows':len(rr),'A':dict(Counter(verdict(r) for r in rr if r['arm']=='baseline')),'B':dict(Counter(verdict(r) for r in rr if r['arm']=='H8')),'paired':len(pp),'changed_contexts':sum(p['context_changed'] for p in pp),'categories':dict(Counter(p['category'] for p in pp)),'changed_gains':[p['id'] for p in pp if p['category']=='观察到改善'],'changed_regressions':[p['id'] for p in pp if p['category']=='疑似退步']}
 for name,predicate in [('original17',lambda n:n<=18),('added31',lambda n:n>18),('formal_without_disputed035',lambda n:n!=35)]:
  rr=[r for r in rows if r['cohort']=='formal48' and predicate(int(r['id'].rsplit('-',1)[1]))]
  summary[name]={'A':dict(Counter(verdict(r) for r in rr if r['arm']=='baseline')),'B':dict(Counter(verdict(r) for r in rr if r['arm']=='H8')),'paired':len([p for p in audits if p['cohort']=='formal48' and predicate(int(p['id'].rsplit('-',1)[1]))]),'changed_contexts':sum(p['context_changed'] for p in audits if p['cohort']=='formal48' and predicate(int(p['id'].rsplit('-',1)[1])))}
 def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 dump('summary.json',summary);dump('paired_audit.json',audits)
 rep=[json.loads(l) for l in (OUT/'repeats.jsonl').read_text(encoding='utf8').splitlines()] if (OUT/'repeats.jsonl').exists() else []
 notes=['# 冻结行级补充方案：完整问答对照验证','','本次实际运行AnswerService.answer；A当前正式方案，B只在上下文组装处增加冻结行级方案。当前配置语义追问和答案二次复核均关闭；保持该配置，不代表验证了这两项功能。两组均执行最终引用校验。模型gpt-5.6-sol，AI索引g000033-71ec3c52f3。','','| 集合 | 配对题数 | A一致 | B一致 | 上下文改变题数 |','|---|---:|---:|---:|---:|']
 for c,s in summary.items():notes.append(f"| {COHORT.get(c,c)} | {s['paired']} | {s['A'].get('consistent',0)} | {s['B'].get('consistent',0)} | {s['changed_contexts']} |")
 notes+=['','只有正式48题进入正式统计；原18题中007排除、024待审。三批各10题均pending，分别呈现。新10题在查看问答前由原资料标注并冻结；资料及行库已使用过，且以表格为主，不能代表全部RAG业务或独立业务盲测。','','上下文相同却答案/裁判变动，不能归因于行级补充。上下文改变题也须结合实际引文、对象和重复结果判断，不仅看总数。','','## 逐题配对','','| 题号 | A | B | 上下文变化 | 判断 | 正向增/失 | 新负例 |','|---|---|---|---|---|---|---|']
 for p in audits:notes.append(f"| {p['id']} | {LABEL.get(p['A'],p['A'])} | {LABEL.get(p['B'],p['B'])} | {p['context_changed']} | {p['category']} | {','.join(p['gold_gains']) or '无'} / {','.join(p['gold_losses']) or '无'} | {','.join(p['new_marked_negatives']) or '无'} |")
 notes+=['','## 判分争议','','035保留语义争议；038核心事实有依据但校准记录引文指向存在缺口；旧验收001附加参数的机型适用性待核。未改Gold或审核状态。详见dispute_review.json。','','## 重复验证','','重复只用于预先规定的上下文变化且一致性改变题、错误或引文退步题；两组各固定额外2次并全部保留，不挑最好结果、不替换主实验结果。重复固定各组本次实际上下文，只重新生成答案和判分，用来检查回答波动；不属于重复执行整条检索链路。']
 for cid in sorted({r['id'] for r in rep}):
  notes.append('- '+cid+'：'+ '；'.join(arm+' '+str(dict(Counter(verdict(r) for r in rows+rep if r['id']==cid and r['arm']==arm))) for arm in ('baseline','H8')))
 notes+=['','## 运行完整性','',f"主实验输出{len(rows)}份，配对{len(audits)}题；原始执行异常{raw_error_count}份，恢复后未解决异常{sum(bool(r.get('error')) for r in rows)}份；额外重复{len(rep)}份。仅网络/超时等执行失败可补跑，成功但质量较差的答案不重试。正文、检查项、引用和原始判断分别留档，历史输出未覆盖。"]
 if (OUT/'live_judge_audit.json').exists():
  notes+=['','## 本轮发现的裁判争议','']
  notes.extend('- '+r['id']+'：'+r['finding']+' '+r['effect'] for r in json.loads((OUT/'live_judge_audit.json').read_text(encoding='utf8')))
 if (OUT/'conclusions.md').exists():notes[2:2]=(OUT/'conclusions.md').read_text(encoding='utf8').splitlines()+['']
 text='\n'.join(notes)+'\n';(OUT/'report.md').write_text(text,encoding='utf8')
 body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>行级补充完整对照</title><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:30px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit}details{border:1px solid #ddd;margin:10px;padding:10px}</style><h1>行级补充完整对照</h1><pre>'+html.escape(text)+'</pre>'
 body+='<h2>分组结果（自动裁判）</h2><table style="border-collapse:collapse"><tr><th>集合</th><th>配对题数</th><th>A一致</th><th>B一致</th><th>上下文变化</th></tr>'
 for c,v in summary.items():body+='<tr>'+''.join('<td style="border:1px solid #ddd;padding:8px">'+html.escape(str(x))+'</td>' for x in [COHORT.get(c,c),v['paired'],v['A'].get('consistent',0),v['B'].get('consistent',0),v['changed_contexts']])+'</tr>'
 body+='</table><h2>逐题答案与实际引用</h2>'
 for p in audits:
  body+='<details><summary>'+html.escape(p['id']+' '+p['question']+' | '+p['A']+' → '+p['B'])+'</summary>'
  for arm in ('baseline','H8'):
   r=pairs[p['id']][arm];body+='<h3>'+arm+'</h3><pre>'+html.escape(r.get('answer',{}).get('text','无答案'))+'</pre><details><summary>判分与实际上下文</summary><pre>'+html.escape(json.dumps({'judgement':r.get('judgement'),'context':r.get('context'),'score':r.get('score'),'error':r.get('error')},ensure_ascii=False,indent=2))+'</pre></details>'
  body+='</details>'
 (OUT/'report.html').write_text(body,encoding='utf8')
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
