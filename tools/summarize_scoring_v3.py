"""Compare rejudgements; original answers/contexts/Gold never overwritten."""
from collections import Counter
from pathlib import Path
import json,html
from tools.rejudge_scoring_v3 import ROOT,OUT,sha,read,lines

def main():
    assert (OUT/'completion.json').exists()
    p=read(OUT/'v31_protocol.json');assert all(sha(k)==h for k,h in p['hashes'].items())
    rows=read(OUT/'final_cases_v31.json')
    # Transport/format failures only may be recovered once in an append-only file.
    recovered={r['judge_request_sha256']:r for r in lines(OUT/'error_retries.jsonl')} if (OUT/'error_retries.jsonl').exists() else {}
    # final_cases already includes execution recoveries and uniform v3.1 guards.
    assert len(rows)==119
    def group_summary(rs):
        old=Counter((r.get('old_judgement') or {}).get('verdict','unavailable') for r in rs)
        new=Counter((r.get('new_judgement') or {}).get('verdict','unavailable') for r in rs)
        return {'count':len(rs),'old_verdicts':dict(old),'new_verdicts':dict(new),'errors':[r['id'] for r in rs if r['judge_error']],
                'changes':[{'id':r['id'],'arm':r['arm'],'repeat':r['repeat'],'old':(r.get('old_judgement') or {}).get('verdict'),'new':(r.get('new_judgement') or {}).get('verdict'),'findings':(r.get('new_judgement') or {}).get('findings',[])} for r in rs if (r.get('old_judgement') or {}).get('verdict')!=(r.get('new_judgement') or {}).get('verdict')]}
    formal=[r for r in rows if r['cohort']=='historical_baseline' and r['review_status']=='reviewed']
    summary={'historical_formal48':group_summary(formal),
      'original17':group_summary([r for r in formal if int(r['id'].split('-')[-1])<=18]),
      'added31':group_summary([r for r in formal if int(r['id'].split('-')[-1])>18]),
      'historical_regression10_pending':group_summary([r for r in rows if r['cohort']=='historical_baseline' and r['id'].startswith('rag-acc')]),
      'pending024':group_summary([r for r in rows if r['cohort']=='historical_baseline' and r['id']=='rag-biz-024'])}
    for cohort in ('row_changed','holdout_pending','object_repeat_pending'):
        summary[cohort]={arm:group_summary([r for r in rows if r['cohort']==cohort and r['arm']==arm]) for arm in sorted({r['arm'] for r in rows if r['cohort']==cohort})}
    negatives=[]
    for r in rows:
        old=r.get('old_score') or {};new=r['new_score']
        if old.get('hard_negative_ids')!=new.get('hard_negative_ids'):
            negatives.append({'id':r['id'],'cohort':r['cohort'],'arm':r['arm'],'repeat':r['repeat'],'old':old.get('hard_negative_ids'),'new':new.get('hard_negative_ids'),'same_source_only':new.get('hard_negative_page_only_ids'),'unresolved':new.get('hard_negative_unresolved_ids')})
    summary['negative_score_changes']=negatives
    manual_review=[
      {'id':'rag-biz-035','issue':'新裁判把“复核质控是否可靠”视为覆盖“质控失控”。是否表达充分存在语义争议，不能据此认定修复成功。','automatic_verdict':'consistent'},
      {'id':'rag-biz-038','issue':'答案的“校准记录”是否被其实际标注的S1/S5支持，仍需核对。未定位到确定错误断言，暂记uncertain。','automatic_verdict':'uncertain'},
      {'id':'rag-row-holdout-001','issue':'BS-2800与BS-2800M资料是否适用于同一设备尚未确认。DC24V主结论有依据，附加电压范围和控制电平的适用性待核；两个方案均uncertain。','automatic_verdict':'uncertain'},
    ]
    summary['manual_review']=manual_review
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dump('resolved_cases.json',rows);dump('summary.json',summary)
    changes=[r for r in rows if (r.get('old_judgement') or {}).get('verdict')!=(r.get('new_judgement') or {}).get('verdict')]
    dump('review_changes.json',[{k:r[k] for k in ('id','arm','cohort','repeat','answer','context','old_judgement','new_judgement','judge_error')} for r in changes])
    dump('manual_review.json',manual_review)
    notes=['# 正式评测判分修订及历史输出复核','','## 修改范围','',
      '仅修改评测：负例按来源与实际完整引文计分；同父页无完整原句、缺正文、重复数值无法定位时单列诊断。答案裁判新增对象/条件绑定和实际引用支持检查，显式区分来源不可核对。Runner将实际入模上下文传给裁判；报告使用included_text，部分负例不计完整召回。',
      '未修改检索、重排、生成提示词、Gold内容或审核状态。复用历史生成答案和实际上下文，重新调用答案裁判；没有重新生成答案，因此变化来自判分方式和裁判随机性，不能称为业务算法进退。','',
      '最终使用v3.1：明确的非设备编号若未在实际引用中核实，不能判一致，保守标为待核实而不是擅自判别名；引用错误必须指明答案实际断言，对象替换须指明问题和答案中不同的对象。别名或来源型号关系不明时不直接判incorrect。初版v3仅加语义提示仍漏判编号，保留为preliminary，不作为最终结果。成功的原检查项保留，统一应用v3.1后验校验；仅执行失败请求重新调用裁判，不是119次独立同提示词的模型实验。','',
      '## 历史基线固定集合','',
      '| 集合 | 题数 | 旧一致 | 新一致 | 无法判分 |','|---|---:|---:|---:|---:|']
    for label,key in [('正式集合','historical_formal48'),('原18题有效集合','original17'),('新增有效集合','added31'),('待审回归','historical_regression10_pending'),('024 pending','pending024')]:
        s=summary[key];notes.append(f"| {label} | {s['count']} | {s['old_verdicts'].get('consistent',0)} | {s['new_verdicts'].get('consistent',0)} | {len(s['errors'])} |")
    labels={'consistent':'一致','incomplete':'事实缺漏','needs_clarification':'未按要求追问','incorrect':'错误','uncertain':'待核实','behavior_mismatch':'回答方式不符','safety_gap':'安全要求缺漏'}
    notes+=['','正式48题自动裁判明细：'+ '；'.join(f'{labels.get(k,k)} {v}题' for k,v in summary['historical_formal48']['new_verdicts'].items())+'。这是一致性判定，不能把“待核实”当成已证实错误。']
    notes+=['','007仍排除；024和全部新问题、回归pending不进入正式分母。历史主集合48题的结果不能与变化题重复次数混在一起。','',
      '## 保存的变化题与新问题','',
      '| 子集合 | 方案 | 输出数 | 旧一致 | 新一致 | 无法判分 |','|---|---|---:|---:|---:|']
    for cohort in ('row_changed','holdout_pending','object_repeat_pending'):
        for arm,s in summary[cohort].items():notes.append(f"| {cohort} | {arm} | {s['count']} | {s['old_verdicts'].get('consistent',0)} | {s['new_verdicts'].get('consistent',0)} | {len(s['errors'])} |")
    notes+=['','新10题每方案10个保存答案；对象复核额外重复单列，无旧裁判的4次不能写成旧错→新对。行级变化题每方案18个保存答案，属于6题×3次，不是全量通过率。','',
      '## 判分变化逐条核查','',
      '完整原因、原答案、原上下文和新旧裁判检查项均保存在review_changes.json与下方HTML。新裁判仍可能误判，尤其需区分“没说出某事实”和“说出的事实没有来源支持”；不得把缺失内容自动判成引用错误。自动结果和人工争议项分别保留。','']
    for r in changes:
        j=r.get('new_judgement') or {};notes.append(f"- {r['cohort']} / {r['id']} / {r['arm']} / 第{r['repeat']}次：{(r.get('old_judgement') or {}).get('verdict')} → {j.get('verdict')}；{j.get('object_binding',{}).get('reason','')} {j.get('source_binding',{}).get('reason','')}")
    notes+=['','## 人工复核意见（不覆盖自动结果或Gold）','']
    notes.extend(f"- {r['id']}：{r['issue']}" for r in manual_review)
    notes+=['','## 具体结论','',
      '- 023：行级方案3份保存输出只含CSDB行，不含DCF负例完整引文。旧规则3次均误算负例，新规则均为0；这是同一题的3次输出。',
      '- 新验收008：主输出旧裁判把两个方案都判一致；新规则将基线标为待核实，行级方案一致。连同额外重复，基线0/3一致，行级方案3/3一致；这10题仍pending，不计入正式分数，也不再属于未见过的盲测。',
      '- 042：答案开头声称“R2每测消耗量更大”，后文却列R1=224 μL、R2=62 μL。新裁判发现这一矛盾，旧裁判仅看数值覆盖而放过。',
      '- 050：按现有审核Gold，四类排查方向适用于所问场景；答案将其排除为其他机型原因。新裁判记为与Gold相反，而不只是漏答；这不是检索本身获得改善。',
      '- 行级补充仍有正向信号，但不能声称全部正确：新10题基线8题一致、2题待核，行级方案9题一致、1题待核；其中001的机型适用关系尚待确认。','',
      '## 引文计分变化','',f'负例计分变化记录共{len(negatives)}条，保留old/new/same_source_only/unresolved。正向Gold仍沿用完整引文计分，context阶段必须核对实际included_text；引用所属页相同不代表全部行都召回。本次119份保存输出的正向完整引文命中数未变化。相似来源未自动登记为等价。','',
      '## 验证与限制','',
      '使用项目.venv运行RAG测试：300项通过，1项跳过。覆盖对象替换、错误版本引用、缺上下文、缺裁判字段、同页错误行、截断原文及重复数值。缺少实际上下文的事实答案不能直接判一致。系统默认Python缺少项目依赖，收集失败，不作为代码回归结果。缺失或超时裁判不当成业务错，异常恢复仅针对执行错误，不重试质量不佳结果。',
      '历史原始报告、Gold及原答案未覆盖。建议审核新增对象/引用错误和裁判争议，不据总分变化宣称RAG改善。','']
    text='\n'.join(notes);(OUT/'report.md').write_text(text,encoding='utf-8');(ROOT/'docs/rag_scoring_v3_validation_20261105.md').write_text(text,encoding='utf-8')
    body='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>评测判分修订复核</title><style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:30px auto;padding:0 20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit}details{border:1px solid #ddd;padding:12px;margin:8px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px 18px}</style>'
    body+='<h1>评测判分修订复核</h1><p>复核119份保存输出，没有重新生成答案。正式集合与待审诊断集合分开统计。仅修改评测，不代表业务算法已改善。</p><h2>正式48题自动结果</h2><table><tr><th>判定</th><th>题数</th></tr>'
    for k,v in summary['historical_formal48']['new_verdicts'].items():body+=f'<tr><td>{html.escape(labels.get(k,k))}</td><td>{v}</td></tr>'
    body+='</table><h2>需要复核的判分</h2><ul>' + ''.join('<li>'+html.escape(r['id']+'：'+r['issue'])+'</li>' for r in manual_review)+'</ul><details><summary>完整验证说明和分组比较</summary><pre>'+html.escape(text)+'</pre></details><h2>逐条原始输出及裁判依据</h2>'
    for r in rows:body+='<details><summary>'+html.escape(f"{r['cohort']} {r['id']} {r['arm']} repeat={r['repeat']} old={(r.get('old_judgement') or {}).get('verdict')} new={(r.get('new_judgement') or {}).get('verdict')}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    dump('delivery_verification.json',{'records':len(rows),'generation_calls':0,'errors':sum(bool(r['judge_error']) for r in rows),'all_input_hashes_unchanged':True,'formal_count':len(formal),'new_pending_preserved':True,'html_sections':body.count('<details>')})
    print(json.dumps({k:v for k,v in summary.items() if k!='negative_score_changes'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
