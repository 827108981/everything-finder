"""Summarize the fixed title-input experiment, including evidence lost to truncation."""
from __future__ import annotations
import json
from pathlib import Path
from statistics import median


def main():
    root=Path('reports/rerank_title_input_20261102_threads4')
    done=json.loads((root/'completion.json').read_text(encoding='utf-8'))
    assert done['count']==60
    rows=[json.loads(s) for s in (root/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    for cohort in ('original18','added32'):
        selected=[r for r in rows if r['cohort']=='development' and r['expected_behavior']!='refuse'
                  and ((int(r['id'][-3:])<=18)==(cohort=='original18'))]
        summary[cohort]={}
        for arm in ('body','title_body'):
            summary[cohort][arm]={str(k):{'n':len(selected),
                'hit':sum(r['arms'][arm]['scores'][str(k)]['gold_hit'] for r in selected),
                'coverage':sum(r['arms'][arm]['scores'][str(k)]['gold_evidence_recall'] for r in selected)/len(selected),
                'mrr':sum(r['arms'][arm]['scores'][str(k)]['reciprocal_rank'] for r in selected)/len(selected),
                'negatives':sum(r['arms'][arm]['scores'][str(k)]['hard_negative_found'] for r in selected)} for k in (1,3,8)}
    lost=[dict(id=r['id'],**q) for r in rows for q in r['single_chunk_quote_audit'] if q['body_visible'] and not q['title_body_visible']]
    timings={a:sum(r['arms'][a]['inference_seconds'] for r in rows) for a in ('body','title_body')}
    truncated={a:sum(t>512 for r in rows for t in r['arms'][a]['untruncated_pair_tokens'].values()) for a in timings}
    lines=['# 当前模型：标题加入重排输入的验证','','## 结论','',
        '本次不建议将“文档标题直接前置正文”加入当前模型的默认输入：按现有Gold，开发集前3条命中从41/46降到36/46；候选题首位命中增加，但前3条指定文件命中从10/10降到9/10；两组前8证据覆盖持平。已发现一处原可见完整引文因标题占用预算而不再完整可见。没有显示稳定的综合收益。',
        '', '这里的排名按现有指定Gold文件计分。候选001、010及部分开发题存在同正文替代来源，002有可解释核心区别的其他手册，因此部分名次变化不等于实际答案证据变化。不能把这些分数直接解读为答案准确率下降；它们也不足以支持上线。',
        '', '这只验证了现有bge-reranker-base在512 token预算下的标题前置格式。正文现有章节未改变，适用机型等结构化字段尚缺；不代表完整上下文输入、其他格式或其他模型一定无效。',
        '', '## 范围','',
        f"AI索引 `{protocol['bundle_id']}`；模型 `{protocol['model']}`，固定修订、CPU、batch=4、总输入512 token。60题每题20条冻结候选，两组各实际推理1200对。模型分排序和同分规则固定。",'',
        '仅正文组与标题＋正文组比较；标题直接来自已核验的文件名，最多64 token；正文中的现有章节、表头、步骤保持原样。索引没有独立适用机型、版本、章节字段，本次未推断补写。加入标题会占用同一个512 token预算。',
        '', '本实验只比较重排模型输入和完整排序，不含生产混合加分、名次保护、后续条件筛选或文档限额；不能据此宣称最终上下文或答案质量变化。10道候选仍为pending，等价来源尚未审核；指定Gold文件排名仅作诊断。全部问题以前均已看过，不是独立验收。',
        '', '## 标注证据排序','', '| 集合 | 前K条 | 仅正文命中 | 标题＋正文命中 | 仅正文覆盖 | 标题＋正文覆盖 | 仅正文MRR | 标题＋正文MRR | 指定反例次数（正文/标题） |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cohort,label in [('development','开发集46题'),('original18','原18题中的16题'),('added32','新增32题中的30题'),('holdout_pending','候选10题pending')]:
        for k in ('1','3','8'):
            a=summary[cohort]['body'][k];b=summary[cohort]['title_body'][k]
            lines.append(f"| {label} | {k} | {a['hit']}/{a['n']} | {b['hit']}/{b['n']} | {a['coverage']:.1%} | {b['coverage']:.1%} | {a['mrr']:.4f} | {b['mrr']:.4f} | {a['negatives']}/{b['negatives']} |")
    lines+=['', '仅正文基线已将候选内可计分正例排入前8条，故前8覆盖没有提升空间；本轮更有区分力的是前1/3排序、证据截断和强负例变化。MRR表示首条完整标注引文的排名，不表示整题全部必答事实已覆盖。',
        '', '007题指定表格由第2降至第7，但同机型流量调试资料仍在第1，不能称整题没有有效证据；002候选题由第3降至第4，前面的手册仍解释核心LIS功能。024题则明确发现标题挤掉了一处完整原文引文。',
        '', '执行记录：首次默认12线程试跑在两题后停止；正式两组均固定4线程、interop=1。正式运行于第23题后因会话中断恢复，保留原结果和冻结协议，恢复脚本哈希另存resume.json。没有因中间结果调整模型或输入。']
    lines+=['','## 截断与计算成本','',f"- 超过512 token的原始问题—输入对：仅正文{truncated['body']}/1200，标题＋正文{truncated['title_body']}/1200。",
        f"- 原来可见、加入标题后不再完整可见的单块完整Gold引文：{len(lost)}处。范围只包括在单个候选正文中能逐字定位完整引文的情形；跨块引文没有按此方式审计，不能当成无截断风险。",
        f"- 两组实际分词＋模型推理总时长：仅正文{timings['body']:.1f}秒，标题＋正文{timings['title_body']:.1f}秒。每题中位数{median(r['arms']['body']['inference_seconds'] for r in rows):.2f}/{median(r['arms']['title_body']['inference_seconds'] for r in rows):.2f}秒。已预热并交替组顺序；不含模型加载、索引核验和审计开销，属于本机离线测量。",
        '', '## 逐题变化','', '| 题号 | 首条指定证据排名（正文/标题） | 前8覆盖（正文/标题） | 新增前8反例 |', '|---|---:|---:|---|']
    changes=[]
    for r in rows:
        a=r['arms']['body']['scores'];b=r['arms']['title_body']['scores'];an=set(a['8']['hard_negative_ids']);bn=set(b['8']['hard_negative_ids'])
        if a['20']['first_gold_rank']!=b['20']['first_gold_rank'] or a['8']['gold_evidence_recall']!=b['8']['gold_evidence_recall'] or an!=bn:
            f=lambda x:'—' if x is None else f'{x:.1%}'
            lines.append(f"| {r['id']} | {a['20']['first_gold_rank'] or '—'}/{b['20']['first_gold_rank'] or '—'} | {f(a['8']['gold_evidence_recall'])}/{f(b['8']['gold_evidence_recall'])} | {','.join(sorted(bn-an)) or '—'} |")
            changes.append({'id':r['id'],'body':a,'title_body':b,'new_negatives':sorted(bn-an)})
    if lost:
        lines+=['','## 加标题后不可完整看到的引文','']
        for q in lost:lines.append(f"- {q['id']}：`{q['file']}`；引文：{q['quote'].replace(chr(10),' ')}")
    lines+=['','## 校验','',f"- 1200个索引正文及文件名逐条核实；仅正文模型分与上一轮冻结原始分最大差：{max(r['baseline_max_score_difference'] for r in rows):.8g}，60题原排序全部复现。",'- Gold哈希、审核状态和AI包完成时再次核对一致；未调用API，未改变生产算法或默认配置。',
        '',f"- [冻结协议]({(root/'protocol.json').resolve().as_posix()})",f"- [逐题结果]({(root/'cases.jsonl').resolve().as_posix()})"]
    Path('docs/rag_rerank_title_input_validation_20261102.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (root/'audit_summary.json').write_text(json.dumps({'lost_quotes':lost,'timings':timings,'truncated':truncated,'changes':changes},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summary,'lost_quote_count':len(lost),'timings':timings,'truncated':truncated},ensure_ascii=False))


if __name__=='__main__':main()
