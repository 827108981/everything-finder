"""Post-run analysis only. Never changes retrieval, labels or frozen choices."""
from tools.validate_vector_packaging_20261116 import *


def main():
    inputs()
    assert read('final_acceptance.json')['completed']
    base=lines(OUT/'W0_exact.jsonl')
    arms=['W0','W1','W2','W3','W4','W5','F10','F11','F10_count_matched']
    tables=[];changes={}
    for arm in arms:
        rows=lines(OUT/f'{arm}_exact.jsonl');m=metrics(rows)
        annpath=OUT/f'{arm}_ann.jsonl';ann=metrics(lines(annpath)) if annpath.exists() else None
        tables.append(dict(arm=arm,exact=m,ann=ann))
        changes[arm]=deltas(base,rows)
    selected=('rag-row-holdout-003','evidence-flow-independent-05','evidence-flow-independent-06',
              'rag-biz-006','rag-biz-019','rag-biz-023','rag-biz-028','rag-biz-050')
    def compact_native(arm):
        records=[]
        with (OUT/f'{arm}_native.jsonl').open(encoding='utf8') as source:
            for line in source:
                row=json.loads(line)
                records.append({k:row[k] for k in ('id','question','cohort','scores')})
                if row['id'] in selected:records[-1]['entries']=row.get('entries',[])
        return records
    native={a:compact_native(a) for a in ('W0','F10','F11')}
    findings=[]
    stages=('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')
    for cid in selected:
        record={'id':cid,'arms':{}}
        for arm in ('W0','F10'):
            r=next(r for r in native[arm] if r['id']==cid)
            record['question']=r['question']
            record['arms'][arm]={s:r['scores'].get(s) for s in stages}
            record['arms'][arm]['context_entries']=r.get('entries',[])
        findings.append(record)
    comparison=read('comparisons.json');gate=read('adoption_gates.json')
    # Native records do not contain vector cutoff_scores. Reusing metrics()
    # here would silently report Recall@k=0. Count actual stage quotes instead.
    native_summary={}
    for arm,records in native.items():
        native_summary[arm]={}
        for stage in stages:
            native_summary[arm][stage]={}
            for cohort in ('formal','diagnostic','source_holdout'):
                scores=[r['scores'][stage] for r in records if r['cohort']==cohort
                        and r['scores'][stage]['gold_evidence_total_declared']>0]
                native_summary[arm][stage][cohort]=dict(
                    positive_cases=len(scores),
                    complete=sum(s['gold_evidence_found']==s['gold_evidence_total_declared'] for s in scores),
                    mean_quote_recall=sum(s['gold_evidence_recall'] for s in scores)/max(1,len(scores)))
    transmission=[]
    for before,after in zip(native['W0'],native['F10']):
        assert before['id']==after['id']
        stage_differences=[]
        for stage in stages:
            b=before['scores'][stage];a=after['scores'][stage]
            if b['gold_evidence_found']!=a['gold_evidence_found']:
                stage_differences.append(dict(stage=stage,before=b['gold_evidence_found'],after=a['gold_evidence_found']))
        if stage_differences:
            transmission.append(dict(id=before['id'],question=before['question'],cohort=before['cohort'],changes=stage_differences))
    remaining=[]
    for row in native['F10']:
        context=row['scores']['context'];total=context['gold_evidence_total_declared']
        if not total or context['gold_evidence_found']==total:continue
        earliest='双路融合后仍未补齐'
        for previous,current in zip(stages[2:],stages[3:]):
            if row['scores'][previous]['gold_evidence_found']==total and row['scores'][current]['gold_evidence_found']<total:
                earliest=current;break
        remaining.append(dict(id=row['id'],question=row['question'],earliest_observed_loss=earliest,
             context_found=context['gold_evidence_found'],total=total))
    raw=[json.loads(x) for x in FIXTURE.read_text(encoding='utf8').splitlines()]
    groups={r['id']:r['annotation_provenance']['source_group'] for r in raw}
    holdbase=lines(OUT/'W0_ann_holdout.jsonl');holdafter=lines(OUT/'F10_ann_holdout.jsonl')
    hold=[]
    for b,a in zip(holdbase,holdafter):
        if a['id'] not in groups:continue
        bs=b['scores']['vector'];ns=a['scores']['vector']
        hold.append(dict(id=a['id'],source_group=groups[a['id']],question=a['question'],before=bs,after=ns,
            rank_changed=bs['first_gold_rank']!=ns['first_gold_rank']))
    write(OUT/'detailed_findings.json',dict(packaging=tables,exact_changes=changes,key_cases=findings,
          independent=hold,native_stage_summary=native_summary,transmission=transmission,remaining=remaining))
    title='包装与结构对照：完整结果解读'
    text='# '+title+'\n\n'
    text+='**采用决定：暂不采用新包装，也暂不将结构补充切入生产。** 新包装有证据丢失；结构补充有局部向量收益，但实际Context没有新增证据，独立触发验证不足。通用实现保留为显式可选能力，默认与生产索引不变。\n\n'
    text+='本轮全部实验执行完成，API 调用为 0；以下只评价检索和实际 Context，不宣称答案正确率。原包装和生产索引保持不变。\n\n'
    text+='## 1. 包装拆解与真实 ANN\n\n下表的证据覆盖是“返回父块内完整引文覆盖”，不是命中窗口本身覆盖，也不是实际入模覆盖。\n\n|方案|正式完整@50（精确）|诊断完整@50（精确）|正式MRR（精确）|正式/诊断完整@50（ANN）|\n|---|---:|---:|---:|---|\n'
    for t in tables:
        m=t['exact'];a=t['ann'];v=f'{a["formal"]["complete"]}/46、{a["diagnostic"]["complete"]}/40' if a else '未执行：精确对照已出现证据丢失，不满足进入条件'
        text+=f'|{t["arm"]}|{m["formal"]["complete"]}/46|{m["diagnostic"]["complete"]}/40|{m["formal"]["mrr"]:.4f}|{v}|\n'
    text+='\nW1只去文件名；W2只去位置；W3只保留正文标签；W4为纯正文；W5为有预算的可信背景。它们均未满足无新增证据丢失的门槛，不能替换W0。这个结论是当前模型与语料下的对照结果，不代表文件名或位置在所有RAG中必需。\n\n'
    text+='F10保留原包装并补原生表格行。F11因P*=W0与F10使用同一表示及ANN索引，所以结果相同，不能说组合有额外收益。W6未满足互补条件，按冻结协议不执行。\n\n'
    text+='## 2. 结构收益与损失\n\n'
    for arm in ('F10','F11'):
        v=comparison['vector'][arm];c=comparison['stages'][arm]['context']
        text+=f'{arm}：新增引文题 {v["gains"]}；丢失引文题 {v["losses"]}；新增已标注负例题 {v["new_negatives"]}。实际Context新增 {c["gains"]}；丢失 {c["losses"]}。\n\n'
    text+='“新增引文”包含部分补齐，只有全部引文齐全才计入完整题。第一条Gold的MRR不等于所有证据到齐名次。新增负例指整个Top50中出现的已标注负例，不直接等同于最终答案误用。\n\n'
    text+='结构行经过“表头：值”重排后，不一定包含原摊平正文中连续的Gold原句。因此best_encoded_window_scores是严格连续引文诊断，不能直接比较它的零值来宣称结构行缺乏事实；也不能用父块命中宣称命中行已独自支持全部答案。实际匹配的单元格和原文位置保存在representation_metadata；API前传递以实际included_text单独核验。通信比较题尤其可能由一个机型行召回整页，再由原页提供另一个机型的证据。\n\n'
    text+='等数量替换保留52,282个向量，F10补充为53,561个。等数量对照用于识别收益是否只是来自更多候选；替换会移走原窗口，不能直接当成无损发布方案。\n\n'
    text+='## 3. 独立来源验证\n\n'
    text+='20道新增题全部pending，按10个来源组单列。\n\n|题号|第一条完整引文名次：前→后|引文覆盖：前→后|\n|---|---|---|\n'
    for h in hold:
        b=h['before'];a=h['after']
        text+=f'|{h["id"]}|{b["first_gold_rank"]} → {a["first_gold_rank"]}|{b["gold_evidence_found"]}/{b["gold_evidence_total_declared"]} → {a["gold_evidence_found"]}/{a["gold_evidence_total_declared"]}|\n'
    text+='\nF10独立来源收益组：'+str(gate['F10']['independent_improved_source_groups'])+'。独立题不触发结构机制时只能说明保留行为，不能算结构有效性证据。\n\n'
    text+='## 4. 重点题在真实链路的传递\n\n|题号|方案|向量|融合|重排|条件|选段|实际Context|\n|---|---|---|---|---|---|---|---|\n'
    for r in findings:
        for arm in ('W0','F10'):
            cells=[]
            for s in ('vector','fusion','rerank','constraints','evidence','context'):
                m=r['arms'][arm][s]
                cells.append('未执行' if m is None else f'{m["gold_evidence_found"]}/{m["gold_evidence_total_declared"]}（首命中{m["first_gold_rank"]}）')
            text+='|'+r['id']+'|'+arm+'|'+'|'.join(cells)+'|\n'
    text+='\n表中为完整引文数量/本题声明引文数量，不是回答得分。实际Context仅按included_text匹配；逐条来源和完整轨迹见report.html和detailed_findings.json。\n\n'
    text+='### 实际Context完整引文覆盖\n\n|方案|正式正向46题|原诊断40题|新来源20题|\n|---|---:|---:|---:|\n'
    for arm in native:
        m=native_summary[arm]['context']
        text+=f'|{arm}|{m["formal"]["complete"]}/46|{m["diagnostic"]["complete"]}/40|{m["source_holdout"]["complete"]}/20|\n'
    text+='\n### 逐题传递变化（F10对W0）\n\n'
    for r in transmission:
        text+=r['id']+'：'+'；'.join(f'{x["stage"]} {x["before"]}→{x["after"]}' for x in r['changes'])+'。\n\n'
    text+='## 5. 采用边界与未解决项\n\n'
    text+='新包装：不采用。结构补充：向量层局部收益已证实；是否具备通用采用依据，以独立来源收益组、各层新增丢失及对象复核分别判断。生产未启用。\n\n'
    text+='原生PPT简单表格之外的PDF、图片、复杂合并单元格未修复。independent-05若通用原理仍未召回，不能宣称该题已完全解决。表格列并列不意味着同一行各列有业务关联；占位符不能当作有效答案。\n\n'
    text+='### 当前实际Context仍缺的Gold引文\n\n|题号|覆盖|最早可观测缺失环节|\n|---|---:|---|\n'
    for row in remaining:
        text+=f'|{row["id"]}|{row["context_found"]}/{row["total"]}|{row["earliest_observed_loss"]}|\n'
    text+='\n019中第13张场景页仍在，缺的是第14张原因、解决思路和40次以上灌注建议。第14张在去重候选第30位，未进入重排输出前20。不能把“DEV场景页命中”写成“原因和处理证据齐全”。这次没有修改该阶段。\n\n'
    text+='Top5变化已对21道题作技术复核，见new_top5_technical_applicability_review.json。新增候选确有同型号错误部件、不同机型、相关但不回答所问参数的片段；这些不能当作正向证据。已标注负例未增加不代表全部候选适用，候选不适用也不直接等于最终答案错误。本轮没有生成答案。技术复核不改变Gold状态。\n\n'
    text+='20道新来源题W0本来已20/20完整；只有1题的Gold父块存在本轮新增表格行。独立收益为0，既没有证明泛化有效，也不能用该批多数未触发的题反证表结构机制无用。下一轮最小验证应冻结能够触发原生表格机制的未调试来源，并加入错型号/错部件对照；本轮到此结束，不根据结果继续调词或换题。\n\n'
    text+='当前全部实测数据在同目录，完整流程和保护哈希见final_acceptance.json。\n'
    path=OUT/'detailed_conclusion.md';path.write_text(text,encoding='utf8')
    (ROOT/'docs/rag_vector_packaging_detailed_result_20261117.md').write_text(text,encoding='utf8')
    print(path)


if __name__=='__main__':main()
