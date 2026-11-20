from pathlib import Path
from collections import Counter
import json,html
from tools.validate_material_steps import ROOT,sha
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset,GoldDatasetError
OUT=ROOT/'reports/rag_row_holdout_20261105'
def main():
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in freeze['hashes'].items())
    assert (OUT/'correctness_completion.json').exists() and (OUT/'answer_completion.json').exists() and (OUT/'scoring_completion.json').exists()
    rows=[json.loads(l) for l in (OUT/'timing_cases.jsonl').open(encoding='utf-8') if json.loads(l)['repeat']==1];assert len(rows)==20
    records={(r['id'],r['arm']):r for r in rows};answers=[json.loads(l) for l in (OUT/'answers.jsonl').open(encoding='utf-8')];assert len(answers)==20
    ans={(a['id'],a['arm']):a for a in answers};cases=load_gold_dataset(ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json',reviewed_only=False).cases
    data=json.loads((ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json').read_text(encoding='utf-8'))['cases'];assert data==[json.loads(l) for l in (ROOT/'tests/rag/fixtures/rag_row_holdout_v1.jsonl').open(encoding='utf-8')]
    assert all(c.review_status=='pending' for c in cases)
    try:load_gold_dataset(ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json')
    except GoldDatasetError:formal_rejected=True
    else:formal_rejected=False
    assert formal_rejected
    source=json.loads((OUT/'source_audit.json').read_text(encoding='utf-8'));assert all(a['original_verified'] and a['index_verified'] and sha(a['file'])==a['file_sha256'] for a in source)
    review=[]
    for c in cases:
        review.append({'id':c.case_id,'status':'pending','baseline_source_review':'object_binding_error' if c.case_id.endswith('008') else 'supported','H8_source_review':'supported','reason':'原文第12张MFH1属于分析仪后端轨道，第13张IMFH1属于ISE模块后端轨道；基线把IMFH1替换成MFH1并声称未见原编号，不能视为对象正确。' if c.case_id.endswith('008') else '核对机型、模块、数值、单位与对应引用；原文复核暂定，没有替代业务审核。'})
    (OUT/'source_correctness_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    proposal={'id':'rag-row-holdout-008','issue':'自动裁判只检查通道和功能，漏掉编号替换和引用对象错误。','proposal':'core事实应明确绑定IMFH1；不得用分析仪后端MFH1替代ISE后端IMFH1；答案引用应能支持完整对象。','applied_to_gold':False,'review_status':'pending'}
    (OUT/'annotation_review_item.json').write_text(json.dumps(proposal,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    score=json.loads((OUT/'scoring_completion.json').read_text(encoding='utf-8'));repeat=json.loads((OUT/'008_repeat_answers.json').read_text(encoding='utf-8'))
    summary={'new_questions':10,'all_pending':True,'auto_baseline_consistent':sum(a['judgement']['verdict']=='consistent' for a in answers if a['arm']=='baseline'),'auto_H8_consistent':sum(a['judgement']['verdict']=='consistent' for a in answers if a['arm']=='H8'),'provisional_source_review_baseline_correct':9,'provisional_source_review_H8_correct':10,'changed_contexts':sum(records[c.case_id,'H8']['context_changed'] for c in cases),'case008_extra_repeats':len(repeat),'evaluation_audit':score,'errors':sum(bool(a.get('error')) for a in answers),'time_omitted_per_user':True}
    (OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 冻结行级方案的新问题正确性验证与计分核对','','## 结论','',
      '按用户最新要求仅判断正确性，不再分析或比较耗时。计时重复已停止，旧原始记录保留但不进入本报告结论。',
      '冻结H8后制作10道此前未用于问答调试的问题。自动裁判判两边10/10一致；逐条原文与引用复核发现008基线把IMFH1替换成MFH1，裁判漏判。因此暂定源证据复核结果为基线9/10、H8 10/10，1题改善、其余9题未见回退。这不是正式评测分数，全部新题pending，需业务逐题确认。',
      '同页误计已确认：旧逻辑将只取正确行也算成负例召回；来源加实际完整引文区间核对的实验修正在3张真实表格、12个视图中符合预设。正式计分代码和历史报告未修改。','',
      '## 冻结、独立性与审核边界','',
      f"AI bundle_id=`{freeze['bundle_id']}`。freeze.json记录冻结时间、H8实现与行库、配置、生产代码、Gold和原资料哈希；先冻结，再选择和撰写新问题，再首次检索。H8仍是BM25前10行+原评分和硬条件，最多补一条，不替换原文，总上限8段/12000字符，未按结果调规则。",
      '样本从未被既有50题/10题声明证据及负例覆盖的页中选择，覆盖泵供电、Medica样本量、启动时间、样本类型、凯特斜率、ISE速度、糖化预处理、IMFH1电机、清洗方式和SV15_ISE。问题与既有业务主题不同，但来源结构曾参与构建行库，也由同一助手制作预期；不是资料未见、独立业务人员盲标的验收集。','',
      '## 逐题结果','',
      '| 编号 | 问题 | 自动基线/H8 | 原文复核基线/H8（暂定） |','|---|---|---|---|']
    for c in cases:lines.append(f"| {c.case_id} | {c.question} | 一致/一致 | {'对象错误/正确' if c.case_id.endswith('008') else '正确/正确'} |")
    lines+=['','## 008对象与引用复核','',
      '来源《IVD_生化_BS-5000系列_14 样本管理系统.pptx》：第12张“分析仪后端轨道”列MFH1，第13张“ ISE模块后端轨道”列IMFH1。两者描述都涉及返回通道传送带，但不能仅因功能文字相同就替换用户给出的编号。',
      '初次基线称“资料中对应编号是MFH1（未见IMFH1）”；H8从第13张补入IMFH1行，并按原编号回答。额外各重复2次：H8两次仍正确并引用该行；基线一次继续改称MFH1，另一次虽然文字写IMFH1，却引用只描述MFH1的第12张。按对象与引用均成立的标准，基线3次未完全通过、H8为3/3。补充重复是发现问题后的确认，不是额外独立盲测题。',
      '自动裁判漏判的原因是临时检查项只写“位于返回通道、驱动返回通道传送带”，没有充分约束对象绑定。annotation_review_item.json记录修订建议，原Gold未改，状态仍pending。不能为提高分数静默覆盖裁判结果。','',
      '## 行级证据计分核对','',
      '使用三张真实表：CSDB/DCF报警表、P05/P12泵表、IMFH1/IMFH2电机表。每表测试正确行、负例行、整页、截断负例四种视图，共12项。',
      '- 正确行：旧逻辑3/3错误计入同页另一行；实验修正均不计入。',
      '- 负例行与整页：6/6仍能检出完整负例，没有为了消除误报而漏掉真实完整引文。',
      '- 截断负例：旧逻辑3/3计为完整召回；实验修正不算完整召回，但保留“同父页未完整覆盖”诊断，不能视为无暴露风险。',
      '- 同一句数值在多行重复时，单纯字符串仍不足以定位；实验实现标为未确定，需来源行/列或原文区间。缺少实际included_text也不能凭ID算命中。',
      f"在历史冻结H8的59题上，旧计分有{score['historical_page_flag_cases']}题负例标记，实际完整负例引文为{score['historical_full_quote_cases']}题，差异对应023仅CSDB行。该变化来自计分口径核对，不是检索算法改善。",
      '正向计分也有边界：004、005、008存在指定来源未命中，但答案来自其他已索引内容或更细的行片段；来源/引文粒度与答案正确性必须分开。positive_mapping_diagnostic.json保留原统计，不自动把相似资料登记为等价来源。','',
      '## 建议','',
      '保留H8候选：冻结后的新问题上有1个明确的对象绑定改善，没有发现另外9题回退；样本仅10题且全pending，不能宣布广泛泛化或直接切到默认。',
      '评测优先补齐两项：负例以来源与实际引文区间计分，保留部分/未确定诊断；回答检查必须核对对象—条件—事实—引用的绑定。只有关键词或数值覆盖，无法识别本次IMFH1/MFH1混用。',
      '本轮新增测试集与历史50题/回归集分别保存，未更改历史分数、review_status或默认数据集。后续正式计分须由业务人员审核新题。','',
      '## 交付核验','',
      '新JSON与JSONL逐条一致；加载器允许pending诊断加载，并拒绝默认正式加载；10条引文在原资料和当前索引均核验；冻结哈希前后通过。新问题20次生成与20次裁判，008补充4次生成，无执行错误。来源区间审计候选8项测试通过；所有相关29项测试通过。','']
    txt='\n'.join(lines);(OUT/'report.md').write_text(txt,encoding='utf-8');(ROOT/'docs/rag_row_holdout_validation_20261105.md').write_text(txt,encoding='utf-8')
    body='<!doctype html><meta charset="utf-8"><title>新问题正确性与行级计分核对</title><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:12px;margin:8px}</style><pre>'+html.escape(txt)+'</pre>'
    for c in cases:
        details={'question':c.question,'review_status':'pending','expected_facts':c.required_facts,'gold_evidence':c.gold_evidence,'baseline_answer':ans[c.case_id,'baseline']['answer'],'H8_answer':ans[c.case_id,'H8']['answer'],'source_review':next(r for r in review if r['id']==c.case_id),'baseline_context':records[c.case_id,'baseline']['context'],'H8_context':records[c.case_id,'H8']['context']}
        body+='<details><summary>'+html.escape(c.case_id+' '+c.question)+'</summary><pre>'+html.escape(json.dumps(details,ensure_ascii=False,indent=2,default=str))+'</pre></details>'
    (OUT/'report.html').write_text(body,encoding='utf-8')
    (OUT/'delivery_verification.json').write_text(json.dumps({'pending_cases':len(cases),'correctness_inputs':len(rows),'main_answers':len(answers),'extra_repeat_answers':len(repeat),'json_equals_jsonl':True,'formal_loader_rejects_pending':formal_rejected,'source_quotes_verified':len(source),'frozen_hashes_unchanged':True,'timing_excluded':True,'scoring_source_tests':8,'related_tests':29},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
