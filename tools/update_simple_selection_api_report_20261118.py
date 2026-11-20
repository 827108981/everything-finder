"""Attach actual generation answers and separate Gold-judge results to report.html."""
from collections import Counter
import base64
import gzip
from hashlib import sha256
import html
import json
from pathlib import Path
import re

from tools.render_general_native_report import packed
from tools.report_vector_packaging_20261116 import SourceRegistry
from tools.validate_simple_selection_20261118 import OUT


def main():
    answer_path = OUT / "api_answers.jsonl"
    answer_rows = [json.loads(line) for line in answer_path.read_text(encoding="utf8").splitlines()]
    contexts = {r["id"]: r for r in (json.loads(line) for line in (OUT / "real_entry_simple.jsonl").read_text(encoding="utf8").splitlines())}
    source_path = OUT / "report_sources.json"
    prior_sources = json.loads(source_path.read_text(encoding="utf8"))
    registry = SourceRegistry([])
    registry.sources.update(prior_sources)

    for row in answer_rows:
        context = contexts[row["id"]]
        entries = {entry["source_id"]: entry for entry in context["entries"]}
        cited = sorted(set(re.findall(r"\[(S\d+)\]", row.get("draft_answer", ""))), key=lambda value: int(value[1:]))
        citation_sources = []
        for citation in cited:
            entry = entries.get(citation)
            if not entry:
                continue
            ids = []
            for source in entry.get("sources", []):
                ids.extend(registry.register({
                    "file_path": source.get("file_path", ""),
                    "location": source.get("location", entry.get("location", "")),
                    "page_number": source.get("page_number"),
                    "slide_number": source.get("slide_number"),
                }))
            citation_sources.append({"citation": citation, "filename": entry.get("filename", ""),
                                     "location": entry.get("location", ""), "source_ids": ids})
        row["citation_sources"] = citation_sources
    if registry.misses:
        raise AssertionError(f"Unresolved cited source links: {registry.misses[:3]}")
    # Enrich only the report payload; keep the saved API answer/judge transcript immutable.
    source_path.write_text(json.dumps(registry.sources, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

    summary = json.loads((OUT / "api_summary.json").read_text(encoding="utf8"))
    labels = {"consistent": "一致", "incomplete": "必答项不全", "incorrect": "明确错误", "needs_clarification": "应澄清", "uncertain": "不确定", "unavailable": "未判定"}
    reviewed = summary["by_review_status"]["reviewed"]
    counts = summary["verdicts"]
    banner = (
        '<!-- API_SUMMARY_START --><section class="panel" id="apiAnswerSummary"><h2>实际 LLM 回答与 Gold 自动裁判</h2>'
        f'<p>固定简化方案的实际 Context，生成模型 <b>{html.escape(summary["generation_model"])}</b> 逐题生成 {summary["generation_calls"]}/111 个答案；独立 Gold 裁判完成 {summary["judge_calls"]}/111 题。生成错误 {len(summary["errors"])}，引用格式有效 {summary["citation_format_valid"]}/111。应用设置的答案二次复核为 {"开启" if summary["answer_review_enabled"] else "关闭"}。</p>'
        f'<p><b>已审核50题：</b>一致 {reviewed["verdicts"].get("consistent", 0)}，必答项不全 {reviewed["verdicts"].get("incomplete", 0)}，明确错误 {reviewed["verdicts"].get("incorrect", 0)}，应澄清 {reviewed["verdicts"].get("needs_clarification", 0)}，不确定 {reviewed["verdicts"].get("uncertain", 0)}。</p>'
        f'<p>全部111题自动裁判：' + '；'.join(f'{html.escape(labels.get(k, k))} {v}' for k, v in counts.items()) + '。pending/disputed 题只作诊断，不计正式结论。自动裁判不是人工复核；裁判“不确定”不按正确计，答案正确率不能从单一总数推断。</p>'
        '<p>7条未含引用的回答均为固定拒答“资料中没有足够依据。”，属于应用允许的拒答，不等同于引用格式出错。104条实质回答的引用编号均有效，但编号有效不代表引用支持结论。</p>'
        '<p>004、050的必答项均已覆盖；自动编号检查未在所引用片段内找到完整型号，因此保留“不确定”并列入复核，不判成事实错误。条码题的码制列表正确，但仅引用单型号材料，被裁判判为来源范围不足，需核查引用而非重写码制答案。</p>'
        '<p>本轮只生成simple方案，旧方案没有配对答案，不能据此宣称删除过滤提高或降低了答案正确率。下方原API前结论保留为历史记录；本节为追加的API验证。Gold仅进入独立裁判，不进入生成请求。</p></section><!-- API_SUMMARY_END -->'
    )
    page_path = OUT / "report.html"
    page = page_path.read_text(encoding="utf8")
    page = re.sub(r'<!-- API_SUMMARY_START -->[\s\S]*?<!-- API_SUMMARY_END -->', '', page)
    # Upgrade a report produced by the first (unmarked) updater as well.
    page = re.sub(r'<section class="panel" id="apiAnswerSummary">[\s\S]*?</section>', '', page)
    page = re.sub(r'const API_ANSWERS=await unpack\([\s\S]*?renderApiAnswer\(\);\n', '', page)
    start = page.index('const PACKED=') + len('const PACKED=')
    packed_arms, end = json.JSONDecoder().raw_decode(page[start:])
    detail = json.loads(gzip.decompress(base64.b64decode(packed_arms['simple'])))
    by_id = {r['id']: r for r in answer_rows}
    for case in detail['cases']:
        record = by_id[case['id']]
        text = record.get('draft_answer', '')
        refusal = text.strip() == '资料中没有足够依据。'
        case['answer'] = dict(mode='full', text=text, status='insufficient_evidence' if refusal else 'answered',
                              model=record['model'], source_ids=record['citation_check']['cited_ids'],
                              warnings=['固定Context直接生成；未重跑检索，未启用应用内答案二次复核。'])
        case['generation'] = dict(draft_answer=text, model=record['model'], replay_mode='frozen_context_generation')
        case['gold_judgement'] = record['judgement']
        case['actual_behavior'] = 'refuse' if refusal else 'answer'
        case['api_validation'] = dict(context_sha256=record['context_sha256'], citation_check=record['citation_check'])
    detail['run']['mode'] = '固定simple Context实际API生成111题；独立LLM裁判111题；其他方案仅LLM前对照'
    detail['summary']['answer_judgement_counts'] = dict(Counter(r['judgement']['verdict'] for r in answer_rows))
    packed_arms['simple'] = packed(detail)
    page = page[:start] + json.dumps(packed_arms) + page[start + end:]
    if 'id="apiAnswerPanel"' not in page:
        page = page.replace('<p id="sourceStatus">', '<div id="apiAnswerPanel"></div><p id="sourceStatus">', 1)
    page = page.replace('API调用0。新增20题为pending，只作诊断。', 'simple方案已追加API问答；其余方案未生成答案。新增20题为pending，只作诊断。')
    page = page.replace('默认阶段 Hit@5', 'LLM前阶段 Hit@5')
    page = page.replace('<main>', '<main>' + banner, 1)
    payload = packed(answer_rows)
    js = "const API_ANSWERS=await unpack(" + json.dumps(payload) + ");\n" + r'''
const API_BY_ID=Object.fromEntries(API_ANSWERS.map(x=>[x.id,x]));
const API_VERDICT_LABELS={consistent:'一致',incomplete:'必答项不全',incorrect:'错误（裁判标记，需核查）',needs_clarification:'应澄清',uncertain:'不确定',unavailable:'未判定'};
const API_STATUS_LABELS={covered:'已覆盖',missing:'缺失',contradicted:'与金标矛盾',uncertain:'不确定',present:'出现禁用主张',absent:'未出现禁用主张'};
function renderApiAnswer(){const id=document.getElementById('compareCase').value;const r=API_BY_ID[id];if(!r)return;const j=r.judgement||{};const c=SIMPLE_COMPARE.simple[id];
const sourceButtons=(r.citation_sources||[]).flatMap(x=>(x.source_ids||[]).map(s=>`<button type="button" data-original-source="${esc(s)}">${esc(x.citation)} · ${esc(x.filename)} · ${esc(x.location)}：打开原资料</button>`)).join('');
const rubrics=(j.rubrics||[]).map(x=>`<div class="api-rubric"><b>${esc(x.id)} · ${esc(API_STATUS_LABELS[x.status]||x.status)} · ${esc(x.role||x.kind||'')}</b><div>${esc(x.gold_text||'')}</div><small>${esc(x.reason||'')}</small></div>`).join('');
document.getElementById('apiAnswerPanel').innerHTML=`<section class="panel"><h3>LLM 实际回答 · ${esc(r.id)} · simple方案</h3><p>审核状态：${esc(r.review_status)} · Gold自动裁判：<b>${esc(API_VERDICT_LABELS[j.verdict]||j.verdict||'未判定')}</b> · 行为：${esc(j.behavior||'—')} · 引用格式：${r.citation_check?.structurally_valid?'有效':r.draft_answer.trim()==='资料中没有足够依据。'?'固定拒答，无需引文':'无效'}</p><pre class="api-answer">${esc(r.draft_answer||r.error||'无回答')}</pre><details><summary>裁判依据（${(j.rubrics||[]).length} 项）</summary><p>${esc(j.behavior_reason||'')}</p><p>对象绑定：${esc(j.object_binding?.status||'—')} · ${esc(j.object_binding?.reason||'')}</p><p>来源绑定：${esc(j.source_binding?.status||'—')} · ${esc(j.source_binding?.reason||'')}</p>${rubrics||'<p>无逐项判分。</p>'}</details><details><summary>答案引用的原资料（${(r.citation_sources||[]).length} 项）</summary>${sourceButtons||'<p>答案没有可定位的 [S#] 引用。</p>'}</details><details><summary>API 与引用诊断</summary><pre>${esc(JSON.stringify({model:r.model,context_sha256:r.context_sha256,context_chars:r.context_chars,citation_check:r.citation_check,identifier_check:j.identifier_check},null,2))}</pre></details></section>`;}
document.getElementById('compareCase').addEventListener('change',renderApiAnswer);renderApiAnswer();
'''
    if "const SIMPLE_COMPARE=await unpack(" not in page:
        raise RuntimeError("Simple comparison payload is missing from report")
    page = page.replace("\ninit();\n", "\n" + js + "\ninit();\n", 1)
    page_path.write_text(page, encoding="utf8")
    write_api_conclusion(answer_rows, summary)
    # Preserve the previous no-API acceptance as historical; append this run explicitly.
    acceptance_path = OUT / 'final_acceptance.json'
    acceptance = json.loads(acceptance_path.read_text(encoding='utf8'))
    acceptance['api_followup'] = dict(summary=summary, report_html_sha256=sha256(page_path.read_bytes()).hexdigest(),
        answer_sha256=sha256(answer_path.read_bytes()).hexdigest(), paired_baseline_generated=False,
        method='Frozen actual Context + existing generator prompt + separate same-model Gold judge')
    acceptance_path.write_text(json.dumps(acceptance, ensure_ascii=False, indent=2)+'\n',encoding='utf8')
    print("API_REPORT_UPDATED", {"cases": len(answer_rows), "citation_source_manifest": len(registry.sources),
                                 "cited_source_links": sum(len(x.get("citation_sources", [])) for x in answer_rows)})


def write_api_conclusion(rows, summary):
    lines = ['# 同一批111题：冻结Context的实际API验证', '',
        '111次生成和111项独立Gold判分均完成，未重跑召回/重排，未修改Gold或运行时算法。生成与裁判使用同一已配置模型gpt-5.6-sol的分离请求；这不是不同模型的独立交叉验证。', '',
        '|审核状态|题数|一致|必答缺失|需澄清|错误标记|不确定|', '|---|---:|---:|---:|---:|---:|---:|']
    for status, group in summary['by_review_status'].items():
        v=group['verdicts']
        lines.append('|'+status+'|'+str(group['cases'])+'|'+ '|'.join(str(v.get(k,0)) for k in ('consistent','incomplete','needs_clarification','incorrect','uncertain'))+'|')
    lines += ['', '正式自动一致率为40/50（80%）；pending/disputed不进入这个分母。不确定不是确定错误，一致也不代替业务审核。只有simple方案调用API，旧方案没有配对回答，本轮不能证明删除条件过滤提高或降低答案正确率。', '',
        '104条实质答案都使用存在的引用编号；7条无引用回答均为应用允许的固定拒答，不能当作7次引用格式故障。独立裁判仍按各题所要求的回答行为检查这些拒答。', '',
        '## 正式题待处理项', '', '|题目|自动判分|需要核对的具体内容|', '|---|---|---|']
    for r in sorted(rows,key=lambda r:r['id']):
        if r['review_status']!='reviewed' or r['judgement']['verdict']=='consistent':continue
        j=r['judgement']
        findings=[x.get('reason','') for x in j.get('rubrics',[]) if x['status'] in ('missing','contradicted','uncertain','present') and x.get('role')!='context']
        if j.get('behavior')!='match':findings.append(j.get('behavior_reason',''))
        lines.append('|'+r['id']+'|'+j['verdict']+'|'+ '；'.join(findings).replace('|','/').replace('\n',' ')+'|')
    lines += ['', '## 解释边界', '',
        '- 019关键原因和处理证据在此前未进入重排入口；实际答案遗漏四类原因、排除污染后灌注的顺序及40次以上建议，还被裁判指出引用未支持“系统参数”。不能把这些遗漏全归因于LLM。',
        '- 016、021：有可回答内容，但模型直接回复资料不足；属于本次实际生成失败，需要结合Context干扰与提示词另作对照。',
        '- 014给出了两个条件分支，并提示先确认BM10，但既有Gold要求明确追问；保留原判分，供业务复核回答方式要求。',
        '- 004、050：必答项均covered，LLM对象/来源核对均match；最终uncertain来自代码检查未在实际被引用的片段里找到完整型号，不能直接说内容答错。050另出现非中文杂字符，原答照录。',
        '- 条码题：列表与Gold一致，问题是只引用CL-2600i单型号材料，而没有引用Context中更广范围的两份资料。incorrect是来源绑定标记，不是码制列表错误。',
        '- 无关产线题017、018和independent-12本次均拒答；新增错场景Context并不必然被模型采用。但单次结果不能保证其他问法都安全。',
        '- 温度传感器row-holdout2-007、ISE对比independent-05缺证据的问题也表现为答案不完整。', '',
        '原始回答和判分：api_answers.jsonl；分组统计：api_summary.json；冻结输入：api_protocol.json；交互查看：report.html。报告中只给simple方案填入本轮答案，其他三个方案仍为未生成。']
    text='\n'.join(lines)+'\n'
    (OUT/'api_conclusion.md').write_text(text,encoding='utf8')
    original=(OUT/'conclusion.md').read_text(encoding='utf8')
    if '\n<!-- API_FOLLOWUP -->' in original:original=original.split('\n<!-- API_FOLLOWUP -->')[0]
    (OUT/'conclusion.md').write_text(original+'\n<!-- API_FOLLOWUP -->\n\n'+text,encoding='utf8')


if __name__ == "__main__":
    main()
