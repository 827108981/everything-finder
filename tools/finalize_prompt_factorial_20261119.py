"""Freeze the outcome, preserving automatic results and manual audit separately."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from tools.validate_prompt_factorial_20261119 import OUT, ROOT, HISTORY, ARMS, TARGETS, check_frozen, write
from tools.report_prompt_factorial_20261119 import LABELS, summarize


def effective_rows():
    original = [json.loads(l) for l in (OUT/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    recovered = [json.loads(l) for l in (OUT/'transport_recovery.jsonl').read_text(encoding='utf8').splitlines()] if (OUT/'transport_recovery.jsonl').exists() else []
    bykey = {(r['id'],r['arm'],r['round']):r for r in recovered}
    return original, [bykey.get((r['id'],r['arm'],r['round']),r) for r in original]


def main():
    protocol=check_frozen()
    original,rows=effective_rows()
    assert len(rows)==500 and len({(r['id'],r['arm'],r['round']) for r in rows})==500
    payloads={}
    for r in rows:
        assert r['context_sha256']==protocol['frozen_context_sha256'][r['id']]
        p=dict(r['api_request'])
        assert p['messages'][0]['content']==protocol['prompts'][r['arm']]
        p['messages']=p['messages'][1:]
        payloads.setdefault(r['id'],set()).add(json.dumps(p,sort_keys=True,ensure_ascii=False))
        assert 'required_facts' not in json.dumps(r['api_request'],ensure_ascii=False)
    assert all(len(v)==1 for v in payloads.values())
    summary=summarize(rows)
    write(OUT/'summary.json',summary)
    service=[json.loads(l) for l in (OUT/'service_entry.jsonl').read_text(encoding='utf8').splitlines()]
    service_recovery=OUT/'service_entry_judge_recovery.jsonl'
    if service_recovery.exists():
        rr={(r['id'],r['arm']):r for r in (json.loads(l) for l in service_recovery.read_text(encoding='utf8').splitlines())}
        service=[rr.get((r['id'],r['arm']),r) for r in service]
    assert len(service)==14
    payload_audit=json.loads((OUT/'real_entry_payload_comparison.json').read_text(encoding='utf8'))
    manual=json.loads((OUT/'manual_review.json').read_text(encoding='utf8')) if (OUT/'manual_review.json').exists() else {'status':'待核实','records':[]}
    target_failures={a:[k for k in TARGETS if summary['target_repeats'][k][a]!=['consistent']*3] for a in ARMS}
    acceptance=dict(bundle_id=protocol['bundle_id'],tasks=500,first_round_cases_per_arm=111,
        frozen_context_and_non_system_payload_checks=500,all_frozen_inputs_unchanged=True,
        initial_execution_errors=sum('error' in r for r in original),remaining_errors=summary['errors'],
        real_service_calls=14,real_service_context_equal=sum(r.get('context_equal_frozen') is True for r in service),
        real_service_saved_payloads_equal=sum(r.get('equal') is True for r in payload_audit),
        real_service_payload_not_saved=[r for r in payload_audit if r.get('status')=='no_saved_request'],
        real_service_errors=[{k:r[k] for k in ('id','arm','error')} for r in service if 'error' in r],
        target_not_stable=target_failures,manual_review=manual,
        recommendation='限定采用型号格式核验；候选提示暂不默认采用',
        runtime_default_prompt='baseline',adapter_default=False,persistent_settings_changed=False,
        no_gold_changes=True,no_reranker_or_selection_changes=True,
        caveats=['自动裁判不是人工正确率','50正式/60待审/1争议分开','019未修复','050范围核验仍待核实','已有10个跨文档共识测试失败未纳入本轮修复'])
    write(OUT/'final_acceptance.json',acceptance)
    lines=['# 固定Context提示词四组对照：实验结论','','结论：型号连接符格式核验可限定采用；本轮提示候选暂不替换默认提示。019保持冻结，Gold、索引、召回、BGE和简单选段均未修改。','',
        '## 实验控制','','- 111题×4组首次对照，7道重点题每组预设3次，总计500个生成与独立判分任务。','- 另有7题×2组共14次真实AnswerService.answer入口调用，真实本地召回与BGE推理，未用Gold干预运行。','- 四组仅system提示不同，同一题Context与其他请求参数逐条一致；没有按质量挑选答案或修改提示。','- 模型为配置中的gpt-5.6-sol；生成和裁判为同配置模型的独立请求。未获得底层模型指纹，不能确认服务商内部是否固定路由。','- 初始HTTP403/无效裁判JSON作为执行故障保留。只对执行缺口补跑，成功生成而判分失败的记录复用原回答。','',
        '## 首次正式50题（自动判分）','','|方案|一致|不完整|需澄清|行为不符|错误|不确定|执行错误|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v=summary['by_arm'][arm]['reviewed']['verdicts']
        lines.append('|'+LABELS[arm]+'|'+'|'.join(str(v.get(k,0)) for k in ('consistent','incomplete','needs_clarification','behavior_mismatch','incorrect','uncertain','error'))+'|')
    lines+=['','旧批次的40/50不能直接作为本轮旧提示的分母；本轮旧提示重新生成，成对比较见上表。原批次只修改确定性评测格式检查，004从不确定变一致，40→41；这是判分修正，答案原文没有变化。','',
        '## 待审和争议集合（不并入正式分）','','|方案|pending60题|disputed1题|','|---|---|---|']
    for a in ARMS:lines.append('|'+LABELS[a]+'|'+json.dumps(summary['by_arm'][a]['pending']['verdicts'],ensure_ascii=False)+'|'+json.dumps(summary['by_arm'][a]['disputed']['verdicts'],ensure_ascii=False)+'|')
    lines+=['','## 相对本轮旧提示的首次变化','','|方案|正式变好|正式变差|待审变好|待审变差|','|---|---|---|---|']
    for a in ARMS[1:]:lines.append('|'+LABELS[a]+'|'+'|'.join('、'.join(summary['by_arm'][a][s][k]) or '无' for s,k in [('reviewed','gains'),('reviewed','losses'),('pending','gains'),('pending','losses')])+'|')
    lines+=['','## 重点题预设三次结果','','|题目|旧提示|回答行为|完整性与引用|组合|','|---|---|---|---|']
    for k in TARGETS:lines.append('|'+k+'|'+'|'.join(' / '.join(summary['target_repeats'][k][a]) for a in ARMS)+'|')
    lines+=['','## 真实问答入口','','|题目|方案|Context等于冻结输入|自动结果|','|---|---|---|---|']
    for r in service:lines.append('|'+r['id']+'|'+LABELS[r['arm']]+'|'+str(r.get('context_equal_frozen'))+'|'+r.get('judgement',{}).get('verdict',r.get('error','未完成'))+'|')
    lines+=['','真实入口遵循现有配置：Adapter关闭、额外语义澄清关闭、答案二次复核关闭。没有为了通过验收偷偷启用其他API层。13次保存的真实生成请求与对应冻结请求完全一致；035组合因裁判失败导致请求快照未落盘，只能从完成trace恢复回答和Context，不能声称该条请求字节已核验。','',
        '## 关键回答与裁判复核','','自动判断的错误、遗漏、不确定与业务错误必须分开。复核记录保留原判，没有改Gold、改自动分或替模型补引用。','']
    for r in manual.get('records',[]):
        lines+=['- '+r['id']+'：'+r.get('finding','')]
    lines+=['','## 测试与剩余问题','','- 本轮重点单元及兼容测试67项通过。','- 全RAG测试579通过、10失败、1跳过、1预期失败；10项失败均来自旧跨文档共识测试，在关闭本轮新增功能的内存对照下仍同样失败。不能写成全库测试通过。','- 004格式修正保留型号的全部数字和后缀，且要求实际来源附近有型号线索；不把不同型号、物料编号或任意错误码视为别名。','- 050四类原因与Gold一致；原引文没有型号，不能仅凭同书其他页出现CL-8000i就继承适用范围，仍单列技术核验不足，不称为业务答案错误。','- 条码的多资料已在Context，提示可能促使使用正确引用，但仍须检查实际所引来源，不能把出现多个文件当成平台共识成立。','- 现有样本均为已有开发/回归资料，pending并不意味着全新独立验收，不能外推到全部未测业务。','',
        '## 采用决定','','1. 限定采用评测格式修正，保留未核实范围的不确定状态。','2. 候选提示保留为显式实验参数，生产请求默认baseline；不修改用户持久化配置。','3. 不能只凭总体一致数更高采用。重点题重复失败及真实入口错误说明本轮方案尚未达到稳定门槛。','4. 下一轮只针对复现的分支错误与过度扩展做固定Context的小对照；先核对来源表达是否足够明确，再决定需要改提示还是资料表达，不恢复题号规则或多层过滤。','',
        '## 交付','','- `report.html`：程序原有诊断格式，四组回答与逐项判分前后对照，包含冻结的8阶段排名和原资料定位按钮。','- `open_report.cmd`：启动本地报告及原资料定位服务。','- `answers.jsonl`：原始500任务，包含原请求、原回答和原判。','- `transport_recovery.jsonl`：只补执行缺口，原失败保留。','- `service_entry.jsonl`、`service_entry_judge_recovery.jsonl`：真实问答入口与仅补判记录。','- `evaluation_only_audit.json`：判分格式修正单独归因。','- `protocol.json`、`final_acceptance.json`、`manual_review.json`：冻结、验收与复核记录。','']
    lines+=['','## 修改边界','','- `local_full_text_search/rag/generation/base.py`：末尾新增可选prompt_variant，兼容原有调用，默认baseline。','- `local_full_text_search/rag/generation/openai_compatible.py`：保留原提示，提供行为、完整性和组合三个显式实验变体，只改变system提示，没有新增运行中的API调用层。','- `local_full_text_search/rag/evaluation/answer_judge.py`：限定的型号连接符等价核验，记录format_equivalences；未推断资料适用范围。','- `tests/rag/test_prompt_factorial_20261119.py`：11项本轮新增测试。','- `tools/validate_prompt_factorial_20261119.py`、`verify_prompt_service_entry_20261119.py`、`recover_prompt_transport_20261119.py`：冻结对照、真实入口和执行故障补跑。','- `tools/report_prompt_factorial_20261119.py`、`check_prompt_factorial_report_20261119.cjs`、`finalize_prompt_factorial_20261119.py`：原程序格式报告、界面验证及结论。','- 没有修改Gold标签、持久化配置、QueryPlan、召回、BGE、条件判断或最终选段；本轮实验没有加入题号、页码或答案特判。','',
        '## 复现命令','','```powershell','.venv/Scripts/python.exe -m pytest tests/rag/test_prompt_factorial_20261119.py tests/rag/test_generation_gateway.py tests/rag/test_answer_judge.py tests/rag/test_binding_scoring.py -q','.venv/Scripts/python.exe -m pytest tests/rag -q','.venv/Scripts/python.exe -m tools.validate_prompt_factorial_20261119 run','.venv/Scripts/python.exe -m tools.verify_prompt_service_entry_20261119','.venv/Scripts/python.exe -m tools.recover_prompt_transport_20261119','.venv/Scripts/python.exe -m tools.finalize_prompt_factorial_20261119','.venv/Scripts/python.exe -m tools.report_prompt_factorial_20261119','```','','已有执行记录会跳过，不能通过删除不满意记录重抽答案。执行故障和无效判分记录保留。UI检查由`tools/check_prompt_factorial_report_20261119.cjs`对本地报告服务执行，实际桌面应用打开动作在自动测试中mock，不能据此宣称已逐个打开所有资料。']
    text='\n'.join(lines)
    (OUT/'conclusion.md').write_text(text,encoding='utf8')
    (ROOT/'docs/rag_prompt_factorial_validation_20261119.md').write_text(text+'\n\n## 冻结提示全文\n\n'+'\n'.join('### '+a+'\n\n```text\n'+p+'\n```\n' for a,p in protocol['prompts'].items()),encoding='utf8')
    write(OUT/'step_status.json',dict(freeze='通过',focused_tests='通过',all_rag_tests='失败（10项既有失败，579通过）',
        evaluation_format='通过（004）',evaluation_scope='待核实（050）',paired_generation='通过' if not summary['errors'] else '失败',
        actual_entry='通过' if all(r.get('context_equal_frozen') and not r.get('error') for r in service) else '失败',
        candidate_adoption='失败（稳定性或回归门槛未满足）',ui='待核实（以最终快照ui_audit为准）'))
    print('FINALIZED',json.dumps({a:summary['by_arm'][a]['reviewed']['verdicts'] for a in ARMS},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
