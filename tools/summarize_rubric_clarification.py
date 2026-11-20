"""Readable audit plus independent label/algorithm comparisons."""
from __future__ import annotations
import collections
import hashlib
import html
import json
import sys
from pathlib import Path

OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('reports/rubric_clarification_20261102_v4')
FIXTURES=Path('tests/rag/fixtures')


def main():
    attempts=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=list({(r['cohort'],r['id']):r for r in attempts}.values())
    assert len(rows)==60
    all_rows=rows
    errors=[r for r in rows if 'error_type'in r]
    completed=[r for r in rows if 'error_type'not in r]
    unavailable=[r for r in completed if any(e.get('stage')=='clarification_assessment_unavailable' for e in r['events'])]
    # Exclude provider-fallback rows from semantic-gate outcome comparisons.
    rows=[r for r in completed if r not in unavailable]
    assert all(r['retrieval_stages_identical'] for r in completed)
    controls=json.loads((OUT/'controls.json').read_text(encoding='utf-8'))
    audit=json.loads((FIXTURES/'rag_gold_v3_rubric_audit.json').read_text(encoding='utf-8'))
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    for d in protocol['datasets'].values():
        assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    dev=[r for r in rows if r['cohort']=='development']
    hold=[r for r in rows if r['cohort']=='holdout_pending']
    values={}
    for cohort,selected in [('development',dev),('original18',[r for r in dev if int(r['id'][-3:])<=18]),
                            ('added32',[r for r in dev if int(r['id'][-3:])>18])]:
        values[cohort]={}
        for axis in ('before_v3','before_v4_proposed','after_v3','after_v4_proposed'):
            judgements=[r['judgements'][axis] for r in selected]
            values[cohort][axis]={'n':len(selected),'verdicts':dict(collections.Counter(j['verdict'] for j in judgements)),
                'behavior':dict(collections.Counter(j['behavior'] for j in judgements))}
    original_cases=[json.loads(l) for l in (FIXTURES/'rag_gold_v3.jsonl').read_text(encoding='utf-8').splitlines()]
    required_clarify_ids={c['id'] for c in original_cases if c['expected_behavior']=='clarify'}
    clarify=[r for r in dev if r['expected_behavior']=='clarify']
    false_clarify=[r['id'] for r in dev if r['expected_behavior']!='clarify' and r['answer']['status']=='needs_clarification']
    missed=[r['id'] for r in clarify if r['answer']['status']!='needs_clarification']
    summary={'audit_counts':audit['counts'],'axes':values,
        'clarification':{'expected':len(required_clarify_ids),'tested':len(clarify),
            'untested':sorted(required_clarify_ids-{r['id'] for r in clarify}),
            'status_correct':len(clarify)-len(missed),
            'missed':missed,'false_clarification_ids':false_clarify,
            'semantic_matches':sum(r['judgements']['after_v3']['behavior']=='match' for r in clarify)},
        'holdout_pending':{'planned':10,'n':len(hold),'false_clarification_ids':[r['id'] for r in hold if r['answer']['status']=='needs_clarification'],
            'judge_behavior':dict(collections.Counter(r['judgements']['after_v3']['behavior'] for r in hold))},
        'controls':{'planned':len(controls),'n':sum('error_type'not in r for r in controls),
            'behavior_matches':sum(r.get('judgement',{}).get('behavior')=='match' for r in controls),
            'infrastructure_failures':[r['id'] for r in controls if 'error_type'in r],
            'failures':[r['id'] for r in controls if 'error_type'not in r and r['judgement']['behavior']!='match']},
        'checks':{'historical_gold_hashes_unchanged':True,'retrieval_identical_completed_pairs':len(completed),
            'raw_attempt_rows':len(attempts),'latest_errors':len(errors),
            'infrastructure_fallback_ids':[r['id'] for r in unavailable],
            'provider_error_ids':[r['id'] for r in errors],
            'status_marker_leaks':[r['id'] for r in rows if '<needs_clarification>' in r['answer']['text']]},
        'request_reuse':{key:sum(r.get('request_reuse',{}).get(key,0) for r in rows)
            for key in ('assessment_calls','generation_reused','fresh_generation_calls')}}
    (OUT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    esc=html.escape
    h=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>标注复核与澄清验证</title>',
       '<style>body{font:16px system-ui;margin:30px;max-width:1550px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}pre{white-space:pre-wrap}summary{padding:10px;cursor:pointer}.note{padding:15px;background:#fff1d0}</style>',
       '<h1>标注复核与澄清验证</h1><p class="note">当前为部分验证：服务商先返回502，重试返回403；澄清候选默认关闭，不能视为完整验收通过。v3历史Gold不变；v4仅为pending草案，不是正式计分。标注变化和程序变化分别比较，模型裁判仅辅助审核。</p>',
       '<pre>'+esc(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>',
       '<h2>全部50题字段复核（含引文）</h2>']
    for c in audit['cases']:
        h.append('<details><summary>'+esc(c['id']+' '+c['proposal']+' '+c['question'])+'</summary><p>'+esc(c['reason'])+'</p><table><tr><th>原检查项</th><th>建议检查项（pending）</th></tr><tr><td><pre>'+esc(json.dumps(c['original_checks'],ensure_ascii=False,indent=2))+'</pre></td><td><pre>'+esc(json.dumps(c['proposed_checks'],ensure_ascii=False,indent=2))+'</pre></td></tr></table><details><summary>证据与逐项说明</summary><pre>'+esc(json.dumps({k:c[k] for k in ('evidence','check_review','scope_note','additional_evidence_for_review') if k in c},ensure_ascii=False,indent=2))+'</pre></details></details>')
    h.append('<h2>实际回答前后对照</h2>')
    for r in sorted(rows,key=lambda r:(r['cohort'],r['id'])):
        h.append('<details><summary>'+esc(r['id']+' '+r['question']+' | 状态 '+r['answer']['status'])+'</summary><table><tr><th>修改前</th><th>修改后</th></tr><tr><td><pre>'+esc(r['baseline_answer']['text'])+'</pre></td><td><pre>'+esc(r['answer']['text'])+'</pre></td></tr></table><details><summary>四组标注与程序结果</summary><pre>'+esc(json.dumps(r['judgements'],ensure_ascii=False,indent=2))+'</pre></details></details>')
    h.append('<h2>未完成验证的记录（不计为通过）</h2>')
    for r in [*errors,*unavailable]:
        h.append('<details><summary>'+esc(r['id']+' '+r.get('error_message','澄清接口失败，回退了原回答流程'))+'</summary><pre>'+esc(json.dumps({k:r.get(k) for k in ('id','cohort','error_type','error_message')},ensure_ascii=False,indent=2))+'</pre></details>')
    h.append('<h2>条件已知、比较题和资料不足控制</h2>')
    for c in controls:
        h.append('<details><summary>'+esc(c['id']+' '+c.get('question','接口失败，未完成控制验证'))+'</summary><pre>'+esc(json.dumps(c,ensure_ascii=False,indent=2))+'</pre></details>')
    h.append('</html>')
    (OUT/'report.html').write_text('\n'.join(h),encoding='utf-8')
    md=['# 标注复核与澄清验证（2026-11-02）','','**结论：标注复核已交付；澄清机制仅部分验证，尚未通过完整业务验收，默认关闭。** 服务商先返回HTTP 502，重试返回HTTP 403，已停止远程请求。历史Gold及reviewed状态未修改；所有新标注建议均为pending，应用默认数据集保持原样。','',
        '## 标注复核','','复核原问题和已有证据，不按生成答案反推标准。全部50题有逐项说明；69条正向引文仍可在当前AI索引定位。本次没有重开全部原文件，没有新增业务事实或引文。',
        '',f'建议分类：{audit["counts"]}。第003题涉及安全项的适用边界，第032/033题涉及个案事实如何转为排查要求；第007题发现同机型资料疑似数值冲突，需业务人员确认。',
        '', '## 两个变量分别比较','','| 集合 | 同答案旧标注 | 同答案新标注草案 | 新程序旧标注 | 新程序新标注草案 |','|---|---:|---:|---:|---:|']
    for cohort in values:
        md.append('| '+cohort+' | '+' | '.join(f'{values[cohort][axis]["verdicts"].get("consistent",0)}/{values[cohort][axis]["n"]}' for axis in ('before_v3','before_v4_proposed','after_v3','after_v4_proposed'))+' |')
    md+=['','表中仅包含完成有效判断的同一批配对问题，排除了网络失败和回退题，不能外推全部50题；模型裁判consistent数量不是人工准确率。横向由旧标注变为新草案的提升只代表标注口径变化，不能归因于程序。判断程序效果须固定同一套标注比较。新草案未获人工批准，不属于正式分数。','',
        '## 澄清机制','','根因：检索前的策略只处理极短问题；missing_fields始终为空，阶段竞争分支无法触发；旧生成提示没有要求证据中存在决定性分支时主动追问。',
        '', '候选实现：检索完成后新增结构化澄清判断，区分通用流程、比较问题、用户条件缺失和资料不足/冲突。仅对当前唯一数值、设备/系统操作能力、互斥操作选择这三类请求考虑中断；必须引用用户原话，确认所有所问内容均被阻断且没有可直接回答的子问题，并提供缺失字段、具体问题和对应S编号的正文原文引文。结构或引文不合法时放弃该判断，继续原生成流程。引文存在不等于业务判断必然正确，任务分类和关系判断仍依赖模型并需检查误追问。普通生成提示保持原样。没有增加按题号、机型或报警码写的生产规则。若启用，普通作答增加一次模型判断；若需澄清，则省去原答案生成。报告已单列该步骤的请求和耗时归属。默认rag_semantic_clarification_enabled=False，当前不影响日常作答；实验显式开启。',
        '',f'原3题澄清结果：{summary["clarification"]}。',f'待审核10题诊断：{summary["holdout_pending"]}。',f'固定12个控制：{summary["controls"]}。',
        '', '## 验证边界与失败记录','','第一轮提示过于宽泛，误追问通用参数处理和资料数值冲突；第二轮收紧提示后，021仍只列分支不追问。两轮均在发现机制问题后停止，保留在同名及_v2目录，不作为完整60题验收。第三轮结构化判断正确澄清014/021/049，但误拦030一般采集安排、034仅凭数值能否确诊、043含定义的多部分问题，因此不采纳该版；其原始结果保存在_v3目录。最后一轮加入任务类型、用户原话及全部子问题被阻断的结构校验，完整验证结果见本目录，仍不是独立盲测。第三轮验证脚本的缓存provider字段名错误已修正，失败记录和重跑保留；最终轮不存在该脚本错误。',
        '',f'实际完成情况：{summary["checks"]}。完成44题结果记录，其中36题获得有效结构化判断，2题沿原检索流程直接返回，无需新判断；6题接口失败后回退，另16题远程调用失败。后22题不能计为澄清验收通过。待审核10题均受接口影响，未完成本轮验收；13项控制也均受接口影响，不能以早期版本的控制结果替代。',
        '', '所有计划题均尝试重新执行检索、条件及上下文阶段，符合原生成条件的题再执行新澄清判断；原始重排分按精确问题、块ID和正文哈希复用。已完成44对的检索至上下文阶段逐项相等。普通生成请求体与已保存请求完全相同时，复用原模型初稿以控制生成随机差异；否则实际生成。原50题baseline为上轮真实回答，不是同时重新采样，不能宣称本轮全部答案均重新生成。新标注对同一答案重新裁判，未改动检查项的记录复用原裁判；LLM裁判可能有波动，不能据单次结果声称统计显著。',
        '', 'v3历史版本、v4草案、实际系统结果分开保存。v4 JSON/JSONL同内容、加载器校验及全pending状态检查通过；正式加载器拒绝pending草案。历史50题仍为reviewed，默认测试集和实验功能未切换。',
        '', '本地检查：243项RAG测试通过，1项跳过；Python编译及git diff --check通过。测试覆盖有效/无效结构判断、缺失引用、非原文引用、已有可答子问题、默认关闭、总澄清开关、回退、澄清状态发布及新增请求/耗时统计。完整本地检查见local_verification.json。',
        '', '后续只需在恢复当前模型接口访问权限后补跑22道受影响题（含10道pending预验收）和13项控制。重跑工具支持--retry-failed，按错误或接口回退记录选择，不按题目是否通过选择；不要用早期版本的结果替代最终版本未完成的验证。','',
        '## 逐题复核清单','','| 题号 | 建议 | 原因 |','|---|---|---|']
    for c in audit['cases']:md.append(f'| {c["id"]} | {c["proposal"]} | {c["reason"]} |')
    md+=['','## 审核入口','',f'- [逐题可读报告]({(OUT/"report.html").resolve().as_posix()})',
        f'- [复核记录]({(FIXTURES/"rag_gold_v3_rubric_audit.json").resolve().as_posix()})',
        f'- [v4待审草案]({(FIXTURES/"rag_gold_v4_proposed_review.json").resolve().as_posix()})',
        f'- [统计]({(OUT/"comparison.json").resolve().as_posix()})']
    Path('docs/rag_rubric_clarification_validation_20261102.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
