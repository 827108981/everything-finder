"""Assemble the final audit trail without silently replacing failed runs."""
import json
from collections import Counter
from pathlib import Path
from tools.rag_eval_common import ROOT,read,lines,write,sha
from tools.validate_business_repair import native
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings

OUT=native.OUT

def main():
    metrics=read(OUT/'final_report_metrics.json')
    focus=lines(OUT/'verified_native_answers.jsonl')
    ablation=lines(OUT/'final_ablation.jsonl')
    assert len(ablation)==24,'All 8 x 3 ablation cells must finish before final report'
    assert len(focus)==8
    first=read(OUT/'report_metrics.json') if (OUT/'report_metrics.json').exists() else read(OUT/'metrics.json')
    settings=read(OUT/'settings_guard.json')
    settings['unchanged']=sha(Path(SETTINGS_PATH))==settings['sha256']
    settings['default_still_disabled']=AppSettings().rag_query_plan_enabled is False
    hashes={str(p):sha(p) for p in native.FILES}
    final_manifest=read(OUT/'verified_all_manifest.json')
    assert hashes==final_manifest['dataset_hashes']
    history=read(OUT/'dataset_validation.json')['historical_hashes']
    historical_unchanged=all(sha(Path(p))==h for p,h in history.items())
    inactive=metrics['inactive_equivalence']
    audits=[]
    for r in ablation:
        raw=r.get('claim_audit_raw','')
        try:
            raw=raw.strip().removeprefix('```json').removesuffix('```').strip();audit=json.loads(raw)
        except (ValueError,TypeError): audit={'parse_error':True,'raw':raw}
        for claim in audit.get('extra_claims',[]):
            quote=claim.get('source_quote','');quote=''.join(quote.split())
            claim['quote_verbatim_verified']=bool(quote and quote in ''.join(r['context'].split()))
            if claim.get('supported')=='yes' and not claim['quote_verbatim_verified']:
                claim['audit_limitation']='Judge supplied a shortened/paraphrased quote; requires review, not an exact-quote proof.'
        audits.append({'id':r['id'],'arm':r['arm'],**audit})
    write(OUT/'extra_claim_audits.json',audits)
    ablation_matrix={cid:{r['arm']:{'verdict':r.get('judgement',{}).get('verdict'),'findings':r.get('judgement',{}).get('findings'),'error':r.get('error')} for r in ablation if r['id']==cid} for cid in dict.fromkeys(r['id'] for r in ablation)}
    write(OUT/'ablation_matrix.json',ablation_matrix)
    focused=[]
    for r in focus:
        focused.append({'id':r['id'],'question':r['question'],'status':r.get('result',{}).get('status'),
            'verdict':r.get('judgement',{}).get('verdict'),'error':r.get('error') or r.get('judge_error'),
            'gold_found':r['scores']['context']['gold_evidence_found'],
            'gold_total':r['scores']['context']['gold_evidence_total_declared'],
            'answer':r.get('result',{}).get('text','')})
    gates={'settings':settings,'history_preserved':historical_unchanged,'dataset_drift':False,
        'inactive_equivalence':inactive,'main49_regression42_total91':metrics['total']==91,
        'retired_case_in_results':any(r['id']=='rag-biz-040' for r in focus),
        'native_focus':focused,'ablation_count':len(ablation),'first_round_losses_preserved':first['answer_losses'],
        'final_answer_losses':metrics['answer_losses'],'default_enable_recommendation':False,
        'earlier_focus_failures':[{'id':r['id'],'verdict':r.get('judgement',{}).get('verdict'),'error':r.get('error')} for r in lines(OUT/'final_native_answers.jsonl') if r.get('error') or r.get('judgement',{}).get('verdict')!='consistent']}
    write(OUT/'final_acceptance.json',gates)
    lines_out=['# 本轮开发与验证最终结论\n',
        '已实际修改代码、Gold、评测和应用当前评测入口。默认Adapter仍关闭；用户持久化设置未修改。不是仅提交分析。\n',
        '## 数据集\n','新主集49题（47 reviewed、007 disputed、024 pending）＋回归42题＝91题。040从本轮所有活动入口和分母删除，历史版本保留。本轮明确确认的条码、电压、密封圈已更新；其他回归题审核状态不变。\n',
        '电压题使用手册封面准确型号BS-2800M；没有推断BS-2800别名。DC24V是标称值，22.8～25.2V是正常供电范围。原文9条引文已双重核验，见source_audit.json。\n',
        '## 最终重点题原生完整问答\n','|题目|完整引文|判分/错误|\n|---|---:|---|\n']
    for r in focused:lines_out.append(f"|{r['id']}|{r['gold_found']}/{r['gold_total']}|{r['verdict'] or r['error']}|\n")
    lines_out+=['\n## 全量开关对比\n',f"Context完整Gold增加：{metrics['gains']}；减少：{metrics['losses']}。\n",
        f"未激活Adapter的{inactive['count']}题，候选/分数/Context比较差异：{len(inactive['failures'])}。\n",
        '全量答案比较采用同一新版题面：A为关闭的原生下游冻结轨迹；B为开启的原生下游冻结轨迹。均为真实LLM调用。最后一次仅调整来源选择后，对问题、QueryPlan、Context均逐字相同的记录复用此前答案及原判分，变化的记录重新运行；不是把重复成功样本替换失败。详见verified_answer_provenance.json。另有八题从真实检索入口运行完整问答，以上表格保留所有失败。\n',
        '|集合|题数|关闭|最终开启|\n|---|---:|---|---|\n']
    for group,d in metrics['cohorts'].items():lines_out.append(f"|{group}|{d['count']}|{d['arms']['off']['answers']}|{d['arms']['on']['answers']}|\n")
    lines_out+=['\n|集合|Context完整Gold题数（关闭→开启）|Context平均证据Recall|Context MRR|\n|---|---:|---:|---:|\n']
    for group,d in metrics['cohorts'].items():
        a=d['arms']['off']['stages']['context'];b=d['arms']['on']['stages']['context']
        lines_out.append(f"|{group}|{a['all_gold_retained']}→{b['all_gold_retained']} / {a['eligible_cases']}|{a['mean_evidence_recall']:.4f}→{b['mean_evidence_recall']:.4f}|{a['MRR']:.4f}→{b['MRR']:.4f}|\n")
    lines_out.append('\n回归诊断组虽未丢失完整Gold证据，但Context MRR从0.7092降至0.6456；代表证据保护改变了前排顺序，不能说排序完全无退化。所有这些pending题仅作诊断，不进入正式总分。\n')
    lines_out += [f"\n答案改善题：{metrics['answer_gains']}。\n",f"答案退化题：{metrics['answer_losses']}。\n",f"其他无法直接判优劣的变化：{metrics['uncertain_changes']}。\n",
        f"证据改善而答案未通过：{metrics['retrieval_gain_answer_not_improved']}。\n",
        f"相同Context未激活Adapter时的生成/裁判波动（不能归因算法）：{metrics['same_context_answer_variation']}。\n",
        '## 原因和修复层\n',
        '- 条码：原链路主要保留单型号说明，缺少平台资料。新增结构化字段检索、跨文档一致性比较、限量代表保护和实际Context复核；高速/中低速/全线共3个来源支持同一列表，不以简单并集冒充共识。\n',
        '- 019：场景在13张，原因及操作在14张。新增可追溯相邻场景连接，并识别“大部分发光项目”的范围；保留两个真实源块与顺序，不将其他DEV场景套用。\n',
        '- 050：第一轮完整引文已召回但答案仍退化。根因是通用原因被范围警告误伤，同时型号命中的其他故障流程抢占答案。最终保护精确诊断标题的原因段，并限制“只问原因”的生成目标；首轮失败完整保留。\n',
        '- 电压：修正准确型号和双证据Gold；未知部件编号可参与参数检索的有限保护，不成为硬过滤。固定Context消融曾出现同时引用BS-2800材料的错误，未隐藏。\n',
        '- 密封圈：修正题目对象，取消明确子件名称被自动判成整件歧义的逻辑。\n',
        '- 计分：合并来源仍需保留全部来源ID，并逐条检查实际引文；不再漏计同正文多来源，也不能借同页另一个内容得分。\n',
        '## 测试及可复现入口\n',
        '```powershell\n.venv/Scripts/python.exe -m pytest tests/rag tests/test_settings_service.py -q\n.venv/Scripts/python.exe -m tools.validate_business_repair --arm off_all --disabled\n.venv/Scripts/python.exe -m tools.validate_business_repair --arm verified_all\n.venv/Scripts/python.exe -m tools.answer_business_repair --candidate verified_all --arms B --output verified_changed_B --workers 4\n.venv/Scripts/python.exe -m tools.ablate_business_repair --candidate verified_all --output final_ablation\n```\n',
        '已存在的结果不可同名覆盖；复现时换arm/output名称。每轮manifest保存代码、Gold哈希和真实bundle。单元与兼容测试412 passed、1 skipped。固定Context消融进程有一次启动级异常退出（尚未产生结果），重新启动后继续，不按答案质量重试。\n',
        '## 采用建议及剩余问题\n',
        '- 可以采用：已确认的新版业务题、040退休、审阅/计分来源修复、默认关闭及活动集合分离。\n',
        '- 需要灰度：跨文档结构化共识、相邻场景连接、参数编号提示及诊断原因保护。当前仅支持可直接核实的结构，不代表解决任意表格或所有领域。\n',
        '- 暂不默认开启：单次LLM判分仍会波动，来源引用和额外主张需逐条看；任何最终原生调用失败也不能用另一轮成功覆盖。不能仅凭总一致数增加宣布全面成功。\n',
        '- 业务待确认：原pending/disputed题仍未正式审核；本轮已确认五题无需重复确认。BS-2800和BS-2800M是否可建立业务别名仍无证明，因此未建立别名。\n',
        '- 尚未彻底解决：019与条码在另一次冻结Context生成中核心事实齐全，但分别增加未充分绑定到DEV场景的一般原因、未限定CL-2600i型号的校验规则；严判仍失败。不能用原生重点题成功掩盖这种不稳定。下一轮最小修复应检查新增主张的来源与适用范围，优先减少用户未问的扩展，不再改Gold迎合答案。\n',
        '- 全量变化中的rag-acc-002在未激活且输入相同下发生引用绑定波动，不能证明Adapter造成退化；independent-11未明确发出澄清请求仍需修复。020在最终原生运行漏提试剂/校准品失效、污染，但其他运行通过，说明方法型答案完整性尚不稳定。\n',
        '逐题实际回答、缺失项、错误主张、全部阶段候选及排除原因见 final_report_focus_full_traces.json、verified_native_answers.jsonl。final_native_answers.jsonl保留前一次020无效引用S20、050型号误写CL-800i、条码范围收缩的失败；不能删除或当作通过。固定Context和旧/新Context消融见final_ablation.jsonl；额外主张及原文核验限制见extra_claim_audits.json。\n']
    lines_out+=['\n## 严格固定提示/固定Context消融\n','旧/新Context两臂使用完全相同的领域提示（排除动态共识提示）；第三臂固定新Context使用通用提示。动态共识提示收益由原生完整问答另行检验，不能混在纯Context消融中。\n','|题目|旧Context＋固定提示|新Context＋相同提示|新Context＋通用提示|\n|---|---|---|---|\n']
    for cid,arms in ablation_matrix.items():
        lines_out.append('|'+cid+'|'+ '|'.join(str(arms[a]['verdict'] or arms[a]['error']) for a in ('old_context_fixed_prompt','new_context_fixed_prompt','new_context_generic_prompt'))+'|\n')
    (OUT/'conclusion.md').write_text('\n'.join(lines_out),encoding='utf8')
    print(json.dumps({'native_focus':[{k:v for k,v in r.items() if k not in ('answer','question')} for r in focused],'settings_unchanged':settings['unchanged'],'history_preserved':historical_unchanged},ensure_ascii=False))

if __name__=='__main__':main()
