"""Record source-bound review notes; never overwrite the frozen automatic verdicts."""
import json
from tools.finalize_unified_prompt_20261119 import read_rows
from tools.validate_unified_prompt_20261119 import OUT, write


NOTES = [
    ('rag-biz-019', 'frozen', '本轮未修复上游缺证据，仍计入原固定集合；不得将任何拒答措辞变化当成召回修复。'),
    ('rag-biz-014', 'unresolved_clarification', '新提示仍列出无标识/BM10条件值，但未向用户明确询问并要求回复盘型；不能说已解决澄清。数值条件未被简单合成唯一数值，与胡乱选定一个参数应区分。'),
    ('rag-biz-016', 'partial_answer_gain', '原提示首次只说资料不足；新提示首次给出可回答的光耦AD诊断、管路检查，并明确不能由现有资料区分板卡/供电。真实入口也改善。1988～2108/900、管路漏气分别存在于实际引用的原资料；这属于现有证据利用改善，不是召回改善。'),
    ('rag-biz-020', 'invalid_citation_and_judge_false_positive', '新提示第2、3轮均引用不存在的S37，其中第3轮仍被自动裁判判一致；真实入口也发生S37无效引用并被硬校验拒绝返回。按确定性引用合法性，该题只有第1轮具备可返回的有效引用，不能用自动2/3一致代替可用率。真实入口失败草稿保存在events.draft_generated，不为取得好答案而重试。'),
    ('rag-biz-021', 'confirmed_condition_action_error', '新提示首次将“本次试剂或校准品异常”写为同时换新试剂和校准品，并在原因未明时建议优先同时更换。S1第30张分别列出试剂、校准品分支；S8第11张是压缩表。两者均已在Context中，错误发生在生成时合并概括与细节。真实入口同样优先按概括表执行；不是本轮检索丢失。'),
    ('rag-biz-035', 'main_causes_completeness', '原提示首次漏掉明确的质控失控方向；新提示的已核查回答列出样本、试剂定标、质控失控，并解释室内质控的覆盖局限。主要原因在原PPT第50张及Context S1均存在。不能把该题某次成功扩展为全部重复都已稳定通过。'),
    ('rag-biz-033', 'conditional_inference_completeness', '原提示首次已给竞争法/夹心法对应及混匀排查，但漏掉这些条件共同提示发光值偏低这一明确结论；新提示首次补全。新回答扩展为8步、明显更长，自动一致不能代表已达到用户希望的简洁程度。条件推论的直接来源为Context S1，未引入额外检索收益。'),
    ('rag-biz-008', 'procedure_gain', '原提示首次直接给泵/光耦分支，漏掉先在软件中控制蠕动泵转动；新提示首次补上该前提。原PPT及PDF第22页实际支持这一操作。'),
    ('rag-biz-044', 'procedure_detail_gain', '原提示首次只概括相邻结果与3mV，省略H-STD第1次活化不计、第2次记录及后续比较；新提示首次按原电解质PPT第20张补全H-STD、L-STD、Cl Bias顺序及判定。收益是步骤完整性。'),
    ('rag-biz-006', 'citation_alignment_regression', '两组开始/停止/换瓶核心均覆盖。新提示额外句“持续进空气或大量气泡”仅引用S3；S3装机指南第49页支持持续进空气，实际“大量气泡”在S1的PPT第7张。属于局部引用绑定问题，不能写成吸液核心原理错误，也不能说库中没有该事实。'),
    ('rag-biz-034', 'judge_dispute', '两组均明确不能仅凭0.3 IU/mL确诊或治疗，并给复测范围和建议；来源判定两组都match，裁判只对新提示判behavior_mismatch。保留自动原判，单列行为判分争议，不能将这一自动降分直接算成确定业务退步。'),
    ('rag-biz-043', 'unsupported_frequency_claim_removed', '原提示首次将系统差异说为比基质效应更常见的解释，但所引S1/S2只列出可能原因，未提供频率比较。新提示首次删去更常见这一额外断言，并保留概念区别与不能仅凭换品牌即归因的结论。'),
    ('rag-row-holdout-001', 'nominal_range_gain', '原提示重复只给24V而漏正常范围，新提示已核查回答同时表达DC24V及22.8～25.2V。两种信息在原Context已有，不是切片或检索本轮新增。'),
    ('rag-row-holdout2-001', 'citation_scope_gain_not_retrieval_gain', '条码列表两组可正确，但原提示真实入口仅引单型号S5/S6；新提示真实入口引S2/S4/S5，高速及通用资料已在冻结Context中。收益是引用范围选择；本轮没有新增共识检索算法。'),
    ('evidence-flow-independent-02', 'pending_cross_object_extension', '原提示首次正确说明进样暂停后等待完成测试再加架。新提示多加放入区指示灯闪烁禁取放；所引S5原PPT第38张明确标题为存放通道/样本架状态，不能直接换成样本放入区。该题pending，不计入正式50题。'),
    ('evidence-flow-independent-11', 'pending_missing_condition_selected_anyway', '新提示开头直接认定该台BS-430按凯特备件处理，后面又承认序列号和模块标签未知。原第6张将凯特绑定到起始序列号，另一原第6张明确旧版本Medica；不能仅凭BS-430先选定。即使整体一致数未受影响，也保留此错误证据。该题pending。'),
    ('rag-row-holdout2-007', 'pending_wrong_numeric_scope', '原提示拒答不满足Gold，但保留了范围不明；新提示给37℃1140Ω，而Gold为约1115Ω。实际S1原热工PPT第7张确实有1140Ω但未明确BS-800，不能据此说该题已修复。原资料适用范围应另核，不改Gold制造通过。该题pending。'),
    ('vector-source-independent-019', 'pending_cross_operation_safety_extension', '核心浮子通断测法两组覆盖。新提示额外以S2证明浮子测量整体断电；实际S2原DOCX的断电句属于交流驱动板保险丝F1通断测量。这里讨论的是证据绑定，不能将该特定出处直接写成浮子传感器通用操作依据。该题pending。'),
    ('vector-source-independent-007', 'pending_citation_identifier_gain', '两组均正确给出FL12持续ON10秒；新提示额外引用S7的型号/代码信息，而原提示仅引S3被判uncertain。核心答案未变化，应称引用核验覆盖变化，不能把它当成新学会故障机制。该题pending。'),
    ('rag-biz-017', 'refusal_runtime_contract', '新提示首次拒答的业务内容被判一致，但实际文本为本次提供的资料中没有足够依据并附解释、没有引用。当前AnswerService只认可原固定拒答句，否则进入引用校验报错。由确定性真实入口模拟验证该协议差异，隔离判分一致不代表应用可返回。'),
    ('rag-biz-050', 'confirmed_business_scope_retained', '四类原因遵循用户已经明确的业务确认，本轮不改Gold、不再次要求用户确认。若自动裁判仅因实际引用中未写型号给uncertain，应作为技术范围核验限制单列，不将其偷换为四类业务原因错误。'),
]


def main():
    rows = read_rows('answers.jsonl')
    repair = {(r['id'],r['arm'],r['round']):r for r in read_rows('transport_recovery.jsonl')}
    rows = [repair.get((r['id'],r['arm'],r['round']),r) for r in rows]
    service = read_rows('service_entry.jsonl')
    records = []
    for k, category, finding in NOTES:
        first = [r for r in rows if r['id']==k and r['round']==1]
        records.append(dict(id=k, category=category, finding=finding,
            review_status=next((r['review_status'] for r in first),None),
            repeats={a:[dict(round=r['round'],verdict=r.get('judgement',{}).get('verdict'),citation_check=r.get('citation_check'))
                for r in sorted(rows,key=lambda r:r['round']) if r['id']==k and r['arm']==a] for a in ('baseline','unified')},
            actual_first_answers={r['arm']:r.get('answer') for r in first},
            service=[dict(arm=r['arm'],verdict=r.get('judgement',{}).get('verdict'),error=r.get('error')) for r in service if r['id']==k]))
    write(OUT/'manual_review.json',dict(status='completed_for_listed_records_only',
        scope='核对列出的实际回答、实际Context及原资料；不是250条回答全部人工审核。不覆盖自动分，不改Gold状态。',
        source_audits=['additional_source_audit.json','source_verification_reuse.json','refusal_contract_audit.json'],
        default_prompt='baseline',records=records))
    print('SOURCE_BOUND_REVIEW',len(records))


if __name__ == '__main__':
    main()
