"""Source-bound notes for the approved four-rule candidate; retain frozen scores."""
from tools.finalize_minimal_four_prompt_20261119 import read_rows
from tools.validate_minimal_four_prompt_20261119 import OUT, write


NOTES = [
    ('rag-biz-019', 'frozen', '本轮上游缺证据问题保持冻结，保留在固定正式集合，不用提示措辞变化声称已修复。'),
    ('rag-biz-014', 'clarification_judge_variance', '精简提示三轮均给无标识/BM10分支值，没有明确请求用户提供并回复盘型；前两轮判需澄清，第三轮却判一致。第三轮“先确认”与另两轮行为相近；保留自动原判并记争议，不能据1/3一致宣称追问能力改善。真实入口仍判需澄清。'),
    ('rag-biz-015', 'partial_answer_gain', '首次原提示若整题拒答，丢失可答部分；精简提示给出CL-6000i和CL-8000i各自入口、CL-8000i AD范围，并明确不能把CL-1000i数值直接套给CL-6000i。S1/S2/S6实际Context与原页支持这些边界。'),
    ('rag-biz-016', 'partial_answer_gain', '原提示三轮整题资料不足，精简提示及真实入口能利用已有光耦诊断与液路依据作部分回答，并说明不能由此判定板卡/供电。收益是使用已有资料，不是新增召回。仍须核对全部三轮及额外操作的依据。'),
    ('rag-biz-020', 'runtime_invalid_citation', '精简提示真实入口生成S37，实际请求只有S1至S20、原文没有S37；程序硬校验阻止返回。属于生成编号错误，不是检索缺少S37。保留草稿，不为成功而重新采样。隔离三轮的自动一致与确定性引用检查分别统计。'),
    ('rag-biz-021', 'branch_not_stable', '精简提示第2/3轮仍把本次异常分支写成同时更换试剂和校准品；S1第30张将两类问题分别处理，S8第11张为概括表。第1轮末尾补了分别处理，但前文仍要求同时更换；自动判一致不能代表表达无歧义。真实入口亦错误。'),
    ('rag-biz-035', 'extra_claim_binding', '首次精简提示已列样本、试剂定标、质控失控，却加“质控规则、靶值设置掩盖失控”等所引S1/S2没有说明的细节，被判错。不能说漏了三类主要原因；问题是无依据扩展。原PPT第49/50张与实际Context用于复核。'),
    ('rag-biz-034', 'behavior_judge_dispute', '两组都明确不能仅凭0.3 IU/mL确诊或决定治疗，且引用相同复测范围和复测规则。原提示被判回答而非拒答，精简提示被判拒答一致，主要是行为标签判分口径差异，不能把这1题自动增分算成明确业务修复。保留原判与两组原文。'),
    ('rag-biz-039', 'extra_claim_binding', '核心双波长意义与向厂家确认两组覆盖。精简提示额外引用S5解释主波长作用及副波长仅用于背景补偿；实际S5只写副波长通常平直不反应。应将新增说明缺来源单列，不能称核心双波长问题答错。'),
    ('rag-biz-045', 'conditional_claim_gain', '原提示首次把校准失败/斜率异常的可能原因直接扩展为钠结果低的原因；精简提示先限定为斜率异常不必然等于损坏，再列对应原因。核心不能由单一症状直接确定电极和物料码，两组都表达。收益是条件绑定，编码还须按模块/区域边界理解。'),
    ('rag-row-holdout-001', 'range_gain_not_retained', '精简提示三轮及真实入口均只给DC24V，未补22.8～25.2V；实际Context S2有J7接口供电范围，S5有P05 DC24V。不能把上轮统一提示3/3成功迁移到本轮，不能声称四项精简已保留参数范围收益。'),
    ('rag-row-holdout2-001', 'citation_scope_gain', '精简提示三轮及真实入口均引用多个来源支持免疫平台条码，原提示本轮多引用单型号S5/S6，导致平台范围证据不足。码制列表原本就能正确，收益是引用范围，不是新检索/共识算法。'),
    ('evidence-flow-independent-03', 'pending_extra_claim_removed', '原提示首次颜色核心正确但补了防护罩呼吸灯的停机推断；精简提示只答状态按钮红色严重报警、黄色警告报警，引用S3。属于减少跨对象扩展，pending不计正式50题。'),
    ('evidence-flow-independent-05', 'pending_reverse_inference', '原提示从模块类型推导原理；精简提示却说标注直接法的就是Medica，反转了单向关系。原PPT第6张及实际S3同时列Medica和凯特为直接法，不能据直接法唯一识别Medica。这是已见实质错误，pending题不计正式50题，但仍用于回归风险判断。'),
    ('evidence-flow-independent-07', 'replacement_judge_dispute', '两组均回答注射器密封圈66000次和040-000765-00，均未显式加达到寿命需更换；基线被判覆盖，精简提示被判缺失同一检查项。对象、寿命和编码无实质差异，不能将此自动掉分称为密封圈对象退化。原PPT第12/164张与实际S1/S5已核实，保留原判等待业务裁决，不改Gold。'),
    ('evidence-flow-independent-11', 'pending_absence_claim', '两组均未仅凭BS-430选定模块，并给序列号/旧版条件。精简提示却说资料没有其他识别方法；实际引用不支持全资料不存在的断言。另有“先确认/查记录”是否等于明确向用户追问的判分差异，保留原判，不将其说成错选凯特。'),
    ('vector-source-independent-011', 'pending_extra_citation_claim', '两组维护/维修/装机WATER编号核心均正确。精简提示多说资料仅给维护和装机具体示例，却仅引用S5，装机示例在S6/S8。属于额外陈述及引用绑定问题，pending不计正式50题。'),
    ('vector-source-independent-007', 'pending_citation_scope_gain', '两组均正确说FL12的ON信号连续10秒；基线只引故障DOCX S3，其实际正文没有BS-2800M型号，编号校验判不确定；精简提示补引S7的BS-2800M/215050602/主真空容器内浮子FL12行，补足实际引用范围。收益是引用绑定，并非报警机制内容从错变对。原DOCX及原XLSX第34行均已核实。'),
]


def main():
    original = read_rows('answers.jsonl')
    recovered = {(r['id'],r['arm'],r['round']):r for r in read_rows('transport_recovery.jsonl')}
    rows = [recovered.get((r['id'],r['arm'],r['round']),r) for r in original]
    service = read_rows('service_entry.jsonl')
    records = []
    for k, category, finding in NOTES:
        first = [r for r in rows if r['id']==k and r['round']==1]
        records.append(dict(id=k,category=category,finding=finding,
            review_status=next((r['review_status'] for r in first),None),
            repeats={a:[dict(round=r['round'],verdict=r.get('judgement',{}).get('verdict'),citation_check=r.get('citation_check'))
                for r in sorted(rows,key=lambda r:r['round']) if r['id']==k and r['arm']==a] for a in ('baseline','minimal_four')},
            actual_first_answers={r['arm']:r.get('answer') for r in first},
            service=[dict(arm=r['arm'],verdict=r.get('judgement',{}).get('verdict'),error=r.get('error'))for r in service if r['id']==k]))
    write(OUT/'manual_review.json',dict(status='completed_for_listed_records_only',
        scope='Only listed observed answers and actual Context/source bindings reviewed; not all250 independently human reviewed. Gold and automatic verdicts unchanged.',
        source_audits=['additional_source_reuse.json','source_verification_reuse.json','additional_source_audit.json','refusal_contract_audit.json'],records=records))
    print('REVIEW',len(records))


if __name__=='__main__':main()
