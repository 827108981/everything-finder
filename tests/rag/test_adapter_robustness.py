from dataclasses import replace
import json
import pytest
from tests.rag.test_query_plan_adapter import plan, candidate, TABLE
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.identifier_evidence import prioritize,annotate
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.domain_adapter import DomainAdapter
from local_full_text_search.rag.evidence_contract import context_audit

@pytest.mark.parametrize('question',[
    '索引中断后如何恢复？','如何修复 CPU 异常？','ZZQX 是什么意思？',
    'P02 和 SV99 分别是什么？','文件传输速率 145 mL/min 怎么计算？'])
def test_non_domain_runtime_fields_and_prompt_remain_identical(question):
    old=QueryInterpreter().interpret(question); new=plan(question)
    assert not new.adapter_active
    for key in ('entities','constraints','lexical_terms','lexical_expression','semantic_query',
                'fallback_terms','lexical_variants','semantic_variants','analyzer_version'):
        assert getattr(old,key)==getattr(new,key)
    assert AnswerService._plan_summary(old)==AnswerService._plan_summary(new)
    items=[candidate('普通流程',1),candidate('报警 ZZQX：无关处理',2)]
    assert prioritize(new,items)==items
    a=ConstraintMatcher().evaluate(old,items)
    decisions=[i.constraint_decision for i in a]
    b=ConstraintMatcher(domain_adapter=DomainAdapter.load('chemiluminescence')).evaluate(new,items)
    assert decisions==[i.constraint_decision for i in b]

def test_scope_types_conditions_and_units_are_source_identified():
    p=plan('CL-2600i 软件版本 V2.3，当前校准品和试剂在定标后超过允许范围，不是样本检测，流量145±5 μL/s如何判断？')
    assert p.adapter_active
    kinds={x.type for x in p.scope_constraints}
    assert {'instrument_model','software_version','calibrator','reagent','stage',
            'time_condition','negative_condition','numeric_range'}<=kinds
    assert all(x.source and x.reason and 0<x.confidence<=1 for x in p.scope_constraints)
    assert not any(c.is_hard and c.value=='sample_test' for c in p.constraints)

def test_method_does_not_promote_explanation_over_original_workflow():
    p=plan('定标报 XYZ，什么时候沿用上次，什么时候更换本次试剂？')
    assert p.adapter_active and any(s.name=='method' for s in p.strategies)
    branch=candidate('定标：先判断偏差是否超过允许范围。上次存在问题时重新检查；本次存在问题时更换。',1)
    row=candidate(TABLE,2)
    detail=candidate('XYZ 排查流程：先确认原因，再处理，仍然异常时检查管路。',3)
    assert prioritize(p,[branch,row,detail])==[branch,row,detail]
    assert detail.query_plan_diagnostics['exact_representative']
    ConstraintMatcher().evaluate(p,[branch,row,detail])
    r=EvidencePolicy(max_per_document=2).evaluate(RagQuery(p.original_text),[branch,row,detail],plan=p)
    assert branch in r.accepted and detail in r.accepted
    assert p.method_branches and not p.missing_fields

def test_exact_representatives_have_global_cap_and_dont_expand_duplicates():
    p=plan('报 XYZ、报警 ABC、报警 DEF 是什么意思？')
    rows=[candidate('报警 '+code+'：故障',i) for i,code in enumerate(['XYZ','ABC','DEF','XYZ','ABC'],1)]
    prioritize(p,rows)
    assert sum(bool(i.query_plan_diagnostics['exact_representative']) for i in rows)==2
    assert sum(bool(i.query_plan_diagnostics['exact_representative']) for i in rows if 'XYZ' in i.chunk.text)==1

def test_source_scope_and_table_header_remain_bound_to_original_offsets():
    p=plan('免疫仪报 XYZ 是什么意思？怎么处理？')
    i=candidate('仅适用于CL-2600i，软件版本 V2.3\n'+TABLE)
    annotate(p,i); ctx=ContextBuilder().build([i])
    entry=ctx.entries[0]
    assert 'CL-2600i' in entry['included_text'] and 'V2.3' in entry['included_text']
    assert 'ABC' not in entry['included_text']
    assert all(i.chunk.text[s['start']:s['end']] in entry['included_text'] for s in entry['source_spans'])
    audit=context_audit(p,ctx.entries)
    assert all(x['retained'] for x in audit['scope_observations'])
    assert audit['semantic_completeness']=='unverified'
    assert '不能省略资料的型号限制' in AnswerService._plan_summary(p)

def test_complete_row_is_not_silently_cut_at_budget():
    p=plan('报 XYZ 是什么意思？怎么处理？')
    i=candidate(TABLE.replace('检查温控后重新检测','检查温控后'+'重新检测'*100))
    annotate(p,i); ctx=ContextBuilder().build([i],max_chars=256)
    assert not ctx.entries
    assert ctx.excluded_entries[0]['reason_code']=='complete_row_exceeds_budget'

def test_single_object_is_not_automatically_ambiguous():
    p=plan('免疫仪注射器什么时候更换？')
    assert p.adapter_active and not any(a.name=='part_whole' for a in p.ambiguities)
    assert '不能据此给整套组件更换周期' in AnswerService._plan_summary(p)
    assert '66000' not in str(p) and '040-000765' not in str(p)

def test_no_domain_configuration_has_no_activation_even_for_alarm(tmp_path):
    p=QueryInterpreter(enhanced=True,domain_config=tmp_path/'missing.json').interpret('定标报 XYZ 如何处理？')
    assert not p.adapter_active and p.domain_config=='generic'
    assert p.unknown_entities

def test_invalid_policy_fails_closed(tmp_path):
    p=tmp_path/'bad.json';p.write_text(json.dumps({'policy':{'max_exact_representatives':100}}))
    assert DomainAdapter.load(p).identity=='generic'

def test_negated_model_is_not_exact_protected_even_if_domain_is_active():
    p=plan('免疫仪，不是CL-8000i，型号不清楚，怎么排查？')
    assert p.adapter_active
    items=[candidate('CL-8000i 排查步骤')]
    prioritize(p,items)
    assert not items[0].query_plan_diagnostics['exact_representative']

def test_scope_conflict_requires_explicit_restriction_not_merely_another_model():
    p=plan('CL-2600i 软件版本 V2.3 条码规则是什么？')
    related=candidate('与CL-8000i比较：部分参数不同。')
    restricted=candidate('仅适用于CL-8000i\n条码规则',2)
    version=candidate('软件版本：V3.0\n条码规则',3)
    ConstraintMatcher(domain_adapter=DomainAdapter.load('chemiluminescence')).evaluate(p,[related,restricted])
    assert related.constraint_decision.accepted
    assert not restricted.constraint_decision.accepted
    assert annotate(p,restricted)['explicit_scope_conflicts']
    assert annotate(p,version)['explicit_scope_conflicts']

@pytest.mark.parametrize('phrase',['先检查哪些环节','怎么处理','如何操作','为什么异常'])
def test_handling_without_definition_is_method(phrase):
    p=plan('CL-3456i '+phrase+'？')
    assert p.adapter_active and any(s.name=='method' for s in p.strategies)

def test_alarm_definition_plus_handling_still_uses_complete_row():
    p=plan('报 XYZ 是什么意思？怎么处理？')
    assert not any(s.name=='method' for s in p.strategies)

def test_excluded_table_row_cannot_leak_objects_through_scope_note():
    p=plan('报 XYZ 是什么意思？怎么处理？')
    i=candidate(TABLE.replace('压力过低','校准品不足').replace('检查泵','更换注射器密封圈'))
    annotate(p,i)
    ctx=ContextBuilder().build([i])
    assert '校准品' not in ctx.text and '密封圈' not in ctx.text and '注射器' not in ctx.text
    assert not any(o['text'] in ('校准品','注射器','密封圈') for o in ctx.entries[0]['query_plan_diagnostics']['scope_observations'])
