from tools.validate_material_repairs import bind_row,select_preserving
from tests.rag.test_evidence_selection_experiment import candidate
from tests.rag.test_material_steps import supported
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.query_interpreter import QueryInterpreter

def item(cid,text='原文',doc='doc'):
    x=supported(candidate(cid,text,doc=doc));x.lexical_rank=1;return x

def test_null_does_not_shift_values_or_drop_units():
    b=bind_row(['机型','流量 μL/s','备注'],['A','140±5',''])
    assert b[1]['header']=='流量 μL/s' and b[1]['value']=='140±5'
    assert b[2]['column']==3 and b[2]['value'] is None
    assert bind_row(['值'],['0'])[0]['value']=='0'

def test_cannot_evict_baseline_at_full_budget():
    original=[item(str(n),f'事实{n}') for n in range(8)]
    assert select_preserving(original,[item('new','新增资料')],EvidencePolicy())[0]==original

def test_soft_quota_only_fills_free_slot():
    original=[item(str(n),f'事实{n}') for n in range(3)]
    new=item('new','另一必要事实')
    selected,trace=select_preserving(original,[new],EvidencePolicy())
    assert selected==original+[new] and trace[0]['document_cap_exception']

def test_unknown_or_conflicting_conditions_not_added():
    plan=QueryInterpreter().interpret('所有项目样本检测发光值偏低')
    xs=[item('unknown','检查仪器'),item('conflict','单个项目样本检测发光值偏低')]
    xs=list(ConstraintMatcher().evaluate(plan,xs))
    assert select_preserving([],xs,EvidencePolicy())[0]==[]

def test_same_document_body_copy_not_added():
    a=item('a','相同事实');b=item('b','相同事实')
    assert select_preserving([a],[b],EvidencePolicy())[0]==[a]
