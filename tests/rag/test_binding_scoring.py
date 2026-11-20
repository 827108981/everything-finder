import json
from dataclasses import replace
import pytest
from tests.rag.test_answer_judge import _case,_payload
from tests.rag.test_stage_scoring import _row
from local_full_text_search.rag.evaluation.answer_judge import _parse_judgement,_request_prompt
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.reporting.matching import match_quote

def test_object_replacement_overrides_keyword_coverage():
    p=json.loads(_payload());p['object_binding']={'status':'mismatch','reason':'将所问部件换成另一部件'}
    r=_parse_judgement(json.dumps(p),_case(),'mock',require_binding=True)
    assert r['verdict']=='incorrect' and 'eval_object_binding' in r['findings']

def test_wrong_source_binding_overrides_correct_literal_answer():
    p=json.loads(_payload());p['source_binding']={'status':'mismatch','reason':'引用仅支持另一版本'}
    assert _parse_judgement(json.dumps(p),_case(),'mock',require_binding=True,has_context=True)['verdict']=='incorrect'

def test_ambiguous_binding_remains_uncertain():
    p=json.loads(_payload());p['object_binding']={'status':'uncertain','reason':'别名关系没有依据'}
    assert _parse_judgement(json.dumps(p),_case(),'mock',require_binding=True)['verdict']=='uncertain'

def test_missing_actual_context_never_proves_citation():
    p=json.loads(_payload());p['source_binding']={'status':'match','reason':'猜测'}
    with pytest.raises(ValueError):_parse_judgement(json.dumps(p),_case(),'mock',require_binding=True)


def test_unavailable_source_cannot_mark_factual_answer_consistent():
    p=json.loads(_payload());p['source_binding']={'status':'unavailable','reason':'未保存实际上下文'}
    r=_parse_judgement(json.dumps(p),_case(),'mock',require_binding=True,answer_text='回答')
    assert r['verdict']=='uncertain' and r['source_binding']['status']=='unavailable'

def test_missing_binding_fields_rejected_in_live_mode():
    p=json.loads(_payload());p.pop('object_binding')
    with pytest.raises(ValueError):_parse_judgement(json.dumps(p),_case(),'mock',require_binding=True)

def test_prompt_carries_actual_context_and_object_conditions():
    value=_request_prompt(_case(),{'text':'answer','evidence_context':'[S1]真实原文'})
    assert 'actual_evidence_context' in value and '真实原文' in value and 'constraints' in value

def setup_mapping():
    c=replace(_case(),gold_evidence=({'file':'机器.pdf','page':4,'quote':'正确行'},),hard_negatives=({'file':'机器.pdf','page':4,'quote':'另一部件错误行'},))
    return map_case_evidence(c,[_row('p','正确行\n另一部件错误行','机器.pdf',0)])

def test_formal_negative_requires_quote_in_actual_context():
    m=setup_mapping();c={'chunk_id':'row','member_chunk_ids':['p'],'text':'正确行\n另一部件错误行','included_text':'正确行'}
    r=score_stage('context',[c],m)
    assert r['hard_negative_found']==0 and r['hard_negative_page_only_ids']
    c['included_text']='另一部件错误行';assert score_stage('context',[c],m)['hard_negative_found']==1

def test_formal_missing_actual_text_is_diagnostic_only():
    r=score_stage('context',[{'chunk_id':'p','text':'另一部件错误行'}],setup_mapping())
    assert r['hard_negative_found']==0 and r['hard_negative_unresolved_ids']

def test_unrelated_text_does_not_resolve_missing_source_text():
    r=score_stage('context',[{'chunk_id':'p'}, {'chunk_id':'other','included_text':'别的正文'}],setup_mapping())
    assert r['hard_negative_found']==0 and r['hard_negative_unresolved_ids']

def test_positive_context_requires_actual_text_too():
    r=score_stage('context',[{'chunk_id':'p','text':'正确行'}],setup_mapping())
    assert r['gold_evidence_found']==0

def test_positive_context_only_counts_included_row():
    r=score_stage('context',[{'chunk_id':'p','text':'正确行\n另一部件错误行','included_text':'另一部件错误行'}],setup_mapping())
    assert r['gold_evidence_found']==0 and r['hard_negative_found']==1

def test_negative_partial_report_match_is_not_countable():
    r=match_quote({'filename':'机器.pdf','page_number':4,'text':'另一部件错误'}, {'file':'机器.pdf','page':4,'quote':'另一部件错误行'},kind='hard_negative',evidence_id='N')
    assert not r.is_countable and r.status=='partial'

def test_shared_value_projection_is_unresolved():
    c=replace(_case(),gold_evidence=({'file':'机器.pdf','page':4,'quote':'P05'},),hard_negatives=({'file':'机器.pdf','page':4,'quote':'DC24V'},))
    m=map_case_evidence(c,[_row('p','P05 DC24V\nP11 DC24V','机器.pdf',0)])
    r=score_stage('context',[{'chunk_id':'row','member_chunk_ids':['p'],'included_text':'P05 DC24V'}],m)
    assert r['hard_negative_found']==0 and r['hard_negative_unresolved_ids']

def test_explicit_part_number_cannot_pass_with_other_number_source():
    c=replace(_case(),question='ZX81电机在哪里？')
    p=json.loads(_payload(case=c));p['source_binding']={'status':'match','reason':'功能相同'}
    result=_parse_judgement(json.dumps(p),c,'mock',require_binding=True,has_context=True,
                           answer_text='对应X81，位于返回通道。[S1]',actual_context='[S1]\nX81位于返回通道')
    assert result['verdict']=='uncertain' and result['identifier_check']['status']=='review'

def test_number_in_actual_citation_passes_guard():
    c=replace(_case(),question='ZX81电机在哪里？');p=json.loads(_payload(case=c));p['source_binding']={'status':'match','reason':'支持'}
    result=_parse_judgement(json.dumps(p),c,'mock',require_binding=True,has_context=True,
                           answer_text='ZX81位于返回通道。[S1]',actual_context='[S1]\nZX81位于返回通道')
    assert result['verdict']=='consistent'

def test_unspoken_fact_cannot_be_source_contradiction():
    c=replace(_case(),question='电机在哪里？');p=json.loads(_payload(case=c));p['source_binding']={'status':'mismatch','reason':'没讲完整维护过程'}
    result=_parse_judgement(json.dumps(p),c,'mock',require_binding=True,has_context=True,
                           answer_text='位于返回通道。[S1]',actual_context='[S1]\n位于返回通道')
    assert result['verdict']=='uncertain' and result['source_binding']['status']=='uncertain'

def test_source_model_dispute_not_unproved_answer_object_error():
    c=replace(_case(),question='设备ZX81的电压是多少？');p=json.loads(_payload(case=c))
    p['object_binding']={'status':'mismatch','reason':'引用型号后缀不同'};p['source_binding']={'status':'uncertain','reason':'型号关系未知'}
    r=_parse_judgement(json.dumps(p),c,'mock',require_binding=True,has_context=True,
                        answer_text='ZX81的电压是24V。[S1]',actual_context='[S1]\nZX81电压24V')
    assert r['verdict']=='uncertain'

def test_proved_answer_object_replacement_is_incorrect():
    c=replace(_case(),question='ZX81在哪？');p=json.loads(_payload(case=c))
    p['object_binding']={'status':'mismatch','reason':'明确替换','question_object':'ZX81','answer_object':'X81'}
    p['source_binding']={'status':'match','reason':'支持X81'}
    r=_parse_judgement(json.dumps(p),c,'mock',require_binding=True,has_context=True,
                        answer_text='对应的是X81，位于返回通道。[S1]',actual_context='[S1]\nX81位于返回通道')
    assert r['verdict']=='incorrect'
