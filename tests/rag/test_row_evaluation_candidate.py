from dataclasses import replace
from tests.rag.test_stage_scoring import _case,_row
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from tools.row_evaluation_candidate import negative_span_audit

def setup():
    positive='CSDB底物注入时检测到气泡，灌注后重新定标'
    negative='DCF光度计错误，重启仪器和软件'
    c=replace(_case(positive),hard_negatives=({'file':'机器.pdf','page':4,'quote':negative},))
    parent=_row('p',positive+'\n'+negative,'机器.pdf',0)
    return positive,negative,map_case_evidence(c,[parent])

def test_same_page_other_row_not_negative_recall():
    pos,neg,m=setup();row={'chunk_id':'rowA','member_chunk_ids':['p'],'included_text':pos}
    assert score_stage('context',[row],m)['hard_negative_found']==0
    assert score_stage('context',[row],m)['hard_negative_page_only_ids']
    assert negative_span_audit([row],m)['negative_quote_hits']==[]

def test_actual_negative_row_is_still_detected():
    pos,neg,m=setup();assert len(negative_span_audit([{'chunk_id':'rowB','member_chunk_ids':['p'],'included_text':neg}],m)['negative_quote_hits'])==1

def test_complete_parent_still_exposes_negative():
    pos,neg,m=setup();assert len(negative_span_audit([{'chunk_id':'p','included_text':pos+'\n'+neg}],m)['negative_quote_hits'])==1

def test_omitted_or_truncated_text_is_not_complete_recall():
    pos,neg,m=setup();assert negative_span_audit([{'chunk_id':'p','included_text':'DCF光度计错误'}],m)['negative_quote_hits']==[]

def test_same_words_wrong_source_do_not_count():
    pos,neg,m=setup();assert negative_span_audit([{'chunk_id':'other','included_text':neg}],m)['negative_quote_hits']==[]

def test_changed_numeric_fact_does_not_match():
    c=replace(_case('目标事实'),hard_negatives=({'file':'机器.pdf','page':4,'quote':'流量145±5 μL/s'},))
    source=_row('p','目标事实\n流量145±5 μL/s','机器.pdf',0);m=map_case_evidence(c,[source])
    assert negative_span_audit([{'chunk_id':'p','included_text':'流量140±5 μL/s'}],m)['negative_quote_hits']==[]

def test_missing_actual_text_is_unknown_not_id_based_success():
    pos,neg,m=setup();audit=negative_span_audit([{'chunk_id':'p'}],m)
    assert audit['negative_quote_hits']==[] and audit['unresolved_mapping']

def test_identical_numeric_values_need_row_provenance():
    c=replace(_case('P05驱动电压DC24V'),hard_negatives=({'file':'机器.pdf','page':4,'quote':'DC24V'},))
    source=_row('p','P05驱动电压DC24V\nP11驱动电压DC24V','机器.pdf',0);m=map_case_evidence(c,[source])
    audit=negative_span_audit([{'chunk_id':'row','member_chunk_ids':['p'],'included_text':'P05驱动电压DC24V'}],m,[source])
    assert audit['negative_quote_hits']==[] and audit['unresolved_mapping']
