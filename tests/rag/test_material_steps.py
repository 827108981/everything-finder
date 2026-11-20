from dataclasses import replace
from local_full_text_search.rag.models import CandidateDecision
from tests.rag.test_evidence_selection_experiment import candidate
from tools.validate_material_steps import conservative_select, anchors

def supported(x):
    x.constraint_decision=CandidateDecision(x.chunk.stable_chunk_id,True,'test',(),0,0,0,0,0,'supported')
    return x

def test_identifiers_are_whole_tokens_not_device_numbers():
    assert anchors('CL-8000i 报CSDB，DCF?')=={'CSDB','DCF'}
    assert 'DEV' not in anchors('DEV_A')

def test_rescue_only_free_slot_never_evicts_existing_facts():
    original=[supported(candidate(str(i),f'保留事实{i}')) for i in range(8)]
    extra=supported(candidate('extra','CSDB处理'))
    assert conservative_select('CSDB?',original+[extra],original)[0]==original
    chosen,_=conservative_select('CSDB?',original[:4]+[extra],original[:4])
    assert chosen==original[:4]+[extra]

def test_unknown_applicability_not_rescued_and_no_unasked_identifier():
    extra=candidate('extra','CSDB处理')
    assert conservative_select('CSDB?', [extra],[])[0]==[]
    supported(extra)
    assert conservative_select('DEV?', [extra],[])[0]==[]

def test_numeric_variants_and_document_boundaries_preserved():
    a=candidate('a','流量140±5 μL/s');b=candidate('b','流量145±5 μL/s')
    c=candidate('c',a.chunk.text,doc='v2')
    assert conservative_select('?', [a,b,c],[a,b,c])[0]==[a,b,c]
