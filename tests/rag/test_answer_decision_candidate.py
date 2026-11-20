import json
from dataclasses import replace
import pytest
from tools.answer_decision_candidate import parse_result
from local_full_text_search.rag.generation.base import GenerationRequest

REQ=GenerationRequest(question='怎么操作？当前设备能否批量处理？',context='source',model='test',api_base_url='https://example.test',api_style='chat_completions',api_key='unused')
ENTRIES=[{'source_id':'S1','included_text':'打开菜单。是否批量处理取决于通讯方式。'}]
def part(state='supported'):
    return {'request_quote':'怎么操作','state':state,'text':'打开菜单。[S1]','evidence':[{'source_id':'S1','quote':'打开菜单。'}],'missing_fields':[],'follow_up':''}
def parse(parts):return parse_result(json.dumps({'parts':parts},ensure_ascii=False),REQ,ENTRIES)
def test_partial_retains_answer_and_question():
    q=part('needs_user_input');q.update(request_quote='当前设备能否批量处理',text='',missing_fields=['通讯方式'],follow_up='当前使用什么通讯方式？',evidence=[{'source_id':'S1','quote':'是否批量处理取决于通讯方式。'}])
    r=parse([part(),q]);assert r['behavior']=='partial_answer' and r['status']=='answered';assert '打开菜单' in r['text'] and '通讯方式？' in r['text']
def test_pure_question_needs_no_answer_citation():
    q=part('needs_user_input');q.update(text='',missing_fields=['通讯方式'],follow_up='使用什么通讯方式？')
    r=parse([q]);assert r['status']=='needs_clarification' and '[S1]' not in r['text']
@pytest.mark.parametrize('mutation',[{'evidence':[{'source_id':'S99','quote':'打开菜单。'}]},{'text':'打开菜单。[S99]'},{'request_quote':'不存在的诉求'},{'evidence':[{'source_id':'S1','quote':'应该更换板卡'}]}])
def test_invalid_provenance_fails_execution(mutation):
    p=part();p.update(mutation)
    with pytest.raises(ValueError):parse([p])
def test_missing_one_part_does_not_erase_supported_part():
    p=part('evidence_missing');p.update(request_quote='当前设备能否批量处理',text='资料未说明当前设备的批量处理能力。',evidence=[])
    r=parse([part(),p]);assert r['behavior']=='partial_answer' and '打开菜单' in r['text']
def test_format_recovery_is_bounded_and_not_business_refusal():
    from tools.answer_decision_candidate import generate
    from local_full_text_search.core.task_manager import CancelToken
    class FakeGateway:
        calls=0
        def complete_text(self,request,token):self.calls+=1;return '{}'
    g=FakeGateway()
    with pytest.raises(ValueError):generate(g,REQ,ENTRIES,CancelToken())
    assert g.calls==2
def test_valid_format_recovery_preserves_supported_answer():
    from tools.answer_decision_candidate import generate
    from local_full_text_search.core.task_manager import CancelToken
    class FakeGateway:
        calls=0
        def complete_text(self,request,token):
            self.calls+=1
            return '{}' if self.calls==1 else json.dumps({'parts':[part()]},ensure_ascii=False)
    g=FakeGateway();r=generate(g,REQ,ENTRIES,CancelToken())
    assert g.calls==2 and r['format_recoveries']==1 and r['status']=='answered'
