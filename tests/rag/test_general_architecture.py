import ast
from dataclasses import replace
import json
from pathlib import Path

import pytest
from local_full_text_search.rag.domain_adapter import DomainAdapter
from local_full_text_search.rag.query_semantics import object_mentions, quantity_scopes
from local_full_text_search.rag.fact_schema import extract_facts, merge_facts, numeric_semantics
from local_full_text_search.rag.source_consensus import summarize, attach_context_consensus
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.generation.structured import validate_plan, render_plan
from tests.rag.test_query_plan_adapter import plan, candidate


@pytest.mark.parametrize('noun', ['检测项目','通道','模块','样本','设备'])
@pytest.mark.parametrize('quantifier,expected', [('单个','single'),('少数','multiple'),('大部分','multiple'),('全部','all')])
def test_quantities_not_bound_to_one_alarm(noun, quantifier, expected):
    assert quantity_scopes(quantifier+noun+'异常')[0][1] == expected


@pytest.mark.parametrize('whole,part', [('压缩机','滤芯'),('电源模块','风扇'),('传送机','皮带')])
def test_unseen_parent_child_names(whole, part):
    aliases = {'component': {'parent': [whole], 'child': [part]}}
    mentions = object_mentions(whole+part+'寿命多少？', aliases)
    assert len(mentions) == 1 and mentions[0]['head']['text'] == part
    assert mentions[0]['modifiers'][0]['text'] == whole
    assert len(object_mentions(whole+'和'+part+'分别寿命多少？', aliases)) == 2


def record(file, scope, values, field='支持协议', obj='控制器'):
    return {'fact': field, 'object': [obj], 'values': values,
            'source': {'file': file, 'scope': scope, 'page': 1, 'chunk_id': file}}


@pytest.mark.parametrize('field', ['支持协议','可用文件格式','允许采集模式'])
def test_same_explicit_scope_agreement(field):
    result = summarize([record('a', {'平台':['甲']}, ['A','B'], field), record('b', {'平台':['甲']}, ['b','a'], field)])
    assert len(result) == 1 and result[0]['supporting_source_count'] == 2
    assert result[0]['consensus_level'] == 'scoped_agreement'


def test_different_scope_separate_unknown_scope_separate_conflict_not_union():
    assert len(summarize([record('a', {'平台':['甲']}, ['A']),record('b', {'平台':['乙']}, ['A'])])) == 2
    assert len(summarize([record('a', {}, ['A']),record('b', {}, ['A'])])) == 2
    rows=summarize([record('a', {'平台':['甲']}, ['A']),record('b', {'平台':['甲']}, ['B'])])
    assert rows[0]['consensus_level']=='conflict' and rows[0]['values']==[]


def test_numeric_roles_are_not_hardcoded_answers():
    assert 'nominal' in numeric_semantics('风机标称电压 48V')[0]['types']
    assert 'normal_range' in numeric_semantics('风机正常工作范围 45～51V')[0]['types']
    assert 'control_signal' in numeric_semantics('风机控制信号 5V')[0]['types']


def test_facts_unknown_object_not_merged_across_sources():
    a=candidate('使用寿命：800次',1);b=candidate('使用寿命：800次',2)
    facts=merge_facts(extract_facts(a.chunk.text,a.chunk.source)+extract_facts(b.chunk.text,b.chunk.source))
    assert len(facts)==2


def test_config_rejects_gold_in_nested_configuration(tmp_path):
    path=tmp_path/'bad.json';path.write_text(json.dumps({'policy':{'rules':[{'expected_answer':'answer'}]}}))
    assert DomainAdapter.load(path).identity=='generic'


def test_query_view_does_not_invent_model_alias_or_answers():
    p=plan('ZX-4567 驱动模块电压是多少？')
    spec=p.parsed_request
    assert spec['requested_fields']==['value']
    assert 'ZX-4567M' not in json.dumps(spec)
    assert not any(v in str(spec) for v in ('22.8','66000'))


def claim(text='风机额定电压为48V', quote='风机额定电压为48V', **kwargs):
    return {'text':text,'claim_type':'value','object':'风机','scope':{},'conditions':[],
            'certainty':'supported','source_refs':[{'source_id':'S1','quote':quote}],**kwargs}


@pytest.mark.parametrize('bad', [claim(text='风机额定电压为99V'),claim(quote='原文没有这句话'),claim(certainty='confirmed')])
def test_unsupported_claims_cannot_render(bad):
    ctx=ContextBuilder().build([candidate('风机额定电压为48V')])
    checked,rejected=validate_plan({'claims':[bad]},ctx,{'requested_fields':['value']})
    assert rejected and not checked['claims']
    assert render_plan(checked)=='资料中没有足够依据。'


def test_possible_cause_never_confirmed_and_unasked_replacement_rejected():
    ctx=ContextBuilder().build([candidate('可能原因：连接线松动；处理：更换连接线')])
    checked,rejected=validate_plan({'claims':[claim('连接线松动','连接线松动',claim_type='cause'),claim('更换连接线','更换连接线',claim_type='replacement')]},ctx,{'requested_fields':['cause']})
    assert len(rejected)==2 and not checked['claims']


def test_valid_claim_deterministic_render():
    ctx=ContextBuilder().build([candidate('风机额定电压为48V')])
    checked,rejected=validate_plan({'claims':[claim()]},ctx,{'requested_fields':['value']})
    assert not rejected and render_plan(checked)=='风机额定电压为48V [S1]'


@pytest.mark.parametrize('supported', [True, False])
def test_paraphrased_condition_requires_semantic_review(supported):
    from local_full_text_search.rag.generation.structured import generate_structured
    from local_full_text_search.rag.generation.base import GenerationRequest
    from local_full_text_search.core.task_manager import CancelToken
    ctx=ContextBuilder().build([candidate('压力超过允许上限时，应停止供气。')])
    item=claim('应停止供气','压力超过允许上限时，应停止供气。',claim_type='solution',conditions=['压力超出允许范围上限'])
    class Gateway:
        calls=[]
        def complete_text(self, request, token):
            self.calls.append(request)
            if len(self.calls)==1:return json.dumps({'claims':[item]})
            payload=json.loads(request.user_prompt)
            assert payload['claims'][0]['conditions']==item['conditions']
            return json.dumps({'checks':[{'claim_index':0,'supported':supported,'reason':'核对了前提和动作'}]})
    gateway=Gateway();logs=[]
    request=GenerationRequest('压力异常时如何处理？',ctx.text,'test','','','')
    result=generate_structured(gateway,request,plan(request.question),ctx,CancelToken(),logs.append)
    assert len(gateway.calls)==2
    assert ('应停止供气' in result.text)==supported
    assert not logs[0]['structural_rejections']


def test_scope_metadata_must_have_text_values():
    ctx=ContextBuilder().build([candidate('风机额定电压为48V')])
    checked,rejected=validate_plan({'claims':[claim(scope={'型号':['invented']})]},ctx,{'requested_fields':['value']})
    assert not checked['claims'] and rejected[0]['reason']=='invalid_scope_structure'


def test_runtime_never_imports_gold_evaluator():
    root=Path(__file__).resolve().parents[2]/'local_full_text_search/rag'
    for path in root.rglob('*.py'):
        if 'evaluation' in path.parts: continue
        tree=ast.parse(path.read_text(encoding='utf8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom): assert not (node.module or '').startswith('local_full_text_search.rag.evaluation'),path
            if isinstance(node,ast.Import): assert all(not n.name.startswith('local_full_text_search.rag.evaluation') for n in node.names),path
