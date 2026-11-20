from dataclasses import asdict
import json
import pytest

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import EvidenceChunk, RagChunk, SourceRef
from local_full_text_search.rag.reranking.evidence_features import evidence_features
from local_full_text_search.rag.reranking.evidence_ordering import EvidenceOrdering
from local_full_text_search.rag.reranking.local_semantic_judge import LocalSemanticJudge, validate_judgement


def item(text, key='one', score=.9):
    source=SourceRef(1,key,1,1,1,'C:/fixtures/source.pdf','source.pdf','.pdf',(), '第 1 页',page_number=1)
    return EvidenceChunk(chunk=RagChunk(key,key,0,text,text,'',source),
                         rerank_score=score, fused_score=.01, rerank_score_combined=.2)


def test_plain_parameter_words_do_not_create_structural_support():
    assert evidence_features('P09 的参数是多少？', 'P09维护试题：口述参数标准。')['structural_support'] == 'unknown'


def test_complete_labelled_record_is_bound_to_literal_identifier():
    body = '代码\n描述\n原因\n解决方法\nABCD\n测试失败\n管路气泡\n灌注后重试'
    f = evidence_features('ABCD是什么，怎么处理？', body)
    assert len(f['verified_rows']) == 1
    assert not evidence_features('EFGH怎么处理？', body)['verified_rows']


def test_incomplete_table_is_unknown():
    assert not evidence_features('ABCD怎么处理？', '代码\n描述\n原因\n解决方法\nABCD\n失败')['verified_rows']


@pytest.mark.parametrize('raw', [
    {'support': 'direct', 'applicability': 'unknown', 'quotes': []},
    {'support': 'direct', 'applicability': 'unknown', 'quotes': ['不存在']},
    {'support': 'related', 'applicability': 'conflict', 'quotes': []},
    {'support': 'yes', 'applicability': 'unknown', 'quotes': []},
    [], '{broken',
])
def test_invalid_or_unquoted_claim_abstains(raw):
    assert not validate_judgement(raw, '原文').valid


def test_repeated_quote_does_not_locate_a_row():
    result = validate_judgement({'support': 'direct', 'applicability': 'unknown', 'quotes': ['24V']}, 'A:24V;B:24V')
    assert not result.valid and 'ambiguous' in result.error


def test_exact_quote_offsets_are_source_bound():
    text = '当低位触发时，启动泵。'
    result = validate_judgement({'support': 'direct', 'applicability': 'unknown', 'quotes': ['启动泵']}, text)
    assert result.valid
    s = result.spans[0]
    assert text[s['start']:s['end']] == s['quote']


def test_unknown_keeps_base_order_and_all_candidates():
    values = [item('正文A', 'a', .8), item('正文B', 'b', .9)]
    out = EvidenceOrdering(rules=True).order('普通问题', values, CancelToken())
    assert [x.chunk.stable_chunk_id for x in out] == ['b', 'a']


def test_partial_evidence_is_not_deleted_or_treated_as_conflict():
    class Judge:
        def judge(self, q, text):
            return validate_judgement({'support': 'partial', 'applicability': 'unknown', 'quotes': [text]}, text)
    out = EvidenceOrdering(judge=Judge()).order('分别如何处理？', [item('如果A执行B'), item('如果C执行D', 'two')], CancelToken())
    assert len(out) == 2 and all(x.rerank_diagnostics['tier'] == 1 for x in out)


def test_semantic_conflict_demotes_without_hard_deletion():
    class Judge:
        def judge(self, q, text):
            return validate_judgement({'support': 'related', 'applicability': 'conflict' if text == '仅适用其他平台' else 'unknown', 'quotes': [text]}, text)
    xs = [item('仅适用其他平台', 'a', .99), item('未说明平台', 'b', .8)]
    out = EvidenceOrdering(judge=Judge()).order('目标平台怎么处理', xs, CancelToken())
    assert [x.chunk.stable_chunk_id for x in out] == ['b', 'a'] and len(out) == 2


def test_generator_failure_abstains_and_preserves_base_order():
    def broken(**kwargs): raise RuntimeError('local worker failure')
    judge = LocalSemanticJudge(generator=broken)
    out = EvidenceOrdering(judge=judge).order('问题', [item('正文')], CancelToken())
    assert out[0].rerank_diagnostics['tier'] == 1
    assert 'worker failure' in out[0].rerank_diagnostics['semantic']['error']


def test_prompt_contains_only_runtime_question_and_candidate():
    received = []
    def generate(**kwargs):
        received.append(kwargs)
        return {'choices': [{'message': {'content': json.dumps({'support': 'related', 'applicability': 'unknown', 'quotes': []})}}]}
    judge = LocalSemanticJudge(generator=generate)
    assert judge.judge('用户原话', '候选正文').valid
    assert json.loads(received[0]['messages'][1]['content']) == {'question': '用户原话', 'passage': '候选正文'}


def test_routing_is_explicit_and_unrouted_does_not_invoke_model():
    class Judge:
        def judge(self, *args): raise AssertionError('should not be invoked')
    out = EvidenceOrdering(judge=Judge(), semantic_ids=set()).order('问题', [item('正文')], CancelToken())
    assert out[0].rerank_diagnostics['semantic']['error'] == 'not_routed'


def test_uncertainty_route_uses_disagreement_without_gold_or_query_dictionary():
    calls = []
    class Judge:
        def judge(self, q, text):
            calls.append(text)
            return validate_judgement({'support':'unknown','applicability':'unknown','quotes':[]},text)
    xs = [item('低CE旧第一','a',.1),item('高CE旧第二','b',.9),item('共同第三','c',.8)]
    out = EvidenceOrdering(judge=Judge(),uncertainty_top_k=1).order('普通问题',xs,CancelToken())
    assert set(calls)=={'低CE旧第一','高CE旧第二'}
    assert [x.chunk.stable_chunk_id for x in out]==['b','c','a']


def test_identical_boundary_causes_zero_local_model_calls():
    class Judge:
        def judge(self,*a):raise AssertionError('unnecessary model invocation')
    EvidenceOrdering(judge=Judge(),uncertainty_top_k=8).order('问题',[item('正文')],CancelToken())


def test_invalid_generated_quote_is_neutral_even_when_model_says_direct():
    def generate(**kwargs):
        return {'choices':[{'message':{'content':json.dumps({'support':'direct','applicability':'compatible','quotes':['伪造文字']})}}]}
    out=EvidenceOrdering(judge=LocalSemanticJudge(generator=generate)).order('问题',[item('真实正文')],CancelToken())
    assert out[0].rerank_diagnostics['tier']==1


def test_default_constructor_keeps_semantic_dependency_optional():
    import inspect
    from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
    assert inspect.signature(LocalBgeReranker).parameters['ordering_policy'].default is None


def test_cache_is_bound_to_exact_runtime_input_and_source_offsets(tmp_path):
    calls=[]
    def generate(**kwargs):
        calls.append(kwargs)
        return {'choices':[{'message':{'content':json.dumps({'support':'direct','applicability':'unknown','quotes':['启动泵']})}}]}
    judge=LocalSemanticJudge(generator=generate,cache_dir=tmp_path)
    first=judge.judge('如何操作','低位时启动泵。')
    assert judge.judge('如何操作','低位时启动泵。')==first and len(calls)==1
    judge.judge('如何操作','高位时启动泵。')
    assert len(calls)==2
    for path in tmp_path.glob('*.json'):path.write_text('{broken',encoding='utf8')
    assert not judge.judge('如何操作','低位时启动泵。').valid
    assert len(calls)==2  # Corrupt cache abstains; no silent quality retry.


def test_cache_cannot_upgrade_or_rebind_quoted_claim(tmp_path):
    def generate(**kwargs):
        return {'choices':[{'message':{'content':json.dumps({'support':'partial','applicability':'unknown','quotes':['启动泵']})}}]}
    judge=LocalSemanticJudge(generator=generate,cache_dir=tmp_path)
    judge.judge('问题','启动泵。')
    path=next(tmp_path.glob('*.json'));value=json.loads(path.read_text(encoding='utf8'));value['support']='direct'
    path.write_text(json.dumps(value),encoding='utf8')
    result=judge.judge('问题','启动泵。')
    assert not result.valid and result.error=='invalid_cached_source_binding'
