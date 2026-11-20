from dataclasses import replace
import pytest
from local_full_text_search.rag.condition_binding import bind_plan, judge_local, scope_mentions
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import ConstraintKind as K, ConstraintStatus as S, QueryConstraint, RagQuery
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from tests.rag.test_selection_factorial import candidate


@pytest.mark.parametrize('question,body,status',[
 ('几乎所有发光项目定标失败','个别项目定标失败，检查试剂。','conflict'),
 ('大多数设备离线','单个设备离线，检查本机。','conflict'),
 ('多个通道报错','所有通道报错，重启通道。','unknown'),
 ('几乎所有项目定标失败','大多数项目定标失败。','match'),
 ('全部项目异常','部分项目异常。','unknown'),
 ('全部项目异常','个别项目异常；全部项目异常时检查公共部分。','match'),
 ('全部项目异常','单个通道故障。','unknown'),
 ('所有生化项目异常','个别免疫项目异常。','unknown'),
 ('几乎所有项目定标失败','检查所有项目，然后重新定标。','unknown'),
 ('全部项目异常','并非个别项目异常。','unknown'),
 ('全部项目异常','个别项目异常；其他模块异常。','conflict'),
 ('全部任务中断','某个任务中断。','conflict'),
])
def test_object_bound_scope(question,body,status):
    plan = QueryInterpreter(bound_conditions=True).interpret(question)
    c = next(c for c in plan.constraints if c.kind is K.SCOPE)
    result = judge_local(c,body,question)
    assert result.status.value == status
    assert result.reason
    if result.matched_text: assert result.matched_text in body


@pytest.mark.parametrize('question',[
 '如果单个项目失败或者全部项目失败，分别怎么处理？',
 '并非所有项目异常，该怎么排查？',
 '手册写“所有项目异常”，实际情况还不确定。',
 '可能所有设备离线了。',
 '检查全部样本需要多久？',
])
def test_non_observed_scope_never_hard_filters(question):
    plan = QueryInterpreter(bound_conditions=True).interpret(question)
    assert not any(c.kind is K.SCOPE and c.is_hard for c in plan.constraints)


def test_retrieval_expressions_and_raw_question_unchanged():
    q='几乎所有发光项目定标失败，发光值低于1000，怎么处理？'
    a=QueryInterpreter().interpret(q); b=QueryInterpreter(bound_conditions=True).interpret(q)
    for field in ('original_text','lexical_terms','lexical_expression','semantic_query','lexical_variants','semantic_variants'):
        assert getattr(a,field)==getattr(b,field)
    for c in b.constraints:
        if c.source_start is not None: assert q[c.source_start:c.source_end]==c.source_text
    from local_full_text_search.rag.answer_service import AnswerService
    snap=AnswerService._plan_snapshot(b)
    scope=next(c for c in snap['constraints'] if c['kind']=='scope')
    assert scope['subject']=='发光项目'
    assert q[scope['source_start']:scope['source_end']]==scope['source_text']


def test_unknown_preserved_through_evidence_policy():
    p=QueryInterpreter(bound_conditions=True).interpret('几乎所有项目定标失败怎么办？')
    x=candidate('unknown','检查管路是否有气泡。')
    checked=ConstraintMatcher(local_conditions=True).evaluate(p,[x])
    d=EvidencePolicy(max_per_document=None).evaluate(RagQuery(p.original_text),checked,plan=p)
    assert d.accepted==(x,) and not d.insufficient


def test_direction_belongs_to_object_and_not_to_filename():
    p=QueryInterpreter(bound_conditions=True).interpret('发光值偏低怎么办？')
    c=next(x for x in p.constraints if x.kind is K.DIRECTION)
    assert judge_local(c,'底物余量偏高。',p.original_text).status is S.UNKNOWN
    assert judge_local(c,'发光值偏高。',p.original_text).status is S.CONFLICT
    assert judge_local(c,'发光值偏低时检查A；发光值偏高时检查B。',p.original_text).status is S.MATCH


def test_model_requires_explicit_exclusive_body_scope():
    c=QueryConstraint(K.MODEL,'CL-8000i',source_text='CL-8000i')
    assert judge_local(c,'以CL-2600i为例。','CL-8000i如何操作').status is S.UNKNOWN
    assert judge_local(c,'仅适用于 CL-2600i。','CL-8000i如何操作').status is S.CONFLICT


def test_local_judgement_does_not_add_rank_bonus():
    p=QueryInterpreter(bound_conditions=True).interpret('几乎所有项目定标失败怎么办？')
    xs=[candidate('a','检查液路。'),candidate('b','大多数项目定标失败。')]
    a=ConstraintMatcher().evaluate(QueryInterpreter().interpret(p.original_text),xs)
    b=ConstraintMatcher(local_conditions=True).evaluate(p,xs)
    assert [x.chunk.stable_chunk_id for x in a]==[x.chunk.stable_chunk_id for x in b]


@pytest.mark.xfail(strict=True, reason='Known blocker: short quantifiers and action quantities lose branch ownership; experimental mode must stay off')
def test_short_quantifier_method_branches_must_not_be_globally_rejected():
    p=QueryInterpreter(bound_conditions=True).interpret('大部分项目定标失败怎么办？')
    x=candidate('branches','多项目定标失败优先考虑系统性故障，单项目失败应综合分析，特别在一次只定一个项目时不能忽略系统故障的可能')
    result=ConstraintMatcher(local_conditions=True).evaluate(p,[x])[0]
    assert result.constraint_decision.accepted
