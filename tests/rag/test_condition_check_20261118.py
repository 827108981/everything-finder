from copy import deepcopy
import pytest
from local_full_text_search.rag.condition_relations import judge_relation
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.models import ConstraintKind as K, ConstraintStatus as S, QueryConstraint
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.trace import candidate_snapshot
from tests.rag.test_selection_factorial import candidate


@pytest.mark.parametrize('kind,value,question,source,body,expected', [
    (K.DIRECTION,'low','电压偏低怎么办','偏低','温度偏高。','unknown'),
    (K.DIRECTION,'low','电压偏低怎么办','偏低','电压偏高。','conflict'),
    (K.DIRECTION,'low','电压偏低怎么办','偏低','电压偏低。','match'),
    (K.DIRECTION,'low','电压偏低怎么办','偏低','电压偏低时检查A；电压偏高时检查B。','match'),
    (K.DIRECTION,'low','电压偏低怎么办','偏低','电压可能偏高。','unknown'),
    (K.DIRECTION,'low','电压偏低怎么办','偏低','电压并非偏高。','unknown'),
    (K.STAGE,'sample_test','样本检测异常怎么办','样本检测','处理后重新定标。','unknown'),
    (K.STAGE,'sample_test','样本检测异常怎么办','样本检测','样本条码支持多种码制。','unknown'),
    (K.STAGE,'sample_test','样本检测异常怎么办','样本检测','样本检测阶段出现异常。','match'),
    (K.STAGE,'sample_test','样本检测异常怎么办','样本检测','故障阶段：定标过程中。','conflict'),
    (K.STAGE,'calibration','定标失败怎么处理','定标','更换校准品后重新定标。','unknown'),
    (K.STAGE,'calibration','定标失败怎么处理','定标','定标失败时检查管路。','match'),
    (K.MODEL,'CL-8000i','CL-8000i参数多少','CL-8000i','与CL-2600i进行比较。','unknown'),
    (K.MODEL,'CL-8000i','CL-8000i参数多少','CL-8000i','适用型号：CL-2600i。','conflict'),
    (K.MODEL,'CL-8000i','CL-8000i参数多少','CL-8000i','CL-2600i参数为A；CL-8000i参数为B。','match'),
    (K.OBJECT,'样本','样本条码码制有哪些','样本','条码码制包括Code128。','unknown'),
    (K.SCOPE,'all','全部项目异常怎么办','全部项目','单个项目异常。','conflict'),
    (K.SCOPE,'all','全部项目异常怎么办','全部项目','检查所有项目，然后重新定标。','unknown'),
    (K.SCOPE,'all','全部项目异常怎么办','全部项目','单个通道异常。','unknown'),
    (K.SCOPE,'all','全部项目异常怎么办','全部项目','单个项目异常；全部项目异常时检查公共系统。','match'),
    (K.SCOPE,'all','全部项目异常怎么办','全部项目','多项目定标失败优先考虑系统性故障，单项目失败应综合分析，特别在一次只定一个项目时不能忽略系统故障的可能','unknown'),
])
def test_relation_proofs(kind,value,question,source,body,expected):
    c=QueryConstraint(kind,value,source_text=source)
    j,p=judge_relation(c,body,question)
    assert j.status.value==expected
    assert p['status']==expected and p['reason']
    if p['source_start'] is not None:
        assert body[p['source_start']:p['source_end']]==p['candidate_quote']==j.matched_text
    else:
        assert not j.matched_text


def test_A_changes_no_labels_and_preserves_every_input_pair():
    plan=QueryInterpreter().interpret('样本条码支持哪些码制？')
    xs=[candidate('unknown','支持码制Code128'),candidate('match','样本条码配置说明')]
    old=ConstraintMatcher().evaluate(plan,deepcopy(xs))
    new=ConstraintMatcher(preserve_order=True).evaluate(plan,deepcopy(xs))
    assert [x.chunk.stable_chunk_id for x in old]==['match','unknown']
    assert [x.chunk.stable_chunk_id for x in new]==['unknown','match']
    assert {x.chunk.stable_chunk_id:x.constraint_decision.judgements for x in old}=={
        x.chunk.stable_chunk_id:x.constraint_decision.judgements for x in new}


def test_B_never_changes_query_or_retrieval_scores():
    plan=QueryInterpreter().interpret('电压偏低怎么办？')
    xs=[candidate('x','温度偏高。')]
    before=(plan,xs[0].rerank_score,xs[0].vector_score)
    checked=ConstraintMatcher(preserve_order=True,judgement_mode='source_bound').evaluate(plan,xs)
    assert before==(plan,checked[0].rerank_score,checked[0].vector_score)
    assert candidate_snapshot(checked[0])['query_plan_diagnostics']['condition_relations']['checks']


def test_filenames_cannot_supply_body_relation():
    p=QueryInterpreter().interpret('样本检测异常怎么处理？')
    x=candidate('x','检查液路。')
    # The ordinary candidate fixture title contains no fact proof.
    checked=ConstraintMatcher(judgement_mode='source_bound').evaluate(p,[x])[0]
    assert all(j.status is S.UNKNOWN for j in checked.constraint_decision.judgements)


def test_invalid_mode_fails_early():
    with pytest.raises(ValueError):
        ConstraintMatcher(judgement_mode='made_up')


def test_C_is_opt_in_and_does_not_bypass_proved_conflicts():
    from local_full_text_search.rag.models import QueryPlan, RagQuery
    from local_full_text_search.rag.evidence_policy import EvidencePolicy
    p=QueryInterpreter().interpret('样本检测异常怎么办？')
    x=candidate('x','检查液路。')
    checked=ConstraintMatcher(preserve_order=True,judgement_mode='source_bound').evaluate(p,[x])
    assert EvidencePolicy().evaluate(RagQuery(p.original_text),checked,plan=p).insufficient
    d=EvidencePolicy(require_confirmed_conditions=False).evaluate(RagQuery(p.original_text),checked,plan=p)
    assert d.accepted and d.reason=='accepted_with_unconfirmed_conditions'
    assert checked[0].query_plan_diagnostics['unconfirmed_conditions']
    bad=candidate('bad','故障阶段：定标过程中。')
    checked=ConstraintMatcher(judgement_mode='source_bound').evaluate(p,[bad])
    assert EvidencePolicy(require_confirmed_conditions=False).evaluate(RagQuery(p.original_text),checked,plan=p).insufficient


def test_actual_context_cannot_advertise_a_truncated_proof():
    from local_full_text_search.rag.context_builder import ContextBuilder
    p=QueryInterpreter().interpret('电压偏低怎么办？')
    x=candidate('x','背景说明。'*90+'电压偏低。')
    checked=ConstraintMatcher(judgement_mode='source_bound').evaluate(p,[x])
    assert any(c['status']=='match' for c in x.query_plan_diagnostics['condition_relations']['checks'])
    ctx=ContextBuilder().build(checked,max_chars=256)
    assert ctx.entries and ctx.entries[0]['truncated']
    checks=ctx.entries[0]['query_plan_diagnostics']['condition_relations']['checks']
    assert all(c['status']=='unknown' and c['reason_code']=='proof_not_included' for c in checks)
