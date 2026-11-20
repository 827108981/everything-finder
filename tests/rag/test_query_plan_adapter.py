from dataclasses import replace
import json

import pytest

from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.models import EvidenceChunk, RagQuery, ConstraintStatus
from local_full_text_search.rag.identifier_evidence import annotate, coverage, exact, structured_rows
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.domain_adapter import DomainAdapter
from tests.rag.test_query_analyzer import _chunk

def plan(text):
    return QueryInterpreter(enhanced=True,domain_config='chemiluminescence').interpret(text)

def candidate(text, n=1, document='same'):
    return EvidenceChunk(replace(_chunk(text,n,'manual.pdf'),document_key=document),lexical_rank=n,rerank_score=.9)

TABLE='标识\n描述\n原因\n解决方法\nXYZ\n温度错误\n温度超出规定范围\n检查温控后重新检测\nABC\n压力错误\n压力过低\n检查泵\n11'

@pytest.mark.parametrize('code',['CSDB','DEV','DUP','ZZQX'])
def test_unregistered_alarms_are_identified_without_inventing_meaning(code):
    p=plan(f'发光定标报 {code} 是什么意思？该怎么处理？')
    e=next(e for e in p.entity_details if e.text==code)
    assert e.type=='alarm_code' and e.confidence>0 and e.reason and e.source
    assert code in p.lexical_terms and any(x.text==code for x in p.unknown_entities)
    assert {s.name for s in p.answer_requirements} >= {'definition','cause','handling'}
    assert any(c.value=='calibration' for c in p.constraints)
    assert not any(c.value==code for c in p.constraints)

def test_simple_unknown_question_does_not_invent_stage_model_or_subqueries():
    p=plan('ZZQX 是什么意思？')
    assert not p.constraints and not p.sub_queries
    assert p.unknown_entities[0].text=='ZZQX'
    assert all(e.type!='instrument_model' for e in p.entity_details)

def test_branch_queries_retain_original_objects_and_conditions():
    q='CL-8000i 样本检测，单个项目偏低时怎么排查；所有项目偏低时怎么排查？'
    p=plan(q)
    assert len(p.sub_queries)==2
    assert all(q in sub.text and sub.focus in q and sub.inherited_constraints==p.constraints for sub in p.sub_queries)
    assert not any(c.kind.value=='scope' and c.is_hard for c in p.constraints)

def test_model_pattern_does_not_turn_component_into_instrument():
    p=plan('P02 和 SV99 分别是什么？')
    assert all(e.type!='instrument_model' for e in p.entity_details)
    assert {'P02','SV99'} <= {e.text for e in p.unknown_entities}

@pytest.mark.parametrize('word,value',[('校准','calibration'),('临床检测','sample_test'),('个别项目','single'),('大部分项目','multiple')])
def test_adapter_aliases(word,value):
    assert any(e.normalized_value==value for e in plan(word+'异常怎么处理？').entity_details)

def test_missing_and_invalid_domain_configuration_fall_back(tmp_path):
    p=QueryInterpreter(enhanced=True,domain_config=tmp_path/'missing.json').interpret('ZZQX 是什么？')
    assert p.domain_config=='generic' and p.original_text=='ZZQX 是什么？'
    assert p.unknown_entities
    (tmp_path/'bad.json').write_text('{"patterns":{"x":"["}}')
    assert DomainAdapter.load(tmp_path/'bad.json').identity=='generic'

def test_legacy_interpreter_fields_stay_compatible():
    p=QueryInterpreter().interpret('索引任务如何恢复？')
    assert p.analyzer_version=='query-plan-v3' and not p.entity_details
    assert all(isinstance(x,str) for x in p.entities)

def test_unknown_is_not_hard_conflict_and_missing_stage_stays_unknown():
    p=plan('定标报 XYZ 是什么意思？')
    i=candidate('XYZ 温度故障')
    ConstraintMatcher().evaluate(p,[i])
    assert i.constraint_decision.accepted
    assert i.constraint_decision.judgements[0].status==ConstraintStatus.UNKNOWN

def test_exact_boundary_does_not_match_another_code():
    assert not exact('XYZ1 ABXYZ XYZ_A','XYZ')
    assert exact('报警 xyz：温度高','XYZ')

def test_regular_row_projection_preserves_binding_and_actual_quotes():
    p=plan('报 XYZ 是什么意思？怎么处理？')
    i=candidate(TABLE)
    annotate(p,i)
    ctx=ContextBuilder().build([i])
    assert ctx.entries[0]['included_text']=='标识\n描述\n原因\n解决方法\nXYZ\n温度错误\n温度超出规定范围\n检查温控后重新检测'
    assert 'ABC' not in ctx.entries[0]['included_text']
    assert '\n'.join(i.chunk.text[span['start']:span['end']] for span in ctx.entries[0]['source_spans'])==ctx.entries[0]['included_text']
    assert all(x['status']=='covered' for x in coverage(p,ctx.entries))

def test_broken_or_multiline_table_does_not_guess_columns():
    assert not structured_rows(TABLE.replace('温度超出规定范围','温度\n超出规定范围'))
    assert not structured_rows('XYZ\n温度错误\n温度高\n检查温控')

def test_exact_evidence_is_not_removed_by_document_quota():
    p=plan('报 XYZ 怎么处理？')
    items=[candidate('XYZ 温度高',n) for n in range(1,5)]
    result=EvidencePolicy(max_per_document=1).evaluate(RagQuery(p.original_text),items,plan=p)
    assert len(result.accepted)==2
    assert items[0].query_plan_diagnostics['protected_from_document_quota']
    assert '标识已有保留代表' in result.rejected[0][1]

def test_common_identifier_does_not_push_out_nonidentifier_explanation():
    p=plan('报 XYZ 怎么处理？')
    context=candidate('温度曲线正常，但与参考值偏差过大',1)
    mentions=[candidate('报警 XYZ：排查异常',n) for n in range(2,12)]
    row=candidate(TABLE,20)
    result=EvidencePolicy(max_per_document=3).evaluate(RagQuery(p.original_text,final_top_k=4),[context,*mentions,row],plan=p)
    assert row in result.accepted and context in result.accepted

def test_different_explicit_alarm_rejected_unknown_identifier_not_filtered():
    p=plan('报 XYZ 怎么处理？')
    wrong=candidate('报警 ABC：更换泵')
    right=candidate(TABLE,2)
    result=EvidencePolicy().evaluate(RagQuery(p.original_text),[wrong,right],plan=p)
    assert result.accepted==(right,)
    assert 'explicit_identifier_conflict' in dict(result.rejected)[wrong.chunk.stable_chunk_id]
    p2=plan('XYZ 是什么？')
    assert not annotate(p2,wrong)['identifier_conflict']

def test_exact_preference_never_bypasses_hard_stage_conflict():
    p=plan('定标报 XYZ 怎么处理？')
    i=candidate('样本检测报 XYZ，检查吸样针')
    ConstraintMatcher().evaluate(p,[i])
    result=EvidencePolicy().evaluate(RagQuery(p.original_text),[i],plan=p)
    assert result.insufficient and not result.accepted

def test_truncated_context_does_not_claim_coverage():
    p=plan('报 XYZ 是什么意思？怎么处理？')
    i=candidate(TABLE)
    annotate(p,i)
    entries=list(ContextBuilder().build([i]).entries)
    entries[0]={**entries[0],'included_text':'XYZ\n温度错误'}
    states={r['requirement']:r['status'] for r in coverage(p,entries)}
    assert states=={'definition':'covered','handling':'unknown','cause':'unknown'}


def test_material_code_and_numeric_unit_are_preserved_without_hard_filters():
    p=plan('物料编码 123-456-789，流量145±5 μL/s是什么意思？')
    assert any(e.type=='material_code' and e.text=='123-456-789' for e in p.entity_details)
    assert any(e.type=='numeric_unit' for e in p.entity_details)
    assert not any(c.value=='123-456-789' for c in p.constraints)

def test_negated_model_and_alias_do_not_become_hard_conditions():
    p=plan('不是CL-8000i，型号不清楚，不是临床检测，是定标问题')
    assert not any(c.is_hard for c in p.constraints if c.kind.value=='model' or c.value=='sample_test')
    assert any(c.is_hard and c.value=='calibration' for c in p.constraints)

def test_independent_objects_are_not_mutually_exclusive():
    p=plan('样本和试剂怎么处理？')
    assert all(c.is_hard for c in p.constraints if c.kind.value=='object')


def test_exact_route_uses_published_text_and_respects_root_filter(tmp_path):
    from local_full_text_search.rag.index_store import RagIndexStore
    from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
    from local_full_text_search.core.task_manager import CancelToken
    store=RagIndexStore(tmp_path/'rag')
    bundle,conn=store.create_candidate(1,embedding_repo_id='test',embedding_revision='a'*40,
        embedding_dimension=3,reranker_repo_id='test',reranker_revision='b'*40)
    store.add_chunk(conn,_chunk(TABLE,1,'table.pdf'))
    store.add_chunk(conn,_chunk('报警 ABC，检查其他模块。',2,'other.pdf'))
    conn.commit()
    try:
        p=plan('报 XYZ 是什么意思？怎么处理？')
        retriever=LexicalRetriever(store,bundle.metadata_path)
        hits=retriever.retrieve(RagQuery(p.original_text),CancelToken(),p)
        assert hits and exact(hits[0].chunk.text,'XYZ')
        assert any(d['route_id']=='identifier_exact' and d['candidate_count']==1 for d in retriever.route_diagnostics)
        assert not retriever.retrieve(RagQuery(p.original_text,root_ids=(999,)),CancelToken(),p)
    finally:store.abort(bundle,conn)
