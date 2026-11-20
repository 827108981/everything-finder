from dataclasses import replace
import json
from pathlib import Path
from tests.rag.test_query_plan_adapter import plan,candidate
from local_full_text_search.rag.source_consensus import list_fields,summarize,source_scope,attach_context_consensus,reserve,protect_topic_causes
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import EvidencePart,EvidenceAlternative,GoldEvidenceMatch
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

def row(values,platform,file):
    return {'fact':'支持通信协议','values':values,'source':{'file':file,'platform_group':platform,'explicit_scope':[]}}

def test_same_field_sets_deduplicate_order_case_whitespace():
    r=summarize([row(['Alpha','BETA'],'fast','a'),row(['beta','Al pha'],'slow','b')])[0]
    assert r['consensus_level']=='platform_consensus' and r['supporting_source_count']==2
    assert len(r['values'])==2 and not r['conflicts']

def test_conflicting_values_are_not_unioned():
    r=summarize([row(['A','B'],'fast','a'),row(['A','C'],'slow','b')])[0]
    assert r['consensus_level']=='conflict' and not r['values'] and len(r['conflicts'])==2

def test_format_copies_cannot_create_platform_consensus():
    r=summarize([row(['A','B'],'fast','a.pdf'),row(['A','B'],'fast','a.pptx')])[0]
    assert r['consensus_level']=='document_agreement'

def test_explicit_scope_blocks_platform_generalization():
    a=row(['A','B'],'fast','a');a['source']['explicit_scope']=['仅适用于AA-123']
    r=summarize([a,row(['A','B'],'slow','b')])[0]
    assert r['consensus_level']=='scoped_agreement'

def test_parser_rejects_unrelated_fields_and_visibly_incomplete_cell():
    assert list_fields('支持通信协议\nA、B、C','支持哪些通信协议？')
    assert not list_fields('支持通信协议\nA、B、C/','支持哪些通信协议？')
    assert not list_fields('支持通信协议\nA、B、C','正常驱动电压多少？')

def test_other_product_line_not_consensus_support():
    c=candidate('支持条形码码制\nA、B')
    c.chunk=replace(c.chunk,source=replace(c.chunk.source,file_path='E:/生化资料/通用.pptx'))
    assert source_scope(c.chunk.source,plan('免疫仪支持哪些条形码码制？')) is None

def test_exact_merged_context_preserves_all_source_ids():
    a=candidate('支持通信协议\nA、B',1,'one');b=candidate(a.chunk.text,2,'two')
    # Candidate helper uses text hashes; dedup requires verified equal bodies.
    merged=CandidateDeduplicator().deduplicate([a,b],limit=8)
    ctx=ContextBuilder().build(merged.candidates)
    assert {a.chunk.source.stable_chunk_id,b.chunk.source.stable_chunk_id}<=set(ctx.entries[0]['member_chunk_ids'])

def test_single_explicit_seal_is_not_ambiguous_or_whole_assembly():
    p=plan('A&T ISE 注射器密封圈的使用寿命是多少？物料编码是什么？')
    assert not any(a.name=='part_whole' for a in p.ambiguities)
    assert '66000' not in str(p)

def test_modified_scope_retains_multiple_projects():
    p=plan('大部分发光项目定标失败，报 XYZ，怎么解决？')
    assert any(c.value=='multiple' and c.is_hard for c in p.constraints)

def test_diagnostic_heading_preserved_without_universal_model_inference():
    p=plan('CL-3456i 温度自检失败，可能是什么原因？')
    popular=candidate('CL-3456i 通讯问题排查',1)
    generic=candidate('故障排查-温度自检\n可能原因：\n传感器连接异常',2)
    other=candidate('温度自检\n例如CL-7890i\n可能原因：专用阀异常',3)
    ordered=protect_topic_causes(p,[popular,other,generic])
    assert ordered[0] is generic
    assert not other.query_plan_diagnostics.get('source_representative')
    assert '阈值' in ContextBuilder().build([generic]).text

def test_cause_contract_never_injects_diagnostic_facts():
    from local_full_text_search.rag.evidence_contract import contract_summary
    p=plan('CL-3456i 温度自检失败，可能是什么原因？')
    text=contract_summary(p)
    assert '本题只问可能原因' in text and 'UPS' not in text and '暗计数' not in text
    assert '这是方法问题' not in text

def test_verified_consensus_reaches_generation_scope_contract():
    from local_full_text_search.rag.source_consensus import generation_consensus_contract
    same=summarize([row(['A','B'],'fast','a'),row(['A','B'],'slow','b')])
    text=generation_consensus_contract(same)
    assert '共同支持' in text and '仅该型号' in text and 'fast' in text and 'slow' in text
    assert generation_consensus_contract([])==''
    conflicting=summarize([row(['A','B'],'fast','a'),row(['A','C'],'slow','b')])
    assert '不能宣称平台共识' in generation_consensus_contract(conflicting)

def test_disabled_diagnostic_reservation_is_noop():
    from local_full_text_search.rag.query_interpreter import QueryInterpreter
    p=QueryInterpreter().interpret('CL-3456i 温度自检失败，可能是什么原因？')
    items=[candidate('其他',1),candidate('温度自检\n可能原因：异常',2)]
    assert protect_topic_causes(p,items)==items

def test_different_alarm_in_method_heading_is_not_generic_evidence():
    from local_full_text_search.rag.identifier_evidence import annotate
    p=plan('发光定标报 XYZ，可能什么原因，怎么处理？')
    other=candidate('ABC排查思路\n先检查前后两天报警再处理')
    same=candidate('XYZ排查思路\n先排除污染再重新定标',2)
    assert annotate(p,other)['identifier_conflict']
    assert not annotate(p,same)['identifier_conflict']

def test_adjacent_scene_cannot_inherit_conflicting_or_missing_scope():
    from local_full_text_search.rag.source_consensus import scene_eligible
    p=plan('单个项目定标失败报 XYZ，怎么解决？')
    assert not scene_eligible(p,candidate('大部分项目定标失败，报XYZ'))
    assert not scene_eligible(p,candidate('定标失败，报XYZ'))
    assert scene_eligible(p,candidate('单个项目定标失败，报XYZ'))

def test_adjacent_detail_is_rejected_when_scene_fails_gate():
    from local_full_text_search.rag.evidence_policy import EvidencePolicy
    from local_full_text_search.rag.models import RagQuery
    p=plan('报 XYZ 怎么解决？');detail=candidate('可能原因：管路污染\n解决思路：清洗')
    detail.query_plan_diagnostics['adjacent_scene']={'scene_chunk_id':'absent','detail_chunk_id':detail.chunk.stable_chunk_id}
    result=EvidencePolicy().evaluate(RagQuery(p.original_text),[detail],plan=p)
    assert not result.accepted and any('场景页' in reason for _,reason in result.rejected)

def test_representative_reservation_is_bounded():
    items=[candidate(str(n),n) for n in range(10)]
    for x in items: x.query_plan_diagnostics['source_representative']=True
    assert reserve(items)==items
    assert sum(bool(i.query_plan_diagnostics.get('source_representative')) for i in items)==4

def test_same_page_or_merged_source_without_actual_quote_does_not_score():
    expected='目标部件正常范围'
    part=EvidencePart('second',0,len(expected),0,len(expected))
    mapping=GoldEvidenceMatch('E1','positive','manual', '2',expected,('second',),'confirmed',alternatives=(EvidenceAlternative('A1','exact',(part,),True),))
    entry={'chunk_id':'first','member_chunk_ids':['first','second'],'included_text':'同页另一个部件'}
    assert score_stage('context',[entry],[mapping])['gold_evidence_found']==0
    entry['included_text']=expected
    assert score_stage('context',[entry],[mapping])['gold_evidence_found']==1

def test_truncated_field_cannot_become_context_consensus():
    item=candidate('支持通信协议\nA、B',1)
    source={'file':item.chunk.source.file_path,'platform_group':'fast','explicit_scope':[]}
    item.query_plan_diagnostics={'matched_list_fields':[{'fact':'支持通信协议','values':['A','B'],'quote':item.chunk.text}],
        'source_consensus':[{'fact':'支持通信协议','supporting_sources':[source],'conflicts':[]}]}
    ctx=ContextBuilder().build([item])
    shortened=replace(ctx,entries=({**ctx.entries[0],'included_text':'支持通信协议\nA'},))
    _,consensus=attach_context_consensus(shortened)
    assert not consensus

def test_conflict_discovered_before_budget_never_silently_becomes_agreement():
    item=candidate('支持通信协议\nA、B',1)
    source={'file':item.chunk.source.file_path,'platform_group':'fast','explicit_scope':[]}
    item.query_plan_diagnostics={'matched_list_fields':[{'fact':'支持通信协议','values':['A','B'],'quote':item.chunk.text}],
        'source_consensus':[{'fact':'支持通信协议','supporting_sources':[source],
            'conflicts':[{'source':{'file':'excluded'},'values':['C']}]}]}
    ctx,consensus=attach_context_consensus(ContextBuilder().build([item]))
    assert consensus[0]['consensus_level']=='conflict'
    assert '不得取并集' in ctx.text

def test_active_suite_retired_case_absent_and_formats_equal():
    root=Path(__file__).resolve().parents[2];manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    allcases=[]
    for d in manifest['datasets']:
        p=root/d['path'];a=load_gold_dataset(p,reviewed_only=False)
        b=load_gold_dataset(p.with_name(p.stem+'_review.json'),reviewed_only=False)
        assert [c.raw for c in a.cases]==[c.raw for c in b.cases]
        assert len(a.cases)==d['count'];allcases.extend(a.cases)
    assert len(allcases)==91 and len({c.case_id for c in allcases})==91
    assert not any(c.case_id=='rag-biz-040' for c in allcases)
    reviewed=load_gold_dataset(root/'tests/rag/fixtures/rag_gold_v5_reviewed.jsonl')
    assert len(reviewed.cases)==47 and all(c.review_status=='reviewed' for c in reviewed.cases)
