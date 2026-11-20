from dataclasses import replace
from local_full_text_search.rag.models import RagChunk, SourceRef, EvidenceChunk
from tools.validate_evidence_selection import select_grouped


def candidate(cid,text,doc='doc',score=.8):
    source=SourceRef(1,cid,None,1,1,doc,doc,'.pptx',(),cid)
    return EvidenceChunk(RagChunk(cid,doc,0,text,text,cid,source),rerank_score=score,rerank_score_combined=score)


def test_linked_pages_are_atomic_and_preserve_original_scores():
    a=candidate('title','故障场景',score=.9);b=candidate('body','原因与处理步骤',score=.8)
    assert select_grouped([a,b],{'title':[['body','场景','页1']]},limit=1)[0]==[]
    selected,_=select_grouped([a,b],{'title':[['body','场景','页1']]},limit=2)
    assert selected==[a,b] and (a.rerank_score,b.rerank_score)==(.9,.8)


def test_relationship_never_crosses_document_versions():
    a=candidate('a','甲',doc='v1');b=candidate('b','乙',doc='v2')
    selected,trace=select_grouped([a,b],{'a':[['b','','']]},limit=1)
    assert len(selected)==1 and not trace[0]['grouped']


def test_numeric_unit_and_negation_variants_are_not_deduplicated():
    xs=[candidate('a','流量140±5 μL/s'),candidate('b','流量145±5 μL/s'),candidate('c','不可更换')]
    selected,_=select_grouped(xs,{},limit=3)
    assert {x.chunk.stable_chunk_id for x in selected}=={'a','b','c'}


def test_exact_duplicates_do_not_consume_extra_slots():
    xs=[candidate('a','同样正文'),candidate('b','同样正文',doc='other'),candidate('c','不同事实')]
    selected,_=select_grouped(xs,{},limit=3)
    assert len(selected)==2


def test_deterministic_and_count_limited():
    xs=[candidate(str(i),f'操作步骤{i}',score=1-i/100) for i in range(12)]
    a,_=select_grouped(xs,{},limit=8);b,_=select_grouped(xs,{},limit=8)
    assert [x.chunk.stable_chunk_id for x in a]==[x.chunk.stable_chunk_id for x in b]
    assert len(a)==8
