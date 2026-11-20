from dataclasses import replace
import json

import pytest

from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset, GoldDatasetError
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.reporting.matching import match_quote


def case(tmp_path, approved=True, alternate_quote='流量 12 mL/min'):
    row=dict(id='generic-1', question='泵甲流量是多少？', group_id='generic-flow',
        expected_behavior='answer', review_status='reviewed', constraints={},
        gold_evidence=[dict(file='E:/docs/a.pdf',page=2,quote='流量 12 mL/min',
            alternative_sources=[dict(file='E:/docs/b.pptx',page=4,quote=alternate_quote,approved=approved)])])
    path=tmp_path/'gold.jsonl'
    path.write_text(json.dumps(row),encoding='utf8')
    return load_gold_dataset(path).cases[0]


def row(cid, file, text, page):
    return dict(chunk_id=cid,ordinal=0,file_path=file,filename=file.split('/')[-1],
                text=text,page_number=page,location=f'第 {page} 页')


def test_primary_or_approved_source_counts_once_and_report_agrees(tmp_path):
    c=case(tmp_path)
    a=row('a','E:/docs/a.pdf','泵甲流量 12 mL/min',2)
    b=row('b','E:/docs/b.pptx','泵甲流量 12 mL/min',4)
    mappings=map_case_evidence(c,[a,b])
    for candidates in ([a],[b],[a,b]):
        score=score_stage('vector',candidates,mappings)
        assert score['gold_evidence_found']==score['gold_evidence_total_declared']==1
    assert match_quote(b,c.gold_evidence[0],kind='positive',evidence_id='E1').status=='exact'


def test_approved_source_still_requires_path_page_and_full_quote(tmp_path):
    c=case(tmp_path)
    b=row('b','E:/docs/b.pptx','泵甲流量 12 mL/min',4)
    for wrong in [dict(b,file_path='E:/other/b.pptx'),dict(b,page_number=9,location='第 9 页'),dict(b,text='流量 12')]:
        mappings=map_case_evidence(c,[wrong])
        assert score_stage('vector',[wrong],mappings)['gold_evidence_found']==0
    assert match_quote(dict(b,file_path='E:/other/b.pptx'),c.gold_evidence[0],kind='positive',evidence_id='E1').status=='unmatched'


def test_missing_primary_does_not_block_explicit_alternative(tmp_path):
    c=case(tmp_path)
    b=row('b','E:/docs/b.pptx','流量 12 mL/min',4)
    assert score_stage('vector',[b],map_case_evidence(c,[b]))['gold_evidence_recall']==1


@pytest.mark.parametrize('approved,quote',[(False,'流量 12 mL/min'),(True,'流量 24 mL/min')])
def test_unapproved_or_different_fact_rejected(tmp_path,approved,quote):
    with pytest.raises(GoldDatasetError):case(tmp_path,approved,quote)


def test_two_different_facts_cannot_be_satisfied_by_one_alternative(tmp_path):
    c=case(tmp_path)
    c=replace(c,gold_evidence=(*c.gold_evidence,dict(file='E:/docs/c.pdf',page=2,quote='额定电压 24 V')))
    b=row('b','E:/docs/b.pptx','流量 12 mL/min',4)
    mappings=map_case_evidence(c,[b])
    score=score_stage('vector',[b],mappings)
    assert score['gold_evidence_found']==1 and score['gold_evidence_total_declared']==2
