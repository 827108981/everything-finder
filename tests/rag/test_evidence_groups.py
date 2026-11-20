import pytest
from local_full_text_search.rag.evaluation.evidence_groups import score_evidence_groups
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch


def mapping(eid, cid, quote):
    return GoldEvidenceMatch(eid, 'positive', 'source', '1', quote, (cid,), 'exact')


def test_approved_or_group_counts_once_and_requires_actual_quote():
    group = {'id': 'checks', 'approved': True, 'alternatives': [
        [mapping('a', 'a', 'original checks')], [mapping('b', 'b', 'equivalent checks')]]}
    assert score_evidence_groups('context', [{'chunk_id': 'b', 'included_text': 'equivalent checks'}], [group])['found'] == 1
    assert score_evidence_groups('context', [{'chunk_id': 'b', 'text': 'equivalent checks', 'included_text': 'other row'}], [group])['found'] == 0
    assert score_evidence_groups('context', [{'chunk_id': 'wrong', 'included_text': 'equivalent checks'}], [group])['found'] == 0
    group['approved'] = False
    with pytest.raises(ValueError):
        score_evidence_groups('context', [], [group])


def test_multiple_required_parts_in_one_alternative_are_and():
    group = {'id': 'procedure', 'approved': True, 'alternatives': [[mapping('a', 'a', 'scene'), mapping('b', 'b', 'steps')]]}
    assert score_evidence_groups('context', [{'chunk_id': 'a', 'included_text': 'scene'}], [group])['found'] == 0
