from copy import deepcopy
from dataclasses import replace
import hashlib
import socket

import pytest

from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import (
    EvidenceChunk, RagChunk, RagQuery, SourceRef, QueryConstraint, ConstraintKind,
    ConstraintStatus, ConstraintJudgement, CandidateDecision,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from tools.selection_factorial import ARMS, SelectionFactors, select_context
from tools.validate_selection_factorial import no_network


def candidate(cid, text, doc=None, score=.9):
    doc = doc or cid
    source = SourceRef(1, cid, None, 1, 1, doc, doc, '.txt', (), cid)
    chunk = RagChunk(cid, doc, 0, text, text, hashlib.sha256(text.encode()).hexdigest(), source)
    return EvidenceChunk(chunk, lexical_rank=1, rerank_score=score)


def run(items, arm, *, budget=12000, k=8, vectors=None, plan=None):
    return select_context(RagQuery('如何处理？', final_top_k=k), items, plan,
                          EvidencePolicy(), budget, ARMS[arm], vectors)


def test_all_off_delegates_native_context_and_decision():
    xs = [candidate(str(i), f'步骤{i}', 'manual') for i in range(6)]
    q = RagQuery('如何处理？'); policy = EvidencePolicy()
    expected = policy.evaluate(q, deepcopy(xs))
    d, ctx, trace = run(xs, 'baseline')
    assert d == expected and ctx == ContextBuilder().build(expected.accepted)
    assert trace == []


def test_a_only_removes_document_cap_not_total_cap():
    xs = [candidate(str(i), f'步骤{i}', 'manual') for i in range(12)]
    assert len(run(xs, 'baseline')[0].accepted) == 3
    assert len(run(xs, 'A')[0].accepted) == 8


def test_all_factors_preserve_quality_and_explicit_conflict_gates():
    constraint = QueryConstraint(ConstraintKind.STAGE, 'x')
    conflict = candidate('conflict', '其他阶段')
    conflict.constraint_decision = CandidateDecision('conflict', False, '冲突',
        (ConstraintJudgement(constraint, ConstraintStatus.CONFLICT, '阶段明确冲突'),))
    good = candidate('good', '正文')
    low = candidate('low', '很低相关', score=.01)
    plan = QueryInterpreter(enhanced=False).interpret('如何处理？')
    for arm in ARMS:
        d, ctx, _ = run([conflict, good, low], arm, plan=plan, vectors={'good':[1.,0.]})
        assert [x.chunk.stable_chunk_id for x in d.accepted] == ['good']


def test_mmr_does_not_merge_similar_different_numbers_or_claim_equivalence():
    xs = [candidate('a', '允许100次'), candidate('b', '允许200次')]
    _, ctx, _ = run(xs, 'B', vectors={'a':[1.,0.], 'b':[1.,0.]})
    assert '100' in ctx.text and '200' in ctx.text and len(ctx.entries) == 2


def test_mmr_can_promote_complement_without_changing_first_choice():
    xs = [candidate(str(i), f'内容{i}') for i in range(10)]
    vs = {str(i):[1.,0.] for i in range(10)}; vs['2'] = [0.,1.]
    d, _, trace = run(xs, 'B', k=2, vectors=vs)
    assert [x.chunk.stable_chunk_id for x in d.accepted] == ['0','2']
    assert trace[1]['compared_to'] == ['0']


def test_missing_vector_fails_closed_instead_of_falling_back_to_remote():
    with pytest.raises(ValueError, match='missing or invalid'):
        run([candidate('a','正文')], 'B', vectors={})


def test_budget_skips_oversized_whole_unit_and_continues():
    xs = [candidate('long', '完整流程' * 200), candidate('short', '必要步骤')]
    _, ctx, trace = run(xs, 'C', budget=256)
    assert [e['chunk_id'] for e in ctx.entries] == ['short']
    assert not ctx.entries[0]['truncated'] and len(ctx.text) <= 256
    assert trace[0]['action'] == 'complete_chunk_exceeds_remaining_budget'


def test_c_counts_headers_and_preserves_multiple_duplicate_sources():
    xs = [candidate('a', '相同正文', 'one'), candidate('b','相同正文','two')]
    _, ctx, _ = run(xs, 'C', budget=256)
    assert len(ctx.entries) == 1 and ctx.entries[0]['source_count'] == 2
    assert len(ctx.text) <= 256


def test_c_allows_more_than_eight_short_units_with_same_budget():
    xs = [candidate(str(i), f'条件与步骤{i}') for i in range(12)]
    assert len(run(xs, 'C')[1].entries) == 12
    assert len(run(xs, 'baseline')[1].entries) == 8


def test_budget_dropped_hard_match_does_not_bypass_final_gate():
    constraint = QueryConstraint(ConstraintKind.STAGE, 'x')
    plan = replace(QueryInterpreter(enhanced=False).interpret('如何处理？'), constraints=(constraint,))
    long = candidate('long','限定阶段的完整流程' * 200)
    long.constraint_decision = CandidateDecision('long', True, '匹配',
        (ConstraintJudgement(constraint, ConstraintStatus.MATCH, '原文限定'),))
    d, ctx, _ = run([long,candidate('short','背景')], 'C', budget=256, plan=plan)
    assert d.insufficient and d.reason.startswith('required_constraint_not_confirmed') and not ctx.text


def test_network_guard_blocks_accidental_connections():
    with no_network() as attempts:
        with socket.socket() as s, pytest.raises(RuntimeError, match='disabled'):
            s.connect(('127.0.0.1', 9))
    assert len(attempts) == 1


def test_experiment_does_not_mutate_candidate_scores_or_annotations():
    xs = [candidate(str(i), f'步骤{i}') for i in range(5)]
    original = deepcopy(xs)
    vectors = {str(i):[1.,float(i)] for i in range(5)}
    for arm in ARMS:
        run(xs, arm, vectors=vectors)
    assert xs == original


def test_enabled_adapter_is_not_silently_reconfigured():
    plan = replace(QueryInterpreter(enhanced=False).interpret('如何处理？'), adapter_active=True)
    with pytest.raises(ValueError, match='adapter-disabled'):
        run([candidate('a','正文')], 'A', plan=plan)
