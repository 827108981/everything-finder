"""Runtime simplification: no business condition gates or hidden fragment caps."""
from copy import deepcopy

import pytest

from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import SimpleEvidencePolicy
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from tests.rag.test_selection_factorial import candidate


def test_simple_selection_does_not_filter_or_resort_by_conditions_or_scores():
    plan = QueryInterpreter().interpret('单个项目样本检测偏低怎么处理？')
    xs = [candidate('conflict', '全部项目定标失败。', score=.001),
          candidate('unknown', '检查连接。', score=.0001)]
    xs = list(ConstraintMatcher(preserve_order=True).evaluate(plan, xs))
    xs[0].lexical_rank = None
    xs[0].vector_score = .001
    before = deepcopy(xs)
    d = SimpleEvidencePolicy().evaluate(RagQuery(plan.original_text, final_top_k=1), xs, plan=plan)
    assert d.accepted == tuple(xs) and not d.insufficient and not d.rejected
    assert xs == before


def test_no_fixed_count_or_document_quota_and_zero_generation():
    xs = [candidate(str(i), f'步骤{i}。', 'same_document') for i in range(12)]
    d = SimpleEvidencePolicy().evaluate(RagQuery('怎么处理？', final_top_k=1), xs)
    ctx = ContextBuilder().build(d.accepted, preserve_complete_chunks=True)
    assert len(ctx.entries) == 12
    assert [e['chunk_id'] for e in ctx.entries] == [str(i) for i in range(12)]


def test_budget_skips_oversized_units_without_partial_instructions():
    xs = [candidate('oversize', '第一步检查，第二步排查，第三步确认。' * 100),
          candidate('fits', '正常电压24V，允许范围22.8～25.2V。')]
    ctx = ContextBuilder().build(xs, max_chars=256, preserve_complete_chunks=True)
    assert [e['chunk_id'] for e in ctx.entries] == ['fits']
    assert not ctx.entries[0]['truncated'] and len(ctx.text) <= 256
    assert ctx.excluded_entries[0]['reason_code'] == 'complete_chunk_exceeds_remaining_budget'


@pytest.mark.parametrize('budget', [0, 1, 40, 256, 800])
def test_budget_counts_source_headers_and_never_truncates(budget):
    xs = [candidate(str(i), f'完整条件和操作步骤{i}。') for i in range(10)]
    ctx = ContextBuilder().build(xs, max_chars=budget, preserve_complete_chunks=True)
    assert len(ctx.text) <= budget
    assert all(not e['truncated'] for e in ctx.entries)


def test_duplicate_body_keeps_every_source_without_merging_distinct_numbers():
    xs = [candidate('a', '电压24V。', 'manual_a'), candidate('b', '电压24V。', 'manual_b'),
          candidate('c', '电压25V。', 'manual_c')]
    ctx = ContextBuilder().build(xs, preserve_complete_chunks=True)
    assert len(ctx.entries) == 2
    assert ctx.entries[0]['source_count'] == 2
    assert len(ctx.entries[0]['sources']) == 2
    assert '24V' in ctx.text and '25V' in ctx.text


def test_empty_candidates_remain_insufficient():
    assert SimpleEvidencePolicy().evaluate(RagQuery('怎么处理？'), []).insufficient


def test_old_persisted_switch_cannot_reenable_runtime_condition_filter(indexed_rag):
    # The fixture is imported below; no network or generation is involved.
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.rag.secrets import MemorySecretStore
    from local_full_text_search.core.task_manager import CancelToken
    from tests.rag.test_rag_end_to_end import _Embedding, _Reranker, _Gateway
    store, rag_store, settings = indexed_rag
    settings.rag_constraint_matching_enabled = True
    settings.rag_min_rerank_score = .99
    events = []
    class MustNotCheck:
        def evaluate(self, *args, **kwargs):
            raise AssertionError('Condition filtering must not run')
    gateway = _Gateway()
    service = AnswerService(store, rag_store, _Embedding(), _Reranker(.001), gateway,
                            MemorySecretStore(), settings, constraint_matcher=MustNotCheck(),
                            progress_callback=events.append)
    result = service.retrieve(RagQuery('只有一个项目的索引任务中断后如何恢复？'), CancelToken())
    assert result.sources and gateway.calls == 0
    event = next(e for e in events if e['stage'] == 'checking_constraints')
    assert not event['condition_check_enabled'] and event['condition_judgement_mode'] == 'disabled'
    assert not next(e for e in events if e['stage'] == 'checking_evidence')['fixed_top_k_applied']


from tests.rag.test_rag_end_to_end import indexed_rag
