from copy import deepcopy

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import RagQuery
from tests.rag.test_selection_factorial import candidate


def service(settings=None, **kwargs):
    return AnswerService(None, None, None, None, None, None, settings or AppSettings(), **kwargs)


def test_application_defaults_ignore_legacy_saved_quota_and_final_count():
    settings = AppSettings(); settings.rag_max_chunks_per_document = 1
    policy = service(settings).evidence_policy
    xs = [candidate(str(i), f'互补步骤{i}', 'same_document') for i in range(12)]
    assert policy.max_per_document is None
    assert len(policy.evaluate(RagQuery('如何处理？'), xs).accepted) == 12


def test_application_policy_retains_bge_order_without_score_cutoffs():
    settings = AppSettings(); settings.rag_min_rerank_score = .2
    current = service(settings)
    xs = [candidate(str(i), f'步骤{i}', 'same_document') for i in range(10)]
    xs.insert(0, candidate('low', '相关性不足', score=.1))
    query = RagQuery('如何处理？', final_top_k=5)
    actual = current.evidence_policy.evaluate(query, xs)
    assert [x.chunk.stable_chunk_id for x in actual.accepted] == [x.chunk.stable_chunk_id for x in xs]
    assert not actual.rejected and current.evidence_policy.preserve_complete_chunks


def test_explicit_policy_injection_remains_available_for_frozen_baselines():
    old = EvidencePolicy(max_per_document=2)
    assert service(evidence_policy=old).evidence_policy is old
    xs = [candidate(str(i), f'步骤{i}', 'same_document') for i in range(4)]
    assert len(old.evaluate(RagQuery('如何处理？'), xs).accepted) == 2


def test_adoption_does_not_enable_domain_adapter_or_change_settings():
    settings = AppSettings(); settings.rag_query_plan_enabled = False
    before = deepcopy(settings)
    current = service(settings)
    assert settings == before
    assert current.query_interpreter.interpret('如何处理？').adapter_active is False
