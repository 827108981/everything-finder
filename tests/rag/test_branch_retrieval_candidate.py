from dataclasses import asdict, replace

import pytest

from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import EvidenceChunk, RagChunk, RagQuery, SourceRef
from tools.branch_retrieval_candidate import Branch, merge_routes, plan_branches, retrieve_with_branches, validate_plan


def hit(key, *, lexical=None, vector=None):
    source = SourceRef(1, key, 1, 1, 1, "E:/manual.pdf", "manual.pdf", ".pdf", (1,), "page 1", page_number=1)
    chunk = RagChunk(key, key, 0, key, key, key, source)
    return EvidenceChunk(chunk=chunk, lexical_rank=lexical, vector_rank=vector)


QUESTION = "M900 电机低速时振动，优先查哪里？如果高速时振动呢？"
PLAN = {"shared_fragments": ["M900"], "branches": [
    {"condition_quote": "低速时振动", "fragments": ["电机", "低速时振动"]},
    {"condition_quote": "高速时振动", "fragments": ["电机", "高速时振动"]},
]}


def test_single_question_does_not_call_planner():
    def forbidden(*_):
        raise AssertionError("Planner should not run")
    branches, audit = plan_branches("电机为什么振动？", forbidden)
    assert branches == () and not audit["triggered"]


def test_condition_queries_are_source_only_and_keep_global_model():
    branches = validate_plan(QUESTION, PLAN)
    assert [b.query_text for b in branches] == ["M900 电机 低速时振动", "M900 电机 高速时振动"]


def test_new_answer_words_are_rejected():
    value = {**PLAN, "branches": [PLAN["branches"][0], {"condition_quote": "高速时振动", "fragments": ["轴承损坏"]}]}
    assert not validate_plan(QUESTION, value)


def test_repeated_condition_does_not_create_independent_routes():
    assert not validate_plan(QUESTION, {**PLAN, "branches": [PLAN["branches"][0]] * 2})


def test_omitted_model_and_multiple_local_models_fall_back():
    assert not validate_plan(QUESTION, {**PLAN, "shared_fragments": []})
    assert not validate_plan(QUESTION + "M800也一样吗？", PLAN)


def test_planner_failure_and_invalid_json_do_not_change_baseline():
    def broken(*_):
        raise RuntimeError("provider unreachable")
    for complete in (broken, lambda *_: "not json"):
        branches, audit = plan_branches(QUESTION, complete)
        assert not branches and audit["fallback"]
        assert "provider unreachable" not in str(audit)


def test_duplicate_routes_are_not_extra_rrf_votes():
    original = [hit("base", lexical=1, vector=1)]
    extra = [hit("other", lexical=2, vector=2)]
    once, _ = merge_routes(original, [extra], rrf_k=60, limit=60)
    repeated, _ = merge_routes(original, [extra] * 3, rrf_k=60, limit=60)
    assert [(x.chunk.stable_chunk_id, x.fused_score) for x in once] == [
        (x.chunk.stable_chunk_id, x.fused_score) for x in repeated]
    assert original[0].fused_score == 0  # No mutation leaks back into baseline.


def test_best_rank_merges_channels_without_replacing_source():
    original = [hit("same", lexical=8)]
    result, _ = merge_routes(original, [[hit("same", lexical=3, vector=5)]], rrf_k=60, limit=60)
    assert result[0].lexical_rank == 3 and result[0].vector_rank == 5
    assert result[0].fused_score == pytest.approx(1 / 63 + 1 / 65)
    assert result[0].chunk.source == original[0].chunk.source


def test_full_budget_records_every_eviction():
    result, audit = merge_routes([hit("base", lexical=40)], [[hit("extra", lexical=1, vector=1)]], rrf_k=60, limit=1)
    assert [x.chunk.stable_chunk_id for x in result] == ["extra"]
    assert audit["removed_candidate_ids"] == ["base"]
    assert audit["excluded"][0]["reason"] == "unchanged_fusion_budget"


def test_no_branch_returns_identical_baseline_object():
    original = [hit("base", lexical=1)]
    result, audit = retrieve_with_branches(RagQuery("one"), original, (), None, None, CancelToken(), rrf_k=60)
    assert result is original and not audit["triggered"]


def test_query_filters_and_all_budgets_are_preserved():
    query = RagQuery(QUESTION, root_ids=(7,), extensions=(".pdf",), include_ocr=False)
    seen = []
    def retrieve(q, *_):
        seen.append(q)
        return [hit(q.text, lexical=1)]
    result, audit = retrieve_with_branches(query, [hit("base", lexical=1)], validate_plan(QUESTION, PLAN),
                                         retrieve, lambda *_: None, CancelToken(), rrf_k=60)
    assert len(seen) == 2 and len(result) == 3 and not audit["fallback"]
    assert all(asdict(replace(q, text=QUESTION)) == asdict(query) for q in seen)


def test_partial_route_failure_is_atomic_fallback():
    original = [hit("base", lexical=1)]
    calls = []
    def retrieve(*_):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("failed")
        return [hit("partial", lexical=1, vector=1)]
    result, audit = retrieve_with_branches(RagQuery(QUESTION), original, validate_plan(QUESTION, PLAN),
                                         retrieve, lambda *_: None, CancelToken(), rrf_k=60)
    assert result is original and audit["fallback"] and not audit["added_candidate_ids"]


def test_cancel_is_not_silently_converted_to_fallback():
    token = CancelToken()
    def retrieve(*_):
        token.cancel()
        token.throw_if_cancelled()
    with pytest.raises(CancelledError):
        retrieve_with_branches(RagQuery(QUESTION), [], (Branch("b", "低速", "低速振动"),),
                               retrieve, lambda *_: None, token, rrf_k=60)
