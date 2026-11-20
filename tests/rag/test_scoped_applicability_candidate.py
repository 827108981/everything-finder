from dataclasses import replace

from local_full_text_search.rag.models import EvidenceChunk, RagChunk, SourceRef
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from tools.scoped_applicability_candidate import ScopedApplicability
from tools.scoped_branch_planner import validate


def evidence(key, body, score):
    ref = SourceRef(1, key, 1, 1, 1, "E:/doc.pdf", "doc.pdf", ".pdf", (1,), "page 1", page_number=1)
    return EvidenceChunk(chunk=RagChunk(key, key, 1, body, body, key, ref),
                         rerank_score=score, rerank_score_combined=score, vector_score=0.8)


def test_unknown_keeps_relevance_order_but_real_conflicts_reject():
    plan = QueryInterpreter().interpret("所有项目重复性异常，怎么排查？")
    items = [evidence("direct", "如果项目都不好，可以检查公共原因。", .95),
             evidence("list", "所有项目样本申请清单。", .3),
             evidence("conflict", "只有一个项目重复性不好。", .9)]
    result = ScopedApplicability(mode="O").evaluate(plan, items)
    assert [x.chunk.stable_chunk_id for x in result[:2]] == ["direct", "list"]
    assert not result[-1].constraint_decision.accepted
    assert result[0].constraint_decision.required_unknown_count > 0


def test_follow_on_scope_is_local_to_branch_and_final_policy():
    question = "单个项目失常，怎么查？如果所有项目失常呢？"
    plan = QueryInterpreter().interpret(question)
    candidate = ScopedApplicability(("单个项目失常", "所有项目失常"), mode="C")
    items = [evidence("single", "只有一个项目失常时，先查局部原因。", .9),
             evidence("all", "全部项目失常时，先查公共原因。", .8)]
    result = candidate.evaluate(plan, items)
    assert all(x.constraint_decision.accepted for x in result)
    assert not any(c.kind.value == "scope" for c in candidate.policy_plan(plan).constraints)
    assert {b["branch"] for b in candidate.last_bindings} == {1, 2}


def test_no_branch_falls_back_to_original_plan():
    plan = QueryInterpreter().interpret("所有项目异常？")
    candidate = ScopedApplicability((), mode="C")
    assert candidate.policy_plan(plan) is plan


def test_elliptical_branch_requires_explicit_inheritance():
    q = "低剂量时响应变差，先看哪？如果高剂量时也变差呢？"
    rows = [{"condition_quote": "低剂量时响应变差", "fragments": ["低剂量时响应变差"], "inherited": []},
            {"condition_quote": "高剂量时也变差", "fragments": ["高剂量时也变差"], "inherited": []}]
    assert validate(q, {"branches": rows}) == ()
    rows[1]["inherited"] = [{"quote": "响应", "role": "metric", "from_quote": "低剂量时响应变差"}]
    result = validate(q, {"branches": rows})
    assert result[1].query_text.endswith("响应")
