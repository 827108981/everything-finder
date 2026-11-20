"""Question-only conditional recall candidate, opt-in; no corpus/Gold access.

Additional routes use the existing per-channel budgets. Each channel keeps its
best rank across routes, then the unchanged two-channel RRF formula is used.
This is explicitly *not* extra votes for documents repeated across branches.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import json
import re
from typing import Callable

from local_full_text_search.rag.models import EvidenceChunk


PLANNER_PROMPT = """你只从用户原话中摘取检索片段，不能回答问题，不能读取或猜测资料。
目标：用户明确比较多个条件时，分别检索每个条件。返回严格 JSON：
{"shared_fragments":[],"branches":[{"condition_quote":"连续原文条件",
"fragments":["连续原文片段", "必要时继承原问省略的指标片段"]}]}。
要求：
1. 仅两个或三个明确条件时输出；否则 branches=[]。同一条件的重复说法不算多个条件。
2. 每个字符串必须是用户原话连续子串。片段可重排，但不能改词或补充原因、部件、答案。
3. 为每个分支保留决定答案的肯定、否定、比较、范围、对象及阶段；不得扩大其范围。
4. 后续分支省略了共同指标时，可摘取前文该指标；不可把前一个分支的具体例子、
局部对象或局部条件带到条件已改变的后续分支。全局明确的机型版本等才放 shared_fragments。
5. fragments 是自包含、简洁的检索语句片段；去掉问句套话，但不丢业务条件。
6. 不确定继承是否正确就返回空分支，不推测。不要输出解释。"""


@dataclass(frozen=True)
class Branch:
    branch_id: str
    condition_quote: str
    query_text: str


def eligible(question: str) -> bool:
    """Conservative rollout scope: explicit follow-on conditional question only."""
    return bool(re.search(r"[？?；;。]\s*(?:如果|假如|若|相反|而当|当)", question))


def validate_plan(question: str, value: dict) -> tuple[Branch, ...]:
    if not eligible(question) or not isinstance(value, dict):
        return ()
    raw = value.get("branches")
    shared = value.get("shared_fragments", [])
    if not isinstance(raw, list) or not 2 <= len(raw) <= 3 or not isinstance(shared, list):
        return ()
    if any(not isinstance(x, str) or not x.strip() or x not in question for x in shared):
        return ()
    # Multiple explicit model/version identifiers require a richer scoped plan.
    # Falling back is safer than transferring one model's condition to another.
    identifiers = set(re.findall(r"[A-Za-z]+[-_]?\d+[A-Za-z\d.\-]*", question))
    if len(identifiers) > 1:
        return ()
    branches = []
    seen = set()
    conditions = set()
    for row in raw:
        if not isinstance(row, dict):
            return ()
        condition = row.get("condition_quote")
        fragments = row.get("fragments")
        if (not isinstance(condition, str) or len(condition.strip()) < 3 or condition not in question
                or not isinstance(fragments, list) or not 1 <= len(fragments) <= 12):
            return ()
        if any(not isinstance(x, str) or not x.strip() or x not in question for x in fragments):
            return ()
        text = " ".join(dict.fromkeys(shared + fragments))
        if not 4 <= len(text) <= 220 or any(identifier not in text for identifier in identifiers):
            return ()
        key = re.sub(r"\W", "", text).casefold()
        condition_key = re.sub(r"\W", "", condition).casefold()
        if key in seen or condition_key in conditions:
            continue
        seen.add(key)
        conditions.add(condition_key)
        branches.append(Branch(f"branch_{len(branches) + 1}", condition, text))
    return tuple(branches) if len(branches) >= 2 else ()


def plan_branches(question: str, complete: Callable[[str, str], str]):
    audit = {"triggered": False, "branches": [], "fallback": False, "reason": "no_explicit_follow_on_condition"}
    if not eligible(question):
        return (), audit
    try:
        raw = complete(PLANNER_PROMPT, question).strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
        value = json.loads(raw)
        branches = validate_plan(question, value)
        audit.update(raw_plan=value, triggered=bool(branches), fallback=not bool(branches),
                     reason="validated_extractive_branches" if branches else "unverifiable_or_insufficient_branches",
                     branches=[vars(b) for b in branches])
        return branches, audit
    except Exception as exc:
        # Do not include provider errors, URLs, or credentials in the candidate audit.
        audit.update(fallback=True, reason="planner_failed", error_type=type(exc).__name__)
        return (), audit


def merge_routes(original: list[EvidenceChunk], routes: list[list[EvidenceChunk]], *,
                 rrf_k: int, limit: int):
    merged = {item.chunk.stable_chunk_id: copy.deepcopy(item) for item in original}
    for route in routes:
        for item in route:
            key = item.chunk.stable_chunk_id
            target = merged.setdefault(key, copy.deepcopy(item))
            for channel in ("lexical", "vector"):
                rank = getattr(item, channel + "_rank")
                old = getattr(target, channel + "_rank")
                if rank is None or (old is not None and old <= rank):
                    continue
                for field in (("lexical_rank", "lexical_score", "lexical_score_kind", "lexical_contributions",
                               "lexical_route_hits", "lexical_route_rrf_score") if channel == "lexical" else
                              ("vector_rank", "vector_score", "vector_route_hits", "vector_route_rrf_score", "vector_unit_hits")):
                    setattr(target, field, copy.deepcopy(getattr(item, field)))
    for item in merged.values():
        item.fused_score = sum(1.0 / (rrf_k + rank) for rank in (item.lexical_rank, item.vector_rank)
                               if rank is not None)
    ordered = sorted(merged.values(), key=lambda item: (-item.fused_score, item.chunk.stable_chunk_id))
    for rank, item in enumerate(ordered, 1):
        item.fused_rank = rank
    return ordered[:limit], {
        "budget": {"before": len(original), "union": len(ordered), "after": min(limit, len(ordered)), "limit": limit},
        "added_candidate_ids": [x.chunk.stable_chunk_id for x in ordered[:limit]
                                if x.chunk.stable_chunk_id not in {i.chunk.stable_chunk_id for i in original}],
        "removed_candidate_ids": [x.chunk.stable_chunk_id for x in original
                                  if x.chunk.stable_chunk_id not in {i.chunk.stable_chunk_id for i in ordered[:limit]}],
        "excluded": [{"chunk_id": x.chunk.stable_chunk_id, "reason": "unchanged_fusion_budget", "rank": x.fused_rank}
                     for x in ordered[limit:]],
    }


def retrieve_with_branches(query, original, branches, retrieve, interpret, token, *, rrf_k):
    """All or nothing: route failure returns baseline, without partially promoted ranks."""
    from dataclasses import replace
    audit = {"triggered": bool(branches), "branches": [], "fallback": False,
             "added_candidate_ids": [], "removed_candidate_ids": [],
             "budget": {"before": len(original), "after": len(original)}}
    if not branches:
        return original, audit
    try:
        routes = []
        for branch in branches:
            branch_query = replace(query, text=branch.query_text)
            hits = retrieve(branch_query, token, interpret(branch.query_text, token))
            routes.append(hits)
            audit["branches"].append({**vars(branch), "candidate_ids": [h.chunk.stable_chunk_id for h in hits],
                                      "reason": "independent_question_condition; same route budgets and RRF"})
        merged, details = merge_routes(original, routes, rrf_k=rrf_k, limit=max(1, query.fusion_top_k) * 2)
        audit.update(details)
        return merged, audit
    except Exception as exc:
        if token.cancelled:
            raise
        audit.update(fallback=True, reason="branch_route_failed", error_type=type(exc).__name__)
        return original, audit
