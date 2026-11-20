"""Experimental route coverage under the unchanged fusion candidate budget."""

from __future__ import annotations

import copy
from collections import defaultdict


def merge_with_route_coverage(original, routes, *, limit=60, per_route=10):
    """Reserve consideration for each question route, retaining original order.

    The routes are question-derived, not selected by file, page, or Gold labels.
    Retrieval scores are left intact; later stages still perform their own checks.
    """
    limit = max(1, int(limit))
    per_route = max(1, int(per_route))
    chosen = []
    seen = set()
    audit = []
    for route_id, candidates in routes.items():
        admitted = 0
        for item in candidates:
            if admitted >= per_route or len(chosen) >= limit:
                break
            cid = item.chunk.stable_chunk_id
            if cid in seen:
                continue
            selected = copy.deepcopy(item)
            chosen.append(selected)
            seen.add(cid)
            admitted += 1
            audit.append({"route": route_id, "chunk_id": cid,
                          "route_rank": candidates.index(item) + 1})
    for item in original:
        if len(chosen) >= limit:
            break
        cid = item.chunk.stable_chunk_id
        if cid in seen:
            continue
        chosen.append(copy.deepcopy(item))
        seen.add(cid)
    for rank, item in enumerate(chosen, 1):
        item.fused_rank = rank
    return chosen, audit
