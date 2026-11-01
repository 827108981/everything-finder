"""将逐题阶段事件转换为互不重叠的耗时区间。"""

from __future__ import annotations

from datetime import datetime
from typing import Any


MODULE_LABELS = {
    "setup": "索引检查与准备",
    "query": "问题解析",
    "lexical": "关键词召回",
    "vector": "向量召回",
    "fusion_dedup": "融合与去重",
    "rerank": "本地重排",
    "constraints": "业务条件检查",
    "evidence": "证据筛选",
    "context": "上下文构建",
    "api_setup": "LLM 请求准备",
    "generation": "LLM 生成及重试",
    "review": "答案二次复核",
    "gold_judge": "金标答案裁判",
    "validation": "引用校验与收尾",
    "other": "其他处理",
}

_EVENT_MODULE = {
    "checking_index": "setup",
    "query_interpreting": "query",
    "query_interpreted": "query",
    "clarification_check": "query",
    "retrieving": "setup",
    "lexical_retrieval": "lexical",
    "vector_retrieval": "vector",
    "fusion": "fusion_dedup",
    "deduplicating": "fusion_dedup",
    "reranking": "rerank",
    "reranked": "constraints",
    "checking_constraints": "evidence",
    "checking_evidence": "evidence",
    "building_context": "context",
    "credential_read": "api_setup",
    "generating": "generation",
    "generation_retry": "generation",
    "draft_generated": "validation",
    "reviewing": "review",
    "review_retry": "review",
    "review_completed": "validation",
    "validating": "validation",
    "gold_judging": "gold_judge",
    "gold_judge_retry": "gold_judge",
    "gold_judged": "validation",
    "completed": "other",
    "retrieval_completed": "other",
}


def case_timing(result: dict[str, Any]) -> dict[str, Any]:
    """以事件起点到下一事件起点计时；失败题的尾段算到异常返回。"""

    duration = result.get("duration_ms")
    if duration is None:
        return {"total_ms": None, "modules_ms": {}, "measured": False}
    total = max(0, int(duration))
    events = [event for event in result.get("events", [])
              if isinstance(event, dict) and event.get("elapsed_ms") is not None]
    points = [(str(event.get("stage") or ""), max(0, min(total, int(event["elapsed_ms"]))))
              for event in events]
    modules = {name: 0 for name in MODULE_LABELS}
    if not points:
        modules["other"] = total
    else:
        modules["setup"] += points[0][1]
        for index, (stage, start) in enumerate(points):
            end = points[index + 1][1] if index + 1 < len(points) else total
            modules[_EVENT_MODULE.get(stage, "other")] += max(0, end - start)
    return {
        "total_ms": total,
        "modules_ms": {name: value for name, value in modules.items() if value},
        "measured": bool(points),
    }


def run_timing(run: dict[str, Any], results: list[dict[str, Any]]) -> dict[str, Any]:
    modules = {name: 0 for name in MODULE_LABELS}
    cases_ms = 0
    measured = 0
    for result in results:
        timing = case_timing(result)
        if timing["total_ms"] is None:
            continue
        cases_ms += timing["total_ms"]
        measured += 1
        for name, value in timing["modules_ms"].items():
            modules[name] += value
    wall_ms: int | None = None
    try:
        start = datetime.fromisoformat(str(run["started_at"]))
        end = datetime.fromisoformat(str(run["finished_at"]))
        wall_ms = max(0, int((end - start).total_seconds() * 1000))
    except (KeyError, TypeError, ValueError):
        pass
    preparation = run.get("preparation_duration_ms")
    preparation_ms = max(0, int(preparation)) if preparation is not None else None
    return {
        "wall_ms": wall_ms,
        "preparation_ms": preparation_ms,
        "overall_recorded_ms": wall_ms + preparation_ms if wall_ms is not None and preparation_ms is not None else None,
        "case_total_ms": cases_ms,
        "outside_cases_ms": max(0, wall_ms - cases_ms) if wall_ms is not None else None,
        "measured_case_count": measured,
        "modules_ms": {name: value for name, value in modules.items() if value},
        "module_labels": dict(MODULE_LABELS),
        "definition": "逐题耗时按相邻阶段事件的时间差归属；融合与去重合并展示。新版运行另记索引预检和本地模型准备耗时；旧报告缺这段原始计时，不能事后推算。报告文件写入耗时尚未纳入。",
    }
