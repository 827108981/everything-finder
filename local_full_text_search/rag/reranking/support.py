"""通用的候选答案支持度计算。

这个模块不维护领域词表。它只使用问题解析器已经提取出的实体、检索词、
硬条件和问题意图，给重排模型补充“正文是否能直接回答问题”的信号。
"""

from __future__ import annotations

import re

from local_full_text_search.rag.models import EvidenceChunk, QueryPlan


_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[，。！？；：、“”‘’（）()\[\]{}<>《》,.;:!?\\/]+")

# 这些是问句关系的通用表达，不包含设备、专业或测试集词项。
_INTENT_MARKERS: dict[str, tuple[str, ...]] = {
    "purpose": ("用于", "作用", "功能", "负责", "目的", "实现"),
    "cause": ("因为", "由于", "原因", "导致", "机理", "原理"),
    "parameter": ("参数", "规格", "范围", "标准", "数量", "多少", "数值", "单位"),
    "procedure": ("步骤", "操作", "检查", "确认", "处理", "排查", "更换", "设置"),
    "location": ("位置", "安装", "放置", "位于", "所在"),
    "definition": ("是指", "定义", "含义", "表示", "是什么", "包括"),
}


def _compact(value: str) -> str:
    return _SPACE_RE.sub("", str(value or "")).casefold()


def _content_terms(plan: QueryPlan) -> tuple[str, ...]:
    values: list[str] = []
    for value in (*plan.entities, *plan.lexical_terms):
        compact = _compact(value)
        if len(compact) < 2 or compact in values:
            continue
        values.append(compact)
    return tuple(values)


def infer_question_intent(plan: QueryPlan) -> str:
    """为旧 QueryPlan 补齐更细的问题关系类型。"""

    if plan.intent not in {"general", "explanation"}:
        return plan.intent
    text = _compact(plan.original_text)
    if any(word in text for word in ("作用", "用途", "功能", "负责")):
        return "purpose"
    if any(word in text for word in ("原因", "为什么", "由于", "导致", "原理", "机理")):
        return "cause"
    if any(word in text for word in ("多少", "参数", "规格", "范围", "标准", "数值")):
        return "parameter"
    if any(word in text for word in ("步骤", "如何", "怎么", "怎样", "操作", "排查", "处理")):
        return "procedure"
    if any(word in text for word in ("位置", "安装在哪里", "放在哪里", "位于")):
        return "location"
    if plan.intent == "explanation":
        return "definition"
    return "general"


def compute_answer_support(plan: QueryPlan, item: EvidenceChunk) -> tuple[float, tuple[str, ...]]:
    """计算 0~1 的通用答案支持度，并返回可解释原因。

    实体/内容词覆盖反映候选是否在谈同一个对象；意图关系词反映正文是否
    出现了与问题相匹配的事实表达。它只作为重排辅助分，不单独拒绝候选。
    """

    text = _compact(item.chunk.text)
    terms = _content_terms(plan)
    if not text or not terms:
        return 0.0, ("没有可计算的内容词覆盖",)
    matched = [term for term in terms if term in text]
    coverage = min(1.0, sum(min(len(term), 8) for term in matched) /
                   max(1.0, sum(min(len(term), 8) for term in terms)))
    reasons = [f"内容词覆盖 {len(matched)}/{len(terms)}"]
    relation = infer_question_intent(plan)
    markers = _INTENT_MARKERS.get(relation, ())
    marker_hits = tuple(marker for marker in markers if _compact(marker) in text)
    sentences = [part for part in re.split(r"[。！？；;\n]", text) if part]
    direct_relation = any(
        any(_compact(marker) in sentence for marker in markers)
        and any(term in sentence for term in matched)
        for sentence in sentences
    )
    relation_score = (
        1.0 if direct_relation else
        0.5 if marker_hits else
        0.0
    )
    if marker_hits:
        reasons.append(
            f"意图={relation}，正文含" + "、".join(marker_hits[:3])
            + ("，且与内容词同句" if direct_relation else "，但未与内容词同句")
        )
    else:
        reasons.append(f"意图={relation}，未发现对应关系表达")
    # 支持度作为辅助奖励而不是硬门禁：只有关系表达和问题对象出现在同一
    # 句子时才提供明显加分，避免标题、目录或维修步骤中的泛化词误抬排名。
    score = 0.35 + 0.15 * coverage + 0.50 * relation_score
    return min(1.0, score), tuple(reasons)


def build_rerank_query(plan: QueryPlan) -> str:
    """给 Cross-Encoder 的通用查询表示，保留原问题并补充已解析结构。"""

    intent = infer_question_intent(plan)
    constraints = "、".join(
        f"{item.kind.value}={item.value}"
        for item in plan.constraints
        if item.is_hard
    ) or "无明确硬条件"
    return (
        f"问题：{plan.original_text}\n"
        f"回答关系：{intent}\n"
        f"问题条件：{constraints}\n"
        "判断候选正文是否直接支持这个问题的答案。"
    )


def retrieval_prior(item: EvidenceChunk) -> float:
    """把已有阶段名次转换成 0~1 的先验，缺失的通道不参与平均。"""

    ranks = [item.lexical_rank, item.vector_rank, item.fused_rank]
    values = [1.0 / max(1, int(rank)) for rank in ranks if rank is not None]
    return sum(values) / len(values) if values else 0.0
