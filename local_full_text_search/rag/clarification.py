"""判断问题是否缺少会改变处理路径的关键信息。"""

from __future__ import annotations

from dataclasses import dataclass

from local_full_text_search.rag.models import QueryPlan


@dataclass(frozen=True, slots=True)
class ClarificationDecision:
    required: bool
    question: str = ""
    missing_fields: tuple[str, ...] = ()
    reason: str = ""


class ClarificationPolicy:
    """避免只因缺少型号或阶段就打断用户；仅拦截明显无法检索的问题。"""

    def evaluate(self, plan: QueryPlan) -> ClarificationDecision:
        meaningful = tuple(term for term in plan.lexical_terms if len(term) >= 3)
        if len(plan.original_text.strip()) < 4 or (not meaningful and not plan.entities):
            return ClarificationDecision(
                True,
                "请补充具体的故障现象、对象或问题关键词，我再帮你检索资料。",
                ("object_or_symptom",),
                "问题中没有足够的检索对象或现象",
            )
        # 阶段只作为待确认字段；是否追问由检索证据是否互相冲突决定。
        return ClarificationDecision(False)

    def evaluate_competing_stages(self, plan: QueryPlan, candidates) -> ClarificationDecision:
        if "stage" not in plan.missing_fields:
            return ClarificationDecision(False)
        stages = {
            stage
            for item in candidates[:8]
            for stage, phrases in {
                "calibration": ("定标", "校准", "校正"),
                "quality_control": ("质控", "质控品"),
                "sample_test": ("样本检测", "样本", "标本"),
                "startup": ("开机", "自检"),
            }.items()
            if any(phrase in item.chunk.text for phrase in phrases)
        }
        if len(stages) < 2:
            return ClarificationDecision(False)
        return ClarificationDecision(
            True,
            "为了避免把不同场景的步骤混在一起，请确认异常发生在定标、质控、样本检测还是开机自检阶段？",
            ("stage",),
            "高排名证据覆盖多个可能互斥的处理阶段",
        )
