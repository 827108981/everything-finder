"""检查检索候选是否与问题明确给出的业务条件冲突。"""

from __future__ import annotations

import re
from collections.abc import Sequence

from local_full_text_search.rag.models import (
    CandidateDecision,
    ConstraintJudgement,
    ConstraintKind,
    ConstraintPolarity,
    ConstraintStatus,
    ConstraintStrength,
    EvidenceChunk,
    QueryConstraint,
    QueryPlan,
)


_VALUES = {
    ConstraintKind.SCOPE: {
        "single": ("只有某个项目", "某个项目", "单个项目", "只有一个项目", "仅一个项目", "单一项目"),
        "multiple": ("多个项目", "部分项目", "几个项目", "一部分项目"),
        "all": ("全部项目", "所有项目", "所有检测项目", "每个项目", "大面积"),
    },
    ConstraintKind.STAGE: {
        "calibration": ("定标", "校准", "校正"),
        "quality_control": ("质控", "质控品", "质量控制"),
        "sample_test": ("样本检测", "标本检测", "样本", "标本"),
        "startup": ("开机", "启动", "自检"),
    },
    ConstraintKind.DIRECTION: {
        "low": ("偏低", "过低", "低于", "下降", "不足"),
        "high": ("偏高", "过高", "高于", "升高"),
        "abnormal": ("异常", "故障", "报错", "错误", "不通过", "失败"),
    },
}


class ConstraintMatcher:
    """只对互斥条件做淘汰；未提及条件的正文标记 UNKNOWN。"""

    def evaluate(
        self, plan: QueryPlan, candidates: Sequence[EvidenceChunk]
    ) -> tuple[EvidenceChunk, ...]:
        for item in candidates:
            filenames = "\n".join(source.filename for source in item.all_sources)
            applicability_text = "\n".join((
                item.chunk.text,
                filenames,
            ))
            judgements = tuple(
                self._judge(constraint, applicability_text, item.chunk.text, filenames,
                            plan.lexical_terms)
                for constraint in plan.constraints
            )
            hard = tuple(result for result in judgements if result.constraint.is_hard)
            soft = tuple(result for result in judgements
                         if result.constraint.polarity is ConstraintPolarity.AFFIRMED
                         and result.constraint.strength is ConstraintStrength.SOFT)
            conflicts = sum(result.status is ConstraintStatus.CONFLICT for result in hard)
            matches = sum(result.status is ConstraintStatus.MATCH for result in hard)
            unknown = sum(result.status is ConstraintStatus.UNKNOWN for result in hard)
            soft_matches = sum(result.status is ConstraintStatus.MATCH for result in soft)
            soft_conflicts = sum(result.status is ConstraintStatus.CONFLICT for result in soft)
            # 软冲突仅做一个确定性的排序扣分；不改变 accepted。
            item.constraint_score = float(matches * 2 + soft_matches - soft_conflicts)
            conflict_detail = next((
                f"{result.constraint.kind.value}={result.constraint.value}：问题“{result.constraint.source_text}”，正文“{result.matched_text}”"
                for result in hard if result.status is ConstraintStatus.CONFLICT
            ), "")
            item.constraint_decision = CandidateDecision(
                stable_chunk_id=item.chunk.stable_chunk_id,
                accepted=conflicts == 0,
                reason=(
                    f"硬条件冲突：{conflict_detail}" if conflicts else
                    "适用性未知，保留为背景但不能单独支撑确定性结论" if unknown else
                    f"软条件冲突 {soft_conflicts} 项，已降权但保留" if soft_conflicts else
                    "条件一致" if matches or soft_matches else "正文未明确说明条件"
                ),
                judgements=judgements,
                required_match_count=matches,
                required_unknown_count=unknown,
                required_conflict_count=conflicts,
                soft_match_count=soft_matches,
                soft_conflict_count=soft_conflicts,
                support_status="conflict" if conflicts else "unknown" if unknown else "supported",
                support_spans=tuple(result.matched_text for result in hard
                                    if result.status is ConstraintStatus.MATCH and result.matched_text),
            )
        # 开放问题没有可验证的适用性排序依据，保留重排原序。
        if not plan.constraints:
            return tuple(candidates)
        if not any(item.is_hard for item in plan.constraints):
            # 只有软条件时仍按条件分数排序，保持原有的“可能匹配优先”行为。
            return tuple(
                sorted(
                    candidates,
                    key=lambda item: (
                        -float(item.constraint_score or 0.0),
                        -float(item.rerank_score_combined
                               if item.rerank_score_combined is not None
                               else item.rerank_score or 0.0),
                        item.chunk.stable_chunk_id,
                    ),
                )
            )
        # 场景明确一致的证据优先；条件判断相同时稳定保留综合重排顺序。
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    bool(item.constraint_decision and not item.constraint_decision.accepted),
                    bool(item.constraint_decision and item.constraint_decision.required_unknown_count),
                    -float(item.constraint_score or 0.0),
                ),
            )
        )

    def _judge(self, constraint: QueryConstraint, text: str,
               body: str = "", filenames: str = "",
               lexical_terms: tuple[str, ...] = ()) -> ConstraintJudgement:
        if constraint.kind is ConstraintKind.TOPIC:
            topic = constraint.value.casefold()
            title_match = topic in filenames.casefold()
            anchors = tuple(term for term in lexical_terms
                            if len(term) >= 3 and topic not in term.casefold()
                            and term.casefold() in body.casefold())
            body_match = any(
                topic in line.casefold() and any(anchor.casefold() in line.casefold() for anchor in anchors)
                for line in re.split(r"[。；;\n]", body)
            )
            supported = body_match or (title_match and bool(anchors))
            return ConstraintJudgement(
                constraint,
                ConstraintStatus.MATCH if supported else ConstraintStatus.UNKNOWN,
                "正文直接确认问题主题" if body_match else
                "来源标题确认范围，且同一正文包含问题中的检索对象" if supported else
                "来源及正文未共同确认问题主题",
                constraint.value if body_match else
                f"来源标题:{constraint.value}；正文:{anchors[0]}" if supported else "",
            )
        if constraint.kind is ConstraintKind.DEVICE:
            target = "".join(constraint.value.casefold().split())
            lines = re.split(r"[。；;\n]", body)
            supported_line = next((line for line in lines
                                   if target in "".join(line.casefold().split())
                                   and sum(line.count(suffix) for suffix in
                                           ("诊断仪", "分析仪", "检测仪", "治疗仪")) <= 1), "")
            title_only = target in "".join(filenames.casefold().split())
            return ConstraintJudgement(
                constraint,
                ConstraintStatus.MATCH if supported_line else ConstraintStatus.UNKNOWN,
                "正文局部确认设备" if supported_line else
                "仅来源标题提示设备，正文适用性未确认" if title_only else
                "正文未能局部确认该设备，或同段包含多个设备",
                constraint.value if supported_line else "",
            )
        if constraint.kind is ConstraintKind.OBJECT:
            found = constraint.value in text
            return ConstraintJudgement(
                constraint,
                ConstraintStatus.MATCH if found else ConstraintStatus.UNKNOWN,
                "正文提及该对象" if found else "正文未明确提及该对象",
                constraint.value if found else "",
            )
        values = _VALUES.get(constraint.kind)
        if values is None:
            return ConstraintJudgement(constraint, ConstraintStatus.UNKNOWN, "没有该类型的确定性判断规则")
        patterns = values.get(constraint.value, ())
        matched = self._find_non_negated(text, patterns)
        if (
            not matched
            and constraint.kind is ConstraintKind.DIRECTION
            and constraint.value == "abnormal"
        ):
            matched = self._find_non_negated(
                text,
                (*values.get("low", ()), *values.get("high", ())),
            )
        if matched:
            return ConstraintJudgement(constraint, ConstraintStatus.MATCH, "正文明确描述了相同条件", matched)
        for value, alternatives in values.items():
            if value == constraint.value:
                continue
            if (
                constraint.kind is ConstraintKind.DIRECTION
                and "abnormal" in {constraint.value, value}
            ):
                continue
            conflict = self._find_non_negated(text, alternatives)
            if conflict:
                return ConstraintJudgement(
                    constraint,
                    ConstraintStatus.CONFLICT,
                    f"正文描述“{conflict}”，与问题条件“{constraint.source_text or constraint.value}”不一致",
                    conflict,
                )
        return ConstraintJudgement(constraint, ConstraintStatus.UNKNOWN, "正文没有明确描述该条件")

    @staticmethod
    def _find_non_negated(text: str, phrases: tuple[str, ...]) -> str:
        # 简单否定窗口：防止“并非所有项目”被判为 all。
        for phrase in phrases:
            start = 0
            while True:
                index = text.find(phrase, start)
                if index < 0:
                    break
                prefix = text[max(0, index - 5) : index]
                if not any(word in prefix for word in ("不是", "并非", "并不", "不属于", "非所有")):
                    return phrase
                start = index + len(phrase)
        return ""
