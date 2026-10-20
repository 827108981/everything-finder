"""检查检索候选是否与问题明确给出的业务条件冲突。"""

from __future__ import annotations

from collections.abc import Sequence

from local_full_text_search.rag.models import (
    CandidateDecision,
    ConstraintJudgement,
    ConstraintKind,
    ConstraintStatus,
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
            judgements = tuple(
                self._judge(constraint, item.chunk.text)
                for constraint in plan.constraints
            )
            conflicts = sum(
                1 for result in judgements
                if result.constraint.required and result.status is ConstraintStatus.CONFLICT
            )
            matches = sum(
                1 for result in judgements
                if result.constraint.required and result.status is ConstraintStatus.MATCH
            )
            unknown = sum(
                1 for result in judgements
                if result.constraint.required and result.status is ConstraintStatus.UNKNOWN
            )
            total = max(1, matches + unknown + conflicts)
            item.constraint_score = (matches - conflicts * 2) / total
            item.constraint_decision = CandidateDecision(
                stable_chunk_id=item.chunk.stable_chunk_id,
                accepted=conflicts == 0,
                reason=("与问题条件冲突" if conflicts else "条件一致" if matches else "正文未明确说明条件"),
                judgements=judgements,
                required_match_count=matches,
                required_unknown_count=unknown,
                required_conflict_count=conflicts,
            )
        # 场景明确一致的证据优先；在同一等级内保留 Cross-Encoder 排名。
        return tuple(
            sorted(
                candidates,
                key=lambda item: (
                    bool(item.constraint_decision and item.constraint_decision.required_unknown_count),
                    -float(item.rerank_score or 0.0),
                    item.chunk.stable_chunk_id,
                ),
            )
        )

    def _judge(self, constraint: QueryConstraint, text: str) -> ConstraintJudgement:
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
