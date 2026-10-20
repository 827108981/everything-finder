"""确定性提取检索词和明确业务条件，不依赖外部模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.rag.models import ConstraintKind, QueryConstraint


_SEPARATOR = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff._+/#-]+")
_IDENTIFIER = re.compile(
    r"(?i)(?=[a-z0-9._+/#-]*[a-z])(?=[a-z0-9._+/#-]*\d)[a-z0-9]+(?:[-._+/#][a-z0-9]+)*"
)

_QUESTION_PREFIXES = (
    "请帮我分析", "请帮我查找", "我想了解", "我想知道", "请问", "麻烦问下",
    "如何解决", "怎样解决", "怎么解决", "如何处理", "怎样处理", "怎么处理",
    "如何排查", "怎样排查", "怎么排查", "为什么", "如何", "怎样", "怎么",
)
_QUESTION_SUFFIXES = (
    "这个时候该怎么排查", "这个时候怎么排查", "这个时候该如何排查",
    "这个时候该怎么处理", "这个时候怎么处理", "这个时候怎么办",
    "应该怎么办", "该怎么办", "怎么办", "怎么排查", "如何排查", "怎样排查",
    "怎么处理", "如何处理", "怎样处理", "是什么原因", "的原因是什么",
    "的问题", "问题", "是什么", "有哪些", "吗", "呢",
)

# 只扩展语义稳定、实际会出现在维修资料中的同义词。
_STABLE_VARIANTS = {
    "不通过": ("未通过", "失败"),
    "未通过": ("不通过", "失败"),
    "报错": ("错误", "异常"),
    "错误": ("报错", "异常"),
    "偏低": ("低", "下降"),
    "偏高": ("高", "升高"),
}

_DOMAIN_TERMS = (
    "免疫底物", "底物本底", "发光值异常", "发光值偏低", "发光值偏高",
    "定标失败", "定标异常", "质控异常", "发光值", "异常", "失败",
    "检查点", "数据库", "搜索索引", "全文索引", "AI索引", "向量索引",
    "重排模型", "向量模型", "搜索任务", "索引任务", "项目", "底物", "本底",
    "定标", "校准", "质控", "试剂", "样本", "标本", "发光", "异常", "失败",
    "不通过", "未通过", "报错", "错误", "恢复", "中断", "索引", "任务",
    "文件", "正文", "数据库", "版本", "模型", "向量", "搜索", "启动",
    "停止", "取消", "线程", "锁", "事务", "快照", "发布", "generation",
)

_SCOPE_PATTERNS = {
    "single": ("只有某个项目", "某个项目", "单个项目", "只有一个项目", "仅一个项目", "单一项目"),
    "multiple": ("多个项目", "部分项目", "几个项目", "一部分项目"),
    "all": ("全部项目", "所有项目", "所有检测项目", "每个项目", "大面积"),
}
_STAGE_PATTERNS = {
    "calibration": ("定标", "校准", "校正"),
    "quality_control": ("质控", "质控品", "质量控制"),
    "sample_test": ("样本检测", "标本检测", "样本", "标本"),
    "startup": ("开机", "启动", "自检"),
}
_DIRECTION_PATTERNS = {
    "low": ("偏低", "过低", "低于", "下降", "不足"),
    "high": ("偏高", "过高", "高于", "升高"),
    "abnormal": ("异常", "故障", "报错", "错误", "不通过", "失败"),
}
_STOP_CONTENT = ("这个时候", "这个问题", "的情况", "的原因", "该如何", "应该如何", "请帮我")


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    original_text: str
    normalized_text: str
    terms: tuple[str, ...]
    fts_expression: str
    fallback_terms: tuple[str, ...]
    constraints: tuple[QueryConstraint, ...] = ()
    entities: tuple[str, ...] = ()
    intent: str = "general"


class QueryAnalyzer:
    """用词典和规则抽取稳定词项，保留原问题供语义检索与精排使用。"""

    def __init__(self, *, chinese_window: int = 3, max_terms: int = 20) -> None:
        # 保留旧参数名兼容调用方；不再按固定窗口切任意长中文。
        self.max_terms = max(1, int(max_terms))

    def analyze(self, text: str) -> QueryAnalysis:
        original = str(text or "").strip()
        normalized = normalize_text(original, ignore_spaces=False)
        core = self._strip_question_shell(normalized)
        identifiers = _IDENTIFIER.findall(normalized)
        terms: list[str] = []
        fallback: list[str] = []
        self._extend_unique(terms, identifiers)
        self._extend_unique(fallback, identifiers)

        matched_terms = self._matched_domain_terms(core)
        self._extend_unique(terms, matched_terms)
        self._extend_unique(fallback, matched_terms)

        for source, replacements in _STABLE_VARIANTS.items():
            if source in core:
                self._extend_unique(terms, (source, *replacements))
                self._extend_unique(fallback, (source, *replacements))

        # 对没有词典命中的自由文本，才使用少量三字片段兜底；过滤问句壳词。
        if len(matched_terms) < 2:
            residual = core
            for phrase in _STOP_CONTENT:
                residual = residual.replace(phrase, " ")
            residual = re.sub(r"(这个时候|时候|这个|怎么|如何|怎样|应该|该|请问|请|吗|呢)", " ", residual)
            segments = [part for part in _SEPARATOR.split(residual) if part]
            for segment in segments:
                if len(segment) >= 3:
                    for start in range(max(1, len(segment) - 2)):
                        self._extend_unique(terms, (segment[start : start + 3],))

        terms = [term for term in terms if len(term) >= 3][: self.max_terms]
        fallback = [term for term in fallback if term][: self.max_terms]
        expression = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        constraints = self._extract_constraints(original, identifiers, matched_terms)
        intent = self._infer_intent(original)
        return QueryAnalysis(
            original_text=original,
            normalized_text=normalized,
            terms=tuple(terms),
            fts_expression=expression,
            fallback_terms=tuple(fallback or terms or (core,)),
            constraints=constraints,
            entities=tuple(dict.fromkeys((*identifiers, *matched_terms))),
            intent=intent,
        )

    @staticmethod
    def _strip_question_shell(value: str) -> str:
        result = value.strip(" ？?，,。.!！")
        changed = True
        while changed and result:
            changed = False
            for prefix in _QUESTION_PREFIXES:
                if result.startswith(prefix):
                    result = result[len(prefix) :].strip()
                    changed = True
                    break
        changed = True
        while changed and result:
            changed = False
            for suffix in _QUESTION_SUFFIXES:
                if result.endswith(suffix):
                    result = result[: -len(suffix)].strip()
                    changed = True
                    break
        return result.strip(" 的这时后")

    @staticmethod
    def _matched_domain_terms(text: str) -> list[str]:
        # 先匹配较长业务词，防止“免疫底物”被拆成泛化的“底物”。
        return [
            term
            for term in sorted(_DOMAIN_TERMS, key=len, reverse=True)
            if term.casefold() in text.casefold()
        ]

    @classmethod
    def _extract_constraints(
        cls,
        text: str,
        identifiers: list[str],
        matched_terms: list[str],
    ) -> tuple[QueryConstraint, ...]:
        normalized = normalize_text(text, ignore_spaces=False)
        constraints: list[QueryConstraint] = []
        for kind, patterns in (
            (ConstraintKind.SCOPE, _SCOPE_PATTERNS),
            (ConstraintKind.STAGE, _STAGE_PATTERNS),
            (ConstraintKind.DIRECTION, _DIRECTION_PATTERNS),
        ):
            for value, phrases in patterns.items():
                matched = next((phrase for phrase in phrases if phrase in normalized), None)
                if matched:
                    constraints.append(QueryConstraint(kind, value, True, matched))
                    break
        if identifiers:
            constraints.append(
                QueryConstraint(ConstraintKind.MODEL, identifiers[0], True, identifiers[0])
            )
        for term in matched_terms:
            object_term = next(
                (
                    value for value in ("发光值", "底物", "本底", "试剂", "样本", "标本", "项目")
                    if value in term
                ),
                None,
            )
            if object_term and not any(
                item.kind is ConstraintKind.OBJECT and item.value == object_term
                for item in constraints
            ):
                constraints.append(QueryConstraint(ConstraintKind.OBJECT, object_term, True, term))
        return tuple(constraints)

    @staticmethod
    def _infer_intent(text: str) -> str:
        if any(word in text for word in ("怎么排查", "如何排查", "怎样排查", "怎么处理", "如何解决", "怎么办")):
            return "troubleshooting"
        if any(word in text for word in ("是什么", "什么意思", "为什么")):
            return "explanation"
        return "general"

    @staticmethod
    def _extend_unique(target: list[str], values) -> None:
        for value in values:
            item = str(value or "").strip()
            if item and item not in target:
                target.append(item)
