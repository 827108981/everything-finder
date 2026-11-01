"""确定性提取检索词和明确业务条件，不依赖外部模型。"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.rag.models import (
    ConstraintKind, ConstraintPolarity, ConstraintStrength, QueryConstraint,
    QuerySpan, QueryTransformation, RetrievalQueryVariant,
)


_SEPARATOR = re.compile(r"[^0-9a-z\u3400-\u4dbf\u4e00-\u9fff._+/#-]+")
_IDENTIFIER = re.compile(
    r"(?i)(?=[a-z0-9._+/#-]*[a-z])(?=[a-z0-9._+/#-]*\d)[a-z0-9]+(?:[-._+/#][a-z0-9]+)*"
)
_DEVICE_PHRASE = re.compile(r"([\u3400-\u9fff]{2,10}(?:诊断仪|分析仪|检测仪|治疗仪))")

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
    "sample_test": ("样本检测", "标本检测"),
    "startup": ("开机", "启动", "自检"),
}
_DIRECTION_PATTERNS = {
    "low": ("偏低", "过低", "低于", "下降", "不足"),
    "high": ("偏高", "过高", "高于", "升高"),
    "abnormal": ("异常", "故障", "报错", "错误", "不通过", "失败"),
}
_STOP_CONTENT = ("这个时候", "这个问题", "的情况", "的原因", "该如何", "应该如何", "请帮我")
_QUESTION_WORDS = re.compile(
    r"(?:什么时候|是什么原因|有什么|为什么|怎么样|怎么办|如何|怎样|怎么|何时|多少|哪些|哪个|是否|请问|吗|呢)"
)
_GRAMMAR_BOUNDARIES = re.compile(r"[的了呢吗会是]")


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
    spans: tuple[QuerySpan, ...] = ()
    transformations: tuple[QueryTransformation, ...] = ()
    lexical_variants: tuple[RetrievalQueryVariant, ...] = ()
    semantic_variants: tuple[RetrievalQueryVariant, ...] = ()


class QueryAnalyzer:
    """用词典和规则抽取稳定词项，保留原问题供语义检索与精排使用。"""

    def __init__(self, *, chinese_window: int = 3, max_terms: int = 20) -> None:
        # 保留旧参数名兼容调用方；不再按固定窗口切任意长中文。
        self.max_terms = max(1, int(max_terms))

    def analyze(self, text: str) -> QueryAnalysis:
        original = str(text or "")
        normalized = normalize_text(original, ignore_spaces=False)
        core = self._strip_question_shell(normalized)
        transformations = (
            (QueryTransformation("remove_question_shell", normalized, core, "去除可定位的问句外壳"),)
            if core != normalized else ()
        )
        spans = self._extract_spans(original)
        identifiers = _IDENTIFIER.findall(normalized)
        constraints = self._extract_constraints(original, identifiers, self._matched_domain_terms(core))
        positive_core = core
        for condition in constraints:
            if condition.polarity is ConstraintPolarity.AFFIRMED and condition.strength is not ConstraintStrength.AMBIGUOUS:
                continue
            phrases = _DIRECTION_PATTERNS.get(condition.value, ()) if condition.kind is ConstraintKind.DIRECTION else (condition.source_text,)
            for phrase in phrases:
                positive_core = positive_core.replace(phrase, " ")
        terms: list[str] = []
        fallback: list[str] = []
        self._extend_unique(terms, identifiers)
        self._extend_unique(fallback, identifiers)

        matched_terms = self._matched_domain_terms(positive_core)
        # 每道题都从原问题提取内容线索；固定词表仅做同义扩展和条件解析，
        # 不能因为识别出型号或某个领域词就跳过其他业务对象。
        residual = positive_core
        for identifier in identifiers:
            residual = re.sub(re.escape(identifier), " ", residual, flags=re.IGNORECASE)
        for phrase in _STOP_CONTENT:
            residual = residual.replace(phrase, " ")
        # 用空格保留边界，窗口只在各片段内部生成。
        residual = _QUESTION_WORDS.sub(" ", residual)
        residual = _GRAMMAR_BOUNDARIES.sub(" ", residual)
        segments = [part for part in _SEPARATOR.split(residual) if part]
        content_terms: list[str] = []
        for segment in segments:
            if len(segment) < 3:
                continue
            if len(segment) <= 8:
                self._extend_unique(content_terms, (segment,))
            # 四字窗口保留相邻概念；三字窗口兼容原有 trigram 索引的断行。
            for width in (4, 3):
                if len(segment) < width:
                    continue
                for start in range(len(segment) - width + 1):
                    self._extend_unique(content_terms, (segment[start : start + width],))
        self._extend_unique(terms, content_terms)
        self._extend_unique(fallback, content_terms)
        self._extend_unique(terms, matched_terms)
        self._extend_unique(fallback, matched_terms)

        for source, replacements in _STABLE_VARIANTS.items():
            if source in positive_core:
                self._extend_unique(terms, (source, *replacements))
                self._extend_unique(fallback, (source, *replacements))

        for condition in constraints:
            if (condition.kind is ConstraintKind.DIRECTION
                    and condition.polarity is ConstraintPolarity.AFFIRMED
                    and condition.strength is not ConstraintStrength.AMBIGUOUS
                    and condition.value in {"low", "high"}):
                direction = "偏低" if condition.value == "low" else "偏高"
                variants = _STABLE_VARIANTS[direction]
                self._extend_unique(terms, (("发光值" + direction,) if "发光值" in positive_core else ()) + (direction, *variants))
                self._extend_unique(fallback, (direction, *variants))

        direction_words = {"偏低", "偏高", "升高", "下降"}
        terms = [term for term in terms if len(term) >= 3 or term in direction_words][: self.max_terms]
        fallback = [term for term in fallback if term][: self.max_terms]
        expression = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        lexical_variants = self._lexical_variants(core, identifiers, expression)
        focused = self._focused_semantic_query(original, core)
        semantic_variants = (RetrievalQueryVariant("semantic_original", original),)
        if focused and focused != original:
            semantic_variants += (RetrievalQueryVariant("semantic_focused", focused),)
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
            spans=spans,
            transformations=transformations,
            lexical_variants=lexical_variants,
            semantic_variants=semantic_variants,
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
        # “P02 是什么泵”只删疑问结构，“泵”仍是可检索对象。
        result = re.sub(r"是什么(?=[^，,。；;！？?]{1,8}$)", " ", result)
        result = re.sub(r"有什么(?=[^，,。；;！？?]{1,8}$)", " ", result)
        return result.strip(" 的？?，,。.!！")

    @staticmethod
    def _extract_spans(original: str) -> tuple[QuerySpan, ...]:
        spans: list[QuerySpan] = []
        for match in _IDENTIFIER.finditer(original):
            spans.append(QuerySpan(match.group(), *match.span(), ("ascii_digit_identifier", "exact_surface")))
        for match in re.finditer(r"[\u3400-\u9fff]{2,}", original):
            spans.append(QuerySpan(match.group(), *match.span(), ("contiguous_chinese",)))
        return tuple(sorted(spans, key=lambda item: (item.start, item.end)))

    @staticmethod
    def _lexical_variants(core: str, identifiers: list[str], broad: str) -> tuple[RetrievalQueryVariant, ...]:
        variants: list[RetrievalQueryVariant] = []
        unique_ids = list(dict.fromkeys(identifiers))
        for identifier in unique_ids:
            if len(identifier) < 3:
                variants.append(RetrievalQueryVariant("lexical_short_identifier", identifier, "", (identifier,)))
        if len(unique_ids) >= 2:
            selected = unique_ids[:3]
            if all(len(item) >= 3 for item in selected):
                expr = " AND ".join(f'"{item}"' for item in selected)
                variants.append(RetrievalQueryVariant("lexical_precise", core, expr, tuple(selected)))
        phrases = [part for part in re.split(r"[^\u3400-\u9fff]+", core) if 3 <= len(part) <= 12]
        for phrase in phrases[:1]:
            expr = f'"{phrase}"'
            if expr != broad:
                variants.append(RetrievalQueryVariant("lexical_phrase", phrase, expr, (phrase,)))
        if broad:
            variants.append(RetrievalQueryVariant("lexical_broad", core, broad))
        return tuple(variants)

    @staticmethod
    def _focused_semantic_query(original: str, core: str) -> str:
        if any(word in original for word in ("不是", "并非", "不属于", "而是", "相比", "与", "还是", "“", "”")):
            return original
        return " ".join(core.split()) or original

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
        quoted = [match.span() for match in re.finditer(r'“[^”]*”|"[^"]*"|‘[^’]*’', normalized)]
        for kind, patterns in (
            (ConstraintKind.SCOPE, _SCOPE_PATTERNS),
            (ConstraintKind.STAGE, _STAGE_PATTERNS),
            (ConstraintKind.DIRECTION, _DIRECTION_PATTERNS),
        ):
            occurrences: list[tuple[int, int, str, str]] = []
            occupied: list[tuple[int, int]] = []
            # 长词先占位，避免“只有一个项目”又被“一个项目”重复命中。
            phrases = sorted(
                ((value, phrase) for value, variants in patterns.items() for phrase in variants),
                key=lambda pair: len(pair[1]), reverse=True,
            )
            for value, phrase in phrases:
                for match in re.finditer(re.escape(phrase), normalized):
                    start, end = match.span()
                    if any(start < right and end > left for left, right in occupied):
                        continue
                    occupied.append((start, end))
                    occurrences.append((start, end, value, phrase))
            for start, end, value, phrase in sorted(occurrences):
                prefix = normalized[max(0, start - 12):start]
                local_prefix = re.split(r"[，,。;；！？?]", prefix)[-1]
                if any(left < end and start < right for left, right in quoted):
                    polarity, strength, reason = (
                        ConstraintPolarity.UNKNOWN, ConstraintStrength.AMBIGUOUS,
                        "条件出现在引文中，不能直接当成用户当前状态",
                    )
                elif re.search(r"(?:并不是|不属于|不是|并非)\s*$", local_prefix):
                    polarity, strength, reason = (
                        ConstraintPolarity.NEGATED, ConstraintStrength.AMBIGUOUS,
                        "相邻否定词明确否定了该取值",
                    )
                elif re.search(r"(?:可能|好像|疑似|似乎)", local_prefix):
                    polarity, strength, reason = (
                        ConstraintPolarity.AFFIRMED, ConstraintStrength.SOFT,
                        "问题使用不确定表达，只能软排序",
                    )
                else:
                    polarity, strength, reason = (
                        ConstraintPolarity.AFFIRMED, ConstraintStrength.HARD,
                        "明确词项且相邻范围内没有否定或不确定表达",
                    )
                constraints.append(QueryConstraint(
                    kind, value, polarity is ConstraintPolarity.AFFIRMED and strength is ConstraintStrength.HARD,
                    phrase, None, polarity, strength, "rule", reason,
                ))
        # 编号形态无法证明它是整机型号；P02、SV99 等均只参加召回。
        # 没有型号时保留用户明确指名的设备；这属于适用性门禁，不参与
        # 固定领域词表的关键词提取。
        if not identifiers:
            for phrase in dict.fromkeys(_DEVICE_PHRASE.findall(normalized)):
                constraints.append(QueryConstraint(
                    ConstraintKind.DEVICE, phrase, source_text=phrase,
                    reason="问题明确指名设备，证据须能确认适用范围",
                ))
        # 问句开头的“X如何……”给出开放主题线索，无须把 X 预先收录为领域词。
        # 若 X 内已有明确阶段/方向等条件，则不要再把它整体硬化为主题。
        topic_match = re.match(r"^\s*([\u3400-\u9fff]{2,8})(?=如何|怎么|怎样)", normalized)
        if topic_match:
            topic = topic_match.group(1)
            if not any(item.is_hard and item.source_text and item.source_text in topic for item in constraints):
                constraints.append(QueryConstraint(
                    ConstraintKind.TOPIC, topic, source_text=topic,
                    reason="问题句首明确给出主题，需结合来源范围和正文事实核对",
                ))
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
                constraints.append(QueryConstraint(
                    ConstraintKind.OBJECT, object_term, source_text=term,
                    reason="问题中明确提及检索对象",
                ))
        for kind in (ConstraintKind.SCOPE, ConstraintKind.STAGE, ConstraintKind.DIRECTION, ConstraintKind.MODEL):
            affirmed = [i for i, item in enumerate(constraints)
                        if item.kind is kind and item.polarity is ConstraintPolarity.AFFIRMED]
            values = {constraints[i].value for i in affirmed}
            if kind is ConstraintKind.DIRECTION and "abnormal" in values and len(values) == 2:
                for i in affirmed:
                    if constraints[i].value == "abnormal":
                        constraints[i] = replace(
                            constraints[i], required=False, strength=ConstraintStrength.SOFT,
                            reason="异常是泛化描述，具体方向优先",
                        )
                values.remove("abnormal")
            if len(values) > 1:
                # 多对象或同类互斥值未能绑定关系时，宁可保留候选也不猜一个全局硬条件。
                for i in affirmed:
                    constraints[i] = replace(
                        constraints[i], required=False, strength=ConstraintStrength.AMBIGUOUS,
                        reason="同类条件出现多个取值，无法可靠绑定对象或消解冲突",
                    )
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
