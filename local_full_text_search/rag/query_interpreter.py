"""把 UI 的原始问题整理为各检索阶段共用的 QueryPlan。"""

from __future__ import annotations

from local_full_text_search.rag.models import (
    ConstraintPolarity, ConstraintStrength, QueryConstraint, QueryPlan,
)
from local_full_text_search.rag.query_analyzer import QueryAnalyzer


class QueryInterpreter:
    """第一版完全确定性；保留注入点供未来可选 LLM 解析器使用。"""

    def __init__(self, analyzer: QueryAnalyzer | None = None, llm_parser=None) -> None:
        self.analyzer = analyzer or QueryAnalyzer()
        self.llm_parser = llm_parser

    def interpret(self, text: str, token=None) -> QueryPlan:
        if token is not None:
            token.throw_if_cancelled()
        analysis = self.analyzer.analyze(text)
        # 可选的结构化 LLM 分析暂不自动启用；规则分析始终能独立运行。
        if self.llm_parser is not None:
            parsed = self.llm_parser.parse(text, token)
            analysis = self._merge_optional_result(analysis, parsed)
        return QueryPlan(
            original_text=analysis.original_text,
            normalized_text=analysis.normalized_text,
            intent=analysis.intent,
            entities=analysis.entities,
            constraints=analysis.constraints,
            lexical_terms=analysis.terms,
            lexical_expression=analysis.fts_expression,
            semantic_query=analysis.original_text,
            fallback_terms=analysis.fallback_terms,
            missing_fields=self._missing_fields(analysis.intent, analysis.constraints),
            analyzer_version="query-plan-v3",
            spans=analysis.spans,
            transformations=analysis.transformations,
            lexical_variants=analysis.lexical_variants,
            semantic_variants=analysis.semantic_variants,
        )

    @staticmethod
    def _missing_fields(intent: str, constraints) -> tuple[str, ...]:
        # 未识别到固定“阶段”不等于问题真的缺少条件。
        return ()

    @staticmethod
    def _merge_optional_result(analysis, parsed):
        """兼容旧注入点；同类不同值一律先降级，不能生成冲突硬条件。"""
        if not isinstance(parsed, dict):
            return analysis
        from dataclasses import replace

        known = {(item.kind, item.value) for item in analysis.constraints}
        constraints = list(analysis.constraints)
        for item in parsed.get("constraints", ()):
            if not isinstance(item, QueryConstraint) or (item.kind, item.value) in known:
                continue
            constraints.append(replace(item, source="optional_parser"))
            known.add((item.kind, item.value))
        for kind in {item.kind for item in constraints}:
            affirmed = [i for i, item in enumerate(constraints)
                        if item.kind is kind and item.polarity is ConstraintPolarity.AFFIRMED]
            if len({constraints[i].value for i in affirmed}) <= 1:
                continue
            for i in affirmed:
                constraints[i] = replace(
                    constraints[i], required=False, strength=ConstraintStrength.AMBIGUOUS,
                    reason="不同解析来源给出同类冲突取值",
                )
        intent = analysis.intent
        if intent == "general" and parsed.get("intent") in {"troubleshooting", "explanation"}:
            intent = parsed["intent"]
        return replace(analysis, constraints=tuple(constraints), intent=intent)
