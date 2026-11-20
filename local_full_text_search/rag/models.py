from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class SourceRef:
    source_generation: int
    stable_chunk_id: str
    document_id: int | None
    file_id: int
    root_id: int
    file_path: str
    filename: str
    extension: str
    block_ids: tuple[int, ...]
    location_text: str
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_type: str = "native_text"
    ocr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class SourceBlock:
    source_generation: int
    block_id: int
    document_id: int | None
    file_id: int
    root_id: int
    file_path: str
    filename: str
    extension: str
    document_key: str
    block_index: int
    text: str
    location_text: str
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_type: str = "native_text"
    ocr_confidence: float | None = None


@dataclass(frozen=True, slots=True)
class RagChunk:
    stable_chunk_id: str
    document_key: str
    ordinal: int
    text: str
    normalized_text: str
    text_hash: str
    source: SourceRef


@dataclass(frozen=True, slots=True)
class RagQuery:
    text: str
    root_ids: tuple[int, ...] = ()
    extensions: tuple[str, ...] = ()
    include_ocr: bool = True
    min_ocr_confidence: float = 0.60
    lexical_top_k: int = 50
    vector_top_k: int = 50
    fusion_top_k: int = 30
    reranker_top_k: int = 20
    final_top_k: int = 8


@dataclass(slots=True)
class EvidenceChunk:
    chunk: RagChunk
    lexical_rank: int | None = None
    lexical_score: float | None = None
    lexical_score_kind: str = ""
    lexical_contributions: tuple[dict[str, object], ...] = ()
    vector_rank: int | None = None
    vector_score: float | None = None
    # 融合后的原始名次；用于重排时保留前置检索证据，避免单一模型把候选完全推走。
    fused_rank: int | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    rerank_score_combined: float | None = None
    answer_support_score: float | None = None
    answer_support_reasons: tuple[str, ...] = ()
    constraint_score: float | None = None
    constraint_decision: CandidateDecision | None = None
    duplicate_of: str | None = None
    # 仅在去重阶段填充；旧的单来源调用方仍可读取 chunk.source。
    member_chunk_ids: tuple[str, ...] = ()
    sources: tuple[SourceRef, ...] = ()
    lexical_route_hits: tuple[RetrievalRouteHit, ...] = ()
    vector_route_hits: tuple[RetrievalRouteHit, ...] = ()
    lexical_route_rrf_score: float | None = None
    vector_route_rrf_score: float | None = None
    vector_unit_hits: tuple[VectorUnitHit, ...] = ()
    query_plan_diagnostics: dict[str, object] = field(default_factory=dict)
    rerank_diagnostics: dict[str, object] = field(default_factory=dict)
    rerank_raw_logit: float | None = None

    @property
    def all_sources(self) -> tuple[SourceRef, ...]:
        return self.sources or (self.chunk.source,)

    @property
    def all_member_chunk_ids(self) -> tuple[str, ...]:
        return self.member_chunk_ids or (self.chunk.stable_chunk_id,)


class ConstraintKind(StrEnum):
    """用户问题中会改变答案适用范围的业务条件。"""

    SCOPE = "scope"
    STAGE = "stage"
    OBJECT = "object"
    DIRECTION = "direction"
    MODEL = "model"
    DEVICE = "device"
    TOPIC = "topic"


class ConstraintStatus(StrEnum):
    MATCH = "match"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


class ConstraintPolarity(StrEnum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    UNKNOWN = "unknown"


class ConstraintStrength(StrEnum):
    HARD = "hard"
    SOFT = "soft"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class QueryConstraint:
    kind: ConstraintKind
    value: str
    required: bool = True
    source_text: str = ""
    confidence: float | None = None
    polarity: ConstraintPolarity = ConstraintPolarity.AFFIRMED
    strength: ConstraintStrength = ConstraintStrength.HARD
    source: str = "rule"
    reason: str = ""
    subject: str = ""
    source_start: int | None = None
    source_end: int | None = None
    role: str = "observed"

    @property
    def is_hard(self) -> bool:
        # 旧调用方仍可传 required，但它本身不能授权删除候选。
        return (
            self.required
            and self.polarity is ConstraintPolarity.AFFIRMED
            and self.strength is ConstraintStrength.HARD
        )


@dataclass(frozen=True, slots=True)
class QuerySpan:
    text: str
    start: int
    end: int
    features: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class QueryTransformation:
    operation: str
    source_text: str
    result_text: str
    reason: str


@dataclass(frozen=True, slots=True)
class RetrievalQueryVariant:
    route_id: str
    query_text: str
    expression: str = ""
    signal_texts: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RetrievalRouteHit:
    route_id: str
    rank: int
    score: float | None = None
    score_kind: str = "unknown"


@dataclass(frozen=True, slots=True)
class VectorUnitHit:
    """向量检索命中的细粒度单元；最终证据仍归并到父 chunk。"""

    unit_id: str
    route_id: str
    raw_rank: int
    score: float
    text: str = ""
    kind: str = "window"
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    embedding_text: str = ""
    representation_metadata: dict | None = None


@dataclass(frozen=True, slots=True)
class QueryEntity:
    text: str
    type: str
    normalized_value: str
    confidence: float
    source: str
    strength: str
    reason: str
    is_strong_constraint: bool = False
    needs_scope_check: bool = False


@dataclass(frozen=True, slots=True)
class PlanSignal:
    name: str
    confidence: float
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class SubQuery:
    text: str
    focus: str
    inherited_constraints: tuple[QueryConstraint, ...]
    confidence: float
    source: str
    reason: str


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original_text: str
    normalized_text: str
    intent: str
    entities: tuple[str, ...]
    constraints: tuple[QueryConstraint, ...]
    lexical_terms: tuple[str, ...]
    lexical_expression: str
    semantic_query: str
    fallback_terms: tuple[str, ...] = ()
    missing_fields: tuple[str, ...] = ()
    analyzer_version: str = "query-plan-v2"
    spans: tuple[QuerySpan, ...] = ()
    transformations: tuple[QueryTransformation, ...] = ()
    lexical_variants: tuple[RetrievalQueryVariant, ...] = ()
    semantic_variants: tuple[RetrievalQueryVariant, ...] = ()
    # entities stays tuple[str] for existing callers; rich records are additive.
    entity_details: tuple[QueryEntity, ...] = ()
    answer_requirements: tuple[PlanSignal, ...] = ()
    sub_queries: tuple[SubQuery, ...] = ()
    retrieval_hints: tuple[PlanSignal, ...] = ()
    unknown_entities: tuple[QueryEntity, ...] = ()
    domain_config: str = ""
    plan_diagnostics: tuple[PlanSignal, ...] = ()
    adapter_active: bool = False
    strategies: tuple[PlanSignal, ...] = ()
    scope_constraints: tuple[QueryEntity, ...] = ()
    protected_evidence: tuple[PlanSignal, ...] = ()
    method_branches: tuple[PlanSignal, ...] = ()
    ambiguities: tuple[PlanSignal, ...] = ()
    # Validated configuration, not retrieved facts or Gold labels.
    adapter_policy: dict = field(default_factory=dict)

    # 兼容 RagIndexStore.lexical_search 当前使用的分析结果字段。
    @property
    def terms(self) -> tuple[str, ...]:
        return self.lexical_terms

    @property
    def fts_expression(self) -> str:
        return self.lexical_expression

    @property
    def parsed_request(self) -> dict:
        """Canonical view; legacy intent/entities/constraints remain compatible."""
        from local_full_text_search.rag.query_semantics import query_spec
        return query_spec(self)


@dataclass(frozen=True, slots=True)
class ConstraintJudgement:
    constraint: QueryConstraint
    status: ConstraintStatus
    reason: str
    matched_text: str = ""


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    stable_chunk_id: str
    accepted: bool
    reason: str
    judgements: tuple[ConstraintJudgement, ...] = ()
    required_match_count: int = 0
    required_unknown_count: int = 0
    required_conflict_count: int = 0
    soft_match_count: int = 0
    soft_conflict_count: int = 0
    support_status: str = "unknown"
    support_spans: tuple[str, ...] = ()


class RagAnswerStatus(StrEnum):
    ANSWERED = "answered"
    NEEDS_CLARIFICATION = "needs_clarification"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True, slots=True)
class RagAnswer:
    answer_text: str
    sources: tuple[SourceRef, ...]
    insufficient_evidence: bool
    source_ids: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    trace_id: str = ""
    provider_name: str = ""
    model_name: str = ""
    status: RagAnswerStatus = RagAnswerStatus.ANSWERED
    clarification_question: str = ""
    audit: AnswerAudit | None = None


@dataclass(frozen=True, slots=True)
class RagRetrievalResult:
    """只完成本地检索与证据准备时返回的结果，不代表生成了答案。"""

    trace_id: str
    status: RagAnswerStatus
    sources: tuple[SourceRef, ...] = ()
    insufficient_evidence: bool = False
    clarification_question: str = ""
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ClaimReview:
    claim: str
    citations: tuple[str, ...]
    status: str
    reason: str


@dataclass(frozen=True, slots=True)
class AnswerReviewResult:
    verdict: str
    summary: str
    claims: tuple[ClaimReview, ...]
    final_answer: str
    provider_name: str = ""
    model_name: str = ""


@dataclass(frozen=True, slots=True)
class AnswerAudit:
    draft_answer: str
    review: AnswerReviewResult | None
    citation_check_reason: str = ""


@dataclass(frozen=True, slots=True)
class VectorRecord:
    point_id: str
    embedding: tuple[float, ...]
    metadata: dict[str, str | int | float | bool]


@dataclass(frozen=True, slots=True)
class VectorHit:
    point_id: str
    score: float
    metadata: dict[str, object] = field(default_factory=dict)
