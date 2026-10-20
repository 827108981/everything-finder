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
    vector_rank: int | None = None
    vector_score: float | None = None
    fused_score: float = 0.0
    rerank_score: float | None = None
    constraint_score: float | None = None
    constraint_decision: CandidateDecision | None = None
    duplicate_of: str | None = None


class ConstraintKind(StrEnum):
    """用户问题中会改变答案适用范围的业务条件。"""

    SCOPE = "scope"
    STAGE = "stage"
    OBJECT = "object"
    DIRECTION = "direction"
    MODEL = "model"


class ConstraintStatus(StrEnum):
    MATCH = "match"
    CONFLICT = "conflict"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryConstraint:
    kind: ConstraintKind
    value: str
    required: bool = True
    source_text: str = ""
    confidence: float = 1.0


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
    analyzer_version: str = "query-plan-v1"

    # 兼容 RagIndexStore.lexical_search 当前使用的分析结果字段。
    @property
    def terms(self) -> tuple[str, ...]:
        return self.lexical_terms

    @property
    def fts_expression(self) -> str:
        return self.lexical_expression


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
