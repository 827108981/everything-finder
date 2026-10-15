from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


class NoPublishedGenerationError(RuntimeError):
    """Raised when no complete, published search generation is available."""


class UnsupportedSnapshotError(RuntimeError):
    """Raised when an active search database cannot provide readable blocks."""


class RagIndexNotReadyError(RuntimeError):
    """Raised when no published sidecar matches the active generation/model."""


class RagBuildCancelled(RuntimeError):
    """Raised when a RAG index candidate is cancelled before publication."""


class RagSourceError(ValueError):
    """Raised when persisted chunk text cannot be traced to its source blocks."""


class InvalidSourceError(ValueError):
    """Raised when a chunk cannot be mapped back to source text."""


@dataclass(frozen=True, slots=True)
class FileSource:
    file_id: int
    root_id: int
    file_path: str
    filename: str
    extension: str
    parser_name: str
    parser_version: str

    def __post_init__(self) -> None:
        if self.file_id <= 0 or self.root_id <= 0:
            raise ValueError("file_id and root_id must be positive")
        if not self.file_path.strip():
            raise ValueError("file_path must not be empty")


@dataclass(frozen=True, slots=True)
class SourceBlock:
    generation: int
    document_id: int
    block_id: int
    block_index: int
    block_kind: str
    raw_text: str
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

    def __post_init__(self) -> None:
        if self.generation <= 0:
            raise ValueError("generation must be positive")
        if self.block_id <= 0:
            raise ValueError("block_id must be positive")
        if not self.raw_text.strip():
            raise ValueError("source block text must not be empty")
        if self.ocr_confidence is not None and not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError("ocr_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class SourceSpan:
    block_id: int
    block_index: int
    raw_start: int
    raw_end: int
    separator_before: str = ""

    def __post_init__(self) -> None:
        if self.block_id <= 0:
            raise ValueError("block_id must be positive")
        if self.raw_start < 0 or self.raw_end <= self.raw_start:
            raise ValueError("source span must be a non-empty half-open range")


@dataclass(frozen=True, slots=True)
class SourceRef:
    generation: int
    stable_chunk_id: str
    file_id: int
    root_id: int
    document_id: int
    file_path: str
    filename: str
    extension: str
    spans: tuple[SourceSpan, ...]
    block_index_start: int
    block_index_end: int
    location_text: str
    page_number: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_type: str = "native_text"
    ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        if self.generation <= 0 or self.file_id <= 0 or self.root_id <= 0 or self.document_id <= 0:
            raise ValueError("generation, file_id, root_id, and document_id must be positive")
        if not self.stable_chunk_id or not self.file_path:
            raise ValueError("source chunk ID and file path are required")
        if not self.spans:
            raise ValueError("at least one source span is required")
        if self.block_index_start > self.block_index_end:
            raise ValueError("block index range is reversed")
        if (self.line_start is None) != (self.line_end is None):
            raise ValueError("line_start and line_end must either both be set or both be empty")
        if self.line_start is not None and self.line_start > self.line_end:
            raise ValueError("line range is reversed")
        if self.ocr_confidence is not None and not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError("ocr_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class RagChunk:
    stable_chunk_id: str
    text: str
    normalized_text: str
    text_hash: str
    chunk_policy_version: str
    sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        if not self.stable_chunk_id or not self.chunk_policy_version:
            raise ValueError("stable_chunk_id and chunk_policy_version are required")
        if not self.text.strip() or not self.normalized_text.strip():
            raise ValueError("chunk text and normalized text must not be empty")
        if not self.text_hash:
            raise ValueError("text_hash is required")
        if not self.sources:
            raise ValueError("a chunk must have at least one source")
        if any(source.stable_chunk_id != self.stable_chunk_id for source in self.sources):
            raise ValueError("all sources must refer to this stable_chunk_id")


@dataclass(frozen=True, slots=True)
class EvidenceChunk:
    chunk: RagChunk
    selected_source: SourceRef

    def __post_init__(self) -> None:
        if self.selected_source not in self.chunk.sources:
            raise ValueError("selected_source must belong to chunk.sources")


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_id: str
    revision: str
    dimension: int
    max_tokens: int

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("model_id and revision are required")
        if self.dimension <= 0 or self.max_tokens <= 0:
            raise ValueError("model dimension and max_tokens must be positive")


@dataclass(frozen=True, slots=True)
class RagQuery:
    question: str
    root_ids: tuple[int, ...] = ()
    extensions: tuple[str, ...] = (".pdf", ".docx", ".txt")
    include_ocr: bool = True
    min_ocr_confidence: float = 0.60
    top_k: int = 8

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if not 1 <= self.top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not 0.0 <= self.min_ocr_confidence <= 1.0:
            raise ValueError("min_ocr_confidence must be between 0 and 1")
        if any(root_id <= 0 for root_id in self.root_ids):
            raise ValueError("root IDs must be positive")


@dataclass(frozen=True, slots=True)
class RankedEvidence:
    evidence: EvidenceChunk
    lexical_rank: int | None
    vector_rank: int | None
    fused_score: float


@dataclass(frozen=True, slots=True)
class Citation:
    label: str
    source: SourceRef
    text_hash: str
    excerpt: str


@dataclass(frozen=True, slots=True)
class RagAnswer:
    status: Literal[
        "answered",
        "insufficient",
        "index_stale",
        "model_unavailable",
        "validation_failed",
        "cancelled",
    ]
    answer_text: str
    citations: tuple[Citation, ...]
    trace_id: str
    source_generation: int | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RagProgress:
    stage: str
    message: str
    completed: int = 0
    total: int = 0
    indeterminate: bool = False
