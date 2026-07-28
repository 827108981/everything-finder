from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class SearchHit:
    block_id: int | None
    location_text: str
    context: str
    hit_count: int
    source_type: str
    ocr_confidence: float | None = None
    is_fuzzy: bool = False


@dataclass(slots=True)
class SearchResult:
    file_id: int
    block_id: int | None
    file_path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: float
    location_text: str
    context: str
    hit_count: int
    source_type: str
    parse_status: str
    score: float
    ocr_confidence: float | None = None
    has_fuzzy_match: bool = False
    matches: list[SearchHit] = field(default_factory=list)
