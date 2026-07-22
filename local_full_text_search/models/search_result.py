from __future__ import annotations

from dataclasses import dataclass


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
