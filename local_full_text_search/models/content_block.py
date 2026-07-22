from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ContentBlock:
    file_path: str
    block_index: int
    block_type: str
    location_text: str
    raw_text: str
    normalized_text: str
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    source_type: str = "native_text"
    ocr_confidence: float | None = None
    extra: dict[str, object] = field(default_factory=dict)
