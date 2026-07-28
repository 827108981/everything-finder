from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(slots=True)
class SearchQuery:
    text: str
    mode: str = "exact"
    root_ids: list[int] = field(default_factory=list)
    extensions: list[str] = field(default_factory=list)
    search_filename: bool = True
    search_path: bool = True
    search_content: bool = True
    include_ocr: bool = True
    include_ocr_fuzzy: bool = False
    ocr_min_confidence: float = 0.60
    case_sensitive: bool = False
    ignore_spaces: bool = False
    ignore_hyphens: bool = False
    date_from: datetime | None = None
    date_to: datetime | None = None
    min_size: int | None = None
    max_size: int | None = None
    page_size: int = 100
    max_results: int = 1000
    page: int = 1
    cursor: str | None = None
