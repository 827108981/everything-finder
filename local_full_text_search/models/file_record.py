from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FileRecord:
    id: int
    root_id: int
    path: str
    filename: str
    extension: str
    size_bytes: int
    modified_time: float
    created_time: float | None
    quick_fingerprint: str
    parse_status: str
    is_deleted: bool = False
