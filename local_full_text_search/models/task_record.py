from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TaskRecord:
    id: int
    file_id: int
    task_type: str
    status: str
    priority: int
    retry_count: int
    error_message: str | None = None
