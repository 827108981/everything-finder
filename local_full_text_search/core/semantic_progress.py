from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SemanticProgress:
    """A parser progress snapshot whose monotonic fields represent real work."""

    phase: str = ""
    completed: int = 0
    total: int = 0
    cursor: str = ""
    bytes_read: int = 0
    output_blocks: int = 0
    checkpoint_version: int = 0

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> "SemanticProgress":
        source = value or {}
        return cls(
            phase=str(source.get("phase") or ""),
            completed=_non_negative_int(source.get("completed")),
            total=_non_negative_int(source.get("total")),
            cursor=str(source.get("cursor") or ""),
            bytes_read=_non_negative_int(source.get("bytes_read")),
            output_blocks=_non_negative_int(source.get("output_blocks")),
            checkpoint_version=_non_negative_int(source.get("checkpoint_version")),
        )


def is_semantic_progress(
    previous: SemanticProgress | Mapping[str, object] | None,
    current: SemanticProgress | Mapping[str, object] | None,
) -> bool:
    """Return true only when a parser advanced a meaningful unit of work.

    Worker heartbeats and ever-increasing transport sequence numbers are
    intentionally absent. A parser may run for hours as long as a page, row,
    member, byte, region, output-block, checkpoint, or legitimate phase moves
    forward.
    """

    before = (
        previous
        if isinstance(previous, SemanticProgress)
        else SemanticProgress.from_mapping(previous)
    )
    after = (
        current
        if isinstance(current, SemanticProgress)
        else SemanticProgress.from_mapping(current)
    )
    if not any(
        (
            before.phase,
            before.completed,
            before.cursor,
            before.bytes_read,
            before.output_blocks,
            before.checkpoint_version,
        )
    ):
        return bool(
            after.phase
            or after.completed
            or after.cursor
            or after.bytes_read
            or after.output_blocks
            or after.checkpoint_version
        )
    if after.phase and after.phase != before.phase:
        return True
    if after.completed > before.completed:
        return True
    if after.bytes_read > before.bytes_read:
        return True
    if after.output_blocks > before.output_blocks:
        return True
    if after.checkpoint_version > before.checkpoint_version:
        return True
    return _cursor_advanced(before.cursor, after.cursor)


def progress_signature(value: SemanticProgress | Mapping[str, object] | None) -> str:
    snapshot = (
        value
        if isinstance(value, SemanticProgress)
        else SemanticProgress.from_mapping(value)
    )
    return (
        f"{snapshot.phase}|{snapshot.cursor}|{snapshot.completed}|"
        f"{snapshot.bytes_read}|{snapshot.output_blocks}|"
        f"{snapshot.checkpoint_version}"
    )


def _cursor_advanced(previous: str, current: str) -> bool:
    if not current or current == previous:
        return False
    if not previous:
        return True
    before_parts = _cursor_parts(previous)
    after_parts = _cursor_parts(current)
    if before_parts is None or after_parts is None:
        return False
    before_labels, before_numbers = before_parts
    after_labels, after_numbers = after_parts
    return before_labels == after_labels and after_numbers > before_numbers


def _cursor_parts(value: str) -> tuple[tuple[str, ...], tuple[int, ...]] | None:
    labels = tuple(re.findall(r"[^\d]+", value))
    numbers = tuple(int(item) for item in re.findall(r"\d+", value))
    if not numbers:
        return None
    return labels, numbers


def _non_negative_int(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
