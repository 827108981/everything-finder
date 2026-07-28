from __future__ import annotations

from collections.abc import Iterable

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.models.content_block import ContentBlock


class BlockCoalescer:
    """Merge small adjacent logical fragments before normalization and FTS."""

    _MERGEABLE = {"docx_paragraph", "docx_table_row", "xlsx_row", "text"}

    def __init__(self, target_chars: int = 4096, max_chars: int = 16384) -> None:
        self.target_chars = max(256, int(target_chars))
        self.max_chars = max(self.target_chars, int(max_chars))

    def coalesce(self, blocks: Iterable[ContentBlock]) -> list[ContentBlock]:
        output: list[ContentBlock] = []
        group: list[ContentBlock] = []
        group_chars = 0

        def flush() -> None:
            nonlocal group, group_chars
            if group:
                output.append(self._merge(group))
                group = []
                group_chars = 0

        for block in blocks:
            raw = block.raw_text.strip()
            if not raw:
                continue
            if block.block_type not in self._MERGEABLE or block.source_type != "native_text":
                flush()
                output.append(self._normalized_copy(block))
                continue
            if group and not self._compatible(group[-1], block):
                flush()
            projected = group_chars + len(raw) + (1 if group else 0)
            row_limit = block.block_type == "xlsx_row" and len(group) >= 20
            if group and (projected > self.max_chars or row_limit):
                flush()
            group.append(block)
            group_chars += len(raw) + (1 if len(group) > 1 else 0)
            if group_chars >= self.target_chars:
                flush()
        flush()
        for index, block in enumerate(output):
            block.block_index = index
        return output

    @staticmethod
    def _compatible(left: ContentBlock, right: ContentBlock) -> bool:
        return (
            left.file_path == right.file_path
            and left.block_type == right.block_type
            and left.sheet_name == right.sheet_name
            and left.source_type == right.source_type
        )

    def _merge(self, blocks: list[ContentBlock]) -> ContentBlock:
        first = blocks[0]
        last = blocks[-1]
        raw_text = "\n".join(block.raw_text.strip() for block in blocks if block.raw_text.strip())
        location = first.location_text
        if len(blocks) > 1 and last.location_text != first.location_text:
            location = f"{first.location_text} - {last.location_text}"
        extra = dict(first.extra or {})
        if len(blocks) > 1:
            extra["coalesced_blocks"] = len(blocks)
            extra["location_start"] = first.location_text
            extra["location_end"] = last.location_text
        return ContentBlock(
            file_path=first.file_path,
            block_index=first.block_index,
            block_type=first.block_type,
            location_text=location,
            raw_text=raw_text,
            normalized_text=normalize_text(raw_text),
            page_number=first.page_number,
            slide_number=first.slide_number,
            sheet_name=first.sheet_name,
            cell_start=first.cell_start,
            cell_end=last.cell_end or first.cell_end,
            line_start=first.line_start,
            line_end=last.line_end or first.line_end,
            source_type=first.source_type,
            ocr_confidence=first.ocr_confidence,
            extra=extra,
        )

    @staticmethod
    def _normalized_copy(block: ContentBlock) -> ContentBlock:
        if not block.normalized_text:
            block.normalized_text = normalize_text(block.raw_text)
        return block
