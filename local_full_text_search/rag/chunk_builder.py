from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.rag.models import RagChunk, SourceBlock, SourceRef


CHUNK_POLICY_VERSION = "paragraph-window-v1"
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;.!?])\s*|\n+")


class ChunkBuilder:
    def __init__(
        self,
        *,
        target_chars: int = 900,
        max_chars: int = 1200,
        overlap_chars: int = 120,
    ) -> None:
        self.target_chars = max(100, int(target_chars))
        self.max_chars = max(self.target_chars, int(max_chars))
        self.overlap_chars = min(max(0, int(overlap_chars)), self.target_chars // 2)

    def iter_chunks(self, blocks: Iterable[SourceBlock]) -> Iterator[RagChunk]:
        current_key: tuple[object, ...] | None = None
        current: list[SourceBlock] = []
        for block in blocks:
            key = self._boundary_key(block)
            if current and key != current_key:
                yield from self._chunks_for_group(current)
                current = []
            current_key = key
            current.append(block)
        if current:
            yield from self._chunks_for_group(current)

    @staticmethod
    def _boundary_key(block: SourceBlock) -> tuple[object, ...]:
        return (
            block.document_key,
            block.file_id,
            block.page_number,
            block.slide_number,
            block.sheet_name,
            block.source_type,
        )

    def _chunks_for_group(self, blocks: list[SourceBlock]) -> Iterator[RagChunk]:
        text_parts: list[str] = []
        spans: list[tuple[int, int, SourceBlock]] = []
        cursor = 0
        for block in blocks:
            text = block.text.strip()
            if not text:
                continue
            if text_parts:
                text_parts.append("\n")
                cursor += 1
            start = cursor
            text_parts.append(text)
            cursor += len(text)
            spans.append((start, cursor, block))
        full_text = "".join(text_parts)
        if not full_text:
            return

        ordinal = 0
        start = 0
        while start < len(full_text):
            end = self._choose_end(full_text, start)
            if end <= start:
                break
            text = full_text[start:end].strip()
            if text:
                used_blocks = tuple(
                    block
                    for span_start, span_end, block in spans
                    if span_end > start and span_start < end
                )
                if used_blocks:
                    yield self._make_chunk(text, ordinal, used_blocks, start, end)
                    ordinal += 1
            if end >= len(full_text):
                break
            next_start = max(start + 1, end - self.overlap_chars)
            # Prefer an overlap boundary that starts at a sentence/line edge.
            overlap_segment = full_text[next_start:end]
            boundary = max(overlap_segment.rfind("\n"), overlap_segment.rfind("。"), overlap_segment.rfind("！"), overlap_segment.rfind("？"))
            if boundary >= max(0, len(overlap_segment) // 3):
                next_start += boundary + 1
            start = next_start

    def _choose_end(self, text: str, start: int) -> int:
        hard_end = min(len(text), start + self.max_chars)
        if hard_end == len(text):
            return hard_end
        target_end = min(hard_end, start + self.target_chars)
        candidate = max(
            text.rfind("\n", start, target_end),
            text.rfind("。", start, target_end),
            text.rfind("！", start, target_end),
            text.rfind("？", start, target_end),
            text.rfind("；", start, target_end),
        )
        if candidate > start + self.target_chars // 2:
            return candidate + 1
        return hard_end

    @staticmethod
    def _make_chunk(
        text: str,
        ordinal: int,
        blocks: tuple[SourceBlock, ...],
        start: int,
        end: int,
    ) -> RagChunk:
        first = blocks[0]
        normalized = normalize_text(text, ignore_punctuation=False)
        text_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        boundary = {
            "blocks": [blocks[0].block_id, blocks[-1].block_id],
            "page": first.page_number,
            "slide": first.slide_number,
            "sheet": first.sheet_name,
            "start": start,
            "end": end,
            "policy": CHUNK_POLICY_VERSION,
        }
        stable_id = hashlib.sha256(
            (first.document_key + json.dumps(boundary, sort_keys=True) + text_hash).encode("utf-8")
        ).hexdigest()
        line_starts = [block.line_start for block in blocks if block.line_start is not None]
        line_ends = [block.line_end for block in blocks if block.line_end is not None]
        location = first.location_text
        if first.page_number is not None:
            location = location or f"第 {first.page_number} 页"
        elif first.slide_number is not None:
            location = location or f"第 {first.slide_number} 张幻灯片"
        elif first.sheet_name:
            location = location or f"工作表：{first.sheet_name}"
        elif line_starts:
            location = f"第 {min(line_starts)}-{max(line_ends or line_starts)} 行"
        source = SourceRef(
            source_generation=first.source_generation,
            stable_chunk_id=stable_id,
            document_id=first.document_id,
            file_id=first.file_id,
            root_id=first.root_id,
            file_path=first.file_path,
            filename=first.filename,
            extension=first.extension,
            block_ids=tuple(dict.fromkeys(block.block_id for block in blocks)),
            location_text=location,
            page_number=first.page_number,
            slide_number=first.slide_number,
            sheet_name=first.sheet_name,
            cell_start=first.cell_start,
            cell_end=blocks[-1].cell_end or first.cell_end,
            line_start=min(line_starts) if line_starts else None,
            line_end=max(line_ends) if line_ends else None,
            source_type=first.source_type,
            ocr_confidence=first.ocr_confidence,
        )
        return RagChunk(stable_id, first.document_key, ordinal, text, normalized, text_hash, source)
