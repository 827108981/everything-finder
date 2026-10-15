from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from collections.abc import Callable

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.rag.block_reader import DocumentBlocks
from local_full_text_search.rag.models import (
    FileSource,
    InvalidSourceError,
    RagChunk,
    SourceBlock,
    SourceRef,
    SourceSpan,
)

_BREAK_RE = re.compile(r"\n+|(?<=[。！？.!?；;])\s+")
_DOCX_TABLE_RE = re.compile(r"表格\s*(\d+)")


@dataclass(frozen=True, slots=True)
class ChunkPolicy:
    version: str = "evidence-v1"
    target_chars: int = 1000
    max_chars: int = 1400
    overlap_chars: int = 100
    embedding_model_revision: str = ""
    max_embedding_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("chunk policy version is required")
        if self.target_chars <= 0 or self.max_chars < self.target_chars:
            raise ValueError("chunk limits must satisfy 0 < target_chars <= max_chars")
        if self.overlap_chars < 0 or self.overlap_chars >= self.max_chars:
            raise ValueError("overlap_chars must be between 0 and max_chars")
        if self.max_embedding_tokens is not None and self.max_embedding_tokens <= 0:
            raise ValueError("max_embedding_tokens must be positive")


@dataclass(frozen=True, slots=True)
class _SourcePiece:
    block: SourceBlock
    joined_start: int
    joined_end: int
    separator_before: str


class ChunkBuilder:
    def __init__(
        self,
        policy: ChunkPolicy = ChunkPolicy(),
        *,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.policy = policy
        self.token_counter = token_counter
        if policy.max_embedding_tokens is not None and token_counter is None:
            raise ValueError("token_counter is required when max_embedding_tokens is set")

    def build(self, document: DocumentBlocks) -> list[RagChunk]:
        if not document.files:
            raise InvalidSourceError("cannot build a chunk without a file source")
        ordered = sorted(document.blocks, key=lambda block: (block.block_index, block.block_id))
        groups = self._groups(ordered, document.files[0].extension.lower())
        document_fingerprint = self._document_fingerprint(document, ordered)
        chunks: list[RagChunk] = []
        for group_index, group in enumerate(groups):
            text, pieces = self._joined_text(group)
            ranges = self._chunk_ranges(text)
            if self.policy.max_embedding_tokens is not None:
                ranges = self._enforce_token_limit(text, ranges)
            for ordinal, (raw_start, raw_end) in enumerate(ranges):
                start, end = self._trim_range(text, raw_start, raw_end)
                if start >= end:
                    continue
                chunk_text = text[start:end]
                if not chunk_text.strip():
                    continue
                spans = self._spans_for_range(start, end, pieces)
                if not spans:
                    raise InvalidSourceError("chunk range has no corresponding source text")
                text_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()
                signature = self._boundary_signature(group, spans)
                stable_id = _digest_parts(
                    [
                        self.policy.version,
                        self.policy.embedding_model_revision,
                        document_fingerprint,
                        str(group_index),
                        signature,
                        str(ordinal),
                        text_hash,
                    ]
                )
                sources = tuple(
                    self._source_ref(
                        file_source=file_source,
                        document=document,
                        group=group,
                        spans=spans,
                        stable_id=stable_id,
                    )
                    for file_source in document.files
                )
                normalized = normalize_text(chunk_text)
                if not normalized:
                    continue
                chunks.append(
                    RagChunk(
                        stable_chunk_id=stable_id,
                        text=chunk_text,
                        normalized_text=normalized,
                        text_hash=text_hash,
                        chunk_policy_version=self.policy.version,
                        sources=sources,
                    )
                )
        return chunks

    def _groups(self, blocks: list[SourceBlock], extension: str) -> list[list[SourceBlock]]:
        groups: list[list[SourceBlock]] = []
        current: list[SourceBlock] = []
        for block in blocks:
            if not block.raw_text.strip():
                continue
            if not current or self._can_join(current[-1], block, extension):
                current.append(block)
            else:
                groups.append(current)
                current = [block]
        if current:
            groups.append(current)
        return groups

    @staticmethod
    def _can_join(previous: SourceBlock, current: SourceBlock, extension: str) -> bool:
        if previous.source_type != current.source_type:
            return False
        if extension == ".pdf":
            # PDF chunks never cross page boundaries.
            return False
        if extension == ".docx":
            if previous.block_kind != current.block_kind:
                return False
            if previous.block_kind == "docx_table_row":
                previous_table = _DOCX_TABLE_RE.search(previous.location_text)
                current_table = _DOCX_TABLE_RE.search(current.location_text)
                return bool(
                    previous_table
                    and current_table
                    and previous_table.group(1) == current_table.group(1)
                )
            return True
        if extension == ".txt":
            return (
                previous.block_kind == current.block_kind == "text"
                and previous.line_end is not None
                and current.line_start == previous.line_end + 1
            )
        return False

    @staticmethod
    def _joined_text(blocks: list[SourceBlock]) -> tuple[str, list[_SourcePiece]]:
        parts: list[str] = []
        pieces: list[_SourcePiece] = []
        cursor = 0
        for block in blocks:
            value = block.raw_text
            if not value:
                continue
            separator = "\n" if parts else ""
            if separator:
                parts.append(separator)
                cursor += len(separator)
            start = cursor
            parts.append(value)
            cursor += len(value)
            pieces.append(_SourcePiece(block, start, cursor, separator))
        return "".join(parts), pieces

    def _chunk_ranges(self, text: str) -> list[tuple[int, int]]:
        if not text:
            return []
        ranges: list[tuple[int, int]] = []
        start = 0
        length = len(text)
        while start < length:
            hard_end = min(length, start + self.policy.max_chars)
            target_end = min(length, start + self.policy.target_chars)
            if hard_end == length:
                end = length
            else:
                end = self._preferred_cut(text, start, target_end, hard_end)
            if end <= start:
                end = min(length, start + self.policy.max_chars)
            ranges.append((start, end))
            if end >= length:
                break
            next_start = max(start + 1, end - self.policy.overlap_chars)
            if next_start < end:
                newline = text.find("\n", next_start, end)
                if newline >= 0 and newline + 1 < end:
                    next_start = newline + 1
            start = next_start
        return ranges

    @staticmethod
    def _trim_range(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end

    def _preferred_cut(self, text: str, start: int, target_end: int, hard_end: int) -> int:
        low = min(target_end, start + max(1, self.policy.target_chars // 2))
        window = text[low:hard_end]
        candidates = [low + match.end() for match in _BREAK_RE.finditer(window)]
        candidates = [value for value in candidates if value > start and value <= hard_end]
        if not candidates:
            return hard_end
        return min(candidates, key=lambda value: (abs(value - target_end), value))

    def _enforce_token_limit(
        self,
        text: str,
        ranges: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        assert self.policy.max_embedding_tokens is not None
        assert self.token_counter is not None
        result: list[tuple[int, int]] = []
        for range_start, range_end in ranges:
            start = range_start
            while start < range_end:
                end = range_end
                if self.token_counter(text[start:end]) > self.policy.max_embedding_tokens:
                    low = start + 1
                    high = end
                    while low < high:
                        middle = (low + high + 1) // 2
                        if self.token_counter(text[start:middle]) <= self.policy.max_embedding_tokens:
                            low = middle
                        else:
                            high = middle - 1
                    fitting_end = low
                    while (
                        fitting_end > start
                        and self.token_counter(text[start:fitting_end])
                        > self.policy.max_embedding_tokens
                    ):
                        fitting_end -= 1
                    if fitting_end <= start:
                        raise InvalidSourceError("one source character exceeds embedding token limit")
                    preferred = self._preferred_cut(
                        text,
                        start,
                        fitting_end,
                        fitting_end,
                    )
                    end = preferred if preferred > start else fitting_end
                result.append((start, end))
                if end >= range_end:
                    break
                start = max(start + 1, end - self.policy.overlap_chars)
                if start < end:
                    newline = text.find("\n", start, end)
                    if newline >= 0 and newline + 1 < end:
                        start = newline + 1
        return result

    @staticmethod
    def _spans_for_range(
        start: int,
        end: int,
        pieces: list[_SourcePiece],
    ) -> tuple[SourceSpan, ...]:
        spans: list[SourceSpan] = []
        previous_block_id: int | None = None
        for piece in pieces:
            overlap_start = max(start, piece.joined_start)
            overlap_end = min(end, piece.joined_end)
            if overlap_start >= overlap_end:
                continue
            raw_start = overlap_start - piece.joined_start
            raw_end = overlap_end - piece.joined_start
            separator = (
                piece.separator_before
                if start < piece.joined_start < end
                and previous_block_id != piece.block.block_id
                else ""
            )
            spans.append(
                SourceSpan(
                    block_id=piece.block.block_id,
                    block_index=piece.block.block_index,
                    raw_start=raw_start,
                    raw_end=raw_end,
                    separator_before=separator,
                )
            )
            previous_block_id = piece.block.block_id
        return tuple(spans)

    def _source_ref(
        self,
        *,
        file_source: FileSource,
        document: DocumentBlocks,
        group: list[SourceBlock],
        spans: tuple[SourceSpan, ...],
        stable_id: str,
    ) -> SourceRef:
        blocks_by_id = {block.block_id: block for block in group}
        first = blocks_by_id[spans[0].block_id]
        last = blocks_by_id[spans[-1].block_id]
        location = first.location_text
        if last.block_id != first.block_id and last.location_text != first.location_text:
            location = f"{first.location_text} - {last.location_text}".strip(" -")
        line_start = _line_for_offset(first, spans[0].raw_start) if first.line_start is not None else None
        line_end = _line_for_offset(last, max(spans[-1].raw_start, spans[-1].raw_end - 1)) if last.line_start is not None else None
        confidences = [
            blocks_by_id[span.block_id].ocr_confidence
            for span in spans
            if blocks_by_id[span.block_id].ocr_confidence is not None
        ]
        return SourceRef(
            generation=document.generation,
            stable_chunk_id=stable_id,
            file_id=file_source.file_id,
            root_id=file_source.root_id,
            document_id=document.document_id,
            file_path=file_source.file_path,
            filename=file_source.filename,
            extension=file_source.extension,
            spans=spans,
            block_index_start=first.block_index,
            block_index_end=last.block_index,
            location_text=location,
            page_number=first.page_number,
            line_start=line_start,
            line_end=line_end,
            source_type=first.source_type,
            ocr_confidence=min(confidences) if confidences else None,
        )

    def _document_fingerprint(
        self,
        document: DocumentBlocks,
        blocks: list[SourceBlock],
    ) -> str:
        file_source = document.files[0]
        parts = [file_source.parser_name, file_source.parser_version]
        for block in blocks:
            parts.append(
                _digest_parts(
                    [
                        block.block_kind,
                        block.location_text,
                        str(block.page_number),
                        str(block.line_start),
                        str(block.line_end),
                        block.source_type,
                        block.raw_text,
                    ]
                )
            )
        return _digest_parts(parts)

    @staticmethod
    def _boundary_signature(group: list[SourceBlock], spans: tuple[SourceSpan, ...]) -> str:
        by_id = {block.block_id: block for block in group}
        pieces: list[str] = []
        for span in spans:
            block = by_id[span.block_id]
            pieces.extend(
                [
                    block.block_kind,
                    block.location_text,
                    str(block.page_number),
                    str(block.line_start),
                    str(block.line_end),
                    block.source_type,
                    str(block.block_index),
                    str(span.raw_start),
                    str(span.raw_end),
                ]
            )
        return _digest_parts(pieces)


def _line_for_offset(block: SourceBlock, offset: int) -> int:
    assert block.line_start is not None
    bounded = max(0, min(len(block.raw_text), int(offset)))
    return int(block.line_start) + block.raw_text[:bounded].count("\n")


def _digest_parts(parts: list[str]) -> str:
    digest = hashlib.sha256()
    for value in parts:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()
