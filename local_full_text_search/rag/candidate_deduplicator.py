"""在 Cross-Encoder 前删除同文档中的重复正文块。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import Sequence

from local_full_text_search.rag.models import EvidenceChunk


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    removed_chunk_id: str
    kept_chunk_id: str
    method: str
    similarity: float


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    candidates: tuple[EvidenceChunk, ...]
    duplicates: tuple[DuplicateRecord, ...]


class CandidateDeduplicator:
    def __init__(self, *, jaccard_threshold: float = 0.92) -> None:
        self.jaccard_threshold = min(1.0, max(0.5, float(jaccard_threshold)))

    def deduplicate(
        self, candidates: Sequence[EvidenceChunk], *, limit: int
    ) -> DeduplicationResult:
        kept: list[EvidenceChunk] = []
        duplicates: list[DuplicateRecord] = []
        by_id: dict[str, EvidenceChunk] = {}
        by_hash: dict[tuple[str, str], EvidenceChunk] = {}
        by_document: dict[str, list[tuple[EvidenceChunk, set[str]]]] = {}

        for candidate in candidates:
            chunk = candidate.chunk
            current_id = chunk.stable_chunk_id
            duplicate_of = by_id.get(current_id)
            method = "stable_chunk_id"
            similarity = 1.0
            if duplicate_of is None:
                duplicate_of = by_hash.get((chunk.document_key, chunk.text_hash))
                method = "text_hash"
            if duplicate_of is None:
                grams = self._char_grams(chunk.normalized_text or chunk.text)
                for previous, previous_grams in by_document.get(chunk.document_key, []):
                    score = self._jaccard(grams, previous_grams)
                    if score >= self.jaccard_threshold:
                        duplicate_of = previous
                        method = "same_document_3gram_jaccard"
                        similarity = score
                        break
            if duplicate_of is not None:
                candidate.duplicate_of = duplicate_of.chunk.stable_chunk_id
                duplicates.append(
                    DuplicateRecord(
                        current_id,
                        duplicate_of.chunk.stable_chunk_id,
                        method,
                        similarity,
                    )
                )
                continue
            kept.append(candidate)
            by_id[current_id] = candidate
            by_hash[(chunk.document_key, chunk.text_hash)] = candidate
            by_document.setdefault(chunk.document_key, []).append(
                (candidate, self._char_grams(chunk.normalized_text or chunk.text))
            )
            if len(kept) >= max(1, int(limit)):
                break
        return DeduplicationResult(tuple(kept), tuple(duplicates))

    @staticmethod
    def _char_grams(value: str) -> set[str]:
        # 忽略空白和标点，仅比较正文字符；分页提取常会改变换行与标点。
        normalized = re.sub(r"[\W_]+", "", str(value or "")).casefold()
        if not normalized:
            return set()
        if len(normalized) <= 3:
            return {normalized}
        return {normalized[index : index + 3] for index in range(len(normalized) - 2)}

    @staticmethod
    def _jaccard(left: set[str], right: set[str]) -> float:
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)
