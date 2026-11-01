"""在 Cross-Encoder 前合并规范化后完全相同的正文，并记录每条输入的去向。"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from local_full_text_search.rag.content_identity import canonical_content, compare_content, content_identity
from local_full_text_search.rag.models import EvidenceChunk


@dataclass(frozen=True, slots=True)
class DuplicateRecord:
    removed_chunk_id: str
    kept_chunk_id: str
    method: str
    similarity: float


@dataclass(frozen=True, slots=True)
class CandidateDisposition:
    chunk_id: str
    status: str
    reason: str
    input_rank: int
    output_rank: int | None
    representative_chunk_id: str | None
    method: str | None
    similarity: float | None
    source_merged: bool


@dataclass(frozen=True, slots=True)
class DeduplicationResult:
    candidates: tuple[EvidenceChunk, ...]
    dispositions: tuple[CandidateDisposition, ...]
    duplicates: tuple[DuplicateRecord, ...]
    exact_groups: tuple[CandidateDisposition, ...]
    near_duplicates: tuple[CandidateDisposition, ...]
    cutoffs: tuple[CandidateDisposition, ...]


class CandidateDeduplicator:
    def __init__(self, *, jaccard_threshold: float = 0.92) -> None:
        self.jaccard_threshold = min(1.0, max(0.5, float(jaccard_threshold)))

    def deduplicate(
        self, candidates: Sequence[EvidenceChunk], *, limit: int
    ) -> DeduplicationResult:
        kept: list[EvidenceChunk] = []
        dispositions: list[CandidateDisposition] = []
        duplicates: list[DuplicateRecord] = []
        by_id: dict[str, EvidenceChunk] = {}
        cutoff_ids: set[str] = set()
        by_hash: dict[str, EvidenceChunk] = {}
        by_content: dict[str, EvidenceChunk] = {}
        by_document: dict[str, list[tuple[EvidenceChunk, set[str]]]] = {}
        maximum = max(1, int(limit))

        for input_rank, candidate in enumerate(candidates, 1):
            chunk = candidate.chunk
            current_id = chunk.stable_chunk_id
            if current_id in cutoff_ids:
                dispositions.append(CandidateDisposition(
                    current_id, "duplicate_stable_chunk_id",
                    "内容块 ID 已处理；此前因 fusion_top_k 达到上限而截断，未进入重排候选池",
                    input_rank, None, None, "stable_chunk_id", 1.0, False,
                ))
                continue
            representative = by_id.get(current_id)
            status = ""
            method: str | None = None
            similarity: float | None = None
            source_merged = False
            if representative is not None:
                status, method, similarity = "duplicate_stable_chunk_id", "stable_chunk_id", 1.0
                reason = f"内容块 ID 已处理，保留 {representative.chunk.stable_chunk_id}"
            else:
                representative = by_hash.get(chunk.text_hash)
                if representative is not None and compare_content(
                    representative.chunk.text, chunk.text
                ).status not in {"exact", "layout_equivalent"}:
                    # 历史 hash 或异常碰撞也不能成为跨事实合并的依据。
                    representative = None
                if representative is not None:
                    status, method, similarity = "grouped_exact_text_hash", "text_hash", 1.0
                    # 完整 SourceRef 判等，保留融合输入顺序；不改代表的任何分数。
                    existing = representative.all_sources
                    merged = tuple(source for source in candidate.all_sources if source not in existing)
                    if merged:
                        representative.sources = existing + merged
                        source_merged = True
                    representative.member_chunk_ids = tuple(dict.fromkeys(
                        (*representative.all_member_chunk_ids, *candidate.all_member_chunk_ids)
                    ))
                    reason = (
                        f"规范化后正文哈希完全相同，来源已合并到代表 {representative.chunk.stable_chunk_id}"
                        if source_merged else
                        f"规范化后正文哈希完全相同，来源已存在于代表 {representative.chunk.stable_chunk_id}"
                    )
                else:
                    canonical = canonical_content(chunk.text)
                    representative = by_content.get(content_identity(chunk.text)) if canonical else None
                    if representative is not None and representative.chunk.document_key == chunk.document_key:
                        # 同文档相邻块继续沿用原有近重复策略，避免把分页来源
                        # 当作跨文件副本合并。
                        representative = None
                    if representative is not None and compare_content(
                        representative.chunk.text, chunk.text
                    ).status not in {"exact", "layout_equivalent"}:
                        representative = None
                    if representative is not None:
                        status, method, similarity = "grouped_layout_equivalent_text", "canonical_content", 1.0
                        existing = representative.all_sources
                        merged = tuple(source for source in candidate.all_sources if source not in existing)
                        if merged:
                            representative.sources = existing + merged
                            source_merged = True
                        representative.member_chunk_ids = tuple(dict.fromkeys(
                            (*representative.all_member_chunk_ids, *candidate.all_member_chunk_ids)
                        ))
                        reason = (
                            "排版规范化后事实字符完全一致，"
                            f"来源合并到代表 {representative.chunk.stable_chunk_id}"
                        )
                    else:
                        grams = self._char_grams(chunk.normalized_text or chunk.text)
                        for previous, previous_grams in by_document.get(chunk.document_key, []):
                            score = self._jaccard(grams, previous_grams)
                            if score >= self.jaccard_threshold and compare_content(
                                previous.chunk.text, chunk.text
                            ).status in {"exact", "layout_equivalent"}:
                                representative = previous
                                status = "duplicate_same_document_near_text"
                                method = "same_document_3gram_jaccard"
                                similarity = score
                                reason = (
                                    f"同文档正文近似重复，保留 {previous.chunk.stable_chunk_id}，"
                                    f"相似度 {score:.2f}；来源未合并"
                                )
                                break
            if status:
                assert representative is not None
                candidate.duplicate_of = representative.chunk.stable_chunk_id
                by_id.setdefault(current_id, representative)
                output_rank = kept.index(representative) + 1
                duplicates.append(DuplicateRecord(
                    current_id, representative.chunk.stable_chunk_id, method or "", similarity or 0.0
                ))
                dispositions.append(CandidateDisposition(
                    current_id, status, reason, input_rank, output_rank,
                    representative.chunk.stable_chunk_id, method, similarity, source_merged,
                ))
                continue

            if len(kept) >= maximum:
                cutoff_ids.add(current_id)
                dispositions.append(CandidateDisposition(
                    current_id, "fusion_top_k_cutoff",
                    f"正文不重复，但去重后候选已达到 fusion_top_k={maximum}，因此未进入重排候选池",
                    input_rank, None, None, None, None, False,
                ))
                continue

            candidate.sources = candidate.all_sources
            candidate.member_chunk_ids = candidate.all_member_chunk_ids
            kept.append(candidate)
            by_id[current_id] = candidate
            by_hash[chunk.text_hash] = candidate
            if canonical_content(chunk.text):
                by_content[content_identity(chunk.text)] = candidate
            by_document.setdefault(chunk.document_key, []).append(
                (candidate, self._char_grams(chunk.normalized_text or chunk.text))
            )
            dispositions.append(CandidateDisposition(
                current_id, "kept", "保留为代表候选，进入去重后候选池",
                input_rank, len(kept), current_id, None, None, False,
            ))

        records = tuple(dispositions)
        return DeduplicationResult(
            tuple(kept), records, tuple(duplicates),
            tuple(item for item in records if item.status == "grouped_exact_text_hash"),
            tuple(item for item in records if item.status == "duplicate_same_document_near_text"),
            tuple(item for item in records if item.status == "fusion_top_k_cutoff"),
        )

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
