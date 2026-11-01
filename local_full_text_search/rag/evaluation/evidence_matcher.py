from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import PurePath
from typing import Any

from local_full_text_search.rag.content_identity import (
    canonical_content, canonicalize_with_offsets, compare_content,
)
from local_full_text_search.rag.evaluation.gold_loader import GoldCase


def normalize_evidence_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in folded if character.isalnum())


def _matches_declared_page(row: dict[str, Any], page: str) -> bool:
    """页号须作为完整数字匹配，不能把金标第 9 页映射到第 19 页。"""
    for key in ("page_number", "slide_number"):
        if str(row.get(key) or "").strip() == page:
            return True
    if not page.isdecimal():
        return page == str(row.get("location") or "").strip()
    return re.search(rf"(?<!\d){re.escape(page)}(?!\d)", str(row.get("location") or "")) is not None


@dataclass(frozen=True, slots=True)
class EvidencePart:
    chunk_id: str
    quote_start: int
    quote_end: int
    source_start: int
    source_end: int
    quote_original_start: int | None = None
    quote_original_end: int | None = None
    source_original_start: int | None = None
    source_original_end: int | None = None


@dataclass(frozen=True, slots=True)
class EvidenceAlternative:
    alternative_id: str
    match_type: str
    parts: tuple[EvidencePart, ...]
    confirmed: bool


@dataclass(frozen=True, slots=True)
class GoldEvidenceMatch:
    evidence_id: str
    kind: str
    file: str
    page: str
    quote: str
    chunk_ids: tuple[str, ...]
    confidence: str
    reason: str = ""
    declared_chunk_ids: tuple[str, ...] = ()
    equivalent_chunk_ids: tuple[str, ...] = ()
    suspected_chunk_ids: tuple[str, ...] = ()
    conflict_chunk_ids: tuple[str, ...] = ()
    alternatives: tuple[EvidenceAlternative, ...] = ()


def _declared_alternatives(
    rows: list[dict[str, Any]], quote: str, evidence_id: str,
) -> tuple[EvidenceAlternative, ...]:
    """只将同一文档内完整覆盖引文的连续块确认为计分证据。"""
    normalized_quote = canonicalize_with_offsets(quote)
    needle = normalized_quote.text
    if not needle:
        return ()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("file_path") or row.get("filename") or "").casefold(), []).append(row)
    alternatives: list[EvidenceAlternative] = []
    for document_rows in grouped.values():
        document_rows.sort(key=lambda row: (int(row.get("ordinal") or 0), str(row.get("chunk_id") or "")))
        for start in range(len(document_rows)):
            window = document_rows[start:start + 3]
            for width in range(1, len(window) + 1):
                selected = window[:width]
                if any(
                    int(right.get("ordinal") or 0) != int(left.get("ordinal") or 0) + 1
                    for left, right in zip(selected, selected[1:])
                ):
                    break
                source_normalizations = [
                    canonicalize_with_offsets(str(row.get("text") or ""))
                    for row in selected
                ]
                normalized = [item.text for item in source_normalizations]
                segments: list[tuple[str, int]] = []
                for position, value in enumerate(normalized):
                    overlap = 0
                    if position:
                        previous = normalized[position - 1]
                        for size in range(min(len(previous), len(value), 512), 7, -1):
                            if previous.endswith(value[:size]):
                                overlap = size
                                break
                    # 只消除可验证的连续重复区间；短巧合不作为重叠依据。
                    segments.append((value[overlap:], overlap))
                joined = "".join(value for value, _ in segments)
                hit_at = joined.find(needle)
                if hit_at < 0:
                    continue
                hit_end = hit_at + len(needle)
                offset = 0
                parts: list[EvidencePart] = []
                for row, source_normalized, (text, source_shift) in zip(
                    selected, source_normalizations, segments
                ):
                    left = max(hit_at, offset)
                    right = min(hit_end, offset + len(text))
                    if left < right:
                        source_start = left - offset + source_shift
                        source_end = right - offset + source_shift
                        quote_start = left - hit_at
                        quote_end = right - hit_at
                        parts.append(EvidencePart(
                            str(row.get("chunk_id") or ""), quote_start, quote_end,
                            source_start, source_end,
                            normalized_quote.offsets[quote_start][0],
                            normalized_quote.offsets[quote_end - 1][1],
                            source_normalized.offsets[source_start][0],
                            source_normalized.offsets[source_end - 1][1],
                        ))
                    offset += len(text)
                if parts and all(part.chunk_id for part in parts):
                    alternatives.append(EvidenceAlternative(
                        f"{evidence_id}:declared:{len(alternatives) + 1}",
                        "declared", tuple(parts), True,
                    ))
                break
    # 重叠窗口只保留最短的完整覆盖组合。
    unique: dict[tuple[str, ...], EvidenceAlternative] = {}
    for alternative in alternatives:
        unique.setdefault(tuple(part.chunk_id for part in alternative.parts), alternative)
    return tuple(unique.values())


def map_case_evidence(case: GoldCase, chunk_rows: list[dict[str, Any]]) -> list[GoldEvidenceMatch]:
    matches: list[GoldEvidenceMatch] = []
    for kind, entries in (("positive", case.gold_evidence), ("hard_negative", case.hard_negatives)):
        for index, entry in enumerate(entries, 1):
            file_value = str(entry.get("file") or "").strip()
            page_value = str(entry.get("page") or "").strip()
            quote = str(entry.get("quote") or "").strip()
            basename = PurePath(file_value.replace("\\", "/")).name.casefold()
            rows = [row for row in chunk_rows if str(row.get("filename") or "").casefold() == basename]
            if page_value:
                page_rows = [
                    row for row in rows
                    if _matches_declared_page(row, page_value)
                ]
                rows = page_rows
            evidence_id = f"{case.case_id}:{kind}:{index}"
            if not rows:
                matches.append(GoldEvidenceMatch(
                    evidence_id, kind, file_value, page_value, quote, (), "unmatched",
                    "当前 AI 索引中未找到相同文件/页码的正文块",
                ))
                continue
            alternatives = list(_declared_alternatives(rows, quote, evidence_id))
            declared_ids = tuple(dict.fromkeys(
                part.chunk_id for alternative in alternatives for part in alternative.parts
            ))
            equivalent_ids: list[str] = []
            suspected_ids: list[str] = []
            # 跨格式自动等价需要整块正文在保真规范化后相同；仅引文
            # 相同但其余 1% 有差异时只提示疑似，不擅自合并事实。
            stem = PurePath(basename).stem
            declared_rows = [row for row in rows if str(row.get("chunk_id") or "") in declared_ids]
            for row in chunk_rows:
                candidate_name = str(row.get("filename") or "").casefold()
                if candidate_name == basename or PurePath(candidate_name).stem != stem:
                    continue
                if page_value and not _matches_declared_page(row, page_value):
                    continue
                candidate_id = str(row.get("chunk_id") or "")
                if not candidate_id or canonical_content(quote) not in canonical_content(str(row.get("text") or "")):
                    continue
                equivalent = any(
                    compare_content(str(source.get("text") or ""), str(row.get("text") or "")).status
                    in {"exact", "layout_equivalent"}
                    for source in declared_rows
                )
                if equivalent:
                    equivalent_ids.append(candidate_id)
                    alternative = _declared_alternatives([row], quote, evidence_id)
                    for item in alternative:
                        alternatives.append(EvidenceAlternative(
                            f"{evidence_id}:equivalent:{len(alternatives) + 1}",
                            "layout_equivalent", item.parts, True,
                        ))
                else:
                    suspected_ids.append(candidate_id)
            normalized_quote = normalize_evidence_text(quote)
            file_groups: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                file_key = str(row.get("file_path") or row.get("filename") or "").casefold()
                file_groups.setdefault(file_key, []).append(row)
            exact_groups: list[tuple[str, ...]] = []
            approximate_groups: list[tuple[str, ...]] = []
            for file_rows in file_groups.values():
                file_rows.sort(key=lambda row: (int(row.get("ordinal") or 0), str(row.get("chunk_id") or "")))
                joined_parts: list[str] = []
                spans: list[tuple[int, int, str]] = []
                offset = 0
                for row in file_rows:
                    text = normalize_evidence_text(str(row.get("text") or ""))
                    start = offset
                    end = start + len(text)
                    spans.append((start, end, str(row.get("chunk_id") or "")))
                    joined_parts.append(text)
                    offset = end
                joined = "".join(joined_parts)
                start_at = joined.find(normalized_quote) if normalized_quote else -1
                if start_at >= 0:
                    end_at = start_at + len(normalized_quote)
                    exact_groups.append(tuple(
                        chunk_id for start, end, chunk_id in spans
                        if chunk_id and start < end_at and end > start_at
                    ))
                    continue
                # OCR 或 PDF 文本抽取可能造成少数字符差异。只把相似匹配列为待复核。
                if normalized_quote and len(normalized_quote) >= 24:
                    ratio = SequenceMatcher(None, normalized_quote, joined, autojunk=False).ratio()
                    if ratio >= 0.72:
                        approximate_groups.append(tuple(
                            str(row.get("chunk_id") or "") for row in file_rows
                            if row.get("chunk_id")
                        ))
            if exact_groups:
                chunk_ids = tuple(dict.fromkeys(chunk_id for group in exact_groups for chunk_id in group))
                confidence = "exact" if len(exact_groups) == 1 else "review"
            elif approximate_groups:
                chunk_ids = tuple(dict.fromkeys(chunk_id for group in approximate_groups for chunk_id in group))
                confidence = "review"
            else:
                chunk_ids = ()
                confidence = "unmatched"
            confirmed_ids = tuple(dict.fromkeys((*declared_ids, *equivalent_ids)))
            matches.append(GoldEvidenceMatch(
                evidence_id=evidence_id,
                kind=kind,
                file=file_value,
                page=page_value,
                quote=quote,
                chunk_ids=confirmed_ids,
                confidence=confidence if confirmed_ids else "review" if suspected_ids or approximate_groups else "unmatched",
                reason=(
                    "同名文件中有多份资料命中，需要人工确认路径" if len(exact_groups) > 1
                    else "引文近似匹配，需要人工确认" if approximate_groups and not exact_groups
                    else "引文未能在索引正文块中定位" if not confirmed_ids
                    else ""
                ),
                declared_chunk_ids=declared_ids,
                equivalent_chunk_ids=tuple(dict.fromkeys(equivalent_ids)),
                suspected_chunk_ids=tuple(dict.fromkeys(suspected_ids)),
                alternatives=tuple(alternatives),
            ))
    return matches


def load_chunk_rows(connection) -> list[dict[str, Any]]:
    rows = connection.execute(
        "SELECT stable_chunk_id, ordinal, text, source_json FROM rag_chunks"
    ).fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        source = json.loads(row["source_json"])
        result.append({
            "chunk_id": str(row["stable_chunk_id"]),
            "ordinal": int(row["ordinal"]),
            "text": str(row["text"]),
            "filename": str(source.get("filename") or ""),
            "file_path": str(source.get("file_path") or ""),
            "page_number": source.get("page_number"),
            "slide_number": source.get("slide_number"),
            "location": str(source.get("location_text") or ""),
        })
    return result
