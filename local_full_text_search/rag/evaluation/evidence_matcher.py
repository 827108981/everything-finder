from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import PurePath
from typing import Any

from local_full_text_search.rag.evaluation.gold_loader import GoldCase


def normalize_evidence_text(value: str) -> str:
    folded = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in folded if character.isalnum())


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
                    if str(row.get("page_number") or "").strip() == page_value
                    or page_value in str(row.get("location") or "")
                ]
                rows = page_rows
            evidence_id = f"{case.case_id}:{kind}:{index}"
            if not rows:
                matches.append(GoldEvidenceMatch(
                    evidence_id, kind, file_value, page_value, quote, (), "unmatched",
                    "当前 AI 索引中未找到相同文件/页码的正文块",
                ))
                continue
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
            matches.append(GoldEvidenceMatch(
                evidence_id=evidence_id,
                kind=kind,
                file=file_value,
                page=page_value,
                quote=quote,
                chunk_ids=tuple(value for value in chunk_ids if value),
                confidence=confidence,
                reason=(
                    "同名文件中有多份资料命中，需要人工确认路径" if len(exact_groups) > 1
                    else "引文近似匹配，需要人工确认" if approximate_groups and not exact_groups
                    else "引文未能在索引正文块中定位" if not chunk_ids
                    else ""
                ),
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
            "location": str(source.get("location_text") or ""),
        })
    return result
