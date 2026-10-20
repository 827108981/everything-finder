from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import PurePath
from typing import Any


_BULLETS = {"•", "·", "‧", "∙", "⋅", "✓", "✔", "➢", "➤", "➔", "→"}
_PAGE_RE = re.compile(r"(?:第\s*)?(\d+)\s*(?:页|面|張|张)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MatchConfig:
    page_tolerance: int = 1
    high_threshold: float = 0.60
    partial_threshold: float = 0.30

    def __post_init__(self) -> None:
        if not (0 < float(self.partial_threshold) < float(self.high_threshold) <= 1):
            raise ValueError("匹配阈值必须满足 0 < partial < high <= 1")
        if int(self.page_tolerance) < 0:
            raise ValueError("页码容差不能为负数")


@dataclass(frozen=True, slots=True)
class NormalizedText:
    value: str
    # 每个规范化字符对应原文中的字符下标，用于把匹配区间映射回原文高亮。
    source_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    kind: str
    evidence_id: str
    status: str
    confidence: str
    coverage: float
    file_match: bool
    page_match: bool | None
    page_delta: int | None
    quote: str
    reason: str
    normalized_start: int | None = None
    normalized_end: int | None = None
    original_start: int | None = None
    original_end: int | None = None

    @property
    def is_countable(self) -> bool:
        return self.status in {"exact", "high"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "evidence_id": self.evidence_id,
            "status": self.status,
            "confidence": self.confidence,
            "coverage": round(float(self.coverage), 6),
            "file_match": self.file_match,
            "page_match": self.page_match,
            "page_delta": self.page_delta,
            "quote": self.quote,
            "reason": self.reason,
            "normalized_start": self.normalized_start,
            "normalized_end": self.normalized_end,
            "original_start": self.original_start,
            "original_end": self.original_end,
        }


def normalize_with_offsets(value: str) -> NormalizedText:
    output: list[str] = []
    offsets: list[int] = []
    for index, character in enumerate(str(value or "")):
        folded = unicodedata.normalize("NFKC", character).casefold()
        for normalized in folded:
            if normalized.isspace() or normalized in _BULLETS:
                continue
            output.append(normalized)
            offsets.append(index)
    return NormalizedText("".join(output), tuple(offsets))


def normalize_text(value: str) -> str:
    return normalize_with_offsets(value).value


def _basename(value: str) -> str:
    return PurePath(str(value or "").replace("\\", "/")).name.casefold()


def _page_number(row: dict[str, Any]) -> int | None:
    for field in ("page_number", "page"):
        value = row.get(field)
        if value is not None and str(value).strip().isdigit():
            return int(str(value).strip())
    match = _PAGE_RE.search(str(row.get("location") or row.get("location_text") or ""))
    return int(match.group(1)) if match else None


def _status(coverage: float, config: MatchConfig) -> tuple[str, str]:
    if coverage >= 0.999999:
        return "exact", "完整文本覆盖"
    if coverage >= config.high_threshold:
        return "high", "高覆盖匹配"
    if coverage >= config.partial_threshold:
        return "partial", "部分匹配"
    return "unmatched", "未达到匹配阈值"


def match_quote(
    chunk: dict[str, Any],
    evidence: dict[str, Any],
    *,
    kind: str,
    evidence_id: str,
    config: MatchConfig | None = None,
) -> EvidenceMatch:
    config = config or MatchConfig()
    quote = str(evidence.get("quote") or "")
    chunk_file = _basename(str(chunk.get("filename") or chunk.get("file_path") or ""))
    evidence_file = _basename(str(evidence.get("file") or ""))
    file_match = bool(chunk_file and evidence_file and chunk_file == evidence_file)
    if not file_match:
        return EvidenceMatch(
            kind, evidence_id, "unmatched", "unknown", 0.0, False, False, None,
            quote, "文件名不一致",
        )

    expected_page = _page_number(evidence)
    actual_page = _page_number(chunk)
    page_delta = abs(actual_page - expected_page) if actual_page is not None and expected_page is not None else None
    page_match: bool | None = None
    if expected_page is not None and actual_page is not None:
        page_match = page_delta <= config.page_tolerance
        if not page_match:
            return EvidenceMatch(
                kind, evidence_id, "unmatched", "unknown", 0.0, True, False, page_delta,
                quote, f"页码相差 {page_delta}，超过容差 {config.page_tolerance}",
            )
    elif expected_page is not None or actual_page is not None:
        page_match = None

    quote_norm = normalize_with_offsets(quote)
    chunk_norm = normalize_with_offsets(str(chunk.get("text") or ""))
    if not quote_norm.value:
        return EvidenceMatch(
            kind, evidence_id, "invalid", "unknown", 0.0, True, page_match, page_delta,
            quote, "引文为空，无法计算覆盖率",
        )
    if not chunk_norm.value:
        return EvidenceMatch(
            kind, evidence_id, "unmatched", "unknown", 0.0, True, page_match, page_delta,
            quote, "候选正文为空",
        )
    match = SequenceMatcher(None, quote_norm.value, chunk_norm.value, autojunk=False).find_longest_match(
        0, len(quote_norm.value), 0, len(chunk_norm.value)
    )
    coverage = match.size / len(quote_norm.value)
    status, label = _status(coverage, config)
    original_start = original_end = None
    if match.size:
        original_start = chunk_norm.source_offsets[match.b]
        original_end = chunk_norm.source_offsets[match.b + match.size - 1] + 1
    return EvidenceMatch(
        kind,
        evidence_id,
        status,
        "exact" if status == "exact" else "review" if status in {"high", "partial"} else "unknown",
        coverage,
        True,
        page_match,
        page_delta,
        quote,
        label,
        match.a,
        match.a + match.size,
        original_start,
        original_end,
    )


def highlight_segments(text: str, matches: list[EvidenceMatch]) -> list[dict[str, Any]]:
    """按原文区间生成安全渲染所需的片段，不直接拼接 HTML。"""

    spans: list[tuple[int, int, set[str]]] = []
    for item in matches:
        if item.original_start is None or item.original_end is None or item.status in {"unmatched", "invalid"}:
            continue
        kinds = {item.kind}
        if item.status == "partial":
            kinds.add("partial")
        spans.append((item.original_start, item.original_end, kinds))
    if not spans:
        return [{"text": str(text or ""), "kinds": []}]
    points = {0, len(text)}
    for start, end, _ in spans:
        points.add(max(0, min(len(text), start)))
        points.add(max(0, min(len(text), end)))
    ordered = sorted(points)
    output: list[dict[str, Any]] = []
    for start, end in zip(ordered, ordered[1:]):
        if start == end:
            continue
        kinds = {kind for left, right, values in spans if left < end and right > start for kind in values}
        output.append({"text": text[start:end], "kinds": sorted(kinds)})
    return output
