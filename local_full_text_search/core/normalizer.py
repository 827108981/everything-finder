from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHEN_PATTERN = re.compile(r"[-‐‑‒–—―﹣－]")
PUNCT_PATTERN = re.compile(r"[\s\.,;:!?，。；：！？、'\"“”‘’（）()\[\]{}<>《》]")


def normalize_text(
    text: str | None,
    *,
    case_sensitive: bool = False,
    ignore_spaces: bool = False,
    ignore_hyphens: bool = False,
    ignore_punctuation: bool = False,
) -> str:
    if not text:
        return ""
    value = unicodedata.normalize("NFKC", text)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = ZERO_WIDTH_PATTERN.sub("", value)
    value = CONTROL_PATTERN.sub(" ", value)
    if ignore_hyphens:
        value = HYPHEN_PATTERN.sub("", value)
    if ignore_punctuation:
        value = PUNCT_PATTERN.sub("", value)
    value = WHITESPACE_PATTERN.sub("" if ignore_spaces else " ", value)
    value = value.strip()
    if not case_sensitive:
        value = value.lower()
    return value


def parse_terms(text: str, mode: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if mode == "phrase":
        return [_strip_wrapping_quotes(text)]
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”', text)
    quoted_terms = [a or b for a, b in quoted if a or b]
    without_quotes = re.sub(r'"[^"]+"|“[^”]+”', " ", text)
    plain_terms = [part for part in re.split(r"\s+", without_quotes.strip()) if part]
    if mode == "exact":
        return [_strip_wrapping_quotes(text)]
    return quoted_terms + plain_terms


def _strip_wrapping_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] in {'"', "“"} and text[-1] in {'"', "”"}:
        return text[1:-1]
    return text


def make_context(raw_text: str, terms: list[str], context_before: int = 80, context_after: int = 120) -> str:
    if not raw_text:
        return ""
    normalized_raw = normalize_text(raw_text)
    best_index = -1
    best_term = ""
    for term in terms:
        norm_term = normalize_text(term)
        if not norm_term:
            continue
        index = normalized_raw.find(norm_term)
        if index >= 0 and (best_index < 0 or index < best_index):
            best_index = index
            best_term = term
    if best_index < 0:
        snippet = raw_text[: context_before + context_after]
        return snippet + ("..." if len(raw_text) > len(snippet) else "")
    # Normalized-to-raw offsets are approximate after NFKC/collapse; this is still useful
    # for result preview while avoiding expensive full mapping in the first version.
    start = max(0, best_index - context_before)
    end = min(len(raw_text), best_index + len(best_term) + context_after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(raw_text) else ""
    return prefix + raw_text[start:end].replace("\n", " ") + suffix


def count_hits(normalized_text: str, normalized_terms: list[str]) -> int:
    total = 0
    for term in normalized_terms:
        if term:
            total += normalized_text.count(term)
    return total
