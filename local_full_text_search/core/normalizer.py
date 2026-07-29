from __future__ import annotations

import re
import unicodedata

ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\ufeff]")
WHITESPACE_PATTERN = re.compile(r"\s+")
CONTROL_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HYPHEN_PATTERN = re.compile(r"[-‐‑‒–—―﹣－]")
PUNCT_PATTERN = re.compile(r"[\s\.,;:!?，。；：！？、'\"“”‘’（）()\[\]{}<>《》]")
CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


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
    if mode == "regex":
        return [text]
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


def make_context(
    raw_text: str,
    terms: list[str],
    context_before: int = 100,
    context_after: int = 180,
    *,
    case_sensitive: bool = False,
    ignore_spaces: bool = False,
    ignore_hyphens: bool = False,
) -> str:
    if not raw_text:
        return ""
    normalized_raw, raw_offsets = _normalize_with_raw_offsets(
        raw_text,
        case_sensitive=case_sensitive,
        ignore_spaces=ignore_spaces,
        ignore_hyphens=ignore_hyphens,
    )
    best_index = -1
    best_length = 0
    for term in terms:
        norm_term = normalize_text(
            term,
            case_sensitive=case_sensitive,
            ignore_spaces=ignore_spaces,
            ignore_hyphens=ignore_hyphens,
        )
        if not norm_term:
            continue
        index = normalized_raw.find(norm_term)
        if index >= 0 and (best_index < 0 or index < best_index):
            best_index = index
            best_length = len(norm_term)
    if best_index < 0:
        snippet = raw_text[: context_before + context_after]
        return snippet + ("..." if len(raw_text) > len(snippet) else "")
    raw_match_start = raw_offsets[best_index][0]
    raw_match_end = raw_offsets[min(len(raw_offsets) - 1, best_index + best_length - 1)][1]
    start = max(0, raw_match_start - context_before)
    end = min(len(raw_text), raw_match_end + context_after)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(raw_text) else ""
    return prefix + raw_text[start:end].replace("\n", " ") + suffix


def _normalize_with_raw_offsets(
    text: str,
    *,
    case_sensitive: bool,
    ignore_spaces: bool,
    ignore_hyphens: bool,
) -> tuple[str, list[tuple[int, int]]]:
    """Normalize text while retaining the source span for every output character."""

    output: list[str] = []
    offsets: list[tuple[int, int]] = []
    pending_space: tuple[int, int] | None = None
    for raw_index, raw_character in enumerate(text):
        value = unicodedata.normalize("NFKC", raw_character)
        for character in value:
            if ZERO_WIDTH_PATTERN.fullmatch(character):
                continue
            if CONTROL_PATTERN.fullmatch(character):
                character = " "
            if ignore_hyphens and HYPHEN_PATTERN.fullmatch(character):
                continue
            if character.isspace():
                if ignore_spaces:
                    continue
                if output and output[-1] != " ":
                    pending_space = (raw_index, raw_index + 1)
                continue
            if pending_space is not None:
                output.append(" ")
                offsets.append(pending_space)
                pending_space = None
            normalized_character = character if case_sensitive else character.lower()
            for item in normalized_character:
                output.append(item)
                offsets.append((raw_index, raw_index + 1))
    if output and output[-1] == " ":
        output.pop()
        offsets.pop()
    return "".join(output), offsets


def count_hits(normalized_text: str, normalized_terms: list[str]) -> int:
    total = 0
    for term in normalized_terms:
        if term:
            total += normalized_text.count(term)
    return total


def contains_cjk(text: str | None) -> bool:
    return bool(text and CJK_PATTERN.search(text))
