"""跨格式正文身份：只自动合并事实字符完全一致的排版变体。"""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher


NORMALIZER_VERSION = "content-identity-v2"
_FACT_SYMBOLS = "%‰℃°≤≥<>≠=±×÷∓~∼→←↔"
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"


@dataclass(frozen=True, slots=True)
class ContentComparison:
    status: str
    canonical_similarity: float
    differing_fragments: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class CanonicalizedContent:
    text: str
    # 每个规范化字符对应的原文半开区间；NFKC 展开时多个字符可对应同一区间。
    offsets: tuple[tuple[int, int], ...]


def canonicalize_with_offsets(text: str, structure=None) -> CanonicalizedContent:
    """去除排版噪声，保留数字、单位、否定词、业务动作和表格内容。

    原始字符串相似度只能提供候选，不能证明事实等价。这里刻意不做
    同义改写或近似阈值合并，避免 99% 相似但关键数值不同的误合并。
    """

    raw = str(text or "")
    output: list[str] = []
    offsets: list[tuple[int, int]] = []
    for index, original in enumerate(raw):
        def append(value: str) -> None:
            for char in value:
                output.append(char)
                offsets.append((index, index + 1))

        # 上标与普通数字的技术含义可能不同；NFKC 会把 m² 错折叠成 m2。
        character = original if original in _SUPERSCRIPTS else unicodedata.normalize("NFKC", original)
        before = raw[index - 1] if index else ""
        after = raw[index + 1] if index + 1 < len(raw) else ""
        if character.isspace():
            if original == "\t":
                # 制表符是常见单元格边界，不能当作普通 PDF 断行删除。
                append("¦")
                continue
            previous = next((value for value in reversed(raw[:index]) if not value.isspace()), "")
            following = next((value for value in raw[index + 1:] if not value.isspace()), "")
            # 数字/英文词的边界及表格单元格不能因为去空白变成同一个值。
            if previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum():
                if previous.isdigit() == following.isdigit():
                    if not output or output[-1] != "¦":
                        append("¦")
            continue
        if character.isalnum():
            append(character)
        elif character in "+-" and (before.isalnum() or after.isalnum()):
            # 正负号、型号连接符、A-B 范围均可能改变事实；行首列表符号不计入。
            append(character)
        elif character in "./" and before.isalnum() and after.isalnum():
            # 小数、型号版本和比例符号均保留。
            append(character)
        elif character == ":":
            previous = next((value for value in reversed(raw[:index]) if not value.isspace()), "")
            following = next((value for value in raw[index + 1:] if not value.isspace()), "")
            if previous.isascii() and previous.isalnum() and following.isalnum():
                # 编号到属性的结构边界不可任意移动，例如 P02:废液泵。
                append(":")
        elif character == "|":
            append("¦")
        elif character in _FACT_SYMBOLS:
            # 比较、单位及算术符号是事实的一部分，即使周围有空格或换行也保留。
            append(character)
    return CanonicalizedContent("".join(output), tuple(offsets))


def canonical_content(text: str) -> str:
    return canonicalize_with_offsets(text).text


def content_identity(text: str) -> str:
    return hashlib.sha256(canonical_content(text).encode("utf-8")).hexdigest()


def compare_content(left: str, right: str) -> ContentComparison:
    """相似度只提出疑似；只有保真规范化完全一致才自动合并。"""
    a, b = canonical_content(left), canonical_content(right)
    if not a or not b:
        return ContentComparison("different", 0.0, (), "至少一侧正文为空")
    if a == b:
        status = "exact" if str(left) == str(right) else "layout_equivalent"
        return ContentComparison(status, 1.0, (), "排版规范化后正文一致")
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    differences = tuple(
        f"{a[i:j]} → {b[k:l]}"
        for operation, i, j, k, l in matcher.get_opcodes()
        if operation != "equal"
    )
    ratio = matcher.ratio()
    status = "suspected_equivalent" if ratio >= 0.95 else "different"
    return ContentComparison(status, ratio, differences[:10], "正文仍有事实字符差异，不自动合并")
