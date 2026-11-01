from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from local_full_text_search.rag.models import RagChunk


VECTOR_UNIT_POLICY_VERSION = "token-window-v2"
_SENTENCE_END = re.compile(r"(?<=[。！？!?；;])")


@dataclass(frozen=True, slots=True)
class VectorUnitPolicy:
    target_tokens: int = 128
    max_tokens: int = 192
    overlap_tokens: int = 32
    include_source_context: bool = True

    def __post_init__(self) -> None:
        if self.target_tokens < 16:
            raise ValueError("向量单元目标 token 数不能小于 16")
        if self.max_tokens < self.target_tokens:
            raise ValueError("向量单元最大 token 数不能小于目标值")
        if self.overlap_tokens < 0 or self.overlap_tokens >= self.target_tokens:
            raise ValueError("向量单元重叠 token 数必须小于目标值")

    @classmethod
    def from_settings(cls, settings) -> "VectorUnitPolicy":
        return cls(
            target_tokens=int(getattr(settings, "rag_vector_unit_target_tokens", 128)),
            max_tokens=int(getattr(settings, "rag_vector_unit_max_tokens", 192)),
            overlap_tokens=int(getattr(settings, "rag_vector_unit_overlap_tokens", 32)),
            include_source_context=bool(getattr(
                settings, "rag_vector_unit_include_source_context", True
            )),
        )

    @property
    def fingerprint(self) -> str:
        payload = {
            "version": VECTOR_UNIT_POLICY_VERSION,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "include_source_context": self.include_source_context,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class VectorRetrievalUnit:
    unit_id: str
    parent_chunk_id: str
    ordinal: int
    text: str
    embedding_text: str
    kind: str
    start_char: int
    end_char: int
    token_count: int


class VectorUnitBuilder:
    """按模型 token 边界切分正文，不依赖领域词表或测试题。"""

    def __init__(self, count_tokens: Callable[[str], int], policy: VectorUnitPolicy) -> None:
        self.count_tokens = count_tokens
        self.policy = policy

    def build(self, chunk: RagChunk) -> tuple[VectorRetrievalUnit, ...]:
        raw_text = chunk.text.replace("\r\n", "\n").replace("\r", "\n")
        leading_offset = len(raw_text) - len(raw_text.lstrip())
        text = raw_text.strip()
        if not text:
            return ()
        atoms = self._atoms(text)
        windows: list[tuple[int, int, str, str]] = []
        start = 0
        while start < len(atoms):
            end = start
            selected: list[tuple[int, int, str, str]] = []
            while end < len(atoms):
                candidate = selected + [atoms[end]]
                candidate_text = text[candidate[0][0]:candidate[-1][1]].strip()
                if selected and self.count_tokens(candidate_text) > self.policy.target_tokens:
                    break
                selected = candidate
                end += 1
                if self.count_tokens(candidate_text) >= self.policy.target_tokens:
                    break
            if not selected:
                selected = [atoms[start]]
                end = start + 1
            unit_text = text[selected[0][0]:selected[-1][1]].strip()
            if self.count_tokens(unit_text) > self.policy.max_tokens:
                replacement = []
                cursor = selected[0][0]
                for part in self._split_long(unit_text):
                    located = text.find(part, cursor)
                    located = cursor if located < 0 else located
                    replacement.append((located, located + len(part), part, "long_split"))
                    cursor = located + len(part)
                atoms[start:end] = replacement
                continue
            windows.append((selected[0][0], selected[-1][1], unit_text, self._kind(selected)))
            if end >= len(atoms):
                break
            overlap = 0
            next_start = end
            while next_start > start + 1 and overlap < self.policy.overlap_tokens:
                next_start -= 1
                overlap += self.count_tokens(atoms[next_start][2])
            start = next_start if next_start > start else end

        result = []
        for ordinal, (start_char, end_char, unit_text, kind) in enumerate(windows):
            embedding_text = self._embedding_text(chunk, unit_text)
            token_count = self.count_tokens(embedding_text)
            if token_count > self.policy.max_tokens:
                # 来源背景不得挤占正文导致静默截断；超限时仅嵌入正文。
                embedding_text = unit_text
                token_count = self.count_tokens(embedding_text)
            if token_count > self.policy.max_tokens:
                raise ValueError(
                    f"向量单元仍超过 token 上限：{chunk.stable_chunk_id}#{ordinal}={token_count}"
                )
            unit_id = hashlib.sha256(
                f"{chunk.stable_chunk_id}\0{ordinal}\0{start_char}\0{end_char}\0{unit_text}".encode("utf-8")
            ).hexdigest()
            result.append(VectorRetrievalUnit(
                unit_id=unit_id, parent_chunk_id=chunk.stable_chunk_id, ordinal=ordinal,
                text=unit_text, embedding_text=embedding_text, kind=kind,
                start_char=start_char + leading_offset,
                end_char=end_char + leading_offset,
                token_count=token_count,
            ))
        return tuple(result)

    def _atoms(self, text: str) -> list[tuple[int, int, str, str]]:
        atoms: list[tuple[int, int, str, str]] = []
        cursor = 0
        for raw_line in text.splitlines(keepends=True):
            line = raw_line.strip()
            line_start = cursor + max(0, raw_line.find(line))
            cursor += len(raw_line)
            if not line:
                continue
            parts = [part.strip() for part in _SENTENCE_END.split(line) if part.strip()]
            search_from = line_start
            for part in parts:
                located = text.find(part, search_from)
                located = search_from if located < 0 else located
                atoms.append((located, located + len(part), part, "sentence"))
                search_from = located + len(part)
        return atoms or [(0, len(text), text, "text")]

    def _split_long(self, text: str) -> list[str]:
        output: list[str] = []
        remaining = text
        while remaining:
            if self.count_tokens(remaining) <= self.policy.max_tokens:
                output.append(remaining)
                break
            low, high, best = 1, len(remaining), 1
            while low <= high:
                middle = (low + high) // 2
                if self.count_tokens(remaining[:middle]) <= self.policy.max_tokens:
                    best = middle
                    low = middle + 1
                else:
                    high = middle - 1
            boundary = max(
                remaining.rfind(mark, 0, best) for mark in ("\n", "。", "；", "，", " ")
            )
            cut = boundary + 1 if boundary >= best // 2 else best
            output.append(remaining[:cut].strip())
            remaining = remaining[cut:].strip()
        return [item for item in output if item]

    def _embedding_text(self, chunk: RagChunk, text: str) -> str:
        if not self.policy.include_source_context:
            return text
        title = Path(chunk.source.filename).stem.strip()
        location = chunk.source.location_text.strip()
        prefix = [f"来源：{title}" if title else "", f"位置：{location}" if location else ""]
        return "\n".join([item for item in prefix if item] + [f"正文：{text}"])

    @staticmethod
    def _kind(atoms: list[tuple[int, int, str, str]]) -> str:
        texts = [item[2] for item in atoms]
        table_signals = sum("|" in item or "\t" in item for item in texts)
        short_lines = sum(len(item) <= 32 for item in texts)
        if table_signals or (len(texts) >= 3 and short_lines / len(texts) >= 0.7):
            return "line_window"
        return "sentence_window"
