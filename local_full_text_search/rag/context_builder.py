from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from local_full_text_search.rag.models import EvidenceChunk, SourceRef


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: tuple[SourceRef, ...]
    source_map: dict[str, SourceRef]
    # 供评测诊断使用的实际上下文条目。正文被预算截断时保留截断后的文本，
    # 报告不能把完整 chunk 误认为已经送进模型。
    entries: tuple[dict[str, object], ...] = ()


class ContextBuilder:
    def build(
        self,
        evidence: Sequence[EvidenceChunk],
        *,
        max_chars: int = 12_000,
    ) -> ContextBundle:
        sections: list[str] = []
        sources: list[SourceRef] = []
        source_map: dict[str, SourceRef] = {}
        entries: list[dict[str, object]] = []
        seen_text: set[str] = set()
        remaining = max(256, int(max_chars))
        for item in evidence:
            chunk = item.chunk
            if chunk.text_hash in seen_text:
                continue
            seen_text.add(chunk.text_hash)
            source_id = f"S{len(sources) + 1}"
            source = chunk.source
            header = (
                f"[{source_id}]\n文件：{source.filename}\n"
                f"位置：{source.location_text or '未提供'}\n内容："
            )
            text = chunk.text.strip()
            section = header + text
            if len(section) > remaining:
                if remaining <= len(header) + 80:
                    break
                section = header + text[: remaining - len(header)]
            included_text = section[len(header):]
            truncated = len(included_text) < len(text)
            sections.append(section)
            sources.append(source)
            source_map[source_id] = source
            entries.append({
                "source_id": source_id,
                "chunk_id": chunk.stable_chunk_id,
                "filename": source.filename,
                "location": source.location_text,
                "included_text": included_text,
                "truncated": truncated,
                "reason": "正文预算截断" if truncated else "已纳入上下文",
            })
            remaining -= len(section) + 2
            if remaining <= 0:
                break
        return ContextBundle("\n\n".join(sections), tuple(sources), source_map, tuple(entries))
