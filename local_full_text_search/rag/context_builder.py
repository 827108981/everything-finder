from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

from local_full_text_search.rag.models import EvidenceChunk, SourceRef


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: tuple[SourceRef, ...]
    source_map: dict[str, tuple[SourceRef, ...]]
    # 供评测诊断使用的实际上下文条目。正文被预算截断时保留截断后的文本，
    # 报告不能把完整 chunk 误认为已经送进模型。
    entries: tuple[dict[str, object], ...] = ()
    excluded_entries: tuple[dict[str, object], ...] = ()


class ContextBuilder:
    def build(
        self,
        evidence: Sequence[EvidenceChunk],
        *,
        max_chars: int = 12_000,
    ) -> ContextBundle:
        sections: list[str] = []
        sources: list[SourceRef] = []
        source_map: dict[str, tuple[SourceRef, ...]] = {}
        entries: list[dict[str, object]] = []
        excluded_entries: list[dict[str, object]] = []
        # 先按哈希整理，再计算字符预算；防御性重复也必须保留后续来源。
        groups: dict[str, tuple[EvidenceChunk, list[SourceRef], bool]] = {}
        for item in evidence:
            key = item.chunk.text_hash
            if key not in groups:
                groups[key] = (item, list(dict.fromkeys(item.all_sources)), False)
                continue
            first, refs, _ = groups[key]
            for source in item.all_sources:
                if source not in refs:
                    refs.append(source)
            groups[key] = (first, refs, True)
        remaining = max(256, int(max_chars))
        budget_exhausted = False
        for item, group_sources, defensively_merged in groups.values():
            chunk = item.chunk
            if budget_exhausted:
                excluded_entries.append({
                    "chunk_id": chunk.stable_chunk_id,
                    "reason_code": "context_budget_cutoff",
                    "reason": "上下文字符预算不足，整块未纳入",
                })
                continue
            source_id = f"S{len(source_map) + 1}"
            source = group_sources[0]
            decision = item.constraint_decision
            relevant_kinds = {
                judgement.constraint.kind.value for judgement in decision.judgements
            } if decision else set()
            if "device" in relevant_kinds or "topic" in relevant_kinds:
                label = "设备" if "device" in relevant_kinds else "主题"
                applicability = (
                    f"\n{label}适用性：正文与来源已共同确认\n"
                    if decision and decision.support_status == "supported" else
                    f"\n{label}适用性：未确认，仅作背景，不得迁移到目标对象\n"
                )
            else:
                applicability = "\n"
            header = (
                f"[{source_id}]\n来源（{len(group_sources)} 处）：\n"
                + "\n".join(
                    f"- {ref.filename} · {ref.location_text or '未提供'}"
                    for ref in group_sources
                ) + applicability + "内容："
            )
            text = chunk.text.strip()
            section = header + text
            if len(section) > remaining:
                if remaining <= len(header) + 80:
                    budget_exhausted = True
                    excluded_entries.append({
                        "chunk_id": chunk.stable_chunk_id,
                        "reason_code": "context_budget_cutoff",
                        "reason": "上下文字符预算不足，整块未纳入",
                    })
                    continue
                section = header + text[: remaining - len(header)]
            included_text = section[len(header):]
            truncated = len(included_text) < len(text)
            sections.append(section)
            sources.extend(group_sources)
            source_map[source_id] = tuple(group_sources)
            entries.append({
                "source_id": source_id,
                "chunk_id": chunk.stable_chunk_id,
                "filename": source.filename,
                "location": source.location_text,
                "source_count": len(group_sources),
                "sources": [
                    {"filename": ref.filename, "location": ref.location_text,
                     "chunk_id": ref.stable_chunk_id}
                    for ref in group_sources
                ],
                "included_text": included_text,
                "truncated": truncated,
                "defensive_merge": defensively_merged,
                "reason": (
                    "防御性合并相同正文来源；正文预算截断" if truncated else
                    "防御性合并相同正文来源" if defensively_merged else "已纳入上下文"
                ),
            })
            remaining -= len(section) + 2
            if remaining <= 0:
                budget_exhausted = True
        return ContextBundle(
            "\n\n".join(sections), tuple(sources), source_map,
            tuple(entries), tuple(excluded_entries),
        )
