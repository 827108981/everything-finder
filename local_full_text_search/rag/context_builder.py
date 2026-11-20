from __future__ import annotations

from dataclasses import dataclass, field, replace
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
    evidence_facts: tuple[dict, ...] = ()
    sections: dict = field(default_factory=dict)


def organize_context(context, plan, excluded=()):
    """Classify only included text; rejected candidates are not reference sources."""
    from local_full_text_search.rag.fact_schema import extract_facts, merge_facts
    facts = []
    sections = {'question': plan.parsed_request, 'primary': [], 'supporting': [],
                'scope': [], 'conditions_and_branches': [], 'conflicts': [], 'inapplicable': list(excluded)}
    for entry in context.entries:
        for source in context.source_map.get(entry['source_id'], ()):
            facts.extend(extract_facts(entry['included_text'], source, entry['source_id']))
    facts = merge_facts(facts)
    for fact in facts:
        name = 'conflicts' if fact['evidence_role'] == 'conflict' else 'primary'
        sections[name].append(fact['fact_id'])
        if fact['scope']: sections['scope'].append(fact['fact_id'])
        if fact['conditions'] or fact['fact_type'] in {'condition', 'solution'}:
            sections['conditions_and_branches'].append(fact['fact_id'])
    return replace(context, evidence_facts=tuple(facts), sections=sections)


class ContextBuilder:
    def build(
        self,
        evidence: Sequence[EvidenceChunk],
        *,
        max_chars: int = 12_000,
        preserve_complete_chunks: bool = False,
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
        remaining = max(0 if preserve_complete_chunks else 256, int(max_chars))
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
            unconfirmed = item.query_plan_diagnostics.get('unconfirmed_conditions', [])
            if unconfirmed:
                applicability += ('用户条件尚未由本段确认：'+ '、'.join(unconfirmed)
                                  +'。只能按原文范围参考，不得据此确认适用或扩大范围。\n')
            text = chunk.text.strip()
            rows = item.query_plan_diagnostics.get('structured_rows', [])
            spans = []
            if rows:
                # Exact source slices only; malformed/incomplete tables are never projected.
                verified = [r for r in rows if chunk.text[r['start']:r['end']] == r['quote']]
                if len(verified) == len(rows):
                    prefix_end=max(r.get('prefix_end',0) for r in verified)
                    spans=([{'start':0,'end':prefix_end,'kind':'scope_and_headers'}] if prefix_end else [])
                    spans += [{'start':r['start'],'end':r['end'],'identifier':r['identifier']} for r in verified]
                    text = '\n'.join(chunk.text[s['start']:s['end']] for s in spans)
            scope_observations=item.query_plan_diagnostics.get('scope_observations',[])
            if spans:
                scope_observations=[x for x in scope_observations if any(s['start']<=x['start'] and x['end']<=s['end'] for s in spans)]
            scope_note=''
            if scope_observations:
                mentions='、'.join(dict.fromkeys(x['text'] for x in scope_observations))
                scope_note='来源范围/对象线索：'+mentions+'。仅表示本段出现，须结合原文绑定，不得泛化或将子部件等同整套组件。\n'
            header = (
                f"[{source_id}]\n来源（{len(group_sources)} 处）：\n"
                + "\n".join(
                    f"- {ref.filename} · {ref.location_text or '未提供'}"
                    for ref in group_sources
                ) + applicability + scope_note + "内容："
            )
            section = header + text
            if len(section) > remaining:
                if preserve_complete_chunks:
                    excluded_entries.append({
                        "chunk_id": chunk.stable_chunk_id,
                        "reason_code": "complete_chunk_exceeds_remaining_budget",
                        "reason": "完整正文及来源超过剩余字符预算；不截断步骤，继续检查后续候选",
                        "required_chars": len(section),
                        "remaining_chars": remaining,
                    })
                    continue
                if spans:
                    excluded_entries.append({'chunk_id':chunk.stable_chunk_id,'reason_code':'complete_row_exceeds_budget',
                        'reason':'完整表头/条件/行超出预算，不能截成部分处理步骤；继续检查后续较短证据'})
                    continue
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
            if truncated and scope_note:
                # Do not advertise scope words which may have been cut from the body.
                header=header.replace(scope_note,'')
                section=header+included_text
            context_diagnostics=dict(item.query_plan_diagnostics)
            relations=context_diagnostics.get('condition_relations')
            if relations:
                # A proof outside the actual sent text cannot certify applicability.
                checks=[]
                for check in relations.get('checks', []):
                    check=dict(check)
                    if check.get('candidate_quote') and check['candidate_quote'] not in included_text:
                        check.update(status='unknown', candidate_quote='', source_start=None,
                                     source_end=None, reason_code='proof_not_included',
                                     reason='适用性依据未完整进入实际正文，不能沿用完整父块的证明')
                    checks.append(check)
                context_diagnostics['condition_relations']={**relations, 'checks':checks}
            if 'scope_observations' in context_diagnostics:
                context_diagnostics['scope_observations']=[x for x in scope_observations if x['text'] in included_text]
            sections.append(section)
            sources.extend(group_sources)
            source_map[source_id] = tuple(group_sources)
            entries.append({
                "source_id": source_id,
                "chunk_id": chunk.stable_chunk_id,
                "member_chunk_ids": list(dict.fromkeys(ref.stable_chunk_id for ref in group_sources)),
                "filename": source.filename,
                "location": source.location_text,
                "source_count": len(group_sources),
                "sources": [
                    {"filename": ref.filename, "file_path": ref.file_path, "location": ref.location_text,
                     "chunk_id": ref.stable_chunk_id}
                    for ref in group_sources
                ],
                "included_text": included_text,
                "query_plan_diagnostics": context_diagnostics,
                "source_spans": spans,
                "projection": "exact_identifier_rows" if spans else "full_chunk",
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
