"""Field agreement with explicit scope. Equal values never enlarge applicability."""
import json
import re
from local_full_text_search.rag.fact_schema import explicit_scope, normalized
from local_full_text_search.rag.models import EvidenceChunk


def list_fields(text, question):
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    found = []
    for key, value in zip(lines, lines[1:]):
        if not 2 <= len(key) <= 24 or not re.search(r'、|,|，|;', value): continue
        if value.endswith(('/', '、', ',', '，', '-', '或')): continue
        values = [s.strip(' 。；;') for s in re.split(r'、|,|，|;', value)]
        if not 2 <= len(values) <= 20 or any(not v or len(v) > 40 for v in values): continue
        if key[-2:] not in question or not any(key[i:i+2] in question for i in range(max(1, len(key)-3))): continue
        found.append({'fact': key, 'values': list({normalized(v): v for v in values}.values()),
                      'quote': key+'\n'+value, 'object': explicit_scope(text).get('对象', [])})
    return found


def source_scope(source, plan, text=''):
    scope = explicit_scope(text)
    return {'file': source.file_path, 'page': source.page_number or source.slide_number,
            'chunk_id': source.stable_chunk_id, 'scope': scope,
            'platform_group': 'unclassified', 'explicit_scope': scope,
            'scope_basis': 'Explicit labelled body restrictions only; filename is a discovery hint, never a scope proof',
            'filename_models': re.findall(plan.adapter_policy.get('patterns', {}).get('instrument_model', r'(?!)'), source.filename)}


def summarize(records):
    grouped = {}
    for row in records:
        source = row['source']; scope = source.get('scope') or source.get('explicit_scope') or {}; obj = row.get('object', [])
        local = '' if scope and obj else source['file']
        key = json.dumps([normalized(row['fact']), scope, obj, local], sort_keys=True, ensure_ascii=False)
        grouped.setdefault(key, []).append(row)
    result = []
    for rows in grouped.values():
        sets = {tuple(sorted({normalized(v) for v in r['values']})) for r in rows}
        sources = list({(r['source']['file'], r['source'].get('page'), r['source'].get('chunk_id')): r['source'] for r in rows}.values())
        conflict = len(sets) > 1
        result.append({'fact': rows[0]['fact'], 'object': rows[0].get('object', []),
            'scope': rows[0]['source'].get('scope') or rows[0]['source'].get('explicit_scope') or {},
            'values': [] if conflict else rows[0]['values'], 'supporting_sources': sources,
            'supporting_source_count': len({s['file'] for s in sources}), 'platform_groups': [],
            'conflicts': [{'source': r['source'], 'values': r['values']} for r in rows] if conflict else [],
            'consensus_level': 'conflict' if conflict else 'scoped_agreement' if len({s['file'] for s in sources}) > 1 else 'single_source',
            'scope_basis': 'Same field, explicit object and explicit scope required for cross-document agreement; never infer all platforms'})
    return result


def supplement(retriever, query, plan, items, token):
    if not plan.adapter_active or 'enumeration' not in plan.parsed_request['requested_fields']: return list(items)
    result = list(items); records, additions = [], []
    connection = retriever.store.connect_read_only(retriever.metadata_path)
    try:
        for row in connection.execute("SELECT stable_chunk_id,text FROM rag_chunks WHERE instr(text,'、')>0 OR instr(text,',')>0 OR instr(text,'，')>0 OR instr(text,';')>0"):
            token.throw_if_cancelled(); fields = list_fields(row['text'], query.text)
            if not fields: continue
            chunk = retriever.store.load_chunk(connection, row['stable_chunk_id'])
            if not retriever.store._matches_filters(chunk, query.root_ids, query.extensions, query.include_ocr, query.min_ocr_confidence): continue
            scope = source_scope(chunk.source, plan, chunk.text)
            records.extend({**field, 'source': scope} for field in fields)
            item = next((x for x in result if x.chunk.stable_chunk_id == chunk.stable_chunk_id), None)
            if item is None:
                item = EvidenceChunk(chunk=chunk, lexical_rank=len(result)+1); result.append(item)
            item.query_plan_diagnostics['matched_list_fields'] = fields; additions.append(item)
        agreement = summarize(records); seen = set()
        for item in additions:
            item.query_plan_diagnostics['source_consensus'] = agreement
            scope = source_scope(item.chunk.source, plan, item.chunk.text)
            key = (item.chunk.document_key, json.dumps(scope['scope'], sort_keys=True))
            if key not in seen and len(seen) < 3:
                item.query_plan_diagnostics['source_representative'] = True
                item.query_plan_diagnostics['reservation_reason'] = 'Bounded distinct-source enumeration; no scope inference'
                seen.add(key)
        retriever.route_diagnostics.append({'route_id': 'structured_field_agreement', 'source_consensus': agreement,
            'candidate_count': len(additions), 'representative_count': len(seen)})
    finally: connection.close()
    return reserve(result)


def reserve(items):
    protected = [i for i in items if i.query_plan_diagnostics.get('source_representative')][:3]
    for item in items:
        if item.query_plan_diagnostics.get('source_representative') and item not in protected:
            item.query_plan_diagnostics['source_representative'] = False
            item.query_plan_diagnostics['protection_limit_reason'] = 'Supplemental representative cap; retain ordinary ranking'
    return protected + [i for i in items if i not in protected]


def attach_context_consensus(context):
    records = []
    for entry in context.entries:
        diagnostics = entry.get('query_plan_diagnostics', {})
        for field in diagnostics.get('matched_list_fields', []):
            if normalized(field['quote']) not in normalized(entry['included_text']): continue
            for source in context.source_map.get(entry['source_id'], ()):
                records.append({**field, 'source': {'file': source.file_path,
                    'page': source.page_number or source.slide_number, 'chunk_id': source.stable_chunk_id,
                    'scope': explicit_scope(entry['included_text'])}})
    return context, summarize(records)


def generation_consensus_contract(consensus):
    # Import compatibility; evidence data must not become dynamic instructions.
    return ''


def protect_topic_causes(plan, items):
    """Legacy import compatibility only; topic-specific promotion was removed."""
    return list(items)


def scene_eligible(plan, item):
    """Legacy inspection helper, not authorization to inherit adjacent scenes."""
    from dataclasses import replace
    from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
    from local_full_text_search.rag.domain_adapter import DomainAdapter
    from local_full_text_search.rag.models import ConstraintStatus
    probe = replace(item)
    adapter = DomainAdapter(aliases=plan.adapter_policy.get('aliases', {}), patterns=plan.adapter_policy.get('patterns', {}))
    ConstraintMatcher(domain_adapter=adapter).evaluate(plan, [probe])
    return all(j.status == ConstraintStatus.MATCH for j in probe.constraint_decision.judgements if j.constraint.is_hard)
