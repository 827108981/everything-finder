"""Experimental evidence audit; production scoring and historical reports unchanged."""
from local_full_text_search.rag.evaluation.stage_scoring import _part_rank
from local_full_text_search.rag.content_identity import canonical_content

def negative_span_audit(candidates,mappings,source_rows=None):
    full=[];page_only=[];unresolved=[]
    actual=[c for c in candidates if isinstance(c.get('included_text'),str)]
    for m in mappings:
        if m.kind!='hard_negative':continue
        source_hit=any({c.get('chunk_id'),*c.get('member_chunk_ids',[])}.intersection(m.chunk_ids) for c in candidates)
        confirmed=[a for a in m.alternatives if a.confirmed]
        if confirmed:
            matched=any(a.parts and all(_part_rank(p,m.quote,actual,'context') is not None for p in a.parts) for a in confirmed)
            if source_hit and not any({c.get('chunk_id'),*c.get('member_chunk_ids',[])}.intersection(m.chunk_ids) for c in actual):
                unresolved.append(m.evidence_id)
        else:
            # Fail closed for scoring: IDs alone cannot prove a quoted row is present.
            matched=False
            if source_hit:unresolved.append(m.evidence_id)
        if matched and source_rows is not None:
            originals={r['chunk_id']:r['text'] for r in source_rows}
            ambiguous=[]
            for a in confirmed:
                for part in a.parts:
                    original=canonical_content(originals.get(part.chunk_id,''))
                    expected=canonical_content(m.quote)[part.quote_start:part.quote_end]
                    if expected and original.count(expected)>1:
                        ambiguous.append((part.chunk_id,original))
            # A projected row containing a shared value cannot identify its source
            # occurrence. Whole original chunks can still prove full-page exposure.
            if ambiguous and not all(any(source in canonical_content(c['included_text']) and {c.get('chunk_id'),*c.get('member_chunk_ids',[])}.intersection({cid}) for c in actual) for cid,source in ambiguous):
                matched=False;unresolved.append(m.evidence_id)
        if matched:full.append(m.evidence_id)
        elif source_hit:page_only.append(m.evidence_id)
    return {'negative_quote_hits':full,'same_parent_without_full_quote':page_only,'unresolved_mapping':unresolved,
            'limitation':'Full source-anchored quote coverage only. Partial negative fragments remain diagnostic; identical values shared by rows need source-row offsets, not string matching alone.'}
