"""Offline outcome audit. Gold is used here only, after runtime selection."""
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
import json
from pathlib import Path
import statistics

from tools.validate_selection_factorial import OUT, ROOT, HISTORY, load_inputs, reconstruct, sha, write
from tools.selection_factorial import ARMS
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.content_identity import canonical_content


def read_arm(arm):
    return {r['id']:r for r in map(json.loads,(OUT/(arm+'.jsonl')).read_text(encoding='utf8').splitlines())}


def mean(values):
    return statistics.mean(values) if values else None


def summarize(rows):
    scored = [r for r in rows if r['scores']['context']['gold_evidence_total_declared']]
    scores = [r['scores']['context'] for r in scored]
    groups = defaultdict(list)
    for r in scored: groups[r['group_id']].append(r['scores']['context']['gold_evidence_recall'])
    return {'cases':len(rows),'positive_cases':len(scored),'complete':sum(s['gold_evidence_recall']==1 for s in scores),
        'any_gold':sum(s['gold_hit'] for s in scores), 'mean_evidence_recall':mean([s['gold_evidence_recall'] for s in scores]),
        'group_macro_recall':mean([mean(v) for v in groups.values()]), 'groups':len(groups),
        'mrr':mean([s['reciprocal_rank'] for s in scores]),
        'recall_at_k':{str(k):mean([sum(rank<=k for rank in s['gold_completion_ranks'].values())/s['gold_evidence_total_declared'] for s in scores]) for k in (1,3,5)},
        'mean_context_chars':mean([len(r['context']) for r in rows]),
        'mean_entry_count':mean([len(r['entries']) for r in rows]),
        'mean_source_documents':mean([len({x['file_path'] for e in r['entries'] for x in e['sources']}) for r in rows]),
        'truncated_cases':sum(any(e['truncated'] for e in r['entries']) for r in rows),
        'changed_context_cases':sum(r['context_changed'] for r in rows),
        'gain_ids':[r['id'] for r in rows if r['gained_gold']], 'loss_ids':[r['id'] for r in rows if r['lost_gold']],
        'added_negative_ids':[r['id'] for r in rows if r['added_hard_negative']],
        'new_negative_quote_count':sum(len(r['added_hard_negative']) for r in rows),
        'existing_negative_quote_count':sum(r['scores']['context']['hard_negative_found'] for r in rows)}


def main():
    originals,s,rawmaps,mappings = load_inputs()
    arms = {a:read_arm(a) for a in ARMS}
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases}
    store=RagIndexStore('E:/everythingfinder-db/rag'); bundle=store.active()
    metrics={a:{cohort:summarize([r for r in rows.values() if r['cohort']==cohort]) for cohort in ('formal','diagnostic')} for a,rows in arms.items()}
    impacts=[]; audits=[]; capacity=[]
    with store.connect_read_only(bundle.metadata_path) as conn:
        for r in originals:
            cid=r['id'];mapping=mappings[cid]
            added=set().union(*(set(rows[cid]['gained_gold']) | set(rows[cid]['added_hard_negative']) for rows in arms.values()))
            for m in mapping:
                if m.evidence_id not in added: continue
                path=Path(m.file)
                if not path.is_file():
                    options={store.load_chunk(conn,k).source.file_path for k in (m.declared_chunk_ids or m.chunk_ids)}
                    options={p for p in options if Path(p).name.casefold()==Path(m.file).name.casefold()}
                    if len(options)==1:path=Path(next(iter(options)))
                raw_original='';error=''
                try: raw_original=original_text(path,int(m.page))
                except Exception as exc:error=str(exc)
                original_present=bool(raw_original and canonical_content(m.quote) in canonical_content(raw_original))
                index_parts=[]
                for alt in m.alternatives:
                    if not alt.confirmed:continue
                    for part in alt.parts:
                        chunk=store.load_chunk(conn,part.chunk_id)
                        expected=canonical_content(m.quote)[part.quote_start:part.quote_end]
                        index_parts.append({'chunk_id':part.chunk_id,'present':expected in canonical_content(chunk.text),
                            'source_file':chunk.source.file_path,'location':chunk.source.location_text})
                rawlist=cases[cid].raw.get('gold_evidence' if m.kind=='positive' else 'hard_negatives',[])
                label=next((x for x in rawlist if canonical_content(x.get('quote',''))==canonical_content(m.quote)),{})
                audits.append({'id':cid,'question':r['question'],'evidence_id':m.evidence_id,'kind':m.kind,
                    'file':str(path),'page':m.page,'quote':m.quote,'gold_reason':label.get('reason',''),
                    'original_quote_present':original_present,'original_error':error,
                    'source_sha256':sha(path) if path.is_file() else None,'original_page_text':raw_original,
                    'index_parts':index_parts, 'index_quote_verified':bool(index_parts) and all(x['present'] for x in index_parts),
                    'added_in':[a for a,rows in arms.items() if m.evidence_id in rows[cid]['gained_gold']+rows[cid]['added_hard_negative']]})
            for a,rows in arms.items():
                rr=rows[cid]
                if not rr['context_changed']: continue
                impacts.append({'id':cid,'arm':a,'gained_gold':rr['gained_gold'],'lost_gold':rr['lost_gold'],
                    'added_hard_negative':rr['added_hard_negative'],'removed_hard_negative':rr['removed_hard_negative'],
                    'added_chunks':rr['added_chunks'],'removed_chunks':rr['removed_chunks'],
                    'context_chars_before':len(r['context']),'context_chars_after':len(rr['context']),
                    'truncated_before':[e['chunk_id'] for e in r['entries'] if e['truncated']],
                    'truncated_after':[e['chunk_id'] for e in rr['entries'] if e['truncated']]})
            if not r['scores']['context']['gold_evidence_total_declared'] or r['scores']['context']['gold_evidence_recall']==1:continue
            plan,candidates=reconstruct(r,store,conn)
            byid={x.chunk.stable_chunk_id:x for x in candidates}
            member_owner={member:x.chunk.stable_chunk_id for x in candidates for member in x.all_member_chunk_ids}
            policy=EvidencePolicy(min_vector_score=s['rag_min_vector_score'],min_rerank_score=s['rag_min_rerank_score'])
            possible=[];missing=[]
            for m in mapping:
                if m.kind!='positive':continue
                choices=[]
                for alt in m.alternatives:
                    if not alt.confirmed:continue
                    if all(p.chunk_id in member_owner for p in alt.parts):
                        owners=frozenset(member_owner[p.chunk_id] for p in alt.parts)
                        if all(not policy.rejection_reason(byid[x],check_constraints=True) for x in owners):choices.append(owners)
                if choices:possible.append(list(set(choices)))
                else:missing.append(m.evidence_id)
            # Exact union enumeration is diagnostic only; it never selects a runtime answer.
            sets={frozenset()}; capped=False
            for choices in possible:
                sets={left|right for left in sets for right in choices}
                if len(sets)>20000:capped=True;break
            best=None
            if not capped:
                for chosen in sets:
                    ctx=ContextBuilder().build([x for x in candidates if x.chunk.stable_chunk_id in chosen],max_chars=2**31-1)
                    if best is None or len(ctx.text)<best['chars']:
                        best={'chars':len(ctx.text),'chunks':list(chosen), 'quote_coverage':score_stage('context',list(ctx.entries),mapping)['gold_evidence_recall']}
            capacity.append({'id':cid,'question':r['question'],'missing_from_eligible_candidates':missing,
                'best_complete_existing_unit_cost':best,'enumeration_capped':capped,
                'fits_budget':bool(not missing and best and best['chars']<=s['rag_context_max_chars']),
                'category':'candidate_or_quality_gate_missing' if missing else 'unit_budget_insufficient' if best and best['chars']>s['rag_context_max_chars'] else 'selection_opportunity' if best else 'unresolved',
                'all_arm_final_scores':{a:rows[cid]['scores']['context']['gold_evidence_found'] for a,rows in arms.items()}})
    comparisons={}
    for a,b in [('baseline','B'),('A','AB'),('C','BC'),('AC','ABC')]:
        comparisons[a+'->'+b]={'context_changed':[cid for cid in arms[a] if arms[a][cid]['context']!=arms[b][cid]['context']],
            'entry_set_changed':[cid for cid in arms[a] if {e['chunk_id'] for e in arms[a][cid]['entries']}!={e['chunk_id'] for e in arms[b][cid]['entries']}],
            'gold_changed':[cid for cid in arms[a] if arms[a][cid]['scores']['context']['gold_completion_ranks'].keys()!=arms[b][cid]['scores']['context']['gold_completion_ranks'].keys()]}
    write(OUT/'metrics.json',metrics);write(OUT/'case_changes.json',impacts)
    write(OUT/'source_audit.json',audits);write(OUT/'capacity_diagnostic.json',capacity)
    write(OUT/'factor_interactions.json',comparisons)
    print('source checks',len(audits),'original missing',[(a['id'],a['evidence_id'],a['original_error']) for a in audits if not a['original_quote_present']])
    print('capacity',[(x['id'],x['category'],x['best_complete_existing_unit_cost']['chars'] if x['best_complete_existing_unit_cost'] else None) for x in capacity])


if __name__=='__main__':main()
