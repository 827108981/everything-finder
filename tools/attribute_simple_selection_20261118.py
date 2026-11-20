"""Sequential controls separate condition removal, TopK, score cutoffs and budget."""
from copy import deepcopy
from dataclasses import asdict
import json
from tools.validate_simple_selection_20261118 import OUT, HISTORY, exp, check_frozen
from tools.validate_selection_factorial import reconstruct, write, no_network
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


def main():
    check_frozen()
    store, bundle, settings = exp.inputs()
    cases = exp.load_cases(True)
    old = exp.read_rows(HISTORY/'baseline_native.jsonl')
    groups = {a:{r['id']:r for r in exp.read_rows(OUT/(a+'_native.jsonl'))}
              for a in ('A','simple_legacy_budget','simple')}
    controls = {'remove_conditions_only':{}, 'remove_conditions_and_top8':{}}
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        chunks = load_chunk_rows(conn)
        maps = {c.case_id:map_case_evidence(c,chunks) for c in cases}
        for row in old:
            plan, xs = reconstruct(row,store,conn)
            byid = {x.chunk.stable_chunk_id:x for x in xs}
            xs = [byid[x['chunk_id']] for x in row['candidate_stages']['rerank']]
            for x in xs:
                x.constraint_decision = None
                x.constraint_score = None
            for name, limit in [('remove_conditions_only',8), ('remove_conditions_and_top8',len(xs))]:
                policy = EvidencePolicy(min_vector_score=settings.rag_min_vector_score,
                    min_rerank_score=settings.rag_min_rerank_score,max_per_document=None)
                decision = policy.evaluate(RagQuery(row['question'],final_top_k=limit),xs,plan=None)
                ctx = ContextBuilder().build(decision.accepted,max_chars=settings.rag_context_max_chars)
                controls[name][row['id']] = dict(id=row['id'],cohort=row['cohort'],
                    scores={'context':score_stage('context',list(ctx.entries),maps[row['id']])},
                    entries=list(ctx.entries), context=ctx.text, exclusions=decision.rejected)
    chain = [('A',groups['A']), *controls.items(), ('remove_conditions_top8_thresholds',groups['simple_legacy_budget']),
             ('complete_chunk_budget',groups['simple'])]
    result = dict(api_calls=0,network_attempts=attempts,steps=[],controls=controls)
    for (before_name,before),(after_name,after) in zip(chain,chain[1:]):
        changes=[]
        for cid,r in after.items():
            b=before[cid];bs=b['scores']['context'];ss=r['scores']['context']
            gain=set(ss['gold_completion_ranks'])-set(bs['gold_completion_ranks'])
            loss=set(bs['gold_completion_ranks'])-set(ss['gold_completion_ranks'])
            neg=set(ss['hard_negative_ids'])-set(bs['hard_negative_ids'])
            removed=set(bs['hard_negative_ids'])-set(ss['hard_negative_ids'])
            if gain or loss or neg or removed or r['context']!=b['context']:
                changes.append(dict(id=cid,gain=sorted(gain),loss=sorted(loss),new_negative=sorted(neg),removed_negative=sorted(removed),
                    entries_before=len(b['entries']),entries_after=len(r['entries'])))
        result['steps'].append(dict(before=before_name,after=after_name,changed_cases=len(changes),
            gains=[c['id'] for c in changes if c['gain']],losses=[c['id'] for c in changes if c['loss']],
            new_negative_quotes=sum(len(c['new_negative']) for c in changes),
            new_negative_cases=[c['id'] for c in changes if c['new_negative']],changes=changes))
    write(OUT/'factor_attribution.json',result)
    print(json.dumps([{k:v for k,v in step.items() if k!='changes'} for step in result['steps']],ensure_ascii=False),flush=True)
    check_frozen()


if __name__=='__main__':main()
