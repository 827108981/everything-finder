"""Four isolated API-free applicability experiments; Gold scoring is post-selection."""
from copy import deepcopy
from dataclasses import asdict
import json
from tools.validate_selection_factorial import ROOT, HISTORY, load_inputs, reconstruct, no_network, sha, write
from tools.analyze_selection_factorial import summarize
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.trace import candidate_snapshot

OUT=ROOT/'reports/rag_condition_binding_20261115'
ARMS={'baseline':(False,False),'extraction':(True,False),'judgement':(False,True),'combined':(True,True)}


def main():
    originals,settings,_,mappings=load_inputs()
    expected={r['id']:r for r in map(json.loads,(ROOT/'reports/rag_selection_a_adoption_20261113/production_A.jsonl').read_text(encoding='utf8').splitlines())}
    store=RagIndexStore('E:/everythingfinder-db/rag'); bundle=store.active()
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    assert bundle.bundle_id==protocol['bundle_id']
    policy=EvidencePolicy(max_per_document=None,min_vector_score=settings['rag_min_vector_score'],min_rerank_score=settings['rag_min_rerank_score'])
    allrows={}; equal=[]; no_retrieval_drift=[]
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        restored={r['id']:reconstruct(r,store,conn)[1] for r in originals}
        for arm,(extract,local) in ARMS.items():
            rows=[]
            for r in originals:
                plan=QueryInterpreter(bound_conditions=extract).interpret(r['question'])
                baseplan=QueryInterpreter().interpret(r['question'])
                for field in ('lexical_terms','lexical_expression','semantic_query','lexical_variants','semantic_variants'):
                    assert getattr(baseplan,field)==getattr(plan,field)
                # restore rerank ordering before applying this arm's matcher
                cs={x.chunk.stable_chunk_id:x for x in deepcopy(restored[r['id']])}
                items=[cs[x['chunk_id']] for x in r['candidate_stages']['rerank']]
                checked=ConstraintMatcher(local_conditions=local).evaluate(plan,items)
                d=policy.evaluate(RagQuery(r['question'],final_top_k=settings['rag_final_top_k']),checked,plan=plan)
                ctx=ContextBuilder().build(d.accepted,max_chars=settings['rag_context_max_chars'])
                selected={x.chunk.stable_chunk_id for x in d.accepted}; reasons=dict(d.rejected)
                snapshots=[candidate_snapshot(x,selected=x.chunk.stable_chunk_id in selected,reason=reasons.get(x.chunk.stable_chunk_id,'selected')) for x in checked]
                scores={**r['scores'],'constraints':score_stage('constraints',[candidate_snapshot(x) for x in checked],mappings[r['id']]),
                        'evidence':score_stage('evidence',snapshots,mappings[r['id']]),
                        'context':score_stage('context',list(ctx.entries),mappings[r['id']])}
                old=expected[r['id']]; b=old['scores']['context'];n=scores['context']
                row={k:r[k] for k in ('id','question','group_id','cohort','review_status')}
                row.update(arm=arm,context=ctx.text,entries=list(ctx.entries),excluded_entries=list(ctx.excluded_entries),
                    query_plan=asdict(plan),evidence=snapshots,scores=scores,selection_trace=[],
                    condition_candidates=[candidate_snapshot(x) for x in checked],
                    decision={'reason':d.reason,'insufficient':d.insufficient,'rejected':d.rejected},
                    gained_gold=sorted(set(n['gold_completion_ranks'])-set(b['gold_completion_ranks'])),
                    lost_gold=sorted(set(b['gold_completion_ranks'])-set(n['gold_completion_ranks'])),
                    added_hard_negative=sorted(set(n['hard_negative_ids'])-set(b['hard_negative_ids'])),
                    removed_hard_negative=sorted(set(b['hard_negative_ids'])-set(n['hard_negative_ids'])),
                    context_changed=ctx.text!=old['context'],answer_status='not_run_no_api',
                    added_chunks=sorted({e['chunk_id'] for e in ctx.entries}-{e['chunk_id'] for e in old['entries']}),
                    removed_chunks=sorted({e['chunk_id'] for e in old['entries']}-{e['chunk_id'] for e in ctx.entries}))
                rows.append(row)
                if arm=='baseline':
                    equal.append({'id':r['id'],'equal':ctx.text==old['context'] and list(ctx.entries)==old['entries'] and n==b})
            allrows[arm]=rows
            (OUT/(arm+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
            print(arm,len(rows),flush=True)
    metrics={a:{cohort:summarize([r for r in rows if r['cohort']==cohort]) for cohort in ('formal','diagnostic')} for a,rows in allrows.items()}
    write(OUT/'metrics.json',metrics)
    write(OUT/'case_changes.json',{a:[{k:r[k] for k in ('id','gained_gold','lost_gold','added_hard_negative','removed_hard_negative','added_chunks','removed_chunks','decision')} for r in rows if r['context_changed']] for a,rows in allrows.items()})
    unchanged=[p for p,h in protocol['frozen'].items() if ('fixtures' in p or p.endswith('settings.json')) and sha(p)!=h]
    write(OUT/'execution_audit.json',{'cases':len(originals),'observations':len(originals)*4,'api_calls':0,'network_attempts':attempts,
          'baseline_reproduction':equal,'baseline_passed':all(x['equal'] for x in equal),'gold_settings_drift':unchanged,
          'retrieval_expressions_unchanged':True,'upstream':'Frozen actual reranked candidates; no new retrieval',
          'candidate_feature_default':False})
    assert all(x['equal'] for x in equal) and not unchanged and not attempts


if __name__=='__main__':main()
