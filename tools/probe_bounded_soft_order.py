"""Isolate soft applicability ordering on frozen scored candidates, no LLM.

The candidate algorithm sees only QueryPlan/candidates. Frozen Gold is used
after Context construction by the unchanged source-quote scorer.
"""
from dataclasses import asdict
import json
from pathlib import Path


def bounded_order(plan, items):
    if not plan.adapter_active:return tuple(items)
    def key(item):
        decision=item.constraint_decision
        base=item.rerank_score_combined if item.rerank_score_combined is not None else item.rerank_score or 0.
        soft=(decision.soft_match_count-decision.soft_conflict_count) if decision else 0
        bonus=max(-.02,min(.02,.01*soft))
        return (bool(decision and not decision.accepted),bool(decision and decision.required_unknown_count),-(base+bonus))
    return tuple(sorted(items,key=key))


def main():
    from local_full_text_search.core.task_manager import CancelToken
    from local_full_text_search.rag.models import EvidenceChunk,RagQuery
    from local_full_text_search.rag.index_store import RagIndexStore
    from local_full_text_search.rag.query_interpreter import QueryInterpreter
    from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
    from local_full_text_search.rag.evidence_policy import EvidencePolicy
    from local_full_text_search.rag.context_builder import ContextBuilder
    from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
    from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
    from local_full_text_search.rag.evaluation.stage_scoring import score_stage
    root=Path(__file__).resolve().parents[1];out=root/'reports/rag_support_graph_20261113'
    manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(root/d['path'],reviewed_only=False).cases}
    rows=[json.loads(l) for l in (out/'B.jsonl').read_text(encoding='utf8').splitlines()]
    s=json.loads((out/'B_manifest.json').read_text(encoding='utf8'))['settings']
    interpreter=QueryInterpreter(enhanced=True,domain_config='chemiluminescence')
    matcher=ConstraintMatcher(domain_adapter=interpreter.domain_adapter)
    policy=EvidencePolicy(min_vector_score=s['rag_min_vector_score'],min_rerank_score=s['rag_min_rerank_score'],max_per_document=s['rag_max_chunks_per_document'])
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active();records=[]
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunk_rows=load_chunk_rows(conn)
        for row in rows:
            plan=interpreter.interpret(row['question']);candidates=[]
            for c in row['candidate_stages'].get('rerank',[]):
                chunk=store.load_chunk(conn,c['chunk_id']);members=tuple(c.get('member_chunk_ids') or [chunk.stable_chunk_id])
                item=EvidenceChunk(chunk=chunk,member_chunk_ids=members,sources=tuple(store.load_chunk(conn,i).source for i in members))
                for key in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined','answer_support_score'):
                    if c.get(key) is not None:setattr(item,key,c[key])
                item.query_plan_diagnostics=dict(c.get('query_plan_diagnostics',{}));candidates.append(item)
            ordered=matcher.evaluate(plan,candidates)
            query=RagQuery(row['question'],final_top_k=s['rag_final_top_k'])
            baseline=policy.evaluate(query,ordered,plan=plan)
            baseline_context=ContextBuilder().build(baseline.accepted,max_chars=s['rag_context_max_chars'])
            changed=policy.evaluate(query,bounded_order(plan,ordered),plan=plan)
            context=ContextBuilder().build(changed.accepted,max_chars=s['rag_context_max_chars'])
            mapping=map_case_evidence(cases[row['id']],chunk_rows)
            score=score_stage('context',list(context.entries),mapping)
            records.append({'id':row['id'],'adapter_active':plan.adapter_active,
                'baseline_reproduced':baseline_context.text==row['context'],
                'before':row['scores']['context']['gold_evidence_found'],'after':score['gold_evidence_found'],
                'context_changed':context.text!=row['context'],'context':context.text,
                'entries':list(context.entries),'score':score,'excluded':changed.rejected})
    summary={'count':len(records),'predeclared_rule':'Relevance score plus soft bonus bounded to +/-0.02; no altered retrieval or hard-conflict gate',
        'baseline_not_reproduced':[r['id'] for r in records if not r['baseline_reproduced']],
        'gains':[r['id'] for r in records if r['after']>r['before']],
        'losses':[r['id'] for r in records if r['after']<r['before']],
        'changed':[r['id'] for r in records if r['context_changed']]}
    (out/'bounded_soft_order_probe.json').write_text(json.dumps({'summary':summary,'records':records},ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
