"""Frozen constrained candidates: isolate evidence policy/context from upstream.

Archived implementation and captured QueryPlan are used for V1; Gold is only
read by score_stage after the candidate selection has finished.
"""
import importlib.util
import sys
import argparse
from dataclasses import replace,fields
from unittest.mock import patch
from pathlib import Path
from local_full_text_search.rag import models as m
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence,load_chunk_rows
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from tools.validate_query_plan_adapter import FILES,OUT as OLD
from tools.rag_eval_common import ROOT,read,lines,write,sha

OUT=ROOT/'reports/rag_adapter_robustness_20261111'

def archived(name):
    spec=importlib.util.spec_from_file_location('archived_'+name,OUT/'before'/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
    return mod

def constraint(d):
    return m.QueryConstraint(**{**d,'kind':m.ConstraintKind(d['kind']),
        'polarity':m.ConstraintPolarity(d['polarity']),'strength':m.ConstraintStrength(d['strength'])})

def old_plan(row):
    p=next(e['query_plan'] for e in row['events'] if 'query_plan' in e)
    base=QueryInterpreter().interpret(row['question'])
    return replace(base,entities=tuple(p['entities']),constraints=tuple(map(constraint,p['constraints'])),
        entity_details=tuple(m.QueryEntity(**x) for x in p['entity_details']),
        answer_requirements=tuple(m.PlanSignal(**x) for x in p['answer_requirements']),
        lexical_terms=tuple(p['lexical_terms']),lexical_expression=p['lexical_expression'],
        analyzer_version=p['analyzer_version'],domain_config=p['domain_config'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='');args=ap.parse_args()
    suffix=('_'+args.tag) if args.tag else ''
    old_id=archived('identifier_evidence');old_policy=archived('evidence_policy');old_ctx=archived('context_builder')
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==read(OLD/'candidate_manifest.json')['bundle']
    cases={c.case_id:c for p in FILES for c in load_gold_dataset(p,reviewed_only=False).cases}
    results=[]
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunk_rows=load_chunk_rows(conn)
        for row in lines(OLD/'candidate.jsonl'):
            pold=old_plan(row);pnew=QueryInterpreter(enhanced=True,domain_config='chemiluminescence').interpret(row['question'])
            # Freeze constraints as well as candidate order, scores and text.
            pnew=replace(pnew,constraints=pold.constraints)
            def items():
                out=[]
                for r in row['candidate_stages'].get('constraints',[]):
                    chunk=store.load_chunk(conn,r['chunk_id']);assert chunk.text==r['text']
                    i=m.EvidenceChunk(chunk)
                    for f in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score',
                              'rerank_score','rerank_score_combined','constraint_score'):
                        setattr(i,f,r.get(f))
                    i.member_chunk_ids=tuple(r.get('member_chunk_ids',[]))
                    i.sources=tuple(store.load_chunk(conn,x['chunk_id']).source for x in r.get('sources',[]))
                    d=r.get('constraint_decision')
                    if d:
                        js=[]
                        for j in d['judgements']:
                            c=next(c for c in pold.constraints if c.kind.value==j['kind'] and c.value==j['value'] and c.source_text==j['source_text'])
                            js.append(m.ConstraintJudgement(c,m.ConstraintStatus(j['status']),j['reason'],j.get('matched_text','')))
                        valid={f.name for f in fields(m.CandidateDecision)}-{'judgements','stable_chunk_id'}
                        i.constraint_decision=m.CandidateDecision(r['chunk_id'],**{k:v for k,v in d.items() if k in valid},judgements=tuple(js))
                    out.append(i)
                return out
            r={'id':row['id'],'question':row['question'],'cohort':row['cohort'],'arms':{}}
            for arm in ('old_selection','new_selection','new_context','combined'):
                new_select=arm in ('new_selection','combined')
                new_context=arm in ('new_context','combined')
                candidates=items();plan=pnew if new_select else pold
                if new_select:
                    selected=EvidencePolicy().evaluate(m.RagQuery(row['question']),candidates,plan=plan)
                else:
                    with patch('local_full_text_search.rag.identifier_evidence.annotate',old_id.annotate),patch('local_full_text_search.rag.identifier_evidence.prioritize',old_id.prioritize):
                        selected=old_policy.EvidencePolicy().evaluate(m.RagQuery(row['question']),candidates,plan=plan)
                if new_context:
                    from local_full_text_search.rag.identifier_evidence import annotate
                    for i in selected.accepted: annotate(pnew,i)
                ctx=(ContextBuilder() if new_context else old_ctx.ContextBuilder()).build(selected.accepted)
                scores=score_stage('context',list(ctx.entries),map_case_evidence(cases[row['id']],chunk_rows))
                r['arms'][arm]={'context':ctx.text,'entries':ctx.entries,'scores':scores,
                                'rejected':selected.rejected,'insufficient':selected.insufficient}
            r['old_context_reproduced']=r['arms']['old_selection']['context']==row['context']
            results.append(r)
    write(OUT/f'selection_ablation{suffix}.json',results)
    write(OUT/f'selection_ablation{suffix}_protocol.json',{'input_hash':sha(OLD/'candidate.jsonl'),
        'bundle':bundle.bundle_id,'count':len(results),'old_context_reproduced':sum(r['old_context_reproduced'] for r in results),
        'limitations':'Frozen post-constraint candidates and scores. Not a fresh retrieval or end-to-end benchmark. New selection includes activation gating. New context uses source observations and complete row/header protection.'})
    print('ablation',len(results),'old context reproduced',sum(r['old_context_reproduced'] for r in results))

if __name__=='__main__':main()
