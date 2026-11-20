"""Fixed-input reranker experiment. All inference is local; Gold is scoring-only."""
from __future__ import annotations
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from tools.validate_selection_factorial import (
    ROOT, RagIndexStore, RagQuery, load_inputs, reconstruct, no_network, score_stage, sha, write,
)
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.reranking.local_semantic_judge import LocalSemanticJudge
from local_full_text_search.rag.reranking.evidence_ordering import EvidenceOrdering

OUT = ROOT/'reports/rag_rerank_semantic_20261116'


class Token:
    def throw_if_cancelled(self): pass


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf8').splitlines()]


def route(row):
    old = row['candidate_stages']['rerank']
    base = sorted(old, key=lambda x: (-float(x['rerank_score'] or 0), -x['fused_score'], x['chunk_id']))
    return {x['chunk_id'] for x in old[:8]} ^ {x['chunk_id'] for x in base[:8]}


def component_inputs(rows):
    lookup = {r['id']:r for r in rows}
    # Source-reviewed diagnostic labels. These labels never enter the judge.
    specs = [
        ('rag-biz-004', 0, 'related', 'unknown', 'Replacing the board does not answer which positions are sampled'),
        ('rag-biz-004', 8, 'direct', 'compatible', 'Three board-to-position assignments explicitly given'),
        ('rag-biz-004', 18, 'related', 'conflict', 'Explicit CL-1000i single-board rule is not CL-8000i'),
        ('rag-biz-006', 17, 'direct', 'unknown', 'The complete intake rule; applicability not inferred from filename'),
        ('rag-biz-021', 0, 'partial', 'compatible', 'LVC meaning only, no decision branches'),
        ('rag-biz-021', 2, 'partial', 'compatible', 'Method branches retain value despite unrelated merged material'),
        ('rag-biz-021', 7, 'partial', 'unknown', 'Alarm table supplies meaning, not complete decision method'),
        ('rag-biz-023', 0, 'related', 'conflict', 'LVC treatment cannot be used for CSDB'),
        ('rag-biz-023', 1, 'related', 'conflict', 'DEV treatment cannot be used for CSDB'),
        ('rag-biz-023', 10, 'direct', 'compatible', 'Complete CSDB row; other alarm rows do not invalidate it'),
        ('rag-biz-020', 1, 'direct', 'compatible', 'Single-item DEV causes and actions'),
        ('evidence-flow-independent-07', 1, 'direct', 'compatible', 'Seal lifetime and part number are explicitly linked'),
    ]
    return [dict(id=f'component-{i:02}', case_id=cid, question=lookup[cid]['question'],
                 candidate=lookup[cid]['candidate_stages']['rerank'][index], expected_support=support,
                 expected_applicability=scope, rationale=why)
            for i,(cid,index,support,scope,why) in enumerate(specs,1)]


def generate(args, rows, protocol):
    components = component_inputs(rows)
    target = OUT/'component_inputs.json'
    if target.exists(): assert json.loads(target.read_text(encoding='utf8')) == components
    else: write(target, components)
    model = protocol['model']
    with no_network() as attempts:
        judge = LocalSemanticJudge(Path(model['path']), model_sha256=model['sha256'],
                                   cache_dir=OUT/'judgement_cache')
        if args.phase in {'components','model'}:
            results=[]
            for pair in components:
                for rep in (1,2):
                    value = judge.judge(pair['question'], pair['candidate']['text'], use_cache=False)
                    results.append(dict(id=pair['id'],case_id=pair['case_id'], repetition=rep,
                                        judgement=asdict(value)))
                    write(OUT/'component_outputs.json', results)
                    print('component', pair['id'], rep, value.support, value.applicability,
                          value.valid, value.error, flush=True)
        if args.phase == 'model':
            records=[]
            for row in rows:
                selected=route(row)
                for candidate in row['candidate_stages']['rerank']:
                    if candidate['chunk_id'] not in selected: continue
                    value=judge.judge(row['question'],candidate['text'])
                    records.append(dict(case_id=row['id'],chunk_id=candidate['chunk_id'],judgement=asdict(value)))
                    with (OUT/'model_progress.jsonl').open('a',encoding='utf8') as stream:
                        stream.write(json.dumps(records[-1],ensure_ascii=False)+'\n')
                    print('model',len(records),row['id'],value.support,value.applicability,value.valid,flush=True)
            write(OUT/'model_outputs.json',records)
        write(OUT/'model_network_audit.json',{'attempts':attempts,'api_calls':0})
        assert not attempts


class SavedJudge:
    def __init__(self, records, case_id):
        self.values={r['chunk_id']:r['judgement'] for r in records if r['case_id']==case_id}
    def judge(self,question,text):
        from local_full_text_search.rag.reranking.local_semantic_judge import SemanticJudgement
        # Lookup by exact body as well as current case; caller provides no Gold.
        cid=next(cid for cid in self.body_to_ids[text] if cid in self.values)
        value=deepcopy(self.values[cid]);value['spans']=tuple(value.get('spans',()))
        return SemanticJudgement(**value)


def replay(rows,settings,mappings,protocol,arms):
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==protocol['bundle']
    old={r['id']:r for r in read_jsonl(ROOT/protocol['baseline_context'])}
    service=AnswerService(None,store,None,None,None,None,AppSettings.from_dict(settings))
    generated=json.loads((OUT/'model_outputs.json').read_text(encoding='utf8')) if (OUT/'model_outputs.json').exists() else []
    records={a:[] for a in arms};checks=[]
    with no_network() as attempts,store.connect_read_only(bundle.metadata_path) as conn:
        for row in rows:
            plan,conditioned=reconstruct(row,store,conn)
            lookup={x.chunk.stable_chunk_id:x for x in conditioned}
            for arm in arms:
                candidates=[deepcopy(lookup[c['chunk_id']]) for c in row['candidate_stages']['rerank']]
                for c in candidates: c.constraint_decision=None;c.constraint_score=0
                if arm!='baseline':
                    judge=None
                    if arm in {'M','RM'}:
                        judge=SavedJudge(generated,row['id'])
                        judge.body_to_ids={}
                        for c in candidates:
                            judge.body_to_ids.setdefault(c.chunk.text,[]).append(c.chunk.stable_chunk_id)
                        assert route(row) <= judge.values.keys()
                    candidates=EvidenceOrdering(rules=arm in {'R','RM'},judge=judge,
                                                semantic_ids=route(row)).order(row['question'],candidates,Token())
                stage_rerank=[candidate_snapshot(c) for c in candidates]
                candidates=ConstraintMatcher().evaluate(plan,candidates)
                stage_condition=[candidate_snapshot(c) for c in candidates]
                q=RagQuery(row['question'],final_top_k=settings['rag_final_top_k'])
                decision=service.evidence_policy.evaluate(q,candidates,plan=plan)
                ctx=service.context_builder.build(decision.accepted,max_chars=settings['rag_context_max_chars'])
                selected={c.chunk.stable_chunk_id for c in decision.accepted}; reasons=dict(decision.rejected)
                evidence=[{**candidate_snapshot(c),'selected':c.chunk.stable_chunk_id in selected,
                    'reason':'selected' if c.chunk.stable_chunk_id in selected else reasons.get(c.chunk.stable_chunk_id,'not_selected')}
                    for c in [*decision.accepted,*[x for x in candidates if x.chunk.stable_chunk_id not in selected]]]
                scores={**row['scores'],**{s:score_stage(s,cs,mappings[row['id']]) for s,cs in (
                    ('rerank',stage_rerank),('constraints',stage_condition),('evidence',evidence),('context',list(ctx.entries)))}}
                before=old[row['id']];now=scores['context'];base=before['scores']['context']
                if arm=='baseline':
                    checks.append(dict(id=row['id'],context_equal=ctx.text==before['context'],
                        entries_equal=list(ctx.entries)==before['entries'],score_equal=now==base))
                record={k:row[k] for k in ('id','question','group_id','cohort','review_status')}
                record.update(arm=arm,query_plan=asdict(plan),context=ctx.text,entries=list(ctx.entries),
                    excluded_entries=list(ctx.excluded_entries),evidence=evidence,
                    condition_candidates=stage_condition,rerank_candidates=stage_rerank,
                    scores=scores,answer_status='not_run_no_api',selection_trace=[{
                        'rank':rank,'chunk_id':c['chunk_id'],'raw_ce_score':c.get('rerank_score'),
                        'combined_score':c.get('rerank_score_combined'),'fused_rank':c.get('fused_rank'),
                        'answer_support_score':c.get('answer_support_score'),
                        'diagnostics':c.get('rerank_diagnostics',{})} for rank,c in enumerate(stage_rerank,1)],
                    decision={'reason':decision.reason,'insufficient':decision.insufficient,'rejected':decision.rejected},
                    gained_gold=sorted(set(now['gold_completion_ranks'])-set(base['gold_completion_ranks'])),
                    lost_gold=sorted(set(base['gold_completion_ranks'])-set(now['gold_completion_ranks'])),
                    added_hard_negative=sorted(set(now['hard_negative_ids'])-set(base['hard_negative_ids'])),
                    removed_hard_negative=sorted(set(base['hard_negative_ids'])-set(now['hard_negative_ids'])),
                    context_changed=ctx.text!=before['context'],
                    added_chunks=sorted({e['chunk_id'] for e in ctx.entries}-{e['chunk_id'] for e in before['entries']}),
                    removed_chunks=sorted({e['chunk_id'] for e in before['entries']}-{e['chunk_id'] for e in ctx.entries}))
                records[arm].append(record)
        for arm,values in records.items():
            (OUT/(arm+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in values),encoding='utf8')
            print('replayed',arm,len(values),flush=True)
        if checks:
            passed=all(all(v for k,v in c.items() if k!='id') for c in checks)
            write(OUT/'baseline_reproduction.json',dict(passed=passed,cases=len(checks),records=checks))
            assert passed,'baseline mismatch'
        drift=[p for p,h in protocol['protected'].items() if sha(p)!=h]
        write(OUT/'execution_audit.json',dict(api_calls=0,network_attempts=attempts,protected_drift=drift,
            bundle_id=bundle.bundle_id,frozen_input_sha=sha(ROOT/protocol['frozen_candidates']),
            arms=[a for a in ['baseline','B','R','M','RM'] if (OUT/(a+'.jsonl')).exists()],
            cases_per_arm=len(rows),method='fixed 20 candidate replay; real condition, evidence and Context implementations'))
        assert not drift and not attempts


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['components','model','rules','replay'],required=True)
    args=parser.parse_args();rows,settings,_,mappings=load_inputs()
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    assert sha(ROOT/protocol['frozen_candidates'])==protocol['frozen_candidates_sha256']
    if args.phase in {'components','model'}: generate(args,rows,protocol)
    else: replay(rows,settings,mappings,protocol,['baseline','B','R'] if args.phase=='rules' else ['baseline','B','R','M','RM'])

if __name__=='__main__': main()
