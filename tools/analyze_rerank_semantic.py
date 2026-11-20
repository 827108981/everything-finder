"""Post-selection evaluation. Gold and source labels are never runtime inputs."""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import statistics

from tools.validate_rerank_semantic import OUT, ROOT, read_jsonl, Token
from tools.validate_selection_factorial import (load_inputs, reconstruct, no_network, write, sha,
    RagIndexStore, RagQuery, score_stage)
from tools.analyze_selection_factorial import summarize
from tools.rag_trace_audit import original_text
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.reranking.evidence_ordering import EvidenceOrdering
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.content_identity import canonical_content


def mean(values):
    return statistics.mean(values) if values else None


def stage_metrics(rows, stage):
    positive = [r['scores'][stage] for r in rows if r['scores'][stage]['gold_evidence_total_declared']]
    return dict(positive_cases=len(positive), complete=sum(s['gold_evidence_recall'] == 1 for s in positive),
        recall=mean([s['gold_evidence_recall'] for s in positive]),
        mrr=mean([s['reciprocal_rank'] for s in positive]),
        recall_at_k={str(k):mean([sum(rank <= k for rank in s['gold_completion_ranks'].values())
            / s['gold_evidence_total_declared'] for s in positive]) for k in (1,3,5)})


def fresh():
    rows, settings, _, _ = load_inputs()
    dataset = load_gold_dataset(OUT/'new_holdout_pending.jsonl', reviewed_only=False)
    assert len(dataset.cases) == 12 and all(c.review_status == 'pending' for c in dataset.cases)
    write(OUT/'new_holdout_review.json', {'_说明':'新题仅诊断；引文文本已核对，未获业务人员审核，不计入正式指标。',
        'cases':[c.raw for c in dataset.cases]})
    review = load_gold_dataset(OUT/'new_holdout_review.json', reviewed_only=False)
    assert [c.raw for c in review.cases] == [c.raw for c in dataset.cases]
    store = RagIndexStore('E:/everythingfinder-db/rag'); bundle = store.active()
    native = {r['id']:r for r in read_jsonl(OUT/'native_retrieval.jsonl') if r['cohort'] == 'new_pending'}
    service = AnswerService(None,store,None,None,None,None,AppSettings.from_dict(settings))
    output = defaultdict(list); mapping_dump = {}; checks = []
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        chunks = load_chunk_rows(conn)
        for case in dataset.cases:
            mapping = map_case_evidence(case,chunks); mapping_dump[case.case_id] = [asdict(m) for m in mapping]
            assert all(m.declared_chunk_ids and any(a.confirmed for a in m.alternatives) for m in mapping)
            row = native[case.case_id]
            plan, candidates = reconstruct(row,store,conn)
            byid = {c.chunk.stable_chunk_id:c for c in candidates}
            upstream = {stage:score_stage(stage,cs,mapping) for stage,cs in row['candidate_stages'].items()}
            base = None
            for arm in ('baseline','B','R'):
                items = [deepcopy(byid[c['chunk_id']]) for c in row['candidate_stages']['rerank']]
                for c in items:c.constraint_decision=None;c.constraint_score=0
                if arm != 'baseline':items=EvidenceOrdering(rules=arm=='R').order(case.question,items,Token())
                rerank = [candidate_snapshot(c) for c in items]
                conditioned = ConstraintMatcher().evaluate(plan,items)
                conditions = [candidate_snapshot(c) for c in conditioned]
                decision = service.evidence_policy.evaluate(RagQuery(case.question,final_top_k=settings['rag_final_top_k']),conditioned,plan=plan)
                context = service.context_builder.build(decision.accepted,max_chars=settings['rag_context_max_chars'])
                ids = {c.chunk.stable_chunk_id for c in decision.accepted}; reasons=dict(decision.rejected)
                evidence=[{**candidate_snapshot(c),'selected':c.chunk.stable_chunk_id in ids,
                    'reason':'selected' if c.chunk.stable_chunk_id in ids else reasons.get(c.chunk.stable_chunk_id,'not_selected')}
                    for c in [*decision.accepted,*[x for x in conditioned if x.chunk.stable_chunk_id not in ids]]]
                scores={**upstream,**{s:score_stage(s,cs,mapping) for s,cs in (
                    ('rerank',rerank),('constraints',conditions),('evidence',evidence),('context',list(context.entries)))}}
                record={k:row[k] for k in ('id','question','group_id','cohort','review_status')}
                record.update(arm=arm,query_plan=asdict(plan),context=context.text,entries=list(context.entries),
                    excluded_entries=list(context.excluded_entries),scores=scores,evidence=evidence,
                    condition_candidates=conditions,rerank_candidates=rerank,
                    decision={'reason':decision.reason,'insufficient':decision.insufficient,'rejected':decision.rejected},
                    answer_status='not_run_no_api',selection_trace=[])
                if base is None:
                    base=record
                    checks.append(dict(id=case.case_id,context_equal=context.text==row['context'],entries_equal=list(context.entries)==row['entries']))
                now=scores['context']; before=base['scores']['context']
                record.update(gained_gold=sorted(set(now['gold_completion_ranks'])-set(before['gold_completion_ranks'])),
                    lost_gold=sorted(set(before['gold_completion_ranks'])-set(now['gold_completion_ranks'])),
                    added_hard_negative=[],removed_hard_negative=[],context_changed=context.text!=base['context'],
                    added_chunks=sorted({e['chunk_id'] for e in context.entries}-{e['chunk_id'] for e in base['entries']}),
                    removed_chunks=sorted({e['chunk_id'] for e in base['entries']}-{e['chunk_id'] for e in context.entries}))
                output[arm].append(record)
        for arm,values in output.items():
            (OUT/('fresh_'+arm+'.jsonl')).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values),encoding='utf8')
        write(OUT/'fresh_mappings.json',mapping_dump)
        write(OUT/'fresh_metrics.json',{a:{s:stage_metrics(values,s) for s in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context') if all(s in r['scores'] for r in values)} for a,values in output.items()})
        write(OUT/'fresh_validation.json',dict(cases=12,quotes=sum(len(c.gold_evidence) for c in dataset.cases),
            pending=12,json_jsonl_equal=True,loader_passed=True,all_quotes_mapped=True,
            baseline_reproduction=checks,api_calls=0,network_attempts=attempts,
            model_acceptance='Not advanced: semantic component gate failed; P/B/R diagnostic only'))
        assert not attempts and all(c['context_equal'] and c['entries_equal'] for c in checks)
    print('Fresh pending P/B/R metrics:',{a:stage_metrics(v,'context') for a,v in output.items()},flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--fresh',action='store_true');args=parser.parse_args()
    if args.fresh:fresh()
    originals, settings, rawmaps, mappings = load_inputs()
    arms={a:read_jsonl(OUT/(a+'.jsonl')) for a in ('baseline','B','R','M','RM') if (OUT/(a+'.jsonl')).exists()}
    byarm={a:{r['id']:r for r in rows} for a,rows in arms.items()}
    metrics={a:{cohort:summarize([r for r in rows if r['cohort']==cohort]) for cohort in ('formal','diagnostic')} for a,rows in arms.items()}
    stages={a:{cohort:{s:stage_metrics([r for r in rows if r['cohort']==cohort],s)
        for s in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context')}
        for cohort in ('formal','diagnostic')} for a,rows in arms.items()}
    write(OUT/'metrics.json',metrics);write(OUT/'stage_metrics.json',stages)
    impacts=[]; audits=[]; review_pack=[]
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    with store.connect_read_only(bundle.metadata_path) as conn:
        for row in originals:
            cid=row['id'];base=byarm['baseline'][cid]
            changed_ids=set().union(*(set(records[cid]['gained_gold'])|set(records[cid]['lost_gold'])|set(records[cid]['added_hard_negative']) for records in byarm.values()))
            for m in mappings[cid]:
                if m.evidence_id not in changed_ids:continue
                path=Path(m.file)
                if not path.is_file():
                    choices={store.load_chunk(conn,c).source.file_path for c in (m.declared_chunk_ids or m.chunk_ids)}
                    choices={p for p in choices if Path(p).name.casefold()==Path(m.file).name.casefold()}
                    if len(choices)==1:path=Path(next(iter(choices)))
                error=''; original=''
                try:original=original_text(path,int(m.page))
                except Exception as exc:error=repr(exc)
                parts=[]
                for alt in m.alternatives:
                    if not alt.confirmed:continue
                    for part in alt.parts:
                        chunk=store.load_chunk(conn,part.chunk_id)
                        expected=canonical_content(m.quote)[part.quote_start:part.quote_end]
                        parts.append(dict(chunk_id=part.chunk_id,source_file=chunk.source.file_path,
                            location=chunk.source.location_text,present=expected in canonical_content(chunk.text)))
                audits.append(dict(id=cid,evidence_id=m.evidence_id,kind=m.kind,file=str(path),page=m.page,quote=m.quote,
                    original_quote_present=bool(original and canonical_content(m.quote) in canonical_content(original)),
                    original_error=error,original_page_text=original,index_parts=parts,
                    index_quote_verified=bool(parts) and all(p['present'] for p in parts),
                    source_sha256=sha(path) if path.is_file() else None))
            for arm,records in byarm.items():
                r=records[cid]
                if arm=='baseline' or not r['context_changed']:continue
                impact={k:r[k] for k in ('id','question','gained_gold','lost_gold','added_hard_negative','added_chunks','removed_chunks')}
                impact.update(arm=arm,entry_set_changed=bool(r['added_chunks'] or r['removed_chunks']),
                    chars_before=len(base['context']),chars_after=len(r['context']))
                impacts.append(impact)
                if impact['entry_set_changed']:
                    review_pack.append(dict(**impact,before_entries=base['entries'],after_entries=r['entries']))
    write(OUT/'case_changes.json',impacts);write(OUT/'source_audit.json',audits)
    (OUT/'context_delta_review.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in review_pack),encoding='utf8')
    components=json.loads((OUT/'component_outputs.json').read_text(encoding='utf8')) if (OUT/'component_outputs.json').exists() else []
    grouped=defaultdict(list)
    for r in components:grouped[r['id']].append(r)
    write(OUT/'component_summary.json',dict(pairs=len(grouped),executions=len(components),
        outputs=Counter((r['judgement']['support']+':'+r['judgement']['applicability']+':'+str(r['judgement']['valid'])) for r in components),
        stable_pairs=sum(len(v)==2 and all(v[0]['judgement'][k]==v[1]['judgement'][k] for k in ('support','applicability','valid','spans')) for v in grouped.values()),
        wrong_identifier_pair='component-08',wrong_identifier_repeated=True,adoption_gate='failed'))
    print('Context metrics:',{a:(m['formal']['complete'],m['diagnostic']['complete'],m['formal']['loss_ids'],m['diagnostic']['loss_ids']) for a,m in metrics.items()},flush=True)
    print('Changed-source audit:',len(audits),'missing original:',[(a['id'],a['evidence_id']) for a in audits if not a['original_quote_present']],flush=True)


if __name__=='__main__':main()
