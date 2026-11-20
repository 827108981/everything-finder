"""Independent, frozen-input evidence experiments. No production or Gold writes.

S1 is freshly executed: stable canonical dedup plus bounded literal-identifier
rescue into free slots only. S2/S3 replay historical A/B independently against
the same baseline. No inference is rerun and no historical answer is regraded.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, CandidateDecision, ConstraintJudgement, ConstraintStatus, RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.trace import candidate_snapshot

ROOT = Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def anchors(text):
    # Whole standalone alphabetic identifiers, never inferred diagnoses.
    return set(re.findall(r'(?<![A-Za-z0-9_.-])[A-Za-z]{2,}(?![A-Za-z0-9_.-])', text.upper()))

def conservative_select(question, eligible, original, limit=8):
    """No scores/Gold/case IDs. Never evict a nonidentical original body.

    Exact-body copies only collapse within the same document. A cap exception
    needs an explicit query identifier absent from selected text, plus all hard
    conditions directly MATCH. It can use a free slot but cannot displace facts.
    This tests a deliberately limited sufficient rule, not semantic coverage.
    """
    selected=[]; seen=set(); trace=[]
    for item in original:
        key=(item.chunk.document_key, canonical_content(item.chunk.text))
        if key in seen:
            trace.append({'action':'exact_duplicate','id':item.chunk.stable_chunk_id})
        else:
            seen.add(key); selected.append(item)
    remaining=anchors(question)-set().union(*(anchors(i.chunk.text) for i in selected))
    for item in eligible:
        if len(selected)>=limit or not remaining:
            break
        if item in selected:
            continue
        decision=item.constraint_decision
        if decision is None or any(j.constraint.is_hard and j.status is not ConstraintStatus.MATCH for j in decision.judgements):
            continue
        matches=remaining & anchors(item.chunk.text)
        if not matches:
            continue
        selected.append(item)
        trace.append({'action':'free_slot_identifier_rescue','id':item.chunk.stable_chunk_id,'identifiers':sorted(matches)})
        remaining-=matches
    return selected, trace

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'reports/rag_retrieval_ab_results_20261103'
    old_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in old_protocol['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active()
    assert bundle.bundle_id==old_protocol['bundle_id']
    def dump(name,obj):
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    paths=[source/'resolved_cases.jsonl',source/'continuation_links.json',source/'query_plans.json',Path(__file__),bundle.metadata_path]
    hashes={str(p):sha(p) for p in paths}
    protocol={'started':str(datetime.now()),'bundle_id':bundle.bundle_id,'hashes':hashes,
              'production_and_gold_hashes':old_protocol['hashes'],
              'S1':'Fresh final-selection replay; exact same-document dedup, explicit alphabetic query-identifier rescue only into free slots; all hard conditions MATCH; max 8 / 12000 chars. No semantic coverage claim.',
              'S2':'Historical A alone vs baseline: narrow title-to-next-causes-page links and scoring context, original candidates retained. No table-row experiment, no full chapter parser.',
              'S3':'Historical B alone vs baseline: frozen question-only subqueries, capped extra retrieval and per-query scoring. Does not stack S1 or S2.',
              'inference':'No new retrieval/model/answer inference; S2/S3 reuse frozen computations after verifying production/settings/Gold hashes. New evidence replay only.',
              'gate':'No lost declared evidence or newly marked negatives. Exact-quote checks and equivalent-source audits supplement strict source matching; a flagged table page is not automatically a wrong answer.',
              'budget':{'chunks':settings.rag_final_top_k,'chars':settings.rag_context_max_chars}}
    dump('protocol.json',protocol)
    rows=[json.loads(l) for l in (source/'resolved_cases.jsonl').open(encoding='utf-8')]
    records={(r['id'],r['arm']):r for r in rows}
    assert len(records)==236
    conn=store.connect_read_only(bundle.metadata_path)
    chunk_rows=load_chunk_rows(conn)
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    maps={cid:map_case_evidence(c,chunk_rows) for cid,c in cases.items() if c.review_status!='disputed'}
    cache={}
    def chunk(cid):
        if cid not in cache:cache[cid]=store.load_chunk(conn,cid)
        assert cache[cid] is not None
        return cache[cid]
    def restore(s,plan):
        ch=chunk(s['chunk_id']);assert ch.text==s['text']
        names=('lexical_rank','lexical_score','lexical_score_kind','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined','answer_support_score','constraint_score')
        item=EvidenceChunk(chunk=ch,**{k:s[k] for k in names if k in s})
        item.member_chunk_ids=tuple(s.get('member_chunk_ids',[]))
        item.sources=tuple(chunk(x['chunk_id']).source for x in s.get('sources',[]))
        d=s.get('constraint_decision')
        if d:
            js=[]
            for j in d['judgements']:
                q=next(c for c in plan.constraints if c.kind.value==j['kind'] and c.value==j['value'] and c.source_text==j['source_text'] and c.strength.value==j['strength'] and c.polarity.value==j['polarity'])
                js.append(ConstraintJudgement(q,ConstraintStatus(j['status']),j['reason'],j['matched_text']))
            item.constraint_decision=CandidateDecision(ch.stable_chunk_id,d['accepted'],d['reason'],tuple(js),**{k:d[k] for k in ('required_match_count','required_unknown_count','required_conflict_count','soft_match_count','soft_conflict_count','support_status')},support_spans=tuple(d['support_spans']))
        return item
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=settings.rag_max_chunks_per_document)
    builder=ContextBuilder();results=[];reproductions=[]
    audits=json.loads((ROOT/'reports/rag_context_ceiling_20261104/source_audit.json').read_text(encoding='utf-8'))
    assert all(a['original_quote_exact_canonical'] and a['index_quote_exact_canonical'] and sha(a['file'])==a['original_sha256'] for a in audits)
    for base in (r for r in rows if r['arm']=='baseline'):
        cid=base['id'];plan=QueryInterpreter().interpret(base['question']);query=RagQuery(text=base['question'],final_top_k=settings.rag_final_top_k)
        for step,arm in (('S1','baseline'),('S2','A'),('S3','B')):
            frozen=records[cid,arm]
            candidates=[restore(s,plan) for s in frozen['candidate_stages']['constraints']]
            native=policy.evaluate(query,candidates,plan=plan)
            native_context=builder.build(native.accepted,max_chars=settings.rag_context_max_chars)
            assert native_context.text==frozen['context'],(cid,arm)
            reproductions.append({'id':cid,'arm':arm,'exact':True})
            trace=[]
            if step=='S1':
                eligible=[i for i in candidates if not policy.rejection_reason(i,check_constraints=True)]
                selected,trace=conservative_select(base['question'],eligible,native.accepted,settings.rag_final_top_k)
                check=EvidencePolicy(min_vector_score=policy.min_vector_score,min_rerank_score=policy.min_rerank_score,max_per_document=settings.rag_final_top_k).evaluate(query,selected,plan=plan)
                context=builder.build(check.accepted,max_chars=settings.rag_context_max_chars)
                by_id={i.chunk.stable_chunk_id:i for i in check.accepted}
                stage=[{**candidate_snapshot(by_id[e['chunk_id']],selected=True),'included_text':e['included_text'],'truncated':e['truncated']} for e in context.entries]
                score=score_stage('context',stage,maps[cid])
            else:
                context=native_context;score=frozen['stages']['context']
            before=base['stages']['context']
            quote_checks=[{'evidence':a['evidence'],'quote':a['quote'],'before':canonical_content(a['quote']) in canonical_content(base['context']), 'after':canonical_content(a['quote']) in canonical_content(context.text)} for a in audits if a['id']==cid]
            cohort=base['cohort']
            if cohort=='formal48':cohort='original17' if int(cid.rsplit('-',1)[1])<=18 else 'added31'
            result={'id':cid,'step':step,'cohort':cohort,'question':base['question'], 'review_status':base['review_status'],
                    'before':before,'after':score,'changed':context.text!=base['context'],
                    'lost':sorted(set(before['gold_completion_ranks'])-set(score['gold_completion_ranks'])),
                    'gained':sorted(set(score['gold_completion_ranks'])-set(before['gold_completion_ranks'])),
                    'new_negative_ids':sorted(set(score['hard_negative_ids'])-set(before['hard_negative_ids'])),
                    'quote_checks':quote_checks,'trace':trace,'context':context.text,'entries':list(context.entries),
                    'before_ids':[e['chunk_id'] for e in base['context_entries']],
                    'baseline_bodies_retained':all(canonical_content(e['included_text']) in canonical_content(context.text) for e in base['context_entries']),
                    'chars':len(context.text),'chunks':len(context.entries)}
            assert result['chunks']<=settings.rag_final_top_k and result['chars']<=settings.rag_context_max_chars
            results.append(result)
    conn.close()
    summary={}
    for step in ('S1','S2','S3'):
        rs=[r for r in results if r['step']==step]
        cohorts={}
        for cohort in ('original17','added31','regression10_pending','pending024'):
            subset=[r for r in rs if r['cohort']==cohort]
            pos=[r for r in subset if r['before']['gold_evidence_recall'] is not None]
            cohorts[cohort]={'count':len(subset),'positive_count':len(pos),'changed':sum(r['changed'] for r in subset),
                             'before_hit':sum(r['before']['gold_hit'] for r in pos),'after_hit':sum(r['after']['gold_hit'] for r in pos),
                             'before_coverage':sum(r['before']['gold_evidence_recall'] for r in pos)/len(pos) if pos else None,
                             'after_coverage':sum(r['after']['gold_evidence_recall'] for r in pos)/len(pos) if pos else None}
        summary[step]={'cohorts':cohorts,'changed':sum(r['changed'] for r in rs),'gains':[r['id'] for r in rs if r['gained']],
                       'losses':[r['id'] for r in rs if r['lost']],'new_negative_cases':[r['id'] for r in rs if r['new_negative_ids']],
                       'quote_gains':[(r['id'],q['evidence']) for r in rs for q in r['quote_checks'] if q['after'] and not q['before']],
                       'quote_losses':[(r['id'],q['evidence']) for r in rs for q in r['quote_checks'] if q['before'] and not q['after']],
                       'gate':not any(r['lost'] or r['new_negative_ids'] for r in rs)}
    assert all(sha(p)==h for p,h in {**hashes,**old_protocol['hashes']}.items())
    dump('summary.json',summary);dump('reproduction.json',reproductions)
    (out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    dump('completion.json',{'finished':str(datetime.now()),'rows':len(results),'exact_reproductions':len(reproductions),'hashes_unchanged':True,'source_quotes_reused_verified':len(audits),'new_model_calls':0,'no_production_changes':True})
    cards=[]
    for r in results:
        cards.append('<details><summary>'+html.escape(f"{r['step']} {r['id']} changed={r['changed']} lost={r['lost']} gained={r['gained']} negatives={r['new_negative_ids']}")+'</summary><pre>'+html.escape(json.dumps(r,ensure_ascii=False,indent=2))+'</pre></details>')
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>三步独立证据回放</title><style>body{font:16px sans-serif;max-width:1200px;margin:32px auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:12px;margin:8px}</style><h1>三步独立证据回放</h1><p>S1新执行选材；S2/S3复用已冻结A/B结果并逐字复现。没有新跑回答，不能称为本轮回答通过率。</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
