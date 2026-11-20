"""Audit exact included text, frozen upstream and causes of new negative quotes."""
from collections import Counter
import json
from pathlib import Path
from tools.validate_simple_selection_20261118 import OUT, ARMS, exp, check_frozen, STAGES
from tools.validate_selection_factorial import write
from local_full_text_search.rag.evaluation.reporting.matching import match_quote, MatchConfig
from local_full_text_search.rag.content_identity import compare_content
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


def main():
    check_frozen()
    store, bundle, settings = exp.inputs()
    cases = {c.case_id:c for c in exp.load_cases(True)}
    arms = {a:{r['id']:r for r in exp.read_rows(OUT/(a+'_native.jsonl'))} for a in ARMS}
    base, new = arms['baseline'], arms['simple']
    issues, negatives, losses, no_gold, gains = [], [], [], [], []
    entry_count = duplicate_sources = layout_equivalent_sources = 0
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunkrows = load_chunk_rows(conn)
        mappings = {cid:map_case_evidence(c, chunkrows) for cid,c in cases.items()}
        for cid, r in new.items():
            old = base[cid]
            for stage in STAGES[:5]:
                assert r['candidate_stages'][stage] == old['candidate_stages'][stage], (cid, stage)
            assert len(r['context']) <= settings.rag_context_max_chars
            assert all(not e['truncated'] for e in r['entries'])
            expected_order = [x['chunk_id'] for x in r['candidate_stages']['rerank']]
            assert [x['chunk_id'] for x in r['candidate_stages']['constraints']] == expected_order
            assert [x['chunk_id'] for x in r['candidate_stages']['evidence']] == expected_order
            included_order = [expected_order.index(e['chunk_id']) for e in r['entries']]
            assert included_order == sorted(included_order)
            rank = {x['chunk_id']:i for i,x in enumerate(r['candidate_stages']['rerank'],1)}
            prior_reasons = dict(old['decision']['rejected'])
            prior_ranks = {x['chunk_id']:i for i,x in enumerate(old['candidate_stages']['constraints'],1)}
            for e in r['entries']:
                chunk = store.load_chunk(conn, e['chunk_id'])
                assert chunk is not None and e['included_text'] == chunk.text.strip(), (cid, e['chunk_id'], 'body')
                for ref in e['sources']:
                    original = store.load_chunk(conn, ref['chunk_id'])
                    assert original is not None
                    identity = compare_content(original.text.strip(), e['included_text']).status
                    assert identity in {'exact', 'layout_equivalent'}, (cid, ref, identity)
                    layout_equivalent_sources += original.text.strip() != e['included_text']
                entry_count += 1
                duplicate_sources += len(e['sources'])-1
            for eid in r['gained_gold']:
                gains.append(dict(id=cid, question=r['question'], gold_id=eid,
                     baseline_context_rank=old['scores']['context']['gold_completion_ranks'].get(eid),
                     bge_rank=r['scores']['rerank']['gold_completion_ranks'].get(eid),
                     new_context_rank=r['scores']['context']['gold_completion_ranks'].get(eid)))
            if r['lost_gold']:
                losses.append(dict(id=cid, gold_ids=r['lost_gold']))
            for eid in r['added_hard_negative']:
                index = int(eid.rsplit(':',1)[1])-1
                evidence = cases[cid].hard_negatives[index]
                matches = []
                for i,e in enumerate(r['entries'],1):
                    matched = score_stage('context', [e], mappings[cid])
                    if eid in matched['hard_negative_ids']:
                        matches.append(dict(rank=i, bge_rank=rank[e['chunk_id']], old_condition_rank=prior_ranks[e['chunk_id']],
                            old_exclusion=prior_reasons.get(e['chunk_id'],'已纳入旧Context或预算未装入'),
                            old_condition=next(x for x in old['candidate_stages']['constraints'] if x['chunk_id']==e['chunk_id']).get('constraint_decision'),
                            entry=e))
                negatives.append(dict(id=cid, question=r['question'], cohort=r['cohort'],
                    negative_id=eid, gold_annotation=evidence, matches=matches,
                    was_negative_in_A=eid in arms['A'][cid]['scores']['context']['hard_negative_ids']))
            if not r['scores']['context']['gold_evidence_total_declared']:
                no_gold.append(dict(id=cid, question=r['question'], expected_behavior=cases[cid].expected_behavior,
                    before_entries=len(old['entries']), after_entries=len(r['entries']),
                    known_negative_before=old['scores']['context']['hard_negative_ids'],
                    known_negative_after=r['scores']['context']['hard_negative_ids'],
                    entries=r['entries']))
            s = r['scores']['context']
            if s['gold_evidence_total_declared'] and s['gold_evidence_found'] != s['gold_evidence_total_declared']:
                issues.append(dict(id=cid, question=r['question'], review_status=r['review_status'], cohort=r['cohort'],
                    missing={eid:next((dict(stage=stage,rank=r['scores'][stage]['gold_completion_ranks'].get(eid)) for stage in reversed(STAGES) if eid in r['scores'][stage]['gold_completion_ranks']), None)
                        for eid in [f'{cid}:positive:{i+1}' for i in range(len(cases[cid].gold_evidence))] if eid not in s['gold_completion_ranks']},
                    scores={stage:r['scores'][stage] for stage in STAGES}))
    summary = dict(cases=111, actual_entries_checked=entry_count, merged_extra_sources=duplicate_sources,
        all_included_text_exact_index_body=True, layout_equivalent_secondary_sources=layout_equivalent_sources,
        all_secondary_sources_exact_or_layout_equivalent=True, all_budgets_respected=True, all_entries_untruncated=True,
        upstream_five_stages_identical=True, bge_order_preserved=True,
        gained_quotes=gains, lost_quotes=losses, new_negative_quotes=negatives, no_gold_cases=no_gold,
        unresolved=issues, api_calls=0)
    write(OUT/'source_bound_audit.json', summary)
    # Audit copy is sufficient for reading; Gold remains untouched.
    lines = ['# 新增标注负例与不计正向分题的逐项复核', '',
             '以下只核实实际Context中的资料与既有标注；不据模型输出改Gold。', '']
    for n in negatives:
        lines += [f"## {n['id']}：{n['question']}", '', json.dumps(n['gold_annotation'],ensure_ascii=False,indent=2), '']
        for m in n['matches']:
            e=m['entry']
            lines += [f"BGE第{m['bge_rank']}；旧条件第{m['old_condition_rank']}；修改后Context第{m['rank']}；旧排除：{m['old_exclusion']}",
                      f"来源：{e['filename']} · {e['location']}", '', e['included_text'], '']
    for n in no_gold:
        lines += [f"## 无正向Gold {n['id']}：{n['question']}",
                  f"预期{n['expected_behavior']}；Context {n['before_entries']}→{n['after_entries']}段", '']
        for e in n['entries']:
            lines += [f"{e['filename']} · {e['location']}", e['included_text'], '']
    (OUT/'negative_source_review.md').write_text('\n'.join(lines),encoding='utf8')
    print('AUDIT', json.dumps(dict(cases=111, entries=entry_count, new_negative_quotes=len(negatives),
          new_negative_questions=sorted({n['id'] for n in negatives}), unresolved=[x['id'] for x in issues], gained=gains),ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
