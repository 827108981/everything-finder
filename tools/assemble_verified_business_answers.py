"""Reuse only demonstrably identical generation inputs; preserve run provenance."""
import json
from tools.rag_eval_common import ROOT,lines,write,sha
OUT=ROOT/'reports/rag_business_repair_20261112'

def changed():
    old={r['id']:r for r in lines(OUT/'final_all.jsonl')}
    new={r['id']:r for r in lines(OUT/'verified_all.jsonl')}
    assert len(old)==len(new)==91
    plan=lambda r:next((e['query_plan'] for e in r['events'] if 'query_plan' in e),{})
    # Retrieval result contains trace IDs/timings; generation equivalence is
    # established by exact question, plan, context and preserved answer status.
    ids=[cid for cid,r in new.items() if r['context']!=old[cid]['context'] or plan(r)!=plan(old[cid])]
    return old,new,ids

def main():
    old,new,ids=changed();write(OUT/'verified_changed_ids.json',ids)
    print('changed contexts/plans:',','.join(ids),flush=True)
    extra=OUT/'verified_changed_B.jsonl'
    if ids and not extra.exists():return
    fresh={r['id']:r for r in lines(extra)} if extra.exists() else {}
    previous={r['id']:r for r in lines(OUT/'final_answers_B.jsonl')}
    assert set(ids)<=set(fresh)
    result=[]
    for cid in new:
        if cid in ids:
            row=dict(fresh[cid]);row['verification_provenance']={'mode':'fresh_final_downstream','capture':'verified_all.jsonl'}
        else:
            row=dict(previous[cid]);assert row['context']==new[cid]['context'] and row.get('context_equal')
            assert row['question']==new[cid]['question']==old[cid]['question']
            row['verification_provenance']={'mode':'reuse_identical_question_plan_context','answer_run':'final_answers_B.jsonl','answer_hash':sha(OUT/'final_answers_B.jsonl'),'verified_capture':'verified_all.jsonl','note':'No new generation claim; original verdict and failures preserved. Last change only affects source selection, not generation requests.'}
        result.append(row)
    (OUT/'verified_answers_B.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in result),encoding='utf8')
    write(OUT/'verified_answer_provenance.json',{'count':91,'fresh_ids':ids,'reused_identical_inputs':91-len(ids),'quality_retries':0})

if __name__=='__main__':main()
