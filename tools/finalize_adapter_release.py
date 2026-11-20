"""Reuse only identical requests; preserve every prior run and its provenance."""
import json
from tools.rag_eval_common import ROOT,read,lines,write,sha
from tools.compare_adapter_release import main as compare
OUT=ROOT/'reports/rag_adapter_robustness_20261111'

def main():
    compare();delta=read(OUT/'release_input_delta.json');assert delta['count']==92
    changed=set(delta['changed_ids'])
    earlier={};fresh={}
    for p in OUT.glob('answers_AB_part*.jsonl'):
        for r in lines(p):
            if r.get('error'):continue
            assert (r['id'],r['arm']) not in earlier
            earlier[(r['id'],r['arm'])]={**r,'run_file':p.name}
    for p in OUT.glob('release_answers_part*.jsonl'):
        for r in lines(p):
            assert not r.get('error') and r['context_equal'],r['id']
            assert (r['id'],r['arm']) not in fresh
            fresh[(r['id'],r['arm'])]={**r,'run_file':p.name}
    assert all((cid,'B') in fresh for cid in changed),changed-{cid for cid,arm in fresh}
    current={r['id']:r for r in lines(OUT/'native_release.jsonl')}
    rows=[]
    for key,r in earlier.items():
        cid,arm=key
        if arm=='B' and key in fresh:
            r={**fresh[key],'validation_provenance':'fresh_release_generation'}
        elif arm=='B':
            assert cid not in changed
            r={**r,'validation_provenance':'identical_actual_context_and_plan_summary_verified'}
        else:
            r={**r,'validation_provenance':'paired_disabled_control_unchanged'}
        if arm=='B':assert r['context']==current[cid]['context'],cid
        rows.append(r)
    assert len(rows)==184
    (OUT/'final_answers_AB.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),'utf-8')
    manifest=read(OUT/'native_release_manifest.json')
    drift=[p for p,h in manifest['code_hashes'].items() if sha(p)!=h]
    assert not drift,drift
    write(OUT/'release_validation_manifest.json',{'changed_input_ids':sorted(changed),'fresh_generation_ids':sorted(cid for cid,arm in fresh),
        'reused_identical_B':sum(r['validation_provenance']=='identical_actual_context_and_plan_summary_verified' for r in rows),
        'A_count':92,'B_count':92,'code_drift':drift,
        'gold_drift':[p for p,h in manifest['dataset_hashes'].items() if sha(p)!=h],
        'note':'184 answers in final comparison, not 184 additional calls. All changed B inputs rerun once; unchanged inputs reuse retained real outputs.'})

if __name__=='__main__':main()
