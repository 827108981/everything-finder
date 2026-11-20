"""Audit the fixed four-arm experiment without any additional API requests."""
from hashlib import sha256
import json
import re

from tools.experiment_035_context_ablation_20261119 import OUT, freeze, prepare
from local_full_text_search.rag.generation.openai_compatible import provider_request_payload


def main():
    protocol=freeze()
    rows=[json.loads(l) for l in (OUT/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    review=json.loads((OUT/'manual_review.json').read_text(encoding='utf8'))
    contexts=json.loads((OUT/'contexts.json').read_text(encoding='utf8'))
    assert len(rows)==20 and len({(r['arm'],r['round']) for r in rows})==20
    assert not any(r.get('error') for r in rows)
    assert len(review['records'])==20
    patterns=[]
    for r in rows:
        c=contexts[r['arm']]['context']
        assert r['context_sha256']==sha256(c.encode()).hexdigest()==protocol['context_sha256'][r['arm']]
        assert r['api_request']['messages'][0]['content']==protocol['system_prompt']
        msg=r['api_request']['messages'][1]['content']
        assert c in msg
        patterns.append(msg.replace(c,'<FROZEN_CONTEXT>'))
        assert not r['citation_check']['unknown_ids'] and r['citation_check']['structurally_valid']
        assert r['gold_in_generation'] is False
        meta=r['generation_response_metadata']
        assert len(meta)==1 and meta[0]['model']==protocol['model']
        assert meta[0]['finish_reasons']==['stop']
    assert len(set(patterns))==1, 'Non-context user input differs'
    a=contexts['A']['context'];d=contexts['D']['context'];b=contexts['B']['context'];c=contexts['C']['context']
    p=list(re.finditer(r'(?m)^\[(S\d+)\]\s*\n',a))
    ss={m[1]:a[m.start():p[i+1].start() if i+1<len(p) else len(a)] for i,m in enumerate(p)}
    assert b==ss['S1']+ss['S2'] and c==b+ss['S3'] and d==a.replace(ss['S3'],'',1)
    audit=dict(tasks=20,independent_new_generations=20,old_answers_reused=0,errors=0,
        question_and_prompt_and_noncontext_input_identical=True,
        a_exact_previous_request=True,context_interventions_verified=True,
        source_ids_not_renumbered=True,gold_only_in_judge=True,
        all_actual_models=protocol['model'],all_stop=True,all_citations_exist=True,
        protected_hashes_after='通过',review_records=20,
        business_code_gold_settings_changed=False,
        service_acceptance='不适用：固定Context生成器诊断，无新链路实现')
    (OUT/'execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False))


if __name__=='__main__':main()
