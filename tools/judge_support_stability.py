"""Apply the unchanged Gold judge to predeclared fixed-context repetitions."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re


def main():
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.core.task_manager import CancelToken
    from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
    from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
    from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
    from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
    root=Path(__file__).resolve().parents[1];out=root/'reports/rag_support_graph_20261113'
    manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(root/d['path'],reviewed_only=False).cases}
    settings=AppSettings.from_dict(json.loads((out/'settings_before.json').read_text(encoding='utf8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    target=out/'stability_judgements.jsonl'
    assert not target.exists()
    work=[]
    for mode in ('fixed_prompt','structured'):
        rows=[json.loads(l) for l in (out/('B_'+mode+'_stability.jsonl')).read_text(encoding='utf8').splitlines()]
        assert len(rows)==24
        work.extend((mode,r) for r in rows)
    def check(mode,row):
        result={'id':row['id'],'mode':mode,'repeat':row['repeat'],'context_sha256':row['context_sha256']}
        try:
            if row.get('error'):result['generation_error']=row['error'];return result
            gateway=OpenAICompatibleGateway(max_transient_retries=2)
            result['judgement']=GoldAnswerJudge(gateway,settings,key).evaluate(cases[row['id']],
                {**row['answer'],'evidence_context':row['context']},CancelToken())
            claims=[c for log in row.get('structured_diagnostics',[]) for c in log.get('answer_plan',{}).get('claims',[])]
            result['fact_ids']=sorted({fid for c in claims for fid in c.get('fact_ids',[])})
            result['claim_bindings']=[{'type':c['claim_type'],'scope':c['scope'],'certainty':c['certainty'],
                'refs':c['source_refs'],'fact_ids':c.get('fact_ids',[])} for c in claims]
            result['answer_values_without_citations']=sorted(set(re.findall(r'\d+(?:\.\d+)?',re.sub(r'\[S\d+\]','',row['answer']['text']))))
        except Exception as exc:result['judge_error']=str(exc).replace(key,'[REDACTED]')
        return result
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(check,*item) for item in work]):
            row=future.result()
            with target.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            print(row['id'],row['mode'],row['repeat'],row.get('judgement',{}).get('verdict','error'),flush=True)


if __name__=='__main__':main()
