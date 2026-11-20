"""Two fixed repeats per prompt for the observed 038 answer omission.

All repeats are retained, regardless of verdict; this is not rerun-until-pass.
"""
import dataclasses
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def main():
    out=Path('reports/project_constraint_fix_20261102')
    assert not (out/'variation_038.json').exists()
    rows=[r for r in map(json.loads,(out/'cases.jsonl').read_text(encoding='utf-8').splitlines()) if r['id']=='rag-biz-038']
    assert len(rows)==2
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    case=next(c for c in load_gold_dataset(Path('tests/rag/fixtures/rag_gold_v3.jsonl')).cases if c.case_id=='rag-biz-038')
    contexts=[next(e['llm_context'] for e in r['events'] if e.get('llm_context')) for r in rows]
    # Coverage equality does not imply identical background selection/order.
    # Preserve each arm's real request and record that distinction explicitly.
    primary=[r['candidate_stages']['context'][0]['chunk_id'] for r in rows]
    assert primary[0]==primary[1]
    def run(row, repeat):
        payload=next(e['api_request'] for e in row['events'] if e.get('stage')=='generating')
        assert set(payload)=={'messages','model','temperature','max_tokens'} and payload['temperature']==0
        assert settings.rag_api_model==payload['model']
        request=TextCompletionRequest(system_prompt=payload['messages'][0]['content'],
            user_prompt=payload['messages'][1]['content'],model=payload['model'],
            api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,
            api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=payload['max_tokens'])
        gateway=OpenAICompatibleGateway(max_transient_retries=1)
        token=CancelToken()
        text=gateway.complete_text(request,token)
        answer={**row['answer'],'text':text}
        verdict=GoldAnswerJudge(gateway,settings,key).evaluate(case,answer,token)
        return {'arm':row['arm'],'repeat':repeat,'answer':answer,'judgement':verdict,
                'payload_sha256':hashlib.sha256(json.dumps(payload,sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,row,i) for row in rows for i in (1,2)]
        results=[f.result() for f in futures]
    value={'case':'rag-biz-038','same_context':contexts[0]==contexts[1],
           'same_first_evidence':True,
           'protocol':'2 additional generations and judgments per arm, fixed before inspecting repeat results; all original and repeat outcomes retained. Real before/after prompts include their different background contexts.',
           'original':[{'arm':r['arm'],'answer':r['answer'],'judgement':r['gold_judgement']} for r in rows],
           'repeats':results}
    (out/'variation_038.json').write_text(json.dumps(value,ensure_ascii=False,indent=2).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    print([(r['arm'],r['repeat'],r['judgement']['verdict']) for r in results])


if __name__=='__main__':
    main()
