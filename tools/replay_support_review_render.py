"""Replay a deterministic review renderer fix over every frozen B response.

No model resampling and no Gold enters the renderer. The unchanged judge is
called only after a changed final answer; unchanged verdicts retain provenance.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from copy import deepcopy
from pathlib import Path
import hashlib
import json


def main():
    from local_full_text_search.rag.answer_review import AnswerReviewParser
    from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
    from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
    from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
    from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.core.task_manager import CancelToken
    root=Path(__file__).resolve().parents[1];out=root/'reports/rag_support_graph_20261113'
    rows=[json.loads(l) for l in (out/'B.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==91
    target=out/'B_effective.jsonl';assert not target.exists()
    manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(root/d['path'],reviewed_only=False).cases}
    settings=AppSettings.from_dict(json.loads((out/'settings_before.json').read_text(encoding='utf8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    def replay(original):
        row=deepcopy(original)
        event=next((e for e in reversed(row['events']) if e.get('stage')=='review_completed'),None)
        provenance={'source':'B.jsonl','source_record_sha256':hashlib.sha256(json.dumps(original,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                    'method':'Deterministic replay of saved reviewer claims through current AnswerReviewParser. No model answer resampled.','answer_changed':False}
        if event and row.get('result',{}).get('status')=='answered':
            parsed=AnswerReviewParser().parse(json.dumps({'verdict':event['verdict'],'summary':event.get('review_summary',''),
                'claims':event['review_claims'],'final_answer':event['reviewed_answer']},ensure_ascii=False),
                allowed_source_ids={e['source_id'] for e in row['entries']},preserve_supported=True).result
            # Preserve a grounded optional follow-up added after review.
            followup=next((e.get('clarification_question','') for e in row['events'] if e.get('stage')=='partial_clarification_deferred'),'')
            final=parsed.final_answer
            if followup and followup not in final:final+='\n\n如需判断当前具体情况：'+followup
            if final!=row['result']['text']:
                provenance['answer_changed']=True
                provenance['old_answer']=row['result']['text']
                row['result']['text']=final
                event['before_render_fix']=event['reviewed_answer'];event['reviewed_answer']=parsed.final_answer
                event['render_fix']='No corpus-absence assertion inferred from a removed unsupported extra claim'
                row['judge_before_render_fix']=row.pop('judgement',None)
                row.pop('judge_error',None)
                try:
                    gateway=OpenAICompatibleGateway(max_transient_retries=2)
                    row['judgement']=GoldAnswerJudge(gateway,settings,key).evaluate(cases[row['id']],{**row['result'],'evidence_context':row['context']},CancelToken())
                except Exception as exc:row['judge_error']=str(exc).replace(key,'[REDACTED]')
        row['verification_provenance']=provenance
        return row
    changed=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(replay,r) for r in rows]):
            row=future.result()
            with target.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if row['verification_provenance']['answer_changed']:changed.append(row['id'])
            print(row['id'],'render_changed' if row['verification_provenance']['answer_changed'] else 'unchanged',row.get('judgement',{}).get('verdict'),flush=True)
    (out/'review_render_replay.json').write_text(json.dumps({'count':91,'changed_ids':changed,'original_run_retained':'B.jsonl',
        'new_results':'B_effective.jsonl','code_hash':hashlib.sha256((root/'local_full_text_search/rag/answer_review.py').read_bytes()).hexdigest(),
        'protocol':'All 91 frozen B records replayed. Only changed answers rejudged. No primary or review model response resampled; no quality-based retry.'},ensure_ascii=False,indent=2),encoding='utf8')


if __name__=='__main__':main()
