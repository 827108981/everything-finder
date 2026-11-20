"""One low-concurrency recovery pass for transport failures, never quality failures."""
from dataclasses import replace
from hashlib import sha256
import json

from tools.validate_prompt_factorial_20261119 import OUT, inputs, check_frozen, recheck, write
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def main():
    check_frozen()
    cases, contexts, settings = inputs()
    rows = [json.loads(l) for l in (OUT/'answers.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows)==500, 'Recovery only after the frozen main experiment finishes'
    failed = [r for r in rows if 'error' in r]
    target = OUT/'transport_recovery.jsonl'
    prior = [json.loads(l) for l in target.read_text(encoding='utf8').splitlines()] if target.exists() else []
    done = {(r['id'],r['arm'],r['round']) for r in prior}
    write(OUT/'transport_recovery_protocol.json',dict(main_tasks=500,failed_tasks=len(failed),
        max_recovery_passes=1,workers=1,reason='HTTP403 burst followed by successful requests with unchanged account/model; recover missing executions only.',
        preserve_failed_records=True,no_quality_based_retry=True,reuse_successful_generation_if_judge_failed=True))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    judge = GoldAnswerJudge(gateway,settings,key)
    for old in failed:
        identity = old['id'],old['arm'],old['round']
        if identity in done:continue
        r = {k:v for k,v in old.items() if k!='error'}
        r.update(transport_recovery=True,original_error=old['error'],recovery_attempt=1)
        case,ctx = cases[r['id']],contexts[r['id']]
        try:
            if 'answer' not in r:
                request = GenerationRequest(question=r['question'],context=ctx['context'],model=settings.rag_api_model,
                    api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,
                    timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=1000,
                    query_plan_summary='',prompt_variant=r['arm'])
                payload = old['api_request']
                # Preserve exactly the first attempt user input including its original plan summary.
                if request.api_style=='chat_completions':
                    from local_full_text_search.rag.generation.base import TextCompletionRequest
                    r['answer']=gateway.complete_text(TextCompletionRequest(system_prompt=payload['messages'][0]['content'],
                        user_prompt=payload['messages'][1]['content'],model=request.model,api_base_url=request.api_base_url,
                        api_style=request.api_style,api_key=key,timeout_seconds=request.timeout_seconds,max_output_tokens=1000),CancelToken())
                else:
                    from local_full_text_search.rag.generation.base import TextCompletionRequest
                    r['answer']=gateway.complete_text(TextCompletionRequest(system_prompt=payload['instructions'],
                        user_prompt=payload['input'],model=request.model,api_base_url=request.api_base_url,
                        api_style=request.api_style,api_key=key,timeout_seconds=request.timeout_seconds,max_output_tokens=1000),CancelToken())
            citation=CitationValidator().inspect(r['answer'],{e['source_id']:() for e in ctx['entries']})
            r['citation_check']=dict(structurally_valid=citation.structurally_valid,cited_ids=list(citation.cited_ids),
                unknown_ids=list(citation.unknown_ids),reason=citation.reason)
            r['judgement']=judge.evaluate(case,dict(status='answered',text=r['answer'],evidence_context=ctx['context']),CancelToken())
            r['legacy_guard_judgement']=recheck(case,r['judgement'],r['answer'],ctx['context'],legacy=True)
        except Exception as exc:
            r['error']=str(exc).replace(key,'[REDACTED]')[:600]
        with target.open('a',encoding='utf8') as stream:
            stream.write(json.dumps(r,ensure_ascii=False).replace(key,'[REDACTED]')+'\n')
        print('RECOVERY',r['id'],r['arm'],r['round'],r.get('judgement',{}).get('verdict'),r.get('error',''),flush=True)
    check_frozen()


if __name__=='__main__':main()
