"""Same actual context and plan, with/without the new answer contract.

Independent generation calls, one per arm. No quality retries; raw answers and
original judge decisions are retained even if the candidate is worse.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from pathlib import Path
import json
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evidence_contract import contract_summary
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from tools.validate_query_plan_adapter import FILES
from tools.rag_eval_common import ROOT,read,lines,write,sha

OUT=ROOT/'reports/rag_adapter_robustness_20261111'
IDS=['rag-biz-020','rag-biz-021','rag-biz-023','rag-row-holdout2-001','evidence-flow-independent-07','rag-row-holdout-001','rag-biz-028']

def main():
    path=OUT/'contract_ablation.jsonl';assert not path.exists()
    captures={r['id']:r for r in lines(OUT/'native_v2.jsonl')}
    cases={c.case_id:c for p in FILES for c in load_gold_dataset(p,reviewed_only=False).cases}
    settings=AppSettings.from_dict(read(SETTINGS_PATH));secrets=WindowsCredentialSecretStore()
    key=secrets.retrieve(settings.rag_api_secret_ref);assert key
    def work(cid,arm):
        row=captures[cid];p=QueryInterpreter(enhanced=True,domain_config='chemiluminescence').interpret(row['question'])
        summary=AnswerService._plan_summary(p);contract=contract_summary(p)
        if arm=='without_contract' and contract:
            assert summary.endswith(contract);summary=summary[:-len(contract)]
        request=GenerationRequest(question=row['question'],context=row['context'],query_plan_summary=summary,
            model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,
            api_key=key,timeout_seconds=settings.rag_api_timeout_seconds)
        token=CancelToken();gateway=OpenAICompatibleGateway(max_transient_retries=2)
        out={'id':cid,'arm':arm,'context_hash':__import__('hashlib').sha256(row['context'].encode()).hexdigest(),
             'plan_summary':summary,'context':row['context']}
        try:
            answer=gateway.generate(request,token)
            record={'mode':'answer','status':'answered','text':answer.text,'insufficient_evidence':False}
            out['answer']=record
            out['judgement']=GoldAnswerJudge(gateway,settings,key).evaluate(cases[cid],{**record,'evidence_context':row['context']},token)
        except Exception as exc:out['error']=str(exc).replace(key,'[REDACTED]')
        return out
    write(OUT/'contract_ablation_protocol.json',{'ids':IDS,'input_hash':sha(OUT/'native_v2.jsonl'),
        'model':settings.rag_api_model,'mode':'fixed_actual_context_plan_summary_contract_only','quality_retry':False,
        'limitations':'Generation-only ablation. One sample per arm cannot establish statistical stability.'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(work,cid,arm) for cid in IDS for arm in ('without_contract','with_contract')]
        for future in as_completed(fs):
            r=future.result()
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
            print(r['id'],r['arm'],r.get('judgement',{}).get('verdict'),r.get('error',''),flush=True)

if __name__=='__main__':main()
