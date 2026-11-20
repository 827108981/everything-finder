"""Paired real LLM answers through native downstream, with frozen upstream replay.

Not a fresh retrieval benchmark: validates generation/gates against the exact
contexts already produced by the native retrieval comparison. Never supplies Gold
to runtime; Gold goes to the separate judge only.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace
from threading import Lock
from unittest.mock import patch
import argparse
import json

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, RagQuery
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.runner import _answer_record
from tools.rag_eval_common import read, lines, write, sha
from tools.validate_business_repair import native
FILES=native.FILES
HISTORICAL=native.OUT
from tools.rag_eval_common import ROOT
OUT=ROOT/'reports/rag_business_repair_20261112'

CURRENT=ContextVar('query_plan_replay')

class ReplayReranker:
    def rerank(self, question, candidates, top_n, token):
        capture=CURRENT.get()
        assert question==capture['question']
        by_id={i.chunk.stable_chunk_id:i for i in candidates}
        self.last_scored_candidates=[]
        for row in capture['candidate_stages'].get('rerank',[]):
            i=by_id[row['chunk_id']]
            for field in ('rerank_score','rerank_score_combined','answer_support_score','answer_support_reasons'):
                setattr(i,field,row.get(field))
            self.last_scored_candidates.append(i)
        assert len(by_id)==len(self.last_scored_candidates)
        return self.last_scored_candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',default='on_all')
    ap.add_argument('--ids',default='');ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--output',default='answers_AB');ap.add_argument('--arms',default='AB');args=ap.parse_args()
    target=OUT/(args.output+'.jsonl');assert not target.exists()
    s=AppSettings.from_dict(read(SETTINGS_PATH))
    secrets=WindowsCredentialSecretStore();key=secrets.retrieve(s.rag_api_secret_ref);assert key
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    source_store=IndexGenerationStore('E:/everythingfinder-db/search_index.db')
    embedding=SimpleNamespace(model_id=bundle.embedding_repo_id,model_revision=bundle.embedding_revision,dimension=bundle.embedding_dimension)
    cases={c.case_id:c for p in FILES for c in load_gold_dataset(p,reviewed_only=False).cases}
    capture_paths={'A':HISTORICAL/'off_all.jsonl','B':OUT/(args.candidate+'.jsonl')}
    captures={arm:{r['id']:r for r in lines(path)} for arm,path in capture_paths.items()}
    retrieval_lock=Lock()
    ids=list(captures['A']) if not args.ids else args.ids.split(',')
    assert all(i in captures[arm] for arm in args.arms for i in ids)
    write(OUT/(args.output+'_protocol.json'),{'mode':'native_downstream_with_frozen_upstream','ids':ids,
          'input_hashes':{arm:sha(path) for arm,path in capture_paths.items()},
          'code_hashes':{str(p):sha(p) for p in (ROOT/'local_full_text_search/rag').rglob('*.py')},
          'dataset_hashes':{str(p):sha(p) for p in FILES},
          'repeat':1,'quality_retry':False,'bundle':bundle.bundle_id,'model':s.rag_api_model,
          'context_equality_required':True,'labels_runtime_access':False})

    def frozen_retrieve(self,query,token,plan=None):
        row=CURRENT.get();assert query.text==row['question']
        result=[]
        with store.connect_read_only(bundle.metadata_path) as conn:
            for r in row['candidate_stages'].get('fusion',[]):
                chunk=store.load_chunk(conn,r['chunk_id']);assert chunk.text==r['text']
                i=EvidenceChunk(chunk)
                for f in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score'):
                    setattr(i,f,r.get(f))
                i.query_plan_diagnostics=dict(r.get('query_plan_diagnostics',{}))
                result.append(i)
        return result

    def work(arm,cid):
        capture=captures[arm][cid];CURRENT.set(capture);events=[];token=CancelToken()
        settings=replace(s,rag_query_plan_enabled=arm=='B')
        gateway=OpenAICompatibleGateway(max_transient_retries=2)
        held=[False]
        def progress(e):
            events.append(e)
            if held[0] and e['stage'] in ('credential_read','retrieval_completed'):
                held[0]=False;retrieval_lock.release()
        service=AnswerService(source_store,store,embedding,ReplayReranker(),gateway,secrets,settings,progress_callback=progress)
        q=RagQuery(capture['question'],include_ocr=True,min_ocr_confidence=s.ocr_min_confidence,
            lexical_top_k=s.rag_lexical_top_k,vector_top_k=s.rag_vector_top_k,fusion_top_k=s.rag_fusion_top_k,
            reranker_top_k=s.rag_reranker_top_k,final_top_k=s.rag_final_top_k)
        result={'id':cid,'arm':arm,'cohort':capture['cohort'],'question':capture['question']}
        try:
            retrieval_lock.acquire();held[0]=True
            result['answer']=_answer_record(service.answer(q,token))
            if held[0]: held[0]=False;retrieval_lock.release()
            context=next((e['llm_context'] for e in reversed(events) if 'llm_context' in e),'')
            result['context']=context;result['context_equal']=context==capture['context']
            if not result['context_equal']: raise ValueError('Frozen native context mismatch; cannot score as paired replay')
            result['judgement']=GoldAnswerJudge(gateway,s,key).evaluate(cases[cid],{**result['answer'],'evidence_context':context},token)
        except Exception as exc: result['error']=str(exc).replace(key,'[REDACTED]')
        finally:
            if held[0]: held[0]=False;retrieval_lock.release()
        result['events']=events
        return result
    with patch.object(HybridRetriever,'retrieve',frozen_retrieve), ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks=[(arm,cid) for n,cid in enumerate(ids) for arm in (('A','B') if n%2 else ('B','A')) if arm in args.arms]
        futures=[pool.submit(work,*t) for t in tasks]
        for f in as_completed(futures):
            row=f.result()
            with target.open('a',encoding='utf-8') as out:out.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
            print(row['id'],row['arm'],row.get('context_equal'),(row.get('judgement') or {}).get('verdict'),row.get('error',''),flush=True)

if __name__=='__main__':main()
