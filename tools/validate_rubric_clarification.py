"""Independent axes: frozen answers/new rubric, and old rubric/new generation.

Existing v3 and reviewed states are never changed. Proposed v4 is diagnostic.
"""
from __future__ import annotations
import dataclasses
import hashlib
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from local_full_text_search.config.constants import SETTINGS_PATH, RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import score_case
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway, provider_request_payload
from local_full_text_search.rag.generation.base import ModelAnswer
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from tools.verify_rerank_scope_bonus import mappings

OUT=Path(sys.argv[1]) if len(sys.argv)>1 else Path('reports/rubric_clarification_20261102')


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    retry='--retry-failed' in sys.argv
    assert retry or not (OUT/'protocol.json').exists()
    original=[json.loads(l) for l in Path('reports/project_constraint_fix_20261102/cases.jsonl').read_text(encoding='utf-8').splitlines()]
    before={(r['cohort'],r['id']):r for r in original if r['arm']=='after'}
    assert len(before)==60 and not any('error_type'in r for r in before.values())
    fixtures=Path('tests/rag/fixtures')
    datasets={'development':load_gold_dataset(fixtures/'rag_gold_v3.jsonl'),
        'holdout_pending':load_gold_dataset(fixtures/'rag_acceptance_v1.jsonl',reviewed_only=False)}
    proposed={c.case_id:c for c in load_gold_dataset(fixtures/'rag_gold_v4_proposed.jsonl',reviewed_only=False).cases}
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    settings.rag_semantic_clarification_enabled=True  # Explicit experimental arm only.
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key
    def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    index_db=Path('E:/everythingfinder-db/search_index.db')
    store=RagIndexStore(index_db.parent/'rag');bundle=store.active()
    previous=json.loads(Path('reports/project_constraint_fix_20261102/protocol.json').read_text(encoding='utf-8'))
    assert bundle.bundle_id==previous['bundle_id']
    assert {k:getattr(settings,k) for k in previous['settings']}==previous['settings']
    dump(OUT/('retry_protocol.json' if retry else 'protocol.json'),{'bundle_id':bundle.bundle_id,'llm_model':settings.rag_api_model,
        'settings':previous['settings'],'semantic_clarification_candidate_enabled':True,
        'datasets':{name:{'path':str(d.path),'sha256':d.sha256} for name,d in datasets.items()},
        'proposed_sha256':sha((fixtures/'rag_gold_v4_proposed.jsonl').read_bytes()),
        'baseline':'Latest successful after rows from project_constraint_fix_20261102; old generation prompt, current retriever.',
        'axes':'Score same baseline answers with v3/v4 proposed, then new answers with both rubrics. Do not attribute changed labels to algorithm improvement.',
        'status':'v4 and holdout pending diagnostic only; no change to official evaluation defaults',
        'script_sha256':sha(Path(__file__).read_bytes()),'concurrency':4})
    cache={(r['question'],c['chunk_id'],sha(c['text'].encode())):c['rerank_score']
           for r in before.values() for c in r['candidate_stages']['rerank']}
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
        repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision)
    vectors={};retrieval_lock=threading.Lock()
    class Embedding:
        model_id=embedding.model_id;model_revision=embedding.model_revision;dimension=embedding.dimension
        def embed_query(self,text):
            if text not in vectors:vectors[text]=embedding.embed_query(text)
            return vectors[text]
    class Reranker(LocalBgeReranker):
        def __init__(self):
            self.model_id=settings.rag_reranker_repo_id;self.revision=settings.rag_reranker_revision
            self.max_rank_drop=settings.rag_rerank_max_rank_drop;self.last_scored_candidates=()
        def rerank(self,question,candidates,top_n,token):
            for c in candidates:
                c.rerank_score=cache[(question,c.chunk.stable_chunk_id,sha(c.chunk.text.encode()))]
                c.rerank_score_combined=(settings.rag_rerank_cross_encoder_weight*c.rerank_score
                    +settings.rag_rerank_retrieval_prior_weight*retrieval_prior(c)
                    +settings.rag_rerank_answer_support_weight*float(c.answer_support_score or 0))
            ordered=sorted(candidates,key=lambda c:(-c.rerank_score_combined,-c.fused_score,c.chunk.stable_chunk_id))
            self.last_scored_candidates=tuple(self._apply_rank_guard(ordered))
            return self._apply_document_diversity(self.last_scored_candidates,top_n)
    fields={'lexical_candidates':'lexical','vector_candidates':'vector','fused_candidates':'fusion',
        'deduplicated_candidates':'deduplicated','reranked_candidates':'rerank','constraint_candidates':'constraints','evidence_candidates':'evidence'}
    def work(cohort,case):
        baseline=before[(cohort,case.case_id)]
        stages={};events=[];held=[False]
        def record(event):
            events.append(event)
            for key,name in fields.items():
                if isinstance(event.get(key),list):stages[name]=event[key]
            if event.get('context_entries'):
                selected={c['chunk_id']:c for c in stages.get('evidence',[]) if c.get('selected')}
                stages['context']=[{**selected[e['chunk_id']],'included_text':e['included_text'],'truncated':e.get('truncated',False)} for e in event['context_entries']]
            if event.get('stage') in ('generating','assessing_clarification') and held[0]:
                held[0]=False;retrieval_lock.release()
        reuse={'generation_reused':False,'assessment_calls':0,'fresh_generation_calls':0}
        class Gateway(OpenAICompatibleGateway):
            def assess_clarification(self,request,token):
                reuse['assessment_calls']+=1
                return super().assess_clarification(request,token)
            def generate(self,request,token):
                old_event=next((e for e in baseline['events'] if e.get('stage')=='generating'),None)
                old_draft=next((e.get('draft_answer') for e in baseline['events'] if e.get('stage')=='draft_generated'),None)
                if cohort=='development' and old_event and old_draft and old_event['api_request']==provider_request_payload(request):
                    reuse['generation_reused']=True
                    return ModelAnswer(old_draft,baseline['answer'].get('provider','OpenAI Compatible'),request.model)
                reuse['fresh_generation_calls']+=1
                return super().generate(request,token)
        token=CancelToken();gateway=Gateway(max_transient_retries=1)
        try:
            service=AnswerService(IndexGenerationStore(index_db),store,Embedding(),Reranker(),gateway,
                WindowsCredentialSecretStore(),settings,progress_callback=record)
            retrieval_lock.acquire();held[0]=True
            answer=_answer_record(service.answer(RagQuery(text=case.question,
                lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,
                final_top_k=settings.rag_final_top_k,min_ocr_confidence=settings.ocr_min_confidence),token))
        finally:
            if held[0]:held[0]=False;retrieval_lock.release()
        for stage in ('lexical','vector','fusion','deduplicated','rerank','constraints','evidence','context'):
            assert stages.get(stage)==baseline['candidate_stages'].get(stage),(case.case_id,stage)
        gold_maps=mappings(baseline,case.raw)
        scored=score_case(case,stages,gold_maps,answer=answer)
        judge=GoldAnswerJudge(gateway,settings,key)
        before_v3=baseline.get('gold_judgement')
        unchanged_answer=(answer['text']==baseline['answer']['text'] and answer['status']==baseline['answer']['status'])
        after_v3=before_v3 if before_v3 and unchanged_answer else judge.evaluate(case,answer,token)
        before_v4=after_v4=None
        if cohort=='development':
            draft=proposed[case.case_id]
            if draft.answer_checks==case.answer_checks:
                before_v4=before_v3;after_v4=after_v3
            else:
                before_v4=judge.evaluate(draft,baseline['answer'],token)
                after_v4=before_v4 if unchanged_answer else judge.evaluate(draft,answer,token)
        return {**scored,'cohort':cohort,'candidate_stages':stages,'events':events,
            'baseline_answer':baseline['answer'],'judgements':{'before_v3':before_v3,'before_v4_proposed':before_v4,
                'after_v3':after_v3,'after_v4_proposed':after_v4},
            'formal_eligible':cohort=='development','proposed_formal_eligible':False,
            'request_reuse':reuse,
            'retrieval_stages_identical':True}
    results=[]
    latest={}
    if retry and (OUT/'cases.jsonl').exists():
        latest={(r['cohort'],r['id']):r for r in map(json.loads,(OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines())}
    with ThreadPoolExecutor(max_workers=2 if retry else 4) as pool:
        futures={pool.submit(work,cohort,c):(cohort,c.case_id) for cohort,d in datasets.items() for c in d.cases
                 if (cohort,c.case_id) not in latest or 'error_type' in latest[(cohort,c.case_id)]
                 or any(e.get('stage')=='clarification_assessment_unavailable' for e in latest[(cohort,c.case_id)].get('events',[]))}
        for f in as_completed(futures):
            cohort,cid=futures[f]
            try:
                r=f.result()
            except Exception as exc:
                import traceback
                r={'id':cid,'cohort':cohort,'error_type':type(exc).__name__,
                    'error_message':str(exc).replace(key,'[REDACTED]')[:300],
                    'frames':[(s.filename,s.lineno,s.name) for s in traceback.extract_tb(exc.__traceback__)]}
            results.append(r)
            with (OUT/'cases.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            print(cohort,cid,r.get('error_type') or {name:(v or {}).get('verdict') for name,v in r['judgements'].items()},flush=True)
    all_attempts=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    results=list({(r['cohort'],r['id']):r for r in all_attempts}.values())
    assert all(sha(d.path.read_bytes())==d.sha256 for d in datasets.values())
    assert store.active().bundle_id==bundle.bundle_id
    dump(OUT/'completion.json',{'n':len(results),'raw_attempt_rows':len(all_attempts),'errors':sum('error_type'in r for r in results),
        'historical_gold_unchanged':True,'same_retrieval_asserted':sum(r.get('retrieval_stages_identical',False) for r in results)})


if __name__=='__main__':main()
