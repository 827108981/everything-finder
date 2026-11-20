"""Validate a narrow production fix against the saved pre-fix analyzer.

Uses current settings and real retrieval/generation. Exact raw rerank scores and
identical LLM requests may be reused; caches never depend on Gold labels.
"""
from __future__ import annotations
import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH, RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import score_case
from local_full_text_search.rag.evaluation.runner import _answer_record, _retrieval_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

OUT = Path('reports/project_constraint_fix_20261102')


def sha(value):
    return hashlib.sha256(value).hexdigest()


def main():
    retry_failed='--retry-failed' in sys.argv
    assert retry_failed or not (OUT/'protocol.json').exists(), 'Use a new directory for a new experiment'
    spec = importlib.util.spec_from_file_location('project_fix_before', OUT/'query_analyzer_before.py')
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    secrets = WindowsCredentialSecretStore()
    api_key = secrets.retrieve(settings.rag_api_secret_ref) or ''
    assert api_key and settings.rag_api_model
    def safe_json(obj):
        return json.dumps(obj, ensure_ascii=False, default=str).replace(api_key, '[REDACTED]')
    def dump(path, obj):
        path.write_text(safe_json(obj)+'\n', encoding='utf-8')
    index_db = Path('E:/everythingfinder-db/search_index.db')
    store = RagIndexStore(index_db.parent/'rag')
    bundle = store.active()
    fixtures = Path('tests/rag/fixtures')
    datasets = {'development':load_gold_dataset(fixtures/'rag_gold_v3.jsonl'),
                'holdout_pending':load_gold_dataset(fixtures/'rag_acceptance_v1.jsonl', reviewed_only=False)}
    assert len(datasets['development'].cases)==50 and len(datasets['holdout_pending'].cases)==10
    assert all(c.corpus_snapshot==bundle.bundle_id for d in datasets.values() for c in d.cases)
    assert all(c.review_status=='pending' for c in datasets['holdout_pending'].cases)
    old_rows = [r for r in map(json.loads, Path('reports/rerank_dev_holdout_20261101/cases.jsonl').read_text(encoding='utf-8').splitlines()) if r['arm']=='current']
    previous_protocol = json.loads(Path('reports/rerank_dev_holdout_20261101/protocol.json').read_text(encoding='utf-8'))
    assert previous_protocol['bundle_id']==bundle.bundle_id
    for field in ('rag_reranker_repo_id','rag_reranker_revision','rag_reranker_max_length'):
        assert getattr(settings,field)==previous_protocol['fixed_settings'][field]
    cache={(r['question'],c['chunk_id'],sha(c['text'].encode())):c['rerank_score']
           for r in old_rows for c in r['candidate_stages']['rerank']}
    protocol={'bundle_id':bundle.bundle_id, 'settings':{k:v for k,v in dataclasses.asdict(settings).items()
        if k.startswith('rag_') and not any(s in k for s in ('api','secret','key'))},
        'llm_model':settings.rag_api_model, 'datasets':{k:{'path':str(d.path),'sha256':d.sha256} for k,d in datasets.items()},
        'before_sha256':sha((OUT/'query_analyzer_before.py').read_bytes()),
        'after_sha256':sha(Path('local_full_text_search/rag/query_analyzer.py').read_bytes()),
        'script_sha256':sha(Path(__file__).read_bytes()), 'concurrency':4,
        'scope':'50 reviewed cases full before/after; 10 pending retrieval-only diagnostic',
        'cache_policy':'Exact question/chunk/body hash for raw scores; exact request excluding credentials for generation/judge. Identical requests reused to avoid random differences.',
        'acceptance':'No new specified negatives or paired evidence regressions; inspect changed answers and core/forbidden checks. Model judge is advisory, not human review.'}
    dump(OUT/('retry_protocol.json' if retry_failed else 'protocol.json'), protocol)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision,'embedding')),
        repo_id=settings.rag_embedding_repo_id, revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size)
    vectors, vector_lock = {}, threading.Lock()
    class CachedEmbedding:
        model_id=embedding.model_id
        model_revision=embedding.model_revision
        dimension=embedding.dimension
        def embed_query(self, text):
            with vector_lock:
                if text not in vectors:
                    vectors[text]=embedding.embed_query(text)
                return vectors[text]
    class CachedReranker(LocalBgeReranker):
        def __init__(self):
            self.model_id=settings.rag_reranker_repo_id
            self.revision=settings.rag_reranker_revision
            self.max_rank_drop=settings.rag_rerank_max_rank_drop
            self.last_scored_candidates=()
        def rerank(self, question, candidates, top_n, token):
            for c in candidates:
                c.rerank_score=cache[(question,c.chunk.stable_chunk_id,sha(c.chunk.text.encode()))]
                c.rerank_score_combined=(settings.rag_rerank_cross_encoder_weight*c.rerank_score
                    +settings.rag_rerank_retrieval_prior_weight*retrieval_prior(c)
                    +settings.rag_rerank_answer_support_weight*float(c.answer_support_score or 0))
            ordered=sorted(candidates,key=lambda c:(-c.rerank_score_combined,-c.fused_score,c.chunk.stable_chunk_id))
            self.last_scored_candidates=tuple(self._apply_rank_guard(ordered))
            return self._apply_document_diversity(self.last_scored_candidates,top_n)
    conn=store.connect_read_only(bundle.metadata_path)
    try:
        chunk_rows=load_chunk_rows(conn)
    finally:
        conn.close()
    maps={c.case_id:map_case_evidence(c,chunk_rows) for d in datasets.values() for c in d.cases}
    retrieval_lock=threading.Lock()
    fields={'lexical_candidates':'lexical','vector_candidates':'vector','fused_candidates':'fusion',
            'deduplicated_candidates':'deduplicated','reranked_candidates':'rerank',
            'constraint_candidates':'constraints','evidence_candidates':'evidence'}
    def pair(cohort, case):
        request_cache={}
        pair_rows=[]
        for arm in ('before','after'):
            token=CancelToken()
            stages,events={},[]
            counts={'generation_fresh':0,'generation_reused':0,'judge_fresh':0,'judge_reused':0}
            request_hashes=[]
            retrieval_held=[False]
            def record(event):
                events.append(event)
                for key,name in fields.items():
                    if isinstance(event.get(key),list):
                        stages[name]=event[key]
                if event.get('context_entries'):
                    selected={c['chunk_id']:c for c in stages.get('evidence',[]) if c.get('selected')}
                    stages['context']=[{**selected[e['chunk_id']],'included_text':e['included_text'],
                        'truncated':e.get('truncated',False)} for e in event['context_entries']]
                # Retrieval owns lazy tokenizer/store state; serialize it. LLM
                # requests may overlap once the immutable context is complete.
                if event.get('stage')=='generating' and retrieval_held[0]:
                    retrieval_held[0]=False
                    retrieval_lock.release()
            gateway=OpenAICompatibleGateway(max_transient_retries=1)
            class RequestCacheGateway:
                def call(self, method, request, tok):
                    data=dataclasses.asdict(request)
                    data.pop('api_key',None)
                    key=(method,sha(json.dumps(data,sort_keys=True,ensure_ascii=False).encode()))
                    request_hashes.append(key)
                    label='generation' if method=='generate' else 'judge'
                    if key in request_cache:
                        counts[label+'_reused']+=1
                        return request_cache[key]
                    counts[label+'_fresh']+=1
                    value=getattr(gateway,method)(request,tok)
                    request_cache[key]=value
                    return value
                def generate(self, request, tok):
                    return self.call('generate',request,tok)
                def complete_text(self, request, tok):
                    return self.call('complete_text',request,tok)
            wrapped=RequestCacheGateway()
            service=AnswerService(IndexGenerationStore(index_db),store,CachedEmbedding(),CachedReranker(),
                wrapped,secrets,settings,progress_callback=record,
                query_interpreter=QueryInterpreter(analyzer=old.QueryAnalyzer()) if arm=='before' else QueryInterpreter())
            query=RagQuery(text=case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,
                final_top_k=settings.rag_final_top_k)
            start=time.monotonic()
            try:
                retrieval_lock.acquire()
                retrieval_held[0]=True
                answer=(_answer_record(service.answer(query,token)) if cohort=='development' else
                        _retrieval_record(service.retrieve(query,token)))
                if retrieval_held[0]:
                    retrieval_held[0]=False
                    retrieval_lock.release()
                judged=None
                if cohort=='development':
                    judged=GoldAnswerJudge(wrapped,settings,api_key).evaluate(case,answer,token)
                scored=score_case(case,stages,maps[case.case_id],answer=answer)
                row={**scored,'cohort':cohort,'arm':arm,'gold_judgement':judged,
                    'candidate_stages':stages,'events':events,'request_counts':counts,
                    'request_hashes':request_hashes,'duration_ms':round((time.monotonic()-start)*1000)}
            except Exception as exc:
                # The provider exception can contain request data; only the class is printed.
                row={'id':case.case_id,'cohort':cohort,'arm':arm,'error_type':type(exc).__name__,
                     'error_message':str(exc).replace(api_key,'[REDACTED]')[:300],
                     'error_frames':[(f.filename,f.lineno,f.name) for f in traceback.extract_tb(exc.__traceback__)],
                     'request_counts':counts,'events':events,'candidate_stages':stages}
            finally:
                if retrieval_held[0]:
                    retrieval_held[0]=False
                    retrieval_lock.release()
            pair_rows.append(row)
            print(cohort,arm,case.case_id,row.get('gold_judgement',{}),row.get('error_type',''),flush=True)
        return pair_rows
    all_rows=[]
    failed_ids=None
    if retry_failed:
        previous=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
        latest={(r['cohort'],r['id'],r['arm']):r for r in previous}
        failed_ids={r['id'] for r in latest.values() if 'error_type' in r}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(pair,cohort,c) for cohort,d in datasets.items() for c in d.cases
                 if failed_ids is None or c.case_id in failed_ids]
        for f in as_completed(futures):
            rows=f.result()
            all_rows.extend(rows)
            with (OUT/'cases.jsonl').open('a',encoding='utf-8') as stream:
                for row in rows:
                    stream.write(safe_json(row)+'\n')
    assert all(sha(d.path.read_bytes())==d.sha256 for d in datasets.values())
    assert store.active().bundle_id==bundle.bundle_id
    stored=[json.loads(l) for l in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    latest={(r['cohort'],r['id'],r['arm']):r for r in stored}
    dump(OUT/'completion.json',{'n':len(latest),'errors':sum('error_type' in r for r in latest.values()),
        'raw_attempt_rows':len(stored),
        'datasets_unchanged':True,'fresh_query_vectors_this_attempt':len(vectors),
        'requests_all_attempts':{k:sum(r['request_counts'][k] for r in stored) for k in
            ('generation_fresh','generation_reused','judge_fresh','judge_reused')}})


if __name__=='__main__':
    main()
