"""Fresh full RAG run using the current app configuration and reviewed v4 Gold.

Local inference is serialized; remote generation and judging may overlap.
Pending diagnostics retain their statuses and never enter the formal score.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import threading
import time
from zoneinfo import ZoneInfo

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR, SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import aggregate, score_case
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.evaluation.transitions import build_candidate_transitions
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=3)
    args = p.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings_hash = sha(SETTINGS_PATH)
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ''
    assert key and settings.rag_api_model
    index_db = Path('E:/everythingfinder-db/search_index.db')
    store = RagIndexStore(index_db.parent / 'rag')
    bundle = store.active()
    fixtures = ROOT / 'tests/rag/fixtures'
    main_gold = load_gold_dataset(fixtures / 'rag_gold_v4_review.json', reviewed_only=False)
    regression = load_gold_dataset(fixtures / 'rag_acceptance_v1_review.json', reviewed_only=False)
    assert len(main_gold.cases) == 50 and len(regression.cases) == 10
    formal = [c for c in main_gold.cases if c.review_status == 'reviewed']
    pending = [c for c in main_gold.cases if c.review_status == 'pending']
    excluded = [c for c in main_gold.cases if c.review_status == 'disputed']
    assert len(formal) == 48 and [c.case_id for c in pending] == ['rag-biz-024']
    assert [c.case_id for c in excluded] == ['rag-biz-007']
    assert all(c.review_status == 'pending' for c in regression.cases)
    tasks = [('formal48', c) for c in formal] + [('pending024', c) for c in pending] + [('regression10_pending', c) for c in regression.cases]
    assert all(c.corpus_snapshot == bundle.bundle_id for _, c in tasks)
    input_hashes = {str(path): sha(path) for path in (main_gold.path, regression.path,
        fixtures / 'rag_gold_v4.jsonl', fixtures / 'rag_acceptance_v1.jsonl')}
    assert [c.raw for c in main_gold.cases] == [c.raw for c in load_gold_dataset(fixtures / 'rag_gold_v4.jsonl', reviewed_only=False).cases]
    assert [c.raw for c in regression.cases] == [c.raw for c in load_gold_dataset(fixtures / 'rag_acceptance_v1.jsonl', reviewed_only=False).cases]

    def safe(value, indent=None):
        return json.dumps(value, ensure_ascii=False, indent=indent, default=str).replace(key, '[REDACTED]')

    def dump(path, value):
        path.write_text(safe(value, 2) + '\n', encoding='utf-8')

    protocol = {'started_at': now(), 'mode': 'full', 'bundle_id': bundle.bundle_id,
        'chunk_count': bundle.chunk_count, 'settings_sha256': settings_hash,
        'llm_model': settings.rag_api_model, 'api_style': settings.rag_api_style,
        'settings': {k: v for k, v in asdict(settings).items() if k.startswith('rag_') and not any(t in k for t in ('api', 'secret', 'key'))},
        'input_sha256': input_hashes, 'formal_ids': [c.case_id for c in formal],
        'pending_main_ids': [c.case_id for c in pending], 'excluded_ids': [c.case_id for c in excluded],
        'regression_pending_ids': [c.case_id for c in regression.cases],
        'production_code_sha256': {str(f.relative_to(ROOT)): sha(f) for f in (ROOT / 'local_full_text_search/rag').rglob('*.py')},
        'harness_sha256': sha(__file__), 'workers': args.workers, 'local_inference_concurrency': 1,
        'cache_policy': 'All query embeddings, rerank scores, generated answers and semantic judgements computed afresh. Existing published index vectors used normally.',
        'retry_policy': 'Up to 3 transient transport retries, matching batch evaluator; Gold judge can reask malformed JSON once. No retry chosen from quality verdicts.',
        'scoring': '48 reviewed formal, 11 pending diagnostic, 007 excluded. Model judge advisory, not human accuracy.',
        'generation_and_judge_use_gold': 'Generator sees only question and retrieved context. Gold checks are passed only to the separate evaluator.',
    }
    dump(out / 'protocol.json', protocol)
    conn = store.connect_read_only(bundle.metadata_path)
    try:
        chunks = load_chunk_rows(conn)
    finally:
        conn.close()
    maps = {c.case_id: map_case_evidence(c, chunks) for _, c in tasks}
    dump(out / 'evidence_audit.json', {'bundle_id': bundle.bundle_id,
        'entries': [{**asdict(e), 'id': cid} for cid, es in maps.items() for e in es]})
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    print('Loading actual local models for fresh full evaluation', flush=True)
    t = time.monotonic()
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(
        settings.rag_embedding_repo_id, settings.rag_embedding_revision, 'embedding')),
        repo_id=settings.rag_embedding_repo_id, revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size)
    reranker = LocalBgeReranker(manager.resolve_local_path(ModelSpec(
        settings.rag_reranker_repo_id, settings.rag_reranker_revision, 'reranker')),
        repo_id=settings.rag_reranker_repo_id, revision=settings.rag_reranker_revision,
        batch_size=settings.rag_reranker_batch_size, max_length=settings.rag_reranker_max_length,
        cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,
        retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,
        answer_support_weight=settings.rag_rerank_answer_support_weight,
        max_rank_drop=settings.rag_rerank_max_rank_drop)
    dump(out / 'model_loading.json', {'duration_ms': round((time.monotonic() - t) * 1000), 'device': 'cpu', 'torch_threads': 4})
    lock = threading.Lock()
    access_denied = threading.Event()
    stage_fields = {'lexical_candidates': 'lexical', 'vector_candidates': 'vector',
        'fused_candidates': 'fusion', 'deduplicated_candidates': 'deduplicated',
        'reranked_candidates': 'rerank', 'constraint_candidates': 'constraints', 'evidence_candidates': 'evidence'}

    def work(cohort, case):
        if access_denied.is_set():
            return {'id': case.case_id, 'cohort': cohort, 'review_status': case.review_status,
                    'formal_scored': cohort == 'formal48', 'execution_status': 'blocked_not_started'}
        queued = time.monotonic()
        lock.acquire()
        held = [True]
        start = time.monotonic()
        queue_ms = round((start - queued) * 1000)
        events, stages = [], {}
        phase = ['generation']
        counts = Counter()
        token = CancelToken()
        previous = [start]

        def record(payload):
            current = time.monotonic()
            for field, stage in stage_fields.items():
                if isinstance(payload.get(field), list):
                    stages[stage] = [dict(x) for x in payload[field]]
            event = {k: v for k, v in payload.items() if k not in stage_fields}
            event.update(elapsed_ms=round((current - start) * 1000),
                         since_previous_event_ms=round((current - previous[0]) * 1000))
            previous[0] = current
            events.append(event)
            if payload.get('stage') in ('generating', 'assessing_clarification') and held[0]:
                held[0] = False
                lock.release()
            if payload.get('stage') == 'assessing_clarification':
                phase[0] = 'clarification'
            elif payload.get('stage') == 'generating':
                phase[0] = 'generation'
            elif payload.get('stage') == 'reviewing':
                phase[0] = 'review'

        class Gateway(OpenAICompatibleGateway):
            def generate(self, request, tok):
                counts['generation_calls'] += 1
                return super().generate(request, tok)

            def complete_text(self, request, tok):
                counts[phase[0] + '_completion_calls'] += 1
                return super().complete_text(request, tok)

            def review_answer(self, request, tok):
                counts['review_calls'] += 1
                return super().review_answer(request, tok)

        def retry(event):
            counts[phase[0] + '_transport_retries'] += 1
            record({**event, 'stage': phase[0] + '_retry'})

        gateway = Gateway(max_transient_retries=3, retry_callback=retry)
        answer, judgement = None, None
        error = None
        try:
            service = AnswerService(IndexGenerationStore(index_db), store, embedding, reranker,
                gateway, WindowsCredentialSecretStore(), settings, progress_callback=record)
            answer = _answer_record(service.answer(RagQuery(text=case.question, include_ocr=True,
                min_ocr_confidence=settings.ocr_min_confidence, lexical_top_k=settings.rag_lexical_top_k,
                vector_top_k=settings.rag_vector_top_k, fusion_top_k=settings.rag_fusion_top_k,
                reranker_top_k=settings.rag_reranker_top_k, final_top_k=settings.rag_final_top_k), token))
        except Exception as exc:
            error = {'stage': phase[0], 'type': type(exc).__name__, 'message': str(exc)[:400]}
            if '403' in str(exc) or '401' in str(exc):
                access_denied.set()
        finally:
            if held[0]:
                held[0] = False
                lock.release()
        if answer is not None:
            phase[0] = 'gold_judge'
            record({'stage': 'gold_judging', 'message': '独立Gold语义判分'})
            try:
                judgement = GoldAnswerJudge(gateway, settings, key).evaluate(case, answer, token)
            except Exception as exc:
                judgement = {'verdict': 'unavailable', 'method': 'llm_gold_judge',
                             'error_type': type(exc).__name__, 'reason': str(exc)[:400]}
                if '403' in str(exc) or '401' in str(exc):
                    access_denied.set()
            record({'stage': 'gold_judged', 'verdict': judgement['verdict']})
        ctx = next((e for e in reversed(events) if e.get('context_entries')), None)
        if ctx:
            selected = {x['chunk_id']: x for x in stages.get('evidence', []) if x.get('selected') is True}
            stages['context'] = [{**selected[e['chunk_id']], 'included_text': e['included_text'],
                'truncated': bool(e.get('truncated')), 'context_rank': i + 1}
                for i, e in enumerate(ctx['context_entries']) if e['chunk_id'] in selected]
        scored = score_case(case, stages, maps[case.case_id], answer=answer or {'mode': 'full'})
        row = {**scored, 'cohort': cohort, 'review_status': case.review_status,
            'formal_scored': cohort == 'formal48', 'started_at': datetime.fromtimestamp(time.time() - (time.monotonic() - start), ZoneInfo('Asia/Shanghai')).isoformat(),
            'duration_ms': round((time.monotonic() - start) * 1000), 'local_queue_wait_ms': queue_ms,
            'candidate_stages': stages, 'candidate_transitions': build_candidate_transitions(stages, events, reranker_top_k=settings.rag_reranker_top_k),
            'events': events, 'gold_judgement': judgement, 'request_counts': dict(counts),
            'execution_status': 'answer_error' if error else 'judge_error' if judgement and judgement['verdict'] == 'unavailable' else 'completed'}
        if error:
            row['error'] = error
        return row

    all_rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(work, cohort, c): c.case_id for cohort, c in tasks}
        for future in as_completed(futures):
            row = future.result()
            all_rows.append(row)
            with (out / 'cases.jsonl').open('a', encoding='utf-8') as f:
                f.write(safe(row) + '\n')
            print(safe({'time': now(), 'id': row['id'], 'cohort': row['cohort'],
                'state': row['execution_status'], 'verdict': (row.get('gold_judgement') or {}).get('verdict'),
                'seconds': row.get('duration_ms', 0) / 1000}), flush=True)
            dump(out / 'progress.json', {'updated_at': now(), 'finished': len(all_rows),
                'total': len(tasks), 'states': dict(Counter(r['execution_status'] for r in all_rows))})
    assert len(all_rows) == 59 and len({r['id'] for r in all_rows}) == 59
    assert sha(SETTINGS_PATH) == settings_hash
    assert all(sha(path) == h for path, h in input_hashes.items())
    assert all(sha(ROOT / path) == h for path, h in protocol['production_code_sha256'].items())
    assert store.active().bundle_id == bundle.bundle_id
    summaries = {cohort: aggregate([r for r in all_rows if r['cohort'] == cohort and r['execution_status'] != 'blocked_not_started'])
                 for cohort in ('formal48', 'pending024', 'regression10_pending')}
    dump(out / 'retrieval_summary.json', summaries)
    dump(out / 'completion.json', {'finished_at': now(), 'total': 59,
        'states': dict(Counter(r['execution_status'] for r in all_rows)), 'bundle_id': bundle.bundle_id,
        'settings_and_source_datasets_unchanged': True, 'production_code_unchanged': True,
        'historical_answers_or_scores_reused': False, 'pending_statuses_unchanged': True,
        'current_semantic_clarification_enabled': settings.rag_semantic_clarification_enabled,
        'formal_count': 48, 'diagnostic_count': 11, 'excluded_ids': ['rag-biz-007']})
    print(safe(json.loads((out / 'completion.json').read_text(encoding='utf-8'))), flush=True)


if __name__ == '__main__':
    main()
