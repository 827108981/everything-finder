"""Fresh current-config RAG validation; keep pending diagnostics out of scoring.

Runs real local models, without reusing historical answers or rerank scores.
This is a harness only: no production algorithm or app settings are changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR, SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.runner import PreparedEval, RagEvalConfig, RagEvalRunner
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'tests/rag/fixtures'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now() -> str:
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    settings_hash = sha(SETTINGS_PATH)
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    token = CancelToken()
    index_db = Path('E:/everythingfinder-db/search_index.db')
    store = RagIndexStore(index_db.parent / 'rag')
    bundle = store.active()
    main_data = load_gold_dataset(FIXTURES / 'rag_gold_v4_review.json', reviewed_only=False)
    regression = load_gold_dataset(FIXTURES / 'rag_acceptance_v1_review.json', reviewed_only=False)
    assert len(main_data.cases) == 50 and len(regression.cases) == 10
    assert all(c.corpus_snapshot == bundle.bundle_id for d in (main_data, regression) for c in d.cases)
    assert all(c.review_status == 'pending' for c in regression.cases)
    selected = tuple(c for c in main_data.cases if c.review_status == 'reviewed')
    pending = tuple(c for c in main_data.cases if c.review_status == 'pending')
    excluded = tuple(c for c in main_data.cases if c.review_status == 'disputed')
    assert len(selected) == 48 and [c.case_id for c in excluded] == ['rag-biz-007']
    assert [c.case_id for c in pending] == ['rag-biz-024']
    saved_hashes = {str(p): sha(p) for p in (
        FIXTURES / 'rag_gold_v4_review.json', FIXTURES / 'rag_gold_v4.jsonl',
        FIXTURES / 'rag_acceptance_v1_review.json', FIXTURES / 'rag_acceptance_v1.jsonl',
        FIXTURES / 'rag_gold_v3_review.json', FIXTURES / 'rag_gold_v3.jsonl')}
    assert [c.raw for c in main_data.cases] == [c.raw for c in load_gold_dataset(FIXTURES / 'rag_gold_v4.jsonl', reviewed_only=False).cases]
    assert [c.raw for c in regression.cases] == [c.raw for c in load_gold_dataset(FIXTURES / 'rag_acceptance_v1.jsonl', reviewed_only=False).cases]
    api_key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ''

    def dump(path, value):
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
        if api_key:
            text = text.replace(api_key, '[REDACTED]')
        path.write_text(text + '\n', encoding='utf-8')

    def progress(event):
        typ = event.get('type')
        if typ == 'case_finished':
            r = event['result']
            print(json.dumps({'time': now(), 'id': r['id'], 'duration_ms': r['duration_ms'],
                'context_gold': r.get('stages', {}).get('context', {}).get('gold_evidence_found'),
                'error': r.get('error')}, ensure_ascii=False), flush=True)
        elif typ in ('run_started', 'run_finished'):
            print(json.dumps({k: event.get(k) for k in ('type', 'run_dir', 'total', 'state')}, ensure_ascii=False), flush=True)

    public_settings = {k: v for k, v in asdict(settings).items() if k.startswith('rag_')
                       and not any(t in k for t in ('api', 'secret', 'key'))}
    protocol = {'started_at': now(), 'bundle_id': bundle.bundle_id, 'chunk_count': bundle.chunk_count,
        'settings_path': str(SETTINGS_PATH), 'settings_sha256': settings_hash,
        'settings': public_settings, 'llm_model': settings.rag_api_model,
        'api_style': settings.rag_api_style, 'api_credential_present': bool(api_key),
        'input_sha256': saved_hashes, 'formal_ids': [c.case_id for c in selected],
        'pending_main_ids': [c.case_id for c in pending], 'excluded_ids': [c.case_id for c in excluded],
        'regression_pending_ids': [c.case_id for c in regression.cases],
        'cache_policy': 'No historical generation/judge/embedding-query/rerank-score reuse; current local models run afresh against the existing published index.',
        'scope': '48 reviewed formal; 024 and regression 10 pending diagnostics; 007 excluded; production configuration unchanged.',
        'limits': 'Pending diagnostics never enter the formal denominator; retrieval metrics are not answer accuracy.',
        'production_code_sha256': {str(p.relative_to(ROOT)): sha(p)
            for p in (ROOT / 'local_full_text_search/rag').rglob('*.py')},
        'harness_sha256': sha(Path(__file__)),
    }
    dump(out / 'protocol.json', protocol)
    conn = store.connect_read_only(bundle.metadata_path)
    try:
        chunks = load_chunk_rows(conn)
    finally:
        conn.close()
    evidence = {c.case_id: tuple(map_case_evidence(c, chunks)) for d in (main_data, regression) for c in d.cases}
    audit = []
    for d in (main_data, regression):
        for c in d.cases:
            for e in evidence[c.case_id]:
                row = asdict(e)
                row.update(id=c.case_id, review_status=c.review_status,
                           original_file_exists=(Path(e.file) if Path(e.file).is_absolute() else Path('E:/免疫资料') / e.file).is_file())
                audit.append(row)
    dump(out / 'evidence_audit.json', {'bundle_id': bundle.bundle_id, 'entries': audit,
        'counts': dict(Counter((e['kind'] + ':' + e['confidence']) for e in audit)),
        'note': 'Native file existence and current index mapping rechecked; does not claim every source page has been visually re-reviewed.'})
    print('Starting fresh local model loading', flush=True)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    started = time.monotonic()
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
    dump(out / 'model_loading.json', {'duration_ms': int((time.monotonic() - started) * 1000),
                                      'fresh_model_load': True, 'device': 'cpu', 'torch_threads': 4})
    def factory(callback):
        return AnswerService(IndexGenerationStore(index_db), store, embedding, reranker,
            OpenAICompatibleGateway(max_transient_retries=0), WindowsCredentialSecretStore(), settings,
            progress_callback=callback)

    results = {}
    cohorts = [('formal48', main_data, selected), ('pending024', main_data, pending),
               ('regression10_pending', regression, regression.cases)]
    for name, dataset, cases in cohorts:
        runner = RagEvalRunner(index_db, settings, token, progress_callback=progress)
        coverage = {
            'case_count': len(cases), 'dataset_case_count': len(dataset.cases),
            'reviewed_case_count': sum(c.review_status == 'reviewed' for c in cases),
            'positive_evidence_total': sum(len(c.gold_evidence) for c in cases),
            'positive_evidence_located': sum(bool(e.chunk_ids) for c in cases for e in evidence[c.case_id] if e.kind == 'positive'),
            'snapshot_mismatch': False, 'snapshot_override': False, 'current_bundle_id': bundle.bundle_id,
            'behavior_counts': dict(Counter(c.expected_behavior for c in cases)),
            'diagnostic_only': name != 'formal48', 'review_status_preserved': True,
        }
        prepared = PreparedEval(dataset, bundle.bundle_id, cases,
            {c.case_id: evidence[c.case_id] for c in cases}, coverage)
        config = RagEvalConfig(dataset_path=dataset.path, mode='retrieval-only', output_root=out / name)
        result = runner.run(config, prepared, factory)
        results[name] = {'run_dir': result['run_dir'], 'state': result['state'], 'count': len(result['results']),
                         'errors': [r['id'] for r in result['results'] if r.get('error')]}
        dump(out / 'completion.json', {'updated_at': now(), 'cohorts': results, 'generation_status': 'not_yet_attempted'})

    # A real business generation request checks whether the external block remains.
    # One access-denied response prevents repeated identical failures over 59 cases.
    formal_dir = Path(results['formal48']['run_dir'])
    rows = [json.loads(l) for l in (formal_dir / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    r = next(r for r in rows if r['id'] == 'rag-biz-001')
    event = next(e for e in reversed(r['events']) if e.get('llm_context'))
    started = time.monotonic()
    remote = {'case_id': r['id'], 'started_at': now(), 'model': settings.rag_api_model,
              'real_business_context': True, 'transient_retries': 0, 'attempts': 1}
    try:
        if not api_key:
            raise RuntimeError('当前保存的API凭据不可用')
        answer = OpenAICompatibleGateway(max_transient_retries=0).generate(GenerationRequest(
            question=r['question'], context=event['llm_context'], model=settings.rag_api_model,
            api_base_url=settings.rag_api_base_url, api_style=settings.rag_api_style, api_key=api_key,
            timeout_seconds=settings.rag_api_timeout_seconds,
            query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(r['question']))), token)
        remote.update(state='reachable', answer=asdict(answer),
                      next_action='Remote access recovered. Full generation run still required; this single probe is not full validation.')
    except Exception as exc:
        remote.update(state='blocked', error_type=type(exc).__name__, safe_error=str(exc)[:350],
                      skipped_remote_ids=[c.case_id for c in (*selected, *pending, *regression.cases) if c.case_id != r['id']],
                      note='This is an infrastructure block, not a per-case answer-quality failure. No frozen answers substituted.')
    remote['duration_ms'] = int((time.monotonic() - started) * 1000)
    dump(out / 'remote_probe.json', remote)
    assert settings_hash == sha(SETTINGS_PATH)
    assert all(sha(Path(p)) == h for p, h in saved_hashes.items())
    assert store.active().bundle_id == bundle.bundle_id
    dump(out / 'completion.json', {'finished_at': now(), 'cohorts': results,
        'generation_status': remote['state'], 'source_datasets_unchanged': True,
        'app_settings_unchanged': True, 'bundle_unchanged': True,
        'formal_answer_accuracy': None, 'candidate_clarification_business_validation': 'not run; default disabled; remote blocked if probe failed'})
    print(json.dumps({'out': str(out), 'cohorts': results, 'remote': remote['state']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
