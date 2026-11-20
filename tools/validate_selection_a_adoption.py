"""Verify production selection against the frozen A experiment, without API calls."""
from copy import deepcopy
from dataclasses import asdict
import json

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.answer_service import AnswerService
from tools.validate_selection_factorial import (
    ROOT, OUT as EXPERIMENT, RagIndexStore, RagQuery, SETTINGS_PATH,
    load_inputs, reconstruct, no_network, score_stage, sha, write,
)

OUT = ROOT / 'reports/rag_selection_a_adoption_20261113'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, settings, _, mappings = load_inputs()
    expected = {r['id']: r for r in map(json.loads, (EXPERIMENT/'A.jsonl').read_text(encoding='utf8').splitlines())}
    store = RagIndexStore('E:/everythingfinder-db/rag'); bundle = store.active()
    protocol = json.loads((EXPERIMENT/'protocol.json').read_text(encoding='utf8'))
    assert bundle.bundle_id == protocol['bundle']
    assert sha(bundle.metadata_path) == protocol['metadata_sha256']
    protected = {p: h for p, h in protocol['protected_hashes'].items()
                 if 'fixtures' in p or p == str(SETTINGS_PATH)}
    assert all(sha(p) == h for p, h in protected.items()), 'Gold/settings drift'
    service = AnswerService(None, store, None, None, None, None, AppSettings.from_dict(settings))
    assert service.evidence_policy.max_per_document is None
    checks = []; actual = []
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        for row in rows:
            plan, candidates = reconstruct(row, store, conn)
            decision = service.evidence_policy.evaluate(
                RagQuery(row['question'], final_top_k=settings['rag_final_top_k']), deepcopy(candidates), plan=plan)
            context = service.context_builder.build(decision.accepted, max_chars=settings['rag_context_max_chars'])
            score = score_stage('context', list(context.entries), mappings[row['id']])
            old = expected[row['id']]
            checks.append({'id': row['id'], 'context_equal': context.text == old['context'],
                           'entries_equal': list(context.entries) == old['entries'],
                           'scores_equal': score == old['scores']['context']})
            actual.append({**old, 'arm': 'production_A', 'context': context.text,
                           'entries': list(context.entries), 'query_plan': asdict(plan),
                           'decision': {'reason': decision.reason, 'insufficient': decision.insufficient,
                                        'rejected': decision.rejected}})
    passed = all(all(v for k, v in c.items() if k != 'id') for c in checks)
    write(OUT/'production_replay.json', {'passed': passed, 'cases': len(checks), 'records': checks,
          'network_attempts': attempts, 'bundle_id': bundle.bundle_id, 'api_calls': 0,
          'method': 'frozen upstream candidates; production AnswerService policy and ContextBuilder',
          'full_retrieval_rerun': False, 'document_quota': None, 'final_top_k': settings['rag_final_top_k'],
          'context_max_chars': settings['rag_context_max_chars'], 'gold_and_settings_unchanged':
              all(sha(p) == h for p, h in protected.items())})
    (OUT/'production_A.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in actual), encoding='utf8')
    assert passed and not attempts
    print(f'Production A identical to experiment A: {len(checks)}/{len(checks)}; API calls: 0')


if __name__ == '__main__':
    main()
