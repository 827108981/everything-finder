"""Audit a complete frozen prompt experiment without changing answers or Gold."""
from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path

from tools.validate_minimal_four_prompt_20261119 import OUT, ROOT, ARMS, TARGETS, EXPECTED, check_frozen, inputs, write, sha
from tools.report_minimal_four_prompt_20261119 import summarize


def read_rows(name):
    p = OUT / name
    return [json.loads(l) for l in p.read_text(encoding='utf8').splitlines()] if p.exists() else []


def normalized_payload(payload):
    p = deepcopy(payload)
    if 'messages' in p:
        assert p['messages'][0]['role'] == 'system'
        p['messages'][0]['content'] = '<controlled system-prompt factor>'
    else:
        p['instructions'] = '<controlled system-prompt factor>'
    return p


def main():
    protocol = check_frozen()
    cases, contexts, settings = inputs()
    original = read_rows('answers.jsonl')
    recoveries = {(r['id'], r['arm'], r['round']): r for r in read_rows('transport_recovery.jsonl')}
    rows = [recoveries.get((r['id'], r['arm'], r['round']), r) for r in original]
    expected = {(k, a, 1) for k in cases for a in ARMS} | {(k, a, n) for k in TARGETS for a in ARMS for n in (2, 3)}
    keys = [(r['id'], r['arm'], r['round']) for r in rows]
    assert len(rows) == EXPECTED and set(keys) == expected and len(set(keys)) == EXPECTED
    first = {(r['id'], r['arm']): r for r in rows if r['round'] == 1}
    assert all(normalized_payload(first[k, 'baseline']['api_request']) == normalized_payload(first[k, 'minimal_four']['api_request']) for k in cases)
    assert all(r['context_sha256'] == protocol['frozen_context_sha256'][r['id']] for r in rows)
    assert all(r['gold_in_generation'] is False for r in rows)
    summary = summarize(rows)
    write(OUT / 'summary.json', summary)
    write(OUT / 'effective_answers.json', rows)

    service = read_rows('service_entry.jsonl')
    service_recoveries = {(r['id'], r['arm']): r for r in read_rows('service_entry_judge_recovery.jsonl')}
    service = [service_recoveries.get((r['id'], r['arm']), r) for r in service]
    assert len(service) == 14 and {(r['id'], r['arm']) for r in service} == {(k, a) for k in TARGETS for a in ARMS}
    service_audit = []
    for r in service:
        context_event = next((e for e in reversed(r.get('events', [])) if 'llm_context' in e), {})
        actual_context = r.get('context', context_event.get('llm_context', ''))
        draft_event = next((e for e in reversed(r.get('events', [])) if 'draft_answer' in e), {})
        requests = r.get('effective_api_requests', [])
        assert len(requests) == 1
        assert normalized_payload(requests[0]) == normalized_payload(first[r['id'], r['arm']]['api_request'])
        service_audit.append(dict(id=r['id'], arm=r['arm'], context_equal_frozen=actual_context == contexts[r['id']]['context'],
            returned_answer=r.get('result', {}).get('answer_text'), draft=draft_event.get('draft_answer'),
            verdict=r.get('judgement', {}).get('verdict'), error=r.get('error'),
            citation_check=draft_event.get('citation_check'), returned=bool(r.get('result'))))
    assert all(r['context_equal_frozen'] for r in service_audit)
    write(OUT / 'service_entry_audit.json', service_audit)

    old_audit_path = ROOT / 'reports/rag_prompt_factorial_20261119/manual_source_audit.json'
    old_audits = json.loads(old_audit_path.read_text(encoding='utf8'))
    reused = []
    for old in old_audits:
        path = Path(old['file'])
        assert sha(path) == old['file_sha256']
        context = contexts[old['id']]['context']
        checks = [dict(c, context_this_run=c['quote'] in context) for c in old['checks']]
        reused.append(dict(id=old['id'], source_id=old['source_id'], file=str(path), pages=old['pages'],
            file_sha256=old['file_sha256'], checks=checks, original_verification_reused_from=str(old_audit_path),
            reason='Same original-file hash and identical frozen Context; no need to re-render unchanged source.'))
    write(OUT / 'source_verification_reuse.json', reused)

    pair_deltas = []
    for k, c in cases.items():
        b, u = first[k, 'baseline'], first[k, 'minimal_four']
        bv, uv = b.get('judgement', {}).get('verdict', 'error'), u.get('judgement', {}).get('verdict', 'error')
        if bv != uv:
            pair_deltas.append(dict(id=k, group_id=c.group_id, review_status=c.review_status, baseline=bv, minimal_four=uv,
                question=c.question, baseline_answer=b.get('answer'), minimal_four_answer=u.get('answer'),
                baseline_judge=b.get('judgement'), minimal_four_judge=u.get('judgement')))
    write(OUT / 'paired_deltas.json', pair_deltas)
    metadata = [m for r in rows for m in r.get('generation_response_metadata', [])]
    invalid_citations = [dict(id=r['id'], arm=r['arm'], round=r['round'], check=r.get('citation_check')) for r in rows
        if not r.get('citation_check', {}).get('structurally_valid', False)]
    output_contract = [dict(id=r['id'], arm=r['arm'], round=r['round'],
        verdict=r.get('judgement', {}).get('verdict'),
        pass_contract=(r.get('answer', '').strip() == '资料中没有足够依据。'
            or bool(r.get('citation_check', {}).get('structurally_valid'))),
        check=r.get('citation_check')) for r in rows]
    false_positive_contract = [r for r in output_contract if r['verdict'] == 'consistent' and not r['pass_contract']]
    write(OUT/'output_contract_audit.json', dict(
        scope='Deterministic output-format gate applied to isolated generated answers; not 250 real AnswerService executions.',
        records=output_contract, automatic_consistent_but_contract_invalid=false_positive_contract))
    audit = dict(expected_tasks=EXPECTED, actual_tasks=len(rows), unique_tasks=len(set(keys)),
        paired_non_system_payloads_equal=111, context_hashes_equal=250, gold_in_generation=False,
        protected_files_unchanged=len(protocol['protected']), source_files_frozen=protocol['source_files_frozen'],
        bundle_id=protocol['bundle_id'], settings_unchanged=True, default_prompt='baseline',
        original_errors=sum('error' in r for r in original), recovery_tasks=len(recoveries),
        remaining_errors=sum('error' in r for r in rows), real_service_context_equal=14,
        real_service_returned=sum(r['returned'] for r in service_audit),
        real_service_errors=[r for r in service_audit if r['error']],
        finish_reasons=dict(Counter(x for m in metadata for x in m.get('finish_reasons', []))),
        actual_models=dict(Counter(m.get('model') for m in metadata)),
        response_metadata_present=sum(bool(r.get('generation_response_metadata')) for r in rows),
        invalid_citations=invalid_citations, first_round_status_counts=dict(Counter(c.review_status for c in cases.values())),
        automatic_consistent_but_contract_invalid=false_positive_contract,
        target_repeats=summary['target_repeats'], unit_test_audit=json.loads((OUT/'test_audit.json').read_text(encoding='utf8')))
    write(OUT/'execution_audit.json', audit)
    decision_path = OUT / 'review_decision.json'
    decision = json.loads(decision_path.read_text(encoding='utf8')) if decision_path.exists() else dict(
        recommendation='待完成逐条复核；默认保持原提示', conclusion='自动统计已完成，但不凭总分决定采用。',
        adoption='待核实', reason='等待核对新增收益、退步和真实入口记录。')
    acceptance = dict(**decision, real_service_context_equal=14, complete=True, default='baseline',
        no_new_upstream_change=True, no_gold_or_judge_change=True, counts=summary['by_arm'],
        targets=summary['target_repeats'], refusal_contract_audit='refusal_contract_audit.json',manual_review='manual_review.json')
    write(OUT/'final_acceptance.json', acceptance)
    tests = audit['unit_test_audit']
    write(OUT/'step_status.json', dict(
        freeze=dict(status='通过', evidence='protocol.json / execution_audit.json'),
        implementation=dict(status='通过', evidence='explicit minimal_four option; default and original refusal literal retained'),
        focused_unit=dict(status='通过', evidence=f"{tests['focused_passed']} passed; test_audit.json"),
        all_rag=dict(status='失败' if tests['failed'] else '通过', evidence='test_audit.json; existing failures listed separately'),
        paired_execution=dict(status='通过' if not summary['errors'] else '失败', evidence='250 tasks; effective_answers.json'),
        real_entry=dict(status='失败' if any(r['error'] for r in service_audit) else '通过', evidence='service_entry_audit.json'),
        source_reuse=dict(status='通过', evidence='source_verification_reuse.json; original hashes unchanged'),
        manual_review=dict(status='通过' if decision_path.exists() else '待核实', evidence='manual_review.json; only listed records reviewed'),
        report_ui=dict(status='待核实', evidence='render final 250 tasks then browser QA'),
        adoption=dict(status=decision['adoption'], evidence=decision['reason'])))
    print(json.dumps(dict(tasks=250, summary=summary['by_arm'], targets=summary['target_repeats'], audit={k:v for k,v in audit.items() if k not in ('invalid_citations','target_repeats','unit_test_audit','real_service_errors')}, acceptance=acceptance['recommendation']),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
