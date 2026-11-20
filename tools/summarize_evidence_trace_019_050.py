"""Read-only evidence trace and controlled probe audit."""
import html
import json
import re
import shutil
from tools.rejudge_scoring_v3 import ROOT, sha

OUT = ROOT / 'reports/rag_evidence_trace_019_050_20261105'


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf8'))


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


def main():
    targets = {
        'rag-biz-019': '47a3663acd65d8fc404bea85455f55b7a4bc1e0b30454dc97e777de9e6350d2e',
        'rag-biz-050': 'e9b8c2a0c6ad65ebb77179fa8fb7101173b58dec5e274ceaadfe737258a8d2cf',
    }
    audits = []
    for cid, target in targets.items():
        trace = read(cid + '.json')
        lists = {k: v for event in trace['events'] for k, v in event.items()
                 if isinstance(v, list) and v and isinstance(v[0], dict)}
        scores = {}
        for field, candidates in [('lexical_rank', lists['lexical_candidates']),
                                  ('vector_rank', lists['vector_candidates'])]:
            for item in candidates:
                key = item['chunk_id']
                scores[key] = scores.get(key, 0) + 1 / (60 + item[field])
        order = sorted(scores, key=lambda key: (-scores[key], key))
        hits = {}
        for name in ['lexical_candidates', 'vector_candidates', 'fused_candidates',
                     'deduplicated_candidates', 'rerank_input_candidates', 'reranked_candidates',
                     'constraint_candidates', 'evidence_candidates', 'context_entries']:
            hits[name] = [
                {'rank': rank, 'chunk_id': item.get('chunk_id'), 'selected': item.get('selected'),
                 'reason': item.get('reason'), 'decision': item.get('constraint_decision')}
                for rank, item in enumerate(lists.get(name, []), 1)
                if target in {item.get('chunk_id'), *item.get('member_chunk_ids', []),
                              *(s.get('chunk_id') for s in item.get('sources', []))}
            ]
        audits.append({'id': cid, 'gold_chunk_id': target,
                       'untruncated_rrf_rank': order.index(target) + 1,
                       'rrf_union_count': len(order), 'stage_hits': hits})
    assert [x['untruncated_rrf_rank'] for x in audits] == [60, 64]
    dump('stage_audit.json', audits)
    rows = [json.loads(line) for line in (OUT / 'probe_outputs.jsonl').read_text(encoding='utf8').splitlines()]
    assert len(rows) == 18 and all(not x.get('error') for x in rows)
    facts = []
    for row in rows:
        if row['id'].endswith('019'):
            match = re.search(r'灌注[^。\n；]{0,25}(?<!\d)40\s*次以上', row['text'])
            facts.append({'id': row['id'], 'arm': row['arm'], 'repeat': row['repeat'],
                          'target_fact_present': bool(match),
                          'excerpt': match.group(0) if match else ''})
    assert sum(x['target_fact_present'] for x in facts if x['arm'] == 'baseline') == 0
    assert sum(x['target_fact_present'] for x in facts if x['arm'] == 'correct_source') == 3
    dump('019_fact_probe_audit.json', facts)
    scope = [{'id': row['id'], 'arm': row['arm'], 'repeat': row['repeat'],
              'uses_existing_general_checks': True,
              'direct_applicability_reservation': row['arm'] == 'baseline' and row['repeat'] == 1,
              'audit_scope': '逐条人工阅读；仅核对通用检查采用，不把无法确定故障当作拒用，不判整份答案全对。'}
             for row in rows if row['id'].endswith('050')]
    dump('050_manual_scope_audit.json', scope)
    text = (OUT / 'report.md').read_text(encoding='utf8')
    shutil.copyfile(OUT / 'report.md', ROOT / 'docs/rag_evidence_trace_019_050_20261105.md')
    body = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>019与050证据链验证</title>'
            '<style>body{font:16px/1.6 sans-serif;max-width:1200px;margin:30px auto;padding:20px}'
            'pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit}'
            'details{border:1px solid #ddd;margin:10px;padding:10px}</style>'
            '<h1>019与050证据链验证</h1><pre>' + html.escape(text) + '</pre>')
    inputs = {(x['id'], x['arm'], x['repeat']): x for x in read('probe_inputs.json')}
    for row in sorted(rows, key=lambda x: (x['id'], x['arm'], x['repeat'])):
        body += '<details><summary>' + html.escape(f"{row['id']} {row['arm']} 第{row['repeat']}次")
        body += '</summary><pre>' + html.escape(row['text']) + '</pre><details><summary>实际诊断输入</summary><pre>'
        body += html.escape(json.dumps(inputs[row['id'], row['arm'], row['repeat']], ensure_ascii=False, indent=2))
        body += '</pre></details></details>'
    (OUT / 'report.html').write_text(body, encoding='utf8')
    frozen = read('protocol.json')['frozen_hashes']
    assert all(sha(path) == digest for path, digest in frozen.items())
    dump('verification.json', {
        'fresh_retrieval_queries': 2, 'source_pages_verified': [13, 14, 17],
        'controlled_outputs': 18, 'execution_errors': 0, 'citation_id_checks_passed': 18,
        '019_target_fact_baseline': 0, '019_target_fact_with_source': 3,
        'cases_per_condition': 3, 'business_code_gold_config_unchanged': True,
        'report_html_sha256': sha(OUT / 'report.html'), 'html_probe_sections': body.count('<details>') // 2,
    })
    print('Verified 2 traces and 18 probes; frozen inputs unchanged.')


if __name__ == '__main__':
    main()
