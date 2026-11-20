"""Regenerate the HTML review without rerunning retrieval, generation or judging."""
import argparse
import json
import sqlite3
from pathlib import Path

from local_full_text_search.rag.evaluation.reporting.comparison import write_comparison_report


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('reports/rag_business_repair_20261112'))
    parser.add_argument('--manifest', type=Path, default=Path('tests/rag/fixtures/rag_active_suite_v5.json'))
    parser.add_argument('--index-root', type=Path, default=Path('E:/everythingfinder-db/rag'))
    parser.add_argument('--baseline', default='off_all')
    parser.add_argument('--candidate', default='verified_all')
    parser.add_argument('--baseline-answers', default='answers_AB')
    parser.add_argument('--candidate-answers', default='verified_answers_B')
    parser.add_argument('--metrics', default='final_report_metrics')
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding='utf8'))
    cases = [case for entry in manifest['datasets'] for case in read_lines(Path(entry['path']))]
    captures = {arm: {r['id']: r for r in read_lines(args.directory / (name + '.jsonl'))} for arm, name in [('off', args.baseline), ('on', args.candidate)]}
    answers = {(r['arm'], r['id']): r for name, arm in [(args.baseline_answers, 'A'), (args.candidate_answers, 'B')] for r in read_lines(args.directory / (name + '.jsonl')) if r['arm'] == arm}
    metrics = json.loads((args.directory / (args.metrics + '.json')).read_text(encoding='utf8'))
    expected = {c['id'] for c in cases}
    assert all(set(cap) == expected for cap in captures.values()), 'Comparison cohorts differ'
    for cap in captures.values():
        assert all(cap[c['id']]['question'] == c['question'] for c in cases), 'Question changed after capture'
    db_path = args.index_root / ('rag.' + manifest['bundle_id']) / 'metadata.db'
    with sqlite3.connect(db_path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        paths = {json.loads(r[0])['file_path'] for r in db.execute('SELECT source_json FROM rag_chunks')}
    print(json.dumps(write_comparison_report(args.directory, cases, captures, answers, metrics, paths)))


if __name__ == '__main__':
    main()
