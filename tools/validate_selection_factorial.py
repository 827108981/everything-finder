"""Frozen candidate replay, eight arms, actual-source-quote scoring; no network."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, fields
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import numpy as np

from tools.selection_factorial import ARMS, select_context
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.rag.models import EvidenceChunk, RagQuery
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch, EvidenceAlternative, EvidencePart
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT / 'reports/rag_support_graph_20261113'
OUT = ROOT / 'reports/rag_selection_factorial_20261113'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf8')


@contextmanager
def no_network():
    old = socket.socket.connect
    attempts = []
    def blocked(self, address):
        attempts.append(str(address))
        raise RuntimeError('Remote/local network disabled in offline selection replay')
    socket.socket.connect = blocked
    try: yield attempts
    finally: socket.socket.connect = old


def mappings_from_cache(raw):
    return [GoldEvidenceMatch(**{**m, 'alternatives': tuple(
        EvidenceAlternative(**{**a, 'parts': tuple(EvidencePart(**p) for p in a['parts'])})
        for a in m['alternatives'])}) for m in raw]


def reconstruct(row, store, conn):
    plan = QueryInterpreter(enhanced=False).interpret(row['question'])
    items = []
    for c in row['candidate_stages']['rerank']:
        chunk = store.load_chunk(conn, c['chunk_id'])
        assert chunk is not None and chunk.text == c['text'], 'indexed text drift'
        members = tuple(c.get('member_chunk_ids') or [chunk.stable_chunk_id])
        item = EvidenceChunk(chunk=chunk, member_chunk_ids=members,
                             sources=tuple(store.load_chunk(conn, cid).source for cid in members))
        for key in ('lexical_rank', 'lexical_score', 'vector_rank', 'vector_score', 'fused_rank', 'fused_score',
                    'rerank_score', 'rerank_score_combined', 'answer_support_score'):
            if c.get(key) is not None: setattr(item, key, c[key])
        item.query_plan_diagnostics = deepcopy(c.get('query_plan_diagnostics', {}))
        items.append(item)
    items = ConstraintMatcher().evaluate(plan, items)
    assert [x.chunk.stable_chunk_id for x in items] == [c['chunk_id'] for c in row['candidate_stages']['constraints']], 'condition order drift'
    return plan, items


def load_local_vectors(bundle, ids):
    """Read fixed-model cached unit embeddings without opening writable Chroma.

    Equal-weight mean of all indexed windows per parent, L2 normalized. These
    are redundancy features only; no new query retrieval/scoring is performed.
    """
    units_path = Path('E:/everythingfinder-db/rag/vector') / bundle.bundle_id / bundle.vector_unit_directory / 'vector_units.db'
    cache_path = Path('E:/everythingfinder-db/rag/embedding_cache.db')
    def ro(path):
        c = sqlite3.connect('file:' + path.as_posix() + '?mode=ro', uri=True)
        c.row_factory = sqlite3.Row; c.execute('PRAGMA query_only=ON'); return c
    vectors, audit = {}, []
    with ro(units_path) as units, ro(cache_path) as cache:
        for cid in sorted(ids):
            rows = units.execute('SELECT * FROM vector_units WHERE parent_chunk_id=? ORDER BY ordinal', (cid,)).fetchall()
            embeddings = []
            for r in rows:
                digest = hashlib.sha256(r['embedding_text'].encode()).hexdigest()
                hit = cache.execute('SELECT vector_blob FROM embedding_cache WHERE model_id=? AND model_revision=? AND dimension=? AND text_sha256=?',
                    (bundle.embedding_repo_id, bundle.embedding_revision, bundle.embedding_dimension, digest)).fetchone()
                if hit is None: raise ValueError('Missing frozen unit embedding: ' + r['unit_id'])
                v = np.frombuffer(hit[0], dtype='<f4').astype(np.float64)
                assert len(v) == bundle.embedding_dimension and np.isfinite(v).all()
                embeddings.append(v)
            if not embeddings: raise ValueError('No indexed vector units for ' + cid)
            mean = np.mean(embeddings, axis=0); mean /= np.linalg.norm(mean)
            vectors[cid] = mean
            audit.append({'chunk_id': cid, 'unit_count': len(rows), 'unit_ids': [r['unit_id'] for r in rows],
                          'vector_sha256': hashlib.sha256(mean.tobytes()).hexdigest()})
    return vectors, {'method': 'normalized mean of all frozen indexed unit embeddings; includes existing source-context prefixes',
                     'model': bundle.embedding_repo_id, 'revision': bundle.embedding_revision,
                     'unit_database': str(units_path), 'unit_database_sha256': sha(units_path), 'parents': audit}


def load_inputs():
    rows = [json.loads(s) for s in (HISTORY / 'A.jsonl').read_text(encoding='utf8').splitlines()]
    s = json.loads((HISTORY / 'A_manifest.json').read_text(encoding='utf8'))['settings']
    rawmaps = json.loads((HISTORY / 'native_gold_mappings.json').read_text(encoding='utf8'))['mappings']
    mappings = {k: mappings_from_cache(v) for k, v in rawmaps.items()}
    return rows, s, rawmaps, mappings


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--baseline-only', action='store_true'); args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, s, rawmaps, mappings = load_inputs()
    store = RagIndexStore('E:/everythingfinder-db/rag'); bundle = store.active()
    assert bundle.bundle_id == 'g000033-71ec3c52f3'
    assert sha(bundle.metadata_path) == '1680e66c214c58e81f3a6206087fe505192863a2d1fd521f34f75ab462ef51f3'
    protected = [* (ROOT/'tests/rag/fixtures').glob('*'), *(ROOT/'local_full_text_search/rag').rglob('*.py'),
                 ROOT/'local_full_text_search/config/defaults.py', SETTINGS_PATH,
                 HISTORY/'A.jsonl', HISTORY/'native_gold_mappings.json', bundle.metadata_path]
    hashes = {str(p): sha(p) for p in protected if p.is_file()}
    protocol = {'bundle': bundle.bundle_id, 'metadata_sha256': sha(bundle.metadata_path), 'cases': len(rows),
        'arms': {k: asdict(v) for k,v in ARMS.items()}, 'context_budget': s['rag_context_max_chars'],
        'fixed_top_k': s['rag_final_top_k'], 'document_cap': s['rag_max_chunks_per_document'],
        'no_remote_calls': True, 'gold_runtime_access': False, 'default_runtime_modified': False,
        'selection_rule': 'A remove doc quota; B .8 normalized upstream rank - .2 maximum cosine to selected; C complete-unit first-fit with native header costs, 64 safety cap',
        'stopping_rule': 'Run all eight once; no quality retry or parameter tuning. Reject adoption on confirmed new required-evidence loss or wrong-scope evidence. Independent acceptance only after a candidate passes development gates.',
        'protected_hashes': hashes}
    freeze = OUT/'protocol.json'
    if freeze.exists():
        old = json.loads(freeze.read_text(encoding='utf8'))
        assert old == protocol, 'frozen inputs or protocol changed'
    else: write(freeze, protocol)
    policy = EvidencePolicy(min_vector_score=s['rag_min_vector_score'], min_rerank_score=s['rag_min_rerank_score'],
                            max_per_document=s['rag_max_chunks_per_document'])
    reproduced, replay = [], {}
    with no_network() as attempts, store.connect_read_only(bundle.metadata_path) as conn:
        for r in rows:
            plan, candidates = reconstruct(r, store, conn)
            q = RagQuery(r['question'], final_top_k=s['rag_final_top_k'])
            decision, ctx, _ = select_context(q, deepcopy(candidates), plan, policy, s['rag_context_max_chars'])
            score = score_stage('context', list(ctx.entries), mappings[r['id']])
            equal = ctx.text == r['context'] and list(ctx.entries) == r['entries'] and score == r['scores']['context']
            reproduced.append({'id': r['id'], 'context_equal': ctx.text == r['context'],
                               'entries_equal': list(ctx.entries) == r['entries'], 'score_equal': score == r['scores']['context']})
            replay[r['id']] = (plan, candidates, q)
        write(OUT/'baseline_reproduction.json', {'cases':len(rows), 'records':reproduced,
             'passed': all(all(v for k,v in x.items() if k != 'id') for x in reproduced), 'network_attempts': attempts})
        assert all(all(v for k,v in x.items() if k != 'id') for x in reproduced), 'baseline failed; do not compare arms'
        print('Baseline actual Context, entries and scores reproduced:', len(rows), flush=True)
        if args.baseline_only: return
        ids = {c.chunk.stable_chunk_id for _, candidates, _ in replay.values() for c in candidates}
        vectors, vector_audit = load_local_vectors(bundle, ids)
        write(OUT/'vector_provenance.json', vector_audit)
        print('Read frozen local vectors:', len(vectors), flush=True)
        for arm, factors in ARMS.items():
            with (OUT/(arm+'.jsonl')).open('w', encoding='utf8') as f:
                for r in rows:
                    plan, candidates, q = replay[r['id']]
                    d, ctx, trace = select_context(q, deepcopy(candidates), plan, policy, s['rag_context_max_chars'], factors, vectors)
                    lookup = {c['chunk_id']: c for c in r['candidate_stages']['constraints']}
                    chosen = [x.chunk.stable_chunk_id for x in d.accepted]
                    reasons = dict(d.rejected)
                    evidence = [{**lookup[cid], 'selected': True, 'reason':'selected'} for cid in chosen]
                    evidence += [{**c, 'selected':False, 'reason':reasons.get(cid,'not_selected')} for cid,c in lookup.items() if cid not in chosen]
                    scores = {**r['scores'], 'evidence':score_stage('evidence', evidence, mappings[r['id']]),
                              'context':score_stage('context', list(ctx.entries), mappings[r['id']])}
                    base = r['scores']['context']; now = scores['context']
                    record = {k:r[k] for k in ('id','question','group_id','cohort','review_status')}
                    record.update(arm=arm, context=ctx.text, entries=list(ctx.entries), excluded_entries=list(ctx.excluded_entries),
                        scores=scores, evidence=evidence, selection_trace=trace, decision={'reason':d.reason,'insufficient':d.insufficient,'rejected':d.rejected},
                        query_plan=asdict(plan), answer_status='not_run_no_api',
                        gained_gold=sorted(set(now['gold_completion_ranks'])-set(base['gold_completion_ranks'])),
                        lost_gold=sorted(set(base['gold_completion_ranks'])-set(now['gold_completion_ranks'])),
                        added_hard_negative=sorted(set(now['hard_negative_ids'])-set(base['hard_negative_ids'])),
                        removed_hard_negative=sorted(set(base['hard_negative_ids'])-set(now['hard_negative_ids'])),
                        context_changed=ctx.text != r['context'],
                        added_chunks=sorted({e['chunk_id'] for e in ctx.entries}-{e['chunk_id'] for e in r['entries']}),
                        removed_chunks=sorted({e['chunk_id'] for e in r['entries']}-{e['chunk_id'] for e in ctx.entries}))
                    f.write(json.dumps(record, ensure_ascii=False)+'\n')
            print('Completed', arm, len(rows), flush=True)
        drift = [p for p,h in hashes.items() if sha(p) != h]
        write(OUT/'execution_audit.json', {'network_attempts':attempts, 'remote_calls':0, 'protected_drift':drift,
              'arms':len(ARMS), 'cases_per_arm':len(rows), 'observations':len(rows)*len(ARMS), 'baseline_reproduced':True})
        assert not drift and not attempts


if __name__ == '__main__': main()
