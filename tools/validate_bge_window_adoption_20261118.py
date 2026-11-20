"""Verify the adopted constructor default against all frozen window-max results."""
from pathlib import Path
from dataclasses import asdict
import json
from tools import validate_bge_windows_20261118 as experiment
from tools.validate_selection_factorial import reconstruct, no_network, write, sha
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence

OUT=experiment.ROOT/'reports/rag_bge_window_adoption_20261118'
HISTORY=experiment.OUT

def main():
    OUT.mkdir(exist_ok=True)
    store,bundle,settings=experiment.inputs()
    cases=experiment.load_cases(True)
    rows=experiment.read_rows(experiment.BASE)
    expected={r['id']:r for r in experiment.read_rows(HISTORY/'window_max_native.jsonl')}
    checks=[]
    with no_network() as attempts,store.connect_read_only(bundle.metadata_path) as conn:
        chunkrows=load_chunk_rows(conn)
        mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker'))
        ranker=experiment.CachedRanker(path,repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,max_length=settings.rag_reranker_max_length)
        assert ranker.input_mode=='window_max'
        ranker.cache=json.loads((HISTORY/'token_scores.json').read_text(encoding='utf8'))
        ranker.new_pairs=ranker.cache_hits=0
        for row in rows:
            plan,items=reconstruct(row,store,conn)
            for x in items:x.constraint_decision=None;x.constraint_score=0;x.rerank_diagnostics={}
            ordered=ranker.rerank(row['question'],items,20,CancelToken())
            r=experiment.finish(row,plan,ordered,'window_max',mappings[row['id']],settings)
            e=expected[row['id']]
            c=dict(id=row['id'],context_equal=r['context']==e['context'],entries_equal=r['entries']==e['entries'],scores_equal=r['scores']==e['scores'],
                stage_orders_equal=all([x['chunk_id'] for x in r['candidate_stages'][s]]==[x['chunk_id'] for x in e['candidate_stages'][s]] for s in ('rerank','constraints','evidence')))
            checks.append(c)
        assert ranker.new_pairs==0 and all(all(v for k,v in c.items() if k!='id') for c in checks)
    write(OUT/'production_replay.json',dict(cases=111,checks=checks,api_calls=0,network_attempts=attempts,constructor_default='window_max',
        method='Current default constructor; exact-token frozen FP32 scores; production downstream. No full retrieval rerun.'))
    experiment.OUT=OUT
    experiment.native_probe(use_default=True,expected_path=HISTORY/'window_max_native.jsonl')
    protected=json.loads((HISTORY/'protocol.json').read_text(encoding='utf8'))['protected']
    drift=[p for p,h in protected.items() if sha(p)!=h]
    assert not drift
    write(OUT/'adoption.json',dict(status='adopted',constructor_default='window_max',window_overlap=64,max_length=512,
        replay_cases=111,real_entry_cases=4,protected_drift=drift,api_calls=0,previous_experiment=str(HISTORY),
        unchanged=['Gold','query','BM25','vector','fusion','candidate top20','condition matching','final top8','context budget','persistent settings'],
        caveat='Existing packaged/running executables are not rebuilt or hot-reloaded by source changes.'))
    print('ADOPTED: default replay 111/111; real entry 4/4; API 0',flush=True)

if __name__=='__main__':main()
