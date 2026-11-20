"""Exhaustive cosine audit of the corrected questions, without changing retrieval."""
import hashlib
import json
from collections import defaultdict
import numpy as np
from tools.retrieve_repaired_material_20261116 import OUT
from tools.diagnose_vector_recall import ro
from tools.validate_selection_factorial import ROOT, no_network, write
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence, load_chunk_rows
from local_full_text_search.rag.evaluation.stage_scoring import score_stage


def main():
    import torch
    torch.set_num_threads(3)
    from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
    from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    protocol=json.loads((ROOT/'reports/rag_vector_diagnosis_20261116/protocol.json').read_text(encoding='utf8'))
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases}
    with ro('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db') as c:chunks=load_chunk_rows(c)
    with ro(protocol['vector_index']['database_path']) as c:units=[dict(r) for r in c.execute('select * from vector_units order by unit_id')]
    ids=list(dict.fromkeys(u['parent_chunk_id'] for u in units));index={p:i for i,p in enumerate(ids)}
    parents=np.array([index[u['parent_chunk_id']] for u in units]);matrix=np.empty((len(units),1024),np.float32)
    with ro('E:/everythingfinder-db/rag/embedding_cache.db') as c:
        for i,u in enumerate(units):
            r=c.execute('select vector_blob from embedding_cache where model_id=? and model_revision=? and dimension=1024 and text_sha256=?',
                (protocol['model'],protocol['revision'],hashlib.sha256(u['embedding_text'].encode()).hexdigest())).fetchone()
            assert r is not None
            matrix[i]=np.frombuffer(r[0],dtype='<f4')
    records=[json.loads(x) for x in (OUT/'fresh_retrieval.jsonl').read_text(encoding='utf8').splitlines()]
    with no_network() as attempts:
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(protocol['model'],protocol['revision'],'embedding'))
        emb=LocalBgeEmbeddingProvider(path,repo_id=protocol['model'],revision=protocol['revision'],batch_size=8)
        queries=[next(e['query_plan']['semantic_query'] for e in r['events'] if 'query_plan' in e) for r in records]
        vectors=emb.embed_documents([emb.QUERY_PREFIX+q for q in queries])
        result=[]
        for r,query,vec in zip(records,queries,vectors):
            cid=r['id'];mappings=map_case_evidence(cases[cid],chunks)
            # Refresh scoring after the final label-only minimality review.
            r['scores']={k:score_stage(k,v,mappings) for k,v in r['stages'].items()}
            r['scores']['context']=score_stage('context',r['entries'],mappings)
            scores=np.full(len(ids),-np.inf,np.float32)
            np.maximum.at(scores,parents,matrix@np.asarray(vec,np.float32))
            ranks={p:int(np.count_nonzero(scores>scores[i]))+1 for i,p in enumerate(ids)}
            result.append(dict(id=cid,semantic_query=query,targets=[dict(evidence_id=m.evidence_id,quote=m.quote,
                exact_completion_rank=min((max(ranks[p.chunk_id] for p in a.parts) for a in m.alternatives if a.confirmed),default=None),
                sources=[dict(chunk_id=p,rank=ranks[p]) for p in m.chunk_ids]) for m in mappings if m.kind=='positive']))
            print(cid,[(t['evidence_id'],t['exact_completion_rank']) for t in result[-1]['targets']],flush=True)
        assert not attempts
    (OUT/'fresh_retrieval.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in records),encoding='utf8')
    write(OUT/'fresh_vector_exact.json',dict(api_calls=0,units=len(units),parents=len(ids),network_attempts=attempts,results=result))


if __name__=='__main__':main()
