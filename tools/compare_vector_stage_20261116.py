"""Whole vector-stage controlled audit; no production changes or generation API.

Real production retriever, complete Chroma/cache integrity comparison, exhaustive
cosine controls, larger ANN depth, query instruction ablation and parent pooling.
Gold is accessed only after retrieval to score frozen candidates.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import numpy as np
from tools.validate_selection_factorial import ROOT, load_inputs, sha, write, no_network
from tools.diagnose_vector_recall import ro
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR, SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

OUT=ROOT/'reports/rag_vector_stage_comparison_20261116'
LABELS={
    'baseline':'A 当前真实向量检索',
    'exact':'B 全库精确余弦，其他不变',
    'deeper':'C ANN窗口搜索深度至少1600，其他不变',
    'no_prefix':'D 只去掉查询编码指令，全库精确',
    'top2':'E 父块前两窗口均分，全库精确',
}


def summarize(records):
    result={}
    for cohort in ('formal','diagnostic'):
        selected=[r for r in records if r['cohort']==cohort and r['scores']['vector']['gold_evidence_total_declared']]
        scores=[r['scores']['vector'] for r in selected]
        ranks=[s['first_gold_rank'] for s in scores]
        result[cohort]=dict(positive_cases=len(scores),complete=sum(s['gold_evidence_recall']==1 for s in scores),
            missing=[r['id'] for r in selected if r['scores']['vector']['gold_evidence_recall']<1],
            mean_recall=float(np.mean([s['gold_evidence_recall'] for s in scores])),
            mrr=sum(1/x if x else 0 for x in ranks)/len(ranks),
            **{f'hit_at_{k}':sum(x is not None and x<=k for x in ranks) for k in (1,3,5,50)})
    return result


def main():
    import torch
    torch.set_num_threads(3)
    OUT.mkdir(parents=True,exist_ok=True)
    _,rawsettings,_,_=load_inputs();settings=AppSettings.from_dict(rawsettings)
    assert not settings.rag_query_plan_enabled
    f=ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json'
    manifest=json.loads(f.read_text(encoding='utf8'))
    cases=[c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases]
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==manifest['bundle_id']
    policy=VectorUnitPolicy.from_settings(settings)
    index=VectorIndexStore(store.root).active(bundle.bundle_id,embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision,embedding_dimension=bundle.embedding_dimension,policy_fingerprint=policy.fingerprint)
    protected_paths=[f,*[ROOT/d['path'] for d in manifest['datasets']],SETTINGS_PATH,bundle.metadata_path,index.database_path,
        store.root/'active.json',index.path.parent/'active.json',
        ROOT/'local_full_text_search/rag/retrieval/vector.py',ROOT/'local_full_text_search/rag/retrieval/vector_units.py',
        ROOT/'local_full_text_search/rag/embeddings/local_bge.py']
    protected={str(p):sha(p) for p in protected_paths if p.is_file()}
    protocol=dict(bundle=bundle.bundle_id,index=asdict(index),settings=rawsettings,case_ids=[c.case_id for c in cases],
        case_count=len(cases),labels=LABELS,protected=protected,api_calls=0,
        contrasts={'baseline-exact':'ANN vs exhaustive search; same query/doc vectors and max pooling',
            'baseline-deeper':'ANN request depth only; same final 50 parents',
            'exact-no_prefix':'query encoding instruction only; document vectors unchanged',
            'exact-top2':'parent pooling only; same query/doc vectors'},
        stopping_rule='One frozen pass of all 91 current questions and four controls; no tuning or quality-based retry; report all gains/losses.',
        interpretation='diagnostic controls, not a deployment or new independent acceptance test',
        model_scope='Same local BGE model. No other model is evaluated; this cannot distinguish all model architecture effects.',
        gold_runtime_access=False)
    write(OUT/'protocol.json',protocol)
    with store.connect_read_only(bundle.metadata_path) as c:chunkrows=load_chunk_rows(c)
    chunks={r['chunk_id']:r for r in chunkrows}
    mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
    with ro(index.database_path) as c:units=[dict(r) for r in c.execute('select * from vector_units order by unit_id')]
    pids=list(dict.fromkeys(u['parent_chunk_id'] for u in units));pi={p:i for i,p in enumerate(pids)}
    unit_parent=np.array([pi[u['parent_chunk_id']] for u in units]);ui={u['unit_id']:i for i,u in enumerate(units)}
    assert set(pids)==set(chunks)
    matrix=np.empty((len(units),bundle.embedding_dimension),np.float32)
    with ro(store.root/'embedding_cache.db') as c:
        for i,u in enumerate(units):
            r=c.execute('select vector_blob from embedding_cache where model_id=? and model_revision=? and dimension=? and text_sha256=?',
                (bundle.embedding_repo_id,bundle.embedding_revision,bundle.embedding_dimension,hashlib.sha256(u['embedding_text'].encode()).hexdigest())).fetchone()
            assert r is not None,u['unit_id']
            matrix[i]=np.frombuffer(r[0],dtype='<f4')
    norms=np.linalg.norm(matrix,axis=1)
    assert np.isfinite(matrix).all() and np.allclose(norms,1,atol=1e-4)
    outputs={a:[] for a in LABELS};details=[];integrity=Counter();fresh_vectors=[]
    with no_network() as attempts:
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
        embedding=LocalBgeEmbeddingProvider(path,repo_id=bundle.embedding_repo_id,revision=bundle.embedding_revision,batch_size=8)
        vectorstore=ChromaVectorStore(index.chroma_path)
        assert vectorstore.count()==len(units)
        # All actual stored vectors and provenance, not the prior 81-vector sample.
        seen=set();metadata_mismatches=[];vector_mismatches=[]
        for start in range(0,len(units),2000):
            response=vectorstore._collection.get(limit=2000,offset=start,include=['embeddings','metadatas'])
            for key,v,meta in zip(response['ids'],response['embeddings'],response['metadatas']):
                seen.add(key)
                if key not in ui:metadata_mismatches.append(dict(unit=key,reason='orphan unit'));continue
                i=ui[key];u=units[i]
                if meta.get('stable_chunk_id')!=u['parent_chunk_id'] or meta.get('unit_id')!=key or int(meta.get('unit_ordinal',-1))!=u['ordinal']:
                    metadata_mismatches.append(dict(unit=key,metadata=meta))
                if not np.allclose(matrix[i],v,atol=1e-7):vector_mismatches.append(key)
                integrity['stored_vectors_compared']+=1
            print('Stored vectors checked',len(seen),'/',len(units),flush=True)
        missing=set(ui)-seen
        assert not missing and not vector_mismatches and not metadata_mismatches
        probe=np.linspace(0,len(units)-1,32,dtype=int).tolist()
        fresh=np.asarray(embedding.embed_documents([units[i]['embedding_text'] for i in probe]),np.float32)
        for i,v in zip(probe,fresh):
            cosine=float(v@matrix[i]);fresh_vectors.append(dict(unit_id=units[i]['unit_id'],cosine=cosine,
                max_absolute_error=float(np.max(np.abs(v-matrix[i])))))
        assert min(r['cosine'] for r in fresh_vectors)>.9999
        with store.connect_read_only(bundle.metadata_path) as c:
            eligibility=np.array([store._matches_filters(store.load_chunk(c,p),(),(),True,settings.ocr_min_confidence) for p in pids])
        interpreter=QueryInterpreter(enhanced=False)
        class RecordingEmbedding:
            last=None
            def embed_query(self,text):
                self.last=np.asarray(embedding.embed_query(text),np.float32)
                return self.last.tolist()
        recorder=RecordingEmbedding()
        def exact_order(unit_scores,pool='max'):
            first=np.full(len(pids),-np.inf,np.float32)
            np.maximum.at(first,unit_parent,unit_scores)
            if pool=='top2':
                # Exclude only one best unit per parent; retain a second equal-scoring unit if present.
                order=np.argsort(-unit_scores,kind='stable');best_indices={}
                for i in order:
                    p=int(unit_parent[i])
                    if p not in best_indices:best_indices[p]=int(i)
                second_scores=unit_scores.copy()
                second_scores[list(best_indices.values())]=-np.inf
                second=np.full(len(pids),-np.inf,np.float32);np.maximum.at(second,unit_parent,second_scores)
                second[~np.isfinite(second)]=first[~np.isfinite(second)]
                ps=(first+second)/2
            else:ps=first
            ps[~eligibility]=-np.inf
            ordered=[pids[i] for i in np.argsort(-ps,kind='stable') if np.isfinite(ps[i])]
            return ordered,ps
        def snapshots(ids,scores):return [{**chunks[p],'vector_rank':i+1,'vector_score':float(scores[pi[p]])} for i,p in enumerate(ids[:50])]
        for n,case in enumerate(cases,1):
            plan=interpreter.interpret(case.question)
            q=RagQuery(case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,vector_top_k=50)
            retriever=VectorRetriever(store,bundle,vectorstore,recorder,backend='unit_v2',unit_index=index,
                oversample_factor=settings.rag_vector_unit_oversample_factor)
            actual=retriever.retrieve(q,CancelToken(),plan)
            uscores=matrix@recorder.last
            order,ps=exact_order(uscores)
            noprefix=np.asarray(embedding.embed_documents([plan.semantic_query])[0],np.float32)
            norder,nps=exact_order(matrix@noprefix)
            torder,tps=exact_order(uscores,'top2')
            depth=max(1600,retriever.route_diagnostics[0]['requested_unit_depth']*4)
            hits=vectorstore.query(recorder.last.tolist(),depth,include_ocr=True)
            deeper=[];seenp=set();dscore={}
            for h in hits:
                p=h.metadata.get('stable_chunk_id')
                if p not in pi or not eligibility[pi[p]] or p in seenp:continue
                seenp.add(p);deeper.append(p);dscore[p]=h.score
                if len(deeper)>=50:break
            baseline=[candidate_snapshot(c) for c in actual]
            candidates=dict(baseline=baseline,exact=snapshots(order,ps),deeper=[{**chunks[p],'vector_rank':i+1,'vector_score':dscore[p]} for i,p in enumerate(deeper)],
                no_prefix=snapshots(norder,nps),top2=snapshots(torder,tps))
            target_ranks=[]
            rank={p:i+1 for i,p in enumerate(order)}
            for m in mappings[case.case_id]:
                if m.kind!='positive':continue
                alternatives=[max(rank.get(p.chunk_id,99999) for p in a.parts) for a in m.alternatives if a.confirmed]
                target_ranks.append(dict(evidence_id=m.evidence_id,quote=m.quote,exact_completion_rank=min(alternatives,default=None)))
            for arm,items in candidates.items():
                record=dict(id=case.case_id,question=case.question,group_id=case.group_id,review_status=case.review_status,
                    cohort='formal' if case.review_status=='reviewed' else 'diagnostic',query_plan=asdict(plan),
                    candidate_stages={'vector':items},scores={'vector':score_stage('vector',items,mappings[case.case_id])},api_calls=0)
                outputs[arm].append(record)
            details.append(dict(id=case.case_id,question=case.question,original_equals_semantic=case.question==plan.semantic_query,
                query_tokens=embedding.count_tokens(embedding.QUERY_PREFIX+plan.semantic_query),
                query_truncated=embedding.count_tokens(embedding.QUERY_PREFIX+plan.semantic_query)>embedding.max_sequence_length,
                candidate_count=len(actual),requested_unit_depth=retriever.route_diagnostics[0]['requested_unit_depth'],
                deep_unit_depth=depth,ann_exact_overlap=len(set(x['chunk_id'] for x in baseline)&set(order[:50])),
                ann_deep_overlap=len(set(x['chunk_id'] for x in baseline)&set(deeper)),target_ranks=target_ranks))
            print(n,case.case_id,{a:(r[-1]['scores']['vector']['gold_evidence_found'],r[-1]['scores']['vector']['first_gold_rank']) for a,r in outputs.items()},flush=True)
        vectorstore.close()
        assert not attempts
    for arm,rows in outputs.items():
        (OUT/(arm+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in rows),encoding='utf8')
    write(OUT/'case_diagnostics.json',details)
    write(OUT/'metrics.json',{a:summarize(rows) for a,rows in outputs.items()})
    differences={}
    for arm,rows in outputs.items():
        differences[arm]=dict(gains=[],losses=[],first_rank_gains=[],first_rank_losses=[],new_negative=[])
        for before,after in zip(outputs['baseline'],rows):
            bs=before['scores']['vector'];ns=after['scores']['vector'];cid=after['id']
            if ns['gold_evidence_found']>bs['gold_evidence_found']:differences[arm]['gains'].append(cid)
            if ns['gold_evidence_found']<bs['gold_evidence_found']:differences[arm]['losses'].append(cid)
            if (ns['first_gold_rank'] or 99999)<(bs['first_gold_rank'] or 99999):differences[arm]['first_rank_gains'].append(cid)
            if (ns['first_gold_rank'] or 99999)>(bs['first_gold_rank'] or 99999):differences[arm]['first_rank_losses'].append(cid)
            if set(ns['hard_negative_ids'])-set(bs['hard_negative_ids']):differences[arm]['new_negative'].append(cid)
    write(OUT/'differences.json',differences)
    drift=[p for p,h in protected.items() if sha(p)!=h]
    assert not drift
    write(OUT/'execution_audit.json',dict(api_calls=0,network_attempts=attempts,protected_drift=drift,cases=len(cases),
        integrity=dict(**integrity,missing_unit_ids=len(missing),vector_mismatches=vector_mismatches,metadata_mismatches=metadata_mismatches,
            cache_norm_range=[float(norms.min()),float(norms.max())],all_parent_ids_present=True),
        fresh_model_probe=fresh_vectors,collection_scope='all52282 stored embeddings/provenance vs metadata/cache; fresh model recompute32 samples',
        query_original_equal_count=sum(r['original_equals_semantic'] for r in details),
        query_truncated_count=sum(r['query_truncated'] for r in details),
        unit_token_range=[min(u['token_count'] for u in units),max(u['token_count'] for u in units)],
        model_max_tokens=embedding.max_sequence_length,eligible_parents=int(eligibility.sum()),
        default_changed=False,source_changed=False,gold_changed=False,model_changed=False,
        api_scope='Vector retrieval only; no rerank/Context re-execution or answers in new arms.'))
    print('FINISHED',json.dumps(differences,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
