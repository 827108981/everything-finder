"""Read-only vector attribution: exact search, ANN, source windows and query controls.

Gold is only used for post-hoc rank/visibility diagnostics, never for retrieval.
No persistent model, index, query or algorithm settings are changed.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import hashlib
import json
import sqlite3
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from tools.validate_selection_factorial import ROOT,load_inputs,no_network,write,sha
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore

OUT=ROOT/'reports/rag_vector_diagnosis_20261116'
# Diagnostic human query controls: all expressions come from question intent,
# never from an answer quote. These do not enter the runtime QueryPlan.
CONTROLS={
 'rag-biz-006':['CL-8000i 清洗液缓冲容器何时开始吸液、何时停止吸液？','CL-8000i 一瓶清洗液空了会怎样？'],
 'rag-biz-028':['发光临床结果普遍偏低，该从哪些方面排查？'],
 'rag-biz-050':['暗计数诊断失败，可能是什么原因？'],
 'rag-row-holdout-003':['BS-2000系列从待机到首个样本结果需要多久？','BS-2800M系列从待机到首个样本结果需要多久？'],
 'rag-row-holdout-004':['BS-2000系列能接收哪些样本类型？','BS-2800M系列能接收哪些样本类型？'],
 'rag-row-holdout-005':['凯特电解质模块的电极斜率合格范围是多少？'],
 'rag-row-holdout-009':['BS-5000样本针如何清洗？','BS-2800M样本针如何清洗？'],
 'evidence-flow-independent-05':['A&T ISE的测量原理是什么？','Medica ISE的测量原理是什么？'],
 'evidence-flow-independent-06':['BS-2800M分析部和电脑的通讯方式是什么？','BS-430分析部和电脑的通讯方式是什么？'],
}


def ro(path):
    c=sqlite3.connect('file:'+Path(path).as_posix()+'?mode=ro',uri=True)
    c.row_factory=sqlite3.Row;c.execute('PRAGMA query_only=ON');return c


def main():
    import torch
    torch.set_num_threads(3)
    OUT.mkdir(parents=True,exist_ok=True)
    rows,s,_,maps=load_inputs();settings=AppSettings.from_dict(s)
    production={r['id']:r for r in map(json.loads,(ROOT/'reports/rag_bge_base_adoption_20261116/B.jsonl').read_text(encoding='utf8').splitlines())}
    selected=[]
    for r in rows:
        v=r['scores']['vector'];c=production[r['id']]['scores']['context']
        if v['gold_evidence_total_declared'] and (v['gold_evidence_recall']<1 or (v.get('first_gold_rank') or 999)>10 or c['gold_evidence_recall']<1):selected.append(r)
    store=RagIndexStore('E:/everythingfinder-db/rag');b=store.active()
    index=VectorIndexStore(store.root).active(b.bundle_id,embedding_repo_id=b.embedding_repo_id,embedding_revision=b.embedding_revision,
        embedding_dimension=b.embedding_dimension,policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    protected={str(b.metadata_path):sha(b.metadata_path),str(index.database_path):sha(index.database_path)}
    prior=json.loads((ROOT/'reports/rag_rerank_semantic_20261116/protocol.json').read_text(encoding='utf8'))
    protected.update(prior['protected'])
    with ro(index.database_path) as conn:units=[dict(r) for r in conn.execute('SELECT * FROM vector_units ORDER BY unit_id')]
    with ro(b.metadata_path) as conn:parents={r['stable_chunk_id']:dict(r) for r in conn.execute('SELECT * FROM rag_chunks')}
    byparent=defaultdict(list)
    for i,u in enumerate(units):byparent[u['parent_chunk_id']].append(i)
    matrix=np.empty((len(units),b.embedding_dimension),dtype=np.float32)
    with ro('E:/everythingfinder-db/rag/embedding_cache.db') as cache:
        for i,u in enumerate(units):
            digest=hashlib.sha256(u['embedding_text'].encode()).hexdigest()
            hit=cache.execute('SELECT vector_blob FROM embedding_cache WHERE model_id=? AND model_revision=? AND dimension=? AND text_sha256=?',
                (b.embedding_repo_id,b.embedding_revision,b.embedding_dimension,digest)).fetchone()
            assert hit is not None,u['unit_id']
            matrix[i]=np.frombuffer(hit[0],dtype='<f4')
    norms=np.linalg.norm(matrix,axis=1);assert np.isfinite(matrix).all() and np.allclose(norms,1,atol=1e-4)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    audit=[];out=[]
    with no_network() as attempts:
        embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(b.embedding_repo_id,b.embedding_revision,'embedding')),
            repo_id=b.embedding_repo_id,revision=b.embedding_revision,batch_size=8)
        chroma=ChromaVectorStore(index.chroma_path)
        assert chroma.count()==len(units)==index.unit_count
        # Verify cached embeddings against the actual ANN collection, not just model ID.
        probe=list(range(0,len(units),max(1,len(units)//80)))
        got=chroma._collection.get(ids=[units[i]['unit_id'] for i in probe],include=['embeddings'])
        check={k:v for k,v in zip(got['ids'],got['embeddings'])}
        assert all(np.allclose(matrix[i],check[units[i]['unit_id']],atol=1e-7) for i in probe)
        cases=[];texts=[]
        for r in selected:
            q=next(e['query_plan']['semantic_query'] for e in r['events'] if e.get('stage')=='query_interpreted')
            cases.append((r,q,CONTROLS.get(r['id'],[])))
            texts.extend([embedding.QUERY_PREFIX+t for t in [q,*CONTROLS.get(r['id'],[])]])
            texts.append(q) # No-prefix ablation only; never a production recommendation by itself.
        query_vectors=np.asarray(embedding.embed_documents(texts),dtype=np.float32)
        write(OUT/'protocol.json',dict(bundle=b.bundle_id,vector_index=as_plain(index),model=b.embedding_repo_id,revision=b.embedding_revision,
            selected=[r['id'] for r in selected],selected_reason='vector incomplete OR first Gold rank>10 OR production Context incomplete',
            unit_count=len(units),parent_count=len(parents),embedding_max_tokens=embedding.max_sequence_length,
            windows=as_plain(VectorUnitPolicy.from_settings(settings)),protected=protected,query_controls=CONTROLS,
            controls_scope='post-hoc attribution only, not a new algorithm evaluation; no new Gold or runtime edits',api_calls=0))
        cursor=0
        def rankings(vec):
            scores=matrix@vec
            order=np.argsort(-scores,kind='stable')
            ps={};raw={}
            for rank,i in enumerate(order,1):
                parent=units[int(i)]['parent_chunk_id']
                if parent not in ps:ps[parent]=(len(ps)+1,int(i),float(scores[i]))
                raw[int(i)]=rank
            return scores,ps,raw,order
        for r,q,variants in cases:
            v=query_vectors[cursor];cursor+=1
            scores,ps,raw,order=rankings(v)
            variant_ranks=[]
            for text in variants:
                _,vp,_,_=rankings(query_vectors[cursor]);cursor+=1
                variant_ranks.append((text,vp))
            _,noprefix,_,_=rankings(query_vectors[cursor]);cursor+=1
            # Exactly match current unit_v2 request depth and parent de-duplication.
            requested=settings.rag_vector_top_k*settings.rag_vector_unit_oversample_factor
            hits=chroma.query(v.tolist(),requested,include_ocr=True)
            while len({h.metadata['stable_chunk_id'] for h in hits})<settings.rag_vector_top_k and requested<len(units):
                requested=min(len(units),requested*2);hits=chroma.query(v.tolist(),requested,include_ocr=True)
            ann={}
            for h in hits:
                parent=h.metadata['stable_chunk_id']
                if parent not in ann:ann[parent]=dict(rank=len(ann)+1,unit_id=h.point_id,score=h.score)
            targets=[]
            for m in maps[r['id']]:
                if m.kind!='positive':continue
                relevant=set(m.chunk_ids)|set(m.declared_chunk_ids)
                for alt in m.alternatives:
                    if alt.confirmed:relevant.update(p.chunk_id for p in alt.parts)
                relevant &= parents.keys()
                alternatives=[]
                for alt in m.alternatives:
                    if alt.confirmed and alt.parts:
                        alternatives.append(dict(ids=list(dict.fromkeys(p.chunk_id for p in alt.parts)),
                            exact_rank=max(ps[p.chunk_id][0] for p in alt.parts),
                            ann_rank=max(ann.get(p.chunk_id,{}).get('rank',99999) for p in alt.parts)))
                source_path=Path(m.file)
                if not source_path.is_file():
                    paths={json.loads(parents[c]['source_json'])['file_path'] for c in relevant if Path(json.loads(parents[c]['source_json'])['file_path']).name==source_path.name}
                    if len(paths)==1:source_path=Path(next(iter(paths)))
                original='';error=''
                try:original=original_text(source_path,int(m.page))
                except Exception as ex:error=repr(ex)
                target=dict(evidence_id=m.evidence_id,file=str(source_path),page=m.page,quote=m.quote,
                    original_verified=canonical_content(m.quote) in canonical_content(original) if original else False,original_error=error,
                    alternatives=alternatives,exact_completion_rank=min((a['exact_rank'] for a in alternatives),default=None),
                    ann_completion_rank=min((a['ann_rank'] for a in alternatives),default=None),parents=[])
                for cid in sorted(relevant,key=lambda c:ps[c][0]):
                    idxs=byparent[cid];best=ps[cid][1]
                    detail=[]
                    for i in sorted(idxs,key=lambda i:-scores[i]):
                        u=units[i]
                        related_parts=[p for a in m.alternatives if a.confirmed for p in a.parts if p.chunk_id==cid]
                        # Offsets refer to normalized quote/source mapping; use canonical substrings as a visibility diagnostic.
                        fragments=[canonical_content(m.quote)[p.quote_start:p.quote_end] for p in related_parts]
                        detail.append(dict(**u,score=float(scores[i]),global_unit_rank=raw[i],
                            contains_full_quote=canonical_content(m.quote) in canonical_content(u['text']),
                            contains_mapped_fragment=any(f and f in canonical_content(u['text']) for f in fragments)))
                    target['parents'].append(dict(chunk_id=cid,source=json.loads(parents[cid]['source_json']),text=parents[cid]['text'],
                        exact_rank=ps[cid][0],ann=ann.get(cid),best_unit=units[best]['unit_id'],
                        no_prefix_rank=noprefix[cid][0],query_control_ranks=[dict(query=t,rank=p[cid][0]) for t,p in variant_ranks],units=detail))
                targets.append(target)
            top=[]
            for cid,(rank,i,score) in list(ps.items())[:20]:
                top.append(dict(parent=cid,rank=rank,score=score,source=json.loads(parents[cid]['source_json']),unit=units[i]))
            frozen_ids=[c['chunk_id'] for c in r['candidate_stages']['vector']]
            ann_ids=list(ann)[:50];exact_ids=list(ps)[:50]
            row=dict(id=r['id'],question=r['question'],semantic_query=q,status=r['review_status'],cohort=r['cohort'],
                stage_scores={s:production[r['id']]['scores'][s] for s in ['lexical','vector','fusion','deduplicated','rerank','constraints','context']},
                ann_top50_overlap_with_frozen=len(set(ann_ids)&set(frozen_ids)),exact_ann_overlap_top50=len(set(ann_ids)&set(exact_ids)),
                requested_unit_depth=requested,raw_hits=len(hits),unique_parents_in_raw=len(ann),targets=targets,top20=top)
            out.append(row)
            (OUT/'diagnosis.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in out),encoding='utf8')
            print(r['id'],'targets',[(t['exact_completion_rank'],t['ann_completion_rank']) for t in targets],
                'ANN/exact overlap',row['exact_ann_overlap_top50'],flush=True)
        chroma.close()
        assert cursor==len(query_vectors)
        audit=dict(api_calls=0,network_attempts=attempts,cached_vectors=len(units),actual_chroma_probes=len(probe),
            cached_chroma_probes_equal=True,norm_range=[float(norms.min()),float(norms.max())],
            protected_drift=[p for p,h in protected.items() if sha(p)!=h],cases=len(out))
        assert not audit['protected_drift'] and not attempts
        write(OUT/'execution_audit.json',audit)
    summary={}
    for cohort in ('formal','diagnostic'):
        records=[r for r in rows if r['cohort']==cohort and r['scores']['vector']['gold_evidence_total_declared']]
        summary[cohort]=dict(positive_cases=len(records),vector_complete=sum(r['scores']['vector']['gold_evidence_recall']==1 for r in records),
            missing=[r['id'] for r in records if r['scores']['vector']['gold_evidence_recall']<1],
            first_gold_after10=[r['id'] for r in records if (r['scores']['vector'].get('first_gold_rank') or 999)>10])
    write(OUT/'summary.json',summary)
    print('Finished, no API/index/default changes',flush=True)


def as_plain(value):
    from dataclasses import asdict
    return {k:str(v) if isinstance(v,Path) else v for k,v in asdict(value).items()}


if __name__=='__main__':main()
