"""Full-corpus isolated encoding experiment. Gold is only used after retrieval."""
from __future__ import annotations
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import argparse
from collections import Counter
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import numpy as np
from tools.prepare_vector_packaging_20261116 import OUT, ROOT, DB, FIXTURE
from tools.validate_selection_factorial import write, sha, no_network
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.vector_index_store import VectorIndexStore, VectorIndexBundle
from local_full_text_search.rag.retrieval.vector_units import VectorUnitBuilder, VectorUnitPolicy, VectorRetrievalUnit
from local_full_text_search.rag.retrieval.embedding_context import EmbeddingContextPolicy
from local_full_text_search.rag.retrieval.source_representation import SourceBackgroundProvider, StructuredUnitBuilder, inspect_pptx
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.embedding_cache import EmbeddingVectorCache
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.models import RagQuery, VectorRecord
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.vector_index_builder import VectorIndexBuilder

REPDB=OUT/'representations.db'
CACHE=OUT/'embedding_cache.db'
BASE=ROOT/'reports/rag_vector_stage_comparison_20261116'

def read(name): return json.loads((OUT/name).read_text(encoding='utf8'))
def lines(path): return [json.loads(x) for x in Path(path).read_text(encoding='utf8').splitlines()]
def dump(name, rows): (OUT/name).write_text(''.join(json.dumps(x,ensure_ascii=False,default=str)+'\n' for x in rows),encoding='utf8')

def inputs():
    p=read('protocol.json')
    drift=[path for path,digest in p['protected'].items() if sha(path)!=digest]
    assert not drift, ('frozen input drift',drift)
    prior=json.loads((BASE/'protocol.json').read_text(encoding='utf8'))
    settings=AppSettings.from_dict(prior['settings'])
    assert not settings.rag_query_plan_enabled
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==p['bundle']
    return store,bundle,settings

def model(bundle):
    import torch
    torch.set_num_threads(6)
    path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
    return LocalBgeEmbeddingProvider(path,repo_id=bundle.embedding_repo_id,revision=bundle.embedding_revision,batch_size=32)

def prepare():
    store,bundle,settings=inputs()
    if REPDB.exists(): raise ValueError('Representations already frozen; do not silently overwrite')
    documents={};source_errors=[]
    with store.connect_read_only(bundle.metadata_path) as c:
        chunks=[store.load_chunk(c,r[0]) for r in c.execute('SELECT stable_chunk_id FROM rag_chunks ORDER BY stable_chunk_id')]
    for path in sorted({c.source.file_path for c in chunks if c.source.extension.lower()=='.pptx'}):
        try:
            if not Path(path).is_file(): raise ValueError('archive/absent native file: fallback to filename, no guessed title/table')
            documents[str(Path(path))]=inspect_pptx(path)
        except Exception as e: source_errors.append(dict(file=path,error=str(e)))
    write(OUT/'source_sidecar.json',documents)
    provider=SourceBackgroundProvider(documents)
    index=VectorIndexStore(store.root).active(bundle.bundle_id,embedding_repo_id=bundle.embedding_repo_id,
        embedding_revision=bundle.embedding_revision,embedding_dimension=bundle.embedding_dimension,policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
    with ro(index.database_path) as c: old=[dict(r) for r in c.execute('SELECT * FROM vector_units ORDER BY parent_chunk_id,ordinal')]
    oldbyparent={}
    for u in old: oldbyparent.setdefault(u['parent_chunk_id'],[]).append(u)
    conn=sqlite3.connect(REPDB)
    conn.execute('CREATE TABLE representations(arm TEXT,unit_id TEXT,logical_id TEXT,parent TEXT,ordinal INTEGER,kind TEXT,body TEXT,encoded TEXT,start_char INTEGER,end_char INTEGER,tokens INTEGER,metadata TEXT,PRIMARY KEY(arm,unit_id))')
    stats={};audits=[];parity=[]
    def put(arm,u):
        meta=u.representation_metadata or {'logical_unit_id':u.unit_id,'view':arm}
        conn.execute('INSERT INTO representations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
            (arm,u.unit_id,meta.get('logical_unit_id',u.unit_id),u.parent_chunk_id,u.ordinal,u.kind,u.text,u.embedding_text,u.start_char,u.end_char,u.token_count,json.dumps(meta,ensure_ascii=False)))
    with no_network() as attempts:
        emb=model(bundle);policy=VectorUnitPolicy.from_settings(settings)
        base=VectorUnitBuilder(emb.count_tokens,policy)
        builders={a:VectorUnitBuilder(emb.count_tokens,policy,context_policy=EmbeddingContextPolicy(a),background_provider=provider) for a in ['W0','W1','W2','W3','W4','W5']}
        # Build ordinary windows once. Alternate encodings must not change them.
        for n,chunk in enumerate(chunks,1):
            built=base.build(chunk)
            expected=oldbyparent[chunk.stable_chunk_id]
            assert len(built)==len(expected),(chunk.stable_chunk_id,'window count')
            for u,e in zip(built,expected):
                for key in ('unit_id','parent_chunk_id','ordinal','kind','text','embedding_text','start_char','end_char','token_count'):
                    assert getattr(u,key)==e[key],(chunk.stable_chunk_id,key)
                for arm,builder in builders.items():
                    bg=provider(chunk,u.start_char,u.end_char)
                    encoded,meta=builder.context_policy.render(chunk,u.text,emb.count_tokens,policy.max_tokens,bg)
                    count=emb.count_tokens(encoded)
                    assert count<=192
                    if arm=='W0': assert encoded==u.embedding_text
                    uid=u.unit_id if arm=='W0' else hashlib.sha256((u.unit_id+'\0'+builder.fingerprint+'\0'+encoded).encode()).hexdigest()
                    meta['logical_unit_id']=u.unit_id
                    put(arm,replace(u,unit_id=uid,embedding_text=encoded,token_count=count,representation_metadata=meta))
            if n%128==0:
                conn.commit();print('Representation parity',n,'/',len(chunks),flush=True)
        for arm in builders:
            rs=conn.execute('SELECT tokens,metadata,encoded,body FROM representations WHERE arm=?',(arm,)).fetchall()
            stats[arm]=dict(units=len(rs),unique_texts=len({r[2] for r in rs}),min_tokens=min(r[0] for r in rs),max_tokens=max(r[0] for r in rs),
                           body_only=sum(r[2]==r[3] for r in rs),context_sources=dict(Counter(json.loads(r[1]).get('scope_authority','unknown') for r in rs)),
                           omitted_reasons=dict(Counter(json.loads(r[1]).get('context_omitted_reason','') for r in rs)))
        # Freeze ALL simple native rows as a separate factor, without inspecting scores.
        structural=StructuredUnitBuilder(builders['W0'],documents)
        rowcount=0;parents=set();rowaudit=[]
        for chunk in chunks:
            for u in structural.build(chunk):
                if u.kind=='native_table_row':
                    put('S_W0',u); rowcount+=1; parents.add(u.parent_chunk_id)
                    if len(rowaudit)<80: rowaudit.append(dict(parent=u.parent_chunk_id,source=asdict(chunk.source),body=u.text,encoded=u.embedding_text,metadata=u.representation_metadata))
        conn.commit()
        assert not attempts
    # Stratified deterministic audit by document; does not select on retrieval rank.
    for path,doc in documents.items():
        for number,slide in doc['slides'].items():
            if slide['title'] or slide['tables']:
                audits.append(dict(file=path,sha256=doc['sha256'],slide=int(number),**slide))
    selected=[];seen=set()
    for a in audits:
        if a['file'] not in seen: selected.append(a);seen.add(a['file'])
        if len(selected)>=30:break
    if len(selected)<30:selected.extend(a for a in audits if a not in selected and len(selected)<30)
    write(OUT/'background_structure_audit.json',dict(all_source_pieces=len(audits),audited_source_pieces=selected,
        source_errors=source_errors,source_files=len(documents),native_tables=sum(len(s['tables']) for d in documents.values() for s in d['slides'].values()),
        eligible_tables=sum(t['eligible'] for d in documents.values() for s in d['slides'].values() for t in s['tables']),
        skip_reasons=dict(Counter(t['skip_reason'] for d in documents.values() for s in d['slides'].values() for t in s['tables'] if not t['eligible'])),
        limitation='30 deterministic pieces checked for actual XML title/table provenance; semantic header validity still requires original-layout review. PDF/image tables not implemented.'))
    write(OUT/'representation_audit.json',dict(stats=stats,legacy_parity_units=len(old),legacy_parity_parents=len(chunks),legacy_fingerprint=policy.fingerprint,
        structural_rows=rowcount,structural_parents=len(parents),row_audit=rowaudit,source_sidecar_sha256=sha(OUT/'source_sidecar.json'),api_calls=0))
    conn.close()
    with ro(store.root/'embedding_cache.db') as src:
        dst=sqlite3.connect(CACHE);src.backup(dst);dst.close()
    print('PREPARED',stats,'structural_rows',rowcount,flush=True)

def encode():
    store,bundle,settings=inputs()
    with ro(REPDB) as conn:
        texts=[r[0] for r in conn.execute('SELECT DISTINCT encoded FROM representations ORDER BY tokens,encoded')]
    with EmbeddingVectorCache(CACHE,model_id=bundle.embedding_repo_id,model_revision=bundle.embedding_revision,dimension=bundle.embedding_dimension) as cache:
        missing=[]
        for n in range(0,len(texts),400):
            batch=texts[n:n+400];found=cache.get_many(batch)
            missing.extend(t for t in batch if t not in found)
        use_runtime=(OUT/'encoding_runtime_audit.json').exists() and read('encoding_runtime_audit.json')['passed']
        write(OUT/'encoding_workload.json',dict(unique_total=len(texts),cached=len(texts)-len(missing),missing_at_start=len(missing),api_calls=0,
                                               batch_size=16 if use_runtime else 32,torch_threads=6,full_precision=True,
                                               device='verified_FP32_DirectML' if use_runtime else 'cpu'))
        print('ENCODING unique',len(texts),'missing',len(missing),flush=True)
        started=time.perf_counter()
        with no_network() as attempts:
            if use_runtime:
                from tools.packaging_encoding_runtime import VerifiedFp32Runtime
                path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
                assert sha(OUT/'embedding_fp32_verified.onnx')==read('encoding_runtime_audit.json')['onnx_sha256']
                emb=VerifiedFp32Runtime(OUT,path,bundle)
            else:emb=model(bundle)
            for n in range(0,len(missing),128):
                batch=missing[n:n+128]
                vectors=emb.embed_documents(batch)
                cache.put_many(dict(zip(batch,vectors,strict=True)))
                print('Encoded',min(n+128,len(missing)),'/',len(missing),'seconds',round(time.perf_counter()-started),flush=True)
            assert not attempts
        write(OUT/'encoding_complete.json',dict(unique_texts=len(texts),newly_encoded=len(missing),network_attempts=attempts,api_calls=0,cache_path=str(CACHE)))

def load_cases(include_holdout=False):
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json').read_text(encoding='utf8'))
    cases=[c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases]
    return cases+(list(load_gold_dataset(FIXTURE,reviewed_only=False).cases) if include_holdout else [])

def matrix_for(arm,bundle):
    with ro(REPDB) as c:
        rows=[dict(r) for r in c.execute('SELECT * FROM representations WHERE arm=? ORDER BY unit_id',(arm,))]
    matrix=np.empty((len(rows),bundle.embedding_dimension),np.float32)
    with EmbeddingVectorCache(CACHE,model_id=bundle.embedding_repo_id,model_revision=bundle.embedding_revision,dimension=bundle.embedding_dimension) as cache:
        for i in range(0,len(rows),400):
            subset=rows[i:i+400];found=cache.get_many([r['encoded'] for r in subset])
            for j,r in enumerate(subset):
                assert r['encoded'] in found, ('encoding incomplete',arm,r['unit_id'])
                matrix[i+j]=found[r['encoded']]
    assert np.isfinite(matrix).all() and np.allclose(np.linalg.norm(matrix,axis=1),1,atol=1e-4)
    return rows,matrix

def metrics(records):
    result={}
    for cohort in ('formal','diagnostic','source_holdout'):
        selected=[r for r in records if r['cohort']==cohort and r['scores']['vector']['gold_evidence_total_declared']]
        scores=[r['scores']['vector'] for r in selected];n=len(scores)
        result[cohort]=dict(positive_cases=n,complete=sum(s['gold_evidence_recall']==1 for s in scores),
            missing=[r['id'] for r in selected if r['scores']['vector']['gold_evidence_recall']<1],
            mrr=sum(1/s['first_gold_rank'] if s['first_gold_rank'] else 0 for s in scores)/max(1,n),
            mean_recall=sum(s['gold_evidence_recall'] for s in scores)/max(1,n),
            **{f'hit_at_{k}':sum(s['first_gold_rank'] is not None and s['first_gold_rank']<=k for s in scores) for k in (1,3,5,50)},
            **{f'recall_at_{k}':sum(r.get('cutoff_scores',{}).get(str(k),{}).get('gold_evidence_recall') or 0 for r in selected)/max(1,n) for k in (1,3,5,50)},
            **{f'complete_at_{k}':sum(r.get('cutoff_scores',{}).get(str(k),{}).get('gold_evidence_recall')==1 for r in selected) for k in (1,3,5,50)})
    return result

def evaluate(arms=None, holdout=False):
    store,bundle,settings=inputs()
    arms=arms or [f'W{i}' for i in range(6)]
    cases=load_cases(holdout)
    with store.connect_read_only(bundle.metadata_path) as c:
        chunkrows=load_chunk_rows(c)
        eligibility={r['chunk_id']:store._matches_filters(store.load_chunk(c,r['chunk_id']),(),(),True,settings.ocr_min_confidence) for r in chunkrows}
    chunks={r['chunk_id']:r for r in chunkrows}
    # Labels are only used below to score completed rankings.
    mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
    interpreter=QueryInterpreter(enhanced=False)
    plans={c.case_id:interpreter.interpret(c.question) for c in cases}
    with no_network() as attempts:
        emb=model(bundle)
        queryvectors=np.asarray([emb.embed_query(plans[c.case_id].semantic_query) for c in cases],np.float32)
        np.save(OUT/('query_vectors_holdout.npy' if holdout else 'query_vectors.npy'),queryvectors)
        for arm in arms:
            units,matrix=matrix_for(arm,bundle)
            pids=sorted({u['parent'] for u in units});pi={p:i for i,p in enumerate(pids)}
            up=np.asarray([pi[u['parent']] for u in units]);output=[]
            unit_scores=matrix@queryvectors.T
            for qi,case in enumerate(cases):
                scores=unit_scores[:,qi];ps=np.full(len(pids),-np.inf,np.float32)
                np.maximum.at(ps,up,scores)
                for p,i in pi.items():
                    if not eligibility[p]:ps[i]=-np.inf
                ordered=[pids[i] for i in np.argsort(-ps,kind='stable') if np.isfinite(ps[i])]
                top=[]
                for rank,p in enumerate(ordered[:50],1):
                    ui=np.flatnonzero(up==pi[p]);best=int(ui[np.argmax(scores[ui])]);u=units[best]
                    top.append({**chunks[p], 'vector_rank':rank,'vector_score':float(ps[pi[p]]),
                        'vector_unit_hits':[dict(unit_id=u['unit_id'],text=u['body'],embedding_text=u['encoded'],score=float(scores[best]),
                                                kind=u['kind'],start_char=u['start_char'],end_char=u['end_char'],representation=json.loads(u['metadata']))]})
                target_ranks=[];rankmap={p:i+1 for i,p in enumerate(ordered)}
                for m in mappings[case.case_id]:
                    if m.kind=='positive':
                        rank=min((max(rankmap.get(p.chunk_id,99999) for p in a.parts) for a in m.alternatives if a.confirmed),default=None)
                        target_ranks.append(dict(evidence_id=m.evidence_id,completion_rank=rank))
                output.append(dict(id=case.case_id,group_id=case.group_id,question=case.question,review_status=case.review_status,
                    cohort='source_holdout' if case.case_id.startswith('vector-source-independent-') else ('formal' if case.review_status=='reviewed' else 'diagnostic'),
                    query_plan=asdict(plans[case.case_id]),candidate_stages={'vector':top},scores={'vector':score_stage('vector',top,mappings[case.case_id])},
                    target_ranks=target_ranks,
                    cutoff_scores={str(k):score_stage('vector',top[:k],mappings[case.case_id]) for k in (1,3,5,50)},
                    best_encoded_window_scores=score_stage('vector',[{**x,'text':x['vector_unit_hits'][0]['text']} for x in top],mappings[case.case_id]),
                    api_calls=0))
            dump(f'{arm}_exact'+('_holdout' if holdout else '')+'.jsonl',output)
            write(fpath:=OUT/(f'{arm}_exact'+('_holdout' if holdout else '')+'_metrics.json'),metrics(output))
            print('EXACT',arm,json.dumps(metrics(output),ensure_ascii=False),flush=True)
        assert not attempts

def deltas(before,after,stage='vector'):
    out=dict(gains=[],losses=[],rank_gains=[],rank_losses=[],new_negatives=[])
    for b,a in zip(before,after):
        assert b['id']==a['id']
        bs=b['scores'][stage];ns=a['scores'][stage];cid=a['id']
        if ns['gold_evidence_found']>bs['gold_evidence_found']:out['gains'].append(cid)
        if ns['gold_evidence_found']<bs['gold_evidence_found']:out['losses'].append(cid)
        if (ns['first_gold_rank'] or 99999)<(bs['first_gold_rank'] or 99999):out['rank_gains'].append(cid)
        if (ns['first_gold_rank'] or 99999)>(bs['first_gold_rank'] or 99999):out['rank_losses'].append(cid)
        if ns.get('hard_negative_found',0)>bs.get('hard_negative_found',0):out['new_negatives'].append(cid)
    return out

class FrozenUnitBuilder:
    def __init__(self,arm):
        self.arm=arm;self.conn=sqlite3.connect(REPDB);self.conn.row_factory=sqlite3.Row
        digest=hashlib.sha256()
        for row in self.conn.execute('SELECT unit_id,encoded,metadata FROM representations WHERE arm=? ORDER BY unit_id',(arm,)):
            digest.update(json.dumps(tuple(row),ensure_ascii=False).encode())
        self.fingerprint=digest.hexdigest()
    def build(self,chunk):
        return tuple(VectorRetrievalUnit(r['unit_id'],r['parent'],r['ordinal'],r['body'],r['encoded'],r['kind'],r['start_char'],r['end_char'],r['tokens'],json.loads(r['metadata']))
                     for r in self.conn.execute('SELECT * FROM representations WHERE arm=? AND parent=? ORDER BY ordinal',(self.arm,chunk.stable_chunk_id)))
    def close(self):self.conn.close()

def candidate_index(arm,store,bundle,settings,emb):
    path=OUT/f'{arm}_index.json'
    if arm=='F11' and read('selection.json')['packaging_candidate']=='W0' and (OUT/'F10_index.json').exists():
        # Degenerate factorial: identical representation must use the identical
        # ANN graph, otherwise random graph differences masquerade as interaction.
        data=read('F10_index.json');write(path,data)
        for k in ('path','database_path','chroma_path'):data[k]=Path(data[k])
        return VectorIndexBundle(**data)
    if path.exists():
        data=json.loads(path.read_text(encoding='utf8'))
        for k in ('path','database_path','chroma_path'):data[k]=Path(data[k])
        index=VectorIndexBundle(**data)
        assert index.source_bundle_id==bundle.bundle_id
        with ro(index.database_path) as c:assert c.execute('SELECT state FROM vector_index_meta').fetchone()[0]=='published'
        return index
    builder=FrozenUnitBuilder(arm)
    def progress(e):
        if e.get('processed_parent_chunks',0)%1024==0:print('INDEX',arm,e.get('processed_parent_chunks'),e.get('unit_count'),flush=True)
    try:
        index,_=VectorIndexBuilder(store,emb,settings,vector_root=OUT/'indexes',cache_path=CACHE,unit_builder=builder).prepare(
            bundle,CancelToken(),progress_callback=progress,publish_pointer=False)
    finally:builder.close()
    write(path,asdict(index))
    return index

def ann(arms=None,holdout=False):
    store,bundle,settings=inputs();cases=load_cases(holdout)
    if arms is None:
        arms=['W0','W4','W5']
        baseline=lines(OUT/'W0_exact.jsonl')
        for a in ('W1','W2','W3'):
            if not deltas(baseline,lines(OUT/f'{a}_exact.jsonl'))['losses']:arms.append(a)
    with store.connect_read_only(bundle.metadata_path) as c:chunkrows=load_chunk_rows(c)
    mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
    with no_network() as attempts:
        emb=model(bundle)
        for arm in arms:
            if arm=='W0':
                index=VectorIndexStore(store.root).active(bundle.bundle_id,embedding_repo_id=bundle.embedding_repo_id,embedding_revision=bundle.embedding_revision,
                    embedding_dimension=bundle.embedding_dimension,policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
                write(OUT/'W0_index.json',asdict(index))
            else:index=candidate_index(arm,store,bundle,settings,emb)
            vectorstore=ChromaVectorStore(index.chroma_path);output=[]
            assert vectorstore.count()==index.unit_count
            retriever=VectorRetriever(store,bundle,vectorstore,emb,backend='unit_v2',unit_index=index,oversample_factor=settings.rag_vector_unit_oversample_factor)
            try:
                for case in cases:
                    plan=QueryInterpreter(enhanced=False).interpret(case.question)
                    query=RagQuery(case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,vector_top_k=50)
                    got=[candidate_snapshot(x) for x in retriever.retrieve(query,CancelToken(),plan)]
                    output.append(dict(id=case.case_id,question=case.question,group_id=case.group_id,review_status=case.review_status,
                        cohort='source_holdout' if case.case_id.startswith('vector-source-independent-') else ('formal' if case.review_status=='reviewed' else 'diagnostic'),
                        query_plan=asdict(plan),candidate_stages={'vector':got},scores={'vector':score_stage('vector',got,mappings[case.case_id])},
                        route_diagnostics=retriever.route_diagnostics,
                        cutoff_scores={str(k):score_stage('vector',got[:k],mappings[case.case_id]) for k in (1,3,5,50)},
                        best_encoded_window_scores=score_stage('vector',[{**x,'text':x['vector_unit_hits'][0]['text']} for x in got],mappings[case.case_id]),
                        api_calls=0))
                name=f'{arm}_ann'+('_holdout' if holdout else '')
                dump(name+'.jsonl',output);write(OUT/(name+'_metrics.json'),metrics(output))
                print('ANN',arm,json.dumps(metrics(output),ensure_ascii=False),flush=True)
            finally:vectorstore.close()
        assert not attempts
    write(OUT/('ann_holdout_audit.json' if holdout else 'ann_audit.json'),dict(arms=arms,api_calls=0,network_attempts=attempts,
        production_index_untouched=True,method='Real Chroma index + production VectorRetriever, ordinary adaptive 400-window entry budget'))

def choose():
    baseline=lines(OUT/'W0_exact.jsonl');annbase=lines(OUT/'W0_ann.jsonl')
    differences={a:deltas(baseline,lines(OUT/f'{a}_exact.jsonl')) for a in [f'W{i}' for i in range(6)]}
    eligible=[]
    for arm in ('W4','W5','W3','W1','W2'):
        if not (OUT/f'{arm}_ann.jsonl').exists():continue
        annrows=lines(OUT/f'{arm}_ann.jsonl');change=deltas(annbase,annrows)
        if differences[arm]['losses'] or change['losses'] or differences[arm]['new_negatives'] or change['new_negatives']:continue
        # Improvement must occur in at least two source groups, not two rewrites.
        improved=set(change['gains']+change['rank_gains'])
        groups={r['group_id'] for r in annrows if r['id'] in improved}
        if len(groups)>=2:eligible.append(arm)
    selected=eligible[0] if eligible else 'W0'
    # This is a frozen development choice, not a deployment decision. Independent
    # acceptance and actual Context regressions can still reject it.
    w4=lines(OUT/'W4_exact.jsonl');w5=lines(OUT/'W5_exact.jsonl')
    complement=deltas(w4,w5)
    dual=bool(complement['gains'] and complement['losses'] and not differences['W4']['losses'] and not differences['W5']['losses'])
    write(OUT/'selection.json',dict(packaging_candidate=selected,eligible=eligible,differences=differences,
        dual_eligible=dual,dual_reason='verified complete-evidence complement' if dual else 'No two non-regressing single views have complementary complete-Gold gains; W6 not applicable.',
        frozen_before_source_holdout=True,adopted=False,
        source_independence_gate='Ranking groups alone are preliminary; final adoption also requires >=2 source groups on frozen independent questions and no actual Context loss.'))
    print('FROZEN CANDIDATE',selected,'dual',dual,flush=True)

def structure():
    store,bundle,settings=inputs();selected=read('selection.json')['packaging_candidate']
    documents=read('source_sidecar.json');provider=SourceBackgroundProvider(documents)
    with no_network() as attempts:
        emb=model(bundle);base=VectorUnitBuilder(emb.count_tokens,VectorUnitPolicy.from_settings(settings),
            context_policy=EmbeddingContextPolicy(selected),background_provider=provider)
        conn=sqlite3.connect(REPDB)
        for arm,pack in [('F10','W0'),('F11',selected),('F10_count_matched','W0')]:
            if conn.execute('SELECT 1 FROM representations WHERE arm=? LIMIT 1',(arm,)).fetchone():continue
            # Copy the baseline logical windows unchanged, then add the same frozen native rows.
            conn.execute('INSERT INTO representations SELECT ?,unit_id,logical_id,parent,ordinal,kind,body,encoded,start_char,end_char,tokens,metadata FROM representations WHERE arm=?',(arm,pack))
            source_rows=conn.execute("SELECT * FROM representations WHERE arm='S_W0' ORDER BY parent,ordinal").fetchall()
            for row in source_rows:
                values=list(row);values[0]=arm
                if pack!='W0':
                    with store.connect_read_only(bundle.metadata_path) as c:chunk=store.load_chunk(c,row[3])
                    encoded,meta=base.context_policy.render(chunk,row[6],emb.count_tokens,192,provider(chunk,0,0))
                    originalmeta=json.loads(row[11]);originalmeta.update(meta)
                    values[1]=hashlib.sha256((row[2]+base.fingerprint+encoded).encode()).hexdigest()
                    values[7]=encoded;values[10]=emb.count_tokens(encoded);values[11]=json.dumps(originalmeta,ensure_ascii=False)
                conn.execute('INSERT INTO representations VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',values)
            if arm=='F10_count_matched':
                # Remove an equal number of original windows per parent. This
                # intentionally lossy diagnostic isolates extra-candidate effects.
                for parent,n in conn.execute("SELECT parent,COUNT(*) FROM representations WHERE arm='S_W0' GROUP BY parent").fetchall():
                    ids=[r[0] for r in conn.execute("SELECT unit_id FROM representations WHERE arm=? AND parent=? AND kind!='native_table_row' ORDER BY ordinal DESC LIMIT ?",(arm,parent,n))]
                    for uid in ids:conn.execute('DELETE FROM representations WHERE arm=? AND unit_id=?',(arm,uid))
                    # More rows than windows: trim supplementary rows too, fixed count.
                    baseline_count=conn.execute("SELECT COUNT(*) FROM representations WHERE arm='W0' AND parent=?",(parent,)).fetchone()[0]
                    ids=[r[0] for r in conn.execute('SELECT unit_id FROM representations WHERE arm=? AND parent=? ORDER BY ordinal',(arm,parent))]
                    for uid in ids[baseline_count:]:conn.execute('DELETE FROM representations WHERE arm=? AND unit_id=?',(arm,uid))
            conn.commit()
        counts=dict(conn.execute('SELECT arm,COUNT(*) FROM representations GROUP BY arm').fetchall());conn.close()
        assert counts['F10']==counts['F11'] and counts['F10_count_matched']==counts['W0']
        assert not attempts
    write(OUT/'factorial_protocol.json',dict(F00='W0',F10='F10',F01=selected,F11='F11',logical_units_equal_F10_F11=True,counts=counts,
        count_sensitivity='F10_count_matched: row substitution, same total original count; cannot attribute all supplementary gains to row semantics',
        structural_scope='Only eligible native PPTX tables; no PDF reconstruction',api_calls=0))

def dual():
    from local_full_text_search.rag.retrieval.embedding_context import fuse_parent_view_ranks
    assert read('selection.json')['dual_eligible'],'Dual factor gate did not pass'
    store,bundle,settings=inputs();cases=load_cases()
    with store.connect_read_only(bundle.metadata_path) as c:
        chunkrows=load_chunk_rows(c)
        eligible={r['chunk_id']:store._matches_filters(store.load_chunk(c,r['chunk_id']),(),(),True,settings.ocr_min_confidence) for r in chunkrows}
    chunks={r['chunk_id']:r for r in chunkrows};mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
    exact={a:lines(OUT/f'{a}_exact.jsonl') for a in ('W4','W5')}
    outputs={a:[] for a in ('W6_exact','W5_duplicate_exact','W6_ann400','W6_ann800','W4_ann800','W5_ann800')}
    with no_network() as attempts:
        emb=model(bundle);indexes={a:candidate_index(a,store,bundle,settings,emb) for a in ('W4','W5')}
        stores={a:ChromaVectorStore(indexes[a].chroma_path) for a in indexes}
        def parent_order(hits):
            seen=set();result=[]
            for h in hits:
                parent=str(h.metadata.get('stable_chunk_id') or '')
                if parent in seen or parent not in chunks or not eligible[parent]:continue
                seen.add(parent);result.append(parent)
                if len(result)>=50:break
            return result
        try:
            for n,case in enumerate(cases):
                plan=QueryInterpreter(enhanced=False).interpret(case.question);qvector=emb.embed_query(plan.semantic_query)
                exactorders={a:[r['chunk_id'] for r in exact[a][n]['candidate_stages']['vector']] for a in exact}
                annorders={(a,depth):parent_order(stores[a].query(qvector,depth,include_ocr=True)) for a in stores for depth in (200,400,800)}
                configs={
                    'W6_exact':[('W4',exactorders['W4']),('W5',exactorders['W5'])],
                    'W5_duplicate_exact':[('W5',exactorders['W5']),('W5',exactorders['W5'])],
                    'W6_ann400':[(a,annorders[a,200]) for a in stores],
                    'W6_ann800':[(a,annorders[a,400]) for a in stores],
                    'W4_ann800':[('W4',annorders['W4',800])],
                    'W5_ann800':[('W5',annorders['W5',800])],
                }
                for arm,routes in configs.items():
                    order,scores,details=fuse_parent_view_ranks(routes,top_k=50,rrf_k=60)
                    items=[{**chunks[p],'vector_rank':i+1,'vector_score':scores[p],'view_votes':details[p]} for i,p in enumerate(order)]
                    outputs[arm].append(dict(id=case.case_id,question=case.question,group_id=case.group_id,review_status=case.review_status,
                        cohort='formal' if case.review_status=='reviewed' else 'diagnostic',query_plan=asdict(plan),candidate_stages={'vector':items},
                        scores={'vector':score_stage('vector',items,mappings[case.case_id])},api_calls=0))
                    if arm=='W5_duplicate_exact':assert order==exactorders['W5']
        finally:
            for s in stores.values():s.close()
        assert not attempts
    for arm,records in outputs.items():dump(arm+'.jsonl',records)
    comparisons={a:deltas(exact['W5'],records) for a,records in outputs.items()}
    write(OUT/'dual_audit.json',dict(status='executed',reason='Full-corpus complement gate triggered; isolated route factor only',
        metrics={a:metrics(rs) for a,rs in outputs.items()},differences=comparisons,api_calls=0,network_attempts=attempts,
        duplicate_view_parity=True,exact_routes_max_parent=50,rrf_constant=60,total_ann_budgets=[400,800],
        production_implemented=False,adopted=False,
        adoption_note='W6 remains an isolated experimental routing factor; single-view four-factor controls stay separate.'))

def downstream(arms=None):
    """Real AnswerService.retrieve, explicit isolated vector handle, no API key.

    Candidate selection/scoring sees only query and corpus. The evaluator is
    called after the native entry point returns. Unchanged CE input pairs can
    reuse their previous local-model outputs; fresh pairs use the same model.
    """
    from unittest.mock import patch
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from tools.validate_query_plan_adapter import STAGES
    store,bundle,settings=inputs();cases=load_cases(True)
    selected=read('selection.json')['packaging_candidate']
    arms=arms or list(dict.fromkeys(['W0',selected,'F10','F11']))
    with store.connect_read_only(bundle.metadata_path) as c:chunkrows=load_chunk_rows(c)
    mappings={c.case_id:map_case_evidence(c,chunkrows) for c in cases}
    class Forbidden:
        def retrieve(self,*a,**k):raise AssertionError('API credentials forbidden')
        def generate(self,*a,**k):raise AssertionError('Generation forbidden')
    class PairCache:
        def __init__(self,ranker):
            self.ranker=ranker;self.scores={};self.reused=0;self.new=0
            if (OUT/'ce_pairs.json').exists():self.scores=json.loads((OUT/'ce_pairs.json').read_text(encoding='utf8'))
        def rerank(self,question,candidates,top_n,token):
            def key(item):return hashlib.sha256((question+'\0'+item.chunk.text).encode()).hexdigest()
            missing=[c for c in candidates if key(c) not in self.scores]
            if missing:
                self.ranker.rerank(question,missing,len(missing),token)
                for c in missing:self.scores[key(c)]=[c.rerank_score,c.rerank_raw_logit]
                self.new+=len(missing)
                write(OUT/'ce_pairs.json',self.scores)
            self.reused+=len(candidates)-len(missing)
            for c in candidates:c.rerank_score,c.rerank_raw_logit=self.scores[key(c)]
            ordered=self.ranker._order_scored(question,candidates,token)
            self.last_scored_candidates=tuple(ordered)
            return ordered[:max(1,top_n)]
    realstore=VectorIndexStore
    with no_network() as attempts:
        emb=model(bundle)
        ranker=PairCache(LocalBgeReranker(LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(
            ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
            repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,
            batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,input_mode='truncate'))
        for arm in arms:
            index=candidate_index(arm,store,bundle,settings,emb) if arm!='W0' else realstore(store.root).active(bundle.bundle_id,
                embedding_repo_id=bundle.embedding_repo_id,embedding_revision=bundle.embedding_revision,embedding_dimension=bundle.embedding_dimension,
                policy_fingerprint=VectorUnitPolicy.from_settings(settings).fingerprint)
            class BoundStore:
                # Test-only dependency replacement, bounded to this one known
                # source bundle/model; does not alter any manifest or setting.
                def __init__(self,*a,**k):pass
                def attached(self,source_bundle_id,*a,**k):
                    assert source_bundle_id==index.source_bundle_id
                    assert k['embedding_repo_id']==index.embedding_repo_id and k['embedding_revision']==index.embedding_revision
                    return index
                active=attached
            records=[]
            existing=OUT/f'{arm}_native.jsonl'
            if existing.exists():
                records=lines(existing)
                assert [r['id'] for r in records]==[c.case_id for c in cases[:len(records)]]
            with patch('local_full_text_search.rag.answer_service.VectorIndexStore',BoundStore):
                for case in cases[len(records):]:
                    events=[];stages={}
                    def progress(e):
                        events.append(e)
                        for key,stage in STAGES.items():
                            if key in e:stages[stage]=e[key]
                    service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,emb,ranker,
                        Forbidden(),Forbidden(),settings,progress_callback=progress)
                    q=RagQuery(case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                        lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                        fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k)
                    result=service.retrieve(q,CancelToken())
                    ctx=next((e for e in reversed(events) if 'llm_context' in e),{})
                    stages['context']=ctx.get('context_entries',[])
                    record=dict(id=case.case_id,group_id=case.group_id,question=case.question,review_status=case.review_status,
                        cohort='source_holdout' if case.case_id.startswith('vector-source-independent-') else ('formal' if case.review_status=='reviewed' else 'diagnostic'),
                        query_plan=next(e['query_plan'] for e in events if 'query_plan' in e),candidate_stages=stages,
                        context=ctx.get('llm_context',''),entries=stages['context'],result=asdict(result),
                        scores={s:score_stage(s,items,mappings[case.case_id]) for s,items in stages.items()},
                        events=events,api_calls=0)
                    records.append(record)
                    with existing.open('a',encoding='utf8') as f:f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
                    print('NATIVE',arm,len(records),case.case_id,'context',record['scores']['context']['gold_evidence_found'],flush=True)
            assert len(records)==111
        assert not attempts
    write(OUT/'native_audit.json',dict(arms=arms,questions=111,api_calls=0,network_attempts=attempts,ce_pairs_reused=ranker.reused,ce_pairs_new=ranker.new,
        ce_model=settings.rag_reranker_repo_id,ce_revision=settings.rag_reranker_revision,
        normal_pipeline=True,vector_index_injection='test-only explicit bundle-bound handle; production attached/active pointers untouched',
        context_scoring='actual included_text; no same-page ID-only credit'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','encode','exact','ann','choose','structure','downstream','dual']);ap.add_argument('--holdout',action='store_true');ap.add_argument('--arms',nargs='+')
    a=ap.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='encode':encode()
    elif a.stage=='exact':evaluate(a.arms,a.holdout)
    elif a.stage=='ann':ann(a.arms,a.holdout)
    elif a.stage=='choose':choose()
    elif a.stage=='structure':structure()
    elif a.stage=='downstream':downstream(a.arms)
    elif a.stage=='dual':dual()

if __name__=='__main__':main()
