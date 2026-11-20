"""Controlled source-local encoding probes on current questions, not global gains."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import json
import hashlib
from pathlib import Path
import numpy as np
from pptx import Presentation
from tools.compare_vector_stage_20261116 import OUT
from tools.validate_selection_factorial import ROOT, no_network, write, sha
from tools.diagnose_vector_recall import ro
from tools.rag_trace_audit import original_text
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence,load_chunk_rows
from local_full_text_search.rag.content_identity import canonical_content


def main():
    import torch
    torch.set_num_threads(3)
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
    from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    records=[json.loads(x) for x in (OUT/'baseline.jsonl').read_text(encoding='utf8').splitlines()]
    selected=[r for r in records if r['scores']['vector']['gold_evidence_total_declared'] and
        (r['scores']['vector']['gold_evidence_recall']<1 or (r['scores']['vector']['first_gold_rank'] or 999)>10)]
    # Also check the known downstream failure as a negative control, not a target to improve.
    selected+=[r for r in records if r['id']=='rag-biz-019']
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases}
    with ro('E:/everythingfinder-db/rag/rag.g000033-71ec3c52f3/metadata.db') as c:chunks=load_chunk_rows(c)
    chunkmap={r['chunk_id']:r for r in chunks}
    with ro(p['index']['database_path']) as c:units=[dict(r) for r in c.execute('select * from vector_units order by unit_id')]
    parents=list(dict.fromkeys(u['parent_chunk_id'] for u in units));pi={cid:i for i,cid in enumerate(parents)}
    up=np.array([pi[u['parent_chunk_id']] for u in units]);matrix=np.empty((len(units),1024),np.float32)
    with ro('E:/everythingfinder-db/rag/embedding_cache.db') as c:
        for i,u in enumerate(units):
            blob=c.execute('select vector_blob from embedding_cache where model_id=? and model_revision=? and dimension=1024 and text_sha256=?',
                (p['index']['embedding_repo_id'],p['index']['embedding_revision'],hashlib.sha256(u['embedding_text'].encode()).hexdigest())).fetchone()
            matrix[i]=np.frombuffer(blob[0],dtype='<f4')
    byparent={cid:[] for cid in parents}
    for u in units:byparent[u['parent_chunk_id']].append(u)
    reps=[];verification=[];presentations={};seenrep=set()
    for r in selected:
        maps=map_case_evidence(cases[r['id']],chunks)
        targets=set()
        for m in maps:
            if m.kind!='positive':continue
            targets.update(m.declared_chunk_ids)
            primary=[chunkmap[c] for c in m.declared_chunk_ids]
            files={c['file_path'] for c in primary}
            if len(files)==1:
                path=next(iter(files));page=int(m.page)
                text=original_text(Path(path),page)
                verification.append(dict(case=r['id'],file=path,page=page,quote=m.quote,
                    original_verified=canonical_content(m.quote) in canonical_content(text),
                    index_verified=any(canonical_content(m.quote) in canonical_content(c['text']) for c in primary)))
        for cid in sorted(targets):
            chunk=chunkmap[cid];us=byparent[cid]
            prefix=us[0]['embedding_text'].split('正文：',1)[0] if '正文：' in us[0]['embedding_text'] else ''
            def add(kind,text,**kw):
                key=(r['id'],cid,kind,text)
                if key in seenrep:return
                seenrep.add(key)
                reps.append(dict(case=r['id'],parent=cid,file=chunk['file_path'],page=chunk['slide_number'] or chunk['page_number'],kind=kind,text=text,**kw))
            add('full_parent',prefix+'正文：'+chunk['text'])
            for u in us:add('unit_body_only',u['text'],unit_id=u['unit_id'])
            if Path(chunk['file_path']).suffix.lower()!='.pptx':continue
            path=chunk['file_path']
            if path not in presentations:presentations[path]=Presentation(path)
            slide=presentations[path].slides[chunk['slide_number']-1]
            heading='\n'.join(s.text for s in slide.shapes if s.has_text_frame and s.text.strip())
            for si,shape in enumerate(slide.shapes):
                if not shape.has_table:continue
                table=shape.table
                if any(c.is_spanned or c.is_merge_origin for row in table.rows for c in row.cells):continue
                headers=[c.text.strip() for c in table.rows[0].cells]
                if len(set(headers))!=len(headers) or sum(bool(x) for x in headers)<2:continue
                for ri,row in enumerate(table.rows):
                    if ri==0:continue
                    # A post-hoc first-row layout representation. This is not
                    # a production semantic-header detector or newly approved Gold.
                    text=prefix+heading+'\n'+'\n'.join(h+'：'+c.text for h,c in zip(headers,row.cells))
                    add('original_table_row_control',text,shape=si,row=ri,
                        header_assumption='unmerged distinct first row; diagnostic only, verify original table before adoption')
    write(OUT/'representation_protocol.json',dict(selected=[r['id'] for r in selected],representations=len(reps),
        comparison='replace one audited parent representation; all other parent vectors fixed, current question, same BGE',
        scope='source-local causal diagnostic; not whole-index results, not independently held-out acceptance',
        gold_runtime_use='Gold selects target parent for offline diagnostics only; no Gold text encoded as a query or added to production'))
    results=[]
    with no_network() as attempts:
        model=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(p['index']['embedding_repo_id'],p['index']['embedding_revision'],'embedding'))
        emb=LocalBgeEmbeddingProvider(model,repo_id=p['index']['embedding_repo_id'],revision=p['index']['embedding_revision'],batch_size=8)
        docs=np.asarray(emb.embed_documents([r['text'] for r in reps]),np.float32)
        queries=emb.embed_documents([emb.QUERY_PREFIX+r['query_plan']['semantic_query'] for r in selected])
        for record,q in zip(selected,queries):
            ps=np.full(len(parents),-np.inf,np.float32);np.maximum.at(ps,up,matrix@np.asarray(q,np.float32))
            for rep,dvec in zip(reps,docs):
                if rep['case']!=record['id']:continue
                score=float(dvec@q);old=float(ps[pi[rep['parent']]])
                rank=int(np.count_nonzero(ps>score))+1-int(old>score)
                token_count=emb.count_tokens(rep['text'])
                results.append(dict(**rep,baseline_rank=int(np.count_nonzero(ps>old))+1,replacement_rank=rank,
                    baseline_score=old,score=score,token_count=token_count,truncated=token_count>emb.max_sequence_length))
            print(record['id'],'representations',sum(x['case']==record['id'] for x in reps),flush=True)
        assert not attempts
    drift=[k for k,h in p['protected'].items() if sha(k)!=h]
    assert not drift
    write(OUT/'representation_controls.json',dict(api_calls=0,network_attempts=attempts,protected_drift=drift,
        cases=len(selected),results=results,original_quote_verification=verification,
        warning='All replacement ranks are source-local controls; cannot be called overall retrieval improvements. Full-parent >512 token controls are marked truncated.'))
    print('DONE',len(results),'local controls; verified',len(verification),flush=True)


if __name__=='__main__':main()
