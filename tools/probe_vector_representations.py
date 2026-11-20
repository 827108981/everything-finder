"""Post-hoc representation controls; never a runtime retriever or acceptance run.

Keep the production model and all rival vectors fixed. Replace only an audited
source's representation with its full parent or original PPT table rows. Rank
against exhaustive parent-max cosine scores. This tests local attribution, not
the effect of rebuilding the complete index (competition would also change).
No answer terms or Gold are supplied to the encoder; Gold selects audit cases.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import hashlib
import json
from pathlib import Path
import numpy as np
from pptx import Presentation
from tools.diagnose_vector_recall import OUT, ro
from tools.validate_selection_factorial import ROOT, no_network, write, sha
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider


def main():
    import torch
    torch.set_num_threads(3)
    protocol = json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    cases = [json.loads(x) for x in (OUT/'diagnosis.jsonl').read_text(encoding='utf8').splitlines()]
    cases = [r for r in cases if any(t['exact_completion_rank'] > 50 for t in r['targets']) or r['id']=='rag-biz-006']
    with ro(protocol['vector_index']['database_path']) as c:
        units = [dict(r) for r in c.execute('SELECT * FROM vector_units ORDER BY unit_id')]
    parent_ids = list(dict.fromkeys(u['parent_chunk_id'] for u in units))
    parent_index = {p:i for i,p in enumerate(parent_ids)}
    unit_parent = np.asarray([parent_index[u['parent_chunk_id']] for u in units])
    matrix = np.empty((len(units),1024),np.float32)
    with ro('E:/everythingfinder-db/rag/embedding_cache.db') as c:
        for i,u in enumerate(units):
            row=c.execute('SELECT vector_blob FROM embedding_cache WHERE model_id=? AND model_revision=? AND dimension=1024 AND text_sha256=?',
                (protocol['model'],protocol['revision'],hashlib.sha256(u['embedding_text'].encode()).hexdigest())).fetchone()
            assert row is not None
            matrix[i]=np.frombuffer(row[0],dtype='<f4')
    representations=[]
    for r in cases:
        seen=set()
        for t in r['targets']:
            for p in t['parents']:
                cid=p['chunk_id']
                if cid in seen: continue
                seen.add(cid)
                context='\n'.join(p['units'][0]['embedding_text'].split('\n')[:2])+'\n正文：'
                representations.append(dict(case=r['id'],parent=cid,kind='full_parent',text=context+p['text']))
                for u in p['units']:
                    representations.append(dict(case=r['id'],parent=cid,kind='unit_body_only',unit_id=u['unit_id'],text=u['text']))
                if Path(t['file']).suffix.lower()!='.pptx':continue
                slide=Presentation(t['file']).slides[int(t['page'])-1]
                heading='\n'.join(s.text for s in slide.shapes if s.has_text_frame and s.text.strip())
                for si,s in enumerate(slide.shapes):
                    if not s.has_table:continue
                    table=s.table
                    headers=[c.text for c in table.rows[0].cells]
                    for ri,row in enumerate(table.rows):
                        if not ri:continue
                        text=context+heading+'\n'+'\n'.join(h+'：'+c.text for h,c in zip(headers,row.cells))
                        representations.append(dict(case=r['id'],parent=cid,kind='original_table_row',table_shape=si,row=ri,text=text))
    extra_queries={
        # Preserve "start detection"; "first result" is a different endpoint.
        'rag-row-holdout-003':['BS-2000系列从待机到首个样本启动检测需要多久？','BS-2800M系列从待机到首个样本启动检测需要多久？'],
        'evidence-flow-independent-05':['A&T 和 Medica ISE 的测量原理有什么不同？'],
        'evidence-flow-independent-06':['BS-2800M 和 BS-430 分析部跟电脑的通讯方式分别是什么？'],
    }
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    with no_network() as attempts:
        emb=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(protocol['model'],protocol['revision'],'embedding')),
            repo_id=protocol['model'],revision=protocol['revision'],batch_size=8)
        docs=np.asarray(emb.embed_documents([p['text'] for p in representations]),np.float32)
        queries=[(r['id'],'original',r['semantic_query']) for r in cases]
        queries += [(k,'question_only_control',q) for k,qs in extra_queries.items() for q in qs]
        qvecs=np.asarray(emb.embed_documents([emb.QUERY_PREFIX+q for _,_,q in queries]),np.float32)
        results=[]
        for (case,kind,q),vec in zip(queries,qvecs):
            ps=np.full(len(parent_ids),-np.inf,np.float32)
            np.maximum.at(ps,unit_parent,matrix@vec)
            ordered=np.sort(ps)[::-1]
            for rep,dvec in zip(representations,docs):
                if rep['case']!=case:continue
                score=float(dvec@vec);old=float(ps[parent_index[rep['parent']]])
                # Replace this parent only; all competitor representations fixed.
                rank=int(np.count_nonzero(ps>score)) + 1 - int(old>score)
                tokens=emb._model.tokenizer(rep['text'],truncation=False)['input_ids']
                visible=emb._model.tokenizer.decode(tokens[:emb.max_sequence_length],skip_special_tokens=True)
                results.append(dict(**rep,query=q,query_kind=kind,baseline_rank=int(np.count_nonzero(ps>old))+1,
                    replacement_rank=rank,score=score,baseline_score=old,token_count=len(tokens),truncated=len(tokens)>emb.max_sequence_length,
                    encoder_visible_text=visible))
        assert not attempts
    write(OUT/'representation_controls.json',dict(
        scope='post-hoc source-local replacement; not full-index rebuild or independent gain',
        global_vectors_fixed=len(units),representations=len(representations),queries=len(queries),api_calls=0,
        invalid_prior_control='003 prior first-result wording changes endpoint; excluded from conclusions; corrected start-detection controls here',
        results=results,protected_drift=[p for p,h in protocol['protected'].items() if sha(ROOT/p)!=h]))
    print('Wrote',len(results),'controlled ranks; no index, Gold or runtime changes.',flush=True)


if __name__=='__main__':main()
