"""Controlled actual-inference experiment: body versus title plus body, same model."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.content_identity import canonicalize_with_offsets, canonical_content
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from tools.verify_rerank_scope_bonus import mappings

SOURCE = Path('reports/rerank_dev_holdout_20261101')
OUT = Path('reports/rerank_title_input_20261102_threads4')


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    old_protocol = json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    records = [json.loads(s) for s in (SOURCE/'cases.jsonl').open(encoding='utf-8')]
    records = [r for r in records if r['arm']=='pure']
    assert len(records)==60
    gold = {}
    for d in old_protocol['datasets'].values():
        p=Path(d['path'])
        assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
        for line in p.read_text(encoding='utf-8').splitlines():
            g=json.loads(line);gold[g['id']]=g
    settings=old_protocol['fixed_settings']
    spec=ModelSpec(settings['rag_reranker_repo_id'],settings['rag_reranker_revision'],'reranker')
    protocol={'frozen_at':datetime.now(timezone.utc).isoformat(), 'bundle_id':old_protocol['bundle_id'],
        'datasets':old_protocol['datasets'], 'model':spec.repo_id,'revision':spec.revision,
        'max_length':512,'batch_size':4,'device':'cpu','torch_threads':4,'interop_threads':1,'title_max_tokens':64,
        'execution_note':'Initial run with default 12 threads stopped after two questions due to excessive CPU overhead; no scoring/formula changes. Partial files retained in rerank_title_input_20261102.',
        'arms':{'body':'original body','title_body':'文档标题：{filename stem}\n正文：{original body}'},
        'ranking':'raw model sigmoid; ties fused score then chunk ID; no other ranking changes',
        'limits':'No structured model/version/section fields available; no inferred applicability. Existing headings and tables kept as body.',
        'scope':'Rerank stage only; no downstream replay, answer generation or official pending scoring.',
        'decision':'Do not adopt on aggregate alone: inspect paired evidence coverage losses, negatives, truncation. No tuning after results.',
        'note':'All 60 questions have previously been inspected; diagnostic/regression, not independent acceptance.',
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (OUT/'protocol.json').exists():
        assert not (OUT/'completion.json').exists(), 'Experiment already completed'
        frozen=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
        for key in ('bundle_id','datasets','model','revision','max_length','batch_size','torch_threads','interop_threads','title_max_tokens','arms','ranking'):
            assert frozen[key]==protocol[key], key
        dump(OUT/'resume.json',{'resumed_at':protocol['frozen_at'],'resume_script_sha256':protocol['script_sha256'],
                              'reason':'User turn interruption stopped the original process; completed cases retained.'})
        protocol=frozen
    else:
        dump(OUT/'protocol.json',protocol)
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active()
    assert bundle.bundle_id==protocol['bundle_id']
    conn=store.connect_read_only(bundle.metadata_path)
    for r in records:
        for c in r['candidate_stages']['rerank']:
            chunk=store.load_chunk(conn,c['chunk_id'])
            assert chunk and chunk.text==c['text'] and chunk.source.filename==c['filename']
    conn.close()
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    model=LocalBgeReranker(LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(spec),
        repo_id=spec.repo_id,revision=spec.revision,batch_size=4,max_length=512)
    tok=model.tokenizer;torch=model._torch
    with torch.inference_mode():
        sample=tok(['模型预热'],['模型预热'],return_tensors='pt');model.model(**sample)
    results=[json.loads(s) for s in (OUT/'cases.jsonl').read_text(encoding='utf-8').splitlines()] if (OUT/'cases.jsonl').exists() else []
    completed={r['id'] for r in results}
    assert len(completed)==len(results)
    for previous in results:
        assert previous['question']==gold[previous['id']]['question']
    started=time.monotonic()
    with (OUT/'cases.jsonl').open('a',encoding='utf-8') as target:
        for num,r in enumerate(records,1):
            if r['id'] in completed:
                continue
            candidates=r['candidate_stages']['rerank'];arms={};audit=[];max_diff=0.0
            # Alternate arm order across questions to reduce warm-up/order timing bias.
            for arm in (('body','title_body') if num%2 else ('title_body','body')):
                texts=[];prefixes=[]
                for c in candidates:
                    title=tok.decode(tok.encode(Path(c['filename']).stem,add_special_tokens=False)[:64])
                    prefix=f'文档标题：{title}\n正文：' if arm=='title_body' else ''
                    texts.append(prefix+c['text']);prefixes.append(prefix)
                scores=[];elapsed=0.0;char_ends=[];token_lengths=[]
                for start in range(0,len(candidates),4):
                    docs=texts[start:start+4];t=time.monotonic()
                    encoded=tok([r['question']]*len(docs),docs,padding=True,truncation=True,max_length=512,return_tensors='pt')
                    with torch.inference_mode():
                        scores.extend(torch.sigmoid(model.model(**encoded).logits.reshape(-1)).cpu().tolist())
                    elapsed+=time.monotonic()-t
                for c,doc,prefix in zip(candidates,texts,prefixes,strict=True):
                    enc=tok(r['question'],doc,truncation=True,max_length=512,return_offsets_mapping=True)
                    ends=[b for (a,b),seq in zip(enc['offset_mapping'],enc.sequence_ids(),strict=True) if seq==1]
                    char_ends.append(max(0,(max(ends) if ends else 0)-len(prefix)))
                    token_lengths.append(len(tok(r['question'],doc,truncation=False)['input_ids']))
                scored=[dict(c,rerank_score=float(s)) for c,s in zip(candidates,scores,strict=True)]
                scored.sort(key=lambda c:(-c['rerank_score'],-c['fused_score'],c['chunk_id']))
                maps=mappings(r,gold[r['id']])
                arms[arm]={'inference_seconds':elapsed,'scores':{str(k):score_stage('rerank',scored[:k],maps) for k in (1,3,8,20)},
                    'order':[c['chunk_id'] for c in scored], 'raw_scores':dict(zip((c['chunk_id'] for c in candidates),scores)),
                    'body_visible_end':dict(zip((c['chunk_id'] for c in candidates),char_ends)),
                    'untruncated_pair_tokens':dict(zip((c['chunk_id'] for c in candidates),token_lengths))}
                if arm=='body':
                    max_diff=max(abs(s-c['rerank_score']) for c,s in zip(candidates,scores,strict=True))
                    assert max_diff<1e-6,(r['id'],max_diff)
                    assert arms[arm]['order']==[c['chunk_id'] for c in candidates]
            # Inspect complete single-chunk gold quotes without using them in model input.
            for e in gold[r['id']]['gold_evidence']:
                quote=canonical_content(e['quote'])
                for c in candidates:
                    canon=canonicalize_with_offsets(c['text']);pos=canon.text.find(quote)
                    if quote and pos>=0:
                        needed=canon.offsets[pos+len(quote)-1][1]
                        audit.append({'file':c['file_path'],'chunk_id':c['chunk_id'],'quote':e['quote'],
                            'char_end_needed':needed,
                            'body_visible':needed<=arms['body']['body_visible_end'][c['chunk_id']],
                            'title_body_visible':needed<=arms['title_body']['body_visible_end'][c['chunk_id']]})
            row={'id':r['id'],'question':r['question'],'cohort':r['cohort'],'group_id':r['group_id'],
                'review_status':r['review_status'],'expected_behavior':r['expected_behavior'],
                'baseline_max_score_difference':max_diff,'arms':arms,'single_chunk_quote_audit':audit}
            target.write(json.dumps(row,ensure_ascii=False)+'\n');target.flush();results.append(row)
            print(f'{num}/60 {r["id"]} body={arms["body"]["scores"]["8"]["gold_evidence_recall"]} title={arms["title_body"]["scores"]["8"]["gold_evidence_recall"]}',flush=True)
    for d in protocol['datasets'].values():assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    assert store.active().bundle_id==protocol['bundle_id']
    summary={}
    for cohort in ('development','holdout_pending'):
        rows=[r for r in results if r['cohort']==cohort and r['expected_behavior']!='refuse'];summary[cohort]={}
        for arm in ('body','title_body'):
            summary[cohort][arm]={str(k):{'n':len(rows),'hit':sum(r['arms'][arm]['scores'][str(k)]['gold_hit'] for r in rows),
                'coverage':sum(r['arms'][arm]['scores'][str(k)]['gold_evidence_recall'] for r in rows)/len(rows),
                'mrr':sum(r['arms'][arm]['scores'][str(k)]['reciprocal_rank'] for r in rows)/len(rows),
                'negatives':sum(r['arms'][arm]['scores'][str(k)]['hard_negative_found'] for r in rows)} for k in (1,3,8)}
    dump(OUT/'summary.json',summary)
    dump(OUT/'completion.json',{'count':len(results),'pairs':2400,'seconds_current_process':time.monotonic()-started,
        'gold_unchanged':True,'api_calls':0,'index_bodies_verified':1200})
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
