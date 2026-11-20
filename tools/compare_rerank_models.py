"""Pinned, resumable local model comparison on frozen candidate bodies."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from tools.verify_rerank_scope_bonus import mappings

ROOT=Path('reports/rerank_models_20261102')
SOURCE=Path('reports/rerank_dev_holdout_20261101')
CACHE=Path('E:/everythingfinder-db/experiments/rerank_model_cache')
MODELS={'bge_v2_m3':('BAAI/bge-reranker-v2-m3','953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'),
        'qwen3_06b':('Qwen/Qwen3-Reranker-0.6B','e61197ed45024b0ed8a2d74b80b4d909f1255473')}
TASK='Given a web search query, retrieve relevant passages that answer the query'
PREFIX='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'


def dump(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def prepare():
    ROOT.mkdir(exist_ok=True,parents=True)
    original=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'))
    gold={}
    for d in original['datasets'].values():
        p=Path(d['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==d['sha256']
        for line in p.read_text(encoding='utf-8').splitlines():
            g=json.loads(line);gold[g['id']]=g
    records=[json.loads(s) for s in (SOURCE/'cases.jsonl').open(encoding='utf-8')]
    records=[r for r in records if r['arm']=='pure'];assert len(records)==60
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'bundle_id':original['bundle_id'],
        'datasets':original['datasets'],'models':MODELS,'base_model':original['fixed_settings']['rag_reranker_repo_id'],
        'base_revision':original['fixed_settings']['rag_reranker_revision'],
        'input':'Same original candidate body text; no title, inferred metadata, rule scores or rank guard.',
        'max_length':512,'batch_size':4,'device':'cpu','torch_threads':4,'interop_threads':1,'dtype':'float32',
        'tokenization_note':'Same total token limit is not the same character budget; official Qwen prompt consumes part of that limit. Record truncation and visible body extent.',
        'qwen_instruction':TASK,'qwen_scoring':'Official template, yes/no logit difference, softmax probability; no generation; logits_to_keep=1.',
        'baseline':'Reuse the 1200 actual body-only scores and timings from completed title-input experiment; same fixed CPU/batch settings.',
        'evaluation':'Rank-only diagnostic, not independent acceptance. Pending stays pending; incomplete equivalent-source Gold limits interpretation.',
        'decision':'Compare top1/3/8 evidence and per-case losses/negatives alongside latency/memory. Do not tune per model on case results.',
        'sources':['https://huggingface.co/BAAI/bge-reranker-v2-m3','https://huggingface.co/Qwen/Qwen3-Reranker-0.6B'],
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    if (ROOT/'protocol.json').exists():
        old=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
        for key in ('bundle_id','datasets','models','input','max_length','batch_size','dtype','qwen_instruction'):
            assert old[key]==json.loads(json.dumps(protocol[key])),key
    else:dump(ROOT/'protocol.json',protocol)
    return original,gold,records


def download():
    prepare();CACHE.mkdir(exist_ok=True,parents=True)
    for name,(repo,sha) in MODELS.items():
        print('Downloading',repo,sha,flush=True)
        p=snapshot_download(repo_id=repo,revision=sha,cache_dir=str(CACHE),max_workers=4,
            allow_patterns=['*.json','*.safetensors','*.model','merges.txt','vocab.txt','README.md'])
        print('Ready',name,p,flush=True)


def run(name):
    import psutil
    import torch
    from transformers import AutoModelForSequenceClassification,AutoModelForCausalLM,AutoTokenizer
    original,gold,records=prepare();repo,sha=MODELS[name]
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==original['bundle_id']
    conn=store.connect_read_only(bundle.metadata_path)
    for r in records:
        for c in r['candidate_stages']['rerank']:
            chunk=store.load_chunk(conn,c['chunk_id']);assert chunk and chunk.text==c['text']
    conn.close()
    path=snapshot_download(repo_id=repo,revision=sha,cache_dir=str(CACHE),local_files_only=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    start=time.monotonic();is_qwen=name.startswith('qwen')
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,padding_side='left' if is_qwen else 'right')
    cls=AutoModelForCausalLM if is_qwen else AutoModelForSequenceClassification
    model=cls.from_pretrained(path,local_files_only=True,dtype=torch.float32).eval()
    load_seconds=time.monotonic()-start
    proc=psutil.Process();peak=proc.memory_info().rss
    pre=tok.encode(PREFIX,add_special_tokens=False) if is_qwen else []
    suf=tok.encode(SUFFIX,add_special_tokens=False) if is_qwen else []
    yes=tok.convert_tokens_to_ids('yes') if is_qwen else None
    no=tok.convert_tokens_to_ids('no') if is_qwen else None
    def forward(query,docs):
        nonlocal peak
        details=[];scoring_start=time.monotonic()
        if is_qwen:
            inputs=[]
            for doc in docs:
                header=f'<Instruct>: {TASK}\n<Query>: {query}\n<Document>: '
                text=header+doc
                budget=512-len(pre)-len(suf)
                ids=tok(text,add_special_tokens=False,truncation=True,max_length=budget)['input_ids']
                inputs.append({'input_ids':pre+ids+suf,'attention_mask':[1]*(len(pre)+len(ids)+len(suf))})
            enc=tok.pad(inputs,padding=True,return_tensors='pt')
        else:
            enc=tok([query]*len(docs),docs,padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.inference_mode():
            if is_qwen:
                logits=model(**enc,logits_to_keep=1,use_cache=False).logits[:,-1,:]
                values=torch.softmax(torch.stack([logits[:,no],logits[:,yes]],dim=1),dim=1)[:,1].tolist()
            else:values=torch.sigmoid(model(**enc).logits.reshape(-1)).tolist()
        scoring_seconds=time.monotonic()-scoring_start
        peak=max(peak,proc.memory_info().rss)
        for doc in docs:
            if is_qwen:
                header=f'<Instruct>: {TASK}\n<Query>: {query}\n<Document>: '
                full=tok(header+doc,add_special_tokens=False,return_offsets_mapping=True)
                offsets=full['offset_mapping'][:512-len(pre)-len(suf)]
                end=max(0,max((b for a,b in offsets),default=0)-len(header))
                details.append({'pair_tokens':len(full['input_ids'])+len(pre)+len(suf),'visible_body_end':end})
            else:
                full=tok(query,doc,truncation=False)
                cut=tok(query,doc,truncation=True,max_length=512,return_offsets_mapping=True)
                end=max((b for (a,b),seq in zip(cut['offset_mapping'],cut.sequence_ids()) if seq==1),default=0)
                details.append({'pair_tokens':len(full['input_ids']),'visible_body_end':end})
        return values,details,scoring_seconds
    sanity_scores,_,_=forward('中国的首都是哪里？',['中国的首都是北京。','液路蠕动泵吸液流量需要调试。'])
    assert sanity_scores[0]>sanity_scores[1],sanity_scores
    dump(ROOT/(name+'_environment.json'),{'repo':repo,'revision':sha,'load_seconds':load_seconds,
        'torch':torch.__version__,'sanity_scores':sanity_scores,'sanity_passed':True,
        'execution_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    output=ROOT/(name+'_cases.jsonl')
    done=[json.loads(s) for s in output.read_text(encoding='utf-8').splitlines()] if output.exists() else []
    completed={r['id'] for r in done};assert len(done)==len(completed)
    start=time.monotonic()
    with output.open('a',encoding='utf-8') as f:
        for num,r in enumerate(records,1):
            if r['id'] in completed:continue
            candidates=r['candidate_stages']['rerank'];scores=[];audit=[];elapsed=0.0
            for i in range(0,len(candidates),4):
                batch=candidates[i:i+4];values,detail,seconds=forward(r['question'],[c['text'] for c in batch])
                scores.extend(values);audit.extend(detail)
                elapsed+=seconds
            scored=[dict(c,rerank_score=s) for c,s in zip(candidates,scores,strict=True)]
            scored.sort(key=lambda c:(-c['rerank_score'],-c['fused_score'],c['chunk_id']))
            maps=mappings(r,gold[r['id']])
            row={'id':r['id'],'question':r['question'],'group_id':r['group_id'],'cohort':r['cohort'],
                'expected_behavior':r['expected_behavior'],'review_status':r['review_status'],
                'seconds':elapsed,'peak_rss_bytes':peak,'order':[c['chunk_id'] for c in scored],
                'candidates':{c['chunk_id']:dict(score=s,**a) for c,s,a in zip(candidates,scores,audit,strict=True)},
                'scores':{str(k):score_stage('rerank',scored[:k],maps) for k in (1,3,8,20)}}
            f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();done.append(row)
            print(name,f'{num}/60',r['id'],'hit8',row['scores']['8']['gold_hit'],f'{elapsed:.2f}s',flush=True)
    assert len(done)==60 and store.active().bundle_id==original['bundle_id']
    for d in original['datasets'].values():assert hashlib.sha256(Path(d['path']).read_bytes()).hexdigest()==d['sha256']
    dump(ROOT/(name+'_completion.json'),{'n':60,'pairs':1200,'peak_rss_bytes':peak,'current_process_seconds':time.monotonic()-start,
        'gold_unchanged':True,'load_seconds':load_seconds,'api_calls':0})
    del model;gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--model',choices=MODELS)
    args=p.parse_args()
    if args.download:download()
    if args.model:run(args.model)
