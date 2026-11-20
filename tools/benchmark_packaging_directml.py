"""Same ONNX FP32 weights, optional integrated-GPU feasibility check."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_vector_packaging_20261116'
sys.path.insert(0,str(OUT/'runtime_directml'))
import onnxruntime as ort
import numpy as np
import time
from tools.validate_vector_packaging_20261116 import inputs,ro,REPDB,write,sha,LocalModelManager,RAG_MODEL_CACHE_DIR,ModelSpec
from transformers import AutoTokenizer
store,bundle,settings=inputs()
path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
with ro(REPDB) as c: texts=[r[0] for r in c.execute('SELECT DISTINCT encoded FROM representations WHERE tokens BETWEEN 120 AND 140 ORDER BY encoded LIMIT 32')]
print('PROVIDERS',ort.get_available_providers(),flush=True)
options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1;options.enable_mem_pattern=False
options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
names=['input_ids','attention_mask','token_type_ids']
encoded=tokenizer(texts,padding=True,truncation=True,max_length=512,return_tensors='np')
cpu=ort.InferenceSession(str(OUT/'embedding_fp32_verified.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
reference=cpu.run(None,dict(encoded))[0];del cpu
session=ort.InferenceSession(str(OUT/'embedding_fp32_verified.onnx'),sess_options=options,providers=['DmlExecutionProvider','CPUExecutionProvider'])
results=[]
for batch in (4,8,16):
    start=time.perf_counter();vectors=[]
    for i in range(0,len(texts),batch):
        inp=tokenizer(texts[i:i+batch],padding=True,truncation=True,max_length=512,return_tensors='np')
        vectors.extend(session.run(None,dict(inp))[0])
    vectors=np.asarray(vectors,np.float32);elapsed=time.perf_counter()-start
    row=dict(batch=batch,texts=len(texts),seconds=elapsed,texts_per_second=len(texts)/elapsed,
        min_cosine=float(np.min(np.sum(vectors*reference,axis=1))),max_abs_error=float(np.max(np.abs(vectors-reference))),
        providers=session.get_providers(),model_sha256=sha(OUT/'embedding_fp32_verified.onnx'),precision='FP32',quantized=False)
    results.append(row);write(OUT/'directml_runtime_benchmark.json',results);print(row,flush=True)
