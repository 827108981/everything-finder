"""Optional same-weight FP32 CPU execution backend check, no retrieval tuning."""
import time
import numpy as np
from tools.validate_vector_packaging_20261116 import *
import torch
import onnxruntime as ort
store,bundle,settings=inputs();emb=model(bundle)
transformer=emb._model[0].auto_model
path=OUT/'embedding_fp32_verified.onnx'
sample=emb._model.tokenizer(['核对同一模型、同一正文及编码数值。'],return_tensors='pt')
names=['input_ids','attention_mask','token_type_ids']
if not path.exists():
    class Encoder(torch.nn.Module):
        def __init__(self,m):super().__init__();self.m=m
        def forward(self,input_ids,attention_mask,token_type_ids):
            output=self.m(input_ids=input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids,return_dict=False)[0]
            # bge-large-zh-v1.5 uses CLS pooling, confirmed below.
            return torch.nn.functional.normalize(output[:,0],p=2,dim=1)
    local_path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
    pooling_config=json.loads((local_path/'1_Pooling/config.json').read_text())
    assert pooling_config['pooling_mode_cls_token'] and not pooling_config['pooling_mode_mean_tokens']
    transformer.set_attn_implementation('eager')
    transformer.eval();torch.set_num_threads(3)
    wrapper=Encoder(transformer).eval()
    with torch.inference_mode():
        torch.onnx.export(wrapper,tuple(sample[n] for n in names),str(path),opset_version=17,
            input_names=names,output_names=['sentence_embedding'],dynamic_axes={**{n:{0:'batch',1:'sequence'} for n in names},'sentence_embedding':{0:'batch'}},
            dynamo=False)
    print('EXPORTED FP32',path,flush=True)
with ro(REPDB) as c:
    texts=[r[0] for r in c.execute('SELECT DISTINCT encoded FROM representations WHERE tokens BETWEEN 120 AND 140 ORDER BY encoded LIMIT 64')]
reference=np.asarray(emb.embed_documents(texts),np.float32)
results=[]
for threads in (3,6):
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    start=time.perf_counter();vectors=[]
    for n in range(0,len(texts),8):
        encoded=emb._model.tokenizer(texts[n:n+8],padding=True,truncation=True,max_length=512,return_tensors='np')
        vectors.extend(session.run(None,{k:v for k,v in encoded.items() if k in names})[0])
    vectors=np.asarray(vectors,np.float32);elapsed=time.perf_counter()-start
    row=dict(threads=threads,batch=8,texts=len(texts),seconds=elapsed,texts_per_second=len(texts)/elapsed,
             min_cosine=float(np.min(np.sum(vectors*reference,axis=1))),max_abs_error=float(np.max(np.abs(vectors-reference))),
             model_sha256=sha(path),quantized=False,precision='FP32',source_weights_revision=bundle.embedding_revision)
    results.append(row);write(OUT/'onnx_runtime_benchmark.json',results);print(json.dumps(row),flush=True)
