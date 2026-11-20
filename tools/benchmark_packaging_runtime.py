"""Resource-only batching benchmark, fixed source strings/model/FP32 vectors."""
import time
import json
import numpy as np
from tools.validate_vector_packaging_20261116 import *
store,bundle,settings=inputs()
emb=model(bundle)
with ro(REPDB) as c:
    texts=[r[0] for r in c.execute('SELECT DISTINCT encoded FROM representations WHERE tokens BETWEEN 120 AND 140 ORDER BY encoded LIMIT 96')]
import torch
results=[];reference=None
for threads,batch in [(3,8),(3,16),(6,16),(6,32),(8,16),(12,16)]:
    torch.set_num_threads(threads);emb._batch_size=batch
    start=time.perf_counter();vectors=np.asarray(emb.embed_documents(texts),np.float32);elapsed=time.perf_counter()-start
    if reference is None:reference=vectors
    row=dict(threads=threads,batch=batch,texts=len(texts),seconds=elapsed,texts_per_second=len(texts)/elapsed,
             min_cosine=float(np.min(np.sum(vectors*reference,axis=1))),max_abs_error=float(np.max(np.abs(vectors-reference))))
    results.append(row);write(OUT/'runtime_benchmark.json',results);print(json.dumps(row),flush=True)
