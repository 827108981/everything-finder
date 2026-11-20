"""Experiment-only FP32 execution backend; never imported by the application."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

class VerifiedFp32Runtime:
    def __init__(self,output,model_path,bundle,batch_size=16):
        # Optional runtime lives in the experiment, not application dependencies.
        sys.path.insert(0,str(Path(output)/'runtime_directml'))
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
        options=ort.SessionOptions();options.intra_op_num_threads=3;options.inter_op_num_threads=1
        options.enable_mem_pattern=False;options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        self.session=ort.InferenceSession(str(Path(output)/'embedding_fp32_verified.onnx'),sess_options=options,
                                        providers=['DmlExecutionProvider','CPUExecutionProvider'])
        assert 'DmlExecutionProvider' in self.session.get_providers()
        self.session_options=options
        self.onnx_path=Path(output)/'embedding_fp32_verified.onnx'
        self.device_switches=[]
        self.batch_size=batch_size;self.model_id=bundle.embedding_repo_id;self.model_revision=bundle.embedding_revision
        self.dimension=bundle.embedding_dimension
    def count_tokens(self,text):return len(self.tokenizer(str(text),add_special_tokens=True,truncation=False)['input_ids'])
    def embed_documents(self,texts):
        if not texts:return []
        # Same output order and input strings; no sampling/quantization/prompt.
        output=[]
        for n in range(0,len(texts),self.batch_size):
            batch=texts[n:n+self.batch_size]
            inputs=dict(self.tokenizer(batch,padding=True,truncation=True,max_length=512,return_tensors='np'))
            try:
                vectors=np.asarray(self.session.run(None,inputs)[0],np.float32)
            except Exception as exc:
                providers=self.session.get_providers()
                if 'DmlExecutionProvider' not in providers: raise
                # DirectML can lose an integrated GPU after long runs. Rebuild
                # the same FP32 graph on CPU and retry this uncommitted batch;
                # earlier cache commits remain valid and are never recomputed.
                self.session=ort.InferenceSession(str(self.onnx_path),sess_options=self.session_options,
                                                  providers=['CPUExecutionProvider'])
                self.device_switches.append(dict(from_provider='DmlExecutionProvider',to_provider='CPUExecutionProvider',
                                                 batch_start=n,error_type=type(exc).__name__,error=str(exc)[:500]))
                vectors=np.asarray(self.session.run(None,inputs)[0],np.float32)
            if vectors.shape!=(len(batch),self.dimension) or not np.isfinite(vectors).all():raise ValueError('Invalid FP32 runtime output')
            output.extend(vectors.tolist())
        return output

def validate():
    from tools.validate_vector_packaging_20261116 import inputs,model,ro,REPDB,OUT,write,sha,no_network,np
    from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    store,bundle,settings=inputs()
    with ro(REPDB) as c:
        rows=[tuple(r) for r in c.execute('SELECT encoded,MAX(tokens) FROM representations GROUP BY encoded ORDER BY MAX(tokens),encoded')]
    indices=np.linspace(0,len(rows)-1,256,dtype=int)
    texts=[rows[i][0] for i in indices]
    with no_network() as attempts:
        reference_model=model(bundle)
        reference=np.asarray(reference_model.embed_documents(texts),np.float32)
        counts=[reference_model.count_tokens(t) for t in texts]
        del reference_model
        import gc;gc.collect()
        path=LocalModelManager(RAG_MODEL_CACHE_DIR).resolve_local_path(ModelSpec(bundle.embedding_repo_id,bundle.embedding_revision,'embedding'))
        runtime=VerifiedFp32Runtime(OUT,path,bundle)
        actual=np.asarray(runtime.embed_documents(texts),np.float32)
        actualcounts=[runtime.count_tokens(t) for t in texts]
    cos=np.sum(reference*actual,axis=1);error=np.max(np.abs(reference-actual))
    passed=bool(min(cos)>.99999 and error<2e-5 and counts==actualcounts and not attempts)
    audit=dict(passed=passed,sample=256,indices=indices.tolist(),token_range=[min(counts),max(counts)],
        min_cosine=float(min(cos)),max_abs_error=float(error),thresholds=dict(min_cosine=.99999,max_abs_error=2e-5),
        quantized=False,precision='FP32',weights_revision=bundle.embedding_revision,
        onnx_sha256=sha(OUT/'embedding_fp32_verified.onnx'),providers=runtime.session.get_providers(),
        network_attempts=attempts,api_calls=0,used_for_queries=False,used_for_rerank=False,
        reason='Execution feasibility only, same revision and tokenization; not a retrieval/model experiment. Query and rerank remain unchanged.')
    write(OUT/'encoding_runtime_audit.json',audit)
    assert passed,audit
    print('FP32 execution parity passed',audit,flush=True)

if __name__=='__main__':validate()
