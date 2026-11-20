"""Fresh disabled lexical retrieval plus frozen-rerank downstream replay, no LLM."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/rag_general_refactor_20261113'

CHILD=r'''
import json,sys
from dataclasses import asdict
from pathlib import Path
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import RagQuery,EvidenceChunk
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
out=Path(sys.argv[1]);arm=sys.argv[2]
store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
s=json.loads((out/'A_manifest.json').read_text(encoding='utf8'))['settings']
lexical=LexicalRetriever(store,bundle.metadata_path,backend=s['rag_lexical_backend'],bundle=bundle)
result=[]
with store.connect_read_only(bundle.metadata_path) as conn:
 for raw in (out/'disabled_replay_inputs.jsonl').read_text(encoding='utf8').splitlines():
  r=json.loads(raw);p=QueryInterpreter(enhanced=False).interpret(r['question']);q=RagQuery(r['question'])
  hits=lexical.retrieve(q,CancelToken(),p)
  candidates=[]
  for c in r['candidates']:
   chunk=store.load_chunk(conn,c['chunk_id'])
   members=tuple(c.get('member_chunk_ids') or [chunk.stable_chunk_id])
   x=EvidenceChunk(chunk=chunk,member_chunk_ids=members,sources=tuple(store.load_chunk(conn,i).source for i in members))
   for k in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined','answer_support_score'):
    if c.get(k) is not None:setattr(x,k,c[k])
   candidates.append(x)
  constrained=ConstraintMatcher().evaluate(p,candidates)
  d=EvidencePolicy(min_vector_score=s['rag_min_vector_score'],min_rerank_score=s['rag_min_rerank_score'],max_per_document=s['rag_max_chunks_per_document']).evaluate(q,constrained,plan=p)
  ctx=ContextBuilder().build(d.accepted,max_chars=s['rag_context_max_chars'])
  result.append({'id':r['id'],'lexical':[(i.chunk.stable_chunk_id,i.lexical_score) for i in hits],
   'constraints':[(i.chunk.stable_chunk_id,asdict(i.constraint_decision)) for i in constrained],
   'accepted':[i.chunk.stable_chunk_id for i in d.accepted],'rejected':d.rejected,'reason':d.reason,'context':ctx.text,'entries':ctx.entries})
(out/('disabled_'+arm+'_replay.json')).write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
'''

def main():
    with (OUT/'A.jsonl').open(encoding='utf8') as source,(OUT/'disabled_replay_inputs.jsonl').open('w',encoding='utf8') as dest:
        for line in source:
            r=json.loads(line)
            dest.write(json.dumps({'id':r['id'],'question':r['question'],'candidates':r['candidate_stages'].get('rerank',[])},ensure_ascii=False)+'\n')
    snapshot=Path(json.loads((OUT/'freeze.json').read_text(encoding='utf8'))['snapshot'])
    for arm,cwd in [('A',snapshot),('B',ROOT)]:
        subprocess.run([sys.executable,'-c',CHILD,str(OUT),arm],cwd=cwd,check=True)
    a=json.loads((OUT/'disabled_A_replay.json').read_text(encoding='utf8'));b=json.loads((OUT/'disabled_B_replay.json').read_text(encoding='utf8'))
    assert len(a)==len(b)==91
    result={'cases':91,'differences':[{'id':x['id'],'fields':[k for k in x if x[k]!=y[k]]} for x,y in zip(a,b) if x!=y],
      'method':'Fresh disabled BM25 against same published bundle; same saved rerank candidates replayed through disabled constraints, evidence policy and Context. Vector/reranker not rerun; not an end-to-end equivalence proof.'}
    (OUT/'disabled_compatibility.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(result)

if __name__=='__main__':main()
