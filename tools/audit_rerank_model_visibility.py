"""Measure actual BGE-visible character range; no new inference or Gold selection."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import json
from pathlib import Path

from tools.validate_rerank_semantic import OUT, route, read_jsonl
from tools.validate_selection_factorial import load_inputs, write, no_network
from tools.run_rag_experiment import LocalModelManager, RAG_MODEL_CACHE_DIR, ModelSpec
from local_full_text_search.rag.content_identity import canonicalize_with_offsets


def main():
    from transformers import AutoTokenizer
    rows, settings, _, mappings=load_inputs()
    records=[]
    with no_network() as attempts:
        manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
        path=manager.resolve_local_path(ModelSpec(settings['rag_reranker_repo_id'],settings['rag_reranker_revision'],'reranker'))
        tokenizer=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
        assert tokenizer.is_fast,'Exact offset audit requires a fast tokenizer'
        for row in rows:
            routed=route(row)
            for c in row['candidate_stages']['rerank']:
                encoded=tokenizer(row['question'],c['text'],truncation=True,max_length=settings['rag_reranker_max_length'],return_offsets_mapping=True)
                ranges=[span for span,side in zip(encoded['offset_mapping'],encoded.sequence_ids()) if side==1 and span[1]>span[0]]
                start=min((s[0] for s in ranges),default=0);end=max((s[1] for s in ranges),default=0)
                gold=[]; norm=canonicalize_with_offsets(c['text'])
                for m in mappings[row['id']]:
                    if m.kind!='positive':continue
                    quote=canonicalize_with_offsets(m.quote).text;found=norm.text.find(quote)
                    # This is a visibility diagnostic, not source-bound scoring.
                    # Formal matching remains the existing EvidenceAlternative matcher.
                    if found<0 or not quote:continue
                    a=norm.offsets[found][0];b=norm.offsets[found+len(quote)-1][1]
                    gold.append(dict(evidence_id=m.evidence_id,start=a,end=b,visible_to_bge=a>=start and b<=end))
                records.append(dict(case_id=row['id'],chunk_id=c['chunk_id'],routed_to_semantic=c['chunk_id'] in routed,
                    body_chars=len(c['text']),bge_body_start=start,bge_body_end=end,
                    bge_truncated=end<len(c['text'].rstrip()),bge_tokens=len(encoded['input_ids']),gold_quotes=gold))
        outputs=read_jsonl(OUT/'model_progress.jsonl') if (OUT/'model_progress.jsonl').exists() else []
        lookup={(r['case_id'],r['chunk_id']):r for r in records}
        semantic=[]
        for o in outputs:
            visible=lookup[o['case_id'],o['chunk_id']]
            j=o['judgement']
            semantic.append(dict(case_id=o['case_id'],chunk_id=o['chunk_id'],support=j['support'],valid=j['valid'],error=j['error'],
                model_input_tokens=j['input_tokens'],model_output_tokens=j['output_tokens'],
                quotes=[{**s,'visible_to_bge':s['start']>=visible['bge_body_start'] and s['end']<=visible['bge_body_end']} for s in j['spans']]))
        write(OUT/'model_visibility.json',dict(candidate_count=len(records),records=records,semantic=semantic,
            tokenizer_path=str(path),bge_max_length=settings['rag_reranker_max_length'],
            semantic_visible_body='full candidate; input over 4096 budget abstains, no silent truncation',
            api_calls=0,network_attempts=attempts,
            semantic_capacity_conclusion='A benefit beyond BGE visibility cannot prove superior semantic inference without the matched-body control'))
        assert not attempts
    print('Visibility audited:',len(records),'model judgements:',len(semantic),flush=True)


if __name__=='__main__':main()
