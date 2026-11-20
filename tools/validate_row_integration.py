"""Frozen-protocol row integration and explicit budget sensitivity; no prod writes."""
from collections import Counter,defaultdict
from dataclasses import replace
from datetime import datetime
import hashlib,json,math
from pathlib import Path
from tools.validate_material_repairs import ROOT,select_preserving,sha
from local_full_text_search.config.constants import SETTINGS_PATH,RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.models import EvidenceChunk,RagQuery
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.retrieval.lexical_analyzer import get_lexical_analyzer
from local_full_text_search.rag.reranking.support import compute_answer_support
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
from local_full_text_search.rag.trace import candidate_snapshot

OUT=ROOT/'reports/rag_row_integration_20261105'
SOURCE=ROOT/'reports/rag_material_repairs_20261105'
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    prior=json.loads((SOURCE/'protocol.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in prior['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    hashes={**prior['hashes'],str(SOURCE/'bound_rows.json'):sha(SOURCE/'bound_rows.json'),__file__:sha(__file__)}
    dump('protocol.json',{'started':str(datetime.now()),'hashes':hashes,'bundle_id':prior['bundle_id'],
      'H8':'2287 corrected bound rows; BM25 existing tokenizer, k1=1.2 b=.75 top10; fresh actual row cross-encoder scores, existing weights and hard conditions. At most ONE best eligible row added to baseline free slot; no evictions; final8/12000. Query/Gold-independent row index. Not a full vector reindex.',
      'S8_S12_S16':'Independent budget diagnostic using baseline and historical A scored source-page pools. Preserve baseline; exact-source duplicate suppression and hard-match-only additions. Page caps explicitly8/12/16; same12000chars. Never claim equal-budget gain or production readiness for12/16.',
      'stop':'No production promotion with lost positive quotes, new marked negative quotes or answer regressions. No tuning after start.',
      'answer_plan':'020 identical-context repetition baseline/candidate each5; H8/023 baseline/candidate each3 if row inclusion succeeds. Budget variants initially evidence only.'})
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==prior['bundle_id']
    conn=store.connect_read_only(bundle.metadata_path);index=load_chunk_rows(conn);pages=defaultdict(list)
    for r in index:pages[r['file_path'],r.get('slide_number') or r.get('page_number')].append(r['chunk_id'])
    units=[]
    for b in json.loads((SOURCE/'bound_rows.json').read_text(encoding='utf-8')):
        value='列绑定（空值表示原单元格为空）：\n'+json.dumps(b['bindings'],ensure_ascii=False)+'\n原文行：\n'+'\n'.join(b['source_values'])
        uid=hashlib.sha256((b['file']+str(b['page'])+str(b['table'])+str(b['row'])+value).encode()).hexdigest()
        units.append({**b,'id':uid,'text':value,'parents':pages[b['file'],b['page']]})
    dump('row_index.json',units)
    analyzer=get_lexical_analyzer();tok=lambda t:[x.normalized for x in analyzer.analyze_text(t)]
    counts=[Counter(tok(' '.join(c['header']+' '+(c['value'] or '') for c in u['bindings']))) for u in units]
    df=Counter(t for c in counts for t in c);avg=sum(sum(c.values()) for c in counts)/len(counts)
    def search(q):
        terms=set(tok(q));scores=[]
        for n,c in enumerate(counts):
            length=sum(c.values());v=sum(math.log(1+(len(counts)-df[t]+.5)/(df[t]+.5))*c[t]*2.2/(c[t]+1.2*(.25+.75*length/avg)) for t in terms if c[t])
            if v:scores.append((v,n))
        return [units[n] for v,n in sorted(scores,reverse=True)[:10]]
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)
    old=[json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')];records={(r['id'],r['arm']):r for r in old}
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    maps={cid:map_case_evidence(c,index) for cid,c in cases.items() if c.review_status!='disputed'};cache={}
    def load(cid):
        if cid not in cache:cache[cid]=store.load_chunk(conn,cid)
        return cache[cid]
    def restore(s):
        ch=load(s['chunk_id']);assert ch.text==s['text']
        return EvidenceChunk(ch,**{k:s[k] for k in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined') if k in s},member_chunk_ids=tuple(s.get('member_chunk_ids',[])),sources=tuple(load(x['chunk_id']).source for x in s.get('sources',[])))
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=3);builder=ContextBuilder();results=[];new_pairs=0
    for b in [r for r in old if r['arm']=='baseline']:
        cid=b['id'];plan=QueryInterpreter().interpret(b['question']);native=list(ConstraintMatcher().evaluate(plan,[restore(s) for s in b['candidate_stages']['constraints']]))
        original=policy.evaluate(RagQuery(text=b['question'],final_top_k=8),native,plan=plan).accepted
        assert builder.build(original,max_chars=12000).text==b['context']
        hits=search(b['question']);rowitems=[]
        for n,u in enumerate(hits,1):
            parent=load(u['parents'][0]);ch=replace(parent,stable_chunk_id=u['id'],text=u['text'],normalized_text=u['text'],text_hash=hashlib.sha256(u['text'].encode()).hexdigest())
            i=EvidenceChunk(ch,lexical_rank=n,fused_rank=20+n,member_chunk_ids=tuple(u['parents']));i.answer_support_score,i.answer_support_reasons=compute_answer_support(plan,i);rowitems.append(i)
        if rowitems:ranker.rerank(b['question'],rowitems,len(rowitems),CancelToken());new_pairs+=len(rowitems)
        ordered_rows=list(ConstraintMatcher().evaluate(plan,ranker.last_scored_candidates))
        combined=list(native);seen={i.chunk.stable_chunk_id for i in native}
        for snap in records[cid,'A']['candidate_stages']['constraints']:
            if snap['chunk_id'] not in seen:combined.append(restore(snap));seen.add(snap['chunk_id'])
        combined=list(ConstraintMatcher().evaluate(plan,combined))
        for arm,limit in [('H8',8),('S8',8),('S12',12),('S16',16)]:
            if arm=='H8':
                selected,trace=select_preserving(original,ordered_rows,policy,min(8,len(original)+1))
            else:selected,trace=select_preserving(original,combined,policy,limit)
            context=builder.build(selected,max_chars=12000);byid={i.chunk.stable_chunk_id:i for i in selected}
            stage=[{**candidate_snapshot(byid[e['chunk_id']],selected=True),'included_text':e['included_text']} for e in context.entries]
            score=score_stage('context',stage,maps[cid]);before=b['stages']['context'];oldbody=canonical_content(b['context']);newbody=canonical_content(context.text)
            neg=[m.evidence_id for m in maps[cid] if m.kind=='hard_negative' and canonical_content(m.quote) and canonical_content(m.quote) in newbody and canonical_content(m.quote) not in oldbody]
            r={'id':cid,'arm':arm,'question':b['question'],'cohort':b['cohort'],'context':context.text,'entries':list(context.entries),'before':before,'after':score,'changed':context.text!=b['context'],'lost':sorted(set(before['gold_completion_ranks'])-set(score['gold_completion_ranks'])),'gained':sorted(set(score['gold_completion_ranks'])-set(before['gold_completion_ranks'])),'new_negative_quotes':neg,'original_bodies_retained':all(canonical_content(e['included_text']) in newbody for e in b['context_entries']),'trace':trace,'row_hits':[u['id'] for u in hits],'row_scores':[candidate_snapshot(i) for i in ordered_rows],'chars':len(context.text),'limit':limit}
            results.append(r)
            with (OUT/'cases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print('done',cid,flush=True)
    conn.close();summary={}
    for arm in ('H8','S8','S12','S16'):
        rs=[r for r in results if r['arm']==arm];pos=[r for r in rs if r['cohort']=='formal48' and r['before']['gold_evidence_recall'] is not None]
        summary[arm]={'changed':sum(r['changed'] for r in rs),'before_hit':sum(r['before']['gold_hit'] for r in pos),'after_hit':sum(r['after']['gold_hit'] for r in pos),'before_coverage':sum(r['before']['gold_evidence_recall'] for r in pos)/len(pos),'after_coverage':sum(r['after']['gold_evidence_recall'] for r in pos)/len(pos),'gains':[r['id'] for r in rs if r['gained']],'losses':[r['id'] for r in rs if r['lost']],'new_negative_quotes':[r['id'] for r in rs if r['new_negative_quotes']],'gate':all(r['original_bodies_retained'] and not r['lost'] and not r['new_negative_quotes'] for r in rs)}
    dump('summary.json',summary);assert all(sha(k)==h for k,h in hashes.items())
    dump('completion.json',{'finished':str(datetime.now()),'rows':len(results),'new_cross_encoder_pairs':new_pairs,'hashes_unchanged':True});print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
