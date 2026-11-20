"""New-question trial with independently rerun native retrieval and frozen H8."""
from tools.validate_row_integration import *
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
import random,time

OUT=ROOT/'reports/rag_row_holdout_20261105'
FIX=ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json'
def main():
    assert not (OUT/'timing_cases.jsonl').exists()
    freeze=json.loads((OUT/'freeze.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in freeze['hashes'].items())
    hashes={**freeze['hashes'],str(FIX):sha(FIX),__file__:sha(__file__)}
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==freeze['bundle_id']
    cases=load_gold_dataset(FIX,reviewed_only=False).cases
    dump('protocol.json',{'started':str(datetime.now()),'hashes':hashes,'case_ids':[c.case_id for c in cases],'review_status':'all_pending','frozen_h8':freeze,'runtime':'Both arms run full fresh native retrieval independently. 3 repetitions sequentially, randomized case/arm order, no query or score memoization. CPU4; no concurrent inference. Initialization and first warm-up excluded and reported separately.','step_order':'Stage1 new-question trial; phase2 evaluator span audit starts after trial completion. Production scorer unmodified.'})
    conn=store.connect_read_only(bundle.metadata_path);index=load_chunk_rows(conn);maps={c.case_id:map_case_evidence(c,index) for c in cases};cache={}
    def load(cid):
        if cid not in cache:cache[cid]=store.load_chunk(conn,cid)
        return cache[cid]
    units=json.loads((ROOT/'reports/rag_row_integration_20261105/row_index.json').read_text(encoding='utf-8'))
    t=time.perf_counter();analyzer=get_lexical_analyzer();tok=lambda text:[i.normalized for i in analyzer.analyze_text(text)]
    counts=[Counter(tok(' '.join(c['header']+' '+(c['value'] or '') for c in u['bindings']))) for u in units];df=Counter(k for c in counts for k in c);avg=sum(sum(c.values()) for c in counts)/len(counts);row_init=(time.perf_counter()-t)*1000
    def search(q):
        terms=set(tok(q));ranked=[]
        for n,c in enumerate(counts):
            length=sum(c.values());v=sum(math.log(1+(len(counts)-df[k]+.5)/(df[k]+.5))*c[k]*2.2/(c[k]+1.2*(.25+.75*length/avg)) for k in terms if c[k])
            if v:ranked.append((v,n))
        return [units[n] for v,n in sorted(ranked,reverse=True)[:10]]
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    t=time.perf_counter();manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
    ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)
    dump('initialization.json',{'local_models_ms':(time.perf_counter()-t)*1000,'row_cpu_index_ms':row_init,'cold_os_cache_not_controlled':True})
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=3);builder=ContextBuilder()
    def run(case,arm,repeat):
        events=[];started=time.perf_counter()
        service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,embedding,ranker,OpenAICompatibleGateway(),WindowsCredentialSecretStore(),settings,progress_callback=events.append)
        q=RagQuery(text=case.question,lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=8,min_ocr_confidence=settings.ocr_min_confidence)
        result=service.retrieve(q,CancelToken());native_ms=(time.perf_counter()-started)*1000
        ce=next((e for e in reversed(events) if 'llm_context' in e),{})
        entries=ce.get('context_entries',[]);original_context=ce.get('llm_context','');context_text=original_context;trace=[];row_ms=0;stage=[]
        ev=next((e['evidence_candidates'] for e in reversed(events) if 'evidence_candidates' in e),[]);snaps={s['chunk_id']:s for s in ev if s.get('selected')}
        plan=QueryInterpreter().interpret(case.question);original=[]
        for e in entries:
            s=snaps[e['chunk_id']];i=EvidenceChunk(load(e['chunk_id']),**{k:s[k] for k in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined') if k in s},member_chunk_ids=tuple(s.get('member_chunk_ids',[])),sources=tuple(load(x['chunk_id']).source for x in s.get('sources',[])))
            original.append(i)
        original=list(ConstraintMatcher().evaluate(plan,original))
        if entries:assert builder.build(original,max_chars=12000).text==original_context
        if arm=='H8':
            rt=time.perf_counter();items=[]
            for n,u in enumerate(search(case.question),1):
                parent=load(u['parents'][0]);ch=replace(parent,stable_chunk_id=u['id'],text=u['text'],normalized_text=u['text'],text_hash=hashlib.sha256(u['text'].encode()).hexdigest());i=EvidenceChunk(ch,lexical_rank=n,fused_rank=20+n,member_chunk_ids=tuple(u['parents']));i.answer_support_score,i.answer_support_reasons=compute_answer_support(plan,i);items.append(i)
            ranker.rerank(case.question,items,len(items),CancelToken());ordered=ConstraintMatcher().evaluate(plan,ranker.last_scored_candidates)
            selected,trace=select_preserving(original,ordered,policy,min(8,len(original)+1));context=builder.build(selected,max_chars=12000);row_ms=(time.perf_counter()-rt)*1000
            context_text=context.text;entries=list(context.entries);byid={i.chunk.stable_chunk_id:i for i in selected};stage=[{**candidate_snapshot(byid[e['chunk_id']],selected=True),'included_text':e['included_text']} for e in entries]
        else:stage=[{**snaps[e['chunk_id']],'included_text':e['included_text']} for e in entries]
        return {'id':case.case_id,'arm':arm,'repeat':repeat,'question':case.question,'status':str(result.status),'local_text':result.clarification_question or '资料中没有足够依据。','context':context_text,'entries':entries,'context_changed':context_text!=original_context,'trace':trace,'stage':stage,'score':score_stage('context',stage,maps[case.case_id]),'native_ms':native_ms,'row_ms':row_ms,'total_ms':(time.perf_counter()-started)*1000,'original_entries':ce.get('context_entries',[])}
    t=time.perf_counter();run(cases[0],'baseline',0);dump('warmup.json',{'ms':(time.perf_counter()-t)*1000,'case':cases[0].case_id,'excluded_from_timing':True})
    jobs=[(c,arm,rep) for rep in range(1,4) for c in cases for arm in ('baseline','H8')];random.Random(20261105).shuffle(jobs)
    for n,(c,arm,rep) in enumerate(jobs,1):
        row=run(c,arm,rep)
        with (OUT/'timing_cases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(n,c.case_id,arm,rep,round(row['total_ms']),row['score']['gold_hit'],row['context_changed'],flush=True)
    conn.close();assert all(sha(k)==h for k,h in hashes.items())
    dump('local_completion.json',{'finished':str(datetime.now()),'rows':len(jobs),'hashes_unchanged':True})
if __name__=='__main__':main()
