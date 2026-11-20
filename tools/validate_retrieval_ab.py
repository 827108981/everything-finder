"""Isolated A/B retrieval experiment; never alters app settings or published index.

A: conservative title-to-continuation-page candidate expansion and scoring context.
B: question-only LLM subqueries, retrieval and per-subquery cross-encoder scoring.
Original 20 rerank candidates retained; at most 20 additional candidates.
"""
from __future__ import annotations
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
import re
import time
from unittest.mock import patch
from collections import Counter, defaultdict

from local_full_text_search.config.constants import SETTINGS_PATH, RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import score_case, aggregate
from local_full_text_search.rag.generation.base import TextCompletionRequest, GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery, EvidenceChunk
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('baseline', 'A', 'B', 'A+B')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def continuation_links(rows):
    """Only short scenario pages followed by a generic causes/steps page; no inferred applicability."""
    pages = defaultdict(lambda: defaultdict(list))
    for r in rows:
        page = r.get('slide_number') or r.get('page_number')
        if page:
            pages[r['file_path']][int(page)].append(r)
    links = {}
    for path, doc in pages.items():
        for page, left in doc.items():
            right = doc.get(page + 1, [])
            if len(left) != 1 or not right:
                continue
            title = left[0]['text'].strip()
            body = '\n'.join(r['text'] for r in sorted(right, key=lambda x:x['ordinal']))
            clean = re.sub(r'\s+', '', body)
            # A deliberately narrow prototype. No generic +/-1 expansion.
            if (len(title) <= 180 and re.search(r'故障|失败|异常|问题|报警', title)
                and not re.search(r'解决思路|处理步骤|可能原因|失败可能原因', title)
                and re.match(r'^(可能原因|故障原因|原因分析|解决思路|处理步骤|排查步骤)[：:]', clean)):
                links[left[0]['chunk_id']] = [(r['chunk_id'], title, left[0]['location']) for r in right]
    return links

def validate_queries(question, value):
    queries = value.get('queries', [])
    if not isinstance(queries, list) or not 2 <= len(queries) <= 3:
        return []
    queries = [q.strip() for q in queries if isinstance(q, str) and 4 <= len(q.strip()) <= 200]
    if len(set(queries)) != len(queries) or len(queries) < 2:
        return []
    # Never invent or silently lose explicit alphanumeric model/version tokens.
    tokens = set(re.findall(r'[A-Za-z]+[-_]?[0-9]+[A-Za-z0-9.\-]*', question))
    if any(any(t.casefold() not in q.casefold() for t in tokens) for q in queries):
        return []
    return queries

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--only', nargs='*')
    p.add_argument('--plans-from', type=Path)
    p.add_argument('--resume-from', type=Path)
    args = p.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ''
    assert key
    def dump(path, value):
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str).replace(key,'[REDACTED]')+'\n', encoding='utf-8')
    def append(path, value):
        with path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(value,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
    fixtures = ROOT/'tests/rag/fixtures'
    main_data = load_gold_dataset(fixtures/'rag_gold_v4_1_review.json', reviewed_only=False)
    regression = load_gold_dataset(fixtures/'rag_acceptance_v1_review.json', reviewed_only=False)
    cases = [c for c in main_data.cases if c.review_status != 'disputed'] + list(regression.cases)
    if args.only:
        cases = [c for c in cases if c.case_id in args.only]
    index_db = Path('E:/everythingfinder-db/search_index.db')
    store = RagIndexStore(index_db.parent/'rag'); bundle = store.active()
    assert all(c.corpus_snapshot == bundle.bundle_id for c in cases)
    paths = [SETTINGS_PATH, main_data.path, regression.path] + list((ROOT/'local_full_text_search/rag').rglob('*.py'))
    hashes = {str(p):sha(p) for p in paths}
    protocol = {'started':str(datetime.now()),'bundle_id':bundle.bundle_id,'arms':ARMS,
        'model':settings.rag_api_model,'embedding':settings.rag_embedding_repo_id,'reranker':settings.rag_reranker_repo_id,
        'case_ids':[c.case_id for c in cases], 'hashes':hashes,'harness_sha256':sha(__file__),
        'budgets':{'original_rerank':20,'additional_max':20,'B_per_route':10,'A_max':10,'final':settings.rag_final_top_k},
        'controls':'Original query, original candidates and downstream policies preserved. No Gold access by retrieval/decomposition/scoring. Fresh run; deterministic identical retrieval and cross-encoder pairs memoized within this experiment only.',
        'timing':'Wall times with shared caches are NOT independent-arm latency measurements. Cache miss inference durations recorded separately. No latency superiority claims.',
        'scope':'Conservative prototypes, not a complete chapter parser or production-ready query planner. Results do not prove zero impact outside evaluated cases.'}
    dump(out/'protocol.json',protocol)
    conn=store.connect_read_only(bundle.metadata_path)
    rows=load_chunk_rows(conn)
    maps={c.case_id:map_case_evidence(c,rows) for c in cases}
    links=continuation_links(rows)
    dump(out/'continuation_links.json',links)
    gateway=OpenAICompatibleGateway(max_transient_retries=2)
    query_plans={}
    def decompose(c):
        t=time.monotonic()
        try:
            response=gateway.complete_text(TextCompletionRequest(
                system_prompt='你只负责检索查询规划。仅当用户提出两个或以上独立诉求时拆分；单问题、同一操作连续步骤、不可分的比较返回空数组。返回JSON {"queries": [...]}，最多3个。每个查询必须自包含，保留原问已给出的机型版本、阶段、范围、否定和不确定性。不得把假设当事实，不得添加答案、诊断结论或用户未给出的条件。只看用户问题，不推测资料内容。',
                user_prompt=c.question, model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,
                api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,
                max_output_tokens=900),CancelToken())
            raw=(response if isinstance(response,str) else response.text).strip(); raw=re.sub(r'^```(?:json)?\s*|\s*```$','',raw)
            val=json.loads(raw)
            return c.case_id,{'queries':validate_queries(c.question,val),'raw':val,'duration_ms':round((time.monotonic()-t)*1000)}
        except Exception as e:
            return c.case_id,{'queries':[],'error':str(e)[:250],'duration_ms':round((time.monotonic()-t)*1000)}
    if args.plans_from:
        query_plans=json.loads(args.plans_from.read_text(encoding='utf-8'))
        assert set(query_plans)=={c.case_id for c in cases}
    else:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for cid, val in pool.map(decompose,cases):
                query_plans[cid]=val
                dump(out/'query_plans_progress.json',query_plans)
                print('planned',cid, len(val['queries']),flush=True)
    dump(out/'query_plans.json',query_plans)
    import torch
    torch.set_num_threads(4); torch.set_num_interop_threads(1)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
        repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
    reranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
        repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,
        max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,
        retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,
        max_rank_drop=settings.rag_rerank_max_rank_drop)
    cache={}; retrieval_events={}; pair_cache={}; inference=[]; active={}
    original_hybrid=HybridRetriever.retrieve
    original_rerank=reranker.rerank
    def cached_retrieve(hybrid,query,token,plan=None):
        k=query.text
        if k not in cache:
            cb=hybrid.progress_callback; recorded=[]
            def capture(*args,**kwargs):
                recorded.append((args,copy.deepcopy(kwargs)))
                if cb:cb(*args,**kwargs)
            hybrid.progress_callback=capture
            t=time.monotonic()
            try:
                cache[k]=copy.deepcopy(original_hybrid(hybrid,query,token,plan))
            finally:
                hybrid.progress_callback=cb
            retrieval_events[k]=recorded
            inference.append({'kind':'hybrid','question':k,'duration_ms':round((time.monotonic()-t)*1000)})
        elif hybrid.progress_callback:
            for args,kwargs in retrieval_events[k]:
                hybrid.progress_callback(*args,**copy.deepcopy(kwargs))
        return copy.deepcopy(cache[k])
    def augmented(hybrid,query,token,plan=None):
        original=cached_retrieve(hybrid,replace(query,fusion_top_k=settings.rag_fusion_top_k),token,plan)
        baseline=CandidateDeduplicator().deduplicate(copy.deepcopy(original),limit=settings.rag_fusion_top_k)
        base=list(baseline.candidates)[:20]
        active['base_ids']=[i.chunk.stable_chunk_id for i in base]
        active['contexts']={}; active['routes']=defaultdict(list); active['route_candidates']=[]; extras=[]
        if 'B' in active['arm']:
            for q in active['queries']:
                cb=hybrid.progress_callback
                hybrid.progress_callback=None
                try:
                    route=cached_retrieve(hybrid,replace(query,text=q,fusion_top_k=settings.rag_fusion_top_k),token,QueryInterpreter().interpret(q,token))
                finally:
                    hybrid.progress_callback=cb
                active['route_candidates'].append({'query':q,'candidates':[{'chunk_id':x.chunk.stable_chunk_id,'lexical_rank':x.lexical_rank,'vector_rank':x.vector_rank,'fused_rank':x.fused_rank} for x in route]})
                route=CandidateDeduplicator().deduplicate(route,limit=10).candidates
                for item in route:
                    active['routes'][item.chunk.stable_chunk_id].append(q)
                    extras.append(item)
        if 'A' in active['arm']:
            seeds=base+extras[:20]
            added=[]
            for seed in seeds:
                for cid,title,loc in links.get(seed.chunk.stable_chunk_id,[]):
                    ch=store.load_chunk(conn,cid)
                    if ch is not None and len(added)<10:
                        # Preserve real retrieval provenance if the linked page was
                        # recalled below the earlier cutoff. Never copy parent scores.
                        recalled=next((x for x in original+extras if x.chunk.stable_chunk_id==cid),None)
                        added.append(copy.deepcopy(recalled) if recalled else EvidenceChunk(chunk=ch))
                        active['contexts'][cid]=f'前页场景（{loc}，仅用于相关性判断）：{title}\n当前页原文：\n'
            extras=added+extras
        all_items=base+extras
        # Keep base representatives first, merge duplicates using the actual production deduplicator.
        merged=CandidateDeduplicator().deduplicate(all_items,limit=40)
        selected=list(merged.candidates)
        assert set(active['base_ids']) <= {i.chunk.stable_chunk_id for i in selected}
        for rank,item in enumerate(selected,1):
            if item.chunk.stable_chunk_id not in active['base_ids']:
                item.fused_rank=rank
        active['added_ids']=[i.chunk.stable_chunk_id for i in selected if i.chunk.stable_chunk_id not in active['base_ids']]
        return selected
    class ExperimentDedup(CandidateDeduplicator):
        def deduplicate(self,candidates,*,limit):
            return super().deduplicate(candidates,limit=40)
    def score_pairs(question,items,token):
        missing=[]
        for i in items:
            if (question,i.chunk.text) not in pair_cache:
                missing.append(copy.deepcopy(i))
        if missing:
            t=time.monotonic(); original_rerank(question,missing,len(missing),token)
            for i in missing:
                pair_cache[(question,i.chunk.text)]=i.rerank_score
            inference.append({'kind':'cross_encoder','pairs':len(missing),'duration_ms':round((time.monotonic()-t)*1000)})
        return [pair_cache[(question,i.chunk.text)] for i in items]
    class ExperimentReranker:
        last_scored_candidates=()
        def rerank(self,question,items,top_n,token):
            views=[]
            for i in items:
                v=copy.deepcopy(i)
                prefix=active.get('contexts',{}).get(i.chunk.stable_chunk_id,'')
                if prefix:
                    v.chunk=replace(v.chunk,text=prefix+v.chunk.text)
                views.append(v)
            scores=score_pairs(question,views,token)
            if 'B' in active['arm']:
                for q in active['queries']:
                    indexes=[n for n,i in enumerate(items) if q in active['routes'].get(i.chunk.stable_chunk_id,[])]
                    sub=score_pairs(q,[views[n] for n in indexes],token)
                    for n,s in zip(indexes,sub): scores[n]=max(scores[n],s)
            for item,s in zip(items,scores):
                item.rerank_score=s
                item.rerank_score_combined=reranker.cross_encoder_weight*s+reranker.retrieval_prior_weight*retrieval_prior(item)+reranker.answer_support_weight*float(item.answer_support_score or 0)
            ranked=sorted(items,key=lambda i:(-i.rerank_score_combined,-i.fused_score,i.chunk.stable_chunk_id))
            ranked=reranker._apply_rank_guard(ranked)
            self.last_scored_candidates=tuple(ranked)
            return reranker._apply_document_diversity(ranked,top_n)
    local=[]
    if args.resume_from:
        previous=json.loads((args.resume_from/'protocol.json').read_text(encoding='utf-8'))
        assert previous['hashes']==hashes and previous['bundle_id']==bundle.bundle_id
        local=[json.loads(l) for l in (args.resume_from/'retrieval_cases.jsonl').read_text(encoding='utf-8').splitlines()]
        local=[r for r in local if not r.get('error')]
        for r in local:
            append(out/'retrieval_cases.jsonl',r)
            # Only baseline/A scores have a unique, recoverable question/body pair.
            if r['arm'] in ('baseline','A'):
                for i in r['candidate_stages'].get('rerank',[]):
                    prefix=r.get('association_contexts',{}).get(i['chunk_id'],'')
                    pair_cache[(r['question'],prefix+i['text'])]=i['rerank_score']
        dump(out/'resume_provenance.json',{'from':str(args.resume_from),'rows':len(local),'prior_harness_sha256':previous['harness_sha256'],
            'change':'Baseline uses identical scoring implementation through pair-cache wrapper. Previously completed rows preserved. No retrieval/selection parameters changed.'})
    completed={(r['id'],r['arm']) for r in local}
    field_stage={'lexical_candidates':'lexical','vector_candidates':'vector','fused_candidates':'fusion','deduplicated_candidates':'deduplicated','reranked_candidates':'rerank','constraint_candidates':'constraints','evidence_candidates':'evidence'}
    for case in cases:
        for arm in ARMS:
            if (case.case_id,arm) in completed:continue
            active.clear(); active.update(arm=arm,queries=query_plans[case.case_id]['queries'])
            stages={}; events=[]
            def record(e):
                for f,s in field_stage.items():
                    if f in e: stages[s]=e[f]
                events.append({k:v for k,v in e.items() if k not in field_stage})
            token=CancelToken(); t=time.monotonic()
            kwargs={}
            if arm!='baseline':kwargs['deduplicator']=ExperimentDedup()
            service=AnswerService(IndexGenerationStore(index_db),store,embedding,ExperimentReranker(),
                gateway,WindowsCredentialSecretStore(),settings,progress_callback=record,**kwargs)
            query=RagQuery(text=case.question,min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=20 if arm=='baseline' else 40,final_top_k=settings.rag_final_top_k)
            try:
                with patch.object(HybridRetriever,'retrieve',cached_retrieve if arm=='baseline' else augmented):
                    result=service.retrieve(query,token)
                error=None
            except Exception as e:
                error={'type':type(e).__name__,'message':str(e)[:400]}; result=None
            ctx=next((e for e in reversed(events) if e.get('context_entries')), {})
            selected={x['chunk_id']:x for x in stages.get('evidence',[]) if x.get('selected')}
            stages['context']=[{**selected[e['chunk_id']],'included_text':e['included_text'],'truncated':bool(e.get('truncated'))} for e in ctx.get('context_entries',[]) if e['chunk_id'] in selected]
            scored=score_case(case,stages,maps[case.case_id],answer={'mode':'retrieval-only'})
            row={**scored,'arm':arm,'review_status':case.review_status,
                'cohort':'regression10_pending' if case.case_id.startswith('rag-acc') else 'formal48' if case.review_status=='reviewed' else 'pending024',
                'local_wall_ms':round((time.monotonic()-t)*1000),'candidate_stages':stages,
                'context':ctx.get('llm_context',''),'context_entries':ctx.get('context_entries',[]),
                'retrieval_status':str(result.status) if result else 'error','local_answer_text':(result.clarification_question or '资料中没有足够依据。') if result and not ctx else '', 'events':events,'error':error,
                'supplement_ids':active.get('added_ids',[]),'association_contexts':active.get('contexts',{}),'subquery_routes':active.get('route_candidates',[])}
            local.append(row); append(out/'retrieval_cases.jsonl',row)
            print(arm,case.case_id,'gold',stages.get('context') and scored['stages']['context']['gold_evidence_found'],'error',error,flush=True)
    conn.close(); dump(out/'inference_cost.json',inference)
    case_map={c.case_id:c for c in cases}
    def remote(row):
        t=time.monotonic();case=case_map[row['id']]; answer=None;judgement=None;error=None
        try:
            if row['error']:raise RuntimeError(str(row['error']))
            if row['context']:
                generated=gateway.generate(GenerationRequest(question=case.question,context=row['context'],
                    model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,
                    api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,
                    query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question))),CancelToken())
                text=generated.text
            else:
                # Reproduce local refusal/clarification as recorded by the service event.
                text=row['local_answer_text']
            answer={'mode':'full','text':text,'status':row['retrieval_status'],'source_ids':[e.get('source_id') for e in row['context_entries']]}
            judgement=GoldAnswerJudge(gateway,settings,key).evaluate(case,answer,CancelToken())
        except Exception as e:
            error={'type':type(e).__name__,'message':str(e)[:400]}
        return {**row,'answer':answer,'gold_judgement':judgement,'remote_error':error,'remote_ms':round((time.monotonic()-t)*1000)}
    full=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(remote,r) for r in local]
        for f in as_completed(futures):
            r=f.result();full.append(r);append(out/'cases.jsonl',r)
            print('answer',r['arm'],r['id'],(r.get('gold_judgement')or{}).get('verdict'),r.get('remote_error'),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    assert store.active().bundle_id==bundle.bundle_id
    dump(out/'completion.json',{'finished':str(datetime.now()),'rows':len(full),'expected':4*len(cases),'hashes_unchanged':True,'bundle_unchanged':True,'production_defaults_unchanged':True,'errors':sum(bool(r['error'] or r['remote_error']) for r in full)})
    summarize(out,full,dump)

def summarize(out,rows,dump):
    summary={}; changes={}; groups={}
    for cohort in ('formal48','regression10_pending','pending024'):
        summary[cohort]={}
        for arm in ARMS:
            subset=[r for r in rows if r['arm']==arm and r['cohort']==cohort]
            summary[cohort][arm]={'count':len(subset),'verdicts':dict(Counter((r.get('gold_judgement')or{}).get('verdict','unavailable') for r in subset)),
                'retrieval':aggregate(subset),'errors':[r['id'] for r in subset if r['error'] or r['remote_error']]}
    for r in rows:groups.setdefault(r['id'],{})[r['arm']]=r
    for arm in ARMS[1:]:
        delta=[]
        for cid,g in groups.items():
            b=g['baseline'];v=g[arm]
            bs=b['stages']['context'];vs=v['stages']['context']
            bv=(b.get('gold_judgement')or{}).get('verdict');vv=(v.get('gold_judgement')or{}).get('verdict')
            if bs!=vs or bv!=vv:
                delta.append({'id':cid,'cohort':b['cohort'],'before_context':bs,'after_context':vs,'before_verdict':bv,'after_verdict':vv,
                    'same_context':b['context']==v['context']})
        changes[arm]=delta
    dump(out/'summary.json',summary);dump(out/'changes.json',changes)
    parts=['<!doctype html><meta charset="utf-8"><title>A/B验证</title><style>body{font:16px sans-serif;margin:30px}pre{white-space:pre-wrap}details{border:1px solid #bbb;padding:12px;margin:12px 0}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px}</style><h1>A、B、A+B验证报告</h1>',
        '<p>A：保守续页关联；B：子问题检索与评分。48题正式、11题待审诊断；007排除。分数是模型辅助评审。共享推理缓存使墙钟耗时不能直接比较；未证明全体业务无副作用。</p>',
        '<h2>汇总数据</h2><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>']
    for cid,g in sorted(groups.items()):
        parts.append('<details><summary>'+html.escape(cid+' '+g['baseline']['question'])+'</summary>')
        for arm in ARMS:
            r=g[arm];parts.append('<h3>'+arm+'</h3><pre>'+html.escape(json.dumps({'context_metrics':r['stages']['context'],'judge':r['gold_judgement'],'error':r['remote_error'],'answer':r['answer'],'context':r['context']},ensure_ascii=False,indent=2))+'</pre>')
        parts.append('</details>')
    (out/'report.html').write_text(''.join(parts),encoding='utf-8')

if __name__=='__main__':main()
