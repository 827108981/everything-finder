"""Independent scene sidecar retrieval, immutable baseline replay, strict OR audit."""
from tools.validate_row_integration import *
from tools.validate_retrieval_ab import continuation_links
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch, _declared_alternatives
from local_full_text_search.rag.evaluation.evidence_groups import score_evidence_groups

OUT=ROOT/'reports/rag_scene_entry_20261105'

def main():
    OUT.mkdir(exist_ok=False)
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf8')
    freeze=json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/freeze.json').read_text(encoding='utf8'))
    # Scoring helper is new; every previously frozen input must still be identical.
    assert all(sha(p)==h for p,h in freeze['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf8')))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==freeze['bundle_id']
    conn=store.connect_read_only(bundle.metadata_path);rows=load_chunk_rows(conn);byid={r['chunk_id']:r for r in rows}
    baseline=[r for r in json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/resolved_cases.json').read_text(encoding='utf8')) if r['arm']=='baseline']
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json','rag_row_holdout_v1_review.json','rag_row_holdout_v2_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    maps={r['id']:map_case_evidence(cases[r['id']],rows) for r in baseline}
    links=continuation_links(rows);units=[]
    for parent,children in links.items():
        ids=[parent]+[c[0] for c in children];body='\n\n'.join(byid[c]['text'] for c in ids)
        units.append({'id':hashlib.sha256(body.encode()).hexdigest(),'ids':ids,'text':body})
    dump('scene_index.json',units)
    dump('protocol.json',{'bundle_id':bundle.bundle_id,'hashes':freeze['hashes'],'settings_hash':sha(SETTINGS_PATH),'arms':['saved_native','scene_sidecar'], 'cases':len(baseline),'frozen_before_execution':True,'algorithm':'Reuse previous conservative source-only continuation detector. DIFFERENCE from old A: independent BM25 top10 over every detected title+body pair; no original top20 seed requirement. DIFFERENCE from C: append original physical pages to free slots, never replace/merge native contexts. Same CE and hard constraints, at most ONE scene pair, <=8 physical chunks and <=12000 chars; document cap3 includes original entries. No H8 stacking.','limits':'Saved native context replay; new auxiliary retrieval/reranking, not fresh full pipeline A/B. Corpus links previously explored, no blind generalization claim. Metadata source relationships not automatically business-reviewed.'})
    analyzer=get_lexical_analyzer();tok=lambda t:[x.normalized for x in analyzer.analyze_text(t)]
    counts=[Counter(tok(u['text'])) for u in units];df=Counter(t for c in counts for t in c);avg=sum(sum(c.values()) for c in counts)/max(1,len(counts))
    def search(q):
        terms=set(tok(q));sc=[]
        for n,c in enumerate(counts):
            length=sum(c.values());v=sum(math.log(1+(len(counts)-df[t]+.5)/(df[t]+.5))*c[t]*2.2/(c[t]+1.2*(.25+.75*length/max(1,avg))) for t in terms if c[t])
            if v:sc.append((v,n))
        return [units[n] for _,n in sorted(sc,reverse=True)[:10]]
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=3)
    results=[]
    for b in baseline:
        cid=b['id'];plan=QueryInterpreter().interpret(b['question']);hits=search(b['question']);items=[];lookup={u['id']:u for u in hits}
        for n,u in enumerate(hits,1):
            parent=store.load_chunk(conn,u['ids'][0]);ch=replace(parent,stable_chunk_id=u['id'],text=u['text'],normalized_text=u['text'],text_hash=hashlib.sha256(u['text'].encode()).hexdigest())
            i=EvidenceChunk(ch,lexical_rank=n,fused_rank=20+n,member_chunk_ids=tuple(u['ids']));i.answer_support_score,i.answer_support_reasons=compute_answer_support(plan,i);items.append(i)
        if items:ranker.rerank(b['question'],items,len(items),CancelToken());ordered=ConstraintMatcher().evaluate(plan,ranker.last_scored_candidates)
        else:ordered=[]
        entries=list(b['entries']);ctx=b.get('context') or '';beforestage=[{'chunk_id':e['chunk_id'],'member_chunk_ids':e.get('member_chunk_ids',[]),'included_text':e['included_text']} for e in entries];stage=list(beforestage);trace=[]
        docs=Counter(byid[e['chunk_id']]['file_path'] for e in entries if e['chunk_id'] in byid);present={e['chunk_id'] for e in entries}
        for i in ordered:
            u=lookup[i.chunk.stable_chunk_id];reason=policy.rejection_reason(i);missing=[c for c in u['ids'] if c not in present];path=byid[u['ids'][0]]['file_path']
            if not reason and not missing:reason='already_present'
            if not reason and not entries:reason='native_no_context_do_not_bypass_behavior'
            if not reason and len(entries)+len(missing)>8:reason='physical_top8'
            if not reason and docs[path]+len(missing)>3:reason='document_cap3'
            extra=''
            for c in missing:
                row=byid[c];extra+=f"\n\n[S{len(entries)+1+missing.index(c)}]\n来源：{row['file_path']} · {row['location']}\n内容：{row['text']}"
            if not reason and len(ctx+extra)>12000:reason='char_budget'
            trace.append({'ids':u['ids'],'ce':i.rerank_score,'combined':i.rerank_score_combined,'reason':reason,'selected':not reason})
            if reason:continue
            ctx+=extra
            for c in missing:
                row=byid[c];entry={'chunk_id':c,'included_text':row['text']};entries.append(entry);stage.append(entry)
            break
        before=score_stage('context',beforestage,maps[cid]);after=score_stage('context',stage,maps[cid])
        r={'id':cid,'cohort':b['cohort'],'question':b['question'],'context':ctx,'entries':entries,'changed':ctx!=(b.get('context') or ''),'before':before,'after':after,'trace':trace,'scene_search_ids':[u['ids'] for u in hits],'lost':sorted(set(before['gold_completion_ranks'])-set(after['gold_completion_ranks'])),'new_negatives':sorted(set(after['hard_negative_ids'])-set(before['hard_negative_ids'])),'original_exact_prefix':ctx.startswith(b.get('context') or '')}
        results.append(r)
        with (OUT/'cases.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print(cid,'changed',r['changed'],'gain',after['gold_evidence_found']-before['gold_evidence_found'],flush=True)
    # Approved business scope applies to the FOUR CHECK DIRECTIONS only.
    alts=json.loads((ROOT/'reports/rag_evidence_trace_019_050_20261105/four_checks_source_alternatives.json').read_text(encoding='utf8'));positive=[];manifest=[]
    for a in alts:
        row=byid[a['chunk_id']];assert row['text']==a['text'];quote=row['text'][row['text'].index('失败可能原因：'):]
        # remove slide page numeral from the end, preserving source quote verbatim.
        quote=quote.rstrip();quote=quote.rsplit('\n',1)[0] if quote.split('\n')[-1].isdigit() else quote
        mappings=_declared_alternatives([row],quote,a['chunk_id']);assert mappings and all(x.confirmed for x in mappings)
        positive.append([GoldEvidenceMatch(a['chunk_id'],'positive',row['file_path'],str(row.get('slide_number') or row.get('page_number')),quote,(a['chunk_id'],),'exact',alternatives=mappings)])
        manifest.append({'chunk_id':a['chunk_id'],'file':row['file_path'],'location':row['location'],'quote':quote,'index_exact':True,'approval':'user: immune scenario common failure directions may be used; no transfer of model-specific parameters'})
    groups=[{'id':'dark_count_four_common_checks','approved':True,'alternatives':positive}]
    audits=[]
    for b in [r for r in json.loads((ROOT/'reports/rag_row_ab_confirm_20261105/resolved_cases.json').read_text(encoding='utf8')) if r['id']=='rag-biz-050']:
        stage=[{'chunk_id':e['chunk_id'],'included_text':e['included_text']} for e in b['entries']]
        audits.append({'arm':b['arm'],'old':score_stage('context',stage,maps[b['id']]),'approved_group':score_evidence_groups('context',stage,groups)})
    dump('050_approved_alternatives.json',manifest);dump('050_scoring_audit.json',audits)
    summary={'cases':len(results),'scene_pairs':len(units),'changed':[r['id'] for r in results if r['changed']],'gains':[r['id'] for r in results if r['after']['gold_evidence_found']>r['before']['gold_evidence_found']],'losses':[r['id'] for r in results if r['lost']],'new_negatives':[r['id'] for r in results if r['new_negatives']],'preservation_gate':all(r['original_exact_prefix'] and not r['lost'] and not r['new_negatives'] for r in results),'frozen_inputs_unchanged':all(sha(p)==h for p,h in freeze['hashes'].items())}
    dump('summary.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
