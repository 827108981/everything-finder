"""Source-derived PPT table/continuation prototypes. Published index read-only."""
from __future__ import annotations
from collections import Counter, defaultdict
import copy
from dataclasses import replace
from datetime import datetime
import hashlib, html, json, math, posixpath, re, time, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
from tools.validate_material_steps import ROOT, sha
from tools.validate_retrieval_ab import continuation_links
from local_full_text_search.config.constants import SETTINGS_PATH,RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.models import EvidenceChunk
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.model_manager import LocalModelManager,ModelSpec
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import compute_answer_support
from local_full_text_search.rag.retrieval.lexical_analyzer import get_lexical_analyzer
from local_full_text_search.rag.trace import candidate_snapshot

OUT=ROOT/'reports/rag_structured_material_20261105'
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def text(cell):return '\n'.join(''.join(t.text or '' for t in p.findall('.//a:t',NS)) for p in cell.findall('.//a:p',NS)).strip()
def slides(path):
    with zipfile.ZipFile(path) as z:
        rel={r.attrib['Id']:r.attrib['Target'] for r in ET.fromstring(z.read('ppt/_rels/presentation.xml.rels'))}
        for page,s in enumerate(ET.fromstring(z.read('ppt/presentation.xml')).findall('.//p:sldId',NS),1):
            target=rel[s.attrib['{'+NS['r']+'}id']]
            name=target.lstrip('/') if target.startswith('/') else posixpath.normpath('ppt/'+target)
            yield page,ET.fromstring(z.read(name))

def extract(rows):
    pages=defaultdict(list)
    for r in rows:
        if r['file_path'].lower().endswith('.pptx'):pages[r['file_path'],r.get('slide_number') or r.get('page_number')].append(r)
    units=[];audits=[];files={}
    for path in sorted({p for p,n in pages}):
        try:
            files[path]=sha(path)
            for page,slide in slides(path):
                indexed=pages.get((path,page),[])
                if not indexed:continue
                body=canonical_content('\n'.join(r['text'] for r in indexed))
                title=''
                for shape in slide.findall('.//p:sp',NS):
                    ph=shape.find('.//p:ph',NS)
                    if ph is not None and ph.attrib.get('type') in ('title','ctrTitle'):
                        title=text(shape);break
                for n,table in enumerate(slide.findall('.//a:tbl',NS),1):
                    raw=[[text(c) for c in row.findall('a:tc',NS)] for row in table.findall('a:tr',NS)]
                    props=table.find('a:tblPr',NS)
                    reason=''
                    if any(any(k in c.attrib and c.attrib[k] not in ('0','1') for k in ('gridSpan','rowSpan')) or c.attrib.get('hMerge')=='1' or c.attrib.get('vMerge')=='1' for c in table.findall('.//a:tc',NS)):reason='merged_cells_require_review'
                    elif not raw or len(raw)<2 or len(raw[0])<2 or any(not c for c in raw[0]) or len(set(raw[0]))!=len(raw[0]):reason='ambiguous_header'
                    elif props is None or props.attrib.get('firstRow')!='1':reason='header_not_marked_in_source'
                    elif any(len(row)!=len(raw[0]) for row in raw):reason='column_count_mismatch'
                    elif any(canonical_content(c) not in body for row in raw for c in row if c):reason='source_cell_not_found_in_index'
                    audit={'file':path,'page':page,'table':n,'headers':raw[0] if raw else [],'rows':len(raw)-1,'status':'fallback' if reason else 'accepted','reason':reason,'raw_cells':raw}
                    audits.append(audit)
                    if reason:continue
                    for rn,row in enumerate(raw[1:],2):
                        if not any(row):continue
                        # Headers identify columns; row original stays contiguous for quote verification.
                        value=(title+'\n列顺序：'+' | '.join(raw[0])+'\n原文行：\n'+'\n'.join(row)).strip()
                        uid=hashlib.sha256((path+str(page)+str(n)+str(rn)+value).encode()).hexdigest()
                        units.append({'id':uid,'file':path,'page':page,'table':n,'row':rn,'headers':raw[0],'cells':row,'text':value,'parent_ids':[r['chunk_id'] for r in indexed],'cell_index_verified':True})
        except Exception as exc:audits.append({'file':path,'status':'file_error','reason':type(exc).__name__+': '+str(exc)[:150]})
    return units,audits,files

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    oldp=json.loads((ROOT/'reports/rag_retrieval_ab_results_20261103/protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in oldp['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==oldp['bundle_id']
    conn=store.connect_read_only(bundle.metadata_path);rows=load_chunk_rows(conn)
    units,audits,files=extract(rows);dump('table_units.json',units);dump('structure_audit.json',audits)
    links=continuation_links(rows);dump('continuation_links.json',links)
    hashes={**oldp['hashes'],**files,str(bundle.metadata_path):sha(bundle.metadata_path),__file__:sha(__file__)}
    dump('protocol.json',{'started':str(datetime.now()),'bundle_id':bundle.bundle_id,'hashes':hashes,
      'arms':{'baseline':'Frozen production baseline contexts and scores.','P':'Control: original20 plus at most10 parent pages found by table-row BM25.','T':'Table intervention: same row search, original20 plus at most10 header+row views. P isolates extra retrieval benefit from table representation.','C':'Continuation intervention: original20 only; associated source title and next cause page combined as view, preserving all source IDs. No T/P stacking.'},
      'scope':'Native PPTX tables only. Unsupported/merged/unmarked headers fall back to original. No PDF/Word/Excel structure implementation; no new vector index. Shadow row BM25 uses existing lexical tokenizer, k1=1.2,b=.75. This route differs from production word_v2 complete matching and is not a full reindex benchmark.',
      'selection':'Original scoring weights, rank guard, constraint checks, doc cap3, final8 and actual12k chars. Continuation pages count as physical pages for final8 and document quota; no hidden expansion.',
      'stop':'No combination or production rollout if an intervention loses prior complete Gold evidence or adds actual marked negative quotes. Broad source-ID negative flags reported separately.',
      'structural_gate':'Every accepted cell must exist in original native table and current index body; all merged/ambiguous tables retained whole. Semantic header correctness additionally sampled, not universally proven.'})
    print('structure',len(files),'files',len(audits),'tables',len(units),'rows',len(links),'links',flush=True)
    analyzer=get_lexical_analyzer();tok=lambda t:[x.normalized for x in analyzer.analyze_text(t)]
    counts=[Counter(tok(u['text'])) for u in units];df=Counter(t for c in counts for t in c);avg=sum(map(lambda c:sum(c.values()),counts))/max(1,len(counts))
    def search(q):
        ts=set(tok(q));scores=[]
        for n,c in enumerate(counts):
            length=sum(c.values());s=sum(math.log(1+(len(counts)-df[t]+.5)/(df[t]+.5))*c[t]*2.2/(c[t]+1.2*(.25+.75*length/max(1,avg))) for t in ts if c[t])
            if s>0:scores.append((s,n))
        return [units[n] for s,n in sorted(scores,reverse=True)[:10]]
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    reranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length,cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,answer_support_weight=settings.rag_rerank_answer_support_weight,max_rank_drop=settings.rag_rerank_max_rank_drop)
    baseline=[r for r in (json.loads(l) for l in (ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl').open(encoding='utf-8')) if r['arm']=='baseline']
    cases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    maps={cid:map_case_evidence(c,rows) for cid,c in cases.items() if c.review_status!='disputed'}
    cache={};score_cache={}
    def load(cid):
        if cid not in cache:cache[cid]=store.load_chunk(conn,cid)
        return cache[cid]
    def restore(s):
        ch=load(s['chunk_id']);assert ch.text==s['text']
        return EvidenceChunk(ch,**{k:s[k] for k in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined') if k in s},member_chunk_ids=tuple(s.get('member_chunk_ids',[])),sources=tuple(load(x['chunk_id']).source for x in s.get('sources',[])))
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=settings.rag_max_chunks_per_document)
    builder=ContextBuilder();results=[]
    for b in baseline:
        cid=b['id'];plan=QueryInterpreter().interpret(b['question']);q=RagQuery(text=b['question'],final_top_k=8)
        base=[restore(s) for s in b['candidate_stages']['deduplicated'][:20]]
        hits=search(b['question'])
        for arm in ('P','T','C'):
            items=copy.deepcopy(base);parents={};physical={}
            if arm in ('P','T'):
                for rank,u in enumerate(hits,1):
                    parent=load(u['parent_ids'][0])
                    if arm=='P':
                        if any(i.chunk.stable_chunk_id==parent.stable_chunk_id for i in items):continue
                        item=EvidenceChunk(parent,lexical_rank=rank,fused_rank=20+rank)
                    else:
                        ch=replace(parent,stable_chunk_id=u['id'],text=u['text'],normalized_text=u['text'],text_hash=hashlib.sha256(u['text'].encode()).hexdigest())
                        item=EvidenceChunk(ch,lexical_rank=rank,fused_rank=20+rank,member_chunk_ids=tuple(u['parent_ids']))
                        parents[u['id']]=u['parent_ids']
                    items.append(item)
            else:
                for item in items:
                    children=links.get(item.chunk.stable_chunk_id,[])
                    if not children:continue
                    ids=[item.chunk.stable_chunk_id]+[c[0] for c in children]
                    parts=[load(i) for i in ids]
                    joined='\n\n'.join(c.text for c in parts)
                    uid=hashlib.sha256(('continuation'+''.join(ids)).encode()).hexdigest()
                    item.chunk=replace(item.chunk,stable_chunk_id=uid,text=joined,normalized_text=joined,text_hash=hashlib.sha256(joined.encode()).hexdigest())
                    item.member_chunk_ids=tuple(ids);item.sources=tuple(c.source for c in parts);parents[uid]=ids;physical[uid]=len(ids)
            missing=[]
            for i in items:
                i.answer_support_score,i.answer_support_reasons=compute_answer_support(plan,i)
                key=(b['question'],i.chunk.text)
                if key not in score_cache:missing.append(i)
            if missing:
                reranker.rerank(b['question'],missing,len(missing),CancelToken())
                for i in missing:score_cache[b['question'],i.chunk.text]=i.rerank_score
            # Recompute auxiliary weights with unchanged production scoring.
            from local_full_text_search.rag.reranking.support import retrieval_prior
            for i in items:
                i.rerank_score=score_cache[b['question'],i.chunk.text]
                i.rerank_score_combined=reranker.cross_encoder_weight*i.rerank_score+reranker.retrieval_prior_weight*retrieval_prior(i)+reranker.answer_support_weight*i.answer_support_score
            items.sort(key=lambda i:(-i.rerank_score_combined,-i.fused_score,i.chunk.stable_chunk_id))
            items=reranker._apply_rank_guard(items);ordered=ConstraintMatcher().evaluate(plan,items)
            selected=[];docs=Counter();used=0
            # Same original filters and stable condition ordering. C charges every page.
            for i in ordered:
                if policy.rejection_reason(i):continue
                cost=physical.get(i.chunk.stable_chunk_id,1)
                if used+cost>8 or docs[i.chunk.document_key]+cost>3:continue
                selected.append(i);used+=cost;docs[i.chunk.document_key]+=cost
            check=EvidencePolicy(min_vector_score=policy.min_vector_score,min_rerank_score=policy.min_rerank_score,max_per_document=8).evaluate(q,selected,plan=plan)
            context=builder.build(check.accepted,max_chars=12000);byid={i.chunk.stable_chunk_id:i for i in check.accepted}
            stage=[{**candidate_snapshot(byid[e['chunk_id']],selected=True),'included_text':e['included_text']} for e in context.entries]
            score=score_stage('context',stage,maps[cid]);before=b['stages']['context']
            # Legacy negative matcher counts entire page. Also measure actual quoted text.
            canon=canonical_content(context.text)
            actual_neg=[m.evidence_id for m in maps[cid] if m.kind=='hard_negative' and canonical_content(m.quote) and canonical_content(m.quote) in canon]
            oldcanon=canonical_content(b['context']);oldneg=[m.evidence_id for m in maps[cid] if m.kind=='hard_negative' and canonical_content(m.quote) and canonical_content(m.quote) in oldcanon]
            r={'id':cid,'arm':arm,'question':b['question'],'cohort':b['cohort'],'before':before,'after':score,'context':context.text,'entries':list(context.entries),'stage':stage,'parents':parents,'physical_pages':used,'chars':len(context.text),'changed':context.text!=b['context'],'lost':sorted(set(before['gold_completion_ranks'])-set(score['gold_completion_ranks'])),'gained':sorted(set(score['gold_completion_ranks'])-set(before['gold_completion_ranks'])),'actual_new_negative_quotes':sorted(set(actual_neg)-set(oldneg)),'row_hits':[u['id'] for u in hits]}
            results.append(r)
            with (OUT/'cases.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
        print('done',cid,flush=True)
    conn.close();assert all(sha(p)==h for p,h in hashes.items())
    summary={}
    for arm in ('P','T','C'):
        rs=[r for r in results if r['arm']==arm];pos=[r for r in rs if r['cohort']=='formal48' and r['before']['gold_evidence_recall'] is not None]
        summary[arm]={'changed':sum(r['changed'] for r in rs),'formal_positive_count':len(pos),'before_hit':sum(r['before']['gold_hit'] for r in pos),'after_hit':sum(r['after']['gold_hit'] for r in pos),'before_coverage':sum(r['before']['gold_evidence_recall'] for r in pos)/len(pos),'after_coverage':sum(r['after']['gold_evidence_recall'] for r in pos)/len(pos),'gains':[r['id'] for r in rs if r['gained']],'losses':[r['id'] for r in rs if r['lost']],'new_actual_negative_quotes':[r['id'] for r in rs if r['actual_new_negative_quotes']],'gate':not any(r['lost'] or r['actual_new_negative_quotes'] for r in rs)}
    dump('summary.json',summary);dump('completion.json',{'finished':str(datetime.now()),'rows':len(results),'hashes_unchanged':True,'new_cross_encoder_pairs':len(score_cache),'combination':'Run only after both independent interventions pass retention gate.'})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
