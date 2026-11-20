"""Two independent prototypes: source header/binding audit and monotone selection replay."""
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime
import copy,hashlib,html,json,re
from pathlib import Path
from tools.validate_material_steps import ROOT,sha
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.models import EvidenceChunk,RagQuery,ConstraintStatus
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.trace import candidate_snapshot

OUT=ROOT/'reports/rag_material_repairs_20261105'
PREV=ROOT/'reports/rag_structured_material_20261105'

def bind_row(headers,cells):
    assert len(headers)==len(cells)
    return [{'column':n,'header':h,'value':c if c else None,'empty':not bool(c)} for n,(h,c) in enumerate(zip(headers,cells),1)]

def select_preserving(original,candidates,policy,limit=8):
    """Preserve baseline order/bodies. Only fill unused slots, hard matches only.
    No Gold, IDs, special identifiers, or cases. Duplicate full bodies merge only
    within exact same document; append limits do not apply to historical originals.
    """
    selected=list(original);trace=[]
    seen={(x.chunk.document_key,canonical_content(x.chunk.text)) for x in selected}
    selected_ids={x.chunk.stable_chunk_id for x in selected}
    for x in candidates:
        if len(selected)>=limit:break
        if x.chunk.stable_chunk_id in selected_ids:continue
        if policy.rejection_reason(x):continue
        d=x.constraint_decision
        if d is None or any(j.constraint.is_hard and j.status is not ConstraintStatus.MATCH for j in d.judgements):continue
        key=(x.chunk.document_key,canonical_content(x.chunk.text))
        # A row fully contained in an already included parent contributes no new text.
        if key in seen or any(x.chunk.document_key==o.chunk.document_key and canonical_content(x.chunk.text) in canonical_content(o.chunk.text) for o in selected):continue
        selected.append(x);selected_ids.add(x.chunk.stable_chunk_id);seen.add(key)
        trace.append({'id':x.chunk.stable_chunk_id,'reason':'free_slot_and_all_hard_conditions_match','document_cap_exception':sum(i.chunk.document_key==x.chunk.document_key for i in selected)>policy.max_per_document})
    return selected,trace

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((PREV/'protocol.json').read_text(encoding='utf-8'));assert all(sha(k)==h for k,h in protocol['hashes'].items())
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    def dump(name,v):(OUT/name).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    inputs=[PREV/'structure_audit.json',PREV/'table_units.json',PREV/'cases.jsonl',ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl',Path(__file__)]
    hashes={**protocol['hashes'],**{str(p):sha(p) for p in inputs}}
    dump('protocol.json',{'started':str(datetime.now()),'bundle_id':protocol['bundle_id'],'hashes':hashes,
      'H':'Header repair only, no QA retrieval changes. Unmarked simple tables: question/Gold-free model decides whether first row is column names from first six original rows. Native cell values never generated. Merged/ambiguous tables still fallback. Explicit column/header/value/null serialization.',
      'S':'Selection repair only; does not use H. Restore baseline context unchanged, deduplicate same-source representations, fill free slots from frozen eligible candidate pools with all hard conditions MATCH; soften document quota only for additions. Final8/12000. No eviction when full.',
      'source_variants':'S_native uses baseline constraint pool; S_table adds previous T final candidates; S_continuation adds previous C source pages. Uses existing scores, no new retrieval or reranking. These are bounded replay variants, not full end-to-end candidates.',
      'gate':'No baseline body/complete quote loss; no newly included marked negative quote. Only improved contexts past gate eligible for repeated fresh answer tests. No tuning or quality retry after freeze.'})
    audit=json.loads((PREV/'structure_audit.json').read_text(encoding='utf-8'))
    pending={}
    for t in audit:
        if t.get('reason')=='header_not_marked_in_source':
            view=t['raw_cells'][:6];uid=hashlib.sha256(json.dumps(view,ensure_ascii=False).encode()).hexdigest();pending.setdefault(uid,view)
    groups=[list(pending.items())[i:i+6] for i in range(0,len(pending),6)]
    def classify(group):
        try:
            g=OpenAICompatibleGateway(max_transient_retries=2)
            res=g.complete_text(TextCompletionRequest(system_prompt='你只判断表格结构。输入是多个表格的前六行原始单元格。判断第一行是否确实为列名，后续行为数据。全是数据、上下层复杂表头或不确定时返回false。不要修改任何单元格，不使用领域答案。仅返回JSON {"tables":[{"id":"原id","is_header":true或false,"reason":"简短结构依据"}]}，每个id一次。',user_prompt=json.dumps([{'id':i,'rows':r} for i,r in group],ensure_ascii=False),model=settings.rag_api_model,api_base_url=settings.rag_api_base_url,api_style=settings.rag_api_style,api_key=key,timeout_seconds=settings.rag_api_timeout_seconds,max_output_tokens=1500),CancelToken())
            raw=res if isinstance(res,str) else res.text
            value=json.loads(re.sub(r'^```(?:json)?\s*|\s*```$','',raw.strip()))['tables']
            assert {x['id'] for x in value}=={i for i,r in group} and len(value)==len(group)
            assert all(type(x['is_header']) is bool and isinstance(x['reason'],str) for x in value)
            return value
        except Exception as exc:return [{'id':i,'is_header':False,'reason':'execution_error: '+str(exc).replace(key,'[REDACTED]')[:200],'error':True} for i,r in group]
    decisions={}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(classify,g) for g in groups]):
            decisions.update({r['id']:r for r in f.result()});print('headers',len(decisions),'/',len(pending),flush=True)
    dump('header_decisions.json',decisions)
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));assert store.active().bundle_id==protocol['bundle_id']
    conn=store.connect_read_only(store.active().metadata_path);indexrows=load_chunk_rows(conn);pages=defaultdict(list)
    for r in indexrows:pages[r['file_path'],r.get('slide_number') or r.get('page_number')].append(r)
    bindings=[];ha=[]
    for t in audit:
        accepted=t['status']=='accepted';reason=t.get('reason')
        if reason=='header_not_marked_in_source':
            uid=hashlib.sha256(json.dumps(t['raw_cells'][:6],ensure_ascii=False).encode()).hexdigest();accepted=decisions[uid]['is_header'];reason=decisions[uid]['reason']
        ha.append({**t,'new_accepted':accepted,'header_decision':reason})
        if not accepted:continue
        rs=pages[t['file'],t['page']];body=canonical_content('\n'.join(r['text'] for r in rs))
        for n,row in enumerate(t['raw_cells'][1:],2):
            if not any(row):continue
            assert len(row)==len(t['headers']) and canonical_content('\n'.join(row)) in body
            bound=bind_row(t['headers'],row)
            assert all(x['empty'] or canonical_content(x['value']) in body for x in bound)
            bindings.append({'file':t['file'],'page':t['page'],'table':t['table'],'row':n,'bindings':bound,'source_values':row,'cell_and_order_verified':True})
    dump('header_audit.json',ha);dump('bound_rows.json',bindings)
    cases={c.case_id:c for n in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/n,reviewed_only=False).cases}
    maps={cid:map_case_evidence(c,indexrows) for cid,c in cases.items() if c.review_status!='disputed'}
    baseline=[r for r in (json.loads(l) for l in inputs[3].open(encoding='utf-8')) if r['arm']=='baseline']
    previous={(r['id'],r['arm']):r for r in (json.loads(l) for l in (PREV/'cases.jsonl').open(encoding='utf-8'))}
    cache={}
    def load(cid):
        if cid not in cache:cache[cid]=store.load_chunk(conn,cid)
        return cache[cid]
    def restore(s):
        ch=load(s['chunk_id']);assert ch is not None and ch.text==s['text']
        return EvidenceChunk(ch,**{k:s[k] for k in ('lexical_rank','lexical_score','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined','answer_support_score') if k in s},member_chunk_ids=tuple(s.get('member_chunk_ids',[])),sources=tuple(load(x['chunk_id']).source for x in s.get('sources',[])))
    policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=3);builder=ContextBuilder();results=[]
    for b in baseline:
        cid=b['id'];plan=QueryInterpreter().interpret(b['question']);native=[restore(s) for s in b['candidate_stages']['constraints']]
        native=list(ConstraintMatcher().evaluate(plan,native))
        decision=policy.evaluate(RagQuery(text=b['question'],final_top_k=8),native,plan=plan)
        orig=builder.build(decision.accepted,max_chars=12000);assert orig.text==b['context']
        for arm in ('S_native','S_table','S_continuation'):
            pool=list(native)
            if arm=='S_table':
                # Row views point back to actual original page IDs. No fabricated scores.
                for s in previous[cid,'T']['stage']:
                    for pid in s.get('member_chunk_ids',[]):
                        if any(x.chunk.stable_chunk_id==pid for x in pool):continue
                        ch=load(pid)
                        if ch:pool.append(EvidenceChunk(ch,lexical_rank=s.get('lexical_rank'),vector_score=s.get('vector_score'),rerank_score=s.get('rerank_score'),rerank_score_combined=s.get('rerank_score_combined')))
            if arm=='S_continuation':
                # Retain native C's existing original-question scores for recalled pages.
                for s in previous[cid,'C']['stage']:
                    for pid in s.get('member_chunk_ids',[]):
                        if any(x.chunk.stable_chunk_id==pid for x in pool):continue
                        ch=load(pid)
                        if ch:pool.append(EvidenceChunk(ch,lexical_rank=s.get('lexical_rank'),vector_score=s.get('vector_score'),rerank_score=s.get('rerank_score'),rerank_score_combined=s.get('rerank_score_combined')))
            pool=list(ConstraintMatcher().evaluate(plan,pool))
            selected,trace=select_preserving(decision.accepted,pool,policy)
            context=builder.build(selected,max_chars=12000);byid={x.chunk.stable_chunk_id:x for x in selected}
            snaps=[{**candidate_snapshot(byid[e['chunk_id']],selected=True),'included_text':e['included_text']} for e in context.entries]
            score=score_stage('context',snaps,maps[cid]);before=b['stages']['context']
            old=canonical_content(b['context']);new=canonical_content(context.text)
            negatives=[m.evidence_id for m in maps[cid] if m.kind=='hard_negative' and canonical_content(m.quote) and canonical_content(m.quote) in new and canonical_content(m.quote) not in old]
            r={'id':cid,'arm':arm,'question':b['question'],'cohort':b['cohort'],'before':before,'after':score,'changed':context.text!=b['context'],'context':context.text,'entries':list(context.entries),'trace':trace,'lost':sorted(set(before['gold_completion_ranks'])-set(score['gold_completion_ranks'])),'gained':sorted(set(score['gold_completion_ranks'])-set(before['gold_completion_ranks'])),'new_negative_quotes':negatives,'original_bodies_retained':all(canonical_content(e['included_text']) in new for e in b['context_entries']),'chars':len(context.text)}
            results.append(r)
    conn.close();dump('selection_cases.json',results)
    summary={}
    for arm in ('S_native','S_table','S_continuation'):
        rs=[r for r in results if r['arm']==arm];pos=[r for r in rs if r['cohort']=='formal48' and r['before']['gold_evidence_recall'] is not None]
        summary[arm]={'changed':sum(r['changed'] for r in rs),'before_hit':sum(r['before']['gold_hit'] for r in pos),'after_hit':sum(r['after']['gold_hit'] for r in pos),'gains':[r['id'] for r in rs if r['gained']],'losses':[r['id'] for r in rs if r['lost']],'new_negative_quotes':[r['id'] for r in rs if r['new_negative_quotes']],'all_bodies_retained':all(r['original_bodies_retained'] for r in rs),'gate':all(r['original_bodies_retained'] and not r['lost'] and not r['new_negative_quotes'] for r in rs)}
    hsummary={'previous_tables':sum(t['status']=='accepted' for t in audit),'now_tables':sum(t['new_accepted'] for t in ha),'old_rows':len(json.loads((PREV/'table_units.json').read_text(encoding='utf-8'))),'now_rows':len(bindings),'blank_rows':sum(any(c['empty'] for c in r['bindings']) for r in bindings),'unique_classifications':len(pending),'batch_calls':len(groups),'classification_errors':sum(bool(x.get('error')) for x in decisions.values())}
    dump('summary.json',{'header':hsummary,'selection':summary});assert all(sha(k)==h for k,h in hashes.items())
    dump('completion.json',{'finished':str(datetime.now()),'hashes_unchanged':True,'production_and_gold_unchanged':True,'selection_rows':len(results)})
    print(json.dumps({'header':hsummary,'selection':summary},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
