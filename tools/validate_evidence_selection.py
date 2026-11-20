"""Frozen replay of final evidence selection; no retrieval/model-score recomputation.

Primary proposal: atomic continuation groups, MMR(lambda=.7), exact-body
deduplication, source diversity as a tie-break; original count/character budgets.
quota_free is a diagnostic ablation only. Gold is never passed to the selector.
"""
from __future__ import annotations
import argparse
import copy
from dataclasses import replace
from datetime import datetime
import hashlib
import html
import json
from pathlib import Path
from collections import Counter
import time

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.models import (EvidenceChunk, CandidateDecision, ConstraintJudgement, ConstraintStatus, RagQuery)
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.content_identity import canonical_content
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

ROOT=Path(__file__).resolve().parents[1]
ARMS=('baseline','A','B','A+B')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def shingles(text):
    value=canonical_content(text)
    return {value[i:i+3] for i in range(max(0,len(value)-2))}

def select_grouped(candidates, links, limit=8, relevance_weight=.7):
    """Only already-eligible candidates; never invent source support or transfer scores."""
    by_id={x.chunk.stable_chunk_id:x for x in candidates}
    parents={k:k for k in by_id}
    def root(k):
        while parents[k]!=k:k=parents[k]
        return k
    for parent,children in links.items():
        for child,*_ in children:
            if parent in by_id and child in by_id:
                # A relationship must remain within the exact document/version.
                if by_id[parent].chunk.document_key==by_id[child].chunk.document_key:
                    parents[root(child)]=root(parent)
    grouped={}
    for rank,item in enumerate(candidates):grouped.setdefault(root(item.chunk.stable_chunk_id),[]).append((rank,item))
    units=[]
    for members in grouped.values():
        members.sort(key=lambda x:x[0])
        items=[x[1] for x in members]
        # Exact normalized identity only; numeric/units/negation variants remain separate.
        seen=set();unique=[]
        for item in items:
            body=canonical_content(item.chunk.text)
            if body not in seen:unique.append(item);seen.add(body)
        units.append({'items':unique,'rank':members[0][0],
            'relevance':max(float(i.rerank_score_combined if i.rerank_score_combined is not None else i.rerank_score or 0) for i in items),
            'grams':shingles('\n'.join(i.chunk.text for i in unique))})
    selected=[];selected_grams=[];body_ids=set();docs=Counter();trace=[]
    while units:
        ranked=[]
        for u in units:
            fresh=[i for i in u['items'] if canonical_content(i.chunk.text) not in body_ids]
            if not fresh or len(selected)+len(fresh)>limit:continue
            similarity=max((len(u['grams']&g)/max(1,len(u['grams']|g)) for g in selected_grams),default=0.)
            score=relevance_weight*u['relevance']-(1-relevance_weight)*similarity
            repeat=max((docs[i.chunk.document_key] for i in fresh),default=0)
            ranked.append((score,-repeat,-u['rank'],u,fresh,similarity))
        if not ranked:break
        best=max(ranked,key=lambda x:x[:3]);score,_,_,unit,fresh,similarity=best
        selected.extend(fresh);selected_grams.append(unit['grams']);units.remove(unit)
        for i in fresh:body_ids.add(canonical_content(i.chunk.text));docs[i.chunk.document_key]+=1
        trace.append({'ids':[i.chunk.stable_chunk_id for i in fresh],'selection_score':score,
            'frozen_relevance':unit['relevance'],'max_text_similarity':similarity,'grouped':len(unit['items'])>1})
    return selected,trace

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=ROOT/'reports/rag_retrieval_ab_results_20261103');p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();source=args.source.resolve();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    old_protocol=json.loads((source/'protocol.json').read_text(encoding='utf-8'))
    assert all(sha(p)==h for p,h in old_protocol['hashes'].items()),'Frozen inputs/config/production code changed'
    settings=AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding='utf-8')))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));bundle=store.active();assert bundle.bundle_id==old_protocol['bundle_id']
    source_path=source/'resolved_cases.jsonl'
    inputs={str(source_path):sha(source_path),str(source/'continuation_links.json'):sha(source/'continuation_links.json')}
    def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str)+'\n',encoding='utf-8')
    protocol={'started':str(datetime.now()),'source':str(source),'bundle_id':bundle.bundle_id,'source_sha256':inputs,
        'frozen_config_and_code_sha256':old_protocol['hashes'],'harness_sha256':sha(__file__),
        'proposal':{'method':'atomic continuation groups + MMR; source only breaks ties','relevance_weight':.7,'max_chunks':settings.rag_final_top_k,'max_chars':settings.rag_context_max_chars,
            'dedup':'Exact canonical body only, retaining numeric/unit/negation differences','gate':'Original thresholds and conflicts retained; no inherited model applicability'},
        'diagnostic':'quota_free removes only per-document cap, not a rollout candidate',
        'acceptance':'Within each fixed upstream arm: reproduce old context exactly; 019 correct evidence retained (where available); no per-case Gold coverage loss; no per-case annotated-negative increase; only then generate changed contexts. No Gold access in selector.'}
    dump(out/'protocol.json',protocol)
    conn=store.connect_read_only(bundle.metadata_path);chunk_rows=load_chunk_rows(conn)
    fixtures=ROOT/'tests/rag/fixtures'
    cases={c.case_id:c for f in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(fixtures/f,reviewed_only=False).cases}
    mappings={cid:map_case_evidence(c,chunk_rows) for cid,c in cases.items() if c.review_status!='disputed'}
    links=json.loads((source/'continuation_links.json').read_text(encoding='utf-8'))
    builder=ContextBuilder();policy=EvidencePolicy(min_vector_score=settings.rag_min_vector_score,min_rerank_score=settings.rag_min_rerank_score,max_per_document=settings.rag_max_chunks_per_document)
    chunk_cache={}
    def chunk(cid):
        if cid not in chunk_cache:chunk_cache[cid]=store.load_chunk(conn,cid)
        assert chunk_cache[cid] is not None
        return chunk_cache[cid]
    def restore(snap,plan):
        ch=chunk(snap['chunk_id']);assert ch.text==snap['text']
        names=('lexical_rank','lexical_score','lexical_score_kind','vector_rank','vector_score','fused_rank','fused_score','rerank_score','rerank_score_combined','answer_support_score','constraint_score')
        item=EvidenceChunk(chunk=ch,**{k:snap[k] for k in names if k in snap})
        item.member_chunk_ids=tuple(snap.get('member_chunk_ids',[]))
        item.sources=tuple(chunk(s['chunk_id']).source for s in snap.get('sources',[]))
        d=snap.get('constraint_decision')
        if d:
            judgements=[]
            for j in d['judgements']:
                q=next(c for c in plan.constraints if c.kind.value==j['kind'] and c.value==j['value'] and c.source_text==j['source_text'] and c.strength.value==j['strength'] and c.polarity.value==j['polarity'])
                judgements.append(ConstraintJudgement(q,ConstraintStatus(j['status']),j['reason'],j['matched_text']))
            item.constraint_decision=CandidateDecision(ch.stable_chunk_id,d['accepted'],d['reason'],tuple(judgements),
                **{k:d[k] for k in ('required_match_count','required_unknown_count','required_conflict_count','soft_match_count','soft_conflict_count','support_status')},support_spans=tuple(d['support_spans']))
        return item
    results=[];context_rows=[];reproduction=[]
    for line in source_path.open(encoding='utf-8'):
        old=json.loads(line);cid=old['id'];plan=QueryInterpreter().interpret(old['question'])
        query=RagQuery(text=old['question'],final_top_k=settings.rag_final_top_k)
        snaps=old['candidate_stages'].get('constraints',[])
        candidates=[restore(s,plan) for s in snaps]
        # Compare the old native policy and context builder before measuring changes.
        decision=policy.evaluate(query,candidates,plan=plan)
        original=builder.build(decision.accepted,max_chars=settings.rag_context_max_chars)
        assert original.text==old['context'],f'Replay mismatch {cid} {old["arm"]}'
        reproduction.append({'id':cid,'arm':old['arm'],'exact':True})
        eligible=[x for x in candidates if not policy.rejection_reason(x,check_constraints=True)]
        for method in ('quota_free','grouped_mmr'):
            if method=='quota_free':selected=eligible[:settings.rag_final_top_k];trace=[]
            else:selected,trace=select_grouped(eligible,links,settings.rag_final_top_k)
            # Retain the native critical-condition coverage check on the selected set.
            check=EvidencePolicy(min_vector_score=policy.min_vector_score,min_rerank_score=policy.min_rerank_score,max_per_document=settings.rag_final_top_k).evaluate(query,selected,plan=plan)
            context=builder.build(check.accepted,max_chars=settings.rag_context_max_chars)
            by_id={i.chunk.stable_chunk_id:i for i in check.accepted}
            stage=[{**candidate_snapshot(by_id[e['chunk_id']],selected=True),'included_text':e['included_text'],'truncated':e['truncated']} for e in context.entries]
            score=score_stage('context',stage,mappings[cid]);before=old['stages']['context']
            lost=sorted(set(before['gold_completion_ranks'])-set(score['gold_completion_ranks']))
            gained=sorted(set(score['gold_completion_ranks'])-set(before['gold_completion_ranks']))
            negatives_added=sorted(set(score['hard_negative_ids'])-set(before['hard_negative_ids']))
            changed=context.text!=old['context']
            row={'id':cid,'arm':old['arm'],'method':method,'cohort':old['cohort'],'review_status':old['review_status'],
                'before':before,'after':score,'changed_context':changed,'lost_gold':lost,'gained_gold':gained,'new_negatives':negatives_added,
                'before_ids':[e['chunk_id'] for e in old['context_entries']],'after_ids':[e['chunk_id'] for e in context.entries],
                'selection_trace':trace,'eligible_count':len(eligible),'critical_condition_decision':check.reason,
                'chars':len(context.text),'chunks':len(context.entries)}
            results.append(row)
            if changed and method=='grouped_mmr':
                context_rows.append({'id':cid,'arm':old['arm'],'question':old['question'],'context':context.text,
                    'context_entries':list(context.entries),'previous_context_sha256':hashlib.sha256(old['context'].encode()).hexdigest(),
                    'cohort':old['cohort'],'old_answer':old.get('answer'),'old_judgement':old.get('gold_judgement')})
    conn.close();assert len(reproduction)==236 and len(results)==472
    assert all(sha(p)==h for p,h in old_protocol['hashes'].items()) and all(sha(p)==h for p,h in inputs.items())
    dump(out/'reproduction.json',reproduction)
    (out/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results),encoding='utf-8')
    (out/'changed_contexts.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in context_rows),encoding='utf-8')
    summary={}
    for method in ('quota_free','grouped_mmr'):
        summary[method]={}
        for arm in ARMS:
            subset=[r for r in results if r['method']==method and r['arm']==arm]
            cohorts={}
            for cohort in ('formal48','regression10_pending','pending024'):
                rs=[r for r in subset if r['cohort']==cohort]
                positive=[r for r in rs if r['before']['gold_evidence_recall'] is not None]
                cohorts[cohort]={'count':len(rs),'positive_cases':len(positive),
                    'before_hits':sum(r['before']['gold_hit'] for r in positive),'after_hits':sum(r['after']['gold_hit'] for r in positive),
                    'before_coverage':sum(r['before']['gold_evidence_recall'] for r in positive)/len(positive),
                    'after_coverage':sum(r['after']['gold_evidence_recall'] for r in positive)/len(positive),
                    'before_negatives':sum(r['before']['hard_negative_found'] for r in rs),'after_negatives':sum(r['after']['hard_negative_found'] for r in rs)}
            summary[method][arm]={'cohorts':cohorts,'changed_count':sum(r['changed_context'] for r in subset),
                'gold_losses':[r['id'] for r in subset if r['lost_gold']], 'gold_gains':[r['id'] for r in subset if r['gained_gold']],
                'negative_increases':[r['id'] for r in subset if r['new_negatives']],
                'selection_gate_pass':not any(r['lost_gold'] or r['new_negatives'] for r in subset)}
    dump(out/'summary.json',summary)
    gate=all(summary['grouped_mmr'][a]['selection_gate_pass'] for a in ARMS)
    dump(out/'completion.json',{'finished':str(datetime.now()),'reproduced_exact':236,'replay_decisions':472,'input_hashes_unchanged':True,
        'candidate_scores_reused_and_unmodified':True,'selection_gate_pass':gate,
        'generation':'Only changed contexts eligible after retention gate. Blocked if any Gold loss or new marked negative.',
        'production_and_gold_unchanged':True})
    report(out,results,summary,gate)
    print(json.dumps(summary,ensure_ascii=True,indent=2),flush=True)

def report(out,rows,summary,gate):
    lines=['# 最终证据选择独立验证（2026-11-04）','',
        '固定上一轮四条上游路径的59题，共236组候选、分数及条件判定；先逐字复现旧上下文，再比较最终选择。正式48题，024与回归10题单列待审，007排除。',
        '规则预先固定：连续页成组、精确正文去重、MMR相关性权重0.7、来源只作平分时偏好；总段数8、字数12000、机型冲突及分数门槛保持不变。无题号/答案/Gold访问。quota_free仅作“去掉文档配额”的消融诊断。','',
        '结论：'+('证据保留门槛通过，需进一步验证变化上下文的回答。' if gate else '**证据保留门槛未通过，停止生成阶段，暂不启用。** 原因是至少一道题丢失原有Gold或增加已标注负例。'),'',
        '|路径|方法|正式题Gold命中|平均覆盖|指定反例次数|全部59题上下文变化|','|---|---|---|---|---|---|']
    for arm in ARMS:
        for method in ('quota_free','grouped_mmr'):
            s=summary[method][arm];c=s['cohorts']['formal48']
            lines.append(f"|{arm}|{method}|{c['before_hits']} → {c['after_hits']}/{c['positive_cases']}|{c['before_coverage']:.1%} → {c['after_coverage']:.1%}|{c['before_negatives']} → {c['after_negatives']}|{s['changed_count']}|")
    for arm in ARMS:
        s=summary['grouped_mmr'][arm]
        lines.extend(['','## '+arm+' 主方案逐题检查','', 'Gold丢失：'+('、'.join(s['gold_losses'])or'无')+'。', 'Gold补回：'+('、'.join(s['gold_gains'])or'无')+'。', '新增指定负例：'+('、'.join(s['negative_increases'])or'无')+'。',''])
        for r in rows:
            if r['method']=='grouped_mmr' and r['arm']==arm and (r['lost_gold'] or r['gained_gold'] or r['new_negatives'] or r['id']=='rag-biz-019'):
                lines.append(f"- {r['id']}（{r['cohort']}）：覆盖 {r['before']['gold_evidence_recall']} → {r['after']['gold_evidence_recall']}；丢失{r['lost_gold']}；补回{r['gained_gold']}；新增负例{r['new_negatives']}。")
    lines.extend(['','## 边界','', '本轮验证最终选择机制；baseline/B路径原本没有019答案页，不可能由选择器凭空补出。A/A+B已有答案页，才可检验它能否被保留。',
        '050的PPT替代来源仍单列参考，未改正式Gold。真实事实覆盖不能仅靠固定Gold命中率判断。候选相关性和条件识别错误不会被本轮修复。',
        '若保留检查失败，按事先约定不继续批量生成；没有新生成结果时不得报告答案准确率提升。',''])
    md='\n'.join(lines);(out/'report.md').write_text(md,encoding='utf-8')
    try:
        import markdown
        body=markdown.markdown(md,extensions=['tables'])
    except ImportError:body='<pre>'+html.escape(md)+'</pre>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px sans-serif;max-width:1200px;margin:30px auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:8px}pre{white-space:pre-wrap}</style>'+body,encoding='utf-8')

if __name__=='__main__':main()
