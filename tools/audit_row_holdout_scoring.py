from dataclasses import replace
from pathlib import Path
from tools.validate_row_integration import ROOT
from tools.validate_material_steps import sha
from tools.row_evaluation_candidate import negative_span_audit
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.content_identity import canonical_content
import json
OUT=ROOT/'reports/rag_row_holdout_20261105'
def main():
    assert (OUT/'correctness_completion.json').exists() and (OUT/'answer_completion.json').exists(),'Stage1 must complete first'
    units=json.loads((ROOT/'reports/rag_row_integration_20261105/row_index.json').read_text(encoding='utf-8'))
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'))
    conn=store.connect_read_only(store.active().metadata_path);rows=load_chunk_rows(conn);conn.close()
    base=load_gold_dataset(ROOT/'tests/rag/fixtures/rag_row_holdout_v1_review.json',reviewed_only=False).cases[0]
    controlled=[]
    specifications=[('alarm','化学发光定标处理思路分析.pptx',11,'CSDB','DCF'),('pump','生化场景化培训教材-液路_1.0.pptx',46,'P05','P12'),('motor','IVD_生化_BS-5000系列_14 样本管理系统.pptx',13,'IMFH1','IMFH2')]
    for label,name,page,a,b in specifications:
        def find(key):return next(u for u in units if u['file'].replace('\\','/').split('/')[-1]==name and u['page']==page and any(c['value']==key for c in u['bindings']))
        left,right=find(a),find(b);path=left['file'];quoteA='\n'.join(left['source_values']);quoteB='\n'.join(right['source_values'])
        case=replace(base,case_id='audit-'+label,group_id='audit-'+label,gold_evidence=({'file':path,'page':page,'quote':quoteA},),hard_negatives=({'file':path,'page':page,'quote':quoteB},))
        maps=map_case_evidence(case,rows);page_rows=[r for r in rows if r['file_path']==path and (r.get('slide_number') or r.get('page_number'))==page]
        full='\n'.join(r['text'] for r in sorted(page_rows,key=lambda r:r['ordinal']));assert canonical_content(quoteA) in canonical_content(full) and canonical_content(quoteB) in canonical_content(full)
        views=[('positive_row',left['text'],left['parents'],False),('negative_row',right['text'],right['parents'],True),('full_page',full,[r['chunk_id'] for r in page_rows],True),('truncated_negative',quoteB[:max(1,len(quoteB)//2)],right['parents'],False)]
        for name_view,text,parents,expected in views:
            stage=[{'chunk_id':'view-'+label+'-'+name_view,'member_chunk_ids':parents,'included_text':text}];legacy=score_stage('context',stage,maps);audit=negative_span_audit(stage,maps,rows)
            assert bool(audit['negative_quote_hits'])==expected
            controlled.append({'table':label,'view':name_view,'file':path,'page':page,'positive_quote':quoteA,'negative_quote':quoteB,'included_text':text,'legacy_negative_count':legacy['hard_negative_found'],'span_negative_count':len(audit['negative_quote_hits']),'expected_complete_negative':expected,'audit':audit})
    (OUT/'controlled_scoring_audit.json').write_text(json.dumps(controlled,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Apply both interpretations to historical frozen H8. This changes no retrieval.
    oldcases={c.case_id:c for name in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/name,reviewed_only=False).cases}
    records=[json.loads(l) for l in (ROOT/'reports/rag_row_integration_20261105/cases.jsonl').open(encoding='utf-8')];auditrows=[];byunit={u['id']:u for u in units}
    for r in [r for r in records if r['arm']=='H8']:
        stage=[]
        for e in r['entries']:
            ids={s['chunk_id'] for s in e.get('sources',[])}
            if e['chunk_id'] in byunit:ids.update(byunit[e['chunk_id']]['parents'])
            stage.append({'chunk_id':e['chunk_id'],'member_chunk_ids':sorted(ids),'included_text':e['included_text']})
        maps=map_case_evidence(oldcases[r['id']],rows);legacy=score_stage('context',stage,maps);audit=negative_span_audit(stage,maps,rows)
        assert legacy['gold_evidence_found']==r['after']['gold_evidence_found']
        auditrows.append({'id':r['id'],'legacy_negative_ids':legacy['hard_negative_ids'],**audit})
    (OUT/'historical_scoring_audit.json').write_text(json.dumps(auditrows,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (OUT/'scoring_completion.json').write_text(json.dumps({'controlled_views':len(controlled),'historical_H8_cases':len(auditrows),'legacy_controlled_false_complete':sum(bool(r['legacy_negative_count']) and not r['expected_complete_negative'] for r in controlled),'span_controlled_mismatches':0,'historical_page_flag_cases':sum(bool(r['legacy_negative_ids']) for r in auditrows),'historical_full_quote_cases':sum(bool(r['negative_quote_hits']) for r in auditrows),'unresolved_mapping_cases':sum(bool(r['unresolved_mapping']) for r in auditrows),'production_scorer_changed':False,'meaning':'The difference is evaluator interpretation, not retrieval/answer improvement. Truncated negative content is not full quote recall; partial exposure still needs auditing.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print((OUT/'scoring_completion.json').read_text(encoding='utf-8'))
if __name__=='__main__':main()
