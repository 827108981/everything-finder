"""Recover failed judge executions, then apply uniform v3.1 validation to saved facts."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from dataclasses import replace
from datetime import datetime
import json
from tools.rejudge_scoring_v3 import ROOT,OUT,sha,read,lines
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge,_parse_judgement
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.index_store import RagIndexStore
from pathlib import Path

def main():
    original=read(OUT/'protocol.json');inputs=read(OUT/'input_records.json');raw=lines(OUT/'cases.jsonl');assert len(raw)==119
    changed=[k for k,h in original['hashes'].items() if sha(k)!=h]
    allowed=('evaluation/answer_judge.py','evaluation/reporting/service.py','evaluation/stage_scoring.py')
    assert all(any(k.replace('\\','/').endswith(s) for s in allowed) for k in changed),changed
    settings=AppSettings.from_dict(read(SETTINGS_PATH));settings=replace(settings,rag_api_timeout_seconds=max(180,settings.rag_api_timeout_seconds))
    key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    hashes={k:sha(k) for k in original['hashes']};hashes[__file__]=sha(__file__)
    dump('v31_protocol.json',{'started':str(datetime.now()),'hashes':hashes,'preliminary_modified_files':changed,
      'preliminary_completion':'119 preliminary records were written. Preliminary hash gate caught subsequent evaluation-only fixes; preliminary counts are not the final evaluation.',
      'v31_policy':'All saved item judgements revalidated uniformly: explicit non-device identifiers missing from cited source require review; source mismatch must identify a real answer claim, otherwise uncertain. No case IDs or domain object dictionaries. Gold/content unchanged. Preliminary v3 facts reused; no quality-based LLM retries.',
      'execution_recovery':'Only22 timeout/format failures rerun;3workers;180s transport timeout runtime only, no saved settings change. Same model. Claims missing from old source judgements become uncertain, not automatically incorrect.',
      'reporting':'Page-only negatives, repeated ambiguous values and missing actual text are diagnostic. Partial negatives not complete recall.'})
    cases={c.case_id:c for n in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json','rag_row_holdout_v1_review.json') for c in load_gold_dataset(ROOT/'tests/rag/fixtures'/n,reviewed_only=False).cases}
    byinput={r['judge_request_sha256']:r for r in inputs};errors={r['judge_request_sha256'] for r in raw if r['judge_error']}
    recovered={r['judge_request_sha256']:r for r in lines(OUT/'error_retries.jsonl')} if (OUT/'error_retries.jsonl').exists() else {}
    def retry(digest):
        r=byinput[digest]
        try:return {'judge_request_sha256':digest,'new_judgement':GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=0),settings,key).evaluate(cases[r['id']],r['request_answer'],CancelToken()),'judge_error':None}
        except Exception as e:return {'judge_request_sha256':digest,'new_judgement':None,'judge_error':str(e).replace(key,'[REDACTED]')[:500]}
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(retry,d) for d in sorted(errors) if d not in recovered]):
            r=f.result();recovered[r['judge_request_sha256']]=r
            with (OUT/'error_retries.jsonl').open('a',encoding='utf-8') as out:out.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            print('recovered',len(recovered),'/',len(errors),(r.get('new_judgement') or {}).get('verdict'),r['judge_error'],flush=True)
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));conn=store.connect_read_only(store.active().metadata_path);chunks=load_chunk_rows(conn);conn.close()
    mappings={cid:map_case_evidence(c,chunks) for cid,c in cases.items() if c.review_status!='disputed'}
    units={u['id']:u for u in read(ROOT/'reports/rag_row_integration_20261105/row_index.json')}
    historical={r['id']:r for r in lines(ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl') if r['arm']=='baseline'}
    rowcontexts={(r['id'],r['arm']):r for r in lines(ROOT/'reports/rag_row_integration_20261105/cases.jsonl')}
    holdout={(r['id'],r['arm']):r for r in lines(ROOT/'reports/rag_row_holdout_20261105/timing_cases.jsonl') if r['repeat']==1}
    result=[]
    for r in raw:
        r=dict(r);r['preliminary_v3_judgement']=r['new_judgement'];r['preliminary_error']=r['judge_error']
        if r['judge_error']:
            recovered_row=recovered[r['judge_request_sha256']];r['new_judgement']=recovered_row.get('new_judgement');r['judge_error']=recovered_row.get('judge_error')
        j=r['new_judgement']
        if j:
            value={'behavior':j['behavior'],'behavior_reason':j['behavior_reason'],'required_facts':[{'id':x['id'],'status':x['status'],'reason':x['reason']} for x in j['rubrics'] if x['kind']=='required_facts'],'forbidden_claims':[{'id':x['id'],'status':x['status'],'reason':x['reason']} for x in j['rubrics'] if x['kind']=='forbidden_claims'],'object_binding':j['object_binding'],'source_binding':j['source_binding']}
            r['new_judgement']=_parse_judgement(json.dumps(value,ensure_ascii=False),cases[r['id']],settings.rag_api_model,require_binding=True,has_context=True,answer_text=r['answer']['text'],actual_context=r['context'])
        if r['cohort']=='historical_baseline':src=historical[r['id']];entries=src['context_entries']
        elif r['cohort']=='row_changed':src=historical[r['id']] if r['arm']=='baseline' else rowcontexts[r['id'],'H8'];entries=src.get('entries',src.get('context_entries'))
        else:src=holdout[r['id'],r['arm']];entries=src['entries']
        stage=[]
        for e in entries:
            ids={s['chunk_id'] for s in e.get('sources',[])}
            if e['chunk_id'] in units:ids.update(units[e['chunk_id']]['parents'])
            stage.append({'chunk_id':e['chunk_id'],'member_chunk_ids':list(ids),'included_text':e['included_text']})
        r['new_score']=score_stage('context',stage,mappings[r['id']])
        if r['cohort']=='historical_baseline':r['rescored_stages']={s:score_stage(s,v,mappings[r['id']]) for s,v in src['candidate_stages'].items()}
        result.append(r)
    dump('final_cases_v31.json',result)
    assert all(sha(k)==h for k,h in hashes.items())
    dump('completion.json',{'finished':str(datetime.now()),'records':len(result),'preliminary_records':119,'unique_preliminary_requests':116,'recovered_execution_requests':len(recovered),'judge_errors':sum(bool(r['judge_error']) for r in result),'hashes_unchanged':True,'generation_calls':0,'gold_unchanged':True,'final_judge_version':'object-source-binding-v3.1'})
    print('finished',len(result),'errors',sum(bool(r['judge_error']) for r in result),flush=True)
if __name__=='__main__':main()
