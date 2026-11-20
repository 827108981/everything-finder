"""Regrade saved outputs only; no answer regeneration or Gold mutation."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
import html,json
from pathlib import Path
from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows,map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.stage_scoring import score_stage,SCORER_VERSION
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge,_request_prompt
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from tools.rag_eval_common import ROOT, lines, read, sha

OUT=ROOT/'reports/rag_scoring_v3_20261105'
def main():
    OUT.mkdir(parents=True,exist_ok=False)
    settings=AppSettings.from_dict(read(SETTINGS_PATH));key=WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or '';assert key
    def dump(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str).replace(key,'[REDACTED]')+'\n',encoding='utf-8')
    fixture=ROOT/'tests/rag/fixtures';datasets=[fixture/n for n in ('rag_gold_v4_1_review.json','rag_acceptance_v1_review.json','rag_row_holdout_v1_review.json')]
    cases={c.case_id:c for p in datasets for c in load_gold_dataset(p,reviewed_only=False).cases}
    oldsource=ROOT/'reports/rag_retrieval_ab_results_20261103/resolved_cases.jsonl'
    baseline={r['id']:r for r in lines(oldsource) if r['arm']=='baseline'}
    integration=ROOT/'reports/rag_row_integration_20261105';holdout=ROOT/'reports/rag_row_holdout_20261105'
    hcontexts={(r['id'],r['arm']):r for r in lines(integration/'cases.jsonl')}
    newcontexts={(r['id'],r['arm']):r for r in lines(holdout/'timing_cases.jsonl') if r['repeat']==1}
    inputs=[oldsource,integration/'cases.jsonl',integration/'row_answers.jsonl',integration/'remaining_row_answers.jsonl',holdout/'timing_cases.jsonl',holdout/'answers.jsonl',holdout/'008_repeat_answers.json']
    hashes={str(p):sha(p) for p in [SETTINGS_PATH,*datasets,*inputs,__file__,*list((ROOT/'local_full_text_search/rag/evaluation').rglob('*.py'))]}
    # Changes outside evaluation are not part of this task.
    previous=read(integration/'protocol.json')['hashes']
    allowed='/local_full_text_search/rag/evaluation/'
    assert all(sha(p)==h for p,h in previous.items() if allowed not in p.replace('\\','/'))
    dump('protocol.json',{'started':str(datetime.now()),'scorer_version':SCORER_VERSION,'judge_version':'object-source-binding-v3','model':settings.rag_api_model,'hashes':hashes,'scope':'59 historical baseline outputs (48 formal+024pending+10regression pending),36 changed-context row experiment outputs,20 new holdout outputs,4 exact-object repeats. Rejudge original saved text/context, no generation. All pending statuses preserved.','separation':'Full fixed original17 and added31 compared separately. Changed-question repeated outputs and new pending holdout never folded into full-set metrics.','dedup':'Identical judge request JSON is evaluated once and reused, not counted as independent model trials. No quality retries; JSON-format retry only via production judge.'})
    store=RagIndexStore(Path('E:/everythingfinder-db/rag'));conn=store.connect_read_only(store.active().metadata_path);chunk_rows=load_chunk_rows(conn);conn.close()
    mappings={cid:map_case_evidence(c,chunk_rows) for cid,c in cases.items() if c.review_status!='disputed'}
    units={u['id']:u for u in read(integration/'row_index.json')}
    def stage(entries):
        result=[]
        for e in entries:
            members={r['chunk_id'] for r in e.get('sources',[])}
            if e['chunk_id'] in units:members.update(units[e['chunk_id']]['parents'])
            result.append({'chunk_id':e['chunk_id'],'member_chunk_ids':sorted(members),'included_text':e['included_text']})
        return result
    records=[]
    def add(r,cohort,context,entries,old_judgement,repeat=1,old_score=None,stages=None):
        cid=r['id'];answer=dict(r.get('answer') or {});answer['evidence_context']=context
        new_score=score_stage('context',stage(entries),mappings[cid]);rescored={}
        if stages:
            rescored={s:score_stage(s,v,mappings[cid]) for s,v in stages.items()}
        records.append({'id':cid,'arm':r.get('arm','baseline'),'cohort':cohort,'repeat':repeat,'review_status':cases[cid].review_status,'answer':{k:v for k,v in answer.items() if k!='evidence_context'},'context':context,'old_judgement':old_judgement,'old_score':old_score,'new_score':new_score,'rescored_stages':rescored,'request_answer':answer})
    for r in baseline.values():add(r,'historical_baseline',r['context'],r['context_entries'],r.get('gold_judgement'),old_score=r['stages']['context'],stages=r['candidate_stages'])
    for name in ('row_answers.jsonl','remaining_row_answers.jsonl'):
        for r in lines(integration/name):
            src=baseline[r['id']] if r['arm']=='baseline' else hcontexts[r['id'],'H8']
            add(r,'row_changed',src['context'],src.get('entries',src.get('context_entries')),r.get('judgement'),r['repeat'],src.get('after',src.get('stages',{}).get('context')))
    for r in lines(holdout/'answers.jsonl'):
        src=newcontexts[r['id'],r['arm']];add(r,'holdout_pending',src['context'],src['entries'],r.get('judgement'),old_score=src['score'])
    for r in read(holdout/'008_repeat_answers.json'):
        src=newcontexts[r['id'],r['arm']];r={**r,'answer':{'text':r['text'],'status':src['status']}}
        add(r,'object_repeat_pending',src['context'],src['entries'],None,r['repeat'],src['score'])
    assert len(records)==119
    requests={}
    for i,r in enumerate(records):
        digest=hashlib.sha256(_request_prompt(cases[r['id']],r['request_answer']).encode()).hexdigest();r['judge_request_sha256']=digest;requests.setdefault(digest,[]).append(i)
    dump('input_records.json',records)
    def judge_job(digest,indices):
        r=records[indices[0]]
        try:return digest,GoldAnswerJudge(OpenAICompatibleGateway(max_transient_retries=2),settings,key).evaluate(cases[r['id']],r['request_answer'],CancelToken()),None
        except Exception as e:return digest,None,str(e).replace(key,'[REDACTED]')[:500]
    done=0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for f in as_completed([pool.submit(judge_job,digest,indices) for digest,indices in requests.items()]):
            digest,judged,error=f.result()
            for i in requests[digest]:
                r=records[i];r.pop('request_answer');r['new_judgement']=judged;r['judge_error']=error
                with (OUT/'cases.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(r,ensure_ascii=False,default=str).replace(key,'[REDACTED]')+'\n')
            done+=1;print('judged',done,'/',len(requests),records[requests[digest][0]]['id'],(judged or {}).get('verdict'),bool(error),flush=True)
    assert all(sha(p)==h for p,h in hashes.items())
    dump('completion.json',{'finished':str(datetime.now()),'records':len(records),'unique_requests':len(requests),'judge_errors':sum(bool(r['judge_error']) for r in records),'hashes_unchanged':True,'generation_calls':0,'gold_unchanged':True})
if __name__=='__main__':main()
