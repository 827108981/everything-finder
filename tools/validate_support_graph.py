"""Native entry validation: serialize local inference, overlap remote calls.

Gold reaches only the unchanged evaluator after AnswerService completes.
Runtime snapshots and explicit settings prevent a long run from mixing edits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from threading import Lock


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime-root',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--suite-root',type=Path,required=True)
    parser.add_argument('--arm',choices=['A','B'],required=True)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    args.runtime_root=args.runtime_root.resolve()
    sys.path.insert(0,str(args.runtime_root.resolve()))
    from tools.validate_evidence_answer_flow import Experiment
    from tools.validate_query_plan_adapter import STAGES
    from tools.recheck_evidence_flow_native import public_settings,now
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.core.task_manager import CancelToken
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.rag.models import RagQuery
    from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
    from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
    from local_full_text_search.rag.evaluation.stage_scoring import score_stage
    from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
    from local_full_text_search.rag.evaluation.runner import _answer_record
    from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
    out=args.output_dir.resolve();root=args.suite_root.resolve()
    exp=Experiment(out,1)
    exp.settings=AppSettings.from_dict(json.loads((out/'settings_before.json').read_text(encoding='utf8')))
    flags={'rag_query_plan_enabled':False,'rag_structured_answer_enabled':False,
           'rag_answer_review_enabled':False,'rag_answer_review_strict':True,'rag_semantic_clarification_enabled':False}
    if args.arm=='B': flags={k:k!='rag_answer_review_strict' for k in flags}
    for k,v in flags.items():setattr(exp.settings,k,v)
    s=exp.settings
    manifest=json.loads((root/'tests/rag/fixtures/rag_active_suite_v5.json').read_text(encoding='utf8'))
    files=[root/d['path'] for d in manifest['datasets']]
    cases=[c for p in files for c in load_gold_dataset(p,reviewed_only=False).cases]
    path=out/(args.arm+'.jsonl')
    done={r['id'] for r in map(json.loads,path.read_text(encoding='utf8').splitlines())} if path.exists() else set()
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.runtime_root/'local_full_text_search/rag').rglob('*.py')}
    protocol={'started':now(),'bundle':exp.bundle.bundle_id,'settings':public_settings(s),
        'code_hashes':hashes,'dataset_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        'cases':[(c.review_status,c.case_id) for c in cases], 'runtime_root':str(args.runtime_root.resolve()),
        'flags':flags,'preserved_completed':sorted(done),'workers':args.workers,
        'protocol':'Real AnswerService; one local inference at a time, parallel independent remote requests. No quality retries. Existing completed records preserved.'}
    manifest_path=out/(args.arm+'_manifest.json')
    if manifest_path.exists() and done:
        old=json.loads(manifest_path.read_text(encoding='utf8'))
        assert old['code_hashes']==hashes,'runtime drift'
        assert old['dataset_hashes']==protocol['dataset_hashes'],'dataset drift'
        (out/(args.arm+'_parallel_resume.json')).write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    else: manifest_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    lock=Lock()
    def work(case):
        row={'id':case.case_id,'question':case.question,'group_id':case.group_id,'review_status':case.review_status,
             'cohort':'formal' if case.review_status=='reviewed' else 'diagnostic'}
        stages={};events=[];token=CancelToken();gateway=OpenAICompatibleGateway(max_transient_retries=2)
        lock.acquire();held=True
        def progress(e):
            nonlocal held
            for field,stage in STAGES.items():
                if field in e:stages[stage]=e[field]
            events.append({k:v for k,v in e.items() if k not in STAGES})
            if held and e.get('stage')=='credential_read':
                held=False;lock.release()
        try:
            service=AnswerService(exp.source_store,exp.store,exp.embedding,exp.ranker,gateway,exp.secrets,s,progress_callback=progress)
            query=RagQuery(case.question,include_ocr=True,min_ocr_confidence=s.ocr_min_confidence,
                lexical_top_k=s.rag_lexical_top_k,vector_top_k=s.rag_vector_top_k,fusion_top_k=s.rag_fusion_top_k,
                reranker_top_k=s.rag_reranker_top_k,final_top_k=s.rag_final_top_k)
            row['result']=_answer_record(service.answer(query,token))
        except Exception as exc:row['error']=str(exc).replace(exp.key,'[REDACTED]')
        finally:
            if held:lock.release()
        context=next((e for e in reversed(events) if 'llm_context' in e),{})
        row.update(context=context.get('llm_context',''),entries=context.get('context_entries',[]),events=events,candidate_stages=stages)
        mapping=map_case_evidence(case,exp.chunk_rows)
        row['scores']={stage:score_stage(stage,items,mapping) for stage,items in {**stages,'context':row['entries']}.items()}
        if 'result' in row:
            try:row['judgement']=GoldAnswerJudge(gateway,s,exp.key).evaluate(case,{**row['result'],'evidence_context':row['context']},token)
            except Exception as exc:row['judge_error']=str(exc).replace(exp.key,'[REDACTED]')
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,c) for c in cases if c.case_id not in done]
        for future in as_completed(futures):
            row=future.result()
            with path.open('a',encoding='utf8') as stream:stream.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
            print(args.arm,row['id'],row['scores']['context']['gold_evidence_found'],row.get('judgement',{}).get('verdict'),row.get('error',''),flush=True)
    final=list(map(json.loads,path.read_text(encoding='utf8').splitlines()))
    assert len(final)==len(cases)==len({r['id'] for r in final})
    (out/(args.arm+'_complete.json')).write_text(json.dumps({'finished':now(),'count':len(final),
        'dataset_drift':[str(p) for p in files if hashlib.sha256(p.read_bytes()).hexdigest()!=protocol['dataset_hashes'][str(p)]],
        'runtime_drift':[p for p,h in hashes.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]},indent=2),encoding='utf8')


if __name__=='__main__':main()
