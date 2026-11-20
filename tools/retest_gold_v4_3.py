"""Fresh production-entry rerun after Gold-only edits; no candidate algorithms."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import subprocess

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from tools.validate_evidence_answer_flow import Experiment
from tools.rag_eval_common import ROOT, read, write, sha
from tools.rag_trace_audit import audit_case_sources
from tools.recheck_evidence_flow_native import public_settings, now

OUT=ROOT/'reports/rag_gold_v4_3_retest_20261110'
F=ROOT/'tests/rag/fixtures'
MODIFIED={'rag-biz-'+x for x in ('019','033','040','049')}
STAGES={'lexical_candidates':'lexical','vector_candidates':'vector','fused_candidates':'fusion',
    'deduplicated_candidates':'deduplicated','reranked_candidates':'rerank',
    'constraint_candidates':'constraints','evidence_candidates':'evidence'}

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    assert args.resume or not (OUT/'runs.jsonl').exists(), 'Fresh output only; use --resume after interruption'
    files=[F/'rag_gold_v4_3_review.json',F/'rag_gold_v4_3.jsonl',F/'rag_gold_v4_2_review.json',F/'rag_gold_v4_2.jsonl',
        F/'rag_acceptance_v1_review.json',F/'rag_row_holdout_v1_review.json',F/'rag_row_holdout_v2_review.json',
        ROOT/'reports/rag_evidence_answer_native_recheck_20261109/acceptance_review.json']
    load=lambda p: load_gold_dataset(p,reviewed_only=False).cases
    main_cases=load(files[0])
    pairs=[('formal',c) for c in main_cases if c.review_status=='reviewed']
    pairs += [('main_pending',c) for c in main_cases if c.review_status=='pending']
    for p in files[4:7]: pairs += [(p.stem,c) for c in load(p)]
    pairs += [('previous_independent_pending',c) for c in load(files[7])]
    assert len(pairs)==91 and len({c.case_id for _,c in pairs})==91
    tasks=[(cohort,c,1) for cohort,c in pairs]+[(cohort,c,r) for cohort,c in pairs if c.case_id in MODIFIED for r in (2,3)]
    exp=Experiment(OUT,args.workers)
    assert all(c.corpus_snapshot==exp.bundle.bundle_id for _,c in pairs)
    source_audit=read(OUT/'source_audit.json')
    assert all(x['original_quote'] and x['index_quote'] for x in source_audit)
    paths=[*files,Path(__file__),ROOT/'tools/validate_evidence_answer_flow.py',SETTINGS_PATH,
        exp.store.manifest_path,exp.bundle.metadata_path,*(ROOT/'local_full_text_search/rag').rglob('*.py'),
        *{Path(x['resolved_path']) for x in source_audit}]
    hashes={str(p):sha(p) for p in paths}
    protocol={'started':now(),'bundle':exp.bundle.bundle_id,'settings':public_settings(exp.settings),
        'input_sha256':hashes,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'first_round_counts':dict(Counter(cohort for cohort,_ in pairs)), 'total_runs':len(tasks),
        'changed_ids':sorted(MODIFIED),'repeat_policy':'四个修订题各三次；其余每题一次；仅传输/JSON结构错误自动重试，不根据质量重跑。',
        'comparison_policy':'Gold-only correction, not algorithm improvement. Changed questions not paired algorithm comparisons. Pending cohorts diagnostic only.',
        'frozen_candidates':'不启用行级、分支召回、作用域或证据清单实验候选。使用当前应用生产AnswerService与原生成网关。',
        'passes':'原文/索引核验已通过；结束后核验代码、设置、历史和新Gold哈希。',
        'stop_policy':'运行全部99次后分析；不因质量结果调整Gold；不另加算法。'}
    prior=[]
    if args.resume:
        original=read(OUT/'protocol.json')
        assert original['settings']==protocol['settings'] and original['bundle']==protocol['bundle']
        changed=[p for p,h in original['input_sha256'].items() if sha(p)!=h]
        assert set(changed)<={str(Path(__file__))}, changed
        prior=[json.loads(line) for line in (OUT/'runs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        done={(r['id'],r['repeat']) for r in prior}
        assert len(done)==len(prior)
        tasks=[t for t in tasks if (t[1].case_id,t[2]) not in done]
        write(OUT/'resume_manifest.json',{'resumed':now(),'preserved_rows':len(prior),'remaining_runs':len(tasks),
            'changed_files':changed,'reason':'Previous process stopped after 69 saved results; resume only missing runs; no quality-based retry.',
            'input_sha256':hashes})
    else:
        write(OUT/'protocol.json',protocol)
        (OUT/'working_tree_before.txt').write_text(subprocess.check_output(['git','status','--short'],text=True),encoding='utf-8')
    print('START',len(tasks),exp.bundle.bundle_id,exp.settings.rag_api_model,flush=True)

    def work(cohort,case,repeat):
        token=CancelToken(); events=[]; stages={}; attempts=[]; held=[True]
        capture={'id':case.case_id,'question':case.question,'cohort':cohort,'repeat':repeat,
            'review_status':case.review_status,'group_id':case.group_id,'expected_behavior':case.expected_behavior,
            'formal_scored':cohort=='formal','started':now()}
        exp.lock.acquire()
        def progress(e):
            for field,stage in STAGES.items():
                if isinstance(e.get(field),list): stages[stage]=e[field]
            events.append({k:v for k,v in e.items() if k not in STAGES})
            if held[0] and e.get('stage') in ('credential_read','retrieval_completed'):
                held[0]=False; exp.lock.release()
        gateway=OpenAICompatibleGateway(max_transient_retries=2,retry_callback=attempts.append)
        s=exp.settings
        service=AnswerService(exp.source_store,exp.store,exp.embedding,exp.ranker,gateway,exp.secrets,s,progress_callback=progress)
        query=RagQuery(text=case.question,include_ocr=True,min_ocr_confidence=s.ocr_min_confidence,
            lexical_top_k=s.rag_lexical_top_k,vector_top_k=s.rag_vector_top_k,fusion_top_k=s.rag_fusion_top_k,
            reranker_top_k=s.rag_reranker_top_k,final_top_k=s.rag_final_top_k)
        try:
            capture['answer']=_answer_record(service.answer(query,token))
        except Exception as exc:
            capture['answer_error']=exp.redact(exc)
        finally:
            if held[0]: held[0]=False;exp.lock.release()
        ctx=next((e for e in reversed(events) if 'llm_context' in e),{})
        entries=ctx.get('context_entries',[])
        capture.update(context=ctx.get('llm_context',''),entries=entries,events=events,candidate_stages=stages)
        if capture.get('answer'):
            answer={**capture['answer'],'evidence_context':capture['context']}
            try: capture['judgement']=GoldAnswerJudge(gateway,s,exp.key).evaluate(case,answer,token)
            except Exception as exc: capture['judge_error']=exp.redact(exc)
        capture['source_audit']=audit_case_sources(case,exp.chunk_rows,entries)
        capture.update(finished=now(),transport_retries=attempts)
        return capture

    results=list(prior)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(work,*task) for task in tasks]
        for future in as_completed(futures):
            row=future.result(); results.append(row)
            with (OUT/'runs.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,ensure_ascii=False,default=str).replace(exp.key,'[REDACTED]')+'\n')
            print(row['id'],row['repeat'],(row.get('judgement') or {}).get('verdict'),row.get('answer_error',''),row.get('judge_error',''),flush=True)
            write(OUT/'progress.json',{'finished':len(results),'total':99,'updated':now()})
    drift=[p for p,h in hashes.items() if sha(p)!=h]
    assert len(results)==99
    write(OUT/'completion.json',{'finished':now(),'total':len(results),'input_drift':drift,
        'answer_errors':sum('answer_error' in r for r in results),'judge_errors':sum('judge_error' in r for r in results),
        'settings_code_index_gold_unchanged':not drift,'bundle':exp.store.active().bundle_id})
    assert not drift

if __name__=='__main__': main()
