"""Frozen native-entry QueryPlan comparison; labels are only used after retrieval."""
from pathlib import Path
from dataclasses import asdict
import argparse
import json

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from tools.validate_evidence_answer_flow import Experiment
from tools.rag_eval_common import ROOT, write, sha
from tools.recheck_evidence_flow_native import public_settings, now

OUT = ROOT / 'reports/rag_query_plan_adapter_20261111'
FILES = [ROOT/'tests/rag/fixtures'/f for f in (
    'rag_gold_v4_3.jsonl', 'rag_acceptance_v1.jsonl',
    'rag_row_holdout_v1.jsonl', 'rag_row_holdout_v2.jsonl')]
FILES += [ROOT/'reports/rag_evidence_answer_native_recheck_20261109/acceptance_review.json']
STAGES = {'lexical_candidates':'lexical', 'vector_candidates':'vector',
          'fused_candidates':'fusion', 'deduplicated_candidates':'deduplicated',
          'reranked_candidates':'rerank', 'constraint_candidates':'constraints',
          'evidence_candidates':'evidence'}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--ids', default='')
    ap.add_argument('--answers', action='store_true')
    ap.add_argument('--disabled', action='store_true')
    ap.add_argument('--resume', action='store_true')
    args=ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path=OUT/(args.arm+'.jsonl')
    assert args.resume or not path.exists(), path
    exp=Experiment(OUT, 1)
    if hasattr(exp.settings, 'rag_query_plan_enabled'):
        exp.settings.rag_query_plan_enabled=not args.disabled
    pairs=[]
    for n,p in enumerate(FILES):
        pairs.extend((('formal' if c.review_status=='reviewed' else 'diagnostic') if n==0 else p.stem,c)
                     for c in load_gold_dataset(p,reviewed_only=False).cases)
    if args.ids: pairs=[(g,c) for g,c in pairs if c.case_id in args.ids.split(',')]
    manifest={'started':now(), 'bundle':exp.bundle.bundle_id, 'settings':public_settings(exp.settings),
              'dataset_hashes':{str(p):sha(p) for p in FILES},
              'code_hashes':{str(p):sha(p) for p in (ROOT/'local_full_text_search/rag').rglob('*.py')},
              'cases':[(g,c.case_id) for g,c in pairs], 'mode':'native_answer' if args.answers else 'native_retrieve',
              'protocol':f'{len(pairs)} selected cases from the listed frozen dataset hashes; disputed/pending diagnostic only. Fixed labels. No quality retry. Exact source/quote scoring. Inspect all losses. Different dataset versions must not be pooled.'}
    original_manifest=OUT/(args.arm+'_manifest.json')
    if args.resume:
        previous=json.loads(original_manifest.read_text(encoding='utf-8'))
        assert previous['code_hashes']==manifest['code_hashes'], 'runtime code drift'
        assert previous['dataset_hashes']==manifest['dataset_hashes'], 'Gold drift'
        prior=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        done={r['id'] for r in prior}
        assert len(done)==len(prior)
        pairs=[(g,c) for g,c in pairs if c.case_id not in done]
        write(OUT/(args.arm+'_resume.json'),{'preserved':len(done),'remaining':len(pairs),'reason':'Local runner interrupted; resume only unfinished questions, no quality retries'})
    else:
        write(original_manifest,manifest)
    for group,case in pairs:
        events=[]; stages={}; token=CancelToken()
        def progress(e):
            for field,stage in STAGES.items():
                if field in e: stages[stage]=e[field]
            events.append({k:v for k,v in e.items() if k not in STAGES})
        gateway=OpenAICompatibleGateway(max_transient_retries=2)
        s=exp.settings
        service=AnswerService(exp.source_store,exp.store,exp.embedding,exp.ranker,gateway,exp.secrets,s,progress_callback=progress)
        query=RagQuery(case.question,include_ocr=True,min_ocr_confidence=s.ocr_min_confidence,
                       lexical_top_k=s.rag_lexical_top_k,vector_top_k=s.rag_vector_top_k,
                       fusion_top_k=s.rag_fusion_top_k,reranker_top_k=s.rag_reranker_top_k,final_top_k=s.rag_final_top_k)
        row={'id':case.case_id,'group_id':case.group_id,'question':case.question,'cohort':group,'review_status':case.review_status}
        try:
            result=service.answer(query,token) if args.answers else service.retrieve(query,token)
            row['result']=_answer_record(result) if args.answers else asdict(result)
        except Exception as exc:
            row['error']=str(exc).replace(exp.key,'[REDACTED]')
        ctx=next((e for e in reversed(events) if 'llm_context' in e),{})
        row.update(context=ctx.get('llm_context',''),entries=ctx.get('context_entries',[]),events=events,candidate_stages=stages)
        mappings=map_case_evidence(case,exp.chunk_rows)
        row['scores']={stage:score_stage(stage,items,mappings) for stage,items in {**stages,'context':row['entries']}.items()}
        if args.answers and 'result' in row:
            try: row['judgement']=GoldAnswerJudge(gateway,s,exp.key).evaluate(case,{**row['result'],'evidence_context':row['context']},token)
            except Exception as exc: row['judge_error']=str(exc).replace(exp.key,'[REDACTED]')
        with path.open('a',encoding='utf-8') as f: f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
        print(args.arm,case.case_id,row['scores']['context']['gold_evidence_found'],row.get('error',''),(row.get('judgement') or {}).get('verdict',''),flush=True)
    write(OUT/(args.arm+'_complete.json'),{'finished':now(),'count':len(pairs),
        'dataset_drift':[p for p,h in manifest['dataset_hashes'].items() if sha(p)!=h],
        'bundle':exp.store.active().bundle_id})

if __name__=='__main__': main()
