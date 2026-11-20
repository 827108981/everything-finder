"""Run corrected human questions locally; never use old-query traces as new runs."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['ANONYMIZED_TELEMETRY']='False'
import json
from dataclasses import asdict
from pathlib import Path
from tools.validate_selection_factorial import ROOT, load_inputs, no_network, write
from tools.validate_bge_base_adoption import Forbidden
from tools.diagnose_vector_recall import ro
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.stage_scoring import score_stage

OUT=ROOT/'reports/rag_gold_material_repair_20261116'


def main():
    import torch
    torch.set_num_threads(3)
    from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
    from local_full_text_search.config.defaults import AppSettings
    from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
    from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
    from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
    from local_full_text_search.rag.index_store import RagIndexStore
    from local_full_text_search.rag.answer_service import AnswerService
    from local_full_text_search.rag.models import RagQuery
    from local_full_text_search.core.index_generation_store import IndexGenerationStore
    from local_full_text_search.core.task_manager import CancelToken
    from tools.validate_query_plan_adapter import STAGES
    _,s,_,_=load_inputs()
    settings=AppSettings.from_dict(s)
    assert not settings.rag_query_plan_enabled
    manifest=json.loads((ROOT/'tests/rag/fixtures/rag_active_suite_v5_1.json').read_text(encoding='utf8'))
    cases={c.case_id:c for d in manifest['datasets'] for c in load_gold_dataset(ROOT/d['path'],reviewed_only=False).cases}
    wanted=['rag-biz-028','evidence-flow-independent-05','evidence-flow-independent-06']
    store=RagIndexStore('E:/everythingfinder-db/rag');bundle=store.active()
    assert bundle.bundle_id==manifest['bundle_id']
    with store.connect_read_only(bundle.metadata_path) as conn: chunks=load_chunk_rows(conn)
    manager=LocalModelManager(RAG_MODEL_CACHE_DIR)
    records=[]
    with no_network() as attempts:
        embed=LocalBgeEmbeddingProvider(manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,'embedding')),
            repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,batch_size=settings.rag_embedding_batch_size)
        ranker=LocalBgeReranker(manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,'reranker')),
            repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,batch_size=settings.rag_reranker_batch_size,max_length=settings.rag_reranker_max_length)
        for cid in wanted:
            case=cases[cid];events=[];stages={}
            def progress(e):
                events.append(e)
                for field,stage in STAGES.items():
                    if field in e:stages[stage]=e[field]
            service=AnswerService(IndexGenerationStore(Path('E:/everythingfinder-db/search_index.db')),store,
                embed,ranker,Forbidden(),Forbidden(),settings,progress_callback=progress)
            query=RagQuery(case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                lexical_top_k=settings.rag_lexical_top_k,vector_top_k=settings.rag_vector_top_k,
                fusion_top_k=settings.rag_fusion_top_k,reranker_top_k=settings.rag_reranker_top_k,final_top_k=settings.rag_final_top_k)
            result=service.retrieve(query,CancelToken())
            ctx=next(e for e in reversed(events) if 'llm_context' in e)
            mappings=map_case_evidence(case,chunks)
            scores={k:score_stage(k,v,mappings) for k,v in stages.items()}
            scores['context']=score_stage('context',ctx['context_entries'],mappings)
            records.append(dict(id=cid,question=case.question,review_status=case.review_status,query=asdict(query),
                result=asdict(result),events=events,stages=stages,scores=scores,
                context=ctx['llm_context'],entries=ctx['context_entries'],api_calls=0,
                adapter_enabled=False,reranking='BGE base'))
            (OUT/'fresh_retrieval.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False,default=str)+'\n' for r in records),encoding='utf8')
            print(cid,{k:(v['first_gold_rank'],v['gold_evidence_found'],v['gold_evidence_total_declared']) for k,v in scores.items()},flush=True)
    assert not attempts and len(records)==3
    write(OUT/'fresh_retrieval_audit.json',dict(api_calls=0,network_attempts=attempts,cases=3,
        bundle_id=bundle.bundle_id,adapter_enabled=False,real_local_models=True,
        interpretation='new question + corrected Gold; question/annotation revision, not an algorithm comparison'))


if __name__=='__main__':main()
