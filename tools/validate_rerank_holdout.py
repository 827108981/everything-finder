"""Frozen A/B reranking validation on development and pending holdout questions.

Uses production retrieval, explicitly diagnostic scoring, and never changes Gold
review status or the application's defaults. Gold is read only after retrieval.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH, RAG_MODEL_CACHE_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.embeddings.local_bge import LocalBgeEmbeddingProvider
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import score_case
from local_full_text_search.rag.evaluation.stage_scoring import score_stage
from local_full_text_search.rag.evaluation.runner import _retrieval_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.reranking.support import retrieval_prior
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore


def sha(value):
    return hashlib.sha256(value).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    fixtures = Path("tests/rag/fixtures")
    datasets = {"development": load_gold_dataset(fixtures/"rag_gold_v3.jsonl"),
                "holdout_pending": load_gold_dataset(fixtures/"rag_acceptance_v1.jsonl", reviewed_only=False)}
    assert len(datasets["development"].cases) == 50 and len(datasets["holdout_pending"].cases) == 10
    assert all(c.review_status == "pending" for c in datasets["holdout_pending"].cases)
    settings = AppSettings.from_dict(json.loads(SETTINGS_PATH.read_text(encoding="utf-8")))
    config = json.loads(Path("reports/rerank_pipeline_20261101/current.json").read_text(encoding="utf-8"))
    for k,v in config.items():
        if k.startswith("rag_"):
            setattr(settings,k,v)
    settings.rag_constraint_matching_enabled = True
    settings.rag_clarification_enabled = True
    settings.rag_max_chunks_per_document = 3
    settings.rag_min_vector_score = .35
    settings.rag_min_rerank_score = .10
    settings.rag_dedup_jaccard_threshold = .92
    index_db = Path(config["generation_base_path"])
    store = RagIndexStore(index_db.parent/"rag")
    bundle = store.active()
    assert bundle.bundle_id == config["expected_bundle_id"]
    assert all(c.corpus_snapshot == bundle.bundle_id for d in datasets.values() for c in d.cases)
    arms = {"current": (.65,.05,.30,3), "pure": (1.,0.,0.,0)}
    frozen = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "bundle_id": bundle.bundle_id,
        "datasets": {k:{"path":str(d.path),"sha256":d.sha256,"n":len(d.cases),
                        "statuses":sorted({c.review_status for c in d.cases})} for k,d in datasets.items()},
        "arms": arms,
        "fixed_settings": {k:v for k,v in dataclasses.asdict(settings).items() if k.startswith("rag_") and "api" not in k},
        "evaluation": "retrieval-only diagnostic; pending cases not admitted to official evaluation",
        "decision_rules": [
            "Rank-only top8 hit, complete evidence recall, MRR and per-case paired changes are primary.",
            "Report final context as a secondary outcome under unchanged downstream policies.",
            "A general improvement requires held-out mean coverage/MRR not decreasing, no paired coverage regressions, and review of newly selected specified negatives.",
            "Any pending Gold prevents formal acceptance; ten answer-only cases do not validate refusal or all unseen domains.",
            "Do not adjust weights, queries, thresholds, Gold or pipeline after viewing holdout results.",
        ],
        "script_sha256":sha(Path(__file__).read_bytes()),
    }
    dump(args.out/"protocol.json", frozen)
    # Cache contains only raw model scores with exact question/body identity.
    olddir = Path("C:/Users/50263194/AppData/Local/LocalFullTextSearch/rag_eval_runs/20261030_143500_8c27af")
    old = [json.loads(s) for s in (olddir/"cases.jsonl").read_text(encoding="utf-8").splitlines()]
    verified = json.loads(Path("reports/rerank_ablation_20261101/live/cases.json").read_text(encoding="utf-8"))
    vp = json.loads(Path("reports/rerank_ablation_20261101/live/protocol.json").read_text(encoding="utf-8"))
    assert vp["model"] == settings.rag_reranker_repo_id and vp["revision"] == settings.rag_reranker_revision
    assert settings.rag_reranker_max_length == 512
    cache = {}
    for a,b in zip(old,verified,strict=True):
        assert a["id"] == b["id"] and b.get("max_score_difference") == 0 and b.get("live_order_matches")
        texts = {x["chunk_id"]:x["text"] for x in a["candidate_stages"]["rerank"]}
        for x in b["candidates"]:
            cache[(a["question"],x["chunk_id"],sha(texts[x["chunk_id"]].encode()))] = x["live_score"]
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    ep = manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id,settings.rag_embedding_revision,"embedding"))
    rp = manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id,settings.rag_reranker_revision,"reranker"))
    embedding = LocalBgeEmbeddingProvider(ep,repo_id=settings.rag_embedding_repo_id,revision=settings.rag_embedding_revision,
                                          batch_size=settings.rag_embedding_batch_size)
    raw = LocalBgeReranker(rp,repo_id=settings.rag_reranker_repo_id,revision=settings.rag_reranker_revision,
                          batch_size=settings.rag_reranker_batch_size,max_length=512)
    token = CancelToken()
    counters = {"fresh_pairs":0,"cached_pairs":0}

    class ScoredReranker:
        last_scored_candidates = ()
        def __init__(self,weights):
            self.weights=weights
        def rerank(self,question,candidates,top_n,token):
            key=lambda x:(question,x.chunk.stable_chunk_id,sha(x.chunk.text.encode()))
            missing=[x for x in candidates if key(x) not in cache]
            if missing:
                raw.rerank(question,missing,len(missing),token)
                for x in raw.last_scored_candidates:cache[key(x)]=x.rerank_score
            counters["fresh_pairs"]+=len(missing)
            counters["cached_pairs"]+=len(candidates)-len(missing)
            ce,prior,support,guard=self.weights
            for x in candidates:
                x.rerank_score=cache[key(x)]
                x.rerank_score_combined=ce*x.rerank_score+prior*retrieval_prior(x)+support*float(x.answer_support_score or 0)
            ranked=sorted(candidates,key=lambda x:(-x.rerank_score_combined,-x.fused_score,x.chunk.stable_chunk_id))
            raw.max_rank_drop=guard
            ranked=raw._apply_rank_guard(ranked)
            self.last_scored_candidates=tuple(ranked)
            return raw._apply_document_diversity(ranked,top_n)

    result=[]
    for cohort,dataset in datasets.items():
        for arm,weights in arms.items():
            for case in dataset.cases:
                events=[];stages={};start=time.monotonic();before=dict(counters)
                fields={"lexical_candidates":"lexical","vector_candidates":"vector","fused_candidates":"fusion",
                        "deduplicated_candidates":"deduplicated","reranked_candidates":"rerank",
                        "constraint_candidates":"constraints","evidence_candidates":"evidence"}
                def record(event):
                    events.append(event)
                    for f,s in fields.items():
                        if isinstance(event.get(f),list):stages[s]=event[f]
                    if event.get("context_entries"):
                        selected={x["chunk_id"]:x for x in stages.get("evidence",[]) if x.get("selected") is True}
                        stages["context"]=[{**selected[e["chunk_id"]],"included_text":e.get("included_text"),
                                            "truncated":e.get("truncated",False)} for e in event["context_entries"]]
                service=AnswerService(IndexGenerationStore(index_db),store,embedding,ScoredReranker(weights),
                    OpenAICompatibleGateway(max_transient_retries=0),WindowsCredentialSecretStore(),settings,progress_callback=record)
                retrieval=service.retrieve(RagQuery(text=case.question,include_ocr=True,min_ocr_confidence=settings.ocr_min_confidence,
                    lexical_top_k=50,vector_top_k=50,fusion_top_k=30,reranker_top_k=20,final_top_k=8),token)
                # Evidence labels only used after all retrieval/selection has completed.
                conn=store.connect_read_only(bundle.metadata_path)
                try:chunk_rows=load_chunk_rows(conn)
                finally:conn.close()
                mappings=map_case_evidence(case,chunk_rows)
                scored=score_case(case,stages,mappings,answer=_retrieval_record(retrieval))
                row={**scored,"cohort":cohort,"arm":arm,"review_status":case.review_status,
                    "formal_eligible":case.review_status=="reviewed","candidate_stages":stages,
                    "rerank_topk":{str(k):score_stage("rerank",stages.get("rerank",[])[:k],mappings) for k in (1,3,5,8,20)},
                    "duration_ms":round((time.monotonic()-start)*1000),
                    "raw_score_cache":{k:counters[k]-before[k] for k in counters},"api_calls":0,
                    "events":events}
                result.append(row)
                with (args.out/"cases.jsonl").open("a",encoding="utf-8") as handle:handle.write(json.dumps(row,ensure_ascii=False)+"\n")
                print(cohort,arm,case.case_id,'done',row['raw_score_cache'],flush=True)
    assert all(sha(d.path.read_bytes())==d.sha256 for d in datasets.values())
    assert store.active().bundle_id==bundle.bundle_id
    dump(args.out/"completion.json",{"n":len(result),"status":"completed",**counters,
        "gold_files_unchanged":True,"api_calls":0,"acceptance_formally_reviewed":0})


if __name__=="__main__":main()
