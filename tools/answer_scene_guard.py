"""Paired generation for changed contexts and fresh native retrieval for new probes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from tools.validate_row_holdout import (
    LocalBgeEmbeddingProvider, LocalBgeReranker, LocalModelManager, ModelSpec, RAG_MODEL_CACHE_DIR,
)
from tools.validate_scene_guard import (
    AppSettings, EvidenceChunk, FIXTURES, OUT, RagIndexStore, SCENE, SETTINGS_PATH,
    delta, dump, load_chunk_rows, load_gold_dataset, map_case_evidence, question_signature,
    read, relation, select, sha,
)


def model_settings():
    return replace(AppSettings.from_dict(read(SETTINGS_PATH)), rag_api_timeout_seconds=180)


def make_jobs(cases, contexts, repeats):
    jobs = []
    for row in contexts:
        case = cases[row["id"]]
        for repeat in range(1, repeats + 1):
            for arm in ("A", "B") if row["changed"] else ("AB",):
                source_arm = arm[0]
                jobs.append((case, arm, repeat, row[f"context_{source_arm}"], row[f"entries_{source_arm}"]))
    return jobs


def run_answer(job, settings, key, *, scene_sources=()):
    case, arm, repeat, context, entries = job
    row = {"id": case.case_id, "arm": arm, "repeat": repeat}
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    try:
        if not context:
            row["status"] = "no_context"
            row["text"] = ""
            return row
        plan = AnswerService._plan_summary(QueryInterpreter().interpret(case.question))
        if scene_sources:
            anchors = "、".join(f"[{sid}]" for sid in scene_sources)
            plan += (
                f"\n已定位与问题的报警、阶段及项目范围一致的证据 {anchors}。"
                "用这些证据构成回答的主线，检查是否完整回答了用户所问的原因、处理及复核。"
                "这些来源若列有多项会影响排查的原因或条件，按原文区分可能原因、处理顺序和触发条件。"
                "其他来源用于补充有明确适用依据的事实。同名报警出现在不同检测方法、机型或场景时，"
                "须有资料明确支持其适用范围，才可引用该定义、数值或操作。"
            )
        result = gateway.generate(GenerationRequest(
            question=case.question, context=context,
            query_plan_summary=plan,
            model=settings.rag_api_model, api_base_url=settings.rag_api_base_url,
            api_style=settings.rag_api_style, api_key=key, timeout_seconds=settings.rag_api_timeout_seconds,
        ), CancelToken())
        row["text"] = result.text
        row["citation"] = CitationValidator().inspect(
            result.text, {e["source_id"]: () for e in entries}
        ).structurally_valid
        row["status"] = "answered" if row["citation"] else "citation_invalid"
        row["judgement"] = GoldAnswerJudge(gateway, settings, key).evaluate(
            case, {"text": result.text, "status": row["status"], "evidence_context": context}, CancelToken()
        )
    except Exception as exc:
        row["error"] = str(exc).replace(key, "[REDACTED]")[:500]
    return row


def run_jobs(pool, jobs, name, settings, key):
    destination = OUT / name
    assert not destination.exists()
    futures = [pool.submit(run_answer, job, settings, key) for job in jobs]
    return destination, futures


def collect(destination, futures, key):
    with destination.open("x", encoding="utf8") as file:
        for future in as_completed(futures):
            row = future.result()
            print(row["id"], row["arm"], row["repeat"],
                  row.get("judgement", {}).get("verdict"), row.get("error"), flush=True)
            file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            file.flush()


def fresh_holdout(settings, store, bundle, *, input_path=None, output_path=None, allow_companion=False):
    input_path = input_path or OUT / "holdout.json"
    output_path = output_path or OUT
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
    embedding = LocalBgeEmbeddingProvider(
        manager.resolve_local_path(ModelSpec(settings.rag_embedding_repo_id, settings.rag_embedding_revision, "embedding")),
        repo_id=settings.rag_embedding_repo_id, revision=settings.rag_embedding_revision,
        batch_size=settings.rag_embedding_batch_size,
    )
    ranker = LocalBgeReranker(
        manager.resolve_local_path(ModelSpec(settings.rag_reranker_repo_id, settings.rag_reranker_revision, "reranker")),
        repo_id=settings.rag_reranker_repo_id, revision=settings.rag_reranker_revision,
        batch_size=settings.rag_reranker_batch_size, max_length=settings.rag_reranker_max_length,
        cross_encoder_weight=settings.rag_rerank_cross_encoder_weight,
        retrieval_prior_weight=settings.rag_rerank_retrieval_prior_weight,
        answer_support_weight=settings.rag_rerank_answer_support_weight,
        max_rank_drop=settings.rag_rerank_max_rank_drop,
    )
    catalogue = read(SCENE / "scene_index.json")
    policy = EvidencePolicy(min_vector_score=settings.rag_min_vector_score,
                            min_rerank_score=settings.rag_min_rerank_score)
    rows, dataset = [], []
    with store.connect_read_only(bundle.metadata_path) as conn:
        index = load_chunk_rows(conn)
        path = store.load_chunk(conn, catalogue[0]["ids"][0]).source.file_path
        for h in read(input_path):
            quotes = [(h["page"], h["quote"])]
            if "second_page" in h:
                quotes.append((h["second_page"], h["second_quote"]))
            dataset.append({
                "id": h["id"], "question": h["question"], "corpus_snapshot": bundle.bundle_id,
                "review_status": "pending", "expected_behavior": "answer",
                "required_facts": h["core"],
                "gold_evidence": [{"file": h.get("file", path), "page": p, "quote": q, "relevance": 2}
                                  for p, q in quotes],
            })
        gold_path = output_path / "holdout_gold.json"
        gold_path.write_text(json.dumps(dataset, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        cases = {c.case_id: c for c in load_gold_dataset(gold_path, reviewed_only=False).cases}
        for case in cases.values():
            mappings = map_case_evidence(case, index)
            assert all(any(a.confirmed for a in m.alternatives) for m in mappings), case.case_id
            events = []
            service = AnswerService(
                IndexGenerationStore(Path("E:/everythingfinder-db/search_index.db")), store,
                embedding, ranker, OpenAICompatibleGateway(), WindowsCredentialSecretStore(), settings,
                progress_callback=events.append,
            )
            query = RagQuery(text=case.question, lexical_top_k=settings.rag_lexical_top_k,
                             vector_top_k=settings.rag_vector_top_k, fusion_top_k=settings.rag_fusion_top_k,
                             reranker_top_k=settings.rag_reranker_top_k, final_top_k=settings.rag_final_top_k,
                             min_ocr_confidence=settings.ocr_min_confidence)
            result = service.retrieve(query, CancelToken())
            event = next((e for e in reversed(events) if "llm_context" in e), {})
            context, entries = event.get("llm_context", ""), event.get("context_entries", [])
            signature = question_signature(case.question)
            plan = QueryInterpreter().interpret(case.question)
            matches = [s for s in catalogue if signature and relation(signature, s["text"])[0] == "match"]
            items = []
            for scene in matches:
                parent = store.load_chunk(conn, scene["ids"][0])
                chunk = replace(parent, stable_chunk_id=scene["id"], text=scene["text"])
                items.append(EvidenceChunk(chunk, lexical_rank=1))
            if items:
                ranker.rerank(case.question, items, len(items), CancelToken())
            ranked = ConstraintMatcher().evaluate(plan, ranker.last_scored_candidates) if items else []
            by_id = {s["id"]: s for s in matches}
            pairs = [
                {"ids": by_id[i.chunk.stable_chunk_id]["ids"], "ce": i.rerank_score,
                 "reason": policy.rejection_reason(i) or "available"}
                for i in ranked
            ]
            after, after_entries, audit = select(
                case.question, context, entries, pairs, lambda cid: store.load_chunk(conn, cid), settings,
                allow_companion=allow_companion,
            )
            rows.append({
                "id": case.case_id, "status": str(result.status), "changed": after != context,
                "context_A": context, "context_B": after, "entries_A": entries, "entries_B": after_entries,
                "audit": audit, **delta(entries, after_entries, mappings),
            })
            print("retrieved", case.case_id, "changed", after != context, flush=True)
    (output_path / "holdout_selection.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    return cases, rows


def main():
    assert read(OUT / "evidence_summary.json")["evidence_gate"]
    hashes = read(OUT / "protocol.json")["hashes"]
    assert all(sha(p) == h for p, h in hashes.items())
    settings = model_settings()
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref) or ""
    assert key
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    assert bundle.bundle_id == read(OUT / "protocol.json")["bundle"]
    cases = {c.case_id: c for c in load_gold_dataset(FIXTURES[0], reviewed_only=False).cases}
    changed = [r for r in read(OUT / "selection.json") if r["changed"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        formal = run_jobs(pool, make_jobs(cases, changed, 3), "formal_answers.jsonl", settings, key)
        # Persist formal answers before the independent local retrieval step.
        collect(*formal, key)
        holdout, contexts = fresh_holdout(settings, store, bundle)
        assert all(not r["lost"] and not r["new_negatives"] for r in contexts)
        jobs = make_jobs(holdout, contexts, 1)
        collect(*run_jobs(pool, jobs, "holdout_answers.jsonl", settings, key), key)
    assert all(sha(p) == h for p, h in hashes.items())


if __name__ == "__main__":
    main()
