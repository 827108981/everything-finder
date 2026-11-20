"""Run the frozen scene candidate through AnswerService on fresh, paired inputs."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
import json
import threading
from xml.etree import ElementTree
from zipfile import ZipFile

from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evaluation.runner import _answer_record
from tools.answer_scene_guard import (
    AnswerService, CancelToken, ConstraintMatcher, EvidencePolicy, GoldAnswerJudge,
    IndexGenerationStore, LocalBgeEmbeddingProvider, LocalBgeReranker, LocalModelManager,
    ModelSpec, OpenAICompatibleGateway, QueryInterpreter, RAG_MODEL_CACHE_DIR, RagQuery,
    WindowsCredentialSecretStore, model_settings,
)
from tools.probe_scene_guard_method import apply_method_guard
from tools.validate_scene_guard import (
    EvidenceChunk, FIXTURES, RagIndexStore, ROOT, SCENE, SETTINGS_PATH, delta, load_chunk_rows,
    load_gold_dataset, map_case_evidence, question_signature, read, relation, select, sha,
)
from tools.validate_scene_quota import canonical_content
from tools.validate_structured_material import slides, text

OUT = ROOT / "reports/rag_scene_cross_source_20261108"
INDEX_DB = Path("E:/everythingfinder-db/search_index.db")


def dump(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def original_text(path, number):
    if path.suffix.lower() == ".pptx":
        return next(text(slide) for page, slide in slides(path) if page == number)
    with ZipFile(path) as package:
        root = ElementTree.fromstring(package.read("word/document.xml"))
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    return "\n".join("".join(n.text or "" for n in paragraph.iter(ns + "t"))
                     for paragraph in root.iter(ns + "p"))


def focus(refs):
    anchors = "、".join(f"[{sid}]" for sid in refs)
    return (
        f"\n已定位与问题的报警、阶段及项目范围一致的证据 {anchors}。"
        "用这些证据构成回答的主线，检查是否完整回答了用户所问的原因、处理及复核。"
        "这些来源若列有多项会影响排查的原因或条件，按原文区分可能原因、处理顺序和触发条件。"
        "其他来源用于补充有明确适用依据的事实。同名报警出现在不同检测方法、机型或场景时，"
        "须有资料明确支持其适用范围，才可引用该定义、数值或操作。"
    )


def main():
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    settings = model_settings()
    secrets = WindowsCredentialSecretStore()
    key = secrets.retrieve(settings.rag_api_secret_ref)
    assert key
    store = RagIndexStore(INDEX_DB.parent / "rag")
    bundle = store.active()
    catalogue = read(SCENE / "scene_index.json")
    with store.connect_read_only(bundle.metadata_path) as conn:
        rows = load_chunk_rows(conn)
        by_id = {row["chunk_id"]: row for row in rows}
        catalogue_files = sorted({by_id[cid]["file_path"] for item in catalogue for cid in item["ids"]})

    def dataset(path):
        return load_gold_dataset(path, reviewed_only=False).cases

    holdout = OUT / "holdout.json"
    tasks = [("cross_source_pending", c) for c in dataset(holdout)]
    tasks += [("formal48", c) for c in dataset(FIXTURES[0]) if c.review_status == "reviewed"]
    tasks += [("regression30_pending", c) for p in FIXTURES[1:] for c in dataset(p)]
    assert len(tasks) == 88 and len({c.case_id for _, c in tasks}) == 88
    maps = {case.case_id: map_case_evidence(case, rows) for _, case in tasks}
    sources = set()
    verified = []
    for case in dataset(holdout):
        assert case.review_status == "pending"
        for index, item in enumerate(case.raw["gold_evidence"]):
            path = Path(item["file"])
            assert str(path) not in catalogue_files
            quote = canonical_content(item["quote"])
            assert quote in canonical_content(original_text(path, item.get("page"))), (case.case_id, index)
            match = maps[case.case_id][index]
            confirmed = any(a.confirmed for a in match.alternatives)
            assert confirmed, (case.case_id, match)
            sources.add(path)
            verified.append({"id": case.case_id, **item, "original_verified": True, "index_verified": confirmed})
    assert all(case.corpus_snapshot == bundle.bundle_id for _, case in tasks)
    inputs = [Path(__file__), holdout, SETTINGS_PATH, store.manifest_path, bundle.metadata_path,
              SCENE / "scene_index.json", SCENE / "050_approved_alternatives.json", *FIXTURES, *sorted(sources),
              *(ROOT / "local_full_text_search/rag").rglob("*.py"),
              *(Path(__file__).with_name(name) for name in ("validate_scene_guard.py", "validate_scene_quota.py",
                 "answer_scene_guard.py", "probe_scene_guard_method.py"))]
    frozen = {str(p): sha(p) for p in inputs}
    dump("protocol.json", {
        "bundle": bundle.bundle_id, "model": settings.rag_api_model, "catalogue_files": catalogue_files,
        "catalogue_pairs": len(catalogue), "hashes": frozen,
        "context_limits": [settings.rag_max_chunks_per_document, settings.rag_final_top_k,
                           settings.rag_context_max_chars],
        "semantic_clarification": settings.rag_semantic_clarification_enabled,
        "answer_review": settings.rag_answer_review_enabled,
        "gate": "Cross-source net gain, source-bound actual answers, no new confirmed critical regression.",
        "comparisons": "Fresh native AnswerService for A. B uses the frozen selector, method guard and"
                       " generation focus inside the same service. Identical requests share their result."
                       " B reruns native retrieval for altered contexts and verifies it matches A.",
        "pending": "10 agent-authored groups, original and index quote checked before any retrieval."
                   " The 30 older pending regressions remain separate from the 48 formal cases.",
        "cases": [{"cohort": cohort, "id": case.case_id} for cohort, case in tasks],
    })
    dump("source_verification.json", verified)
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
    policy = EvidencePolicy(min_vector_score=settings.rag_min_vector_score,
                            min_rerank_score=settings.rag_min_rerank_score)
    lock = threading.Lock()

    def work(cohort, case):
        variant = {}
        diagnostics = {}
        matches = []
        signature = question_signature(case.question)
        if signature:
            matches = [s for s in catalogue if relation(signature, s["text"])[0] == "match"]
        audit = {"signature": signature, "catalogue_matches": [s["ids"] for s in matches]}

        class Builder(ContextBuilder):
            def __init__(self, arm):
                self.arm = arm

            def build(self, evidence, *, max_chars):
                native = super().build(evidence, max_chars=max_chars)
                if self.arm == "B":
                    assert native == variant["A"], "native retrieval differs between arms"
                    return variant["B"]
                variant.update(A=native, B=native)
                if not matches or not native.entries:
                    return native
                with store.connect_read_only(bundle.metadata_path) as conn:
                    def load(cid):
                        return store.load_chunk(conn, cid)

                    items = [EvidenceChunk(replace(load(s["ids"][0]), stable_chunk_id=s["id"],
                                                  text=s["text"]), lexical_rank=1) for s in matches]
                    ranker.rerank(case.question, items, len(items), CancelToken())
                    ranked = ConstraintMatcher().evaluate(QueryInterpreter().interpret(case.question),
                                                          ranker.last_scored_candidates)
                    ids = {s["id"]: s["ids"] for s in matches}
                    pairs = [{"ids": ids[item.chunk.stable_chunk_id], "ce": item.rerank_score,
                              "reason": policy.rejection_reason(item) or "available"} for item in ranked]
                    ctx, entries, selection = select(case.question, native.text, native.entries, pairs,
                                                     load, settings, allow_companion=True)
                    audit["selection"] = selection
                    if not selection["selected"]:
                        return native
                    anchors = set(selection["selected"]["pair"])
                    ctx, entries, removed = apply_method_guard(case.question, ctx, entries, anchors)
                    audit["method_removed"] = removed
                    audit["focus"] = tuple(e["source_id"] for e in entries if e["chunk_id"] in anchors)
                    prior = {e["chunk_id"]: native.source_map[e["source_id"]] for e in native.entries}
                    mapping = {e["source_id"]: prior.get(e["chunk_id"]) or (load(e["chunk_id"]).source,)
                               for e in entries}
                    variant["B"] = replace(native, text=ctx, entries=tuple(entries), source_map=mapping,
                                           sources=tuple(ref for refs in mapping.values() for ref in refs))
                return native

        def answer(arm):
            token = CancelToken()
            lock.acquire()
            held = True
            events = []

            def progress(payload):
                nonlocal held
                stage = payload.get("stage")
                if stage in ("credential_read", "clarification_assessed", "draft_generated"):
                    events.append({k: payload[k] for k in ("stage", "citation_check", "needs_clarification")
                                   if k in payload})
                if held and stage in ("generating", "assessing_clarification"):
                    held = False
                    lock.release()

            class Gateway(OpenAICompatibleGateway):
                def generate(self, request, tok):
                    if arm == "B" and audit.get("focus"):
                        request = replace(request, query_plan_summary=request.query_plan_summary + focus(audit["focus"]))
                    return super().generate(request, tok)

            gateway = Gateway(max_transient_retries=3)
            try:
                service = AnswerService(IndexGenerationStore(INDEX_DB), store, embedding, ranker,
                                        gateway, secrets, settings, progress_callback=progress)
                service.context_builder = Builder(arm)
                result = _answer_record(service.answer(RagQuery(
                    text=case.question, include_ocr=True, min_ocr_confidence=settings.ocr_min_confidence,
                    lexical_top_k=settings.rag_lexical_top_k, vector_top_k=settings.rag_vector_top_k,
                    fusion_top_k=settings.rag_fusion_top_k, reranker_top_k=settings.rag_reranker_top_k,
                    final_top_k=settings.rag_final_top_k), token))
            except Exception as exc:
                return {"error": str(exc).replace(key, "[REDACTED]")[:500], "events": events}
            finally:
                if held:
                    lock.release()
            context = variant[arm].text if arm in variant else ""
            try:
                judge = GoldAnswerJudge(gateway, settings, key).evaluate(
                    case, {**result, "evidence_context": context}, token)
            except Exception as exc:
                judge = {"verdict": "unavailable", "error": str(exc).replace(key, "[REDACTED]")[:500]}
            return {"answer": result, "judgement": judge, "events": events}

        diagnostics["A"] = answer("A")
        first, second = variant.get("A"), variant.get("B")
        changed = first != second or bool(audit.get("focus"))
        diagnostics["B"] = answer("B") if changed else {"reused_from": "A"}
        entries_a = list(first.entries) if first else []
        entries_b = list(second.entries) if second else []
        evidence = delta(entries_a, entries_b, maps[case.case_id])
        for arm, entries in (("A", entries_a), ("B", entries_b)):
            actual = diagnostics["A" if not changed else arm]
            labels = {src["source_id"] for src in actual.get("answer", {}).get("sources", [])}
            diagnostics[arm]["cited_evidence"] = delta(
                [], [e for e in entries if e["source_id"] in labels], maps[case.case_id])["after_gold"]
        return {"id": case.case_id, "cohort": cohort, "question": case.question, "changed": changed,
                "audit": audit, "context_A": first.text if first else "", "context_B": second.text if second else "",
                "entries_A": entries_a, "entries_B": entries_b, "evidence": evidence, **diagnostics}

    with (OUT / "cases.jsonl").open("x", encoding="utf8") as file, ThreadPoolExecutor(max_workers=5) as pool:
        for cohort in ("cross_source_pending", "formal48", "regression30_pending"):
            futures = [pool.submit(work, group, case) for group, case in tasks if group == cohort]
            for future in as_completed(futures):
                row = future.result()
                file.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
                file.flush()
                print(row["id"], row["changed"], row["A"].get("judgement", {}).get("verdict"),
                      row["B"].get("judgement", {}).get("verdict"),
                      row["A"].get("error") or row["B"].get("error") or "", flush=True)
    assert all(sha(p) == h for p, h in frozen.items())
    dump("completion.json", {"cases": len(tasks), "frozen_inputs_verified": True})


if __name__ == "__main__":
    main()
