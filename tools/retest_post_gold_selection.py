"""Paired native AnswerService pilot for the frozen selection candidate."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.models import RagQuery
from tools.rag_eval_common import ROOT, read, sha, write
from tools.source_bound_selection import CoverageEvidencePolicy, SELECTOR_PROMPT, REMOVAL_PROMPT
from tools.validate_evidence_answer_flow import Experiment
from tools.recheck_evidence_flow_native import now, public_settings


OUT = ROOT / "reports/rag_post_gold_selection_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_3_review.json"
PILOT = ("rag-biz-006", "rag-biz-023", "rag-biz-015", "rag-biz-022", "rag-biz-028")
STAGES = {"lexical_candidates": "lexical", "vector_candidates": "vector",
          "fused_candidates": "fusion", "deduplicated_candidates": "deduplicated",
          "reranked_candidates": "rerank", "constraint_candidates": "constraints",
          "evidence_candidates": "evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol = read(OUT / "candidate_protocol_v2.json")
    # Native injection was added after the frozen static replay; selector logic and prompts are unchanged.
    assert protocol["candidate_prompt_sha256"] == __import__("hashlib").sha256(SELECTOR_PROMPT.encode()).hexdigest()
    assert protocol["removal_prompt_sha256"] == __import__("hashlib").sha256(REMOVAL_PROMPT.encode()).hexdigest()
    assert protocol["settings_sha256"] == sha(SETTINGS_PATH)
    exp = Experiment(OUT, args.workers)
    assert exp.bundle.bundle_id == protocol["bundle_id"]
    assert sha(GOLD) == protocol["gold_sha256"]
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    path = OUT / "pilot_runs.jsonl"
    if path.exists() and not args.resume:
        raise FileExistsError("Use --resume after interruption")
    prior = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []
    completed = {(row["id"], row["arm"], row["repeat"])
                 for row in prior if row.get("answer") and row.get("judgement")}
    failed_attempts = Counter((row["id"], row["arm"], row["repeat"]) for row in prior)
    tasks = [(cases[case_id], arm, repeat)
             for case_id in PILOT for repeat in (1, 2, 3) for arm in ("A", "B")
             if (case_id, arm, repeat) not in completed]
    if not args.resume:
        write(OUT / "pilot_protocol.json", {
            "started": now(), "bundle_id": exp.bundle.bundle_id,
            "settings": public_settings(exp.settings),
            "settings_sha256": sha(SETTINGS_PATH),
            "gold_sha256": sha(GOLD),
            "selector_sha256": sha(ROOT / "tools/source_bound_selection.py"),
            "static_selector_sha256": protocol["candidate_code_sha256"],
            "selector_code_change_after_static": "Only added CoverageEvidencePolicy injection class; propose_selection and prompts unchanged.",
            "harness_sha256": sha(__file__),
            "question_ids": list(PILOT), "arms": ["A", "B"], "repeats": 3,
            "A": "native AnswerService current EvidencePolicy",
            "B": "native AnswerService injected experimental CoverageEvidencePolicy",
            "no_quality_based_retry": True,
        })

    def run(case, arm, repeat):
        token = CancelToken()
        events, stages, audit = [], {}, {}
        held = [True]
        exp.lock.acquire()

        def progress(event):
            for key, label in STAGES.items():
                if isinstance(event.get(key), list):
                    stages[label] = event[key]
            events.append({k: v for k, v in event.items() if k not in STAGES})
            if held[0] and event.get("stage") in ("credential_read", "retrieval_completed"):
                held[0] = False
                exp.lock.release()

        gateway = OpenAICompatibleGateway(max_transient_retries=2)
        s = exp.settings
        policy = (CoverageEvidencePolicy(settings=s, gateway=gateway, api_key=exp.key,
                                         token=token, audit=audit)
                  if arm == "B" else None)
        service = AnswerService(
            exp.source_store, exp.store, exp.embedding, exp.ranker, gateway,
            exp.secrets, s, progress_callback=progress, evidence_policy=policy,
        )
        query = RagQuery(
            text=case.question, include_ocr=True, min_ocr_confidence=s.ocr_min_confidence,
            lexical_top_k=s.rag_lexical_top_k, vector_top_k=s.rag_vector_top_k,
            fusion_top_k=s.rag_fusion_top_k, reranker_top_k=s.rag_reranker_top_k,
            final_top_k=s.rag_final_top_k,
        )
        row = {"id": case.case_id, "question": case.question,
               "arm": arm, "repeat": repeat,
               "transport_attempt": failed_attempts[(case.case_id, arm, repeat)] + 1,
               "started": now()}
        try:
            row["answer"] = _answer_record(service.answer(query, token))
        except Exception as exc:
            row["answer_error"] = exp.redact(exc)
        finally:
            if held[0]:
                held[0] = False
                exp.lock.release()
        context = next((event for event in reversed(events) if "llm_context" in event), {})
        row.update(context=context.get("llm_context", ""),
                   entries=context.get("context_entries", []),
                   stages=stages, events=events, selector_audit=audit)
        if row.get("answer"):
            try:
                row["judgement"] = GoldAnswerJudge(gateway, s, exp.key).evaluate(
                    case, {**row["answer"], "evidence_context": row["context"]}, token)
            except Exception as exc:
                row["judge_error"] = exp.redact(exc)
        row["finished"] = now()
        return row

    results = list(prior)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, *task) for task in tasks]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str).replace(exp.key, "[REDACTED]") + "\n")
            print(row["id"], row["arm"], row["repeat"],
                  (row.get("judgement") or {}).get("verdict"),
                  row.get("selector_audit", {}).get("reason", ""),
                  row.get("answer_error", ""), flush=True)
    valid = {(row["id"], row["arm"], row["repeat"])
             for row in results if row.get("answer") and row.get("judgement")}
    write(OUT / "pilot_completion.json", {
        "finished": now(), "attempts": len(results), "valid_pairs": len(valid),
        "expected_valid_pairs": len(PILOT) * 6,
        "verdicts": dict(Counter((row.get("judgement") or {}).get("verdict", "missing") for row in results)),
        "errors": sum(bool(row.get("answer_error") or row.get("judge_error")) for row in results),
    })


if __name__ == "__main__":
    main()
