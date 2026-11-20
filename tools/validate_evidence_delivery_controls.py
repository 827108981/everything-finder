"""Run preselected unchanged-context controls for evidence delivery diagnostics."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

from tools.rag_eval_common import ROOT, lines, pair_order, read, sha, write
from tools.validate_evidence_delivery import verify_freeze


OUT = ROOT / "reports/rag_evidence_delivery_20261109_01"
NATIVE = ROOT / "reports/rag_evidence_answer_native_recheck_20261109/e2e_runs.jsonl"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_2_review.json"
CONTROL_IDS = (
    "rag-biz-004",  # Normal sourced answer
    "rag-biz-021",  # Conditional method answer
    "rag-biz-014",  # Clarification
    "rag-biz-017",  # Out-of-corpus refusal
    "rag-biz-048",  # Similar model counterexample
    "rag-biz-009",  # Ordered procedure
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare() -> None:
    verify_freeze()
    assert sha(NATIVE) == read(OUT / "input_manifest_before.json")[str(NATIVE)]
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    latest = {row["id"]: row for row in lines(NATIVE) if row["repeat"] == 1}
    rows = []
    for case_id in CONTROL_IDS:
        case = cases[case_id]
        assert case.review_status == "reviewed"
        baseline = latest[case_id]["arms"]["A"]
        context = baseline["context"]
        entries = baseline["entries"]
        rows.append({
            "id": case_id, "group_id": case.group_id,
            "question": case.question, "expected_behavior": case.expected_behavior,
            "context": context, "entries": entries,
            "context_sha256": digest(context),
            "A_equals_B": True, "source_run": str(NATIVE),
        })
    output = OUT / "controls.json"
    assert not output.exists(), "Control set is already frozen"
    write(output, {"protocol_sha256": sha(OUT / "protocol.json"),
                   "code_sha256": sha(Path(__file__)), "runs_per_arm": 3,
                   "control_ids": CONTROL_IDS, "cases": rows})
    print("Frozen controls:", [row["id"] for row in rows])


def run(workers: int) -> None:
    verify_freeze()
    frozen = read(OUT / "controls.json")
    assert sha(Path(__file__)) == frozen["code_sha256"]
    assert sha(OUT / "protocol.json") == frozen["protocol_sha256"]
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    controls = {row["id"]: row for row in frozen["cases"]}
    output = OUT / "control_runs.jsonl"
    existing = {(row["id"], row["repeat"]) for row in lines(output)} if output.exists() else set()

    def execute(case_id: str, repeat: int, order: str) -> dict:
        case = cases[case_id]
        item = controls[case_id]
        row = {"id": case_id, "repeat": repeat, "order": order,
               "context_sha256": item["context_sha256"], "arms": {}}
        for arm in order:
            gateway = OpenAICompatibleGateway(max_transient_retries=2)
            token = CancelToken()
            context = item["context"]
            entries = item["entries"]
            try:
                request = GenerationRequest(
                    case.question, context, settings.rag_api_model,
                    settings.rag_api_base_url, settings.rag_api_style, key,
                    settings.rag_api_timeout_seconds,
                    query_plan_summary=AnswerService._plan_summary(
                        QueryInterpreter().interpret(case.question)),
                )
                answer = gateway.generate(request, token).text.strip()
                citations = CitationValidator().inspect(
                    answer, {entry["source_id"]: () for entry in entries})
                result = {"text": answer,
                          "status": "insufficient" if answer == "资料中没有足够依据。" else "answered",
                          "source_ids": list(citations.cited_ids),
                          "citation_valid": citations.structurally_valid}
                judgement = GoldAnswerJudge(gateway, settings, key).evaluate(
                    case, {**result, "evidence_context": context}, token)
                row["arms"][arm] = {"answer": result, "judgement": judgement}
            except Exception as exc:
                row["arms"][arm] = {"error": str(exc).replace(key, "[REDACTED]")[:500]}
        return row

    jobs = [(case_id, repeat, pair_order(index, repeat))
            for index, case_id in enumerate(CONTROL_IDS) for repeat in (1, 2, 3)
            if (case_id, repeat) not in existing]
    with output.open("a", encoding="utf-8") as target, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            target.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            target.flush()
            print(row["id"], row["repeat"],
                  *(row["arms"][arm].get("judgement", {}).get("verdict", "error")
                    for arm in "AB"), flush=True)
    verify_freeze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    else:
        run(args.workers)


if __name__ == "__main__":
    main()
