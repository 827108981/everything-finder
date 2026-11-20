"""Pre-freeze and run one complete D plus protection-control evidence batch."""

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
from local_full_text_search.rag.generation.openai_compatible import (
    OpenAICompatibleGateway, provider_request_payload,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

from tools.rag_eval_common import ROOT, lines, pair_order, read, sha, write
from tools.validate_evidence_delivery import verify_freeze


ROOT_OUT = ROOT / "reports/rag_evidence_delivery_20261109_01"
OUT = ROOT_OUT / "complete_batch_v2"
DIAGNOSTIC = ROOT_OUT / "diagnostic_contexts.jsonl"
CONTROLS = ROOT_OUT / "controls.json"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_2_review.json"


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def frozen_cases() -> list[dict]:
    diagnostic = [dict(row, cohort="D") for row in lines(DIAGNOSTIC)]
    controls = [
        {"id": row["id"], "question": row["question"],
         "A_context": row["context"], "B_context": row["context"],
         "A_entries": row["entries"], "B_entries": row["entries"],
         "cohort": "C"}
        for row in read(CONTROLS)["cases"]
    ]
    assert len(diagnostic) == 4 and len(controls) == 6
    return diagnostic + controls


def prepare() -> None:
    verify_freeze()
    OUT.mkdir(exist_ok=False)
    cases = frozen_cases()
    assert len({row["id"] for row in cases}) == len(cases)
    for row in cases:
        for arm in "AB":
            assert len(row[f"{arm}_entries"]) <= 8
            assert len(row[f"{arm}_context"]) <= 12_000
        if row["cohort"] == "C":
            assert row["A_context"] == row["B_context"]
            assert row["A_entries"] == row["B_entries"]
    protocol = {
        "input_protocol_sha256": sha(ROOT_OUT / "protocol.json"),
        "input_manifest_sha256": sha(ROOT_OUT / "input_manifest_before.json"),
        "diagnostic_contexts_sha256": sha(DIAGNOSTIC),
        "controls_sha256": sha(CONTROLS),
        "gold_sha256": sha(GOLD),
        "settings_sha256": sha(SETTINGS_PATH),
        "runner_sha256": sha(Path(__file__)),
        "case_ids": [row["id"] for row in cases],
        "cohorts": {row["id"]: row["cohort"] for row in cases},
        "repeats_per_arm": 3, "pair_order": "alternating AB/BA",
        "transport_retries": 2, "quality_retries": 0,
        "arm_definition": "D: pre-audited same-budget source completion; C: A and B identical",
    }
    write(OUT / "protocol.json", protocol)
    print("Frozen complete batch:", len(cases), "groups, 3 pairs each")


def verify() -> dict:
    verify_freeze()
    protocol = read(OUT / "protocol.json")
    for path, key in (
        (ROOT_OUT / "protocol.json", "input_protocol_sha256"),
        (ROOT_OUT / "input_manifest_before.json", "input_manifest_sha256"),
        (DIAGNOSTIC, "diagnostic_contexts_sha256"),
        (CONTROLS, "controls_sha256"),
        (GOLD, "gold_sha256"),
        (SETTINGS_PATH, "settings_sha256"),
        (Path(__file__), "runner_sha256"),
    ):
        assert sha(path) == protocol[key], f"Frozen input changed: {path}"
    assert [row["id"] for row in frozen_cases()] == protocol["case_ids"]
    return protocol


def run(workers: int) -> None:
    protocol = verify()
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key
    cases = {case.case_id: case for case in load_gold_dataset(GOLD, reviewed_only=False).cases}
    prepared = frozen_cases()
    output = OUT / "runs.jsonl"
    existing = {(row["id"], row["repeat"]) for row in lines(output)} if output.exists() else set()

    def execute(item: dict, repeat: int, order: str) -> dict:
        case = cases[item["id"]]
        assert case.review_status == "reviewed"
        record = {"id": item["id"], "cohort": item["cohort"],
                  "repeat": repeat, "order": order, "arms": {}}
        for arm in order:
            context = item[f"{arm}_context"]
            entries = item[f"{arm}_entries"]
            gateway = OpenAICompatibleGateway(max_transient_retries=2)
            token = CancelToken()
            try:
                request = GenerationRequest(
                    case.question, context, settings.rag_api_model,
                    settings.rag_api_base_url, settings.rag_api_style, key,
                    settings.rag_api_timeout_seconds,
                    query_plan_summary=AnswerService._plan_summary(
                        QueryInterpreter().interpret(case.question)),
                )
                payload = provider_request_payload(request)
                answer = gateway.generate(request, token).text.strip()
                citation_check = CitationValidator().inspect(
                    answer, {entry["source_id"]: () for entry in entries})
                answer_record = {
                    "text": answer,
                    "status": "insufficient" if answer == "资料中没有足够依据。" else "answered",
                    "source_ids": list(citation_check.cited_ids),
                    "citation_valid": citation_check.structurally_valid,
                }
                judgement = GoldAnswerJudge(gateway, settings, key).evaluate(
                    case, {**answer_record, "evidence_context": context}, token)
                record["arms"][arm] = {
                    "context_sha256": hash_text(context),
                    "request_sha256": hash_text(json.dumps(payload, ensure_ascii=False, sort_keys=True)),
                    "request_snapshot": payload,
                    "answer": answer_record, "judgement": judgement,
                }
            except Exception as exc:
                record["arms"][arm] = {
                    "context_sha256": hash_text(context),
                    "error": str(exc).replace(key, "[REDACTED]")[:500],
                }
        return record

    jobs = [(item, repeat, pair_order(index, repeat))
            for index, item in enumerate(prepared) for repeat in (1, 2, 3)
            if (item["id"], repeat) not in existing]
    with output.open("a", encoding="utf-8") as target, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            target.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            target.flush()
            print(row["id"], row["repeat"],
                  *(row["arms"][arm].get("judgement", {}).get("verdict", "error")
                    for arm in "AB"), flush=True)
    verify()
    assert len({(row["id"], row["repeat"]) for row in lines(output)}) == len(protocol["case_ids"]) * 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "verify"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "run":
        run(args.workers)
    else:
        print("Frozen cases:", len(verify()["case_ids"]))


if __name__ == "__main__":
    main()
