"""Freeze, audit, and diagnose missing RAG evidence without changing production."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.generation.base import GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore

from tools.rag_eval_common import ROOT, lines, pair_order, read, sha, write


OUT = ROOT / "reports/rag_evidence_delivery_20261109_01"
PRIOR = ROOT / "reports/rag_code_consolidation_20261109"
NATIVE = ROOT / "reports/rag_evidence_answer_native_recheck_20261109"
HISTORIC = ROOT / "reports/rag_context_ceiling_20261104"
FIXTURES = ROOT / "tests/rag/fixtures"
PLAN = ROOT / "docs/rag_evidence_delivery_development_validation_plan_20261109.md"
MODEL_FILES = (
    ROOT / "local_full_text_search/rag/answer_service.py",
    ROOT / "local_full_text_search/rag/context_builder.py",
    ROOT / "local_full_text_search/rag/evidence_policy.py",
    ROOT / "local_full_text_search/rag/generation/openai_compatible.py",
    ROOT / "local_full_text_search/rag/evaluation/answer_judge.py",
)
DATA_FILES = (
    FIXTURES / "rag_gold_v4_2_review.json",
    FIXTURES / "rag_gold_v4_2.jsonl",
    FIXTURES / "rag_acceptance_v1_review.json",
    FIXTURES / "rag_row_holdout_v1_review.json",
    FIXTURES / "rag_row_holdout_v2_review.json",
    NATIVE / "e2e_runs.jsonl",
    PRIOR / "native_source_audit.jsonl",
    HISTORIC / "prepared.json",
    HISTORIC / "historical_loss_stages.json",
)
DIAGNOSTIC_IDS = ("rag-biz-006", "rag-biz-019", "rag-biz-023", "rag-biz-040")


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def effective_settings(settings: AppSettings) -> dict:
    values = asdict(settings)
    result = {key: value for key, value in values.items()
              if key.startswith("rag_") and "secret" not in key and "base_url" not in key}
    result["api_base_url_sha256"] = hashlib.sha256(settings.rag_api_base_url.encode()).hexdigest()
    result["api_secret_ref_sha256"] = hashlib.sha256(settings.rag_api_secret_ref.encode()).hexdigest()
    return result


def frozen_paths(store: RagIndexStore, bundle) -> list[Path]:
    sources = {Path(row["resolved_path"]) for row in lines(PRIOR / "native_source_audit.jsonl")
               if not row["actual_context_quote"] and Path(row["resolved_path"]).is_file()}
    return [PLAN, Path(__file__).resolve(), SETTINGS_PATH, store.manifest_path,
            bundle.metadata_path, *MODEL_FILES, *DATA_FILES, *sorted(sources)]


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    bundle = store.active()
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    paths = frozen_paths(store, bundle)
    manifest = {str(path): sha(path) for path in paths}
    prior = read(NATIVE / "protocol.json")
    assert bundle.bundle_id == prior["bundle"], "Published AI bundle changed"
    assert effective_settings(settings) == prior["effective_rag_settings"], "RAG settings changed"
    counts = {path.name: dict(Counter(case.review_status for case in
                                     load_gold_dataset(path, reviewed_only=False).cases))
              for path in DATA_FILES if path.name.endswith("_review.json") and path.parent == FIXTURES}
    protocol = {
        "created": now(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "published_ai_bundle": bundle.bundle_id, "model": settings.rag_api_model,
        "effective_rag_settings": effective_settings(settings),
        "settings_sha256": sha(SETTINGS_PATH), "data_statuses": counts,
        "diagnostic_case_ids": DIAGNOSTIC_IDS,
        "diagnostic_intervention": "Pre-audited source-completed contexts from 2026-11-04, only where current contexts match byte-for-byte; fixed at <= 8 entries and <= 12000 characters.",
        "pair_repeats": 3, "transport_retries": 2, "quality_retries": 0,
        "model_answer_candidate": "Production OpenAICompatibleGateway; no evidence-map prompt.",
        "prior_source": str(PRIOR / "native_source_audit.jsonl"),
        "plan": str(PLAN),
    }
    write(OUT / "protocol.json", protocol)
    write(OUT / "input_manifest_before.json", manifest)
    (OUT / "working_tree_status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True), encoding="utf-8")
    (OUT / "working_tree_before.diff").write_bytes(subprocess.check_output(["git", "diff", "--binary"]))
    print(bundle.bundle_id, counts)


def verify_freeze() -> dict:
    protocol = read(OUT / "protocol.json")
    store = RagIndexStore(Path("E:/everythingfinder-db/rag"))
    assert store.active().bundle_id == protocol["published_ai_bundle"]
    assert effective_settings(AppSettings.from_dict(read(SETTINGS_PATH))) == protocol["effective_rag_settings"]
    manifest = read(OUT / "input_manifest_before.json")
    changed = [path for path, old in manifest.items() if sha(path) != old]
    assert not changed, f"Frozen input changed: {changed}"
    return protocol


def audit() -> None:
    verify_freeze()
    raw_cases = {}
    for path in (
        FIXTURES / "rag_gold_v4_2_review.json",
        FIXTURES / "rag_row_holdout_v1_review.json",
        FIXTURES / "rag_row_holdout_v2_review.json",
        NATIVE / "acceptance_review.json",
    ):
        data = read(path)
        raw_cases.update({case["id"]: case for case in (data["cases"] if isinstance(data, dict) else data)})
    runs = {row["id"]: row for row in lines(NATIVE / "e2e_runs.jsonl") if row["repeat"] == 1}
    source_rows = [row for row in lines(PRIOR / "native_source_audit.jsonl")
                   if row["arm"] == "A" and not row["actual_context_quote"]]
    assert len(source_rows) == 19

    # These are post-hoc audit decisions only; no candidate or generator imports them.
    labels = {
        "rag-biz-004": ("equivalent_present", "实际 S4 含三块板的对应位置，首轮回答完整。"),
        "rag-biz-006": ("partial_support", "实际 S5 含低位/高位控制；S1/S3 有切桶，未发现原题所需的警告提示。"),
        "rag-biz-023": ("necessary_gap", "实际上下文只有 LVC、DEV、DUP，缺 CSDB 表格行和专属处理。"),
        "rag-biz-021": ("equivalent_present", "实际第 27/30 张足以支持两条条件分支，首轮回答完整。"),
        "rag-biz-019": ("necessary_gap", "实际第 17/20/53 张未含第 14 张 40 次以上灌注和对应处理顺序。"),
        "rag-biz-030": ("equivalent_present", "现有其他质控资料支持初始均值与 SD 的核心处理，回答首轮正确；仍保留来源绑定复核。"),
        "rag-biz-028": ("equivalent_present", "已入上下文的其他定标和临床结果资料支持所问方向，首轮回答正确。"),
        "rag-biz-040": ("necessary_gap", "实际 S6 只含低样本量项目分支；所有项目变差时查冲洗头的第 16 张缺席。"),
        "rag-biz-049": ("optional_missing", "需先确认 SDM 版本；实际第 35 张已区分串口/网口，缺恢复细节不妨碍先追问。"),
        "rag-biz-050": ("equivalent_present", "实际 S5 已含四类通用方向；用户确认通用排查适用，本轮漏答属于资料使用问题。"),
        "rag-row-holdout-004": ("equivalent_present", "实际来源含两机型样本类型，首轮回答正确；待审标注不计正式分。"),
        "rag-row-holdout-005": ("equivalent_present", "实际来源含 Na/K 与 Cl 的斜率范围和单位，首轮回答正确。"),
        "rag-row-holdout-008": ("gold_disputed", "问题写 IMFH1，实际表格及回答写 MFH1；须先复核题目编号。"),
        "rag-row-holdout2-001": ("equivalent_present", "实际 S2 完整列出六类码制；首轮虽判错，指定引文缺席不是原因。"),
        "rag-row-holdout2-007": ("applicability_unresolved", "实际 S3 有 1115Ω/37℃，但与其他机型温度材料同在上下文，需核实 BS-800 归属。"),
        "evidence-flow-independent-05": ("equivalent_present", "实际其他来源的直接/间接 ISE 对照已足以回答；pending 不计正式分。"),
    }
    rows = []
    for item in source_rows:
        case = raw_cases[item["id"]]
        run = runs[item["id"]]
        category, reason = labels[item["id"]]
        if item["id"] == "rag-biz-006" and "高低位液位传感器" in item["quote"]:
            category, reason = "equivalent_present", "实际 S5 已完整包含低位启动与高位停止的泵阀动作。"
        entry_summaries = [{"source_id": entry["source_id"], "filename": entry.get("filename"),
                            "location": entry.get("location"),
                            "included_text_sha256": hashlib.sha256(entry["included_text"].encode()).hexdigest(),
                            "included_text_excerpt": entry["included_text"][:300]}
                           for entry in run["arms"]["A"]["entries"]]
        rows.append({
            "id": item["id"], "group_id": case.get("group_id"), "question": case["question"],
            "review_status": case["review_status"], "expected_behavior": case["expected_behavior"],
            "core_facts": [check["text"] for check in case.get("answer_checks", []) if check["role"] == "core"],
            "classification": category, "audit_reason": reason,
            "gold_source": {key: item.get(key) for key in ("resolved_path", "page", "position", "quote", "source_sha", "index_alternatives", "original_quote", "index_quote")},
            "actual_context_entries": entry_summaries,
            "actual_answer": run["arms"]["A"]["answer"]["text"],
            "judge_verdict": run["arms"]["A"].get("judgement", {}).get("verdict"),
            "semantic_applicability": "confirmed" if category not in {"applicability_unresolved", "gold_disputed"} else "needs_business_review",
        })
    with (OUT / "fact_gap_audit.jsonl").open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(OUT / "fact_gap_summary.json", {"quotes": len(rows), "cases": len({row["id"] for row in rows}),
                                          "classes": dict(Counter(row["classification"] for row in rows))})
    print(read(OUT / "fact_gap_summary.json"))


def diagnose() -> None:
    verify_freeze()
    prepared = {row["id"]: row for row in read(HISTORIC / "prepared.json")}
    latest = {row["id"]: row for row in lines(NATIVE / "e2e_runs.jsonl") if row["repeat"] == 1}
    rows = []
    for case_id in DIAGNOSTIC_IDS:
        old = prepared[case_id]
        current = latest[case_id]["arms"]["A"]
        assert old["current"]["context"] == current["context"]
        source_completed = old["source_completed"]
        assert len(source_completed["context_entries"]) <= 8
        assert len(source_completed["context"]) <= 12_000
        rows.append({"id": case_id, "question": old["question"],
                     "A_context": current["context"], "A_entries": current["entries"],
                     "B_context": source_completed["context"], "B_entries": source_completed["context_entries"],
                     "added_chunk_ids": old["added_ids"], "removed_chunk_ids": old["removed_original_ids"],
                     "same_budget": True,
                     "provenance": str(HISTORIC / "prepared.json")})
    with (OUT / "diagnostic_contexts.jsonl").open("x", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    print([(row["id"], len(row["A_entries"]), len(row["B_entries"]),
            len(row["A_context"]), len(row["B_context"]), row["removed_chunk_ids"])
           for row in rows])


def run_diagnostic(workers: int) -> None:
    verify_freeze()
    contexts = {row["id"]: row for row in lines(OUT / "diagnostic_contexts.jsonl")}
    cases = {case.case_id: case for case in load_gold_dataset(
        FIXTURES / "rag_gold_v4_2_review.json", reviewed_only=False).cases}
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key, "Saved model credential unavailable"
    output = OUT / "diagnostic_runs.jsonl"
    existing = {(row["id"], row["repeat"]) for row in lines(output)} if output.exists() else set()

    def execute(case_id: str, repeat: int, order: str) -> dict:
        case = cases[case_id]
        data = contexts[case_id]
        row = {"id": case_id, "repeat": repeat, "order": order, "arms": {}}
        for arm in order:
            gateway = OpenAICompatibleGateway(max_transient_retries=2)
            token = CancelToken()
            context = data[f"{arm}_context"]
            entries = data[f"{arm}_entries"]
            try:
                request = GenerationRequest(
                    case.question, context, settings.rag_api_model, settings.rag_api_base_url,
                    settings.rag_api_style, key, settings.rag_api_timeout_seconds,
                    query_plan_summary=AnswerService._plan_summary(QueryInterpreter().interpret(case.question)),
                )
                answer = gateway.generate(request, token).text.strip()
                citations = CitationValidator().inspect(answer, {entry["source_id"]: () for entry in entries})
                answer_record = {"text": answer, "status": "insufficient" if answer == "资料中没有足够依据。" else "answered",
                                 "source_ids": list(citations.cited_ids), "citation_valid": citations.structurally_valid}
                judgement = GoldAnswerJudge(gateway, settings, key).evaluate(
                    case, {**answer_record, "evidence_context": context}, token)
                row["arms"][arm] = {"context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                                     "answer": answer_record, "judgement": judgement}
            except Exception as exc:
                row["arms"][arm] = {"context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                                     "error": str(exc).replace(key, "[REDACTED]")[:500]}
        return row

    jobs = [(case_id, repeat, pair_order(index, repeat))
            for index, case_id in enumerate(DIAGNOSTIC_IDS) for repeat in (1, 2, 3)
            if (case_id, repeat) not in existing]
    with output.open("a", encoding="utf-8") as target, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(execute, *job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            target.write(json.dumps(row, ensure_ascii=False).replace(key, "[REDACTED]") + "\n")
            target.flush()
            print(row["id"], row["repeat"],
                  *(row["arms"][arm].get("judgement", {}).get("verdict", "error") for arm in "AB"), flush=True)
    verify_freeze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "audit", "diagnose", "run_diagnostic", "verify"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze()
    elif args.stage == "audit":
        audit()
    elif args.stage == "diagnose":
        diagnose()
    elif args.stage == "run_diagnostic":
        run_diagnostic(args.workers)
    else:
        print(verify_freeze()["published_ai_bundle"])


if __name__ == "__main__":
    main()
