"""用新版答案判分项复核已保存的批量答案，不重复运行检索或生成。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.reporting.service import generate_report
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.services.settings_service import SettingsService


def rejudge_saved_run(
    cases_path: Path,
    source_dir: Path,
    output_dir: Path,
    judge_factory: Callable[[Callable[[dict[str, Any]], None]], Any],
) -> dict[str, Any]:
    """保留原运行目录；重判产物写入独立目录并记录来源。"""
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == source_dir or output_dir.exists():
        raise ValueError("输出目录必须是尚不存在、且不同于原运行目录的新目录")
    metadata = json.loads((source_dir / "run.json").read_text(encoding="utf-8"))
    source_bytes = (source_dir / "cases.jsonl").read_bytes()
    source_rows = [json.loads(line) for line in source_bytes.decode("utf-8-sig").splitlines() if line.strip()]
    dataset = load_gold_dataset(cases_path, reviewed_only=False)
    gold_by_id = {case.case_id: case for case in dataset.cases}
    if any(str(row.get("id") or "") not in gold_by_id for row in source_rows):
        raise ValueError("新版 Gold 缺少原运行中的题号")
    if any(case.corpus_snapshot != metadata.get("ai_bundle_id") for case in dataset.cases):
        raise ValueError("新版 Gold 的 AI 索引版本与原运行不一致")
    output_dir.mkdir(parents=True, exist_ok=False)
    token = CancelToken()
    started = time.monotonic()
    attempts_total = 0
    comparisons: list[dict[str, str]] = []
    new_rows: list[dict[str, Any]] = []
    for source in source_rows:
        row = dict(source)
        case = gold_by_id[str(row["id"])]
        previous = (row.get("gold_judgement") or {}).get("verdict") or "未运行"
        answer = row.get("answer") or {}
        retries = []
        case_started = time.monotonic()
        if not str(answer.get("text") or "").strip():
            judgement = {"verdict": "unavailable", "reason": "原运行没有可复核的答案"}
        else:
            attempts_total += 1
            try:
                judgement = judge_factory(retries.append).evaluate(case, answer, token)
            except Exception as exc:
                # 避免第三方异常文本包含请求头等敏感内容，细节留在受控运行日志中。
                judgement = {"verdict": "unavailable", "reason": f"裁判请求失败：{type(exc).__name__}",
                             "method": "llm_gold_judge"}
            attempts_total += len(retries)
        judgement["previous_verdict"] = previous
        judgement["rejudged_from_saved_answer"] = True
        judgement["rejudge_duration_ms"] = int((time.monotonic() - case_started) * 1000)
        row["gold_judgement"] = judgement
        new_rows.append(row)
        comparisons.append({"id": case.case_id, "previous_verdict": previous,
                            "new_verdict": str(judgement.get("verdict") or "unavailable"),
                            "findings": ", ".join(judgement.get("findings") or [])})
        print(f"{case.case_id}: {previous} -> {judgement['verdict']}", flush=True)
        # 每题落盘，网络故障或人工中断时仍可检查已完成的结果。
        with (output_dir / "cases.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    revised_metadata = dict(metadata)
    revised_metadata["dataset_path"] = str(dataset.path)
    revised_metadata["dataset_sha256"] = dataset.sha256
    revised_metadata["rejudge"] = {
        "source_run_dir": str(source_dir),
        "source_cases_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "rejudged_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.monotonic() - started) * 1000),
        "api_attempt_count": attempts_total,
        "retrieval_and_answers_reused": True,
    }
    (output_dir / "run.json").write_text(
        json.dumps(revised_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "judgement_comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "previous_verdict", "new_verdict", "findings"])
        writer.writeheader()
        writer.writerows(comparisons)
    detail = generate_report(dataset.path, output_dir, output_dir)
    return {"output_dir": str(output_dir), "comparison": comparisons,
            "summary": detail["summary"], "api_attempt_count": attempts_total}


def main() -> int:
    parser = argparse.ArgumentParser(description="用修订版 Gold 复核保存的答案，原运行目录保持不变")
    parser.add_argument("--cases", type=Path, required=True, help="修订版 Gold JSON/JSONL")
    parser.add_argument("--run", type=Path, required=True, help="原批量评测运行目录")
    parser.add_argument("--out", type=Path, required=True, help="新的输出目录，不得已存在")
    args = parser.parse_args()
    settings = SettingsService().load()
    api_key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    if not api_key:
        parser.error("设置页未保存 LLM API Key")

    def make_judge(callback):
        gateway = OpenAICompatibleGateway(max_transient_retries=3, retry_callback=callback)
        return GoldAnswerJudge(gateway, settings, api_key)

    result = rejudge_saved_run(args.cases, args.run, args.out, make_judge)
    print(f"报告：{Path(result['output_dir']) / 'report.html'}")
    print(f"裁判 API 请求尝试：{result['api_attempt_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
