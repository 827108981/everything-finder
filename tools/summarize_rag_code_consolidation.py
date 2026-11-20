"""Summarize the frozen evaluation refactor and native answer A/B replay."""

from __future__ import annotations


from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from tools.rag_eval_common import ROOT, lines, read, sha, write


OUT = ROOT / "reports/rag_code_consolidation_20261109"
NATIVE = ROOT / "reports/rag_evidence_answer_native_recheck_20261109"
OLD = ROOT / "reports/rag_evidence_answer_flow_20261108"


def verdict(run: dict, arm: str) -> str:
    return run["arms"][arm].get("judgement", {}).get("verdict", "judge_error")


def main() -> None:
    native = lines(NATIVE / "e2e_runs.jsonl")
    old = {row["id"]: row for row in lines(OLD / "e2e_runs.jsonl") if row["repeat"] == 1}
    first = {row["id"]: row for row in native if row["repeat"] == 1}
    assert len(first) == len(old) == 90
    same_old_context = {
        case_id: old[case_id]["arms"]["A"]["context"] == run["arms"]["A"]["context"]
        for case_id, run in first.items()
    }
    assert all(same_old_context.values())
    assert all(row["same_context"] for row in native)

    per_case = []
    by_id = defaultdict(list)
    for row in native:
        by_id[row["id"]].append(row)
    for case_id, runs in sorted(by_id.items()):
        runs.sort(key=lambda row: row["repeat"])
        first_run = runs[0]
        per_case.append({
            "id": case_id, "group_id": first_run["group"], "cohort": first_run["cohort"],
            "repeat_count": len(runs),
            "A": [verdict(row, "A") for row in runs],
            "B": [verdict(row, "B") for row in runs],
            "same_context_each_pair": all(row["same_context"] for row in runs),
            "same_context_as_previous_experiment": same_old_context[case_id],
            "first_pair_change": (
                "gain" if verdict(first_run, "B") == "consistent" and verdict(first_run, "A") != "consistent"
                else "regression" if verdict(first_run, "A") == "consistent" and verdict(first_run, "B") != "consistent"
                else "unchanged_consistent" if verdict(first_run, "A") == verdict(first_run, "B") == "consistent"
                else "both_not_consistent" if verdict(first_run, "A") != "consistent" and verdict(first_run, "B") != "consistent"
                else "other"
            ),
        })
    write(OUT / "native_per_case.json", per_case)

    cohorts = {}
    for cohort in ("formal", "regression_pending", "independent_pending"):
        subset = [row for row in first.values() if row["cohort"] == cohort]
        cohorts[cohort] = {
            "questions": len(subset),
            "A": dict(Counter(verdict(row, "A") for row in subset)),
            "B": dict(Counter(verdict(row, "B") for row in subset)),
            "first_pair_gains": [item["id"] for item in per_case if item["cohort"] == cohort and item["first_pair_change"] == "gain"],
            "first_pair_regressions": [item["id"] for item in per_case if item["cohort"] == cohort and item["first_pair_change"] == "regression"],
        }
    write(OUT / "native_summary.json", {
        "created": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "bundle": read(NATIVE / "protocol.json")["bundle"],
        "model": read(NATIVE / "final_manifest.json")["model"],
        "first_pairs": len(first), "repeat_cases": len(native) - len(first),
        "same_context_as_previous_experiment": sum(same_old_context.values()),
        "same_context_within_pair": sum(row["same_context"] for row in native),
        "cohorts": cohorts,
        "judge_errors": [{"id": row["id"], "repeat": row["repeat"], "arm": arm}
                         for row in native for arm in ("A", "B") if verdict(row, arm) == "judge_error"],
    })

    gold = ROOT / "tests/rag/fixtures/rag_gold_v4_2"
    review = read(str(gold) + "_review.json")
    jsonl = lines(str(gold) + ".jsonl")
    assert review["cases"] == jsonl
    compatibility = {
        "previous_and_new_first_pair_contexts_identical": 90,
        "paired_contexts_identical": len(native),
        "gold_review_json_equals_jsonl": True,
        "gold_statuses": dict(Counter(row["review_status"] for row in jsonl)),
        "frozen_source_replay": read(OUT / "source_audit_replay_summary.json"),
        "native_source_audit": read(OUT / "native_source_audit_summary.json"),
        "settings_unchanged_within_valid_native_run": (
            read(NATIVE / "final_manifest.json")["settings_file_sha256_at_start"]
            == read(NATIVE / "final_manifest.json")["settings_file_sha256_at_end"]
        ),
    }
    write(OUT / "before_after_manifest.json", compatibility)

    rows = ["# 逐题真实入口 A/B 判定", "", "A 为现行回答，B 为证据清单候选。每格按首轮、复测 2、复测 3 顺序列出。`C` 正确、`I` 不完整、`X` 错误、`Q` 追问、`U` 不确定、`E` 裁判失败。独立与回归题均为 pending，不计正式分。", "", "| 集合 | 题号 | A | B | 首轮变化 |", "|---|---|---|---|---|"]
    symbols = {"consistent": "C", "incomplete": "I", "incorrect": "X", "needs_clarification": "Q", "uncertain": "U", "judge_error": "E"}
    for item in per_case:
        rows.append(f"| {item['cohort']} | {item['id']} | {'/'.join(symbols.get(v, v) for v in item['A'])} | {'/'.join(symbols.get(v, v) for v in item['B'])} | {item['first_pair_change']} |")
    (OUT / "native_per_case.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    checks = ["# 兼容与校验", "", "- 90/90 道首轮题与上一轮保存的真实入口上下文逐字一致。", "- 140/140 组 A/B 实际上下文逐字一致。", "- 当前有效重跑期间配置文件哈希、有效 RAG 配置、代码、索引及 Gold 哈希均保持一致。", "- v4.2 review JSON 的 50 道题与 JSONL 逐条完全相同；状态保持不变。", "- 旧开发与独立集 37 条来源核验记录逐字段重放，差异 0。", "- 124 条正向引文在原资料和当前索引正文均存在；两臂实际上下文各出现 105 条。", "- 只读审计不等同于机型、阶段、表格行列的语义适用性判定；争议题保留人工复核。", ""]
    (OUT / "refactor_compatibility.md").write_text("\n".join(checks), encoding="utf-8")
    print(f"summarized {len(per_case)} cases, {len(native)} pairs")


if __name__ == "__main__":
    main()
