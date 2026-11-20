"""Freeze and replay the behavior-neutral RAG evaluation refactor."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.evaluation.evidence_matcher import load_chunk_rows
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.index_store import RagIndexStore
from tools.rag_eval_common import ROOT, lines, read, sha, write
from tools.rag_trace_audit import audit_case_sources


OUT = ROOT / "reports/rag_code_consolidation_20261109"
NATIVE = ROOT / "reports/rag_evidence_answer_native_recheck_20261109"
OLD = ROOT / "reports/rag_evidence_answer_flow_20261108"
FIXTURES = ROOT / "tests/rag/fixtures"
INDEX = Path("E:/everythingfinder-db/rag")
DATASETS = (
    FIXTURES / "rag_gold_v4_2_review.json",
    FIXTURES / "rag_acceptance_v1_review.json",
    FIXTURES / "rag_row_holdout_v1_review.json",
    FIXTURES / "rag_row_holdout_v2_review.json",
)
PRODUCTION = (
    ROOT / "local_full_text_search/rag/answer_service.py",
    ROOT / "local_full_text_search/rag/generation/openai_compatible.py",
    ROOT / "local_full_text_search/rag/evaluation/answer_judge.py",
    ROOT / "local_full_text_search/rag/evaluation/runner.py",
    ROOT / "local_full_text_search/rag/evaluation/reporting/service.py",
)
TOOLS = (
    ROOT / "tools/rag_eval_common.py",
    ROOT / "tools/rag_trace_audit.py",
    ROOT / "tools/rejudge_scoring_v3.py",
    ROOT / "tools/validate_scene_quota.py",
    ROOT / "tools/validate_scene_guard.py",
    ROOT / "tools/validate_evidence_answer_flow.py",
)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def inventory() -> dict:
    helper_names = {"read", "lines", "sha", "dump", "write", "model_settings", "run_jobs", "collect"}
    helper_occurrences: dict[str, list[str]] = {name: [] for name in helper_names}
    imports = []
    for path in sorted((ROOT / "tools").glob("*.py")):
        source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in source.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in helper_names:
                helper_occurrences[node.name].append(f"{path.relative_to(ROOT)}:{node.lineno}")
            elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tools."):
                imports.append({
                    "source": str(path.relative_to(ROOT)), "line": node.lineno,
                    "module": node.module,
                    "names": [item.name for item in node.names],
                })
    return {
        "created": now(),
        "helper_definitions": helper_occurrences,
        "helper_counts": {name: len(rows) for name, rows in helper_occurrences.items()},
        "tool_to_tool_imports": imports,
        "classification": {
            "shared_now": ["JSON/JSONL read/write", "SHA-256", "redaction", "paired order"],
            "read_only_audit": ["original page extraction", "index quote mapping", "actual sent text check"],
            "experiment_only": ["scene selector", "method guard", "evidence-map prompt", "experimental gateway"],
        },
    }


def freeze() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    previous = read(OLD / "protocol.json")
    store = RagIndexStore(INDEX)
    bundle = store.active()
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    datasets = [load_gold_dataset(path, reviewed_only=False) for path in DATASETS]
    old_hashes = previous["hashes"]
    stable = [SETTINGS_PATH, store.manifest_path, bundle.metadata_path, *DATASETS, *PRODUCTION]
    comparisons = [
        {"file": str(path), "old": old_hashes.get(str(path)), "current": sha(path),
         "same": old_hashes.get(str(path)) == sha(path)}
        for path in stable
    ]
    protocol = {
        "created": now(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "published_bundle": bundle.bundle_id, "model": settings.rag_api_model,
        "plan": str(ROOT / "docs/rag_evidence_answer_next_plan_20261109.md"),
        "previous_report": str(OLD / "report.md"),
        "scope": "Evaluation helper consolidation and read-only source audit; no production behavior change.",
        "quality_candidate": "Previously rejected evidence-map candidate, unchanged prompt and gateway behavior.",
        "datasets": [{"file": str(path), "count": len(data.cases),
                      "statuses": dict(Counter(case.review_status for case in data.cases))}
                     for path, data in zip(DATASETS, datasets)],
        "stable_input_comparison": comparisons,
    }
    assert bundle.bundle_id == previous["bundle"]
    assert all(item["same"] for item in comparisons if item["file"] != str(SETTINGS_PATH)), (
        "Prior experiment index, Gold, or production code drifted"
    )
    protocol["external_settings_rewrite"] = next(
        item for item in comparisons if item["file"] == str(SETTINGS_PATH)
    )
    write(OUT / "protocol.json", protocol)
    write(OUT / "input_manifest_before.json", {
        "created": now(), "published_bundle": bundle.bundle_id,
        "previous_candidate_sha256": old_hashes[str(ROOT / "tools/validate_evidence_answer_flow.py")],
        "current": {str(path): sha(path) for path in [*stable, *TOOLS]},
    })
    (OUT / "working_tree_status.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True), encoding="utf-8"
    )
    found = inventory()
    write(OUT / "code_overlap_inventory.json", found)
    lines_md = ["# 评测脚本依赖清单", "", f"- 盘点时间：{found['created']}",
                f"- tools 间导入：{len(found['tool_to_tool_imports'])} 处", "",
                "| 函数名 | 仍在各脚本独立定义的数量 |", "|---|---:|"]
    lines_md.extend(f"| `{name}` | {count} |" for name, count in sorted(found["helper_counts"].items()))
    lines_md.extend(["", "已抽取公共读写、哈希、脱敏和配对顺序。来源审计在独立只读模块中。",
                     "仍保留的场景选择、问答提示和模型网关属于实验策略，需单独验证。", "",
                     "## 仍存在的工具间依赖", ""])
    lines_md.extend(f"- `{item['source']}:{item['line']}` -> `{item['module']}`: {', '.join(item['names'])}"
                    for item in found["tool_to_tool_imports"])
    (OUT / "code_dependency_graph.md").write_text("\n".join(lines_md) + "\n", encoding="utf-8")
    (OUT / "execution_log.md").write_text(
        "# 阶段记录\n\n| 阶段 | 结论 | 核对结果 |\n|---|---|---|\n"
        "| 0 输入冻结 | 通过，配置有外部改写 | 发布包、Gold 和生产代码与上一轮一致；配置哈希变化，模型重跑另行冻结新配置。 |\n"
        "| 1 重复依赖盘点 | 通过 | 详见 code_overlap_inventory.json 和 code_dependency_graph.md。 |\n",
        encoding="utf-8",
    )
    print(f"frozen {bundle.bundle_id}; {len(found['tool_to_tool_imports'])} tool imports")


def replay() -> None:
    frozen = read(OUT / "input_manifest_before.json")
    for path, expected in frozen["current"].items():
        assert sha(path) == expected, f"Input changed since freeze: {path}"
    store = RagIndexStore(INDEX)
    bundle = store.active()
    assert bundle.bundle_id == frozen["published_bundle"]
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunk_rows = load_chunk_rows(conn)
    checks = []
    all_rows = []
    for cohort, dataset_name, context_name, old_name in (
        ("development", DATASETS[0], "dev_contexts.jsonl", "source_audit.json"),
        ("independent", OLD / "acceptance_review.json", "acceptance_contexts.jsonl", "acceptance_source_audit.json"),
    ):
        cases = {case.case_id: case for case in load_gold_dataset(dataset_name, reviewed_only=False).cases}
        contexts = {row["id"]: row for row in lines(OLD / context_name)}
        old = read(OLD / old_name)
        current = [row for cid, context in contexts.items()
                   for row in audit_case_sources(cases[cid], chunk_rows, context["entries"])]
        key = lambda row: (row["id"], row["file"], row.get("page"), row["quote"])
        old_by_key = {key(row): row for row in old}
        new_by_key = {key(row): row for row in current}
        fields = ("resolved_path", "source_sha", "original_quote", "original_error",
                  "index_quote", "actual_context_quote")
        differences = [
            {"id": item[0], "file": item[1], "page": item[2], "field": field,
             "old": old_by_key[item].get(field), "new": new_by_key[item].get(field)}
            for item in sorted(old_by_key.keys() & new_by_key.keys()) for field in fields
            if old_by_key[item].get(field) != new_by_key[item].get(field)
        ]
        check = {"cohort": cohort, "previous_count": len(old), "current_count": len(current),
                 "missing": len(old_by_key.keys() - new_by_key.keys()),
                 "added": len(new_by_key.keys() - old_by_key.keys()), "differences": differences}
        checks.append(check)
        all_rows.extend({"cohort": cohort, **row} for row in current)
    assert all(not c["missing"] and not c["added"] and not c["differences"] for c in checks)
    with (OUT / "source_audit_replay.jsonl").open("x", encoding="utf-8") as file:
        for row in all_rows:
            import json
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    write(OUT / "source_audit_replay_summary.json", checks)
    md = ["# 原文—索引—实际上下文核验", "", "重放上一轮已保存的真实上下文，不调用模型。", "",
          "| 集合 | 引文数 | 与上一轮核验结果的差异 |", "|---|---:|---:|"]
    md.extend(f"| {c['cohort']} | {c['current_count']} | {len(c['differences']) + c['missing'] + c['added']} |"
              for c in checks)
    md.extend(["", "`actual_context_quote` 由现行 `score_stage` 根据实际发送的 included_text 和来源片段判定。",
               "`original_quote`、`index_quote` 只证明原句存在，不能自动证明机型、项目范围和方法适用。",
               "对象适用性的人工复核仍以原资料版式和上一轮逐题审计为准。", ""])
    (OUT / "source_audit_replay.md").write_text("\n".join(md), encoding="utf-8")
    with (OUT / "execution_log.md").open("a", encoding="utf-8") as file:
        file.write("| 2 公共工具兼容 | 通过 | 旧输入与生产代码哈希一致，公共函数单测另见 checks.md。 |\n")
        file.write("| 3 来源审计重放 | 通过 | 开发与独立集引文逐字段对照上一轮记录，差异为 0。 |\n")
    print(f"source replay: {sum(c['current_count'] for c in checks)} quotes; zero differences")


def finish() -> None:
    frozen = read(OUT / "input_manifest_before.json")
    unchanged = {path: sha(path) == expected for path, expected in frozen["current"].items()}
    assert all(unchanged.values()), "Frozen inputs changed after replay"
    write(OUT / "final_manifest.json", {
        "created": now(), "published_bundle": frozen["published_bundle"],
        "input_hashes_verified": True, "files": frozen["current"],
        "result_hashes": {str(path): sha(path) for path in sorted(OUT.iterdir()) if path.is_file()},
    })
    print("final input hashes verified")


def native_audit() -> None:
    frozen = read(OUT / "input_manifest_before.json")
    assert all(sha(path) == expected for path, expected in frozen["current"].items())
    store = RagIndexStore(INDEX)
    bundle = store.active()
    assert bundle.bundle_id == frozen["published_bundle"]
    with store.connect_read_only(bundle.metadata_path) as conn:
        chunk_rows = load_chunk_rows(conn)
    cases = {
        case.case_id: case
        for source in (*DATASETS, NATIVE / "acceptance_review.json")
        for case in load_gold_dataset(source, reviewed_only=False).cases
    }
    all_rows = []
    for run in lines(NATIVE / "e2e_runs.jsonl"):
        if run["repeat"] != 1:
            continue
        for arm in ("A", "B"):
            all_rows.extend({"cohort": run["cohort"], "arm": arm, **row}
                            for row in audit_case_sources(
                                cases[run["id"]], chunk_rows, run["arms"][arm]["entries"]
                            ))
    assert len({(row["id"], row["arm"]) for row in all_rows}) > 0
    with (OUT / "native_source_audit.jsonl").open("x", encoding="utf-8") as file:
        import json
        for row in all_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "audited_quotes_per_arm": len(all_rows) // 2,
        "original_quote": sum(row["original_quote"] for row in all_rows if row["arm"] == "A"),
        "index_quote": sum(row["index_quote"] for row in all_rows if row["arm"] == "A"),
        "actual_context_quote_A": sum(row["actual_context_quote"] for row in all_rows if row["arm"] == "A"),
        "actual_context_quote_B": sum(row["actual_context_quote"] for row in all_rows if row["arm"] == "B"),
        "failed_original": sorted({row["id"] for row in all_rows if not row["original_quote"]}),
        "failed_index": sorted({row["id"] for row in all_rows if not row["index_quote"]}),
    }
    write(OUT / "native_source_audit_summary.json", summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "replay", "native_audit", "finish"))
    stage = parser.parse_args().stage
    {"freeze": freeze, "replay": replay, "native_audit": native_audit, "finish": finish}[stage]()


if __name__ == "__main__":
    main()
