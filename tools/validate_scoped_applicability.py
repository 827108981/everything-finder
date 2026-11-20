"""Fixed-pool applicability ablation for the published RAG bundle."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import subprocess
from unittest.mock import patch
import xml.etree.ElementTree as ET

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import EvidenceChunk, RagQuery
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.secrets import WindowsCredentialSecretStore
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from tools.rag_eval_common import ROOT, read, sha, write
from tools.scoped_applicability_candidate import ScopedApplicability
from tools.scoped_branch_planner import PROMPT, plan as plan_scoped, validate as validate_scoped


PREVIOUS = ROOT / "reports/rag_040_retrieval_20261110_01"
GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_2_review.json"
MANIFEST = Path("E:/everythingfinder-db/rag/rag.manifest.json")
SOURCE = Path("E:/生化免疫临床知识/生化试剂应用及常见问题-SH-20190110.pptx")
TARGETS = {14: "54f845034ca4b65d74b0d131f67ec6bf7740d53675b8994ad9b64557d8e79080",
           16: "841cbed3a28424fc897842775372df8012c08d6539e9cc701dc1bdf3a93ed68c"}


def norm(s):
    return "".join(str(s).casefold().split())


def freeze(out):
    assert not (out / "protocol.json").exists()
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    bundle = RagIndexStore(MANIFEST.parent).active()
    assert bundle.bundle_id == read(PREVIOUS / "protocol.json")["bundle_id"]
    fields = asdict(settings)
    effective = {k: v for k, v in fields.items() if k.startswith("rag_")
                 and not any(x in k for x in ("secret", "api_base_url"))}
    inputs = [SETTINGS_PATH, MANIFEST, bundle.metadata_path, GOLD,
              GOLD.with_name("rag_gold_v4_2.jsonl"), SOURCE,
              ROOT / "local_full_text_search/rag/constraint_matcher.py",
              ROOT / "local_full_text_search/rag/query_analyzer.py",
              ROOT / "local_full_text_search/rag/query_interpreter.py",
              ROOT / "local_full_text_search/rag/evidence_policy.py",
              ROOT / "local_full_text_search/rag/context_builder.py",
              PREVIOUS / "baseline_rag-biz-040.json", PREVIOUS / "candidate_rag-biz-040.json",
              PREVIOUS / "branch_plans.json"]
    hashes = {str(p): sha(p) for p in inputs}
    write(out / "protocol.json", {"time": datetime.now().isoformat(), "bundle_id": bundle.bundle_id,
          "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
          "settings": effective, "input_sha256": hashes,
          "arms": {"A": "native", "O": "keep rerank order; original constraints",
                   "S": "separate branches; original sort", "C": "separate branches; keep rerank order"},
          "gate": "Both source-bound 040 quotes must occur in actual included_text of C on fixed R0 pool; true conflicts remain excluded.",
          "stop": "If fixed-pool C fails, do not expand retrieval, run answer generation, or claim adoption."})
    (out / "working_tree_at_start.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True, encoding="utf8"), encoding="utf8")
    write(out / "dataset_manifest.json", {
        "main": str(GOLD), "formal_reviewed": 48,
        "disputed": ["rag-biz-007"], "pending": ["rag-biz-024"],
        "development_ids": ["rag-biz-040", "rag-biz-020", "rag-biz-022", "rag-biz-028"],
        "heldout_rule": "Only if fixed-pool gate passes: 6 triggering, 4 confusable negative, 2 single; distinct business groups and >=3 sources; freeze before running.",
    })
    (out / "execution_log.md").write_text("# Execution log\n\n- Stage 0 passed: published bundle and frozen inputs recorded.\n", encoding="utf8")
    return hashes


def verify(out):
    protocol = read(out / "protocol.json")
    assert all(sha(p) == h for p, h in protocol["input_sha256"].items()), "Frozen input changed"
    assert RagIndexStore(MANIFEST.parent).active().bundle_id == protocol["bundle_id"]


def fixed_pool(store, bundle, snapshot):
    reranked = next(event["reranked_candidates"] for event in snapshot["events"] if event["stage"] == "reranked")
    with store.connect_read_only(bundle.metadata_path) as conn:
        items = []
        for row in reranked:
            chunk = store.load_chunk(conn, row["chunk_id"])
            assert chunk is not None and chunk.text == row["text"]
            member_ids = tuple(row.get("member_chunk_ids") or ())
            sources = tuple(store.load_chunk(conn, member_id).source for member_id in member_ids)
            item = EvidenceChunk(chunk=chunk, lexical_rank=row["lexical_rank"], lexical_score=row["lexical_score"],
                                 vector_rank=row["vector_rank"], vector_score=row["vector_score"],
                                 fused_rank=row["fused_rank"], fused_score=row["fused_score"],
                                 rerank_score=row["rerank_score"], rerank_score_combined=row["rerank_score_combined"],
                                 answer_support_score=row["answer_support_score"],
                                 member_chunk_ids=member_ids, sources=sources)
            items.append(item)
        return items


def run(out):
    verify(out)
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    store = RagIndexStore(MANIFEST.parent)
    bundle = store.active()
    case = next(c for c in read(GOLD)["cases"] if c["id"] == "rag-biz-040")
    plan = QueryInterpreter().interpret(case["question"])
    query = RagQuery(case["question"], final_top_k=settings.rag_final_top_k)
    previous = read(PREVIOUS / "branch_plans.json")[case["question"]]
    branches = tuple(b["query_text"] for b in previous["branches"])
    policy = EvidencePolicy(min_vector_score=settings.rag_min_vector_score,
                            min_rerank_score=settings.rag_min_rerank_score,
                            max_per_document=settings.rag_max_chunks_per_document)
    rows = []
    bindings = []
    for pool, filename in (("A", "baseline_rag-biz-040.json"), ("R0", "candidate_rag-biz-040.json")):
        snapshot = read(PREVIOUS / filename)
        original_ids = [r["chunk_id"] for e in snapshot["events"] if e["stage"] == "checking_constraints"
                        for r in e["constraint_candidates"]]
        original_selected = [e["chunk_id"] for e in snapshot["entries"]]
        for arm in ("A", "O", "S", "C"):
            candidates = fixed_pool(store, bundle, snapshot)
            matcher = ScopedApplicability(branches if arm in {"S", "C"} else (), mode=arm if arm != "A" else "S")
            # Native A uses the original plan and the original sort.
            if arm == "A":
                matcher = ScopedApplicability((), mode="S")
            ordered = matcher.evaluate(plan, candidates)
            ranked = [item.chunk.stable_chunk_id for item in ordered]
            if arm == "A":
                assert ranked == original_ids, "Cannot reproduce native condition order"
            decision = policy.evaluate(query, ordered, plan=matcher.policy_plan(plan))
            context = ContextBuilder().build(decision.accepted, max_chars=settings.rag_context_max_chars)
            selected = [entry["chunk_id"] for entry in context.entries]
            if arm == "A":
                assert selected == original_selected, "Cannot reproduce native final sources"
                assert context.text == snapshot["context"], "Cannot reproduce native actual context"
            cover = {str(page): any(entry["chunk_id"] == cid and norm(evidence["quote"]) in norm(entry["included_text"])
                                    for entry in context.entries)
                     for page, cid in TARGETS.items()
                     for evidence in case["gold_evidence"] if evidence["page"] == page}
            rows.append({"pool": pool, "arm": arm, "ranked_ids": ranked,
                         "target_ranks": {str(page): ranked.index(cid) + 1 if cid in ranked else None
                                          for page, cid in TARGETS.items()},
                         "selected_ids": selected, "quote_bound_in_context": cover,
                         "decision_reason": decision.reason,
                         "rejected": decision.rejected,
                         "context_chars": len(context.text),
                         "context_entries": context.entries})
            bindings.append({"pool": pool, "arm": arm, "bindings": matcher.last_bindings})
    with (out / "fixed_pool_ablation.jsonl").open("w", encoding="utf8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    with (out / "condition_bindings.jsonl").open("w", encoding="utf8") as stream:
        for row in bindings:
            stream.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    candidate = next(r for r in rows if r["pool"] == "R0" and r["arm"] == "C")
    gate = all(candidate["quote_bound_in_context"].values())
    write(out / "stage1_gate.json", {"passed": gate, "R0_C": {
        "target_ranks": candidate["target_ranks"], "quote_bound_in_context": candidate["quote_bound_in_context"],
        "selected_ids": candidate["selected_ids"]},
        "rule": "Exact target quotes present in actual source-bound context within unchanged budget"})
    summary = ["# Fixed-pool condition ablation", "", "The pool and rerank scores were fixed; no retrieval or answer-model calls were made.", "",
               "| Pool | Arm | Slide 14 rank | Slide 16 rank | Bound quotes |", "|---|---|---:|---:|---|" ]
    for row in rows:
        summary.append(f"| {row['pool']} | {row['arm']} | {row['target_ranks']['14']} | {row['target_ranks']['16']} | {row['quote_bound_in_context']} |")
    summary.append(f"\nStatic gate: {'PASS' if gate else 'FAIL'}.\n")
    (out / "evidence_changes.md").write_text("\n".join(summary), encoding="utf8")
    with (out / "execution_log.md").open("a", encoding="utf8") as stream:
        stream.write(f"- Stage 1 {'passed' if gate else 'failed'}: fixed R0 pool C quote coverage {candidate['quote_bound_in_context']}.\n")
    verify(out)
    print("stage1 gate", gate, candidate["target_ranks"], candidate["quote_bound_in_context"], flush=True)


def plan_branches(out):
    verify(out)
    assert read(out / "stage1_gate.json")["passed"]
    target = out / "R1_plan.json"
    assert not target.exists(), "Planner output is frozen; start another run for a new version"
    settings = AppSettings.from_dict(read(SETTINGS_PATH))
    key = WindowsCredentialSecretStore().retrieve(settings.rag_api_secret_ref)
    assert key, "Saved model credential unavailable"
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    question = next(c["question"] for c in read(GOLD)["cases"] if c["id"] == "rag-biz-040")
    def complete(prompt, text):
        response = gateway.complete_text(TextCompletionRequest(
            system_prompt=prompt, user_prompt=text, model=settings.rag_api_model,
            api_base_url=settings.rag_api_base_url, api_style=settings.rag_api_style,
            api_key=key, timeout_seconds=settings.rag_api_timeout_seconds,
            max_output_tokens=1100), CancelToken())
        return response if isinstance(response, str) else response.text
    branches, audit = plan_scoped(question, complete)
    write(target, {"question": question, "planner_prompt_sha256": __import__("hashlib").sha256(PROMPT.encode()).hexdigest(),
                   "model": settings.rag_api_model, "audit": audit})
    with (out / "execution_log.md").open("a", encoding="utf8") as stream:
        stream.write(f"- Stage 2 planner: {'passed' if branches and branches[1].inherited else 'failed'}; one planning request.\n")
    print("R1 planner", audit["reason"], [b.query_text for b in branches], flush=True)


def static(out):
    verify(out)
    frozen = read(out / "R1_validated_plan.json")
    assert frozen["audit"]["triggered"] and frozen["audit"]["branches"][1]["inherited"], "R1 lacks explicit inherited metric"
    from tools.validate_branch_retrieval import Runtime, evidence_checks
    from tools.branch_retrieval_candidate import Branch, retrieve_with_branches
    from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
    from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
    from local_full_text_search.rag.evidence_policy import EvidencePolicy as NativeEvidencePolicy
    from local_full_text_search.rag.trace import candidate_snapshot
    exp = Runtime(out)
    case = exp.cases["rag-biz-040"]
    assert case.question == frozen["question"]
    question_to_branches = {case.question: tuple(b["query_text"] for b in frozen["audit"]["branches"])}
    class ExperimentHybrid(HybridRetriever):
        def retrieve(self, query, token, plan=None):
            original = super().retrieve(query, token, plan)
            if exp.tls.arm not in {"R", "RC"} or query.text not in question_to_branches:
                return original
            branches = tuple(Branch(b["branch_id"], b["condition_quote"], b["query_text"])
                             for b in frozen["audit"]["branches"])
            native = HybridRetriever(self.lexical, self.vector, rrf_k=self.rrf_k)
            merged, audit = retrieve_with_branches(query, original, branches,
                native.retrieve, self.query_interpreter.interpret, token, rrf_k=self.rrf_k)
            if self.progress_callback:
                self.progress_callback("branch_recall", "分支召回实验", branch_audit=audit,
                                       branch_fused_candidates=[candidate_snapshot(x) for x in merged])
            return merged
    class ExperimentMatcher(ConstraintMatcher):
        def evaluate(self, plan, candidates):
            if exp.tls.arm not in {"C", "RC"} or plan.original_text not in question_to_branches:
                return super().evaluate(plan, candidates)
            candidate = ScopedApplicability(question_to_branches[plan.original_text], mode="C")
            result = candidate.evaluate(plan, list(candidates))
            exp.tls.bindings = candidate.last_bindings
            return result
    class ExperimentPolicy(NativeEvidencePolicy):
        def evaluate(self, query, candidates, *, plan=None):
            if exp.tls.arm in {"C", "RC"} and plan and plan.original_text in question_to_branches:
                plan = ScopedApplicability(question_to_branches[plan.original_text], mode="C").policy_plan(plan)
            return super().evaluate(query, candidates, plan=plan)
    results = []
    protected = []
    with patch("local_full_text_search.rag.answer_service.HybridRetriever", ExperimentHybrid), \
         patch("local_full_text_search.rag.answer_service.ConstraintMatcher", ExperimentMatcher), \
         patch("local_full_text_search.rag.answer_service.EvidencePolicy", ExperimentPolicy):
        for arm in ("A", "R", "C", "RC"):
            exp.tls.arm = arm
            data = exp.native_full(case, arm=arm)
            data["condition_bindings"] = getattr(exp.tls, "bindings", []) if arm in {"C", "RC"} else []
            write(out / f"native_{arm}_040.json", data)
            found = evidence_checks(case, data, exp.chunk_rows)
            results.append({"arm": arm, "status": data["status"],
                            "quote_coverage": {str(x["page"]): x["bound_in_context"] for x in found},
                            "sources": [{"filename": e["filename"], "location": e["location"],
                                         "chunk_id": e["chunk_id"]} for e in data["entries"]],
                            "context_chars": len(data["context"])})
            print("static", arm, results[-1]["quote_coverage"], flush=True)
        for case_id in ("rag-biz-020", "rag-biz-022", "rag-biz-028"):
            item = exp.cases[case_id]
            before = exp.native_full(item, arm="A")
            after = exp.native_full(item, arm="RC")
            same = before["context"] == after["context"] and before["entries"] == after["entries"]
            assert same, f"Protected context changed: {case_id}"
            protected.append({"id": case_id, "triggered": False,
                              "same_context_and_sources": same,
                              "entry_count": len(after["entries"])})
            print("protected", case_id, same, flush=True)
    assert read(out / "native_A_040.json")["context"] == read(PREVIOUS / "baseline_rag-biz-040.json")["context"]
    success = next(r for r in results if r["arm"] == "RC")
    gate = (success["status"] == "answered" and all(success["quote_coverage"].values())
            and len(success["sources"]) <= settings_from_file().rag_final_top_k
            and success["context_chars"] <= settings_from_file().rag_context_max_chars)
    write(out / "four_arm_static_040.json", {"passed": gate, "arms": results})
    write(out / "protected_static.json", {"cases": protected,
          "meaning": "Untriggered-path preservation only; not evidence of generalization for conditional questions."})
    with (out / "execution_log.md").open("a", encoding="utf8") as stream:
        stream.write(f"- Stage 2 four-arm native static gate {'passed' if gate else 'failed'}; all contexts saved.\n")
    verify(out)
    print("four-arm gate", gate, flush=True)


def validate_plan(out):
    verify(out)
    original = read(out / "R1_plan.json")
    result = validate_scoped(original["question"], original["audit"]["raw"])
    assert result and result[1].inherited
    audit = {"triggered": True, "reason": "revalidated_with_condition_in_query",
             "branches": [{"branch_id": b.branch_id, "condition_quote": b.condition_quote,
                           "query_text": b.query_text, "inherited": list(b.inherited)} for b in result]}
    write(out / "R1_validated_plan.json", {"question": original["question"], "audit": audit,
          "original_output": "R1_plan.json", "planner_calls_added": 0,
          "validator_sha256": sha(ROOT / "tools/scoped_branch_planner.py")})
    with (out / "execution_log.md").open("a", encoding="utf8") as stream:
        stream.write("- Stage 2 correction: the first generated query dropped its condition. Validator now always includes the exact condition; revalidated saved raw plan without a second model request.\n")
    print("R1 revalidated", [b.query_text for b in result], flush=True)


def settings_from_file():
    return AppSettings.from_dict(read(SETTINGS_PATH))


def finish(out):
    verify(out)
    prior = read(PREVIOUS / "source_audit.json")
    assert all(all(e["in_original"] and e["in_index"] for e in page["gold_quote_checks"])
               for page in prior), "Previous source audit did not verify each declared quote"
    fixed = [json.loads(line) for line in (out / "fixed_pool_ablation.jsonl").read_text(encoding="utf8").splitlines()]
    static_result = read(out / "four_arm_static_040.json")
    protected = read(out / "protected_static.json")
    assert not static_result["passed"]
    assert all(row["same_context_and_sources"] for row in protected["cases"])
    root = ET.parse(out / "pytest_full.xml").getroot()
    suites = list(root) if root.tag == "testsuites" else [root]
    tests = {name: sum(int(suite.get(name, 0)) for suite in suites)
             for name in ("tests", "failures", "errors", "skipped")}
    assert tests["tests"] == 63 and tests["failures"] == tests["errors"] == 0
    stage1 = read(out / "stage1_gate.json")
    assert stage1["passed"]
    r0_c = next(row for row in fixed if row["pool"] == "R0" and row["arm"] == "C")
    assert all(r0_c["quote_bound_in_context"].values())
    model_plan = read(out / "R1_plan.json")
    validated = read(out / "R1_validated_plan.json")
    assert validated["planner_calls_added"] == 0
    code_paths = [ROOT / "tools/scoped_applicability_candidate.py",
                  ROOT / "tools/scoped_branch_planner.py",
                  ROOT / "tools/validate_scoped_applicability.py",
                  ROOT / "tests/rag/test_scoped_applicability_candidate.py"]
    write(out / "answer_run_status.json", {
        "answer_generations": 0, "answer_judges": 0,
        "branch_planner_calls": 1,
        "reason": "Four-arm native static gate failed: slide 16 absent from actual context.",
        "later_stages": {"development_answer_AB": "not_applicable",
                         "new_heldout_questions": "not_applicable",
                         "full_48_42_answer_regression": "not_applicable",
                         "production_integration": "not_applicable"},
        "no_answer_accuracy_claim": True,
    })
    write(out / "end_manifest.json", {
        "finished": datetime.now().isoformat(), "decision": "DO_NOT_ADOPT",
        "bundle_id": read(out / "protocol.json")["bundle_id"],
        "frozen_inputs_unchanged": True,
        "input_sha256": read(out / "protocol.json")["input_sha256"],
        "candidate_code_sha256": {str(p): sha(p) for p in code_paths},
        "previous_source_audit_sha256": sha(PREVIOUS / "source_audit.json"),
        "tests": tests, "fixed_pool_gate": stage1["passed"],
        "native_four_arm_gate": static_result["passed"],
        "production_default_changed": False,
    })
    report = f"""# 条件作用域与分支召回：开发验证结果

日期：2026-11-10。**结论：本轮候选不采用，040 仍未完整修复。**

固定候选池中，条件层候选让两段正确资料都进入实际上下文；接入真实检索链路后，新分支查询虽然找到第 16 张，却在融合与去重之间丢失。因此不能把固定候选池的通过说成真实问答的改善。

## 1. 来源、范围与版本

- 发布 AI 索引：`{read(out / 'protocol.json')['bundle_id']}`，原资料和索引未改。不是普通全文 generation。
- Gold：`tests/rag/fixtures/rag_gold_v4_2_review.json`，正式 48 reviewed；007 disputed、024 pending 均未修改或混入正式计分。
- 原资料：`{SOURCE}` 第 14、15、16 张。沿用上轮的 [`source_audit.json`](../rag_040_retrieval_20261110_01/source_audit.json)：两条正向引文和第 15 张强负例均在原始幻灯片及当前 AI 索引正文中复核通过。
- 基线与上轮有效原生 trace 逐字复现。输入哈希、Git HEAD、设置、历史未提交工作区状态在 `protocol.json` 和 `working_tree_at_start.txt` 冻结，结束时全部输入哈希再次核对一致。
- 本轮新增行为只有 `tools/scoped_applicability_candidate.py` 和 `tools/scoped_branch_planner.py` 中的隔离候选。生产算法、Gold、模型、提示、索引和应用默认设置未修改。

## 2. 固定候选池消融：能保住证据，但不足以采用

从上轮真实自动 trace 重建 A 与 R0 的模型重排结果。重建时恢复去重合并的全部来源，因此 A 的条件排序、选中 ID 和实际上下文与原 trace 逐字一致。每一臂使用新对象，冻结模型重排得分和顺序；没有再次调用回答模型。

| 固定候选池 | 条件方式 | 第 14/16 张条件后位置 | 两条完整引文进入实际正文 | 说明 |
|---|---|---|---|---|
| A 原始召回 | 原条件层 | 7 / 缺席 | 1/2 | 只能回答第一分支 |
| A 原始召回 | C 组合条件层 | 1 / 缺席 | 1/2 | 排序改善，无法创造未召回证据 |
| R0 旧分支召回 | 原条件层 | 12 / 13 | 0/2 | 两段都被最终 8 段截掉 |
| R0 旧分支召回 | O 只保留重排顺序 | 1 / 2 | 0/2 | 整题 `scope=all` 最终门槛清空上下文 |
| R0 旧分支召回 | S 只分开条件作用域 | 12 / 13 | 0/2 | 仍受条件排序影响 |
| R0 旧分支召回 | C 作用域加排序 | 1 / 2 | **2/2** | 第 14/16 张和原文均在来源绑定的实际正文中 |

C 同时改变条件解释、条件后的排序，以及最终证据门槛使用的条件范围；这一组合的完整效益不能单独归功于排序。O 和 S 隔离结果分别指出，两处门槛都必须处理。第 14/16 张在 R0 固定池里第 1/2，不涉及扩大 8 段、单文件 3 段或 12,000 字预算。

这一步只证明**已有正确候选时** C 可以保住它们。第 14/16 张同时进入上下文并不保证其余六段都适用；这一风险需要触发后的负例和真实回答验证。

## 3. 新查询规划与四臂真实链路

R1 重新要求对后半句省略的指标明确给出继承来源。模型实际输出一次，第一次输出的第一分支 `query_text` 错误地只含“优先查哪里？”。校验器于是增加“每个查询必须包含其条件原文”；用**同一份保存的原始输出**重新验证，没有为了挑选好答案再调用模型。原始错误、修正后文本、规划提示哈希分别在 `R1_plan.json`、`R1_validated_plan.json`，不能把初次规划写成直接成功。

本次两个实际独立查询：

1. `ALB、GLU 等小样本量项目重复性变差，ALT、AST 较稳`
2. `如果所有项目都变差呢？ 重复性`

后者补出了原问中的“重复性”，但也改变了查询外壳。未单独消融这两个成分，不能断言是哪一个导致排名变化。

在真实 AnswerService 检索与选文链路上，四臂结果：

| 臂 | 召回 | 条件层 | 第 14 张实际正文 | 第 16 张实际正文 |
|---|---|---|---|---|
| A | 当前 | 当前 | 有 | 无 |
| R | R1 | 当前 | 有 | 无 |
| C | 当前 | C | 有 | 无 |
| R+C | R1 | C | 有 | 无 |

R1 的第二分支确实召回第 16 张；它只在词法通道排第 16，向量前 50 未命中，合并后排第 60。去重后不在 30 个候选内，未进入重排，C 没有机会判断它。旧 R0 第二分支是“所有项目都变差”，曾让第 16 张进入重排。这是**本轮 R1 新查询与固定预算共同造成的最早失败点**；不能据此说第 16 张已进入最终模型上下文。

040 的两条正向引文在四臂中的实际覆盖分别都是 1/2。此处是上下文覆盖，不是回答正确率。所有实际来源、片段、条件判断与截断位置见 `native_A_040.json`、`native_R_040.json`、`native_C_040.json`、`native_RC_040.json` 和 `four_arm_static_040.json`。

## 4. 保护、测试与未执行阶段

020、022、028 的 A 与 R+C 上下文及来源逐字相同，详见 `protected_static.json`。它们未触发新分支机制，只证明未触发路径未变化，不能证明跨业务泛化。

隔离候选和现有条件安全/端到端相关测试 **{tests['tests']} 通过，0 失败，0 错误**，结果在 `pytest_full.xml`。新测试包括真实冲突仍排除、UNKNOWN 保持未知、局部条件不硬化为整题、缺省指标须明确继承。测试通过不覆盖 R1 的实际召回失败。

计划步骤 0 与 1 通过；步骤 2 的四臂静态门槛失败。按预先冻结的停止条件，步骤 3 的回答 A/B、步骤 4 的新独立验收与固定 48+42 回归回答、步骤 5 的生产集成均**不适用**：没有足够正确证据可交给模型。实际回答生成 0 次、答案裁判 0 次，分支规划 1 次。`answer_run_status.json` 明确记录这些数量，没有虚构回答一致率。

本轮没有制作新的 pending 验收题，因为候选在先决的静态证据门槛已失败；旧独立题也未被冒充新验收。用户审核状态未变。

## 5. 共同原因、层次与采用决定

上一轮的阻塞是整题条件误用于两个分支，以及 UNKNOWN/表面 MATCH 的排序覆盖模型相关性。本轮固定池验证支持**条件作用域和最终条件门槛**确实是下游阻塞之一。真实链路又证明：即使修好下游，完整查询 R1 在当前融合/去重预算下仍会把第二分支证据放到重排入口外。两层都存在，不能把 040 简化成单一条件判断问题。

本轮原型仅改实验层条件匹配和分支查询验证；没有改生产层。040 在实际链路中未改善，未触发题没有退步，但不足以证明安全或有效。因此结论是 **DO_NOT_ADOPT**，当前应用继续使用原行为。

后续如果重新开发，应先在未使用 Gold 答案的条件下，设计**分支覆盖在融合候选中的保留规则**，保证每个明确条件的候选有进入同一重排预算的机会，同时保留原问候选和真实冲突过滤。必须对不同资料、不同条件的触发题与错对象负例重新冻结验证；不能为 040 固定加入第 16 张，不能把“全部项目”词项匹配当成适用性证明。

## 6. 复查文件

- `protocol.json`、`dataset_manifest.json`、`end_manifest.json`：冻结与结束校验。
- `fixed_pool_ablation.jsonl`、`condition_bindings.jsonl`、`evidence_changes.md`：O/S/C 局部因果对照。
- `R1_plan.json`、`R1_validated_plan.json`：初次规划与复核后的真实查询。
- `native_*_040.json`、`four_arm_static_040.json`：四臂原生检索、实际正文和来源。
- `protected_static.json`、`pytest_full.xml`、`answer_run_status.json`：保护题、自动化检查及未运行阶段。
"""
    (out / "report.md").write_text(report, encoding="utf8")
    with (out / "execution_log.md").open("a", encoding="utf8") as stream:
        stream.write("- Stage 2 failed: R1 recalled slide 16, fused rank 60, not in deduplicated top 30.\n"
                     "- Stages 3-5 not applicable under the predeclared static gate.\n"
                     "- Final decision DO_NOT_ADOPT; input hashes verified.\n")
    verify(out)
    print("DO_NOT_ADOPT; fixed pool passed, native gate failed; 63 tests passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "ablate", "plan", "validate-plan", "static", "finish"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"freeze": freeze, "ablate": run, "plan": plan_branches,
     "validate-plan": validate_plan, "static": static, "finish": finish}[args.stage](args.out)


if __name__ == "__main__":
    main()
