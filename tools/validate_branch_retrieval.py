"""Opt-in, source-bound branch recall experiment. Never changes application defaults."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import posixpath
import subprocess
from threading import local
from unittest.mock import patch
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from local_full_text_search.config.constants import SETTINGS_PATH
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.answer_service import AnswerService
from local_full_text_search.rag.evaluation.gold_loader import load_gold_dataset
from local_full_text_search.rag.evaluation.runner import _answer_record
from local_full_text_search.rag.generation.openai_compatible import OpenAICompatibleGateway
from local_full_text_search.rag.generation.base import TextCompletionRequest
from local_full_text_search.rag.models import RagQuery
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.trace import candidate_snapshot
from tools.rag_eval_common import ROOT, read, lines, write, sha
from tools.validate_evidence_answer_flow import Experiment

GOLD = ROOT / "tests/rag/fixtures/rag_gold_v4_2_review.json"
SOURCE = Path("E:/生化免疫临床知识/生化试剂应用及常见问题-SH-20190110.pptx")
IDS = ("rag-biz-040", "rag-biz-020", "rag-biz-022", "rag-biz-028")
OLD = ROOT / "reports/rag_evidence_answer_native_recheck_20261109/e2e_runs.jsonl"


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def canonical(text):
    return "".join(str(text).split()).casefold()


def append(path, value):
    with path.open("a", encoding="utf8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")


class Runtime(Experiment):
    def __init__(self, out):
        super().__init__(out, 4)
        self.tls = local()
        self.cases = {c.case_id: c for c in load_gold_dataset(GOLD, reviewed_only=False).cases}
        assert all(self.cases[cid].review_status == "reviewed" for cid in IDS)
        assert all(self.cases[cid].corpus_snapshot == self.bundle.bundle_id for cid in IDS)

    def query(self, question):
        s = self.settings
        return RagQuery(text=question, include_ocr=True, min_ocr_confidence=s.ocr_min_confidence,
                        lexical_top_k=s.rag_lexical_top_k, vector_top_k=s.rag_vector_top_k,
                        fusion_top_k=s.rag_fusion_top_k, reranker_top_k=s.rag_reranker_top_k,
                        final_top_k=s.rag_final_top_k)

    def native_full(self, case, *, arm="A", retrieve=True):
        self.tls.arm = arm
        data = {"id": case.case_id, "question": case.question, "arm": arm,
                "events": [], "context": "", "entries": []}
        def progress(event):
            data["events"].append(copy.deepcopy(event))
            if "llm_context" in event:
                data.update(context=event["llm_context"], entries=event.get("context_entries", []))
        service = AnswerService(self.source_store, self.store, self.embedding, self.ranker,
                                OpenAICompatibleGateway(max_transient_retries=2), self.secrets,
                                self.settings, progress_callback=progress)
        result = (service.retrieve if retrieve else service.answer)(self.query(case.question), CancelToken())
        data.update(status=result.status.value, reason=getattr(result, "reason", ""))
        if not retrieve:
            data["answer"] = _answer_record(result)
        return data


def input_hashes(exp):
    paths = [GOLD, GOLD.with_name("rag_gold_v4_2.jsonl"), SETTINGS_PATH, SOURCE,
             exp.store.manifest_path, exp.bundle.metadata_path, OLD,
             ROOT / "tools/validate_evidence_answer_flow.py",
             ROOT / "docs/rag_040_retrieval_development_validation_plan_20261110.md"]
    paths += list((ROOT / "local_full_text_search/rag").rglob("*.py"))
    paths += [ROOT / "local_full_text_search/config/defaults.py"]
    return {str(p): sha(p) for p in paths}


def verify_frozen(exp):
    protocol = read(exp.out / "protocol.json")
    assert protocol["bundle_id"] == exp.bundle.bundle_id
    assert protocol["input_hashes"] == input_hashes(exp), "Frozen input changed; stop comparison"


def freeze(exp):
    assert not (exp.out / "protocol.json").exists(), "Use a new run directory"
    raw = asdict(exp.settings)
    effective = {k: v for k, v in raw.items() if k.startswith("rag_")
                 and not any(w in k for w in ("secret", "api_base_url"))}
    write(exp.out / "protocol.json", {
        "started": datetime.now().isoformat(), "bundle_id": exp.bundle.bundle_id,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "effective_settings": effective, "input_hashes": input_hashes(exp),
        "scope": list(IDS), "repeats": 3, "arms": {"A": "unchanged production", "B": "question-only branch recall"},
        "controls": "No Gold changes, no source/index changes, no downstream algorithm changes; default off.",
        "candidate_status": "not implemented; diagnosis first", "harness_sha_at_freeze": sha(__file__),
    })
    (exp.out / "worktree_at_start.txt").write_text(
        subprocess.check_output(["git", "status", "--short"], text=True, encoding="utf8"), encoding="utf8")
    with ZipFile(SOURCE) as archive:
        sources = []
        for page in (14, 15, 16):
            tree = ET.fromstring(archive.read(f"ppt/slides/slide{page}.xml"))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            # Read text runs only; do not conflate XML formatting attributes with source text.
            paragraphs = ["".join(t.text or "" for t in p.findall(".//a:t", ns))
                          for p in tree.findall(".//a:p", ns)]
            original = "\n".join(p for p in paragraphs if p.strip())
            chunks = [r for r in exp.chunk_rows
                      if Path(r["file_path"]).name == SOURCE.name and r.get("slide_number") == page]
            assert chunks, f"Missing indexed source slide {page}"
            sources.append({"file": str(SOURCE), "slide_1based": page, "original_text": original,
                            "chunks": chunks, "indexed_text_present_in_original": all(
                                canonical(r["text"]) in canonical(original) for r in chunks)})
    write(exp.out / "source_audit.json", sources)
    old = {r["id"]: r["arms"]["A"] for r in lines(OLD) if r["repeat"] == 1 and r["id"] in IDS}
    for cid in IDS:
        data = exp.native_full(exp.cases[cid])
        write(exp.out / f"baseline_{cid}.json", data)
        assert not data.get("error") and data["context"], f"No usable baseline {cid}"
        assert data["context"] == old[cid]["context"], f"Cannot reproduce historical context for {cid}"
        print("reproduced", cid, len(data["entries"]), len(data["context"]), flush=True)
    verify_frozen(exp)


def diagnose(exp):
    verify_frozen(exp)
    audit = read(exp.out / "source_audit.json")
    targets = {r["chunk_id"]: page["slide_1based"] for page in audit for r in page["chunks"]}
    captured = {}
    class Probe(HybridRetriever):
        def retrieve(self, query, token, plan=None):
            base = super().retrieve(query, token, plan)
            deep = replace(query, lexical_top_k=1000, vector_top_k=1000)
            for kind, retriever in (("lexical", self.lexical), ("vector", self.vector)):
                hits = retriever.retrieve(deep, token, plan)
                captured[kind] = {str(page): [candidate_snapshot(x) for x in hits
                                             if x.chunk.stable_chunk_id in targets and targets[x.chunk.stable_chunk_id] == page]
                                  for page in (14, 15, 16)}
                captured[kind + "_count"] = len(hits)
            return base
    with patch("local_full_text_search.rag.answer_service.HybridRetriever", Probe):
        trace = exp.native_full(exp.cases[IDS[0]])
    write(exp.out / "diagnostic_trace.json", trace)
    write(exp.out / "extended_ranks.json", captured)
    assert trace["context"] == read(exp.out / f"baseline_{IDS[0]}.json")["context"]
    verify_frozen(exp)
    for kind in ("lexical", "vector"):
        print(kind, {p: [(r[kind + "_rank"], round(r[kind + "_score"], 5)) for r in rs]
                     for p, rs in captured[kind].items()}, flush=True)

    source_checks(exp)


def source_checks(exp):
    audit = read(exp.out / "source_audit.json")
    for page in audit:
        number = page["slide_1based"]
        with ZipFile(SOURCE) as archive:
            relations = ET.fromstring(archive.read(f"ppt/slides/_rels/slide{number}.xml.rels"))
            note_part = next(posixpath.normpath(posixpath.join("ppt/slides", r.get("Target")))
                             for r in relations if r.get("Type", "").endswith("/notesSlide"))
            notes = ET.fromstring(archive.read(note_part))
            ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
            notes_text = "\n".join("".join(t.text or "" for t in p.findall(".//a:t", ns))
                                   for p in notes.findall(".//a:p", ns)).strip()
        page["original_notes_part"] = note_part
        page["original_notes_text"] = notes_text
        page["original_text_with_notes"] = page["original_text"] + "\n" + notes_text
        page["indexed_text_present_in_original_with_notes"] = all(
            canonical(r["text"]) in canonical(page["original_text_with_notes"]) for r in page["chunks"])
        page["indexed_body_present_in_original"] = all(
            canonical(r["text"].removesuffix("\n" + str(page["slide_1based"])))
            in canonical(page["original_text"]) for r in page["chunks"])
        page["footer_difference"] = "Trailing page number exists in linked notesSlide XML; index combines slide body and its notes."
        page["gold_quote_checks"] = [
            {"quote": e["quote"], "in_original": canonical(e["quote"]) in canonical(page["original_text"]),
             "in_index": any(canonical(e["quote"]) in canonical(r["text"]) for r in page["chunks"])}
            for e in exp.cases[IDS[0]].gold_evidence + exp.cases[IDS[0]].hard_negatives
            if e.get("page") == page["slide_1based"]]
        assert page["indexed_body_present_in_original"]
        assert page["indexed_text_present_in_original_with_notes"]
        assert all(e["in_original"] and e["in_index"] for e in page["gold_quote_checks"])
    write(exp.out / "source_audit.json", audit)


def branch_factory(exp, plans):
    from tools.branch_retrieval_candidate import Branch, retrieve_with_branches
    class CandidateHybrid(HybridRetriever):
        def retrieve(self, query, token, plan=None):
            original = super().retrieve(query, token, plan)
            if getattr(exp.tls, "arm", "A") != "B":
                return original
            frozen_plan = plans.get(query.text, {})
            branches = tuple(Branch(**r) for r in frozen_plan.get("branches", []))
            # A separate native hybrid keeps branch events apart from the original trace.
            native = HybridRetriever(self.lexical, self.vector, rrf_k=self.rrf_k)
            result, audit = retrieve_with_branches(
                query, original, branches, native.retrieve, self.query_interpreter.interpret,
                token, rrf_k=self.rrf_k)
            if self.progress_callback:
                self.progress_callback("branch_recall", "独立条件分支召回实验",
                                       branch_audit=audit,
                                       branch_fused_candidates=[candidate_snapshot(x) for x in result])
            return result
    return CandidateHybrid


def evidence_checks(case, trace, index_rows):
    by_id = {r["chunk_id"]: r for r in index_rows}
    rows = []
    for e in case.gold_evidence:
        hits = []
        for entry in trace["entries"]:
            for source in entry.get("sources", []):
                row = by_id.get(source.get("chunk_id"), {})
                source_file = row.get("file_path", "")
                source_page = row.get("slide_number") or row.get("page_number")
                if (str(source_file).casefold() == str(e["file"]).casefold()
                        and str(source_page) == str(e["page"])
                        and source.get("filename") == row.get("filename")
                        and canonical(e["quote"]) in canonical(entry.get("included_text", ""))
                        and canonical(e["quote"]) in canonical(row.get("text", ""))):
                    hits.append(entry["source_id"])
        rows.append({"page": e["page"], "file": e["file"], "quote": e["quote"],
                     "bound_in_context": bool(hits), "source_ids": hits})
    return rows


def candidate(exp):
    from tools.branch_retrieval_candidate import plan_branches, PLANNER_PROMPT
    verify_frozen(exp)
    source_checks(exp)
    plan_path = exp.out / "branch_plans.json"
    assert not plan_path.exists(), "Plans are immutable; use static stage to replay"
    gateway = OpenAICompatibleGateway(max_transient_retries=2)
    def complete(system, question):
        s = exp.settings
        response = gateway.complete_text(TextCompletionRequest(
            system_prompt=system, user_prompt=question, model=s.rag_api_model,
            api_base_url=s.rag_api_base_url, api_style=s.rag_api_style, api_key=exp.key,
            timeout_seconds=s.rag_api_timeout_seconds, max_output_tokens=1200), CancelToken())
        return response if isinstance(response, str) else response.text
    plans = {}
    for cid in IDS:
        question = exp.cases[cid].question
        _, audit = plan_branches(question, complete)
        plans[question] = audit
        print("plan", cid, audit["triggered"], audit["fallback"], flush=True)
    write(plan_path, plans)
    write(exp.out / "candidate_spec.json", {
        "candidate_sha": sha(ROOT / "tools/branch_retrieval_candidate.py"),
        "planner_prompt_sha": digest(PLANNER_PROMPT), "planner_prompt": PLANNER_PROMPT,
        "mechanism": "Question-only extractive condition fragments; best rank per channel across original and branch routes, same 2-channel RRF; no repeated-route votes.",
        "scope": "Explicit follow-on conditional clauses only; other phrasing falls back.",
        "budgets": {"per_query_lexical": exp.settings.rag_lexical_top_k,
                    "per_query_vector": exp.settings.rag_vector_top_k, "fused": exp.settings.rag_fusion_top_k * 2,
                    "dedup": exp.settings.rag_fusion_top_k, "rerank": exp.settings.rag_reranker_top_k,
                    "final": exp.settings.rag_final_top_k},
        "raw_recall_cost": "At most 2-3 added queries; downstream budgets unchanged. No latency conclusion.",
    })
    static(exp)


def static(exp):
    verify_frozen(exp)
    assert sha(ROOT / "tools/branch_retrieval_candidate.py") == read(exp.out / "candidate_spec.json")["candidate_sha"]
    plans = read(exp.out / "branch_plans.json")
    results = {}
    with patch("local_full_text_search.rag.answer_service.HybridRetriever", branch_factory(exp, plans)):
        for cid in IDS:
            case = exp.cases[cid]
            before = read(exp.out / f"baseline_{cid}.json")
            after = exp.native_full(case, arm="B")
            write(exp.out / f"candidate_{cid}.json", after)
            results[cid] = {"same_context": before["context"] == after["context"],
                            "A": evidence_checks(case, before, exp.chunk_rows), "B": evidence_checks(case, after, exp.chunk_rows),
                            "entry_count": len(after["entries"]), "chars": len(after["context"])}
            print("static", cid, "same", results[cid]["same_context"],
                  "bound", [r["bound_in_context"] for r in results[cid]["B"]], flush=True)
    # Source-bound checks, not page-level guesses; protected questions must be byte-identical.
    gate = (all(r["bound_in_context"] for r in results[IDS[0]]["B"])
            and all(results[cid]["same_context"] for cid in IDS[1:])
            and all(r["entry_count"] <= exp.settings.rag_final_top_k and r["chars"] <= exp.settings.rag_context_max_chars
                    for r in results.values()))
    write(exp.out / "static_gate.json", {"passed": gate, "cases": results,
          "rule": "040 both exact quotes bound to source; protections byte-identical; original final/context budgets"})
    verify_frozen(exp)
    print("static_gate", gate, flush=True)


def finish(exp):
    verify_frozen(exp)
    source_checks(exp)
    spec = read(exp.out / "candidate_spec.json")
    assert spec["candidate_sha"] == sha(ROOT / "tools/branch_retrieval_candidate.py")
    gate = read(exp.out / "static_gate.json")
    assert gate["passed"] is False, "This finalizer describes the failed static-gate branch only"
    junit = ET.parse(exp.out / "pytest.xml").getroot()
    suites = list(junit) if junit.tag == "testsuites" else [junit]
    tests = {key: sum(int(s.get(key, 0)) for s in suites) for key in ("tests", "failures", "errors", "skipped")}
    assert tests["tests"] == 59 and tests["failures"] == tests["errors"] == 0
    audit = read(exp.out / "source_audit.json")
    ids = {r["chunk_id"]: p["slide_1based"] for p in audit for r in p["chunks"]}
    matrix = {}
    for arm, prefix in (("A", "baseline_"), ("B", "candidate_")):
        trace = read(exp.out / f"{prefix}{IDS[0]}.json")
        rows = []
        for event in trace["events"]:
            for key in ("lexical_candidates", "vector_candidates", "branch_fused_candidates",
                        "deduplicated_candidates", "rerank_input_candidates", "reranked_candidates",
                        "constraint_candidates", "evidence_candidates"):
                if key not in event:
                    continue
                for position, item in enumerate(event[key], 1):
                    if item["chunk_id"] in ids:
                        rows.append({"stage": event["stage"], "collection": key, "position": position,
                                     "slide": ids[item["chunk_id"]], **item})
        matrix[arm] = rows
    write(exp.out / "stage_matrix.json", matrix)
    write(exp.out / "answer_run_status.json", {
        "planned_generations": 24, "executed_generations": 0,
        "executed_answer_judges": 0, "executed_planner_calls": 1,
        "reason": "Static evidence protection failed; per plan stages 5/6 answer calls must not run.",
        "not_an_answer_quality_score": True,
    })
    write(exp.out / "regression_summary.json", {
        "scope": "Three reviewed protection questions, static native contexts only",
        "cases": {cid: {"byte_identical_context": gate["cases"][cid]["same_context"],
                        "evidence_A": gate["cases"][cid]["A"], "evidence_B": gate["cases"][cid]["B"]}
                  for cid in IDS[1:]},
        "answer_scores": "Not measured; unchanged context does not prove deterministic LLM answers.",
    })
    protocol = read(exp.out / "protocol.json")
    files = [ROOT / "tools/branch_retrieval_candidate.py", Path(__file__),
             ROOT / "tests/rag/test_branch_retrieval_candidate.py"]
    write(exp.out / "end_manifest.json", {
        "finished": datetime.now().isoformat(), "bundle_id": exp.bundle.bundle_id,
        "frozen_inputs_unchanged": True, "input_hashes": input_hashes(exp),
        "new_code_hashes": {str(p): sha(p) for p in files}, "tests": tests,
        "decision": "DO_NOT_ADOPT", "production_default_changed": False,
        "answer_experiments": "not_run_static_gate_failed",
        "harness_changed_during_development": protocol["harness_sha_at_freeze"] != sha(__file__),
        "candidate_unchanged_since_spec": True,
    })
    report = f"""# 040 条件分支召回：开发验证结果

日期：2026-11-10。结论：**不采用本轮候选，保留现行基线。召回补齐了，但最终证据覆盖变差。040 尚未修复。**

## 1. 结果先看这张表

| 检查 | A：现行基线 | B：独立分支召回 | 判定 |
|---|---|---|---|
| 第 14 张进入重排输入 | 是，融合第 1 | 是，融合第 1 | 保留 |
| 第 16 张进入重排输入 | 否 | 是，融合第 13 | 召回改善 |
| 第 14 张模型重排位置 | 第 1 | 第 1 | 仍认为高度相关 |
| 第 16 张模型重排位置 | 未进入，无法评估 | 第 2 | 仍认为高度相关 |
| 第 14 张条件检查后位置 | 第 7 | 第 12 | 退步 |
| 第 16 张条件检查后位置 | 不存在 | 第 13 | 被排至末段 |
| 最终上下文完整且绑定来源的 Gold 引文 | 1/2 | 0/2 | **退步，禁止采用** |
| 020、022、028 上下文 | 原上下文 | 三题全部逐字相同 | 静态保护通过 |
| 回答生成、答案判分 | 未执行 | 未执行 | 静态门槛失败，不报告虚构的回答分数 |

这里的 1/2、0/2 只统计 040 的两个完整 Gold 引文是否在实际送入模型的正文里，并与索引文件、页号、chunk 绑定；不是总体回答正确率。保护题没有因为上下文相同就被算成回答通过。028 的第二条完整引文本来就不在上下文中，本轮未改善、未恶化，详见 `regression_summary.json`。

## 2. 执行范围和冻结条件

- 依据：[040 开发验证计划](../../docs/rag_040_retrieval_development_validation_plan_20261110.md)。
- 发布 AI bundle：`{exp.bundle.bundle_id}`，不是普通全文 generation。
- 正式 Gold：`tests/rag/fixtures/rag_gold_v4_2_review.json`，四题均 reviewed，标签和状态未改。
- 模型、索引、原文件、生产召回/重排/条件匹配/选文/生成代码及有效设置的冻结哈希核对通过。
- 四题 A 上下文均与 2026-11-09 的原生 trace 第一轮 A 逐字一致，未用人工补页上下文替代。
- 候选只在独立工具中启用，生产代码未引用该工具；应用默认行为未切换。
- 原有大量未提交改动已记录在 `worktree_at_start.txt`，本轮未覆盖它们，也没有提交 Git。
- 实验工具从冻结阶段持续补齐诊断和报告功能，入口哈希前后均记录；候选机制在 `candidate_spec.json` 后没有变化。未伪称开发中的工具文件始终不变。

## 3. 原始召回缺口：并非切片丢失

文件：`{SOURCE}`。页号指从 1 开始的幻灯片序号。

第 14 张包含“小样本量项目重复性不好，大样本量受影响小”；第 16 张包含“如果项目都不好，那可能冲洗头堵了”。第 15 张是水质/反应温度等其他分支，不能替代第 16 张。

三页原始 OOXML 正文、关联备注和当前索引正文已核对，三条用于正负证据的 Gold 引文均在相应原页及索引中找到。索引末尾的页码来自关联 notesSlide XML，不是缺失业务正文。完整内容与引文位置保存在 `source_audit.json`。

用原问题扩展到每通道前 1000 做只读诊断：

| 资料页 | 词法位置 | 向量位置 | 当前每通道前 50 能否召回 |
|---|---:|---:|---|
| 14 | 1 | 1 | 能 |
| 15，强负例 | 37 | 79 | 词法能，不能充当正确答案 |
| 16 | 99 | 209 | 两通道都不能 |

因此原基线最早失败在**原问题召回的前 50 截断**，不是去重、重排或解析时删掉第 16 张。原文用“项目都不好”，问题使用“所有项目重复性变差”，并混合了两个条件；这些用词和表达差异可解释候选机制的动机，但本轮未单独做词汇因素的因果消融，不把它断言为唯一原因。

## 4. 实际开发了什么

候选：`tools/branch_retrieval_candidate.py`。仅对明确后续条件句触发；由只读用户原问的规划调用摘取连续原文片段，程序拒绝非原文片段、不可验证格式和多机型作用域不明的情况。不得传入题号、Gold、标准答案或正确页号。

本轮冻结的实际分支是：

1. `ALB、GLU 等小样本量项目重复性变差 ALT、AST 较稳`
2. `所有项目都变差`

仍保留原始查询通道。每个分支使用原来的词法 50、向量 50 预算。各通道取不同查询中最好的已有名次，再用原来的双通道 RRF 公式融合；重复命中不额外投票。这**会改变合并候选及其检索先验**，不能说结果排序完全不变；不改融合公式、模型重排或条件匹配代码。

040 原始融合候选 60，分支去重并集 129，仍截成 60。之后沿用现行去重 30、重排输入上限 20、最终 8 段、单文档最多 3 段、12,000 字。基线实际重排输入 20 中包含去重后的融合名次大于 20 的记录，这是现有去重行为，不是候选扩大预算。

分支执行失败时原子退回基线，取消任务不会被吞掉。触发条件和查询都是题目结构驱动，没有写入业务答案词或邻页补页逻辑。

限制：规划器虽被要求继承省略的指标，本次第二分支仍未写出“重复性”；程序仅核验片段来自原问，无法据此证明语义完整。该候选是失败的独立原型，不是可上线的通用查询规划器。三个保护题都未触发新机制，所以它们只能证明未触发路径不变，不能证明触发后的泛化。

## 5. 新暴露的确定性阻塞点

第 14、16 张已进入 B 的重排输入，原模型重排分别排第 1、第 2，Cross-Encoder 分数分别约 0.9351、0.9412。随后发生了以下过程：

1. 当前问题解释器把后半句“所有项目”抽成整题统一的硬范围 `scope=all`，另抽出对象“样本”；没有保留“前半句局部项目，后半句所有项目”的条件作用域。
2. 条件匹配器在正确正文中没有认出“项目都不好”对应所有项目，也没有将“样品/标本”对应当前对象“样本”；两段都被标为两个硬条件 UNKNOWN，**没有判成冲突**。
3. `ConstraintMatcher.evaluate` 优先按是否未知、匹配数排序；这一优先级能覆盖模型重排顺序。包含“所有项目、样本”等词的校准、检测项目列表和其他资料被放在正确故障证据前。
4. B 增加了这类候选后，两段正确证据被推至条件排序第 12/13。最终证据策略只取前 8，理由均为“最终 TopK 截断”。不是重排分数低于阈值，也不是这两段触发单文档 3 段上限。

完整阶段、分数、判断理由、实际上下文见 `stage_matrix.json` 和 `candidate_rag-biz-040.json`。这里不是断言模型不识别两段内容；相反，模型把它们排在最前。

可确认的共同机制是：**跨条件问题被压成统一硬条件，再由表面条件匹配优先级覆盖高相关证据的排序。** 这说明只补召回不足以修好本题；不证明所有题都存在同一问题。

## 6. 各阶段完成情况

| 计划阶段 | 执行结果 |
|---|---|
| 0 冻结与复现 | 完成：四题基线上下文逐字复现，输入哈希一致 |
| 1 原资料、索引与排名定位 | 完成：原始召回截断，三页引用核对通过 |
| 2/3 单候选开发 | 完成：条件分支召回独立模块、来源片段验证、失败回退及预算追踪 |
| 4 静态保护门槛 | **失败**：040 来源完整引文覆盖 1/2→0/2；三保护题上下文相同 |
| 5 A/B 各 3 次回答 | 按门槛未执行；计划 24 次，实际 0 次。仅执行过 1 次分支规划调用 |
| 6 回归 | 已完成三保护题本地静态回归；回答回归因门槛失败未执行，不冒充答案验证 |
| 7 采用/回退结论 | 不采用；原基线不变，无需迁移默认设置 |

这不是因耗时停止，也不是只做第一步便停下；已完成开发与静态失败处置。计划明确要求静态失败后不再调用回答模型，更不能在同一对照中改条件匹配或扩大段落数以追求通过。

## 7. 自动化验证和一致性

命令：

```powershell
.venv/Scripts/python.exe -m pytest tests/rag/test_branch_retrieval_candidate.py tests/rag/test_query_constraint_safety.py tests/rag/test_rag_end_to_end.py -q --junitxml=reports/rag_040_retrieval_20261110_01/pytest.xml
```

结果：**{tests['tests']} passed，0 failed，0 errors**。新增候选测试覆盖单问不触发、两条件分支、重复条件去重、原问之外片段拒绝、机型保留、多机型不明时退回、规划失败、RRF 重复命中不重复投票、预算排除记录、输入不变、过滤条件保留、部分通道失败原子回退以及取消传播。

`end_manifest.json` 核对发布 bundle、设置、生产代码、Gold JSON/JSONL、原始资料和基线 trace 哈希不变；`pytest.xml` 保留完整测试结果。未改变正式评测题目数量、状态和判分定义，未输出任何新旧总体算法进步结论。

## 8. 下一步应改哪层

下一项应聚焦**问题条件的作用域及条件匹配后的证据排序**，而不是继续扩大召回或换重排模型：

- 一个条件只作用于它所属的分支，不能把后半句“所有”强加给前半句；
- 条件相关性必须绑定同一对象、现象和分支；一般项目列表里碰巧出现范围词，不等于支持当前故障；
- UNKNOWN 不等于不相关，不能在未确认真实冲突时，让很低的语义相关度仅靠条件词压过直接故障证据；
- 使用未参与调试的跨领域对比条件题，以及明确冲突负例，验证排序修复能否兼顾分支覆盖和错对象保护。

这些是根据本次 trace 定位的下一层修复约束，**本轮没有实现或验证其效果**。不能承诺只改这一层就一定解决所有问题，也不能用新增几个同义词替代通用作用域设计。

## 9. 复查入口

- `protocol.json` / `end_manifest.json`：输入、设置、代码哈希和最后决策。
- `source_audit.json`：三页原文、关联备注、索引 chunk、逐条引文核验。
- `extended_ranks.json` / `diagnostic_trace.json`：原问题扩展排名和原生链路。
- `branch_plans.json` / `candidate_spec.json`：唯一一次规划的实际输出、候选代码哈希和预算。
- `baseline_rag-biz-*.json` / `candidate_rag-biz-*.json`：四题逐层 trace 与实际上下文。
- `static_gate.json` / `stage_matrix.json`：来源绑定门槛和最早失败环节。
- `regression_summary.json` / `answer_run_status.json`：静态保护与未执行回答的明确边界。
- `pytest.xml`：59 项自动化检查。
"""
    (exp.out / "report.md").write_text(report, encoding="utf8")
    doc = ROOT / "docs/rag_040_retrieval_validation_result_20261110.md"
    doc.write_text("# 040 召回修复验证结论\n\n"
                   "已执行开发和静态验证；结论为不采用本轮候选，保留当前生产基线。\n\n"
                   "040 的最终来源引文覆盖从 1/2 降至 0/2；020、022、028 上下文逐字不变。"
                   "59 项自动化检查通过；静态门槛失败，未运行回答 A/B。\n\n"
                   "已定位：正确两段在模型重排后是第 1/2，条件匹配重排后成为第 12/13，被最终 8 段截断。"
                   "下一步应修条件作用域与适用性排序，本轮未跨层修改。\n\n"
                   "[完整报告和证据链](../reports/rag_040_retrieval_20261110_01/report.md)\n", encoding="utf8")
    verify_frozen(exp)
    print("finished DO_NOT_ADOPT; frozen inputs unchanged; 59 tests passed; no answer calls", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "diagnose", "candidate", "static", "finish"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    exp = Runtime(args.out)
    {"freeze": freeze, "diagnose": diagnose, "candidate": candidate, "static": static, "finish": finish}[args.stage](exp)


if __name__ == "__main__":
    main()
