from __future__ import annotations

import json
from pathlib import Path

from local_full_text_search.rag.evaluation.gold_loader import GoldCase, GoldDataset
import pytest

from local_full_text_search.rag.evaluation.reporting.service import (
    ReportConfig,
    SnapshotMismatchError,
    build_report_detail,
    generate_report,
    write_report_files,
)


def _dataset() -> GoldDataset:
    case = GoldCase(
        case_id="case-1",
        question="流量是多少？",
        expected_behavior="answer",
        required_facts=(),
        forbidden_claims=(),
        constraints={},
        missing_fields=(),
        gold_evidence=({"file": "guide.pdf", "page": 19, "quote": "正确流量 12"},),
        hard_negatives=({"file": "guide.pdf", "page": 19, "quote": "旧机型流量 8"},),
        group_id="g",
        corpus_snapshot="bundle-test",
        review_status="reviewed",
        raw={},
    )
    return GoldDataset(Path("gold.jsonl"), "sha", (case,))


def test_detail_contains_stage_ranks_and_mixed_category(tmp_path: Path) -> None:
    result = {
        "id": "case-1",
        "case_index": 1,
        "question": "流量是多少？",
        "expected_behavior": "answer",
        "actual_behavior": "not_run",
        "candidate_stages": {
            "lexical": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8", "lexical_rank": 1}],
            "vector": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8", "vector_rank": 2}],
            "fusion": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8"}],
            "evidence": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8", "selected": True, "final_rank": 1}],
            "context": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8", "included_text": "正确流量 12；旧机型流量 8"}],
        },
        "events": [],
    }
    detail = build_report_detail(_dataset(), {"run_id": "r", "ai_bundle_id": "bundle-test"}, [result], ReportConfig())
    case = detail["cases"][0]
    assert case["stage_data"]["fusion"]["candidates"][0]["category"] == "mixed"
    assert case["stage_summaries"]["context"]["first_gold_rank"] == 1
    assert case["stage_summaries"]["context"]["ranked_candidate_count"] == 1
    assert any(item["code"] == "mixed_chunk" for item in case["diagnostics"])
    write_report_files(detail, tmp_path)
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report_cases.csv").is_file()
    payload = json.loads((tmp_path / "report_detail.json").read_text(encoding="utf-8"))
    assert payload["cases"][0]["stage_data"]["evidence"]["candidates"][0]["stage_rank"] == 1


def test_report_shows_timing_and_gold_judge_in_heatmap_and_detail(tmp_path: Path) -> None:
    result = {
        "id": "case-1", "duration_ms": 1000,
        "events": [{"stage": "generating", "elapsed_ms": 100},
                   {"stage": "gold_judging", "elapsed_ms": 600},
                   {"stage": "gold_judged", "elapsed_ms": 950}],
        "gold_judgement": {"verdict": "inconsistent", "behavior": "mismatch",
                            "behavior_reason": "答案说成了旧机型流量", "model": "judge-model",
                            "rubrics": [{"id": "F1", "kind": "required_facts", "gold_text": "正确流量 12",
                                         "status": "missing", "reason": "没有提到 12"}]},
    }
    run = {"run_id": "r", "ai_bundle_id": "bundle-test", "preparation_duration_ms": 250,
           "started_at": "2026-10-23T00:00:00+00:00", "finished_at": "2026-10-23T00:00:02+00:00"}
    detail = build_report_detail(_dataset(), run, [result])
    assert detail["summary"]["timing"]["modules_ms"]["generation"] == 500
    assert detail["summary"]["timing"]["modules_ms"]["gold_judge"] == 350
    assert detail["summary"]["timing"]["overall_recorded_ms"] == 2250
    assert detail["cases"][0]["gold_judgement"]["verdict"] == "inconsistent"
    write_report_files(detail, tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "答案与金标" in html
    assert "金标答案裁判" in html
    assert "答案说成了旧机型流量" in html


def test_baseline_marks_wrong_to_right_and_csv_exposes_change(tmp_path: Path) -> None:
    result = {
        "id": "case-1",
        "candidate_stages": {
            "context": [{
                "chunk_id": "c1", "filename": "guide.pdf", "page_number": 19,
                "location": "第 19 页", "text": "正确流量 12", "included_text": "正确流量 12",
            }],
        },
        "events": [],
    }
    run = {"run_id": "r", "ai_bundle_id": "bundle-test"}
    baseline = build_report_detail(_dataset(), run, [result], ReportConfig())
    baseline["cases"][0]["stage_summaries"]["context"]["hit_at_5"] = False
    current = build_report_detail(_dataset(), run, [result], ReportConfig(), baseline=baseline)
    assert current["baseline"]["case_changes"]["case-1"]["state"] == "错→对"
    write_report_files(current, tmp_path)
    assert "错→对" in (tmp_path / "report_cases.csv").read_text(encoding="utf-8-sig")


def test_report_explains_gold_cutoff_and_exposes_executed_keywords(tmp_path: Path) -> None:
    gold = {"chunk_id": "gold", "filename": "guide.pdf", "page_number": 19,
            "location": "第 19 页", "text": "正确流量 12；旧机型流量 8", "vector_rank": 22}
    filler = [{"chunk_id": f"other-{i}", "filename": "other.pdf", "text": "其他内容"}
              for i in range(43)]
    result = {
        "id": "case-1", "question": "流量是多少？",
        "candidate_stages": {"vector": [gold], "fusion": [*filler, gold], "deduplicated": []},
        "events": [
            {"stage": "query_interpreted", "query_plan": {
                "original_text": "流量是多少？", "lexical_terms": ["流量"],
                "lexical_expression": '"流量"', "semantic_query": "流量是多少？",
            }},
            {"stage": "lexical_retrieval", "lexical_terms": ["流量标准"],
             "lexical_fts_expression": '"流量标准"'},
            {"stage": "deduplicating", "candidate_dispositions": [{
                "chunk_id": "gold", "status": "fusion_top_k_cutoff",
                "reason": "去重后候选已达到 fusion_top_k=30",
            }]},
        ],
    }
    detail = build_report_detail(_dataset(), {"ai_bundle_id": "bundle-test"}, [result])
    case = detail["cases"][0]
    assert case["query"]["lexical_terms"] == ["流量标准"]
    assert case["gold_evidence"][0]["exit"]["status"] == "fusion_top_k_cutoff"
    assert "fusion_top_k=30" in case["gold_evidence"][0]["exit"]["reason"]
    assert any(item["code"] == "gold_lost_after_recall" for item in case["diagnostics"])
    write_report_files(detail, tmp_path)
    assert "本题实际检索关键词" in (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "流量标准" in (tmp_path / "report_cases.csv").read_text(encoding="utf-8-sig")


def test_old_run_context_uses_representative_entries_not_flat_sources() -> None:
    representative = {"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19,
                      "location": "第 19 页", "text": "正确流量 12", "selected": True, "final_rank": 1}
    extra_source = {"chunk_id": "c2", "filename": "guide.pptx", "text": "正确流量 12"}
    result = {
        "id": "case-1",
        "candidate_stages": {"evidence": [representative], "context": [representative, extra_source]},
        "events": [{"stage": "retrieval_completed", "context_sources": [{
            "chunk_id": "c1", "included_text": "正确流量 12", "truncated": False,
            "sources": [{"chunk_id": "c1"}, {"chunk_id": "c2"}],
        }]}],
    }
    detail = build_report_detail(_dataset(), {"ai_bundle_id": "bundle-test"}, [result])

    context = detail["cases"][0]["stage_data"]["context"]["candidates"]
    assert [item["chunk_id"] for item in context] == ["c1"]
    assert detail["cases"][0]["stage_summaries"]["context"]["candidate_count"] == 1


def test_report_keeps_final_answer_and_failed_model_refusal(tmp_path: Path) -> None:
    run = {"run_id": "r", "ai_bundle_id": "bundle-test", "mode": "full"}
    answered = {
        "id": "case-1", "actual_behavior": "answer",
        "answer": {"status": "answered", "text": "正确流量是 12 [S1]。"},
        "events": [{"stage": "draft_generated", "draft_answer": "正确流量是 12 [S1]。"}],
    }
    detail = build_report_detail(_dataset(), run, [answered], ReportConfig())
    assert detail["cases"][0]["answer"]["text"] == "正确流量是 12 [S1]。"

    refused = {
        "id": "case-1", "actual_behavior": "error", "failed_stage": "validating",
        "error": {"type": "RagGenerationError", "message": "模型答案没有引用检索到的资料"},
        "events": [{
            "stage": "draft_generated",
            "draft_answer": "抱歉，本服务不支持此类协助。",
            "citation_check": {"structurally_valid": False, "reason": "答案没有引用检索到的资料"},
        }],
    }
    detail = build_report_detail(_dataset(), run, [refused], ReportConfig())
    case = detail["cases"][0]
    assert case["generation"]["draft_answer"] == "抱歉，本服务不支持此类协助。"
    assert case["generation"]["failed_stage"] == "validating"
    assert case["error"]["message"] == "模型答案没有引用检索到的资料"
    write_report_files(detail, tmp_path)
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "模型／服务商返回的拒答说明" in html
    assert "未命中/${total}" in html


def test_heatmap_rank_denominator_excludes_rejected_candidates() -> None:
    selected = {"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19,
                "text": "正确流量 12", "selected": True, "final_rank": 1}
    rejected = {"chunk_id": "c2", "filename": "guide.pdf", "page_number": 19,
                "text": "错误流量 8", "selected": False}
    result = {"id": "case-1", "candidate_stages": {"evidence": [selected, rejected]}}
    detail = build_report_detail(_dataset(), {"ai_bundle_id": "bundle-test"}, [result])
    summary = detail["cases"][0]["stage_summaries"]["evidence"]
    assert summary["candidate_count"] == 2
    assert summary["ranked_candidate_count"] == 1
    assert summary["first_gold_rank"] == 1


def test_snapshot_mismatch_writes_warning_and_requires_explicit_override(tmp_path: Path) -> None:
    gold = {
        "_阅读说明": {"AI索引目录": "E:/rag.g000001-test"},
        "cases": [_dataset().cases[0].raw | {
            "id": "case-1", "question": "流量是多少？", "expected_behavior": "answer",
            "required_facts": [], "forbidden_claims": [], "constraints": {},
            "missing_fields": [], "gold_evidence": [{"file": "guide.pdf", "page": 19, "quote": "正确流量 12"}],
            "hard_negatives": [], "group_id": "g", "corpus_snapshot": "g000001-test", "review_status": "reviewed",
        }],
    }
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run.json").write_text(json.dumps({"run_id": "r", "ai_bundle_id": "g000002-other"}), encoding="utf-8")
    (run_dir / "cases.jsonl").write_text(json.dumps({"id": "case-1", "candidate_stages": {}, "events": []}, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(SnapshotMismatchError):
        generate_report(cases_path, run_dir, tmp_path / "out")
    assert (tmp_path / "out" / "report.html").is_file()
    detail = generate_report(
        cases_path, run_dir, tmp_path / "out-allowed",
        ReportConfig(allow_snapshot_mismatch=True),
    )
    assert detail["validation"]["mismatch"] is True
    assert detail["validation"]["allowed"] is True
