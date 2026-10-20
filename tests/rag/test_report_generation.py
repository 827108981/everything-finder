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
            "context": [{"chunk_id": "c1", "filename": "guide.pdf", "page_number": 19, "location": "第 19 页", "text": "正确流量 12；旧机型流量 8"}],
        },
        "events": [],
    }
    detail = build_report_detail(_dataset(), {"run_id": "r", "ai_bundle_id": "bundle-test"}, [result], ReportConfig())
    case = detail["cases"][0]
    assert case["stage_data"]["fusion"]["candidates"][0]["category"] == "mixed"
    assert case["stage_summaries"]["context"]["first_gold_rank"] == 1
    assert any(item["code"] == "mixed_chunk" for item in case["diagnostics"])
    write_report_files(detail, tmp_path)
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report_cases.csv").is_file()
    payload = json.loads((tmp_path / "report_detail.json").read_text(encoding="utf-8"))
    assert payload["cases"][0]["stage_data"]["evidence"]["candidates"][0]["stage_rank"] == 1


def test_baseline_marks_wrong_to_right_and_csv_exposes_change(tmp_path: Path) -> None:
    result = {
        "id": "case-1",
        "candidate_stages": {
            "context": [{
                "chunk_id": "c1", "filename": "guide.pdf", "page_number": 19,
                "location": "第 19 页", "text": "正确流量 12",
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
