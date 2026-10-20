from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.evidence_matcher import GoldEvidenceMatch, map_case_evidence
from local_full_text_search.rag.evaluation.gold_loader import GoldCase, GoldDatasetError, load_gold_dataset
from local_full_text_search.rag.evaluation.metrics import aggregate, build_rank_history, score_case
from local_full_text_search.rag.evaluation.runner import RagEvalConfig, RagEvalRunner
from local_full_text_search.rag.models import RagAnswerStatus, RagRetrievalResult, SourceRef


FIXTURE = Path(__file__).parent / "fixtures" / "rag_gold_v1.jsonl"


def _case(**overrides) -> GoldCase:
    values = {
        "case_id": "case-1",
        "question": "如何检查交换机端口灯？",
        "expected_behavior": "answer",
        "required_facts": ("确认交换机已上电",),
        "forbidden_claims": (),
        "constraints": {},
        "missing_fields": (),
        "gold_evidence": ({"file": "guide.pdf", "page": 4, "quote": "确认交换机已上电并检查端口灯"},),
        "hard_negatives": (),
        "group_id": "switch-light",
        "corpus_snapshot": "bundle-test",
        "review_status": "reviewed",
        "raw": {},
    }
    values.update(overrides)
    return GoldCase(**values)


def test_current_gold_dataset_is_reviewed_and_has_two_refusal_cases() -> None:
    dataset = load_gold_dataset(FIXTURE)
    assert len(dataset.cases) == 18
    assert all(case.review_status == "reviewed" for case in dataset.cases)
    assert sum(case.expected_behavior == "refuse" for case in dataset.cases) == 2
    assert sum(len(case.gold_evidence) for case in dataset.cases) == 24


def test_gold_loader_rejects_pending_rows(tmp_path: Path) -> None:
    row = {
        "id": "case-1", "question": "q", "expected_behavior": "refuse",
        "required_facts": [], "forbidden_claims": [], "constraints": {},
        "missing_fields": [], "gold_evidence": [], "hard_negatives": [],
        "group_id": "g", "corpus_snapshot": "bundle", "review_status": "pending",
    }
    fixture = tmp_path / "gold.jsonl"
    fixture.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(GoldDatasetError, match="尚未标记为 reviewed"):
        load_gold_dataset(fixture)


def test_gold_quote_is_mapped_across_two_chunks() -> None:
    case = _case()
    rows = [
        {"chunk_id": "left", "ordinal": 0, "filename": "guide.pdf", "page_number": 4,
         "location": "第 4 页", "text": "确认交换机已上电并检查"},
        {"chunk_id": "right", "ordinal": 1, "filename": "guide.pdf", "page_number": 4,
         "location": "第 4 页", "text": "端口灯"},
    ]
    mapped = map_case_evidence(case, rows)
    assert mapped[0].confidence == "exact"
    assert mapped[0].chunk_ids == ("left", "right")


def test_gold_quote_is_not_joined_across_different_same_named_files() -> None:
    rows = [
        {"chunk_id": "left", "ordinal": 0, "filename": "guide.pdf", "file_path": "C:/one/guide.pdf",
         "page_number": 4, "location": "第 4 页", "text": "确认交换机已上电并检查"},
        {"chunk_id": "right", "ordinal": 1, "filename": "guide.pdf", "file_path": "C:/two/guide.pdf",
         "page_number": 4, "location": "第 4 页", "text": "端口灯"},
    ]
    mapped = map_case_evidence(_case(), rows)
    assert mapped[0].confidence == "unmatched"
    assert mapped[0].chunk_ids == ()


def test_duplicate_same_named_gold_sources_require_review() -> None:
    quote = "确认交换机已上电并检查端口灯"
    rows = [
        {"chunk_id": "one", "ordinal": 0, "filename": "guide.pdf", "file_path": "C:/one/guide.pdf",
         "page_number": 4, "location": "第 4 页", "text": quote},
        {"chunk_id": "two", "ordinal": 0, "filename": "guide.pdf", "file_path": "C:/two/guide.pdf",
         "page_number": 4, "location": "第 4 页", "text": quote},
    ]
    mapped = map_case_evidence(_case(), rows)
    assert mapped[0].confidence == "review"
    assert "多份资料" in mapped[0].reason


def test_refusal_cases_are_excluded_from_positive_recall_denominator() -> None:
    case = _case(expected_behavior="refuse", gold_evidence=(), hard_negatives=())
    result = score_case(case, {}, [], answer={"mode": "retrieval-only", "status": "answered", "text": ""})
    summary = aggregate([result])
    assert result["actual_behavior"] == "not_run"
    assert result["behavior_match"] is None
    assert summary["stages"]["fusion"]["cases_with_gold"] == 0
    assert summary["refusal_cases"] == 1
    assert summary["behavior_accuracy"] is None


def test_evidence_stage_recall_counts_only_candidates_that_passed_the_gate() -> None:
    case = _case()
    mapping = GoldEvidenceMatch(
        evidence_id="gold-1", kind="positive", file="guide.pdf", page="4",
        quote="确认交换机已上电并检查端口灯", chunk_ids=("gold-chunk",),
        confidence="exact", reason="",
    )
    result = score_case(
        case,
        {
            "evidence": [
                {"chunk_id": "gold-chunk", "selected": False},
                {"chunk_id": "other", "selected": True},
            ]
        },
        [mapping],
        answer={"mode": "retrieval-only", "status": "answered", "text": ""},
    )

    assert result["stages"]["evidence"]["gold_hit"] is False
    assert result["stages"]["evidence"]["gold_evidence_recall"] == 0.0


def test_rank_history_uses_final_selected_order_and_omits_rejected_evidence() -> None:
    history = build_rank_history({
        "fusion": [{"chunk_id": "a"}, {"chunk_id": "b"}],
        "rerank": [{"chunk_id": "b"}, {"chunk_id": "a"}],
        "evidence": [
            {"chunk_id": "a", "selected": True, "final_rank": 2},
            {"chunk_id": "b", "selected": True, "final_rank": 1},
            {"chunk_id": "rejected", "selected": False, "final_rank": None},
        ],
    })

    assert history["b"] == [("fusion", 2), ("rerank", 1), ("evidence", 1)]
    assert history["a"][-1] == ("evidence", 2)
    assert "rejected" not in history


def test_runner_preflights_and_writes_per_case_report(tmp_path: Path, monkeypatch) -> None:
    from local_full_text_search.rag.evaluation import runner as runner_module

    case = _case()
    dataset_file = tmp_path / "gold.jsonl"
    dataset_file.write_text(json.dumps(case.raw | {
        "id": case.case_id,
        "question": case.question,
        "expected_behavior": case.expected_behavior,
        "required_facts": list(case.required_facts),
        "forbidden_claims": list(case.forbidden_claims),
        "constraints": case.constraints,
        "missing_fields": list(case.missing_fields),
        "gold_evidence": list(case.gold_evidence),
        "hard_negatives": list(case.hard_negatives),
        "group_id": case.group_id,
        "corpus_snapshot": case.corpus_snapshot,
        "review_status": case.review_status,
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    class FakeStore:
        def __init__(self, _root):
            pass

        def active(self):
            return SimpleNamespace(bundle_id="bundle-test", metadata_path=tmp_path / "metadata.db")

        def connect_read_only(self, _path):
            return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(runner_module, "RagIndexStore", FakeStore)
    monkeypatch.setattr(runner_module, "load_chunk_rows", lambda _connection: [{
        "chunk_id": "gold-chunk", "ordinal": 0, "filename": "guide.pdf", "page_number": 4,
        "location": "第 4 页", "text": "确认交换机已上电并检查端口灯",
    }])

    settings = SimpleNamespace(
        rag_lexical_top_k=50, rag_vector_top_k=50, rag_fusion_top_k=30,
        rag_reranker_top_k=20, rag_final_top_k=8, ocr_min_confidence=0.6,
        rag_embedding_repo_id="embed", rag_embedding_revision="a" * 40,
        rag_reranker_repo_id="rerank", rag_reranker_revision="b" * 40,
        rag_api_model="model", rag_api_style="responses", rag_answer_review_enabled=False,
    )
    runner = RagEvalRunner(tmp_path / "search_index.db", settings, CancelToken())
    config = RagEvalConfig(dataset_file, mode="retrieval-only", output_root=tmp_path / "runs")
    prepared = runner.prepare(config)

    candidate = {
        "chunk_id": "gold-chunk", "filename": "guide.pdf", "location": "第 4 页",
        "text": "确认交换机已上电并检查端口灯", "lexical_rank": 1,
        "vector_score": 0.8, "fused_score": 0.03, "rerank_score": 0.9,
    }

    class FakeService:
        def __init__(self, callback):
            self.callback = callback

        def retrieve(self, _query, _token, *, trace_id):
            self.callback({"stage": "vector_retrieval", "lexical_candidates": [candidate]})
            self.callback({"stage": "fusion", "vector_candidates": [candidate]})
            self.callback({"stage": "deduplicating", "fused_candidates": [candidate], "deduplicated_candidates": [candidate]})
            self.callback({"stage": "reranked", "reranked_candidates": [candidate]})
            self.callback({"stage": "checking_constraints", "constraint_candidates": [candidate]})
            self.callback({"stage": "checking_evidence", "evidence_candidates": [{**candidate, "selected": True}]})
            return RagRetrievalResult(
                trace_id, RagAnswerStatus.ANSWERED,
                sources=(SourceRef(
                    source_generation=1, stable_chunk_id="gold-chunk", document_id=1,
                    file_id=1, root_id=1, file_path="C:/docs/guide.pdf", filename="guide.pdf",
                    extension=".pdf", block_ids=(1,), location_text="第 4 页", page_number=4,
                ),),
            )

    report = runner.run(config, prepared, lambda callback: FakeService(callback))
    assert report["summary"]["stages"]["fusion"]["gold_hit_rate"] == 1.0
    assert report["summary"]["stages"]["fusion"]["mean_reciprocal_rank"] == 1.0
    assert report["results"][0]["stages"]["fusion"]["first_gold_rank"] == 1
    assert report["results"][0]["stages"]["context"]["gold_hit"] is True
    assert (Path(report["run_dir"]) / "cases.jsonl").is_file()
    assert (Path(report["run_dir"]) / "summary.md").is_file()
    html_report = Path(report["run_dir"]) / "report.html"
    assert html_report.is_file()
    html = html_report.read_text(encoding="utf-8")
    assert "本阶段排名" in html
    assert "跨阶段排名轨迹" in html
    assert "MRR" in html
    assert report["summary"]["metric_definitions"]["mean_reciprocal_rank"]
