from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rejudge_report import rejudge_saved_run


def test_rejudge_saved_run_preserves_original_and_records_comparison(tmp_path: Path) -> None:
    cases = Path(__file__).parent / "fixtures" / "rag_gold_v2_review.json"
    source = tmp_path / "original"
    source.mkdir()
    (source / "run.json").write_text(json.dumps({
        "run_id": "old-run", "mode": "full-answer", "ai_bundle_id": "g000008-9f01fc283b",
    }), encoding="utf-8")
    row = {"id": "rag-biz-007", "answer": {"text": "吸液流量为 140±5 μL/s。"},
           "gold_judgement": {"verdict": "inconsistent"}, "candidate_stages": {}}
    original = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    (source / "cases.jsonl").write_bytes(original)

    class FakeJudge:
        def evaluate(self, case, answer, token):
            assert case.case_id == "rag-biz-007"
            assert answer["text"] == row["answer"]["text"]
            return {"verdict": "consistent", "findings": ["C1", "C2"],
                    "rubric_version": "answer-checks-v2"}

    output = tmp_path / "rejudged"
    result = rejudge_saved_run(cases, source, output, lambda callback: FakeJudge())
    assert (source / "cases.jsonl").read_bytes() == original
    revised = json.loads((output / "cases.jsonl").read_text(encoding="utf-8"))
    assert revised["answer"] == row["answer"]
    assert revised["gold_judgement"]["previous_verdict"] == "inconsistent"
    assert revised["gold_judgement"]["verdict"] == "consistent"
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["rejudge"]["source_cases_sha256"] == hashlib.sha256(original).hexdigest()
    assert metadata["rejudge"]["retrieval_and_answers_reused"] is True
    assert result["api_attempt_count"] == 1
    html = (output / "report.html").read_text(encoding="utf-8")
    assert "沿用原运行已保存的检索结果和答案" in html
    assert "沿用原运行已保存的答案重新判分" in html
    assert "judgement_comparison.csv" in {p.name for p in output.iterdir()}
