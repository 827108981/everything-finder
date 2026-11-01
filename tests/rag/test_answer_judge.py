from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.evaluation.answer_judge import GoldAnswerJudge, _parse_judgement
from local_full_text_search.rag.evaluation.gold_loader import GoldDatasetError, load_gold_dataset


def _case():
    path = Path(__file__).parent / "fixtures" / "rag_gold_v1_review.json"
    return load_gold_dataset(path).cases[1]


def _payload(*, fact_status="covered", forbidden_status="absent", behavior="match", case=None,
             overrides=None):
    case = case or _case()
    checks = case.answer_checks if "answer_checks" in case.raw else [
        {"id": f"F{i}"} for i in range(1, len(case.required_facts) + 1)]
    overrides = overrides or {}
    return json.dumps({
        "behavior": behavior, "behavior_reason": "检查预期行为",
        "required_facts": [{"id": check["id"], "status": overrides.get(check["id"], fact_status), "reason": "对照答案"}
                           for check in checks],
        "forbidden_claims": [{"id": f"N{i}", "status": forbidden_status, "reason": "对照答案"}
                             for i in range(1, len(case.forbidden_claims) + 1)],
    }, ensure_ascii=False)


@pytest.mark.parametrize(("fact", "forbidden", "behavior", "expected"), [
    ("covered", "absent", "match", "consistent"),
    ("missing", "absent", "match", "incomplete"),
    ("covered", "present", "match", "incorrect"),
    ("uncertain", "absent", "match", "uncertain"),
    ("covered", "absent", "mismatch", "behavior_mismatch"),
])
def test_judge_verdict_uses_each_gold_item(fact, forbidden, behavior, expected):
    judged = _parse_judgement(_payload(fact_status=fact, forbidden_status=forbidden,
                                        behavior=behavior), _case(), "test-model")
    assert judged["verdict"] == expected
    assert len(judged["rubrics"]) == len(_case().required_facts) + len(_case().forbidden_claims)


def test_judge_rejects_incomplete_rubric():
    data = json.loads(_payload())
    data["required_facts"].pop()
    with pytest.raises(ValueError, match="条数"):
        _parse_judgement(json.dumps(data), _case(), "test-model")


def test_judge_accepts_single_json_object_with_compatibility_preamble() -> None:
    payload = _payload()
    judged = _parse_judgement("以下是核查结果：\n```json\n" + payload + "\n```\n核查完毕。", _case(), "test-model")
    assert judged["verdict"] == "consistent"
    with pytest.raises(ValueError, match="唯一"):
        _parse_judgement(payload + "\n" + payload, _case(), "test-model")


def test_judge_sends_answer_and_gold_without_secret_in_prompt():
    requests = []

    class _Gateway:
        def complete_text(self, request, token):
            requests.append(request)
            return _payload()

    settings = SimpleNamespace(rag_api_model="test-model", rag_api_base_url="https://api.example.test/v1",
                               rag_api_style="responses", rag_api_timeout_seconds=10)
    judgement = GoldAnswerJudge(_Gateway(), settings, "private-key").evaluate(
        _case(), {"status": "answered", "text": "D63 是 +12V 光子计数电源灯。"}, CancelToken())
    assert judgement["verdict"] == "consistent"
    assert "D63" in requests[0].user_prompt
    assert "private-key" not in requests[0].user_prompt


def test_judge_retries_once_when_rubric_ids_are_missing():
    requests = []

    class _Gateway:
        def complete_text(self, request, token):
            requests.append(request)
            if len(requests) == 1:
                broken = json.loads(_payload())
                broken["forbidden_claims"] = []
                return json.dumps(broken, ensure_ascii=False)
            return _payload()

    settings = SimpleNamespace(rag_api_model="test-model", rag_api_base_url="https://api.example.test/v1",
                               rag_api_style="responses", rag_api_timeout_seconds=10)
    result = GoldAnswerJudge(_Gateway(), settings, "private-key").evaluate(
        _case(), {"status": "answered", "text": "已有引用的答案。"}, CancelToken())

    assert result["verdict"] == "consistent"
    assert len(requests) == 2
    assert "没有项目时返回空数组" in requests[1].user_prompt
    assert all("private-key" not in request.user_prompt for request in requests)


def test_v2_gold_files_match_and_preserve_v1_retrieval_evidence():
    root = Path(__file__).parent / "fixtures"
    v1 = load_gold_dataset(root / "rag_gold_v1_review.json")
    v2 = load_gold_dataset(root / "rag_gold_v2_review.json")
    v2_jsonl = load_gold_dataset(root / "rag_gold_v2.jsonl")
    assert len(v1.cases) == len(v2.cases) == len(v2_jsonl.cases) == 18
    for old, revised, jsonl_case in zip(v1.cases, v2.cases, v2_jsonl.cases):
        assert revised.raw == jsonl_case.raw
        assert revised.question == old.question
        assert revised.gold_evidence == old.gold_evidence
        assert revised.hard_negatives == old.hard_negatives
        assert revised.corpus_snapshot == old.corpus_snapshot


def test_v2_rejects_invalid_answer_checks(tmp_path):
    source = Path(__file__).parent / "fixtures" / "rag_gold_v2_review.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    data["cases"][0]["answer_checks"] = None
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(GoldDatasetError, match="answer_checks 必须是数组"):
        load_gold_dataset(broken)


def test_v2_optional_background_does_not_fail_flow_rate_answer():
    case = next(c for c in load_gold_dataset(Path(__file__).parent / "fixtures" / "rag_gold_v2.jsonl").cases
                if c.case_id == "rag-biz-007")
    result = _parse_judgement(_payload(case=case, overrides={"C1": "missing", "C2": "missing"}), case, "mock")
    assert result["verdict"] == "consistent"
    assert result["findings"] == ["C1", "C2"]
    uncertain = _parse_judgement(_payload(case=case, overrides={"C1": "uncertain"}), case, "mock")
    assert uncertain["verdict"] == "consistent"
    assert uncertain["findings"] == ["C1"]


def test_v2_safety_gap_and_clarification_are_distinct():
    cases = {c.case_id: c for c in load_gold_dataset(Path(__file__).parent / "fixtures" / "rag_gold_v2.jsonl").cases}
    safety = cases["rag-biz-005"]
    judged = _parse_judgement(_payload(case=safety, overrides={"S1": "missing"}), safety, "mock")
    assert judged["verdict"] == "safety_gap"
    clarify = cases["rag-biz-014"]
    judged = _parse_judgement(_payload(case=clarify, behavior="mismatch", overrides={"F1": "missing"}), clarify, "mock")
    assert judged["verdict"] == "needs_clarification"


def test_v2_fixed_refusal_passes_without_optional_explanation():
    case = next(c for c in load_gold_dataset(Path(__file__).parent / "fixtures" / "rag_gold_v2.jsonl").cases
                if c.case_id == "rag-biz-017")
    judged = _parse_judgement(_payload(case=case, fact_status="missing"), case, "mock")
    assert judged["verdict"] == "consistent"
