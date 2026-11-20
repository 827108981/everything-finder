"""Structural safeguards for the isolated scope annotation candidate."""

import json

import pytest

from tools.source_scope_candidate_v2 import validate


def test_rejects_support_not_in_sent_text():
    payload = {
        "relation": "direct", "question_quote": "状态 A",
        "support_quote": "应更换组件", "scope_quote": "",
        "scope_kind": "none", "reason": "",
    }
    with pytest.raises(ValueError, match="support_quote"):
        validate("状态 A 怎么处理？", "状态 A 应检查组件。", json.dumps(payload))


def test_rejects_qualified_result_without_source_scope():
    payload = {
        "relation": "qualified", "question_quote": "状态 A",
        "support_quote": "状态 A 应检查组件", "scope_quote": "",
        "scope_kind": "none", "reason": "",
    }
    with pytest.raises(ValueError, match="qualified relation"):
        validate("状态 A 怎么处理？", "状态 A 应检查组件。", json.dumps(payload))


def test_accepts_exact_quote_but_does_not_claim_semantic_validation():
    payload = {
        "relation": "qualified", "question_quote": "状态 A",
        "support_quote": "状态 A 应检查组件", "scope_quote": "状态 A",
        "scope_kind": "condition", "reason": "需要人工核对范围与结论的关系",
    }
    result = validate("状态 A 怎么处理？", "状态 A 应检查组件。", json.dumps(payload))
    assert result.relation == "qualified"
