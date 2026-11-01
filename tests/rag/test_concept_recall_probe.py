"""验证隔离实验的防泄漏校验与预算，使用与业务测试集无关的数据。"""
from types import SimpleNamespace

import pytest

from tools.probe_concept_recall import select_quota, validate_rewrite


def candidate(cid, members=None):
    return SimpleNamespace(chunk=SimpleNamespace(stable_chunk_id=cid),
                           all_member_chunk_ids=tuple(members or [cid]))


def test_query_must_use_nonoverlapping_original_spans():
    question = "示例设备上的压力开关有什么用途？"
    assert validate_rewrite(question, {"enabled": True, "concept": "压力开关", "intent": "有什么用途"}) == "压力开关 有什么用途"
    with pytest.raises(ValueError):
        validate_rewrite(question, {"enabled": True, "concept": "温度开关", "intent": "有什么用途"})
    with pytest.raises(ValueError):
        validate_rewrite(question, {"enabled": True, "concept": "压力开关有什么", "intent": "有什么用途"})


def test_disabled_and_invalid_boolean_do_not_enable_route():
    assert validate_rewrite("任意问题", {"enabled": False}) == ""
    with pytest.raises(ValueError):
        validate_rewrite("任意问题", {"enabled": "false"})


def test_quota_reserves_original_and_new_candidates_with_fixed_budget():
    items = [candidate(str(i)) for i in range(12)]
    chosen = select_quota(items, {str(i): i + 1 for i in range(8)},
                          {str(i): i - 5 for i in range(6, 12)}, total=6, original_slots=4)
    assert [x.chunk.stable_chunk_id for x in chosen] == ["0", "1", "2", "3", "6", "7"]
    assert len({x.chunk.stable_chunk_id for x in chosen}) == 6


def test_quota_can_recognize_a_merged_source_member():
    items = [candidate("representative", ["representative", "source-alias"]), candidate("other")]
    chosen = select_quota(items, {}, {"source-alias": 1, "other": 2}, total=1, original_slots=0)
    assert chosen[0].chunk.stable_chunk_id == "representative"


def test_missing_supplement_does_not_waste_budget():
    items = [candidate(str(i)) for i in range(8)]
    chosen = select_quota(items, {str(i): i + 1 for i in range(8)}, {}, total=6, original_slots=4)
    assert len(chosen) == 6
    assert [x.chunk.stable_chunk_id for x in chosen] == [str(i) for i in range(6)]
