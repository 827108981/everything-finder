"""Question-only conditional retrieval planner for an isolated v4.3 experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass


PROMPT = """只分析用户原话，提取用户明确比较的两个或三个不同条件；不能回答问题，不能根据资料猜答案。
每个分支提取一段连续的用户原文作为 condition_quote。后一分支如果省略了前面共有的检测指标或对象，从原问题另摘一段连续短语作为 inherited_quote；不要继承前一分支特有的项目、数值、机型或否定。若条件不明确、不同机型归属不清，返回空分支。
只输出 JSON：{"branches":[{"condition_quote":"原问题中的连续条件片段","inherited_quote":"原问题中共有指标或对象的连续片段，允许为空"}]}。字符串必须逐字来自原问题，不写部件、原因、处理方法或资料内容。"""


@dataclass(frozen=True)
class Branch:
    condition_quote: str
    inherited_quote: str
    query_text: str


def validate(question: str, raw: str):
    value = json.loads(raw)
    source = value.get("branches")
    if not isinstance(source, list) or not 2 <= len(source) <= 3:
        return ()
    branches = []
    seen = set()
    for position, row in enumerate(source):
        if not isinstance(row, dict):
            return ()
        condition = row.get("condition_quote")
        inherited = row.get("inherited_quote")
        if (not isinstance(condition, str) or len(condition.strip()) < 4
                or condition not in question or not isinstance(inherited, str)
                or (inherited and inherited not in question)):
            return ()
        if position == 0 and inherited:
            return ()
        if inherited and inherited in condition:
            return ()
        if condition in seen:
            return ()
        seen.add(condition)
        branches.append(Branch(condition, inherited,
                               " ".join(part for part in (condition, inherited) if part)))
    if any(a.condition_quote in b.condition_quote for a in branches for b in branches if a != b):
        return ()
    return tuple(branches)
