"""Question-only planner for conditional retrieval and applicability experiments."""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from tools.branch_retrieval_candidate import eligible


PROMPT = """只分析用户原话的条件关系，不能回答问题，也不能推断资料内容。
如果用户问多个明确不同条件，返回 JSON：
{"branches":[{"condition_quote":"原文连续条件", "fragments":["原问连续片段"],
"inherited":[{"quote":"前文被省略的同一指标或对象原文连续片段", "role":"metric|object", "from_quote":"前一条件原文连续片段"}]}]}。
若不是多条件问法或无法确定省略的关系，返回 {"branches":[]}。
规则：每个片段必须逐字出自用户原问。后续“都变差、也这样”等省略说法，必须从前一条件补回同一变化指标或对象；仅补原话能确认的共享信息。
前一条件里的具体项目、机型、例子、局部阶段、否定或数值不可偷渡到另一条件。各分支条件和继承片段一起要能独立表达用户的检索意图。
一题有多个机型或版本且条件归属不清就返回空。不要加答案词、部件或诊断原因。只输出 JSON。"""


@dataclass(frozen=True)
class ScopedBranch:
    branch_id: str
    condition_quote: str
    query_text: str
    inherited: tuple[dict[str, str], ...]


def validate(question: str, data: dict) -> tuple[ScopedBranch, ...]:
    if not eligible(question) or not isinstance(data, dict):
        return ()
    rows = data.get("branches")
    if not isinstance(rows, list) or not 2 <= len(rows) <= 3:
        return ()
    ids = set(re.findall(r"[A-Za-z]+[-_]?\d+[A-Za-z\d.\-]*", question))
    if len(ids) > 1:
        return ()
    result = []
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            return ()
        condition, fragments, inherited = (row.get("condition_quote"), row.get("fragments"),
                                            row.get("inherited", []))
        if (not isinstance(condition, str) or len(condition) < 3 or condition not in question
                or not isinstance(fragments, list) or not fragments
                or not isinstance(inherited, list)):
            return ()
        if any(not isinstance(part, str) or not part.strip() or part not in question for part in fragments):
            return ()
        if index == 0 and inherited:
            return ()
        for ref in inherited:
            if (not isinstance(ref, dict) or ref.get("role") not in {"metric", "object"}
                    or any(not isinstance(ref.get(k), str) or ref[k] not in question
                           for k in ("quote", "from_quote"))
                    or ref["from_quote"] not in rows[index - 1]["condition_quote"]
                    or ref["quote"] not in ref["from_quote"]
                    or ref["quote"] in condition):
                return ()
        # An elliptical follow-on branch must identify what it inherits.
        if index and re.search(r"(?:都|也|呢|一样|同样|此时)", condition) and not inherited:
            return ()
        supplements = [part for part in fragments
                       if part not in condition and not re.fullmatch(r"(?:优先)?(?:查|看|怎么|如何|哪)[^，。？！?]*[？?]?", part)]
        text = " ".join(dict.fromkeys([*sorted(ids), condition, *supplements,
                                       *[r["quote"] for r in inherited]]))
        if not 4 <= len(text) <= 220 or any(identifier not in text for identifier in ids):
            return ()
        if condition in seen:
            return ()
        seen.add(condition)
        result.append(ScopedBranch(f"branch_{index + 1}", condition, text,
                                   tuple({k: ref[k] for k in ("quote", "role", "from_quote")} for ref in inherited)))
    return tuple(result)


def plan(question: str, complete):
    if not eligible(question):
        return (), {"triggered": False, "fallback": False, "reason": "single_or_unsupported"}
    try:
        raw = complete(PROMPT, question).strip()
        data = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw))
        result = validate(question, data)
        return result, {"triggered": bool(result), "fallback": not bool(result),
                        "reason": "validated" if result else "invalid_or_incomplete_scope",
                        "raw": data, "branches": [{"branch_id": b.branch_id,
                                                   "condition_quote": b.condition_quote,
                                                   "query_text": b.query_text,
                                                   "inherited": list(b.inherited)} for b in result]}
    except Exception as exc:
        return (), {"triggered": False, "fallback": True,
                    "reason": "planner_failed", "error_type": type(exc).__name__}
