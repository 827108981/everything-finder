"""Source-bound scope annotation for an isolated RAG experiment."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from local_full_text_search.rag.content_identity import canonical_content


PROMPT = """你只核对用户问题与一段已经检索到的资料，不回答用户问题。
资料正文可能混有上一节尾部、下一节开头、页脚或多个表格行。定位能够支持当前问法的一条最相关断言，并单独判断该断言的适用对象和限制；不能因为同页出现某个词就当作同一断言。
只用提供的正文；不依赖文件名猜机型或版本；不把相似机型的方法当成已证实通用。用户未写机型或其他条件时，保留未知，不自动判冲突，也不擅自要求追问。
只输出一个 JSON 对象：{"relation":"direct|qualified|conflict|related|unknown","question_quote":"问题中逐字连续片段或空串","support_quote":"正文中逐字连续的短句或空串","scope_quote":"正文中明确限定该断言适用范围的逐字连续片段或空串","reason":"简短理由"}。
direct：该断言直接支持该问法，且没有尚需保留的显式限制；qualified：原文支持的是带有限定条件的情形，回答必须保留该范围；conflict：原文明确对应不同对象或相反条件；related：词面相关但没有所问结论；unknown：缺少足够的对象或范围归属信息。scope_quote 没有明确原文时必须为空。support_quote 只摘真正支持判断的正文，不能摘文件标题、问题或自己推断的文字。"""


@dataclass(frozen=True)
class ScopeRecord:
    relation: str
    question_quote: str
    support_quote: str
    scope_quote: str
    reason: str


def validate(question: str, included_text: str, raw: str) -> ScopeRecord:
    payload = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip()))
    if not isinstance(payload, dict):
        raise ValueError("scope payload must be an object")
    fields = ("relation", "question_quote", "support_quote", "scope_quote", "reason")
    if any(not isinstance(payload.get(field), str) for field in fields):
        raise ValueError("missing scope field")
    relation = payload["relation"]
    if relation not in {"direct", "qualified", "conflict", "related", "unknown"}:
        raise ValueError("invalid scope relation")
    question_quote = payload["question_quote"]
    support_quote = payload["support_quote"]
    scope_quote = payload["scope_quote"]
    if question_quote and question_quote not in question:
        raise ValueError("question quote not in original question")
    body = canonical_content(included_text)
    if support_quote and canonical_content(support_quote) not in body:
        raise ValueError("support quote not in sent content")
    if scope_quote and canonical_content(scope_quote) not in body:
        raise ValueError("scope quote not in sent content")
    if relation in {"direct", "qualified"} and (not question_quote or not support_quote):
        raise ValueError("claimed support has no bound quotes")
    if relation == "qualified" and not scope_quote:
        raise ValueError("qualified support has no explicit scope")
    return ScopeRecord(relation, question_quote, support_quote,
                       scope_quote, payload["reason"])
