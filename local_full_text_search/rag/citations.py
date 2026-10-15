from __future__ import annotations

import json
import re
from dataclasses import dataclass

from local_full_text_search.rag.context_builder import ContextBundle
from local_full_text_search.rag.models import Citation

_SOURCE_TAG = re.compile(r"\[(S\d+)\]")
_DRIVE_PATH = re.compile(r"\b[A-Za-z]:[\\/]")


class CitationValidationError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ValidatedModelAnswer:
    insufficient: bool
    answer: str
    citations: tuple[Citation, ...]


class CitationValidator:
    def validate(self, raw: str, context: ContextBundle) -> ValidatedModelAnswer:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CitationValidationError("模型没有返回有效 JSON") from exc
        if not isinstance(value, dict):
            raise CitationValidationError("模型答案 JSON 必须是对象")
        if not isinstance(value.get("insufficient"), bool):
            raise CitationValidationError("模型答案缺少布尔字段 insufficient")
        if value["insufficient"]:
            if str(value.get("answer") or "").strip():
                raise CitationValidationError("不足证据答案必须为空，由程序显示固定拒答")
            return ValidatedModelAnswer(True, "", ())
        answer = str(value.get("answer") or "").strip()
        used = value.get("used_sources")
        if not answer or not isinstance(used, list) or not used:
            raise CitationValidationError("回答为空或没有 used_sources")
        allowed = {citation.label: citation for citation in context.citations}
        referenced = set(_SOURCE_TAG.findall(answer))
        used_labels = {str(label) for label in used}
        if not referenced or not used_labels:
            raise CitationValidationError("回答必须包含至少一个有效来源编号")
        if referenced - set(allowed) or used_labels - set(allowed):
            raise CitationValidationError("模型引用了本次证据以外的编号")
        if referenced != used_labels:
            raise CitationValidationError("正文引用编号与 used_sources 不一致")
        for line in (part.strip() for part in answer.splitlines() if part.strip()):
            if not _SOURCE_TAG.search(line):
                raise CitationValidationError("每个回答要点都必须带来源编号")
        if _DRIVE_PATH.search(answer):
            raise CitationValidationError("模型不得生成文件路径；来源路径由程序显示")
        citations = tuple(allowed[label] for label in sorted(referenced, key=_source_number))
        return ValidatedModelAnswer(False, answer, citations)


def _source_number(label: str) -> int:
    try:
        return int(label[1:])
    except (ValueError, IndexError):
        return 10**9
