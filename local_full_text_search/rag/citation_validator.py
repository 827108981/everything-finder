from __future__ import annotations

import re
from dataclasses import dataclass

from local_full_text_search.rag.errors import RagGenerationError
from local_full_text_search.rag.models import SourceRef


_CITATION = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited_ids: tuple[str, ...]
    unknown_ids: tuple[str, ...]
    has_citations: bool
    structurally_valid: bool
    reason: str


class CitationValidator:
    def inspect(
        self, answer_text: str, source_map: dict[str, SourceRef]
    ) -> CitationCheck:
        citations = tuple(dict.fromkeys(f"S{value}" for value in _CITATION.findall(answer_text or "")))
        unknown = tuple(sorted(set(citations).difference(source_map)))
        if unknown:
            reason = f"引用不存在：{', '.join(unknown)}"
        elif not (answer_text or "").strip():
            reason = "答案为空"
        elif not citations:
            reason = "答案没有引用检索到的资料"
        else:
            reason = "引用编号格式与来源映射有效"
        return CitationCheck(
            cited_ids=citations,
            unknown_ids=unknown,
            has_citations=bool(citations),
            structurally_valid=bool(answer_text.strip()) and bool(citations) and not unknown,
            reason=reason,
        )

    def validate(
        self, answer_text: str, source_map: dict[str, SourceRef]
    ) -> tuple[str, ...]:
        check = self.inspect(answer_text, source_map)
        if check.unknown_ids:
            raise RagGenerationError(
                f"模型返回了无法验证的引用：{', '.join(check.unknown_ids)}"
            )
        if (answer_text or "").strip() and not check.has_citations:
            raise RagGenerationError("模型答案没有引用检索到的资料")
        return check.cited_ids
