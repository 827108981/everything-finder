"""严格解析 LLM 的答案审查 JSON，避免把自由格式直接交给 UI。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from local_full_text_search.rag.errors import RagGenerationError
from local_full_text_search.rag.models import AnswerReviewResult, ClaimReview


_CITATION = re.compile(r"\[S(\d+)\]")
_ALLOWED_VERDICTS = {"pass", "revised", "insufficient"}
_ALLOWED_CLAIM_STATUS = {"supported", "contradicted", "not_supported"}
_INSUFFICIENT_TEXT = "资料中没有足够依据。"


@dataclass(frozen=True, slots=True)
class ParsedAnswerReview:
    result: AnswerReviewResult
    raw_json: str


class AnswerReviewParser:
    def parse(
        self,
        raw_text: str,
        *,
        allowed_source_ids: set[str] | tuple[str, ...],
        provider_name: str = "",
        model_name: str = "",
    ) -> ParsedAnswerReview:
        payload_text = self._extract_json(str(raw_text or ""))
        try:
            payload = json.loads(payload_text)
        except (json.JSONDecodeError, TypeError) as exc:
            raise RagGenerationError("答案审查模型返回的内容不是有效 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {
            "verdict", "summary", "claims", "final_answer"
        }:
            raise RagGenerationError("答案审查 JSON 字段不完整或包含未支持字段")

        verdict = payload.get("verdict")
        summary = payload.get("summary")
        final_answer = payload.get("final_answer")
        raw_claims = payload.get("claims")
        if verdict not in _ALLOWED_VERDICTS:
            raise RagGenerationError("答案审查 verdict 不受支持")
        if not isinstance(summary, str) or len(summary) > 2_000:
            raise RagGenerationError("答案审查摘要格式无效")
        if not isinstance(final_answer, str) or not final_answer.strip() or len(final_answer) > 20_000:
            raise RagGenerationError("答案审查 final_answer 为空或超出长度限制")
        if not isinstance(raw_claims, list) or len(raw_claims) > 30:
            raise RagGenerationError("答案审查 claims 格式无效或数量超限")

        allowed = set(allowed_source_ids)
        if verdict != "insufficient" and not raw_claims:
            raise RagGenerationError("答案审查没有返回可核验的结论列表")
        claims: list[ClaimReview] = []
        for item in raw_claims:
            if not isinstance(item, dict) or set(item) != {
                "claim", "citations", "status", "reason"
            }:
                raise RagGenerationError("答案审查 claim 字段格式无效")
            claim = item.get("claim")
            citations = item.get("citations")
            status = item.get("status")
            reason = item.get("reason")
            if not isinstance(claim, str) or not claim.strip() or len(claim) > 2_000:
                raise RagGenerationError("答案审查 claim 内容无效")
            if not isinstance(reason, str) or len(reason) > 2_000:
                raise RagGenerationError("答案审查 claim 理由无效")
            if status not in _ALLOWED_CLAIM_STATUS:
                raise RagGenerationError("答案审查 claim 状态不受支持")
            if not isinstance(citations, list) or len(citations) > 20:
                raise RagGenerationError("答案审查 claim 引用格式无效")
            citation_ids = tuple(str(value) for value in citations)
            if any(value not in allowed for value in citation_ids):
                raise RagGenerationError("答案审查引用了上下文中不存在的来源编号")
            if status == "supported" and not citation_ids:
                raise RagGenerationError("标记为 supported 的结论没有引用来源")
            claims.append(ClaimReview(claim.strip(), citation_ids, status, reason.strip()))

        if verdict == "insufficient":
            if final_answer.strip() != _INSUFFICIENT_TEXT:
                raise RagGenerationError("证据不足审查结果必须使用固定答案")
        else:
            compact_final = re.sub(r"\s+", "", final_answer)
            for claim in claims:
                if claim.status != "supported" and re.sub(r"\s+", "", claim.claim) in compact_final:
                    raise RagGenerationError("未获支持的结论仍出现在最终答案中")

            final_ids = {f"S{value}" for value in _CITATION.findall(final_answer)}
            if not final_ids:
                raise RagGenerationError("审查后的答案没有引用来源")
            if final_ids.difference(allowed):
                raise RagGenerationError("审查后的答案包含无效引用编号")
            supported_ids = {
                citation
                for claim in claims
                if claim.status == "supported"
                for citation in claim.citations
            }
            if final_ids.difference(supported_ids):
                raise RagGenerationError("最终答案引用未关联到任何 supported 结论")
            if verdict == "pass" and any(claim.status != "supported" for claim in claims):
                raise RagGenerationError("verdict=pass 但仍存在未获支持的结论")

        result = AnswerReviewResult(
            verdict=verdict,
            summary=summary.strip(),
            claims=tuple(claims),
            final_answer=final_answer.strip(),
            provider_name=provider_name,
            model_name=model_name,
        )
        return ParsedAnswerReview(result, payload_text)

    @staticmethod
    def _extract_json(value: str) -> str:
        text = value.strip()
        if text.startswith("```json") and text.endswith("```"):
            text = text[len("```json") : -3].strip()
        elif text.startswith("```") or text.endswith("```"):
            raise RagGenerationError("答案审查只允许单层 JSON 代码围栏")
        if not text.startswith("{") or not text.endswith("}"):
            raise RagGenerationError("答案审查响应包含 JSON 以外的文本")
        return text
