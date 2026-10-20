from __future__ import annotations

import uuid
from collections.abc import Callable

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.clarification import ClarificationPolicy
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.errors import RagIndexStaleError, RagModelMissingError
from local_full_text_search.rag.evidence_policy import EvidencePolicy
from local_full_text_search.rag.generation.base import AnswerReviewRequest, GenerationRequest
from local_full_text_search.rag.generation.openai_compatible import (
    OpenAICompatibleGateway,
    provider_request_payload,
    provider_review_payload,
)
from local_full_text_search.rag.diagnostics import provider_host
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import (
    AnswerAudit,
    RagAnswer,
    RagAnswerStatus,
    RagQuery,
    RagRetrievalResult,
)
from local_full_text_search.rag.query_interpreter import QueryInterpreter
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever
from local_full_text_search.rag.retrieval.lexical import LexicalRetriever
from local_full_text_search.rag.retrieval.vector import VectorRetriever
from local_full_text_search.rag.reranking.local_bge import LocalBgeReranker
from local_full_text_search.rag.secrets import SecretStore
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


class AnswerService:
    def __init__(
        self,
        source_generation_store: IndexGenerationStore,
        rag_store: RagIndexStore,
        embedding,
        reranker,
        gateway,
        secret_store: SecretStore,
        settings,
        *,
        query_interpreter: QueryInterpreter | None = None,
        clarification_policy: ClarificationPolicy | None = None,
        deduplicator: CandidateDeduplicator | None = None,
        constraint_matcher: ConstraintMatcher | None = None,
        evidence_policy: EvidencePolicy | None = None,
        progress_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.source_generation_store = source_generation_store
        self.rag_store = rag_store
        self.embedding = embedding
        self.reranker = reranker
        self.gateway = gateway
        self.secret_store = secret_store
        self.settings = settings
        self.progress_callback = progress_callback
        self.query_interpreter = query_interpreter or QueryInterpreter()
        self.clarification_policy = clarification_policy or ClarificationPolicy()
        self.deduplicator = deduplicator or CandidateDeduplicator(
            jaccard_threshold=getattr(settings, "rag_dedup_jaccard_threshold", 0.92)
        )
        self.constraint_matcher = constraint_matcher or ConstraintMatcher()
        self.context_builder = ContextBuilder()
        self.citation_validator = CitationValidator()
        self.evidence_policy = evidence_policy or EvidencePolicy(
            min_vector_score=settings.rag_min_vector_score,
            min_rerank_score=settings.rag_min_rerank_score,
            max_per_document=getattr(settings, "rag_max_chunks_per_document", 3),
        )

    def answer(
        self, query: RagQuery, token: CancelToken, *, trace_id: str | None = None
    ) -> RagAnswer:
        result = self._execute(query, token, trace_id=trace_id, retrieval_only=False)
        if not isinstance(result, RagAnswer):
            raise TypeError("完整问答流程未返回 RagAnswer")
        return result

    def retrieve(
        self, query: RagQuery, token: CancelToken, *, trace_id: str | None = None
    ) -> RagRetrievalResult:
        """运行与正式问答相同的本地链路，在读取 API Key 前结束。"""
        result = self._execute(query, token, trace_id=trace_id, retrieval_only=True)
        if not isinstance(result, RagRetrievalResult):
            raise TypeError("仅检索流程未返回 RagRetrievalResult")
        return result

    def _execute(
        self,
        query: RagQuery,
        token: CancelToken,
        *,
        trace_id: str | None,
        retrieval_only: bool,
    ) -> RagAnswer | RagRetrievalResult:
        text = query.text.strip()
        if not text:
            raise ValueError("问题不能为空")
        trace_id = trace_id or f"rag-{uuid.uuid4().hex}"
        self._report(
            "checking_index", "正在确认已发布的 AI 索引",
            trace_id=trace_id,
            query_limits={
                "lexical_top_k": query.lexical_top_k,
                "vector_top_k": query.vector_top_k,
                "fusion_top_k": query.fusion_top_k,
                "reranker_top_k": query.reranker_top_k,
                "final_top_k": query.final_top_k,
            },
        )
        self._report("query_interpreting", "正在识别问题中的检索词和业务条件")
        plan = self.query_interpreter.interpret(text, token)
        plan_snapshot = self._plan_snapshot(plan)
        self._report(
            "query_interpreted",
            "已整理关键词、语义问题和业务条件",
            query_plan=plan_snapshot,
        )
        self._report("clarification_check", "正在检查问题是否缺少必要信息")
        clarification_enabled = bool(getattr(self.settings, "rag_clarification_enabled", True))
        clarification = (
            self.clarification_policy.evaluate(plan)
            if clarification_enabled
            else None
        )
        if clarification is not None and clarification.required:
            answer = self._clarification(trace_id, clarification.question, clarification.reason)
            return self._retrieval_result(answer) if retrieval_only else answer

        with self.source_generation_store.acquire_active() as (source, _lease):
            bundle = self.rag_store.active()
            stale_warnings = (
                ("全文索引已有更新；本次回答仍基于已发布的旧 AI 索引，可能不包含最新资料。",)
                if (bundle.verified_source_generation or bundle.source_generation) != source.generation
                else ()
            )
            if (
                bundle.embedding_repo_id != self.embedding.model_id
                or bundle.embedding_revision != self.embedding.model_revision
                or bundle.embedding_dimension != self.embedding.dimension
            ):
                raise RagIndexStaleError("向量模型与 AI 索引不匹配，请重新构建 AI 索引")
            vector_store = ChromaVectorStore(bundle.chroma_path)
            try:
                lexical = LexicalRetriever(self.rag_store, bundle.metadata_path)
                vector = VectorRetriever(self.rag_store, bundle, vector_store, self.embedding)
                hybrid = HybridRetriever(
                    lexical,
                    vector,
                    rrf_k=self.settings.rag_rrf_k,
                    progress_callback=self._report,
                )
                self._report(
                    "retrieving",
                    "正在执行关键词和向量混合检索",
                    source_generation=bundle.source_generation,
                    active_source_generation=source.generation,
                    rag_bundle_id=bundle.bundle_id,
                )
                fused = hybrid.retrieve(query, token, plan)
                token.throw_if_cancelled()
                if not fused:
                    answer = self._insufficient(trace_id, "no_candidates", stale_warnings)
                    return self._retrieval_result(answer) if retrieval_only else answer

                deduplication = self.deduplicator.deduplicate(
                    fused, limit=max(1, int(query.fusion_top_k))
                )
                deduped = list(deduplication.candidates)
                deduped_ids = {item.chunk.stable_chunk_id for item in deduped}
                self._report(
                    "deduplicating",
                    "已在重排前合并重复内容块",
                    input_count=len(fused),
                    output_count=len(deduped),
                    duplicate_count=len(deduplication.duplicates),
                    fused_candidates=[
                        candidate_snapshot(
                            item,
                            selected=item.chunk.stable_chunk_id in deduped_ids,
                            reason=(
                                "保留并送入去重后候选"
                                if item.chunk.stable_chunk_id in deduped_ids
                                else f"重复内容，保留 {item.duplicate_of or '更高排名候选'}"
                            ),
                        )
                        for item in fused
                    ],
                    duplicate_records=[
                        {
                            "removed_chunk_id": item.removed_chunk_id,
                            "kept_chunk_id": item.kept_chunk_id,
                            "method": item.method,
                            "similarity": item.similarity,
                        }
                        for item in deduplication.duplicates
                    ],
                    deduplicated_candidates=[candidate_snapshot(item) for item in deduped],
                )

                rerank_input = deduped[: max(1, int(query.reranker_top_k))]
                self._report(
                    "reranking",
                    "正在使用 Cross-Encoder 精排去重后的候选",
                    candidate_count=len(rerank_input),
                    reranker_top_k=query.reranker_top_k,
                    fused_candidates=[
                        candidate_snapshot(item, selected=item in rerank_input)
                        for item in fused
                    ],
                )
                reranked = self.reranker.rerank(
                    plan.original_text,
                    rerank_input,
                    max(1, int(query.reranker_top_k)),
                    token,
                )
                all_scored = list(getattr(self.reranker, "last_scored_candidates", reranked))
                reranked_ids = {item.chunk.stable_chunk_id for item in reranked}
                self._report(
                    "reranked",
                    "重排完成，尚未按最终 TopK 截断",
                    reranked_count=len(all_scored),
                    reranked_candidates=[
                        candidate_snapshot(
                            item,
                            selected=item.chunk.stable_chunk_id in reranked_ids,
                            reason="进入条件检查" if item.chunk.stable_chunk_id in reranked_ids else "重排候选未选中",
                        )
                        for item in all_scored
                    ],
                )

                constraint_matching_enabled = bool(
                    getattr(self.settings, "rag_constraint_matching_enabled", True)
                )
                constrained = (
                    self.constraint_matcher.evaluate(plan, all_scored)
                    if constraint_matching_enabled
                    else tuple(all_scored)
                )
                self._report(
                    "checking_constraints",
                    "正在检查证据是否符合问题范围和场景",
                    constraint_candidates=[
                        candidate_snapshot(
                            item,
                            selected=bool(
                                item.constraint_decision
                                and item.constraint_decision.required_conflict_count == 0
                            ),
                            reason=(item.constraint_decision.reason if item.constraint_decision else "未判断"),
                        )
                        for item in constrained
                    ],
                    required_constraint_count=len(plan.constraints),
                    conflict_count=sum(
                        1 for item in constrained
                        if item.constraint_decision and item.constraint_decision.required_conflict_count
                    ),
                )

                competing = (
                    self.clarification_policy.evaluate_competing_stages(plan, constrained)
                    if clarification_enabled
                    else None
                )
                if competing is not None and competing.required:
                    answer = self._clarification(
                        trace_id, competing.question, competing.reason, stale_warnings
                    )
                    return self._retrieval_result(answer) if retrieval_only else answer

                decision = self.evidence_policy.evaluate(
                    query,
                    constrained,
                    plan=plan if constraint_matching_enabled else None,
                )
                accepted_ids = {item.chunk.stable_chunk_id for item in decision.accepted}
                rejected_reasons = dict(decision.rejected)
                accepted_ranks = {
                    item.chunk.stable_chunk_id: rank
                    for rank, item in enumerate(decision.accepted, 1)
                }
                evidence_ordered = [
                    *decision.accepted,
                    *(
                        item for item in constrained
                        if item.chunk.stable_chunk_id not in accepted_ids
                    ),
                ]
                self._report(
                    "checking_evidence",
                    "正在筛选可用于回答的最终证据",
                    reranked_count=len(constrained),
                    accepted_count=len(decision.accepted),
                    min_vector_score=self.evidence_policy.min_vector_score,
                    min_rerank_score=self.evidence_policy.min_rerank_score,
                    evidence_candidates=[
                        {
                            **candidate_snapshot(
                                item,
                                selected=item.chunk.stable_chunk_id in accepted_ids,
                                reason=(
                                    "通过证据门禁"
                                    if item.chunk.stable_chunk_id in accepted_ids
                                    else rejected_reasons.get(item.chunk.stable_chunk_id)
                                    or "未进入最终 TopK"
                                ),
                            ),
                            "final_rank": accepted_ranks.get(item.chunk.stable_chunk_id),
                        }
                        for item in evidence_ordered
                    ],
                )
                if decision.insufficient:
                    answer = self._insufficient(trace_id, decision.reason, stale_warnings)
                    return self._retrieval_result(answer) if retrieval_only else answer

                self._report(
                    "building_context", "正在整理引用上下文",
                    accepted_count=len(decision.accepted),
                )
                context = self.context_builder.build(
                    decision.accepted,
                    max_chars=self.settings.rag_context_max_chars,
                )
                if not context.sources:
                    answer = self._insufficient(trace_id, "empty_context", stale_warnings)
                    return self._retrieval_result(answer) if retrieval_only else answer

                if retrieval_only:
                    self._report(
                        "retrieval_completed",
                        "本地检索与证据整理完成，按检索模式跳过 LLM",
                        source_count=len(context.sources),
                        context_chars=len(context.text),
                        llm_context=context.text,
                        context_sources=[
                            {
                                "source_id": entry.get("source_id"),
                                "chunk_id": entry.get("chunk_id"),
                                "filename": entry.get("filename"),
                                "location": entry.get("location"),
                                "included_text": entry.get("included_text"),
                                "truncated": entry.get("truncated", False),
                                "reason": entry.get("reason", ""),
                            }
                            for entry in context.entries
                        ],
                    )
                    return RagRetrievalResult(
                        trace_id=trace_id,
                        status=RagAnswerStatus.ANSWERED,
                        sources=context.sources,
                    )

                self._report(
                    "credential_read",
                    "正在读取已保存的 API Key",
                    source_count=len(context.sources),
                    context_chars=len(context.text),
                    llm_context=context.text,
                    context_sources=[
                        {
                            "source_id": source_id,
                            "chunk_id": source.stable_chunk_id,
                            "filename": source.filename,
                            "location": source.location_text,
                            "file_path": source.file_path,
                        }
                        for source_id, source in context.source_map.items()
                    ],
                    context_entries=list(context.entries),
                )
                api_key = self.secret_store.retrieve(self.settings.rag_api_secret_ref)
                if not api_key:
                    raise RagModelMissingError("请先在设置页填写并保存 LLM API Key")
                common = {
                    "model": self.settings.rag_api_model,
                    "api_base_url": self.settings.rag_api_base_url,
                    "api_style": self.settings.rag_api_style,
                    "api_key": api_key,
                    "timeout_seconds": self.settings.rag_api_timeout_seconds,
                }
                plan_summary = self._plan_summary(plan)
                request = GenerationRequest(
                    question=text,
                    context=context.text,
                    query_plan_summary=plan_summary,
                    **common,
                )
                self._report(
                    "generating",
                    "正在根据证据生成答案初稿",
                    source_count=len(context.sources),
                    context_chars=len(context.text),
                    model=request.model,
                    api_style=request.api_style,
                    timeout_seconds=request.timeout_seconds,
                    provider_host=provider_host(request.api_base_url),
                    api_request=provider_request_payload(request),
                )
                token.throw_if_cancelled()
                model_answer = self.gateway.generate(request, token)
                citation_check = self.citation_validator.inspect(
                    model_answer.text, context.source_map
                )
                self._report(
                    "draft_generated",
                    "答案初稿已生成并完成引用格式预检查",
                    answer_chars=len(model_answer.text),
                    citation_check={
                        "cited_ids": list(citation_check.cited_ids),
                        "unknown_ids": list(citation_check.unknown_ids),
                        "structurally_valid": citation_check.structurally_valid,
                        "reason": citation_check.reason,
                    },
                    draft_answer=model_answer.text,
                )

                review_result = None
                warnings = list(stale_warnings)
                if bool(getattr(self.settings, "rag_answer_review_enabled", False)):
                    review_request = AnswerReviewRequest(
                        question=text,
                        query_plan_summary=plan_summary,
                        context=context.text,
                        draft_answer=model_answer.text,
                        citation_check_summary=citation_check.reason,
                        allowed_source_ids=tuple(context.source_map),
                        max_output_tokens=int(
                            getattr(self.settings, "rag_answer_review_max_output_tokens", 1400)
                        ),
                        **common,
                    )
                    self._report(
                        "reviewing",
                        "正在逐条核对答案结论与引用证据",
                        model=review_request.model,
                        api_style=review_request.api_style,
                        timeout_seconds=review_request.timeout_seconds,
                        provider_host=provider_host(review_request.api_base_url),
                        review_request=provider_review_payload(review_request),
                    )
                    token.throw_if_cancelled()
                    try:
                        review_result = self.gateway.review_answer(review_request, token)
                    except Exception as exc:
                        token.throw_if_cancelled()
                        if bool(getattr(self.settings, "rag_answer_review_strict", True)):
                            raise
                        warnings.append("LLM 答案二次复核失败，以下答案未经语义复核。")
                        self._report(
                            "review_failed_fallback",
                            "答案复核失败，按设置回退到初稿",
                            error_type=type(exc).__name__,
                        )
                        review_result = None
                    if review_result is None:
                        final_text = model_answer.text.strip()
                    else:
                        self._report(
                            "review_completed",
                            "LLM 答案复核已完成",
                            verdict=review_result.verdict,
                            claim_count=len(review_result.claims),
                            supported_count=sum(1 for claim in review_result.claims if claim.status == "supported"),
                            review_summary=review_result.summary,
                            review_claims=[
                                {
                                    "claim": claim.claim,
                                    "citations": list(claim.citations),
                                    "status": claim.status,
                                    "reason": claim.reason,
                                }
                                for claim in review_result.claims
                            ],
                            reviewed_answer=review_result.final_answer,
                        )
                        final_text = review_result.final_answer
                else:
                    warnings.append("未启用 LLM 答案二次复核。")
                    final_text = model_answer.text.strip()

                if final_text == "资料中没有足够依据。":
                    self._report(
                        "completed",
                        "审查后判定现有证据不足",
                        review_verdict=review_result.verdict if review_result else "disabled_or_failed",
                    )
                    return RagAnswer(
                        answer_text=final_text,
                        sources=(),
                        insufficient_evidence=True,
                        warnings=tuple(warnings),
                        trace_id=trace_id,
                        provider_name=(review_result.provider_name if review_result else model_answer.provider_name),
                        model_name=model_answer.model_name,
                        status=RagAnswerStatus.INSUFFICIENT,
                        audit=AnswerAudit(model_answer.text, review_result, citation_check.reason),
                    )

                self._report(
                    "validating", "正在执行最终引用硬校验",
                    answer_chars=len(final_text),
                    review_enabled=bool(getattr(self.settings, "rag_answer_review_enabled", False)),
                )
                cited_ids = self.citation_validator.validate(final_text, context.source_map)
                cited_sources = tuple(context.source_map[item] for item in cited_ids)
                self._report(
                    "completed", "答案与引用已验证",
                    citation_count=len(cited_ids),
                    review_verdict=review_result.verdict if review_result else "disabled",
                )
                return RagAnswer(
                    answer_text=final_text,
                    sources=cited_sources,
                    insufficient_evidence=False,
                    warnings=tuple(warnings),
                    source_ids=cited_ids,
                    trace_id=trace_id,
                    provider_name=(review_result.provider_name if review_result else model_answer.provider_name),
                    model_name=model_answer.model_name,
                    status=RagAnswerStatus.ANSWERED,
                    audit=AnswerAudit(model_answer.text, review_result, citation_check.reason),
                )
            finally:
                vector_store.close()

    @staticmethod
    def _retrieval_result(answer: RagAnswer) -> RagRetrievalResult:
        return RagRetrievalResult(
            trace_id=answer.trace_id,
            status=answer.status,
            sources=answer.sources,
            insufficient_evidence=answer.insufficient_evidence,
            clarification_question=answer.clarification_question,
            reason=(answer.warnings[-1] if answer.warnings else ""),
        )

    @staticmethod
    def _plan_summary(plan) -> str:
        constraints = ", ".join(
            f"{item.kind.value}={item.value}（{item.source_text or '规则识别'}）"
            for item in plan.constraints
        ) or "无明确条件"
        return f"意图={plan.intent}; 条件={constraints}; 检索词={', '.join(plan.lexical_terms)}"

    @staticmethod
    def _plan_snapshot(plan) -> dict[str, object]:
        return {
            "intent": plan.intent,
            "entities": list(plan.entities),
            "constraints": [
                {
                    "kind": item.kind.value,
                    "value": item.value,
                    "required": item.required,
                    "source_text": item.source_text,
                }
                for item in plan.constraints
            ],
            "lexical_terms": list(plan.lexical_terms),
            "lexical_expression": plan.lexical_expression,
            "semantic_query": plan.semantic_query,
            "missing_fields": list(plan.missing_fields),
            "analyzer_version": plan.analyzer_version,
        }

    def _clarification(
        self,
        trace_id: str,
        question: str,
        reason: str,
        warnings: tuple[str, ...] = (),
    ) -> RagAnswer:
        self._report(
            "needs_clarification",
            "需要补充信息后才能安全回答",
            reason=reason,
            clarification_question=question,
        )
        return RagAnswer(
            answer_text=question,
            sources=(),
            insufficient_evidence=False,
            warnings=warnings,
            trace_id=trace_id,
            status=RagAnswerStatus.NEEDS_CLARIFICATION,
            clarification_question=question,
        )

    def _insufficient(
        self, trace_id: str, reason: str, warnings: tuple[str, ...] = ()
    ) -> RagAnswer:
        self._report("insufficient", "资料中没有足够依据，未调用 LLM", reason=reason)
        return RagAnswer(
            answer_text="资料中没有足够依据。",
            sources=(),
            insufficient_evidence=True,
            warnings=(*warnings, reason),
            trace_id=trace_id,
            status=RagAnswerStatus.INSUFFICIENT,
        )

    def _report(self, stage: str, message: str, **metrics: object) -> None:
        if self.progress_callback:
            self.progress_callback({"stage": stage, "message": message, **metrics})
