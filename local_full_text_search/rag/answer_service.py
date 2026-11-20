from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import asdict, replace

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.candidate_deduplicator import CandidateDeduplicator
from local_full_text_search.rag.citation_validator import CitationValidator
from local_full_text_search.rag.clarification import ClarificationPolicy
from local_full_text_search.rag.constraint_matcher import ConstraintMatcher
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.errors import (
    RagIndexStaleError, RagModelMissingError, VectorIndexNotReadyError,
)
from local_full_text_search.rag.evidence_policy import EvidencePolicy, SimpleEvidencePolicy
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
from local_full_text_search.rag.reranking.support import compute_answer_support
from local_full_text_search.rag.secrets import SecretStore
from local_full_text_search.rag.trace import candidate_snapshot
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore


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
        evidence_policy: EvidencePolicy | SimpleEvidencePolicy | None = None,
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
        self.query_interpreter = query_interpreter or QueryInterpreter(
            enhanced=bool(getattr(settings, 'rag_query_plan_enabled', False)),
            domain_config=getattr(settings, 'rag_domain_config', ''),
        )
        self.clarification_policy = clarification_policy or ClarificationPolicy()
        self.deduplicator = deduplicator or CandidateDeduplicator(
            jaccard_threshold=getattr(settings, "rag_dedup_jaccard_threshold", 0.92)
        )
        self.constraint_matcher = constraint_matcher or ConstraintMatcher(
            domain_adapter=getattr(self.query_interpreter, 'domain_adapter', None))
        self.context_builder = ContextBuilder()
        self.citation_validator = CitationValidator()
        # Historical policies can be injected for controlled comparisons. Normal
        # questions use ranked, complete evidence under the Context budget only.
        self.evidence_policy = evidence_policy or SimpleEvidencePolicy()

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
        text = query.text
        if not text.strip():
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
            vector_backend = str(getattr(self.settings, "rag_vector_backend", "legacy"))
            unit_index = None
            if vector_backend == "legacy":
                if getattr(bundle, "parent_vector_mode", "chroma") != "chroma":
                    raise RagIndexStaleError(
                        "当前 AI 索引未包含旧版父块向量；请切换细粒度向量 V2，或重新构建旧版索引"
                    )
                vector_path = bundle.chroma_path
            elif vector_backend == "unit_v2":
                policy = VectorUnitPolicy.from_settings(self.settings)
                vector_index_store = VectorIndexStore(self.rag_store.root)
                if getattr(bundle, "vector_unit_directory", ""):
                    unit_index = vector_index_store.attached(
                        bundle.bundle_id,
                        bundle.vector_unit_directory,
                        embedding_repo_id=self.embedding.model_id,
                        embedding_revision=self.embedding.model_revision,
                        embedding_dimension=self.embedding.dimension,
                        policy_fingerprint=policy.fingerprint,
                        index_fingerprint=bundle.vector_unit_index_fingerprint,
                    )
                else:
                    # 兼容升级前由独立 active.json 选择的向量 V2 索引。
                    unit_index = vector_index_store.active(
                        bundle.bundle_id,
                        embedding_repo_id=self.embedding.model_id,
                        embedding_revision=self.embedding.model_revision,
                        embedding_dimension=self.embedding.dimension,
                        policy_fingerprint=policy.fingerprint,
                    )
                vector_path = unit_index.chroma_path
            else:
                raise ValueError(f"不支持的向量检索后端：{vector_backend}")
            vector_store = ChromaVectorStore(vector_path)
            if unit_index is not None and vector_store.count() != unit_index.unit_count:
                vector_store.close()
                raise VectorIndexNotReadyError(
                    "向量 V2 的单元记录与 Chroma 数量不一致，请重新构建附属索引"
                )
            try:
                lexical = LexicalRetriever(
                    self.rag_store, bundle.metadata_path,
                    multi_route_enabled=bool(getattr(self.settings, "rag_lexical_multi_route_enabled", False)),
                    backend=str(getattr(self.settings, "rag_lexical_backend", "legacy")),
                    bundle=bundle,
                    title_weight=float(getattr(self.settings, "rag_lexical_title_weight", 0.2)),
                    complete_match_enabled=bool(getattr(
                        self.settings, "rag_lexical_complete_match_enabled", True
                    )),
                    k1=float(getattr(self.settings, "rag_lexical_k1", 1.2)),
                    b=float(getattr(self.settings, "rag_lexical_b", 0.75)),
                )
                vector = VectorRetriever(
                    self.rag_store, bundle, vector_store, self.embedding,
                    focused_enabled=bool(getattr(self.settings, "rag_vector_focused_query_enabled", False)),
                    backend=vector_backend, unit_index=unit_index,
                    oversample_factor=int(getattr(
                        self.settings, "rag_vector_unit_oversample_factor", 8
                    )),
                )
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
                    vector_backend=vector_backend,
                    vector_index_fingerprint=(
                        unit_index.index_fingerprint if unit_index else ""
                    ),
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
                disposition_rows = [
                    {
                        "chunk_id": item.chunk_id,
                        "status": item.status,
                        "reason": item.reason,
                        "input_rank": item.input_rank,
                        "output_rank": item.output_rank,
                        "representative_chunk_id": item.representative_chunk_id,
                        "method": item.method,
                        "similarity": item.similarity,
                        "source_merged": item.source_merged,
                    }
                    for item in deduplication.dispositions
                ]
                self._report(
                    "deduplicating",
                    "已在重排前合并重复内容块",
                    input_count=len(fused),
                    output_count=len(deduped),
                    duplicate_count=len(deduplication.duplicates),
                    grouped_count=len(deduplication.exact_groups),
                    cutoff_count=len(deduplication.cutoffs),
                    evidence_count=len(deduped),
                    source_count=sum(len(item.all_sources) for item in deduped),
                    fused_candidates=[
                        candidate_snapshot(
                            item,
                            selected=disposition.status == "kept",
                            reason=(
                                f"保留为代表候选，合并 {len(item.all_sources)} 个来源，进入重排候选池"
                                if disposition.status == "kept" else disposition.reason
                            ),
                            dedup_status=disposition.status,
                            representative_chunk_id=disposition.representative_chunk_id,
                        )
                        for item, disposition in zip(fused, deduplication.dispositions)
                    ],
                    candidate_dispositions=disposition_rows,
                    duplicate_records=[
                        {
                            "removed_chunk_id": item.removed_chunk_id,
                            "kept_chunk_id": item.kept_chunk_id,
                            "method": item.method,
                            "similarity": item.similarity,
                        }
                        for item in deduplication.duplicates
                    ],
                    deduplicated_candidates=[
                        candidate_snapshot(
                            item, dedup_status="kept",
                            representative_chunk_id=item.chunk.stable_chunk_id,
                        )
                        for item in deduped
                    ],
                )

                rerank_input = deduped[: max(1, int(query.reranker_top_k))]
                for item in deduped:
                    # Read-only historical diagnostics, never a BGE ranking input.
                    item.answer_support_score, item.answer_support_reasons = compute_answer_support(
                        plan, item
                    )
                rerank_input_ids = {item.chunk.stable_chunk_id for item in rerank_input}
                self._report(
                    "reranking",
                    "正在使用 Cross-Encoder 精排去重后的候选",
                    candidate_count=len(rerank_input),
                    reranker_top_k=query.reranker_top_k,
                    reranker_input_mode=getattr(self.reranker, "input_mode", "unrecorded"),
                    reranker_window_overlap=getattr(self.reranker, "window_overlap", None),
                    reranker_max_length=getattr(self.reranker, "max_length", None),
                    rerank_input_candidates=[
                        candidate_snapshot(
                            item, selected=item.chunk.stable_chunk_id in rerank_input_ids,
                            reason=("进入重排" if item.chunk.stable_chunk_id in rerank_input_ids
                                    else "超过重排入口 TopK"),
                        )
                        for item in deduped
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

                constraint_matching_enabled = self.evidence_policy.uses_constraints and bool(
                    getattr(self.settings, "rag_constraint_matching_enabled", True)
                )
                constrained = (
                    self.constraint_matcher.evaluate(plan, all_scored)
                    if constraint_matching_enabled
                    else tuple(all_scored)
                )
                self._report(
                    "checking_constraints",
                    "正在检查证据是否符合问题范围和场景" if constraint_matching_enabled
                    else "条件过滤已移除，保留 BGE 顺序",
                    condition_check_enabled=constraint_matching_enabled,
                    selection_mode=self.evidence_policy.selection_mode,
                    condition_ordering=("preserve_bge" if getattr(self.constraint_matcher, "preserve_order", False)
                                        else "legacy_condition_sort") if constraint_matching_enabled else "preserve_bge",
                    condition_judgement_mode=(getattr(self.constraint_matcher, "judgement_mode", "legacy")
                                              if constraint_matching_enabled else "disabled"),
                    condition_confirmation_gate=getattr(self.evidence_policy, "require_confirmed_conditions", True),
                    constraint_candidates=[
                        candidate_snapshot(
                            item,
                            selected=bool(
                                not constraint_matching_enabled
                                or (item.constraint_decision and item.constraint_decision.accepted)
                            ),
                            reason=(
                                item.constraint_decision.reason if item.constraint_decision
                                else "条件检查已关闭，候选保留"
                            ),
                        )
                        for item in constrained
                    ],
                    required_constraint_count=sum(item.is_hard for item in plan.constraints),
                    soft_constraint_count=sum(
                        item.strength.value == "soft" for item in plan.constraints
                    ),
                    conflict_count=sum(
                        1 for item in constrained
                        if constraint_matching_enabled and item.constraint_decision
                        and item.constraint_decision.required_conflict_count
                    ),
                )

                competing = (
                    self.clarification_policy.evaluate_competing_stages(plan, constrained)
                    if clarification_enabled and constraint_matching_enabled
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
                    selection_mode=self.evidence_policy.selection_mode,
                    fixed_top_k_applied=self.evidence_policy.uses_constraints,
                    constraint_filter_applied=constraint_matching_enabled,
                    decision_reason=decision.reason,
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
                                    (
                                        "软条件冲突，降权后通过证据门禁"
                                        if item.constraint_decision
                                        and item.constraint_decision.soft_conflict_count
                                    else "保留 BGE 顺序，待 Context 长度预算选入"
                                    if not self.evidence_policy.uses_constraints else "通过证据门禁"
                                    )
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
                    preserve_complete_chunks=self.evidence_policy.preserve_complete_chunks,
                )
                if plan.adapter_active:
                    from local_full_text_search.rag.context_builder import organize_context
                    context = organize_context(context, plan, decision.rejected)
                    self._report('structured_evidence', '绑定实际正文的事实、范围与条件',
                        parsed_request=plan.parsed_request, evidence_facts=list(context.evidence_facts),
                        context_sections=context.sections, special_rule_invocations=[])
                    from local_full_text_search.rag.source_consensus import attach_context_consensus
                    context, source_consensus = attach_context_consensus(context)
                    self._report('source_consensus', '核对实际引用的跨文档事实一致性', source_consensus=source_consensus)
                    from local_full_text_search.rag.identifier_evidence import coverage
                    from local_full_text_search.rag.evidence_contract import context_audit
                    self._report('answer_slot_coverage', '核对实际上下文中的答案槽位',
                                 answer_slot_coverage=coverage(plan, context.entries),
                                 context_contract_audit=context_audit(plan,context.entries))
                if not context.sources:
                    answer = self._insufficient(trace_id, "empty_context", stale_warnings)
                    return self._retrieval_result(answer) if retrieval_only else answer

                if retrieval_only:
                    self._report(
                        "retrieval_completed",
                        "本地检索与证据整理完成，按检索模式跳过 LLM",
                        evidence_count=len(context.source_map),
                        source_count=len(context.sources),
                        context_chars=len(context.text),
                        llm_context=context.text,
                        context_entries=list(context.entries),
                        context_excluded=list(context.excluded_entries),
                        context_sources=[
                            {
                                "source_id": entry.get("source_id"),
                                "chunk_id": entry.get("chunk_id"),
                                "filename": entry.get("filename"),
                                "location": entry.get("location"),
                                "source_count": entry.get("source_count"),
                                "sources": entry.get("sources"),
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
                    evidence_count=len(context.source_map),
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
                        for source_id, refs in context.source_map.items()
                        for source in refs
                    ],
                    context_entries=list(context.entries),
                    context_excluded=list(context.excluded_entries),
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
                if plan.adapter_active:
                    from local_full_text_search.rag.source_consensus import generation_consensus_contract
                    plan_summary += generation_consensus_contract(source_consensus)
                request = GenerationRequest(
                    question=text,
                    context=context.text,
                    query_plan_summary=plan_summary,
                    **common,
                )
                token.throw_if_cancelled()
                assess = getattr(self.gateway, "assess_clarification", None)
                if (clarification_enabled
                        and bool(getattr(self.settings, "rag_semantic_clarification_enabled", False))
                        and callable(assess)):
                    assessment = None
                    self._report("assessing_clarification", "正在核对是否缺少决定处理路径的信息")
                    try:
                        assessment = assess(request, token)
                    except Exception:
                        token.throw_if_cancelled()
                        self._report("clarification_assessment_unavailable", "未获得有效澄清判断，继续按现有证据回答")
                    self._report(
                        "clarification_assessed", "澄清条件判断完成",
                        decision_available=assessment is not None,
                        needs_clarification=bool(assessment and assessment.needs_clarification),
                        missing_fields=list(assessment.missing_fields) if assessment else [],
                        evidence=list(assessment.evidence) if assessment else [],
                        decision_reason=assessment.reason if assessment else "结构或引用未通过核验",
                        decision_type=assessment.decision_type if assessment else "",
                        request_quote=assessment.request_quote if assessment else "",
                    )
                    if assessment and assessment.needs_clarification:
                        return replace(
                            self._clarification(trace_id, assessment.question, assessment.reason, stale_warnings),
                            provider_name="OpenAI Compatible", model_name=request.model,
                        )
                self._report(
                    "generating", "正在根据证据生成答案初稿",
                    evidence_count=len(context.source_map), source_count=len(context.sources),
                    context_chars=len(context.text), model=request.model,
                    api_style=request.api_style, timeout_seconds=request.timeout_seconds,
                    provider_host=provider_host(request.api_base_url),
                    api_request=provider_request_payload(request),
                )
                if plan.adapter_active and bool(getattr(self.settings, 'rag_structured_answer_enabled', False)):
                    from local_full_text_search.rag.generation.structured import generate_structured
                    model_answer = generate_structured(self.gateway, request, plan, context, token,
                        lambda data: self._report('structured_answer_plan', '核验逐条答案计划与实际引文', **data))
                else:
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
                cited_sources = tuple(source for item in cited_ids for source in context.source_map[item])
                cited_source_ids = tuple(
                    item for item in cited_ids for _source in context.source_map[item]
                )
                self._report(
                    "completed", "答案与引用已验证",
                    citation_count=len(cited_ids),
                    evidence_count=len(cited_ids),
                    source_count=len(cited_sources),
                    review_verdict=review_result.verdict if review_result else "disabled",
                )
                return RagAnswer(
                    answer_text=final_text,
                    sources=cited_sources,
                    insufficient_evidence=False,
                    warnings=tuple(warnings),
                    source_ids=cited_source_ids,
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
        summary = f"意图={plan.intent}; 条件={constraints}; 检索词={', '.join(plan.lexical_terms)}"
        if plan.adapter_active:
            ids = [e.text for e in plan.entity_details if e.type in ('alarm_code', 'identifier_candidate', 'instrument_model')]
            summary += f"; 用户指定标识={', '.join(ids)}; 答案目标={', '.join(s.name for s in plan.answer_requirements)}"
            summary += '; 分别回答有依据的问项，缺乏依据的问项说明缺口。标识相似不代表相同，禁止套用其他标识的处理。解析是问题线索，不是事实证据。'
            from local_full_text_search.rag.evidence_contract import contract_summary
            summary += contract_summary(plan)
        return summary

    @staticmethod
    def _plan_snapshot(plan) -> dict[str, object]:
        return {
            "original_text": plan.original_text,
            "intent": plan.intent,
            "entities": list(plan.entities),
            "entity_details": [asdict(e) for e in plan.entity_details],
            "answer_requirements": [asdict(e) for e in plan.answer_requirements],
            "sub_queries": [asdict(e) for e in plan.sub_queries],
            "retrieval_hints": [asdict(e) for e in plan.retrieval_hints],
            "unknown_entities": [asdict(e) for e in plan.unknown_entities],
            "domain_config": plan.domain_config,
            "adapter_active": plan.adapter_active,
            "strategies": [asdict(e) for e in plan.strategies],
            "scope_constraints": [asdict(e) for e in plan.scope_constraints],
            "protected_evidence": [asdict(e) for e in plan.protected_evidence],
            "method_branches": [asdict(e) for e in plan.method_branches],
            "ambiguities": [asdict(e) for e in plan.ambiguities],
            "plan_diagnostics": [asdict(e) for e in plan.plan_diagnostics],
            "constraints": [
                {
                    "kind": item.kind.value,
                    "value": item.value,
                    "required": item.required,
                    "source_text": item.source_text,
                    "polarity": item.polarity.value,
                    "strength": item.strength.value,
                    "confidence": item.confidence,
                    "source": item.source,
                    "reason": item.reason,
                    "subject": item.subject,
                    "source_start": item.source_start,
                    "source_end": item.source_end,
                    "role": item.role,
                }
                for item in plan.constraints
            ],
            "lexical_terms": list(plan.lexical_terms),
            "lexical_expression": plan.lexical_expression,
            "semantic_query": plan.semantic_query,
            "spans": [
                {"text": item.text, "start": item.start, "end": item.end, "features": list(item.features)}
                for item in plan.spans
            ],
            "transformations": [
                {"operation": item.operation, "source_text": item.source_text, "result_text": item.result_text, "reason": item.reason}
                for item in plan.transformations
            ],
            "lexical_variants": [
                {"route_id": item.route_id, "expression": item.expression, "signal_texts": list(item.signal_texts)}
                for item in plan.lexical_variants
            ],
            "semantic_variants": [
                {"route_id": item.route_id, "query_text": item.query_text}
                for item in plan.semantic_variants
            ],
            "negated_constraints": [
                {"kind": item.kind.value, "value": item.value, "source_text": item.source_text}
                for item in plan.constraints if item.polarity.value == "negated"
            ],
            "unresolved_ambiguities": [
                {"kind": item.kind.value, "value": item.value, "reason": item.reason}
                for item in plan.constraints if item.strength.value == "ambiguous"
                and item.polarity.value != "negated"
            ],
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
