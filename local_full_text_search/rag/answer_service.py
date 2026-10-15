from __future__ import annotations

import uuid
from collections.abc import Callable

from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.context_builder import ContextBuilder
from local_full_text_search.rag.citations import CitationValidationError, CitationValidator
from local_full_text_search.rag.embeddings.base import EmbeddingProvider
from local_full_text_search.rag.generation.base import LLMGateway, LLMUnavailableError
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.models import RagAnswer, RagIndexNotReadyError, RagProgress, RagQuery
from local_full_text_search.rag.policy import EvidencePolicy
from local_full_text_search.rag.retrieval.hybrid import HybridRetriever


class AnswerService:
    def __init__(
        self,
        generation_base_path,
        *,
        hybrid: HybridRetriever | None = None,
        context_builder: ContextBuilder | None = None,
        citation_validator: CitationValidator | None = None,
        context_max_tokens: int = 2300,
        answer_max_tokens: int = 512,
    ) -> None:
        self.generation_base_path = generation_base_path
        self.search_store = IndexGenerationStore(generation_base_path)
        self.rag_store = RagIndexStore(generation_base_path)
        self.hybrid = hybrid or HybridRetriever()
        self.context_builder = context_builder or ContextBuilder(context_max_tokens)
        self.citation_validator = citation_validator or CitationValidator()
        self.answer_max_tokens = max(64, int(answer_max_tokens))
        self.policy = EvidencePolicy()

    def answer(
        self,
        query: RagQuery,
        *,
        embedding: EmbeddingProvider,
        llm: LLMGateway,
        token: CancelToken,
        progress: Callable[[RagProgress], None] | None = None,
    ) -> RagAnswer:
        trace_id = uuid.uuid4().hex
        active = self.search_store.active()
        if active is None:
            return RagAnswer(
                "index_stale", "搜索索引尚未发布，暂时无法进行智能问答。", (),
                trace_id, None,
            )
        try:
            with self.search_store.acquire_active() as (source, _source_lease):
                token.throw_if_cancelled()
                with self.rag_store.acquire_matching(source.generation, embedding.identity) as handle:
                    self._emit(progress, "retrieving", "正在检索相关证据")
                    ranked = self.hybrid.retrieve(
                        handle.connection,
                        query,
                        embedding,
                        token,
                        top_n=max(query.top_k, 20),
                    )
                    token.throw_if_cancelled()
                    if not ranked:
                        return self.policy.insufficient(trace_id, source.generation)
                    context = self.context_builder.build(query, ranked, llm)
                    if not context.citations:
                        return self.policy.insufficient(trace_id, source.generation)
                    self._emit(progress, "generating", "正在依据证据生成回答")
                    raw = llm.generate(
                        context.messages,
                        max_tokens=self.answer_max_tokens,
                        cancel_token=token,
                    )
                    try:
                        checked = self.citation_validator.validate(raw, context)
                    except CitationValidationError as first_error:
                        repair_messages = list(context.messages)
                        repair_messages.append(
                            {
                                "role": "assistant",
                                "content": raw,
                            }
                        )
                        repair_messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "上一个 JSON 未通过程序校验："
                                    f"{first_error}。请只根据原 evidence 修正 JSON，"
                                    "used_sources 只能使用 evidence 中存在的编号。"
                                ),
                            }
                        )
                        raw = llm.generate(
                            repair_messages,
                            max_tokens=self.answer_max_tokens,
                            cancel_token=token,
                        )
                        try:
                            checked = self.citation_validator.validate(raw, context)
                        except CitationValidationError:
                            return RagAnswer(
                                "validation_failed",
                                "回答未通过来源校验。你可以查看下面检索到的证据。",
                                context.citations,
                                trace_id,
                                source.generation,
                                ("模型返回的引用格式无效",),
                            )
                    self._emit(progress, "complete", "智能回答已完成")
                    if checked.insufficient:
                        return self.policy.insufficient(
                            trace_id,
                            source.generation,
                            context.citations,
                        )
                    return RagAnswer(
                        status="answered",
                        answer_text=checked.answer,
                        citations=checked.citations,
                        trace_id=trace_id,
                        source_generation=source.generation,
                        warnings=(
                            (f"上下文裁掉 {context.dropped_chunks} 个低排名证据块",)
                            if context.dropped_chunks
                            else ()
                        ),
                    )
        except RagIndexNotReadyError as exc:
            return RagAnswer(
                "index_stale",
                "智能问答索引正在更新或需要重建。",
                (),
                trace_id,
                active.generation,
                (str(exc),),
            )
        except LLMUnavailableError as exc:
            return RagAnswer(
                "model_unavailable",
                "本地模型暂不可用，请检查模型服务设置。",
                (),
                trace_id,
                active.generation,
                (str(exc),),
            )

    @staticmethod
    def _emit(callback, stage: str, message: str) -> None:
        if callback is not None:
            callback(RagProgress(stage, message, 0, 0, True))
