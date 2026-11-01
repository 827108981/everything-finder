"""启动时只读检查可复用的本地 RAG 模型与已发布索引。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.constants import RAG_MODEL_CACHE_DIR
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.rag.index_store import RagIndexStore
from local_full_text_search.rag.lexical_index_store import LexicalIndexStore
from local_full_text_search.rag.retrieval.lexical_analyzer import get_lexical_analyzer
from local_full_text_search.rag.embedding_model_catalog import embedding_model_option
from local_full_text_search.rag.retrieval.vector_units import VectorUnitPolicy
from local_full_text_search.rag.vector_index_store import VectorIndexStore
from local_full_text_search.rag.model_manager import LocalModelManager, ModelSpec
from local_full_text_search.rag.vector_store.chroma import ChromaVectorStore


class RagLocalStateWorker(QObject):
    """不下载、不重建，只验证上一次落盘的模型和 Chroma bundle。"""

    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(self, generation_base_path: Path, settings) -> None:
        super().__init__()
        self.generation_base_path = Path(generation_base_path).resolve()
        self.settings = settings
        self.token = CancelToken()

    @Slot()
    def run(self) -> None:
        try:
            self.progress.emit(
                {"stage": "local_state", "message": "正在读取本地 AI 模型和已发布索引"}
            )
            manager = LocalModelManager(RAG_MODEL_CACHE_DIR)
            missing_models: list[str] = []
            for spec in (
                ModelSpec(
                    self.settings.rag_embedding_repo_id,
                    self.settings.rag_embedding_revision,
                    "embedding",
                ),
                ModelSpec(
                    self.settings.rag_reranker_repo_id,
                    self.settings.rag_reranker_revision,
                    "reranker",
                ),
            ):
                self.token.throw_if_cancelled()
                try:
                    manager.resolve_local_path(spec)
                except Exception:
                    missing_models.append(spec.repo_id)

            index_error = ""
            bundle = None
            local_vector_count = 0
            configured_option = embedding_model_option(
                self.settings.rag_embedding_repo_id,
                self.settings.rag_embedding_revision,
            )
            try:
                self.token.throw_if_cancelled()
                bundle = RagIndexStore(
                    self.generation_base_path.parent / "rag"
                ).active()
                configured_backend = str(
                    getattr(self.settings, "rag_vector_backend", "legacy")
                )
                if configured_backend == "legacy":
                    if getattr(bundle, "parent_vector_mode", "chroma") != "chroma":
                        index_error = "当前 AI 索引未构建旧版父块向量，请重新构建或切换细粒度向量 V2"
                    else:
                        vector_store = ChromaVectorStore(bundle.chroma_path)
                        try:
                            local_vector_count = vector_store.count()
                        finally:
                            vector_store.close()
                        if local_vector_count != bundle.vector_count:
                            index_error = (
                                "Chroma 向量数量与索引清单不一致："
                                f"{local_vector_count}/{bundle.vector_count}"
                            )
                if not index_error and (
                    bundle.embedding_repo_id != self.settings.rag_embedding_repo_id
                    or bundle.embedding_revision != self.settings.rag_embedding_revision
                    or (
                        configured_option is not None
                        and int(getattr(bundle, "embedding_dimension", 0) or 0)
                        not in {0, configured_option.dimension}
                    )
                ):
                    index_error = "已发布 AI 索引使用的向量模型与当前设置不一致"
                elif not index_error and configured_backend == "unit_v2":
                    policy = VectorUnitPolicy.from_settings(self.settings)
                    unit_store = VectorIndexStore(
                        self.generation_base_path.parent / "rag"
                    )
                    if getattr(bundle, "vector_unit_directory", ""):
                        unit_index = unit_store.attached(
                            bundle.bundle_id,
                            bundle.vector_unit_directory,
                            embedding_repo_id=self.settings.rag_embedding_repo_id,
                            embedding_revision=self.settings.rag_embedding_revision,
                            embedding_dimension=int(bundle.embedding_dimension),
                            policy_fingerprint=policy.fingerprint,
                            index_fingerprint=bundle.vector_unit_index_fingerprint,
                        )
                    else:
                        unit_index = unit_store.active(
                            bundle.bundle_id,
                            embedding_repo_id=self.settings.rag_embedding_repo_id,
                            embedding_revision=self.settings.rag_embedding_revision,
                            embedding_dimension=int(bundle.embedding_dimension),
                            policy_fingerprint=policy.fingerprint,
                        )
                    unit_vectors = ChromaVectorStore(unit_index.chroma_path)
                    try:
                        local_vector_count = unit_vectors.count()
                    finally:
                        unit_vectors.close()
                    if local_vector_count != unit_index.unit_count:
                        index_error = (
                            "细粒度向量数量与索引清单不一致："
                            f"{local_vector_count}/{unit_index.unit_count}"
                        )
                if not index_error and str(
                    getattr(self.settings, "rag_lexical_backend", "legacy")
                ) == "word_v2":
                    # 启动检查与查询使用同一个索引指纹，避免界面显示可用但提问时报错。
                    LexicalIndexStore(self.generation_base_path.parent / "rag").active(
                        bundle.bundle_id, get_lexical_analyzer().fingerprint,
                    )
            except Exception as exc:
                index_error = str(exc) or type(exc).__name__

            active_source = IndexGenerationStore(self.generation_base_path).active()
            active_source_generation = (
                int(active_source.generation) if active_source is not None else 0
            )
            self.finished.emit(
                {
                    "models_ready": not missing_models,
                    "missing_models": tuple(missing_models),
                    "index_ready": bundle is not None and not index_error,
                    "index_error": index_error,
                    "bundle_id": bundle.bundle_id if bundle is not None else "",
                    "chunk_count": bundle.chunk_count if bundle is not None else 0,
                    "vector_count": local_vector_count,
                    "source_generation": bundle.source_generation if bundle is not None else 0,
                    "active_source_generation": active_source_generation,
                    "using_previous_index": bool(
                        bundle is not None
                        and active_source_generation
                        and bundle.verified_source_generation != active_source_generation
                    ),
                    "configured_embedding_model": self.settings.rag_embedding_repo_id,
                    "configured_embedding_revision": self.settings.rag_embedding_revision,
                    "configured_embedding_dimension": (
                        configured_option.dimension if configured_option is not None else 0
                    ),
                    "index_embedding_model": (
                        str(getattr(bundle, "embedding_repo_id", "")) if bundle is not None else ""
                    ),
                    "index_embedding_revision": (
                        str(getattr(bundle, "embedding_revision", "")) if bundle is not None else ""
                    ),
                    "index_embedding_dimension": (
                        int(getattr(bundle, "embedding_dimension", 0) or 0)
                        if bundle is not None else 0
                    ),
                }
            )
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
