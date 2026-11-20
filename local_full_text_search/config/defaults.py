from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILE_PATTERNS


@dataclass(slots=True)
class AppSettings:
    settings_version: int = 15
    default_search_mode: str = "exact"
    case_sensitive: bool = False
    auto_search: bool = False
    search_debounce_ms: int = 300
    page_size: int = 100
    max_results: int = 1000
    parser_workers: int = 4
    # Two OCR workers are the balanced default. Runtime admission still
    # reduces this to one when memory, CPU, or I/O pressure is sustained.
    ocr_workers: int = 2
    slow_file_workers: int = 1
    process_parser_workers: int = 1
    pdf_parser_workers: int = 1
    index_write_batch_size: int = 32
    normal_pending_tasks: int = 8
    ocr_pending_tasks: int = 2
    ocr_microbatch_parent_jobs: int = 4
    ocr_microbatch_max_requests: int = 64
    ocr_microbatch_max_pixels: int = 8_000_000
    ocr_microbatch_memory_mb: int = 256
    ocr_microbatch_wait_ms: int = 30
    slow_pending_tasks: int = 2
    process_pending_tasks: int = 2
    pdf_pending_tasks: int = 2
    pdf_page_batch_size: int = 4
    max_pending_parse_tasks: int = 96
    large_office_process_min_bytes: int = 4 * 1024 * 1024
    process_max_tasks_per_child: int = 16
    process_recycle_min_tasks: int = 16
    process_recycle_max_tasks: int = 64
    index_performance_preset: str = "balanced"
    index_cpu_token_budget: int = 0
    index_memory_budget_mb: int = 2048
    process_memory_budget_mb: int = 768
    normal_inflight_bytes: int = 256 * 1024 * 1024
    office_inflight_bytes: int = 1024 * 1024 * 1024
    pdf_inflight_bytes: int = 1024 * 1024 * 1024
    ocr_inflight_bytes: int = 256 * 1024 * 1024
    slow_inflight_bytes: int = 512 * 1024 * 1024
    fast_ooxml_enabled: bool = True
    xlsx_sheet_workers: int = 2
    xlsx_shared_strings_disk_threshold_bytes: int = 16 * 1024 * 1024
    enable_parse_cache: bool = True
    block_target_chars: int = 4096
    block_max_chars: int = 16384
    db_write_batch_blocks: int = 2000
    db_write_batch_bytes: int = 16 * 1024 * 1024
    db_write_max_delay_ms: int = 500
    parse_result_spool_threshold_bytes: int = 4 * 1024 * 1024
    defer_fts_during_full_scan: bool = True
    partial_search_during_index: bool = True
    pdf_parallel_min_bytes: int = 64 * 1024 * 1024
    pdf_parallel_min_pages: int = 500
    legacy_conversion_cache: bool = True
    auto_scan_on_start: bool = True
    monitor_file_changes: bool = False
    retry_failed_files: bool = False
    normal_no_progress_timeout_seconds: int = 300
    ocr_no_progress_timeout_seconds: int = 900
    archive_no_progress_timeout_seconds: int = 900
    legacy_no_progress_timeout_seconds: int = 1800
    process_no_progress_timeout_seconds: int = 300
    planning_no_progress_timeout_seconds: int = 300
    planning_startup_timeout_seconds: int = 30
    ocr_thread_startup_timeout_seconds: int = 30
    ocr_model_load_timeout_seconds: int = 120
    ocr_detect_timeout_seconds: int = 300
    ocr_recognize_timeout_seconds: int = 900
    finalization_no_progress_timeout_seconds: int = 30
    planning_discovery_batch_size: int = 128
    no_progress_max_retries: int = 3
    compute_full_hash: bool = False
    include_hidden_files: bool = False
    enable_ocr: bool = True
    ocr_language: str = "ch"
    ocr_images: bool = True
    ocr_scanned_pdf: bool = True
    ocr_min_confidence: float = 0.60
    ocr_cpu_threads: int = 2
    min_ocr_image_pixels: int = 12_000
    max_ocr_image_side: int = 960
    save_search_history: bool = True
    # RAG remains optional: absent RAG packages or models never block ordinary search.
    rag_enabled: bool = False
    # Opt-in until paired native-entry regression has passed. Empty config = generic.
    rag_query_plan_enabled: bool = False
    rag_structured_answer_enabled: bool = False
    rag_domain_config: str = "chemiluminescence"
    rag_lexical_multi_route_enabled: bool = False
    # V2 在完成真实评测前保持显式选择，避免升级后悄悄改变用户结果。
    rag_lexical_backend: str = "legacy"
    rag_lexical_title_weight: float = 0.2
    rag_lexical_complete_match_enabled: bool = True
    rag_lexical_k1: float = 1.2
    rag_lexical_b: float = 0.75
    rag_vector_focused_query_enabled: bool = False
    # legacy 使用父 chunk 单向量；unit_v2 使用独立细粒度索引并归并回父 chunk。
    rag_vector_backend: str = "legacy"
    rag_vector_unit_target_tokens: int = 128
    rag_vector_unit_max_tokens: int = 192
    rag_vector_unit_overlap_tokens: int = 32
    rag_vector_unit_include_source_context: bool = True
    rag_vector_unit_oversample_factor: int = 8
    rag_embedding_repo_id: str = "BAAI/bge-small-zh-v1.5"
    rag_embedding_revision: str = "7999e1d3359715c523056ef9478215996d62a620"
    rag_reranker_repo_id: str = "BAAI/bge-reranker-base"
    rag_reranker_revision: str = "2cfc18c9415c912f9d8155881c133215df768a70"
    rag_chunk_target_chars: int = 900
    rag_chunk_max_chars: int = 1200
    rag_chunk_overlap_chars: int = 120
    # 本机 BGE Large 实测 16 比 32 更快；运行时遇到内存不足仍会自动减半重试。
    rag_embedding_batch_size: int = 16
    rag_embedding_length_bucketing: bool = True
    rag_resume_partial_build: bool = True
    rag_index_write_batch_size: int = 128
    rag_hnsw_m: int = 16
    rag_hnsw_construction_ef: int = 200
    rag_hnsw_search_ef: int = 100
    rag_max_chunks: int = 200_000
    rag_reranker_batch_size: int = 4
    rag_reranker_max_length: int = 512
    # Compatibility fields for older settings files; BGE-only ordering ignores them.
    rag_rerank_cross_encoder_weight: float = 1.0
    rag_rerank_retrieval_prior_weight: float = 0.0
    rag_rerank_answer_support_weight: float = 0.0
    rag_rerank_max_rank_drop: int = 0
    rag_lexical_top_k: int = 50
    rag_vector_top_k: int = 50
    rag_fusion_top_k: int = 30
    rag_reranker_top_k: int = 20
    rag_final_top_k: int = 8
    rag_rrf_k: int = 60
    rag_min_vector_score: float = 0.35
    rag_min_rerank_score: float = 0.10
    rag_context_max_chars: int = 12_000
    rag_max_chunks_per_document: int = 3
    rag_clarification_enabled: bool = True
    # Source-grounded semantic routing remains opt-in until business validation.
    rag_semantic_clarification_enabled: bool = False
    rag_dedup_jaccard_threshold: float = 0.92
    # Legacy experiment switch; the simple evidence policy never filters by it.
    rag_constraint_matching_enabled: bool = False
    rag_answer_review_enabled: bool = True
    rag_answer_review_max_output_tokens: int = 1400
    rag_answer_review_strict: bool = True
    rag_api_base_url: str = "https://api.openai.com/v1"
    rag_api_model: str = ""
    rag_api_style: str = "chat_completions"
    rag_api_timeout_seconds: int = 90
    rag_api_secret_ref: str = "local-full-text-search.rag.api-key"
    max_zip_file_count: int = 2000
    max_zip_uncompressed_bytes: int = 1024 * 1024 * 1024
    max_zip_depth: int = 2
    excluded_dirs: list[str] = field(default_factory=lambda: sorted(DEFAULT_EXCLUDED_DIRS))
    excluded_file_patterns: list[str] = field(
        default_factory=lambda: list(DEFAULT_EXCLUDED_FILE_PATTERNS)
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "AppSettings":
        defaults = cls()
        values = defaults.to_dict()
        values.update({k: v for k, v in data.items() if k in values})
        # Migrations may supply defaults for missing keys, but an explicit user
        # value must always win. The dataclass defaults already cover missing
        # OCR fields from older configuration files.
        previous_version = int(data.get("settings_version", 0) or 0)
        if previous_version < 4 and data.get("max_ocr_image_side", 2400) == 2400:
            values["max_ocr_image_side"] = 960
        if previous_version < 10:
            # Terminal failures now require an explicit retry request. A
            # changed source identity is still scheduled automatically.
            values["retry_failed_files"] = False
        if previous_version < 13 and "rag_answer_review_enabled" not in data:
            # 避免升级后静默增加每次问答的 LLM 调用次数；用户可在设置页主动开启。
            values["rag_answer_review_enabled"] = False
        values["settings_version"] = 15
        return cls(**values)


DEFAULT_SETTINGS = AppSettings()
