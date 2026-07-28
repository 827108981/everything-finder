from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILE_PATTERNS


@dataclass(slots=True)
class AppSettings:
    settings_version: int = 4
    default_search_mode: str = "exact"
    case_sensitive: bool = False
    auto_search: bool = False
    search_debounce_ms: int = 300
    page_size: int = 100
    max_results: int = 1000
    parser_workers: int = 4
    ocr_workers: int = 1
    slow_file_workers: int = 1
    process_parser_workers: int = 1
    index_write_batch_size: int = 32
    normal_pending_tasks: int = 8
    ocr_pending_tasks: int = 2
    slow_pending_tasks: int = 2
    process_pending_tasks: int = 2
    max_pending_parse_tasks: int = 96
    large_office_process_min_bytes: int = 4 * 1024 * 1024
    process_max_tasks_per_child: int = 16
    process_recycle_min_tasks: int = 16
    process_recycle_max_tasks: int = 64
    index_performance_preset: str = "balanced"
    index_memory_budget_mb: int = 2048
    process_memory_budget_mb: int = 768
    normal_inflight_bytes: int = 256 * 1024 * 1024
    office_inflight_bytes: int = 1024 * 1024 * 1024
    ocr_inflight_bytes: int = 256 * 1024 * 1024
    slow_inflight_bytes: int = 512 * 1024 * 1024
    fast_ooxml_enabled: bool = True
    enable_parse_cache: bool = True
    block_target_chars: int = 4096
    block_max_chars: int = 16384
    db_write_batch_blocks: int = 2000
    db_write_batch_bytes: int = 16 * 1024 * 1024
    db_write_max_delay_ms: int = 500
    parse_result_spool_threshold_bytes: int = 4 * 1024 * 1024
    defer_fts_during_full_scan: bool = True
    pdf_parallel_min_bytes: int = 64 * 1024 * 1024
    pdf_parallel_min_pages: int = 500
    legacy_conversion_cache: bool = True
    auto_scan_on_start: bool = False
    monitor_file_changes: bool = False
    retry_failed_files: bool = True
    single_file_timeout_seconds: int = 120
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
        values["settings_version"] = 4
        return cls(**values)


DEFAULT_SETTINGS = AppSettings()
