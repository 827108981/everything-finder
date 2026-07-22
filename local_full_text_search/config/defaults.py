from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .constants import DEFAULT_EXCLUDED_DIRS, DEFAULT_EXCLUDED_FILE_PATTERNS


@dataclass(slots=True)
class AppSettings:
    settings_version: int = 2
    default_search_mode: str = "exact"
    case_sensitive: bool = False
    auto_search: bool = False
    search_debounce_ms: int = 300
    page_size: int = 100
    max_results: int = 1000
    parser_workers: int = 4
    ocr_workers: int = 1
    slow_file_workers: int = 1
    index_write_batch_size: int = 32
    max_pending_parse_tasks: int = 96
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
    min_ocr_image_pixels: int = 12_000
    max_ocr_image_side: int = 2400
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
        # 旧版配置文件在 OCR 默认关闭时生成。团队 OCR 版升级后，
        # 没有 settings_version 的配置按新版默认完整索引迁移。
        if int(data.get("settings_version", 0) or 0) < 2:
            values["settings_version"] = 2
            values["enable_ocr"] = True
            values["ocr_images"] = True
            values["ocr_scanned_pdf"] = True
        return cls(**values)


DEFAULT_SETTINGS = AppSettings()
