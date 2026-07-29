from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class FileTiming:
    file_id: int
    extension: str
    size_bytes: int
    queue_name: str
    queue_wait_ms: int = 0
    parse_ms: int = 0
    block_count: int = 0
    text_chars: int = 0
    spool_bytes: int = 0
    worker_pid: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class IndexRunMetrics:
    run_id: str
    mode: str = "incremental"
    discovered_files: int = 0
    discovered_bytes: int = 0
    scan_ms: int = 0
    fingerprint_ms: int = 0
    parse_ms_by_lane: dict[str, int] = field(default_factory=dict)
    normalize_ms: int = 0
    spool_write_ms: int = 0
    database_write_ms: int = 0
    fts_build_ms: int = 0
    total_ms: int = 0
    peak_rss_bytes: int = 0
    process_spawn_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dedup_candidate_count: int = 0
    dedup_full_hash_count: int = 0
    dedup_verified_source_count: int = 0
    dedup_parse_avoided_count: int = 0
    dedup_bytes_avoided: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
