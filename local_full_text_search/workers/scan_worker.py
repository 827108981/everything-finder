from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager, utc_now
from local_full_text_search.core.hardware_profile import (
    build_performance_profile,
    detect_hardware_profile,
    settings_for_profile,
)
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.task_manager import CancelToken

logger = logging.getLogger(__name__)


class ScanWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path, settings: AppSettings, *, performance_mode: bool = False) -> None:
        super().__init__()
        self.db_path = db_path
        self.settings = settings
        self.performance_mode = performance_mode
        self.token = CancelToken()
        self.manager: IndexManager | None = None

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            db.initialize()
            scan_started_at = utc_now()
            effective_settings = self.settings
            profile_payload: dict[str, object] | None = None
            if self.performance_mode:
                roots = db.list_roots(enabled_only=True)
                index_path = Path(str(roots[0]["path"])) if roots else None
                profile = build_performance_profile(detect_hardware_profile(index_path))
                profile_payload = profile.to_dict()
                effective_settings = settings_for_profile(self.settings, profile)
                self.progress.emit(
                    {
                        "stage": "performance_profile",
                        "phase_label": "性能模式配置已生成",
                        "performance_profile": profile_payload,
                    }
                )
            self.manager = IndexManager(db, effective_settings)
            summary = self.manager.index_enabled_roots(self.token, self.progress.emit)
            run_reports = db.recent_index_runs_since(scan_started_at)
            self.finished.emit(
                {
                    "summary": summary.to_dict(),
                    "runs": run_reports,
                    "run_metrics": _aggregate_run_metrics(run_reports),
                    "performance_mode": self.performance_mode,
                    "performance_profile": profile_payload,
                }
            )
        except Exception as exc:
            logger.exception("Index scan failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            self.manager = None

    @Slot()
    def cancel(self, *, force: bool = False) -> None:
        self.token.cancel(force=force)
        if force and self.manager is not None:
            self.manager.force_terminate_processes()

    @Slot()
    def pause(self) -> None:
        self.token.pause()

    @Slot()
    def resume(self) -> None:
        self.token.resume()


def _aggregate_run_metrics(run_reports: list[dict[str, object]]) -> dict[str, object]:
    totals: dict[str, int] = {
        "discovered_files": 0,
        "discovered_bytes": 0,
        "scan_ms": 0,
        "parse_ms": 0,
        "write_ms": 0,
        "fts_ms": 0,
        "total_ms": 0,
        "peak_rss_bytes": 0,
        "process_spawn_count": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "dedup_candidate_count": 0,
        "dedup_full_hash_count": 0,
        "dedup_verified_source_count": 0,
        "dedup_parse_avoided_count": 0,
        "dedup_bytes_avoided": 0,
    }
    for report in run_reports:
        totals["discovered_files"] += _int_metric(report, "discovered_files")
        totals["discovered_bytes"] += _int_metric(report, "discovered_bytes")
        totals["scan_ms"] += _int_metric(report, "scan_ms")
        totals["parse_ms"] += _int_metric(report, "parse_ms")
        totals["write_ms"] += _int_metric(report, "write_ms")
        totals["fts_ms"] += _int_metric(report, "fts_ms")
        totals["total_ms"] += _int_metric(report, "total_ms")
        totals["peak_rss_bytes"] = max(
            totals["peak_rss_bytes"],
            _int_metric(report, "peak_rss_bytes"),
        )
        summary = report.get("summary")
        metrics = summary.get("metrics") if isinstance(summary, dict) else None
        if not isinstance(metrics, dict):
            continue
        for key in (
            "process_spawn_count",
            "cache_hits",
            "cache_misses",
            "dedup_candidate_count",
            "dedup_full_hash_count",
            "dedup_verified_source_count",
            "dedup_parse_avoided_count",
            "dedup_bytes_avoided",
        ):
            totals[key] += _int_metric(metrics, key)
    return totals


def _int_metric(source: dict[str, object], key: str) -> int:
    try:
        return int(source.get(key) or 0)
    except (TypeError, ValueError):
        return 0
