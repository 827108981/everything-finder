from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager, utc_now
from local_full_text_search.core.hardware_profile import (
    build_performance_profile,
    detect_hardware_profile_for_roots,
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
        self._pause_monitor_lock = threading.Lock()
        self._pause_monitor_generation = 0
        self._hardware: object | None = None
        self._root_disk_classes: dict[str, str] = {}
        self._profile_payload: dict[str, object] | None = None

    @Slot()
    def run(self) -> None:
        try:
            db = DatabaseManager(self.db_path)
            db.initialize()
            scan_started_at = utc_now()
            effective_settings = self.settings
            profile_payload: dict[str, object] | None = None
            roots = db.list_roots(enabled_only=True)
            root_paths = [Path(str(root["path"])) for root in roots]
            hardware, root_disk_classes = detect_hardware_profile_for_roots(root_paths)
            self._hardware = hardware
            self._root_disk_classes = dict(root_disk_classes)
            if self.performance_mode:
                profile = build_performance_profile(hardware)
                profile_payload = profile.to_dict()
                self._profile_payload = dict(profile_payload)
                effective_settings = settings_for_profile(self.settings, profile)
                self.progress.emit(
                    {
                        "stage": "performance_profile",
                        "phase_label": "性能模式配置已生成",
                        "performance_profile": profile_payload,
                    }
                )
            effective_profile = profile_payload or {
                "mode": "normal",
                "parser_workers": effective_settings.parser_workers,
                "process_parser_workers": effective_settings.process_parser_workers,
                "pdf_parser_workers": effective_settings.pdf_parser_workers,
                "ocr_workers": effective_settings.ocr_workers,
                "ocr_cpu_threads": effective_settings.ocr_cpu_threads,
                "slow_file_workers": effective_settings.slow_file_workers,
                "memory_budget_mb": effective_settings.index_memory_budget_mb,
                "cpu_token_budget": effective_settings.index_cpu_token_budget,
                "disk_class": hardware.disk_class,
            }
            self.manager = IndexManager(
                db,
                effective_settings,
                run_context={
                    "execution_mode": "performance" if self.performance_mode else "normal",
                    "hardware": hardware.to_dict(),
                    "root_disk_classes": root_disk_classes,
                    "effective_profile": effective_profile,
                },
            )
            if self.token.paused:
                self.manager.request_pause()
            summary = self.manager.index_enabled_roots(self.token, self.progress.emit)
            run_reports = db.recent_index_runs_since(scan_started_at)
            self.finished.emit(
                {
                    "summary": summary.to_dict(),
                    "runs": run_reports,
                    "run_metrics": _aggregate_run_metrics(run_reports),
                    "performance_mode": self.performance_mode,
                    "performance_profile": self._profile_payload,
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
        if self.manager is not None:
            self.manager.request_pause()
        self.progress.emit(
            {
                "stage": "pausing",
                "pause_state": "pausing",
                "phase_label": "正在暂停，等待活动任务到达安全检查点",
            }
        )
        with self._pause_monitor_lock:
            self._pause_monitor_generation += 1
            generation = self._pause_monitor_generation
        threading.Thread(
            target=self._monitor_safe_pause,
            args=(generation,),
            name="lfts-safe-pause-monitor",
            daemon=True,
        ).start()

    @Slot()
    def resume(self) -> None:
        if self.manager is not None:
            self.manager.request_resume()
        self.token.resume()
        with self._pause_monitor_lock:
            self._pause_monitor_generation += 1
        self.progress.emit(
            {
                "stage": "resuming",
                "pause_state": "resuming",
                "phase_label": "正在从安全检查点继续",
            }
        )

    @Slot(bool)
    def switch_mode(self, performance_mode: bool) -> None:
        manager = self.manager
        if (
            manager is None
            or not self.token.paused
            or not manager.is_safely_paused()
            or self._hardware is None
        ):
            self.progress.emit(
                {
                    "stage": "mode_switch_rejected",
                    "pause_state": "pausing" if self.token.paused else "running",
                    "phase_label": "只有完全暂停后才能切换模式",
                }
            )
            return
        if performance_mode:
            profile = build_performance_profile(self._hardware)
            profile_payload = profile.to_dict()
            effective_settings = settings_for_profile(self.settings, profile)
            execution_mode = "performance"
        else:
            profile_payload = {
                "mode": "normal",
                "parser_workers": self.settings.parser_workers,
                "process_parser_workers": self.settings.process_parser_workers,
                "pdf_parser_workers": self.settings.pdf_parser_workers,
                "ocr_workers": self.settings.ocr_workers,
                "ocr_cpu_threads": self.settings.ocr_cpu_threads,
                "slow_file_workers": self.settings.slow_file_workers,
                "memory_budget_mb": self.settings.index_memory_budget_mb,
                "cpu_token_budget": self.settings.index_cpu_token_budget,
                "disk_class": getattr(self._hardware, "disk_class", "unknown"),
            }
            effective_settings = self.settings
            execution_mode = "normal"
        try:
            applied = manager.apply_settings_while_paused(
                effective_settings,
                execution_mode=execution_mode,
                effective_profile=profile_payload,
            )
        except Exception as exc:
            logger.exception("Unable to switch indexing mode while paused")
            self.progress.emit(
                {
                    "stage": "mode_switch_failed",
                    "pause_state": "paused",
                    "phase_label": f"模式切换失败：{str(exc) or exc.__class__.__name__}",
                }
            )
            return
        if not applied:
            self.progress.emit(
                {
                    "stage": "mode_switch_rejected",
                    "pause_state": "paused",
                    "phase_label": "模式切换未应用，索引仍保持暂停",
                }
            )
            return
        self.performance_mode = bool(performance_mode)
        self._profile_payload = dict(profile_payload) if performance_mode else None
        self.progress.emit(
            {
                "stage": "paused",
                "pause_state": "paused",
                "performance_mode": self.performance_mode,
                "performance_profile": profile_payload,
                "phase_label": (
                    "性能模式 · 已暂停"
                    if self.performance_mode
                    else "普通模式 · 已暂停"
                ),
            }
        )

    def _monitor_safe_pause(self, generation: int) -> None:
        while self.token.paused and not self.token.cancelled:
            with self._pause_monitor_lock:
                if generation != self._pause_monitor_generation:
                    return
            manager = self.manager
            if manager is not None and manager.is_safely_paused():
                self.progress.emit(
                    {
                        "stage": "paused",
                        "pause_state": "paused",
                        "phase_label": "已暂停",
                    }
                )
                return
            time.sleep(0.1)


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
