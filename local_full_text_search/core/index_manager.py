from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import pickle
import shutil
import threading
import time
import traceback
import uuid
from collections import deque
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Executor, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from local_full_text_search.config.constants import (
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    LEGACY_OFFICE_EXTENSIONS,
    PARSER_VERSIONS,
    SUPPORTED_EXTENSIONS,
    TEMP_DIR,
    VIDEO_EXTENSIONS,
)
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.block_coalescer import BlockCoalescer
from local_full_text_search.core.content_fingerprint import ContentFingerprint, fingerprint_file
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import (
    CancelledError,
    ParserDependencyError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
from local_full_text_search.core.index_scheduler import estimate_parse_cost
from local_full_text_search.core.index_time_estimator import IndexTimeEstimator
from local_full_text_search.core.index_writer import IndexWriter
from local_full_text_search.core.scanner import iter_files
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import FileTiming, IndexRunMetrics
from local_full_text_search.parsers.parser_registry import ParserRegistry
from local_full_text_search.parsers.legacy_office_parser import cleanup_registered_office_processes

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class IndexSummary:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    unsupported: int = 0
    metadata_only: int = 0
    partial_success: int = 0
    excluded_video: int = 0
    deleted: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class ParseJob:
    file_id: int
    file_path: Path
    task_id: int | None = None
    alias_file_ids: tuple[int, ...] = ()
    content_key: str = ""
    parser_name: str = "unknown"
    parser_version: str = "1"
    lane: str = "normal"
    size_bytes: int = 0
    relevant_bytes: int = 0
    estimated_cost: float = 1.0
    queued_monotonic: float = 0.0
    started_monotonic: float = 0.0
    retry_count: int = 0
    watchdog_timed_out: bool = False
    resume_cursor: int = 0
    progress_sequence: int = 0
    progress_phase: str = ""
    progress_completed: int = 0
    progress_total: int = 0
    progress_detail: str = ""
    last_progress_monotonic: float = 0.0
    timeout_seconds: int = 0


@dataclass(slots=True)
class ParseOutcome:
    file_id: int
    file_path: Path
    blocks: list[ContentBlock]
    parser_name: str
    status: str
    error_code: str | None = None
    error_message: str | None = None
    task_id: int | None = None
    alias_file_ids: tuple[int, ...] = ()
    content_key: str = ""
    parser_version: str = "1"
    lane: str = "normal"
    size_bytes: int = 0
    estimated_cost: float = 1.0
    queue_wait_ms: int = 0
    parse_ms: int = 0
    normalize_ms: int = 0
    worker_pid: int | None = None
    spool_path: Path | None = None
    spool_checksum: str | None = None
    spool_bytes: int = 0
    spool_write_ms: int = 0
    resume_cursor: int = 0
    progress_phase: str = ""
    progress_completed: int = 0
    progress_total: int = 0


@dataclass(slots=True)
class SpoolParseResult:
    file_id: int
    file_path: Path
    spool_path: Path
    worker_pid: int
    result_bytes: int
    checksum: str
    spool_write_ms: int = 0


ParseResult = ParseOutcome | SpoolParseResult


@dataclass(slots=True)
class ParseLane:
    name: str
    executor: Executor
    max_in_flight: int
    max_inflight_bytes: int
    process_based: bool = False
    worker_count: int = 1
    pending: deque[ParseJob] = field(default_factory=deque)
    futures: set[Future[ParseResult]] = field(default_factory=set)
    jobs: dict[Future[ParseResult], ParseJob] = field(default_factory=dict)
    inflight_bytes: int = 0


class ProcessResourceMonitor:
    """Sample child memory outside the scheduler's critical path."""

    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = max(0.2, float(interval_seconds))
        self.peak_rss_bytes = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="lfts-resource-monitor",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.2)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    current_process_rss(),
                )
            except Exception:
                logger.debug("Resource monitor sample failed", exc_info=True)
            self._stop.wait(self.interval_seconds)


class ProcessLaneWatchdog:
    """Terminate overdue process workers even if the scheduler is backpressured."""

    def __init__(
        self,
        lanes: dict[str, ParseLane],
        spool_dir: Path,
        settings: AppSettings,
    ) -> None:
        self.lanes = lanes
        self.spool_dir = spool_dir
        self.settings = settings
        self._stop = threading.Event()
        self._terminated: set[ProcessPoolExecutor] = set()
        self._thread = threading.Thread(
            target=self._run,
            name="lfts-process-watchdog",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.5)

    def _run(self) -> None:
        while not self._stop.wait(0.25):
            now = time.perf_counter()
            for lane in list(self.lanes.values()):
                if not lane.process_based or not isinstance(lane.executor, ProcessPoolExecutor):
                    continue
                overdue = []
                for future, job in list(lane.jobs.items()):
                    if future.done() or not job.started_monotonic:
                        continue
                    refresh_job_progress(job, self.spool_dir, now)
                    timeout_seconds = no_progress_timeout(self.settings, job)
                    last_progress = job.last_progress_monotonic or job.started_monotonic
                    if now - last_progress >= timeout_seconds:
                        job.watchdog_timed_out = True
                        job.timeout_seconds = timeout_seconds
                        overdue.append(job)
                if not overdue:
                    continue
                executor = lane.executor
                if executor in self._terminated:
                    continue
                self._terminated.add(executor)
                logger.warning(
                    "Process lane %s made no progress for the configured interval with %s overdue task(s); terminating worker pool",
                    lane.name,
                    len(overdue),
                )
                terminate_process_pool_workers(executor, self.spool_dir)


_worker_state = threading.local()
_process_registry: ParserRegistry | None = None


class IndexManager:
    def __init__(self, db: DatabaseManager, settings: AppSettings) -> None:
        self.db = db
        self.settings = settings
        self._executor_lock = threading.Lock()
        self._active_process_executors: set[ProcessPoolExecutor] = set()
        self._active_process_registry_dirs: set[Path] = set()

    def force_terminate_processes(self) -> None:
        self.db.interrupt_active_connections()
        with self._executor_lock:
            executors = list(self._active_process_executors)
            registry_dirs = list(self._active_process_registry_dirs)
        for executor in executors:
            terminate_process_pool_workers(executor)
        for registry_dir in registry_dirs:
            cleanup_registered_office_processes(registry_dir)

    def index_enabled_roots(
        self,
        cancel_token: CancelToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexSummary:
        token = cancel_token or CancelToken()
        summary = IndexSummary()
        for root in self.db.list_roots(enabled_only=True):
            token.wait_if_paused()
            token.throw_if_cancelled()
            root_path = Path(str(root["path"]))
            if not root_path.exists():
                logger.warning("Root does not exist: %s", root_path)
                continue
            root_summary = self.index_root(int(root["id"]), token, progress_callback)
            merge_summary(summary, root_summary)
        return summary

    def index_root(
        self,
        root_id: int,
        cancel_token: CancelToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexSummary:
        token = cancel_token or CancelToken()
        root = next((row for row in self.db.list_roots() if int(row["id"]) == root_id), None)
        if root is None:
            raise ValueError(f"根目录不存在: {root_id}")

        run_started = time.perf_counter()
        run_id = uuid.uuid4().hex
        requires_full_rebuild = self.db.requires_full_rebuild()
        resumed_full_batch = self.db.has_incomplete_full_batch()
        full_batch = self.db.active_file_count() == 0 or requires_full_rebuild or resumed_full_batch
        metrics = IndexRunMetrics(run_id=run_id, mode="full_batch" if full_batch else "incremental")
        summary = IndexSummary()
        self.db.start_index_run(metrics)
        self.db.update_root_scan_time(root_id, "indexing")
        root_path = Path(str(root["path"]))
        include_subfolders = bool(root["include_subfolders"])
        previous_paths = self.db.active_paths_for_root(root_id)
        spool_dir = TEMP_DIR / "process_results" / run_id
        lanes: dict[str, ParseLane] = {}
        writer: IndexWriter | None = None
        resource_monitor: ProcessResourceMonitor | None = None
        watchdog: ProcessLaneWatchdog | None = None
        executors: list[Executor] = []
        process_executors: list[ProcessPoolExecutor] = []
        file_timings: list[FileTiming] = []
        worker_pids: set[int] = set()
        run_status = "failed"
        defer_fts = bool(self.settings.defer_fts_during_full_scan and full_batch)

        try:
            recovered = self._recover_spooled_tasks(root_id)
            if recovered:
                summary.indexed += recovered

            scan_started = time.perf_counter()
            discovered = self._discover_files(
                root_path,
                include_subfolders,
                token,
                summary,
                progress_callback,
            )
            metrics.scan_ms = int((time.perf_counter() - scan_started) * 1000)
            metrics.discovered_files = len(discovered)
            metrics.discovered_bytes = sum(_safe_file_size(path) for path in discovered)
            eligible_total = sum(
                1 for path in discovered if path.suffix.lower() not in VIDEO_EXTENSIONS
            )
            seen_paths = {str(path) for path in discovered}
            self._emit(
                progress_callback,
                "planning",
                summary,
                current_file="",
                pending=0,
                total_files=eligible_total,
                discovered_files=len(discovered),
                excluded_video=sum(
                    1 for path in discovered if path.suffix.lower() in VIDEO_EXTENSIONS
                ),
                total_bytes=metrics.discovered_bytes,
                phase_label="正在分析文件成本和重复内容",
            )

            fingerprint_started = time.perf_counter()
            jobs = self._prepare_jobs(root_id, discovered, run_id, summary, metrics, token)
            metrics.fingerprint_ms = int((time.perf_counter() - fingerprint_started) * 1000)
            missing_paths = previous_paths - seen_paths
            summary.deleted += self.db.mark_deleted_paths(missing_paths)

            if defer_fts and jobs:
                self.db.begin_deferred_fts()

            lanes, executors, process_executors = self._create_lanes(jobs, spool_dir)
            estimator = IndexTimeEstimator(
                lane_costs(jobs),
                {name: lane.worker_count for name, lane in lanes.items()},
            )
            with self._executor_lock:
                self._active_process_executors.update(process_executors)
                self._active_process_registry_dirs.add(spool_dir)

            resource_monitor = ProcessResourceMonitor()
            resource_monitor.start()
            watchdog = ProcessLaneWatchdog(
                lanes,
                spool_dir,
                self.settings,
            )
            watchdog.start()

            writer = IndexWriter(
                self.db,
                update_fts=not defer_fts,
                batch_blocks=self.settings.db_write_batch_blocks,
                batch_bytes=self.settings.db_write_batch_bytes,
                max_delay_ms=self.settings.db_write_max_delay_ms,
                queue_size=max(8, int(self.settings.index_write_batch_size or 32) * 2),
            )
            writer.start()

            last_heartbeat = 0.0
            while pending_lane_tasks(lanes.values()):
                recycled = self._recycle_unhealthy_process_lanes(
                    lanes,
                    executors,
                    process_executors,
                    spool_dir,
                )
                submitted = schedule_parse_lanes(lanes.values(), self.settings, token, spool_dir)
                if submitted:
                    task_ids = [task_id for task_id in submitted if task_id is not None]
                    if task_ids and not self.db.try_mark_tasks_running(task_ids):
                        logger.debug(
                            "Skipped parse task running-state update while SQLite is busy: %s tasks",
                            len(task_ids),
                        )
                now = time.perf_counter()
                if now - last_heartbeat >= 0.75:
                    self._emit_active_progress(
                        progress_callback,
                        summary,
                        lanes,
                        estimator,
                        eligible_total,
                    )
                    last_heartbeat = now
                completed = recycled + drain_completed_lanes(
                    lanes.values(), token, spool_dir, block=True
                )
                for lane_name, job, result, descriptor_bytes in completed:
                    outcome = self._outcome_from_result(job, result, spool_dir)
                    if (
                        outcome.error_code in {"PROCESS_WORKER_CRASH", "PARSE_NO_PROGRESS"}
                        and job.retry_count < max(0, int(self.settings.no_progress_max_retries))
                    ):
                        job.retry_count += 1
                        checkpoint = load_partial_parse_checkpoint(job, spool_dir, consume=False)
                        if (
                            checkpoint is not None
                            and checkpoint.resume_cursor > 0
                            and job.parser_name in {"pdf", "zip"}
                        ):
                            job.resume_cursor = checkpoint.resume_cursor
                        reset_job_for_retry(job, time.perf_counter())
                        lanes[lane_name].pending.appendleft(job)
                        continue
                    writer.submit(outcome, cancel_token=token)
                    record_parse_outcome(
                        summary,
                        outcome.status,
                        1 + len(outcome.alias_file_ids),
                        extension=outcome.file_path.suffix.lower(),
                    )
                    metrics.parse_ms_by_lane[lane_name] = (
                        metrics.parse_ms_by_lane.get(lane_name, 0) + outcome.parse_ms
                    )
                    metrics.normalize_ms += outcome.normalize_ms
                    metrics.spool_write_ms += outcome.spool_write_ms
                    if outcome.worker_pid:
                        worker_pids.add(outcome.worker_pid)
                    file_timings.append(
                        FileTiming(
                            file_id=outcome.file_id,
                            extension=outcome.file_path.suffix.lower(),
                            size_bytes=outcome.size_bytes,
                            queue_name=lane_name,
                            queue_wait_ms=outcome.queue_wait_ms,
                            parse_ms=outcome.parse_ms,
                            block_count=len(outcome.blocks),
                            text_chars=sum(len(block.raw_text) for block in outcome.blocks),
                            spool_bytes=outcome.spool_bytes,
                            worker_pid=outcome.worker_pid,
                        )
                    )
                    estimator.observe(
                        lane_name,
                        job.estimated_cost,
                        max(0.001, outcome.parse_ms / 1000.0),
                    )
                    self._emit_active_progress(
                        progress_callback,
                        summary,
                        lanes,
                        estimator,
                        eligible_total,
                        completed_queue=lane_name,
                        completed_file=str(outcome.file_path),
                        worker_pid=outcome.worker_pid,
                        process_result_bytes=outcome.spool_bytes or None,
                        process_descriptor_bytes=descriptor_bytes,
                    )

            writer_summary = writer.finish()
            metrics.database_write_ms = writer_summary.write_ms
            self.db.record_file_metrics(run_id, file_timings)

            if defer_fts and (jobs or resumed_full_batch or requires_full_rebuild):
                self._emit(
                    progress_callback,
                    "fts",
                    summary,
                    current_file="",
                    pending=0,
                    total_files=eligible_total,
                    completed_files=eligible_total,
                    phase_label="正在一次性构建全文索引",
                )
                metrics.fts_build_ms = self.db.rebuild_content_fts()

            completion = self.db.root_completion(root_id)
            is_complete = completion["blocking"] == 0
            self.db.update_root_scan_time(root_id, "ready" if is_complete else "incomplete")
            if is_complete and full_batch:
                self.db.mark_full_batch_complete()
            if is_complete and requires_full_rebuild:
                self.db.mark_full_rebuild_complete()
            metrics.process_spawn_count = len(worker_pids)
            metrics.total_ms = int((time.perf_counter() - run_started) * 1000)
            if resource_monitor is not None:
                metrics.peak_rss_bytes = max(
                    metrics.peak_rss_bytes,
                    resource_monitor.peak_rss_bytes,
                )
            run_status = "complete" if is_complete else "incomplete"
            self._emit(
                progress_callback,
                "finished" if is_complete else "incomplete",
                summary,
                current_file="",
                pending=0,
                total_files=completion["eligible"],
                completed_files=completion["complete"],
                blocking_files=completion["blocking"],
                excluded_video=completion["video_excluded"],
                elapsed_ms=metrics.total_ms,
                phase_label=(
                    "完整索引已完成"
                    if is_complete
                    else f"索引未完成，仍有 {completion['blocking']} 个文件需要处理"
                ),
            )
        except CancelledError:
            summary.cancelled = True
            for lane in lanes.values():
                lane.pending.clear()
                for future in lane.futures:
                    future.cancel()
            for executor in process_executors:
                terminate_process_pool_workers(executor)
            if writer is not None:
                writer.abort()
                metrics.database_write_ms = writer.summary.write_ms
            self.db.update_root_scan_time(root_id, "cancelled")
            run_status = "cancelled"
            self._emit(
                progress_callback,
                "cancelled",
                summary,
                current_file="",
                pending=sum(len(lane.futures) for lane in lanes.values()),
                phase_label="索引已停止",
            )
        except Exception:
            if token.cancelled:
                summary.cancelled = True
                for lane in lanes.values():
                    lane.pending.clear()
                    for future in lane.futures:
                        future.cancel()
                for executor in process_executors:
                    terminate_process_pool_workers(executor)
                if writer is not None:
                    try:
                        writer.abort()
                        metrics.database_write_ms = writer.summary.write_ms
                    except Exception:
                        logger.debug("Writer stopped during forced cancellation", exc_info=True)
                self.db.update_root_scan_time(root_id, "cancelled")
                run_status = "cancelled"
                self._emit(
                    progress_callback,
                    "cancelled",
                    summary,
                    current_file="",
                    pending=0,
                    phase_label="索引已停止",
                )
            else:
                if writer is not None:
                    try:
                        writer.finish()
                    except Exception:
                        logger.exception("Index writer shutdown failed")
                self.db.update_root_scan_time(root_id, "failed")
                raise
        finally:
            if watchdog is not None:
                watchdog.stop()
            if resource_monitor is not None:
                resource_monitor.stop()
                metrics.peak_rss_bytes = max(
                    metrics.peak_rss_bytes,
                    resource_monitor.peak_rss_bytes,
                )
            wait_for_workers = not token.force_cancelled and not summary.cancelled
            if token.force_cancelled or summary.cancelled:
                for executor in process_executors:
                    terminate_process_pool_workers(executor)
            for executor in executors:
                try:
                    executor.shutdown(wait=wait_for_workers, cancel_futures=True)
                except Exception:
                    logger.debug("Executor shutdown failed", exc_info=True)
            with self._executor_lock:
                self._active_process_executors.difference_update(process_executors)
                self._active_process_registry_dirs.discard(spool_dir)
            cleanup_registered_office_processes(spool_dir)
            if spool_dir.exists():
                shutil.rmtree(spool_dir, ignore_errors=True)
            metrics.total_ms = max(metrics.total_ms, int((time.perf_counter() - run_started) * 1000))
            self.db.finish_index_run(metrics, run_status, asdict(summary))
        return summary

    def _discover_files(
        self,
        root_path: Path,
        include_subfolders: bool,
        token: CancelToken,
        summary: IndexSummary,
        progress_callback: ProgressCallback | None,
    ) -> list[Path]:
        discovered: list[Path] = []
        last_emit = 0.0
        discovered_bytes = 0
        for file_path in iter_files(
            root_path,
            include_subfolders=include_subfolders,
            settings=self.settings,
            cancel_token=token,
        ):
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            discovered.append(file_path)
            discovered_bytes += _safe_file_size(file_path)
            summary.scanned += 1
            now = time.monotonic()
            if len(discovered) == 1 or len(discovered) % 100 == 0 or now - last_emit >= 0.2:
                self._emit(
                    progress_callback,
                    "discovering",
                    summary,
                    current_file=str(file_path),
                    pending=0,
                    total_files=len(discovered),
                    total_bytes=discovered_bytes,
                    phase_label="正在发现文件",
                )
                last_emit = now
        return discovered

    def _prepare_jobs(
        self,
        root_id: int,
        file_paths: list[Path],
        run_id: str,
        summary: IndexSummary,
        metrics: IndexRunMetrics,
        token: CancelToken,
    ) -> list[ParseJob]:
        prepared_rows: list[tuple[Path, int]] = []
        batch_size = max(64, min(1024, int(self.settings.index_write_batch_size or 32) * 8))
        for offset in range(0, len(file_paths), batch_size):
            token.throw_if_cancelled()
            chunk = file_paths[offset : offset + batch_size]
            versions = {
                str(path): parser_identity_for_path(path, self.settings)[1]
                for path in chunk
            }
            prepared, errors = self.db.upsert_file_metadata_many(
                root_id,
                chunk,
                retry_failed_files=self.settings.retry_failed_files,
                compute_full_hash=self.settings.compute_full_hash,
                mark_processing=False,
                parser_versions=versions,
            )
            for file_path, exc in errors:
                logger.error("Failed to read metadata for %s: %s", file_path, exc)
                summary.failed += 1
            for file_path, file_id, changed in prepared:
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    if changed:
                        self.db.mark_video_excluded([file_id])
                    summary.excluded_video += 1
                elif changed:
                    prepared_rows.append((file_path, file_id))
                else:
                    summary.skipped += 1

        fingerprinted: list[ParseJob] = []
        for file_path, file_id in prepared_rows:
            token.wait_if_paused()
            token.throw_if_cancelled()
            parser_name, parser_version = parser_identity_for_path(file_path, self.settings)
            fingerprint = self._fingerprint_for(file_path)
            if not self.settings.enable_parse_cache:
                content_key = f"path:{file_path}:{fingerprint.key}"
            else:
                content_key = fingerprint.key
            size_bytes = _safe_file_size(file_path)
            lane = lane_for(file_path, self.settings)
            fingerprinted.append(
                ParseJob(
                    file_id=file_id,
                    file_path=file_path,
                    content_key=content_key,
                    parser_name=parser_name,
                    parser_version=parser_version,
                    lane=lane,
                    size_bytes=size_bytes,
                    relevant_bytes=fingerprint.relevant_bytes,
                    estimated_cost=estimate_parse_cost(file_path, size_bytes, fingerprint.relevant_bytes),
                    queued_monotonic=time.perf_counter(),
                )
            )

        groups: dict[tuple[str, str, str], list[ParseJob]] = {}
        for job in fingerprinted:
            groups.setdefault((job.content_key, job.parser_name, job.parser_version), []).append(job)

        cache = (
            self.db.find_cached_documents(list(groups))
            if self.settings.enable_parse_cache and groups
            else {}
        )
        jobs: list[ParseJob] = []
        task_specs: list[tuple[int, str, str, int]] = []
        task_jobs: list[ParseJob] = []
        for identity, group in groups.items():
            cached = cache.get(identity)
            file_ids = [job.file_id for job in group]
            if cached is not None:
                document_id, status = cached
                self.db.link_cached_document(
                    file_ids,
                    document_id,
                    identity[0],
                    identity[1],
                    identity[2],
                    status,
                )
                metrics.cache_hits += len(group)
                if group[0].file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    summary.excluded_video += len(group)
                else:
                    summary.skipped += len(group)
                continue
            primary = group[0]
            primary.alias_file_ids = tuple(job.file_id for job in group[1:])
            jobs.append(primary)
            metrics.cache_misses += 1
            priority = max(1, min(1_000_000, int(primary.estimated_cost * 100)))
            task_specs.append((primary.file_id, run_id, primary.lane, priority))
            task_jobs.append(primary)

        task_ids = self.db.create_parse_tasks(task_specs)
        for job, task_id in zip(task_jobs, task_ids, strict=True):
            job.task_id = task_id
        jobs.sort(key=lambda job: (-job.estimated_cost, str(job.file_path).lower()))
        return jobs

    @staticmethod
    def _fingerprint_for(file_path: Path) -> ContentFingerprint:
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            stat = file_path.stat()
            key = hashlib.sha256(
                f"unsupported:{file_path}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
            ).hexdigest()
            return ContentFingerprint(f"metadata:{key}", 0, "metadata")
        return fingerprint_file(file_path)

    def _create_lanes(
        self,
        jobs: list[ParseJob],
        spool_dir: Path,
    ) -> tuple[dict[str, ParseLane], list[Executor], list[ProcessPoolExecutor]]:
        normal_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.settings.parser_workers or 1)),
            thread_name_prefix="lfts-parser",
        )
        zip_executor = _new_process_executor(self.settings, 1, spool_dir)
        ocr_executor = _new_process_executor(self.settings, 1, spool_dir)
        office_executor = _new_process_executor(
            self.settings,
            self.settings.process_parser_workers,
            spool_dir,
        )
        legacy_executor = _new_process_executor(self.settings, 1, spool_dir)
        lanes = {
            "normal": ParseLane(
                "normal",
                normal_executor,
                submission_window(
                    self.settings.normal_pending_tasks,
                    self.settings.parser_workers,
                ),
                effective_lane_budget(self.settings, "normal"),
                worker_count=max(1, int(self.settings.parser_workers or 1)),
            ),
            "zip": ParseLane(
                "zip",
                zip_executor,
                1,
                effective_lane_budget(self.settings, "zip"),
                process_based=True,
                worker_count=1,
            ),
            "ocr": ParseLane(
                "ocr",
                ocr_executor,
                1,
                effective_lane_budget(self.settings, "ocr"),
                process_based=True,
                worker_count=1,
            ),
            "office_process": ParseLane(
                "office_process",
                office_executor,
                max(
                    1,
                    min(
                        int(self.settings.process_pending_tasks or 1),
                        int(self.settings.process_parser_workers or 1),
                    ),
                ),
                effective_lane_budget(self.settings, "office_process"),
                process_based=True,
                worker_count=max(1, int(self.settings.process_parser_workers or 1)),
            ),
            "legacy_office": ParseLane(
                "legacy_office",
                legacy_executor,
                1,
                effective_lane_budget(self.settings, "legacy_office"),
                process_based=True,
                worker_count=1,
            ),
        }
        by_lane: dict[str, list[ParseJob]] = {name: [] for name in lanes}
        for job in jobs:
            by_lane[job.lane].append(job)
        for lane_name, lane_jobs in by_lane.items():
            lanes[lane_name].pending.extend(
                sorted(lane_jobs, key=lambda job: (-job.estimated_cost, str(job.file_path).lower()))
            )
        executors: list[Executor] = [
            normal_executor,
            zip_executor,
            ocr_executor,
            office_executor,
            legacy_executor,
        ]
        return lanes, executors, [
            zip_executor,
            ocr_executor,
            office_executor,
            legacy_executor,
        ]

    def _recycle_unhealthy_process_lanes(
        self,
        lanes: dict[str, ParseLane],
        executors: list[Executor],
        process_executors: list[ProcessPoolExecutor],
        spool_dir: Path,
    ) -> list[tuple[str, ParseJob, ParseResult, int | None]]:
        now = time.perf_counter()
        groups: dict[ProcessPoolExecutor, list[ParseLane]] = {}
        for lane in lanes.values():
            if lane.process_based and isinstance(lane.executor, ProcessPoolExecutor):
                groups.setdefault(lane.executor, []).append(lane)
        completed: list[tuple[str, ParseJob, ParseResult, int | None]] = []
        for executor, group in list(groups.items()):
            broken = bool(getattr(executor, "_broken", False))
            timed_out = any(
                job.watchdog_timed_out
                for lane in group
                for future, job in lane.jobs.items()
                if not future.done()
            )
            if not broken and not timed_out:
                continue
            impacted: list[tuple[ParseLane, ParseJob]] = []
            for lane in group:
                for future, job in list(lane.jobs.items()):
                    if timed_out and not broken and future.done():
                        continue
                    future.cancel()
                    impacted.append((lane, job))
                    lane.futures.discard(future)
                    lane.jobs.pop(future, None)
                    lane.inflight_bytes = max(
                        0,
                        lane.inflight_bytes - max(1, job.size_bytes),
                    )
            terminate_process_pool_workers(executor, spool_dir)
            try:
                executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                logger.debug("Unable to close recycled process pool", exc_info=True)
            with self._executor_lock:
                self._active_process_executors.discard(executor)
            if executor in process_executors:
                process_executors.remove(executor)
            if executor in executors:
                executors.remove(executor)
            workers = max(lane.worker_count for lane in group)
            replacement = _new_process_executor(self.settings, workers, spool_dir)
            for lane in group:
                lane.executor = replacement
            process_executors.append(replacement)
            executors.append(replacement)
            with self._executor_lock:
                self._active_process_executors.add(replacement)
            for lane, job in reversed(impacted):
                job_timed_out = bool(job.watchdog_timed_out)
                if job_timed_out:
                    checkpoint = load_partial_parse_checkpoint(job, spool_dir, consume=False)
                    max_retries = max(0, int(self.settings.no_progress_max_retries))
                    if job.retry_count < max_retries:
                        job.retry_count += 1
                        if (
                            checkpoint is not None
                            and checkpoint.resume_cursor > 0
                            and job.parser_name in {"pdf", "zip"}
                        ):
                            job.resume_cursor = checkpoint.resume_cursor
                        else:
                            partial_parse_path(job, spool_dir).unlink(missing_ok=True)
                            job.resume_cursor = 0
                        reset_job_for_retry(job, now)
                        lane.pending.appendleft(job)
                        continue
                    reason_code = "PARSE_NO_PROGRESS"
                    reason_text = (
                        f"连续 {max_retries + 1} 次在“{job.progress_phase or '解析'}”阶段无有效进展，"
                        f"最近一次等待 {max(1, job.timeout_seconds)} 秒；完整索引尚未发布"
                    )
                else:
                    # Pool recycling can interrupt another healthy task sharing
                    # the same executor. Requeue it without consuming a retry.
                    reset_job_for_retry(job, now)
                    lane.pending.appendleft(job)
                    continue
                outcome = _diagnostic_outcome(
                    job,
                    "process_worker",
                    "failed_retryable",
                    reason_code,
                    reason_text,
                    job.started_monotonic or now,
                )
                completed.append((lane.name, job, outcome, None))
        return completed

    def _outcome_from_result(
        self,
        job: ParseJob,
        result: ParseResult,
        spool_dir: Path,
    ) -> ParseOutcome:
        descriptor = result if isinstance(result, SpoolParseResult) else None
        if descriptor is not None and job.task_id is not None:
            if not self.db.try_mark_task_spooled(
                job.task_id,
                descriptor.spool_path,
                descriptor.checksum,
            ):
                logger.debug(
                    "Skipped parse task spooled-state update while SQLite is busy: task_id=%s",
                    job.task_id,
                )
        outcome = materialize_parse_result(result, spool_dir)
        hydrate_outcome(outcome, job)
        if descriptor is not None:
            outcome.worker_pid = descriptor.worker_pid
            outcome.spool_bytes = descriptor.result_bytes
            outcome.spool_path = descriptor.spool_path
            outcome.spool_checksum = descriptor.checksum
            outcome.spool_write_ms = descriptor.spool_write_ms
        return outcome

    def _recover_spooled_tasks(self, root_id: int) -> int:
        recovered = 0
        for row in self.db.recoverable_spooled_tasks(root_id):
            spool_text = str(row["spool_path"] or "")
            checksum = str(row["spool_checksum"] or "")
            if not spool_text:
                continue
            spool_path = Path(spool_text)
            try:
                expected_root = (TEMP_DIR / "process_results").resolve()
                resolved_spool = spool_path.resolve()
                if expected_root not in resolved_spool.parents:
                    raise ValueError("Spool path is outside the controlled run directory")
                if not spool_path.is_file() or sha256_path(spool_path) != checksum:
                    raise ValueError("Spool checksum mismatch")
                with spool_path.open("rb") as stream:
                    outcome = pickle.load(stream)
                if not isinstance(outcome, ParseOutcome):
                    raise TypeError("Invalid spool artifact")
                outcome.task_id = int(row["id"])
                outcome.spool_path = spool_path
                self.db.replace_document_blocks_many([IndexWriter._to_item(outcome)])
                spool_path.unlink(missing_ok=True)
                recovered += 1 + len(outcome.alias_file_ids)
            except Exception as exc:
                logger.warning("Unable to recover spool %s: %s", spool_path, exc)
                self.db.mark_task_failed(int(row["id"]), "SPOOL_INVALID", str(exc))
                self.db.invalidate_file(str(row["path"]))
        return recovered

    def _emit_active_progress(
        self,
        progress_callback: ProgressCallback | None,
        summary: IndexSummary,
        lanes: dict[str, ParseLane],
        estimator: IndexTimeEstimator,
        total_files: int,
        **extra: object,
    ) -> None:
        snapshot = lane_progress_snapshot(lanes.values())
        estimate = estimator.estimate(
            snapshot["remaining_cost_by_lane"],
            snapshot["active_elapsed_by_lane"],
        )
        self._emit(
            progress_callback,
            "indexing",
            summary,
            current_file=snapshot["current_file"],
            pending=pending_lane_tasks(lanes.values()),
            queue=snapshot["current_lane"],
            active_elapsed_seconds=snapshot["active_elapsed_seconds"],
            active_file_count=snapshot["active_file_count"],
            active_phase=snapshot["current_phase"],
            active_completed_units=snapshot["current_completed"],
            active_total_units=snapshot["current_total"],
            active_progress_detail=snapshot["current_detail"],
            no_progress_seconds=snapshot["no_progress_seconds"],
            retry_count=snapshot["retry_count"],
            total_files=total_files,
            completed_files=(
                summary.indexed
                + summary.skipped
                + summary.failed
                + summary.unsupported
                + summary.metadata_only
                + summary.partial_success
            ),
            eta_lower_seconds=(estimate.lower_seconds if estimate else 0),
            eta_upper_seconds=(estimate.upper_seconds if estimate else 0),
            eta_sample_count=(estimate.sample_count if estimate else 0),
            phase_label="正在解析并写入索引",
            index_phase="parsing",
            **extra,
        )

    def _emit(
        self,
        progress_callback: ProgressCallback | None,
        stage: str,
        summary: IndexSummary,
        **extra: object,
    ) -> None:
        if progress_callback is None:
            return
        payload: dict[str, object] = {
            "stage": stage,
            "scanned": summary.scanned,
            "indexed": summary.indexed,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "unsupported": summary.unsupported,
            "metadata_only": summary.metadata_only,
            "partial_success": summary.partial_success,
            "excluded_video": summary.excluded_video,
            "deleted": summary.deleted,
            "cancelled": summary.cancelled,
        }
        payload.update(extra)
        progress_callback(payload)


def pending_lane_tasks(lanes: Iterable[ParseLane]) -> int:
    return sum(len(lane.pending) + len(lane.futures) for lane in lanes)


def lane_costs(jobs: Iterable[ParseJob]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for job in jobs:
        costs[job.lane] = costs.get(job.lane, 0.0) + max(0.0, job.estimated_cost)
    return costs


def lane_progress_snapshot(lanes: Iterable[ParseLane]) -> dict[str, object]:
    now = time.perf_counter()
    remaining: dict[str, float] = {}
    active_elapsed: dict[str, float] = {}
    active_jobs: list[tuple[float, str, ParseJob]] = []
    first_pending: tuple[str, ParseJob] | None = None
    active_count = 0
    for lane in lanes:
        remaining[lane.name] = sum(
            max(0.0, job.estimated_cost)
            for job in lane.pending
        ) + sum(
            max(0.0, job.estimated_cost)
            for job in lane.jobs.values()
        )
        if first_pending is None and lane.pending:
            first_pending = (lane.name, lane.pending[0])
        for future, job in list(lane.jobs.items()):
            if future.done() or not job.started_monotonic:
                continue
            elapsed = max(0.0, now - job.started_monotonic)
            active_count += 1
            active_elapsed[lane.name] = max(active_elapsed.get(lane.name, 0.0), elapsed)
            active_jobs.append((elapsed, lane.name, job))
    if active_jobs:
        elapsed, lane_name, current_job = max(active_jobs, key=lambda item: item[0])
    elif first_pending is not None:
        lane_name, current_job = first_pending
        elapsed = 0.0
    else:
        lane_name, current_job, elapsed = "", None, 0.0
    return {
        "remaining_cost_by_lane": remaining,
        "active_elapsed_by_lane": active_elapsed,
        "current_file": str(current_job.file_path) if current_job is not None else "",
        "current_lane": lane_name,
        "active_elapsed_seconds": int(elapsed),
        "active_file_count": active_count,
        "current_phase": current_job.progress_phase if current_job is not None else "",
        "current_completed": current_job.progress_completed if current_job is not None else 0,
        "current_total": current_job.progress_total if current_job is not None else 0,
        "current_detail": current_job.progress_detail if current_job is not None else "",
        "no_progress_seconds": (
            int(max(0.0, now - (current_job.last_progress_monotonic or current_job.started_monotonic)))
            if current_job is not None and current_job.started_monotonic
            else 0
        ),
        "retry_count": current_job.retry_count if current_job is not None else 0,
    }


def terminate_process_pool_workers(
    executor: ProcessPoolExecutor,
    registry_dir: Path | None = None,
) -> None:
    processes = list((getattr(executor, "_processes", None) or {}).values())
    descendants: list[object] = []
    try:
        import psutil

        for process in processes:
            try:
                descendants.extend(psutil.Process(process.pid).children(recursive=True))
            except (psutil.Error, OSError):
                continue
    except ImportError:
        descendants = []
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            logger.debug("Unable to terminate process-pool worker", exc_info=True)
    for process in processes:
        try:
            process.join(timeout=0.2)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        except (OSError, ValueError):
            logger.debug("Unable to join process-pool worker", exc_info=True)
    for child in reversed(descendants):
        try:
            if child.is_running():
                child.terminate()
        except Exception:
            logger.debug("Unable to terminate parser child process", exc_info=True)
    for child in reversed(descendants):
        try:
            child.wait(timeout=0.5)
        except Exception:
            try:
                if child.is_running():
                    child.kill()
            except Exception:
                logger.debug("Unable to kill parser child process", exc_info=True)
    if registry_dir is not None:
        cleanup_registered_office_processes(registry_dir)


def _new_process_executor(
    settings: AppSettings,
    workers: int,
    registry_dir: Path | None = None,
) -> ProcessPoolExecutor:
    max_tasks = max(1, int(settings.process_max_tasks_per_child or settings.process_recycle_max_tasks or 32))
    return ProcessPoolExecutor(
        max_workers=max(1, int(workers or 1)),
        max_tasks_per_child=max_tasks,
        initializer=initialize_process_worker,
        initargs=(settings, registry_dir),
    )


def initialize_process_worker(
    settings: AppSettings,
    registry_dir: Path | None = None,
) -> None:
    global _process_registry
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        import psutil

        process = psutil.Process()
        if os.name == "nt":
            process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            process.nice(5)
    except Exception:
        logger.debug("Unable to lower parser worker priority", exc_info=True)
    if registry_dir is not None:
        os.environ["LFTS_PROCESS_REGISTRY_DIR"] = str(registry_dir)
    _process_registry = ParserRegistry(settings)


def submission_window(configured: int, workers: int) -> int:
    """Bound executor prefetch while the lane deque owns remaining work."""

    worker_count = max(1, int(workers or 1))
    return max(1, min(int(configured or 1), worker_count * 2))


def schedule_parse_lanes(
    lanes: Iterable[ParseLane],
    settings: AppSettings,
    token: CancelToken,
    spool_dir: Path,
) -> list[int | None]:
    submitted_task_ids: list[int | None] = []
    lane_list = list(lanes)
    global_budget = max(128, int(settings.index_memory_budget_mb)) * 1024 * 1024
    global_inflight = sum(lane.inflight_bytes for lane in lane_list)
    for lane in lane_list:
        while lane.pending and len(lane.futures) < lane.max_in_flight:
            if lane.process_based:
                shared_inflight = sum(
                    len(other.futures)
                    for other in lane_list
                    if other.executor is lane.executor
                )
                if shared_inflight >= lane.worker_count:
                    break
            token.wait_if_paused()
            token.throw_if_cancelled()
            job = lane.pending[0]
            job_bytes = max(1, job.size_bytes)
            if lane.futures and lane.inflight_bytes + job_bytes > lane.max_inflight_bytes:
                break
            if global_inflight and global_inflight + job_bytes > global_budget:
                break
            lane.pending.popleft()
            job.started_monotonic = time.perf_counter()
            job.watchdog_timed_out = False
            job.last_progress_monotonic = job.started_monotonic
            job.timeout_seconds = no_progress_timeout(settings, job)
            try:
                if lane.process_based:
                    future = lane.executor.submit(parse_file_process_worker, job, settings, spool_dir)
                else:
                    future = lane.executor.submit(parse_file_worker, job, settings, token)
            except Exception:
                if token.cancelled:
                    raise CancelledError("任务已取消")
                raise
            lane.futures.add(future)
            lane.jobs[future] = job
            lane.inflight_bytes += job_bytes
            global_inflight += job_bytes
            submitted_task_ids.append(job.task_id)
    return submitted_task_ids


def drain_completed_lanes(
    lanes: Iterable[ParseLane],
    token: CancelToken,
    spool_dir: Path,
    *,
    block: bool = False,
) -> list[tuple[str, ParseJob, ParseResult, int | None]]:
    lane_list = list(lanes)
    futures = {future for lane in lane_list for future in lane.futures}
    if not futures:
        return []
    if block:
        done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
    else:
        done = {future for future in futures if future.done()}
    if not done:
        token.wait_if_paused()
        token.throw_if_cancelled()
        return []
    completed: list[tuple[str, ParseJob, ParseResult, int | None]] = []
    for lane in lane_list:
        lane_done = lane.futures.intersection(done)
        lane.futures.difference_update(lane_done)
        for future in lane_done:
            token.wait_if_paused()
            token.throw_if_cancelled()
            job = lane.jobs.pop(future)
            lane.inflight_bytes = max(0, lane.inflight_bytes - max(1, job.size_bytes))
            descriptor_bytes: int | None = None
            try:
                result = future.result()
                if isinstance(result, SpoolParseResult):
                    descriptor_bytes = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            except CancelledError:
                raise
            except Exception as exc:
                if token.cancelled:
                    raise CancelledError("任务已取消") from exc
                if job.watchdog_timed_out:
                    result = _diagnostic_outcome(
                        job,
                        "process_worker",
                        "failed_retryable",
                        "PARSE_NO_PROGRESS",
                        (
                            f"“{job.progress_phase or '解析'}”阶段连续 "
                            f"{max(1, job.timeout_seconds)} 秒无有效进展"
                        ),
                        job.started_monotonic or time.perf_counter(),
                    )
                else:
                    logger.error(
                        "Parse lane %s failed for %s\n%s",
                        lane.name,
                        job.file_path,
                        traceback.format_exc(),
                    )
                    result = failed_parse_outcome(job, exc)
            completed.append((lane.name, job, result, descriptor_bytes))
    return completed


def parse_file_worker(job: ParseJob, settings: AppSettings, cancel_token: CancelToken) -> ParseOutcome:
    return parse_file_with_registry(job, worker_registry(settings), cancel_token, settings)


def parse_file_process_worker(job: ParseJob, settings: AppSettings, spool_dir: Path) -> SpoolParseResult:
    registry = _process_registry or ParserRegistry(settings)
    checkpoint_path = partial_parse_path(job, spool_dir)
    outcome = parse_file_with_registry(
        job,
        registry,
        CancelToken(),
        settings,
        checkpoint_path=checkpoint_path,
        progress_path=process_progress_path(job, spool_dir),
    )
    outcome.worker_pid = os.getpid()
    spool_dir.mkdir(parents=True, exist_ok=True)
    spool_path = spool_dir / f"{job.file_id}_{uuid.uuid4().hex}.pickle"
    temporary_path = spool_path.with_suffix(".tmp")
    started = time.perf_counter()
    with temporary_path.open("wb") as stream:
        pickle.dump(outcome, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary_path.replace(spool_path)
    checkpoint_path.unlink(missing_ok=True)
    process_progress_path(job, spool_dir).unlink(missing_ok=True)
    spool_write_ms = int((time.perf_counter() - started) * 1000)
    checksum = sha256_path(spool_path)
    return SpoolParseResult(
        job.file_id,
        job.file_path,
        spool_path,
        worker_pid=os.getpid(),
        result_bytes=spool_path.stat().st_size,
        checksum=checksum,
        spool_write_ms=spool_write_ms,
    )


def parse_file_with_registry(
    job: ParseJob,
    registry: ParserRegistry,
    cancel_token: CancelToken,
    settings: AppSettings | None = None,
    *,
    checkpoint_path: Path | None = None,
    progress_path: Path | None = None,
) -> ParseOutcome:
    started = time.perf_counter()
    try:
        cancel_token.wait_if_paused()
        cancel_token.throw_if_cancelled()
        parser = registry.parser_for(job.file_path)
        parser.reset_status()
        logical_blocks: list[ContentBlock] = []
        if parser.supports_resume and job.resume_cursor > 0 and checkpoint_path is not None:
            checkpoint = load_partial_parse_checkpoint(job, checkpoint_path.parent, consume=False)
            if checkpoint is not None and checkpoint.resume_cursor == job.resume_cursor:
                logical_blocks.extend(checkpoint.blocks)
            else:
                job.resume_cursor = 0
        checkpointed_at = 0.0
        progress_sequence = max(0, int(job.progress_sequence))
        last_checkpoint_block_count = len(logical_blocks)

        def report_parser_progress(payload: dict[str, object]) -> None:
            nonlocal checkpointed_at, progress_sequence, last_checkpoint_block_count
            progress_sequence += 1
            phase = str(payload.get("phase") or parser.name)
            completed = max(0, int(payload.get("completed") or 0))
            total = max(0, int(payload.get("total") or 0))
            detail = str(payload.get("detail") or "")
            cursor_value = payload.get("cursor")
            cursor = int(cursor_value) if isinstance(cursor_value, int) else job.resume_cursor
            progress_payload = {
                "job_id": job.task_id,
                "file_id": job.file_id,
                "file_path": str(job.file_path),
                "parser_name": parser.name,
                "worker_pid": os.getpid(),
                "progress_sequence": progress_sequence,
                "phase": phase,
                "completed": completed,
                "total": total,
                "unit_type": str(payload.get("unit_type") or ""),
                "cursor": cursor,
                "detail": detail,
                "updated_at": time.time(),
            }
            job.progress_sequence = progress_sequence
            job.progress_phase = phase
            job.progress_completed = completed
            job.progress_total = total
            job.progress_detail = detail
            job.last_progress_monotonic = time.perf_counter()
            if progress_path is not None:
                write_process_progress(progress_path, progress_payload)
            now = time.perf_counter()
            should_checkpoint = (
                checkpoint_path is not None
                and (
                    parser.supports_resume
                    or (
                        bool(logical_blocks)
                        and (
                            last_checkpoint_block_count == 0
                            or len(logical_blocks) - last_checkpoint_block_count >= 8
                            or now - checkpointed_at >= 2.0
                        )
                    )
                )
            )
            if should_checkpoint:
                write_partial_parse_checkpoint(
                    checkpoint_path,
                    job,
                    parser.name,
                    logical_blocks,
                    settings or registry.settings,
                    started,
                    resume_cursor=cursor,
                    progress_phase=phase,
                    progress_completed=completed,
                    progress_total=total,
                )
                checkpointed_at = now
                last_checkpoint_block_count = len(logical_blocks)

        parser.configure_runtime(
            resume_cursor=job.resume_cursor,
            progress_callback=report_parser_progress,
        )
        report_parser_progress(
            {
                "phase": "starting",
                "completed": job.resume_cursor,
                "cursor": job.resume_cursor,
                "detail": job.file_path.name,
            }
        )
        for block in parser.parse(job.file_path, cancel_token):
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            if block.raw_text.strip():
                logical_blocks.append(block)
                if not parser.supports_resume:
                    report_parser_progress(
                        {
                            "phase": parser.name,
                            "completed": len(logical_blocks),
                            "unit_type": block.block_type,
                            "detail": block.location_text,
                        }
                    )
        parse_ms = int((time.perf_counter() - started) * 1000)
        normalize_started = time.perf_counter()
        effective_settings = settings or registry.settings
        blocks = BlockCoalescer(
            effective_settings.block_target_chars,
            effective_settings.block_max_chars,
        ).coalesce(logical_blocks)
        normalize_ms = int((time.perf_counter() - normalize_started) * 1000)
        status = parser.last_status or "success"
        return ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=blocks,
            parser_name=parser.name,
            status=status,
            error_code=(parser.last_error_code if status != "success" else None),
            error_message=(parser.last_error_message if status != "success" else None),
            task_id=job.task_id,
            alias_file_ids=job.alias_file_ids,
            content_key=job.content_key,
            parser_version=job.parser_version,
            lane=job.lane,
            size_bytes=job.size_bytes,
            estimated_cost=job.estimated_cost,
            queue_wait_ms=max(0, int((job.started_monotonic - job.queued_monotonic) * 1000)),
            parse_ms=parse_ms,
            normalize_ms=normalize_ms,
            worker_pid=os.getpid(),
            resume_cursor=0,
            progress_phase="complete",
            progress_completed=progress_sequence,
        )
    except UnsupportedFormatError as exc:
        return _diagnostic_outcome(job, "metadata", "unsupported", "UNSUPPORTED_FORMAT", str(exc), started)
    except CancelledError:
        raise
    except Exception as exc:
        logger.error("Failed to parse %s\n%s", job.file_path, traceback.format_exc())
        retryable = isinstance(exc, OSError) and not isinstance(exc, (PermissionError, FileNotFoundError))
        status = "password_protected" if isinstance(exc, PasswordProtectedError) else (
            "failed_retryable" if retryable else "failed"
        )
        return _diagnostic_outcome(
            job,
            exc.__class__.__name__,
            status,
            error_code_for_exception(exc),
            user_message_for_exception(exc),
            started,
        )


def _diagnostic_outcome(
    job: ParseJob,
    parser_name: str,
    status: str,
    error_code: str,
    error_message: str,
    started: float,
) -> ParseOutcome:
    return ParseOutcome(
        file_id=job.file_id,
        file_path=job.file_path,
        blocks=[],
        parser_name=parser_name,
        status=status,
        error_code=error_code,
        error_message=error_message,
        task_id=job.task_id,
        alias_file_ids=job.alias_file_ids,
        content_key=job.content_key,
        parser_version=job.parser_version,
        lane=job.lane,
        size_bytes=job.size_bytes,
        estimated_cost=job.estimated_cost,
        queue_wait_ms=max(0, int((job.started_monotonic - job.queued_monotonic) * 1000)),
        parse_ms=int((time.perf_counter() - started) * 1000),
        worker_pid=os.getpid(),
    )


def materialize_parse_result(result: ParseResult, spool_dir: Path) -> ParseOutcome:
    if isinstance(result, ParseOutcome):
        return result
    expected_root = spool_dir.resolve()
    result_path = result.spool_path.resolve()
    if expected_root not in result_path.parents:
        raise ValueError(f"Unexpected process result path: {result_path}")
    if sha256_path(result_path) != result.checksum:
        raise ValueError("Process parse result checksum mismatch")
    with result_path.open("rb") as stream:
        outcome = pickle.load(stream)
    if not isinstance(outcome, ParseOutcome):
        raise TypeError("Invalid process parse result")
    if outcome.file_id != result.file_id or outcome.file_path != result.file_path:
        raise ValueError("Process parse result identity mismatch")
    return outcome


def hydrate_outcome(outcome: ParseOutcome, job: ParseJob) -> None:
    outcome.task_id = job.task_id
    outcome.alias_file_ids = job.alias_file_ids
    outcome.content_key = job.content_key
    outcome.parser_version = job.parser_version
    outcome.lane = job.lane
    outcome.size_bytes = job.size_bytes
    outcome.estimated_cost = job.estimated_cost
    if not outcome.queue_wait_ms and job.started_monotonic and job.queued_monotonic:
        outcome.queue_wait_ms = max(0, int((job.started_monotonic - job.queued_monotonic) * 1000))


def failed_parse_outcome(job: ParseJob, exc: Exception) -> ParseOutcome:
    if isinstance(exc, BrokenProcessPool):
        return _diagnostic_outcome(
            job,
            "process_worker",
            "failed_retryable",
            "PROCESS_WORKER_CRASH",
            "解析子进程异常退出，任务可重试",
            job.started_monotonic or time.perf_counter(),
        )
    retryable = isinstance(exc, OSError) and not isinstance(exc, (PermissionError, FileNotFoundError))
    return _diagnostic_outcome(
        job,
        exc.__class__.__name__,
        "failed_retryable" if retryable else "failed",
        error_code_for_exception(exc),
        user_message_for_exception(exc),
        job.started_monotonic or time.perf_counter(),
    )


def worker_registry(settings: AppSettings) -> ParserRegistry:
    settings_key = repr(settings.to_dict())
    if getattr(_worker_state, "settings_key", None) != settings_key:
        _worker_state.settings_key = settings_key
        _worker_state.registry = ParserRegistry(settings)
    return _worker_state.registry


def parser_identity_for_path(file_path: Path, settings: AppSettings) -> tuple[str, str]:
    suffix = file_path.suffix.lower()
    if suffix == ".docx":
        name = "docx_stream" if settings.fast_ooxml_enabled else "docx"
    elif suffix in {".xlsx", ".xlsm"}:
        name = "xlsx_stream" if settings.fast_ooxml_enabled else "xlsx"
    elif suffix == ".pptx":
        name = "pptx_stream" if settings.fast_ooxml_enabled else "pptx"
    elif suffix == ".pdf":
        name = "pdf"
    elif suffix in IMAGE_EXTENSIONS:
        name = "image_ocr"
    elif suffix in ARCHIVE_EXTENSIONS:
        name = "zip"
    elif suffix in LEGACY_OFFICE_EXTENSIONS:
        name = "legacy_office"
    elif suffix in {".txt", ".log", ".csv", ".md", ".json", ".xml", ".ini"}:
        name = "text"
    else:
        name = "metadata"
    version = PARSER_VERSIONS.get(name, "1")
    dependency = {
        "docx": "docx",
        "docx_stream": "lxml",
        "xlsx": "openpyxl",
        "xlsx_stream": "lxml",
        "pptx": "pptx",
        "pptx_stream": "lxml",
        "pdf": "fitz",
        "image_ocr": "PIL",
    }.get(name)
    if dependency:
        version += f":dep={int(importlib.util.find_spec(dependency) is not None)}"
    if name == "pdf":
        version += f":ocr={int(settings.enable_ocr and settings.ocr_scanned_pdf)}:{settings.ocr_language}"
    elif name == "image_ocr":
        version += (
            f":ocr={int(settings.enable_ocr and settings.ocr_images)}:{settings.ocr_language}"
            f":min={settings.min_ocr_image_pixels}:detect={settings.max_ocr_image_side}"
            ":adaptive=1:tile=1280:overlap=160"
        )
    elif name == "zip":
        version += (
            f":fast={int(settings.fast_ooxml_enabled)}:depth={settings.max_zip_depth}"
            f":ocr={int(settings.enable_ocr)}"
        )
    elif name == "legacy_office":
        version += f":fast={int(settings.fast_ooxml_enabled)}"
    return name, version


def lane_for(file_path: Path, settings: AppSettings) -> str:
    suffix = file_path.suffix.lower()
    if settings.enable_ocr and settings.ocr_images and suffix in IMAGE_EXTENSIONS:
        return "ocr"
    if settings.enable_ocr and settings.ocr_scanned_pdf and suffix == ".pdf":
        return "ocr"
    if suffix in ARCHIVE_EXTENSIONS:
        return "zip"
    if suffix in LEGACY_OFFICE_EXTENSIONS:
        return "legacy_office"
    if suffix == ".pdf":
        return "office_process"
    if suffix in {".docx", ".xlsx", ".xlsm", ".pptx"}:
        return "office_process"
    return "normal"


def effective_lane_budget(settings: AppSettings, lane: str) -> int:
    configured = {
        "normal": settings.normal_inflight_bytes,
        "ocr": settings.ocr_inflight_bytes,
        "zip": settings.slow_inflight_bytes,
        "office_process": settings.office_inflight_bytes,
        "legacy_office": settings.slow_inflight_bytes,
    }[lane]
    multiplier = {"low_resource": 0.5, "balanced": 1.0, "fastest": 1.5}.get(
        settings.index_performance_preset,
        1.0,
    )
    global_budget = max(128, int(settings.index_memory_budget_mb)) * 1024 * 1024
    if lane == "office_process":
        configured = min(
            configured,
            max(128, int(settings.process_memory_budget_mb))
            * 1024
            * 1024
            * max(1, int(settings.process_parser_workers)),
        )
    elif lane == "legacy_office":
        configured = min(
            configured,
            max(128, int(settings.process_memory_budget_mb)) * 1024 * 1024,
        )
    return max(16 * 1024 * 1024, min(int(configured * multiplier), global_budget))


def merge_summary(target: IndexSummary, source: IndexSummary) -> None:
    target.scanned += source.scanned
    target.indexed += source.indexed
    target.skipped += source.skipped
    target.failed += source.failed
    target.unsupported += source.unsupported
    target.metadata_only += source.metadata_only
    target.partial_success += source.partial_success
    target.excluded_video += source.excluded_video
    target.deleted += source.deleted
    target.cancelled = target.cancelled or source.cancelled


def record_parse_outcome(
    summary: IndexSummary,
    status: str,
    count: int = 1,
    *,
    extension: str = "",
) -> None:
    if extension.lower() in VIDEO_EXTENSIONS:
        summary.excluded_video += count
        return
    if status == "success":
        summary.indexed += count
        return
    if status in {"failed", "failed_retryable", "password_protected"}:
        summary.failed += count
        return
    if status == "unsupported":
        summary.unsupported += count
        return
    if status == "skipped":
        summary.skipped += count
        return
    if status in {"metadata_only", "ocr_disabled", "converter_missing"}:
        summary.metadata_only += count
    elif status == "partial_success":
        summary.partial_success += count
    else:
        summary.failed += count


def record_index_status(summary: IndexSummary, status: str) -> None:
    record_parse_outcome(summary, status)


def error_code_for_exception(exc: Exception) -> str:
    if isinstance(exc, ParserDependencyError):
        return "PARSER_DEPENDENCY_MISSING"
    if isinstance(exc, PasswordProtectedError):
        return "PASSWORD_PROTECTED"
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED"
    if isinstance(exc, FileNotFoundError):
        return "FILE_NOT_FOUND"
    if isinstance(exc, OSError):
        return "FILE_IN_USE"
    return "PARSER_ERROR"


def user_message_for_exception(exc: Exception) -> str:
    if isinstance(exc, ParserDependencyError):
        return str(exc)
    if isinstance(exc, PasswordProtectedError):
        return "文件已加密，需要密码"
    if isinstance(exc, PermissionError):
        return "没有权限读取该文件"
    if isinstance(exc, FileNotFoundError):
        return "文件不存在或已被移动"
    return str(exc) or exc.__class__.__name__


def partial_parse_path(job: ParseJob, spool_dir: Path) -> Path:
    """Return the per-job checkpoint path inside the current controlled run spool."""

    return spool_dir / f"{job.file_id}.partial.pickle"


def process_progress_path(job: ParseJob, spool_dir: Path) -> Path:
    return spool_dir / f"{job.file_id}.progress.json"


def write_process_progress(path: Path, payload: dict[str, object]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        logger.debug("Unable to write parser progress %s", path, exc_info=True)


def refresh_job_progress(job: ParseJob, spool_dir: Path, observed_at: float) -> None:
    path = process_progress_path(job, spool_dir)
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("file_id") or -1) != job.file_id:
            return
        sequence = max(0, int(payload.get("progress_sequence") or 0))
        if sequence <= job.progress_sequence:
            return
        job.progress_sequence = sequence
        job.progress_phase = str(payload.get("phase") or "")
        job.progress_completed = max(0, int(payload.get("completed") or 0))
        job.progress_total = max(0, int(payload.get("total") or 0))
        job.progress_detail = str(payload.get("detail") or "")
        job.last_progress_monotonic = observed_at
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.debug("Unable to read parser progress %s", path, exc_info=True)


def no_progress_timeout(settings: AppSettings, job: ParseJob) -> int:
    base = {
        "ocr": settings.ocr_no_progress_timeout_seconds,
        "zip": settings.archive_no_progress_timeout_seconds,
        "legacy_office": settings.legacy_no_progress_timeout_seconds,
        "office_process": settings.process_no_progress_timeout_seconds,
        "normal": settings.normal_no_progress_timeout_seconds,
    }.get(job.lane, settings.normal_no_progress_timeout_seconds)
    if "ocr" in job.progress_phase.lower():
        base = max(base, settings.ocr_no_progress_timeout_seconds)
    multiplier = 1 << min(max(0, int(job.retry_count)), 2)
    return max(1, int(base)) * multiplier


def reset_job_for_retry(job: ParseJob, queued_at: float) -> None:
    job.started_monotonic = 0.0
    job.watchdog_timed_out = False
    job.last_progress_monotonic = 0.0
    job.timeout_seconds = 0
    job.queued_monotonic = queued_at


def write_partial_parse_checkpoint(
    checkpoint_path: Path,
    job: ParseJob,
    parser_name: str,
    logical_blocks: list[ContentBlock],
    settings: AppSettings,
    started: float,
    *,
    resume_cursor: int = 0,
    progress_phase: str = "",
    progress_completed: int = 0,
    progress_total: int = 0,
) -> None:
    """Persist an atomic parse checkpoint so worker recycling is loss-bounded."""

    try:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        blocks = BlockCoalescer(
            settings.block_target_chars,
            settings.block_max_chars,
        ).coalesce(logical_blocks)
        outcome = ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=blocks,
            parser_name=parser_name,
            status="partial_success",
            error_code="PARSE_CHECKPOINT",
            error_message="解析仍在进行，已保存已完成内容",
            task_id=job.task_id,
            alias_file_ids=job.alias_file_ids,
            content_key=job.content_key,
            parser_version=job.parser_version,
            lane=job.lane,
            size_bytes=job.size_bytes,
            estimated_cost=job.estimated_cost,
            queue_wait_ms=max(0, int((job.started_monotonic - job.queued_monotonic) * 1000)),
            parse_ms=int((time.perf_counter() - started) * 1000),
            worker_pid=os.getpid(),
            resume_cursor=max(0, int(resume_cursor)),
            progress_phase=progress_phase,
            progress_completed=max(0, int(progress_completed)),
            progress_total=max(0, int(progress_total)),
        )
        temporary_path = checkpoint_path.with_suffix(f".tmp.{os.getpid()}")
        with temporary_path.open("wb") as stream:
            pickle.dump(outcome, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_path.replace(checkpoint_path)
    except Exception:
        # Checkpointing is best effort and must not turn a healthy parser into a failure.
        logger.debug("Unable to write parse checkpoint %s", checkpoint_path, exc_info=True)


def load_partial_parse_checkpoint(
    job: ParseJob,
    spool_dir: Path,
    *,
    consume: bool = True,
) -> ParseOutcome | None:
    checkpoint_path = partial_parse_path(job, spool_dir)
    if not checkpoint_path.is_file():
        return None
    try:
        with checkpoint_path.open("rb") as stream:
            outcome = pickle.load(stream)
        if not isinstance(outcome, ParseOutcome):
            raise TypeError("Invalid partial parse checkpoint")
        if outcome.file_id != job.file_id or outcome.file_path != job.file_path:
            raise ValueError("Partial parse checkpoint identity mismatch")
        hydrate_outcome(outcome, job)
        if consume:
            checkpoint_path.unlink(missing_ok=True)
        return outcome
    except Exception:
        logger.warning("Unable to load partial parse checkpoint %s", checkpoint_path, exc_info=True)
        checkpoint_path.unlink(missing_ok=True)
        return None


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_process_tree_rss() -> int:
    try:
        import psutil

        process = psutil.Process()
        return int(process.memory_info().rss) + sum(
            int(child.memory_info().rss)
            for child in process.children(recursive=True)
            if child.is_running()
        )
    except Exception:
        return 0


def current_process_rss() -> int:
    """Return only parent RSS; child enumeration is kept off the hot path."""

    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0
