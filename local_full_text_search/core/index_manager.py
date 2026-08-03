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
import tempfile
import uuid
import zipfile
from collections import deque
from collections.abc import Iterable
from concurrent.futures import (
    FIRST_COMPLETED,
    Executor,
    Future,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
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
from local_full_text_search.core.atomic_fts_publish import (
    IndexPublishGateError,
    IndexVersionPublisher,
)
from local_full_text_search.core.content_fingerprint import (
    ContentFingerprint,
    fingerprint_file,
    fingerprint_file_with_spool,
    sha256_file,
    sha256_file_to_spool,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import (
    CancelledError,
    PlanningNoProgressError,
    PlanningWorkerError,
    PauseRequestedError,
    ParserDependencyError,
    PasswordProtectedError,
    UnsupportedFormatError,
    ZipMemberContentChangedError,
    ZipMemberDirectoryChangedError,
    ZipMemberEncryptedError,
    ZipMemberSizeChangedError,
)
from local_full_text_search.core.index_scheduler import estimate_parse_cost
from local_full_text_search.core.index_time_estimator import IndexTimeEstimator
from local_full_text_search.core.index_writer import IndexWriter
from local_full_text_search.core.planning_tasks import (
    FingerprintBatchResult,
    FingerprintSourceResult,
    PdfDocumentScanResult,
    PreparedZipMemberResult,
    PreparedFileMetadata,
    StatBatchResult,
    discover_file_batches,
    fingerprint_source_batch,
    prepare_zip_member_task,
    scan_zip_manifest_task,
    scan_pdf_document_task,
    stat_file_batch,
)
from local_full_text_search.core.planning_worker import (
    PlanningProgress,
    RecoverablePlanningRunner,
)
from local_full_text_search.core.pdf_task_graph import (
    PdfPagePlan,
    PdfTaskGraphRepository,
)
from local_full_text_search.core.ocr_scheduler import OcrRequestRepository
from local_full_text_search.core.run_control import (
    planning_pause_acknowledgements,
    request_process_pause,
    resume_processes,
)
from local_full_text_search.core.runtime_resource_controller import (
    ResourceDecision,
    ResourceSample,
    RuntimeResourceController,
)
from local_full_text_search.core.scanner import iter_files
from local_full_text_search.core.semantic_progress import (
    SemanticProgress,
    is_semantic_progress,
    progress_signature,
)
from local_full_text_search.core.task_manager import CancelToken, ProcessRunControlToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import FileTiming, IndexRunMetrics
from local_full_text_search.ocr.ocr_engine import ADAPTIVE_OCR_VERSION
from local_full_text_search.ocr.ocr_cache import ocr_models_fingerprint
from local_full_text_search.parsers.parser_registry import ParserRegistry
from local_full_text_search.parsers.legacy_office_parser import (
    cleanup_registered_office_processes,
    registered_office_processes_alive,
)
from local_full_text_search.parsers.pdf_parser import PDF_DYNAMIC_OCR_VERSION
from local_full_text_search.parsers.zip_parser import (
    ZipMemberDescriptor,
    decoded_zip_member_name,
    hash_zip_member,
    safe_zip_member_name,
    scan_zip_manifest,
)

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class PreparedSource:
    file_path: Path
    file_id: int
    size_bytes: int | None = None
    exact_sha256: str | None = None
    archive_path: Path | None = None
    archive_member_index: int | None = None
    archive_member_name: str = ""
    archive_member_crc32: int | None = None
    archive_internal_path: str = ""
    source_spool_path: Path | None = None
    image_width: int = 0
    image_height: int = 0


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
    memory_estimate_bytes: int = 0
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
    progress_cursor: str = ""
    progress_bytes_read: int = 0
    progress_output_blocks: int = 0
    progress_checkpoint_version: int = 0
    progress_worker_pid: int | None = None
    last_progress_monotonic: float = 0.0
    timeout_seconds: int = 0
    stall_signature: str = ""
    repeated_stall_count: int = 0
    archive_path: Path | None = None
    archive_member_index: int | None = None
    archive_member_name: str = ""
    archive_member_crc32: int | None = None
    archive_internal_path: str = ""
    exact_sha256: str = ""
    content_hash_full: str | None = None
    source_spool_path: Path | None = None
    checkpoint_path: Path | None = None
    validation_hang_stage: str = ""
    pdf_document_task_id: int | None = None
    pdf_page_number: int | None = None
    pdf_task_type: str = ""
    pdf_source_digest: str = ""
    batch_jobs: tuple[ParseJob, ...] = ()
    pdf_confirmation_batch_end: bool = True
    source_modified_time_ns: int = 0
    source_size_bytes: int = 0
    ocr_request_id: int | None = None
    ocr_request_owner: str = ""
    ocr_width: int = 0
    ocr_height: int = 0


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
    content_hash_full: str | None = None
    diagnostics: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class SpoolParseResult:
    file_id: int
    file_path: Path
    spool_path: Path
    worker_pid: int
    result_bytes: int
    checksum: str
    spool_write_ms: int = 0
    task_id: int | None = None


ParseResult = ParseOutcome | SpoolParseResult
ProcessParseResult = ParseResult | list[SpoolParseResult]


@dataclass(slots=True)
class ParseLane:
    name: str
    executor: Executor
    max_in_flight: int
    max_inflight_bytes: int
    process_based: bool = False
    persistent_process: bool = False
    worker_count: int = 1
    pending: deque[ParseJob] = field(default_factory=deque)
    futures: set[Future[ProcessParseResult]] = field(default_factory=set)
    jobs: dict[Future[ProcessParseResult], ParseJob] = field(default_factory=dict)
    inflight_bytes: int = 0


def _is_network_path(path: Path) -> bool:
    text = str(Path(path))
    if text.startswith(("\\\\", "//")):
        return True
    if os.name != "nt":
        return False
    drive = Path(path).drive
    if not drive:
        return False
    try:
        import ctypes

        return int(
            ctypes.windll.kernel32.GetDriveTypeW(  # type: ignore[attr-defined]
                drive + "\\"
            )
        ) == 4
    except Exception:
        return False


class ProcessResourceMonitor:
    """Sample process-tree CPU/RSS and host pressure outside scheduling locks."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        *,
        network_probe_path: Path | None = None,
    ) -> None:
        self.interval_seconds = max(0.2, float(interval_seconds))
        self.network_probe_path = (
            Path(network_probe_path)
            if network_probe_path is not None
            else None
        )
        self.peak_rss_bytes = 0
        self._sample_lock = threading.Lock()
        self._latest: dict[str, object] = {
            "timestamp": time.monotonic(),
            "total_cpu_percent": 0.0,
            "app_cpu_percent": 0.0,
            "memory_available_bytes": 0,
            "app_rss_bytes": 0,
            "disk_busy_percent": 0.0,
            "network_read_latency_ms": 0.0,
            "worker_rss_bytes": {},
        }
        self._last_disk_busy_ms: int | None = None
        self._last_disk_sample_monotonic: float | None = None
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
                sample = self._collect()
                self.peak_rss_bytes = max(
                    self.peak_rss_bytes,
                    int(sample["app_rss_bytes"]),
                )
                with self._sample_lock:
                    self._latest = sample
            except Exception:
                logger.debug("Resource monitor sample failed", exc_info=True)
            self._stop.wait(self.interval_seconds)

    def snapshot(
        self,
        *,
        queue_depth: int,
        active_tasks: int,
        ocr_pending_pixels: int,
        writer_queue_depth: int,
        paused: bool,
        completion_rate: float = 0.0,
        worker_failure_rate: float = 0.0,
    ) -> ResourceSample:
        with self._sample_lock:
            sample = dict(self._latest)
        return ResourceSample(
            timestamp=float(sample.get("timestamp") or time.monotonic()),
            total_cpu_percent=float(sample.get("total_cpu_percent") or 0.0),
            app_cpu_percent=float(sample.get("app_cpu_percent") or 0.0),
            memory_available_bytes=int(
                sample.get("memory_available_bytes") or 0
            ),
            app_rss_bytes=int(sample.get("app_rss_bytes") or 0),
            disk_busy_percent=float(sample.get("disk_busy_percent") or 0.0),
            network_read_latency_ms=float(
                sample.get("network_read_latency_ms") or 0.0
            ),
            queue_depth=max(0, int(queue_depth)),
            active_tasks=max(0, int(active_tasks)),
            ocr_pending_pixels=max(0, int(ocr_pending_pixels)),
            writer_queue_depth=max(0, int(writer_queue_depth)),
            completion_rate=max(0.0, float(completion_rate)),
            worker_failure_rate=max(0.0, float(worker_failure_rate)),
            paused=bool(paused),
            worker_rss_bytes={
                int(pid): max(0, int(rss))
                for pid, rss in dict(
                    sample.get("worker_rss_bytes") or {}
                ).items()
            },
        )

    def _collect(self) -> dict[str, object]:
        import psutil

        process = psutil.Process()
        processes = [process, *process.children(recursive=True)]
        worker_rss: dict[int, int] = {}
        app_cpu = 0.0
        app_rss = 0
        for child in processes:
            try:
                rss = int(child.memory_info().rss)
                cpu = float(child.cpu_percent(interval=None))
            except (psutil.Error, OSError):
                continue
            worker_rss[int(child.pid)] = rss
            app_rss += rss
            app_cpu += cpu
        memory = psutil.virtual_memory()
        disk_busy = 0.0
        try:
            counters = psutil.disk_io_counters()
            now = time.monotonic()
            current_busy_ms = int(
                getattr(counters, "busy_time", 0) or 0
            )
            if (
                self._last_disk_busy_ms is not None
                and self._last_disk_sample_monotonic is not None
            ):
                elapsed_ms = max(
                    1.0,
                    (now - self._last_disk_sample_monotonic) * 1000,
                )
                disk_busy = max(
                    0.0,
                    min(
                        100.0,
                        (current_busy_ms - self._last_disk_busy_ms)
                        / elapsed_ms
                        * 100.0,
                    ),
                )
            self._last_disk_busy_ms = current_busy_ms
            self._last_disk_sample_monotonic = now
        except Exception:
            pass
        return {
            "timestamp": time.monotonic(),
            "total_cpu_percent": float(psutil.cpu_percent(interval=None)),
            "app_cpu_percent": app_cpu,
            "memory_available_bytes": int(memory.available),
            "app_rss_bytes": app_rss,
            "disk_busy_percent": disk_busy,
            "network_read_latency_ms": self._probe_network_latency_ms(),
            "worker_rss_bytes": worker_rss,
        }

    def _probe_network_latency_ms(self) -> float:
        path = self.network_probe_path
        if path is None or not _is_network_path(path):
            return 0.0
        started = time.perf_counter()
        try:
            path.stat()
            with os.scandir(path) as entries:
                entry = next(entries, None)
                if entry is not None:
                    entry.stat(follow_symlinks=False)
        except OSError:
            return 10_000.0
        return max(0.0, (time.perf_counter() - started) * 1000.0)


class ProcessLaneWatchdog:
    """Terminate overdue process workers even if the scheduler is backpressured."""

    def __init__(
        self,
        lanes: dict[str, ParseLane],
        spool_dir: Path,
        settings: AppSettings,
        database: DatabaseManager | None = None,
        metrics: IndexRunMetrics | None = None,
        diagnostic_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.lanes = lanes
        self.spool_dir = spool_dir
        self.settings = settings
        self.database = database
        self.metrics = metrics
        self.diagnostic_callback = diagnostic_callback
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
            if self.metrics is not None:
                self.metrics.hang_metrics["watchdog_scan_count"] = (
                    int(
                        self.metrics.hang_metrics.get(
                            "watchdog_scan_count",
                            0,
                        )
                    )
                    + 1
                )
            now = time.perf_counter()
            for lane in list(self.lanes.values()):
                if not lane.process_based or not isinstance(lane.executor, ProcessPoolExecutor):
                    continue
                overdue = []
                for future, job in list(lane.jobs.items()):
                    if future.done() or not job.started_monotonic:
                        continue
                    for active_job in _submission_jobs(job):
                        semantic_progress = refresh_job_progress(
                            active_job,
                            self.spool_dir,
                            now,
                        )
                        if (
                            semantic_progress
                            and self.database is not None
                            and active_job.task_id is not None
                        ):
                            self.database.try_update_task_progress(
                                active_job.task_id,
                                phase=active_job.progress_phase,
                                completed=active_job.progress_completed,
                                total=active_job.progress_total,
                                unit_type="",
                                cursor=active_job.progress_cursor,
                                bytes_read=active_job.progress_bytes_read,
                                output_blocks=active_job.progress_output_blocks,
                                checkpoint_version=(
                                    active_job.progress_checkpoint_version
                                ),
                                worker_pid=active_job.progress_worker_pid,
                                checkpoint_path=(
                                    str(active_job.checkpoint_path)
                                    if active_job.checkpoint_path is not None
                                    else None
                                ),
                            )
                            child_progress = pdf_child_progress(active_job)
                            if child_progress is not None:
                                (
                                    child_task_type,
                                    child_unit_key,
                                    child_status,
                                ) = child_progress
                                self.database.try_record_child_task_progress(
                                    active_job.task_id,
                                    task_type=child_task_type,
                                    unit_key=child_unit_key,
                                    status=child_status,
                                    phase=active_job.progress_phase,
                                    completed=active_job.progress_completed,
                                    total=active_job.progress_total,
                                    worker_pid=active_job.progress_worker_pid,
                                )
                        timeout_seconds = no_progress_timeout(
                            self.settings,
                            active_job,
                        )
                        last_progress = (
                            active_job.last_progress_monotonic
                            or active_job.started_monotonic
                        )
                        if now - last_progress >= timeout_seconds:
                            if (
                                not active_job.watchdog_timed_out
                                and self.metrics is not None
                            ):
                                self.metrics.hang_metrics[
                                    "no_progress_timeout_count"
                                ] = (
                                    int(
                                        self.metrics.hang_metrics.get(
                                            "no_progress_timeout_count",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                self.metrics.hang_metrics[
                                    "parser_worker_timeout_count"
                                ] = (
                                    int(
                                        self.metrics.hang_metrics.get(
                                            "parser_worker_timeout_count",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                            active_job.watchdog_timed_out = True
                            active_job.timeout_seconds = timeout_seconds
                            job.watchdog_timed_out = True
                            overdue.append(active_job)
                if not overdue:
                    continue
                executor = lane.executor
                if executor in self._terminated:
                    continue
                self._terminated.add(executor)
                representative = max(
                    overdue,
                    key=lambda item: now
                    - (
                        item.last_progress_monotonic
                        or item.started_monotonic
                        or now
                    ),
                )
                diagnostic = {
                    "source": "process_lane_watchdog",
                    "lane": lane.name,
                    "file": str(representative.file_path),
                    "phase": representative.progress_phase,
                    "cursor": representative.progress_cursor,
                    "retry_count": int(representative.retry_count),
                    "no_progress_seconds": int(
                        max(
                            0.0,
                            now
                            - (
                                representative.last_progress_monotonic
                                or representative.started_monotonic
                                or now
                            ),
                        )
                    ),
                    "timeout_seconds": int(representative.timeout_seconds),
                    "reason": (
                        f"{representative.progress_phase or '解析'} 阶段停在"
                        f" {representative.progress_cursor or '未报告游标'}"
                    ),
                }
                self._emit_diagnostic(
                    {**diagnostic, "state": "reclaiming_no_progress"}
                )
                self._emit_diagnostic(
                    {**diagnostic, "state": "terminating_worker"}
                )
                if self.metrics is not None:
                    self.metrics.hang_metrics[
                        "worker_process_kill_count"
                    ] = (
                        int(
                            self.metrics.hang_metrics.get(
                                "worker_process_kill_count",
                                0,
                            )
                        )
                        + 1
                    )
                logger.warning(
                    "Process lane %s made no progress for the configured interval with %s overdue task(s); terminating worker pool",
                    lane.name,
                    len(overdue),
                )
                terminate_process_pool_workers(executor, self.spool_dir)

    def _emit_diagnostic(self, payload: dict[str, object]) -> None:
        if self.diagnostic_callback is None:
            return
        try:
            self.diagnostic_callback(dict(payload))
        except Exception:
            logger.debug(
                "Unable to emit process-watchdog diagnostic state",
                exc_info=True,
            )


_worker_state = threading.local()
_process_registry: ParserRegistry | None = None


def _pool_health_probe() -> bool:
    return True


class IndexManager:
    def __init__(
        self,
        db: DatabaseManager,
        settings: AppSettings,
        *,
        run_context: dict[str, object] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.run_context = dict(run_context or {})
        self._executor_lock = threading.Lock()
        self._active_process_executors: set[ProcessPoolExecutor] = set()
        self._active_process_registry_dirs: set[Path] = set()
        self._pause_lock = threading.Lock()
        self._pause_lanes: dict[str, ParseLane] | None = None
        self._pause_writer: IndexWriter | None = None
        self._pause_executors: list[Executor] | None = None
        self._pause_process_executors: list[ProcessPoolExecutor] | None = None
        self._pause_spool_dir: Path | None = None
        self._pause_estimator: IndexTimeEstimator | None = None
        self._current_metrics: IndexRunMetrics | None = None
        self._planning_runner: RecoverablePlanningRunner | None = None
        self._planning_control_dir: Path | None = None
        self._planning_metadata: dict[str, PreparedFileMetadata] = {}
        self._pdf_document_jobs: dict[int, ParseJob] = {}
        self._pdf_failed_documents: set[int] = set()
        self._pdf_confirmation_buffers: dict[
            int, list[tuple[int, Path, str]]
        ] = {}
        self._pause_requested_monotonic: float | None = None
        self._pause_acknowledgements: dict[int, dict[str, object]] = {}
        self._pause_required_acknowledgements: set[int] = set()
        self._pause_state = "running"
        self._pause_resource_observation: dict[str, object] | None = None
        self._latest_resource_sample: ResourceSample | None = None
        self._resource_resize_cooldown_seconds = 0.0
        self._diagnostic_lock = threading.Lock()
        self._diagnostic_events: deque[dict[str, object]] = deque(maxlen=128)
        self._diagnostic_sequence = 0

    def request_pause(self) -> None:
        requested_at = time.perf_counter()
        with self._pause_lock:
            self._pause_requested_monotonic = requested_at
            self._pause_acknowledgements = {}
            self._pause_required_acknowledgements = {
                _pause_job_key(active_job)
                for lane in (self._pause_lanes or {}).values()
                for future, job in lane.jobs.items()
                if not future.done()
                for active_job in _submission_jobs(job)
            }
            self._pause_state = "pause_requested"
            if self._current_metrics is not None:
                self._current_metrics.pause_metrics[
                    "pause_request_count"
                ] = (
                    int(
                        self._current_metrics.pause_metrics.get(
                            "pause_request_count",
                            0,
                        )
                    )
                    + 1
                )
                self._current_metrics.pause_metrics[
                    "pause_request_to_submission_stop_ms"
                ] = 0
                _record_eta_replay_control(
                    self._current_metrics,
                    "pause",
                    self._pause_lanes or {},
                )
        with self._executor_lock:
            registry_dirs = list(self._active_process_registry_dirs)
        if self._planning_control_dir is not None:
            registry_dirs.append(self._planning_control_dir)
        for registry_dir in registry_dirs:
            request_process_pause(registry_dir)
        with self._pause_lock:
            if self._pause_estimator is not None:
                self._pause_estimator.pause()

    def request_resume(self) -> None:
        self._finish_pause_resource_observation()
        with self._executor_lock:
            registry_dirs = list(self._active_process_registry_dirs)
        if self._planning_control_dir is not None:
            registry_dirs.append(self._planning_control_dir)
        for registry_dir in registry_dirs:
            resume_processes(registry_dir)
        with self._pause_lock:
            self._pause_state = "resuming"
            if self._pause_estimator is not None:
                self._pause_estimator.resume()
            if self._current_metrics is not None:
                self._current_metrics.pause_metrics["resume_count"] = (
                    int(
                        self._current_metrics.pause_metrics.get(
                            "resume_count",
                            0,
                        )
                    )
                    + 1
                )
                _record_eta_replay_control(
                    self._current_metrics,
                    "resume",
                    self._pause_lanes or {},
                )
            self._pause_required_acknowledgements = set()
            self._pause_state = "running"

    def is_safely_paused(self) -> bool:
        with self._pause_lock:
            lanes = self._pause_lanes
            writer = self._pause_writer
            if lanes is None:
                return True
            active = any(
                not future.done()
                for lane in lanes.values()
                for future in lane.futures
            )
            spool_dir = self._pause_spool_dir
            planning_pids = set(
                self._planning_runner.active_pids
                if self._planning_runner is not None
                else ()
            )
            planning_control_dir = self._planning_control_dir
            acknowledgements_complete = (
                self._pause_required_acknowledgements.issubset(
                    self._pause_acknowledgements
                )
            )
        planning_acknowledgements = (
            planning_pause_acknowledgements(planning_control_dir)
            if planning_control_dir is not None and planning_pids
            else {}
        )
        planning_acknowledgements_complete = planning_pids.issubset(
            planning_acknowledgements
        )
        office_active = bool(
            spool_dir is not None
            and registered_office_processes_alive(spool_dir)
        )
        safe = (
            not active
            and (writer is None or writer.is_idle())
            and planning_acknowledgements_complete
            and not office_active
            and acknowledgements_complete
        )
        became_paused = False
        if safe:
            with self._pause_lock:
                if self._pause_state in {
                    "pause_requested",
                    "pausing",
                }:
                    self._pause_state = "paused"
                    became_paused = True
                    if (
                        self._current_metrics is not None
                        and self._pause_requested_monotonic is not None
                    ):
                        self._current_metrics.pause_metrics[
                            "safe_pause_latency_seconds"
                        ] = round(
                            time.perf_counter()
                            - self._pause_requested_monotonic,
                            3,
                        )
        if became_paused:
            self._begin_pause_resource_observation()
        return safe

    def pause_status(self) -> dict[str, object]:
        safe = self.is_safely_paused()
        with self._pause_lock:
            planning_pids = set(
                self._planning_runner.active_pids
                if self._planning_runner is not None
                else ()
            )
            planning_control_dir = self._planning_control_dir
            parse_required = len(self._pause_required_acknowledgements)
            parse_received = len(
                self._pause_required_acknowledgements.intersection(
                    self._pause_acknowledgements
                )
            )
            parse_acknowledgements = list(
                self._pause_acknowledgements.values()
            )
        planning_acknowledgements_by_pid = (
            planning_pause_acknowledgements(planning_control_dir)
            if planning_control_dir is not None and planning_pids
            else {}
        )
        planning_acknowledgements_list = [
            planning_acknowledgements_by_pid[pid]
            for pid in sorted(
                planning_pids.intersection(
                    planning_acknowledgements_by_pid
                )
            )
        ]
        planning_required = len(planning_pids)
        planning_received = len(planning_acknowledgements_list)
        with self._pause_lock:
            return {
                "state": self._pause_state,
                "safe": safe,
                "acknowledgements": (
                    parse_acknowledgements
                    + planning_acknowledgements_list
                ),
                "required_acknowledgements": (
                    parse_required + planning_required
                ),
                "received_acknowledgements": (
                    parse_received + planning_received
                ),
                "planning_acknowledgements": (
                    planning_acknowledgements_list
                ),
                "planning_required_acknowledgements": planning_required,
                "planning_received_acknowledgements": planning_received,
            }

    def _record_pause_acknowledgement(
        self,
        job: ParseJob,
        outcome: ParseOutcome,
        *,
        completed_during_pause: bool = False,
    ) -> None:
        checkpoint_path = job.checkpoint_path
        checkpoint_checksum = ""
        if checkpoint_path is not None and checkpoint_path.is_file():
            try:
                checkpoint_checksum = sha256_path(checkpoint_path)
            except OSError:
                checkpoint_checksum = ""
        acknowledgement_cursor = int(
            job.pdf_page_number
            if job.pdf_task_type and job.pdf_page_number is not None
            else outcome.resume_cursor or job.resume_cursor
        )
        acknowledgement = {
            "task_id": int(job.task_id or 0),
            "worker_pid": int(outcome.worker_pid or 0),
            "safe_unit_type": str(
                job.pdf_task_type
                or outcome.progress_phase
                or job.progress_phase
                or job.parser_name
            ),
            "cursor": acknowledgement_cursor,
            "completed_units": int(
                outcome.progress_completed
                or job.progress_completed
                or 0
            ),
            "total_units": int(
                outcome.progress_total
                or job.progress_total
                or 0
            ),
            "checkpoint_version": int(
                job.progress_checkpoint_version
                or outcome.resume_cursor
                or 0
            ),
            "checkpoint_checksum": checkpoint_checksum,
            "returned_at_epoch": time.time(),
            "holds_external_process": False,
            "completed_during_pause": bool(completed_during_pause),
        }
        with self._pause_lock:
            job_key = _pause_job_key(job)
            if completed_during_pause and (
                self._pause_state not in {"pause_requested", "pausing"}
                or job_key not in self._pause_required_acknowledgements
            ):
                return
            self._pause_state = "pausing"
            self._pause_acknowledgements[job_key] = acknowledgement

    def _record_completed_pause_requirement(
        self,
        job: ParseJob,
        outcome: ParseOutcome,
    ) -> None:
        """Treat a task that finishes after the pause request as safely idle."""

        self._record_pause_acknowledgement(
            job,
            outcome,
            completed_during_pause=True,
        )

    def _begin_pause_resource_observation(
        self,
        *,
        reset: bool = False,
    ) -> None:
        with self._pause_lock:
            if (
                self._pause_resource_observation is not None
                and not reset
            ):
                return
        processes: list[object] = []
        read_bytes = 0
        try:
            import psutil

            parent = psutil.Process()
            processes = [
                parent,
                *parent.children(recursive=True),
            ]
            for process in processes:
                try:
                    process.cpu_percent(interval=None)
                    read_bytes += int(
                        process.io_counters().read_bytes
                    )
                except (psutil.Error, OSError, AttributeError):
                    continue
        except Exception:
            processes = []
        observation = {
            "started": time.perf_counter(),
            "processes": processes,
            "read_bytes": read_bytes,
            "database_signature": (
                self._pause_database_signature()
            ),
        }
        with self._pause_lock:
            self._pause_resource_observation = observation

    def _finish_pause_resource_observation(self) -> None:
        with self._pause_lock:
            observation = self._pause_resource_observation
            self._pause_resource_observation = None
            metrics = self._current_metrics
        if observation is None or metrics is None:
            return
        elapsed = max(
            0.0,
            time.perf_counter()
            - float(observation.get("started") or 0.0),
        )
        read_bytes = 0
        cpu_percent = 0.0
        try:
            import psutil

            for process in list(
                observation.get("processes") or []
            ):
                try:
                    read_bytes += int(
                        process.io_counters().read_bytes
                    )
                    cpu_percent += float(
                        process.cpu_percent(interval=None)
                    )
                except (psutil.Error, OSError, AttributeError):
                    continue
        except Exception:
            pass
        before_signature = observation.get(
            "database_signature"
        )
        after_signature = self._pause_database_signature()
        before_progress = (
            int(before_signature[0])
            if isinstance(before_signature, tuple)
            and before_signature
            else 0
        )
        after_progress = (
            int(after_signature[0])
            if isinstance(after_signature, tuple)
            and after_signature
            else 0
        )
        pause_metrics = metrics.pause_metrics
        pause_metrics["paused_observation_seconds"] = round(
            elapsed,
            3,
        )
        pause_metrics[
            "paused_observation_progress_delta"
        ] = max(0, after_progress - before_progress)
        pause_metrics["paused_read_bytes_delta"] = max(
            0,
            read_bytes
            - int(observation.get("read_bytes") or 0),
        )
        pause_metrics["paused_database_write_count"] = int(
            after_signature != before_signature
        )
        pause_metrics["paused_cpu_average"] = round(
            cpu_percent / max(1, int(os.cpu_count() or 1)),
            6,
        )

    def _pause_database_signature(self) -> tuple[object, ...]:
        try:
            with self.db.connect(timeout_seconds=0.1) as connection:
                progress = int(
                    connection.execute(
                        """
                        SELECT COALESCE(
                            SUM(progress_completed),
                            0
                        )
                        FROM parse_tasks
                        """
                    ).fetchone()[0]
                )
                blocks = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM content_blocks"
                    ).fetchone()[0]
                )
                states = tuple(
                    tuple(row)
                    for row in connection.execute(
                        """
                        SELECT status, COUNT(*)
                        FROM parse_tasks
                        GROUP BY status
                        ORDER BY status
                        """
                    )
                )
            return progress, blocks, states
        except Exception:
            return ()

    def apply_settings_while_paused(
        self,
        settings: AppSettings,
        *,
        execution_mode: str,
        effective_profile: dict[str, object],
    ) -> bool:
        """Rebuild idle parser pools while a run is at a confirmed safe pause."""

        if not self.is_safely_paused():
            return False
        with self._pause_lock:
            lanes = self._pause_lanes
            executors = self._pause_executors
            process_executors = self._pause_process_executors
            spool_dir = self._pause_spool_dir
            estimator = self._pause_estimator
            writer = self._pause_writer
            metrics = self._current_metrics
            if (
                lanes is None
                or executors is None
                or process_executors is None
                or spool_dir is None
            ):
                return False
            if any(not future.done() for lane in lanes.values() for future in lane.futures):
                return False
            previous_mode = str(self.run_context.get("execution_mode") or "normal")
            old_executors = list(executors)
            old_process_executors = list(process_executors)
            previous_settings = self.settings
            self.settings = settings
            try:
                replacement_lanes, replacement_executors, replacement_processes = (
                    self._create_lanes([], spool_dir)
                )
                probes = [
                    executor.submit(_pool_health_probe)
                    for executor in dict.fromkeys(replacement_executors)
                ]
                if not all(
                    bool(probe.result(timeout=30))
                    for probe in probes
                ):
                    raise RuntimeError("候选解析进程池健康检查失败")
            except Exception as exc:
                self.settings = previous_settings
                for executor in locals().get(
                    "replacement_executors",
                    [],
                ):
                    try:
                        executor.shutdown(
                            wait=False,
                            cancel_futures=True,
                        )
                    except Exception:
                        pass
                if metrics is not None:
                    metrics.pause_metrics[
                        "mode_switch_failure_count"
                    ] = (
                        int(
                            metrics.pause_metrics.get(
                                "mode_switch_failure_count",
                                0,
                            )
                        )
                        + 1
                    )
                    metrics.pause_metrics[
                        "mode_switch_rollback_count"
                    ] = (
                        int(
                            metrics.pause_metrics.get(
                                "mode_switch_rollback_count",
                                0,
                            )
                        )
                        + 1
                    )
                    metrics.profile_transitions.append(
                        {
                            "from": previous_mode,
                            "to": execution_mode,
                            "status": "rolled_back",
                            "error": str(exc),
                            "applied_at_epoch": time.time(),
                        }
                    )
                return False
            for lane_name, lane in lanes.items():
                replacement = replacement_lanes[lane_name]
                lane.executor = replacement.executor
                lane.max_in_flight = replacement.max_in_flight
                lane.max_inflight_bytes = replacement.max_inflight_bytes
                lane.process_based = replacement.process_based
                lane.persistent_process = replacement.persistent_process
                lane.worker_count = replacement.worker_count
            executors[:] = replacement_executors
            process_executors[:] = replacement_processes
            self.run_context["execution_mode"] = execution_mode
            self.run_context["effective_profile"] = dict(effective_profile)
            if estimator is not None:
                estimator.reset_for_context_change(
                    {
                        name: lane.worker_count
                        for name, lane in lanes.items()
                    }
                )
            if writer is not None:
                writer.batch_blocks = max(1, int(settings.db_write_batch_blocks))
                writer.batch_bytes = max(1024, int(settings.db_write_batch_bytes))
                writer.max_delay = max(0.01, int(settings.db_write_max_delay_ms) / 1000)
            with self._executor_lock:
                self._active_process_executors.difference_update(old_process_executors)
                self._active_process_executors.update(replacement_processes)
            if metrics is not None:
                metrics.pause_metrics["mode_switch_count"] = (
                    int(
                        metrics.pause_metrics.get(
                            "mode_switch_count",
                            0,
                        )
                    )
                    + 1
                )
                metrics.execution_mode = execution_mode
                metrics.effective_profile = dict(effective_profile)
                metrics.lane_worker_limits = {
                    name: lane.worker_count for name, lane in lanes.items()
                }
                metrics.profile_transitions.append(
                    {
                        "from": previous_mode,
                        "to": execution_mode,
                        "applied_at_epoch": time.time(),
                        "checkpointed_tasks": sum(
                            len(lane.futures) for lane in lanes.values()
                        ),
                        "pending_tasks": sum(
                            len(lane.pending) for lane in lanes.values()
                        ),
                    }
                )
                _record_eta_replay_control(
                    metrics,
                    "mode_switch",
                    lanes,
                    mode=execution_mode,
                )
        for executor in old_executors:
            try:
                executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                logger.debug("Unable to close parser pool during mode switch", exc_info=True)
        # Rebuilding parser pools is an authorized mode-switch operation, not
        # evidence that a safely paused run is still consuming resources.
        # Start a fresh observation window only after the old pools are gone.
        self._begin_pause_resource_observation(reset=True)
        return True

    def force_terminate_processes(self) -> None:
        self.db.interrupt_active_connections()
        planning_runner = self._planning_runner
        if planning_runner is not None:
            planning_runner.cancel_active()
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
        metrics = IndexRunMetrics(
            run_id=run_id,
            mode="full_batch" if full_batch else "incremental",
            execution_mode=str(self.run_context.get("execution_mode") or "normal"),
            hardware=dict(self.run_context.get("hardware") or {}),
            root_disk_classes={
                str(key): str(value)
                for key, value in dict(self.run_context.get("root_disk_classes") or {}).items()
            },
            effective_profile=dict(self.run_context.get("effective_profile") or {}),
        )
        metrics.resource_metrics["rss_budget_bytes"] = (
            max(
                128,
                int(self.settings.index_memory_budget_mb),
            )
            * 1024
            * 1024
        )
        metrics.eta_metrics["_run_started_monotonic"] = run_started
        with self._pause_lock:
            self._current_metrics = metrics
        summary = IndexSummary()
        self.db.start_index_run(metrics)
        self.db.update_root_scan_time(root_id, "indexing")
        root_path = Path(str(root["path"]))
        include_subfolders = bool(root["include_subfolders"])
        previous_paths = self.db.active_paths_for_root(root_id)
        spool_dir = TEMP_DIR / "process_results" / run_id
        planning_control_dir = (
            spool_dir / "planning_run_control"
        )
        planning_runner = RecoverablePlanningRunner(
            spool_dir / "planning_tasks",
            no_progress_timeout_seconds=max(
                1,
                int(self.settings.planning_no_progress_timeout_seconds or 300),
            ),
            startup_timeout_seconds=max(
                1,
                int(self.settings.planning_startup_timeout_seconds or 30),
            ),
            pause_control_dir=planning_control_dir,
        )
        self._planning_runner = planning_runner
        self._planning_control_dir = planning_control_dir
        self._planning_metadata = {}
        self._pdf_document_jobs = {}
        self._pdf_failed_documents = set()
        self._pdf_confirmation_buffers = {}
        lanes: dict[str, ParseLane] = {}
        writer: IndexWriter | None = None
        resource_monitor: ProcessResourceMonitor | None = None
        resource_controller: RuntimeResourceController | None = None
        watchdog: ProcessLaneWatchdog | None = None
        executors: list[Executor] = []
        process_executors: list[ProcessPoolExecutor] = []
        file_timings: list[FileTiming] = []
        worker_pids: set[int] = set()
        ocr_model_loads_by_pid: dict[int, int] = {}
        ocr_model_load_ms_by_pid: dict[int, int] = {}
        ocr_runtime_metrics_by_pid: dict[int, dict[str, float]] = {}
        run_status = "failed"
        defer_fts = bool(self.settings.defer_fts_during_full_scan and full_batch)
        candidate_version_id: int | None = None
        fts_published = True

        try:
            recovered = self._recover_spooled_tasks(root_id)
            if recovered:
                summary.indexed += recovered

            lanes, executors, process_executors = self._create_lanes([], spool_dir)
            with self._pause_lock:
                self._pause_lanes = lanes
                self._pause_executors = executors
                self._pause_process_executors = process_executors
                self._pause_spool_dir = spool_dir
            metrics.lane_worker_limits = {
                name: lane.worker_count for name, lane in lanes.items()
            }
            lane_submission_limits = {
                name: lane.max_in_flight for name, lane in lanes.items()
            }
            resource_controller = RuntimeResourceController(
                memory_budget_bytes=max(
                    128,
                    int(self.settings.index_memory_budget_mb),
                )
                * 1024
                * 1024,
                ocr_hard_max=min(
                    2,
                    max(1, int(self.settings.ocr_workers or 1)),
                ),
                initial_ocr_inflight=lanes["ocr"].max_in_flight,
                consecutive_samples=3,
                min_state_seconds=5,
                resize_cooldown_seconds=15,
            )
            self._resource_resize_cooldown_seconds = float(
                resource_controller.resize_cooldown_seconds
            )
            estimator = IndexTimeEstimator(
                {},
                {name: lane.worker_count for name, lane in lanes.items()},
            )
            with self._pause_lock:
                self._pause_estimator = estimator
            with self._executor_lock:
                self._active_process_executors.update(process_executors)
                self._active_process_registry_dirs.add(spool_dir)

            resource_monitor = ProcessResourceMonitor(
                network_probe_path=Path(str(root["path"]))
            )
            resource_monitor.start()
            watchdog = ProcessLaneWatchdog(
                lanes,
                spool_dir,
                self.settings,
                self.db,
                metrics,
                diagnostic_callback=self._record_scheduler_diagnostic,
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
            with self._pause_lock:
                self._pause_writer = writer

            all_jobs: list[ParseJob] = []
            active_jobs: dict[tuple[str, str, str], ParseJob] = {}
            seen_paths: set[str] = set()
            eligible_total = 0
            excluded_video_total = 0
            planning_batch: list[Path] = []
            planning_batch_size = max(
                64,
                min(512, int(self.settings.index_write_batch_size or 32) * 4),
            )
            scan_and_plan_started = time.perf_counter()
            fingerprint_elapsed = 0.0
            deferred_fts_started = False
            last_prepare_progress_emit = 0.0

            def report_prepare_progress(
                progress: PlanningProgress,
            ) -> None:
                nonlocal last_prepare_progress_emit
                now = time.monotonic()
                if (
                    now - last_prepare_progress_emit < 0.2
                    and progress.completed > 1
                ):
                    return
                self._emit(
                    progress_callback,
                    "planning",
                    summary,
                    current_file=progress.detail,
                    pending=pending_lane_tasks(lanes.values()),
                    total_files=eligible_total,
                    discovered_files=metrics.discovered_files,
                    excluded_video=excluded_video_total,
                    total_bytes=metrics.discovered_bytes,
                    planning_phase=progress.phase,
                    planning_cursor=progress.cursor,
                    planning_completed=progress.completed,
                    planning_total=progress.total,
                    planning_bytes_read=progress.bytes_read,
                    planning_worker_pid=progress.worker_pid,
                    phase_label=_planning_phase_label(progress.phase),
                )
                last_prepare_progress_emit = now

            def enqueue_planning_batch(paths: list[Path]) -> None:
                nonlocal fingerprint_elapsed, deferred_fts_started
                nonlocal candidate_version_id, fts_published
                if not paths:
                    return
                fingerprint_started = time.perf_counter()
                batch_jobs = self._prepare_jobs(
                    root_id,
                    paths,
                    run_id,
                    summary,
                    metrics,
                    token,
                    active_jobs,
                    report_prepare_progress,
                )
                fingerprint_elapsed += time.perf_counter() - fingerprint_started
                if not batch_jobs:
                    return
                if defer_fts and not deferred_fts_started:
                    self.db.begin_deferred_fts()
                    candidate_version_id = IndexVersionPublisher(
                        self.db
                    ).begin_candidate(
                        root_id=root_id,
                        run_id=run_id,
                        version_key=f"root:{root_id}:run:{run_id}",
                    )
                    deferred_fts_started = True
                    fts_published = False
                all_jobs.extend(batch_jobs)
                for job in batch_jobs:
                    lane = lanes[job.lane]
                    lane.pending.append(job)
                    estimator.total_cost_by_lane[job.lane] = (
                        estimator.total_cost_by_lane.get(job.lane, 0.0)
                        + max(0.0, job.estimated_cost)
                    )
                    metrics.lane_input_bytes[job.lane] = (
                        metrics.lane_input_bytes.get(job.lane, 0)
                        + max(0, job.size_bytes)
                    )
                    if job.archive_path is not None:
                        lane_key = f"member_lane_{job.lane}_count"
                        metrics.zip_metrics[lane_key] = (
                            metrics.zip_metrics.get(lane_key, 0) + 1
                        )
                submitted = schedule_parse_lanes(
                    lanes.values(),
                    self.settings,
                    token,
                    spool_dir,
                    metrics=metrics,
                )
                task_ids = [task_id for task_id in submitted if task_id is not None]
                self._mark_submitted_tasks_running(task_ids)
                self._claim_submitted_ocr_requests(
                    all_jobs,
                    task_ids,
                    run_id,
                )

            for file_path in self._iter_discovered_files(
                root_path,
                include_subfolders,
                token,
                summary,
                progress_callback,
            ):
                seen_paths.add(str(file_path))
                metrics.discovered_files += 1
                cached_metadata = self._planning_metadata.get(str(file_path))
                if cached_metadata is not None:
                    metrics.discovered_bytes += cached_metadata.size_bytes
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    excluded_video_total += 1
                else:
                    eligible_total += 1
                planning_batch.append(file_path)
                if len(planning_batch) >= planning_batch_size:
                    self._emit(
                        progress_callback,
                        "planning",
                        summary,
                        current_file=str(file_path),
                        pending=pending_lane_tasks(lanes.values()),
                        total_files=eligible_total,
                        discovered_files=metrics.discovered_files,
                        excluded_video=excluded_video_total,
                        total_bytes=metrics.discovered_bytes,
                        phase_label="正在流水分析并解析已发现文件",
                    )
                    enqueue_planning_batch(planning_batch)
                    planning_batch = []
            enqueue_planning_batch(planning_batch)
            metrics.fingerprint_ms = int(fingerprint_elapsed * 1000)
            metrics.scan_ms = max(
                0,
                int(
                    (
                        time.perf_counter()
                        - scan_and_plan_started
                        - fingerprint_elapsed
                    )
                    * 1000
                ),
            )
            missing_paths = previous_paths - seen_paths
            summary.deleted += self.db.mark_deleted_paths(missing_paths)

            last_heartbeat = 0.0
            last_resource_control = 0.0
            last_resource_completed = 0
            pending_ocr_worker_target: int | None = None
            pending_ocr_resize_reason = ""
            while pending_lane_tasks(lanes.values()):
                recycled = self._recycle_unhealthy_process_lanes(
                    lanes,
                    executors,
                    process_executors,
                    spool_dir,
                )
                self._emit_pending_scheduler_diagnostics(
                    progress_callback,
                    summary,
                    lanes,
                    estimator,
                    eligible_total,
                )
                if (
                    pending_ocr_worker_target is not None
                    and not token.paused
                ):
                    resized, resize_state = (
                        self._resize_idle_ocr_process_lane(
                            lanes,
                            executors,
                            process_executors,
                            spool_dir,
                            target_workers=(
                                pending_ocr_worker_target
                            ),
                            reason=pending_ocr_resize_reason,
                        )
                    )
                    if resized:
                        pending_ocr_worker_target = None
                        pending_ocr_resize_reason = ""
                    elif resize_state != "active_tasks":
                        pending_ocr_worker_target = None
                        pending_ocr_resize_reason = ""
                submitted = schedule_parse_lanes(
                    lanes.values(),
                    self.settings,
                    token,
                    spool_dir,
                    metrics=metrics,
                )
                if submitted:
                    task_ids = [task_id for task_id in submitted if task_id is not None]
                    self._mark_submitted_tasks_running(task_ids)
                    self._claim_submitted_ocr_requests(
                        all_jobs,
                        task_ids,
                        run_id,
                    )
                now = time.perf_counter()
                if (
                    resource_monitor is not None
                    and resource_controller is not None
                    and now - last_resource_control >= 1.0
                ):
                    completed_now = (
                        summary.indexed
                        + summary.failed
                        + summary.partial_success
                        + summary.metadata_only
                    )
                    elapsed_since_sample = max(
                        1.0,
                        now - (last_resource_control or now - 1.0),
                    )
                    sample = resource_monitor.snapshot(
                        queue_depth=sum(
                            len(lane.pending) for lane in lanes.values()
                        ),
                        active_tasks=sum(
                            len(lane.futures) for lane in lanes.values()
                        ),
                        ocr_pending_pixels=sum(
                            max(
                                1,
                                int(job.memory_estimate_bytes or 0) // 3,
                            )
                            for job in lanes["ocr"].pending
                        ),
                        writer_queue_depth=(
                            writer.queue_depth() if writer is not None else 0
                        ),
                        paused=token.paused,
                        completion_rate=max(
                            0.0,
                            (completed_now - last_resource_completed)
                            / elapsed_since_sample,
                        ),
                        worker_failure_rate=(
                            sum(
                                1
                                for job in all_jobs
                                if job.watchdog_timed_out
                            )
                            / max(1, len(all_jobs))
                        ),
                    )
                    self._latest_resource_sample = sample
                    decision = resource_controller.observe(sample)
                    if decision.target_ocr_inflight is not None:
                        pending_ocr_worker_target = int(
                            decision.target_ocr_inflight
                        )
                        pending_ocr_resize_reason = decision.reason
                    _apply_resource_decision(
                        lanes,
                        lane_submission_limits,
                        decision,
                        sample,
                        metrics,
                    )
                    last_resource_control = now
                    last_resource_completed = completed_now
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
                    if outcome.status == "paused":
                        if job.pdf_document_task_id:
                            self._flush_pdf_page_confirmations(
                                int(job.pdf_document_task_id)
                            )
                        self._update_ocr_request_for_outcome(
                            job,
                            outcome,
                            requeue=True,
                        )
                        self._record_pause_acknowledgement(job, outcome)
                        checkpoint = load_partial_parse_checkpoint(job, spool_dir, consume=False)
                        if checkpoint is not None and checkpoint.resume_cursor > 0:
                            job.resume_cursor = checkpoint.resume_cursor
                        elif outcome.resume_cursor > 0:
                            job.resume_cursor = outcome.resume_cursor
                        reset_job_for_retry(job, time.perf_counter())
                        lanes[lane_name].pending.appendleft(job)
                        continue
                    self._record_completed_pause_requirement(
                        job,
                        outcome,
                    )
                    if (
                        outcome.error_code in {"PROCESS_WORKER_CRASH", "PARSE_NO_PROGRESS"}
                        and job.retry_count < max(0, int(self.settings.no_progress_max_retries))
                        and (
                            outcome.error_code != "PARSE_NO_PROGRESS"
                            or register_stall(job) < 2
                        )
                    ):
                        if job.pdf_document_task_id:
                            self._flush_pdf_page_confirmations(
                                int(job.pdf_document_task_id)
                            )
                        if job.task_id is not None:
                            self.db.mark_task_attempt_interrupted(
                                int(job.task_id),
                                str(outcome.error_code),
                                (
                                    outcome.error_message
                                    or "解析 worker 异常退出，任务将重新排队"
                                ),
                            )
                        self._update_ocr_request_for_outcome(
                            job,
                            outcome,
                            requeue=True,
                        )
                        job.retry_count += 1
                        checkpoint = load_partial_parse_checkpoint(job, spool_dir, consume=False)
                        if (
                            checkpoint is not None
                            and checkpoint.resume_cursor > 0
                            and job.parser_name in {
                                "pdf",
                                "zip",
                                "xlsx_stream",
                                "image_ocr",
                            }
                        ):
                            job.resume_cursor = checkpoint.resume_cursor
                        reset_job_for_retry(job, time.perf_counter())
                        lanes[lane_name].pending.appendleft(job)
                        continue
                    self._update_ocr_request_for_outcome(
                        job,
                        outcome,
                    )
                    if job.pdf_task_type:
                        page_job = job
                        merged = self._record_pdf_page_outcome(
                            page_job,
                            outcome,
                        )
                        metrics.pdf_metrics["page_task_completed_count"] = (
                            metrics.pdf_metrics.get(
                                "page_task_completed_count",
                                0,
                            )
                            + 1
                        )
                        if merged is None:
                            if outcome.status not in {
                                "success",
                                "partial_success",
                                "metadata_only",
                            }:
                                parent_id = int(
                                    page_job.pdf_document_task_id or 0
                                )
                                if parent_id not in self._pdf_failed_documents:
                                    self._pdf_failed_documents.add(parent_id)
                                    parent = self._pdf_document_jobs.get(parent_id)
                                    record_parse_outcome(
                                        summary,
                                        outcome.status,
                                        (
                                            1 + len(parent.alias_file_ids)
                                            if parent is not None
                                            else 1
                                        ),
                                        extension=".pdf",
                                    )
                            continue
                        outcome, job = merged
                        metrics.pdf_metrics["document_merge_count"] = (
                            metrics.pdf_metrics.get("document_merge_count", 0)
                            + 1
                        )
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
                    metrics.spool_write_bytes += outcome.spool_bytes
                    metrics.lane_output_blocks[lane_name] = (
                        metrics.lane_output_blocks.get(lane_name, 0) + len(outcome.blocks)
                    )
                    if job.archive_path is not None:
                        metrics.zip_metrics["member_parse_count"] = (
                            metrics.zip_metrics.get("member_parse_count", 0) + 1
                        )
                    for block in outcome.blocks:
                        if block.extra.get("legacy_lane"):
                            lane_metric = (
                                f"{str(block.extra['legacy_lane'])}_block_count"
                            )
                            metrics.legacy_office_metrics[lane_metric] = (
                                metrics.legacy_office_metrics.get(lane_metric, 0)
                                + 1
                            )
                            converter = str(
                                block.extra.get("legacy_converter") or "unknown"
                            )
                            converter_metric = f"converter_{converter}_block_count"
                            metrics.legacy_office_metrics[converter_metric] = (
                                metrics.legacy_office_metrics.get(
                                    converter_metric,
                                    0,
                                )
                                + 1
                            )
                        if block.block_type == "xlsx_row":
                            metrics.xlsx_metrics["row_block_count"] = (
                                metrics.xlsx_metrics.get("row_block_count", 0) + 1
                            )
                            if block.extra.get("sheet_parallel"):
                                metrics.xlsx_metrics["parallel_row_block_count"] = (
                                    metrics.xlsx_metrics.get(
                                        "parallel_row_block_count",
                                        0,
                                    )
                                    + 1
                                )
                            if block.extra.get("shared_strings_mode") == "disk":
                                metrics.xlsx_metrics["disk_shared_string_row_count"] = (
                                    metrics.xlsx_metrics.get(
                                        "disk_shared_string_row_count",
                                        0,
                                    )
                                    + 1
                                )
                        if block.block_type == "pdf_page":
                            metrics.pdf_metrics["native_page_count"] = (
                                metrics.pdf_metrics.get("native_page_count", 0) + 1
                            )
                            metrics.pdf_metrics["pdf_native_pages"] = (
                                metrics.pdf_metrics.get("pdf_native_pages", 0)
                                + 1
                            )
                        if block.source_type != "ocr":
                            continue
                        accumulate_required_ocr_block_metrics(
                            metrics,
                            block,
                        )
                        metric_key = (
                            "ocr_page_count"
                            if block.block_type == "pdf_page_ocr"
                            else "ocr_image_count"
                        )
                        metrics.ocr_metrics[metric_key] = (
                            metrics.ocr_metrics.get(metric_key, 0) + 1
                        )
                        if outcome.worker_pid:
                            worker_runtime = (
                                ocr_runtime_metrics_by_pid.setdefault(
                                    int(outcome.worker_pid),
                                    {},
                                )
                            )
                            for runtime_key in (
                                "detect_requests",
                                "detect_inference_calls",
                                "detect_batch_count",
                                "detect_pixels",
                                "recognize_requests",
                                "recognize_inference_calls",
                                "recognize_batch_count",
                                "recognize_pixels",
                                "microbatch_wait_ms_p50",
                                "microbatch_wait_ms_p95",
                                "microbatch_wait_ms_max",
                                "oversize_single_count",
                                "cancelled_before_batch_count",
                            ):
                                worker_runtime[runtime_key] = max(
                                    float(
                                        worker_runtime.get(runtime_key)
                                        or 0
                                    ),
                                    float(
                                        block.extra.get(runtime_key)
                                        or 0
                                    ),
                                )
                        if block.extra.get("fallback_used"):
                            metrics.ocr_metrics["tiled_fallback_count"] = (
                                metrics.ocr_metrics.get("tiled_fallback_count", 0) + 1
                            )
                        for extra_key, metric_key in (
                            ("tiles_planned", "tiles_planned"),
                            ("tiles_processed", "tiles_processed"),
                            ("tiles_pruned", "tiles_pruned"),
                            ("tile_regions_detected", "tile_regions_detected"),
                            ("tile_regions_recognized", "tile_regions_recognized"),
                            (
                                "tile_regions_skipped_resolved",
                                "tile_regions_skipped_resolved",
                            ),
                            ("crop_dedup_hits", "crop_dedup_hits"),
                            ("recognizer_batches", "recognizer_batches"),
                        ):
                            metrics.ocr_metrics[metric_key] = (
                                metrics.ocr_metrics.get(metric_key, 0)
                                + max(0, int(block.extra.get(extra_key) or 0))
                            )
                        if block.block_type == "pdf_page_ocr":
                            metrics.pdf_metrics["ocr_page_count"] = (
                                metrics.pdf_metrics.get("ocr_page_count", 0) + 1
                            )
                            if block.extra.get("pdf_dynamic_dpi"):
                                metrics.pdf_metrics["dynamic_dpi_page_count"] = (
                                    metrics.pdf_metrics.get(
                                        "dynamic_dpi_page_count",
                                        0,
                                    )
                                    + 1
                                )
                            if block.extra.get("pdf_full_page_fallback"):
                                metrics.pdf_metrics["full_page_fallback_count"] = (
                                    metrics.pdf_metrics.get(
                                        "full_page_fallback_count",
                                        0,
                                    )
                                    + 1
                                )
                            metrics.pdf_metrics["upgraded_region_count"] = (
                                metrics.pdf_metrics.get("upgraded_region_count", 0)
                                + max(
                                    0,
                                    int(block.extra.get("pdf_upgraded_regions") or 0),
                                )
                            )
                            for extra_key, metric_key in (
                                (
                                    "pdf_preview_render_ms",
                                    "pdf_preview_render_ms",
                                ),
                                (
                                    "pdf_region_200dpi_render_ms",
                                    "pdf_region_render_ms",
                                ),
                                (
                                    "pdf_region_300dpi_render_ms",
                                    "pdf_region_render_ms",
                                ),
                            ):
                                metrics.pdf_metrics[metric_key] = (
                                    metrics.pdf_metrics.get(metric_key, 0)
                                    + max(
                                        0,
                                        int(
                                            block.extra.get(extra_key)
                                            or 0
                                        ),
                                    )
                                )
                        if outcome.worker_pid:
                            ocr_model_loads_by_pid[outcome.worker_pid] = max(
                                ocr_model_loads_by_pid.get(outcome.worker_pid, 0),
                                int(block.extra.get("ocr_model_load_count") or 0),
                            )
                            ocr_model_load_ms_by_pid[outcome.worker_pid] = max(
                                ocr_model_load_ms_by_pid.get(outcome.worker_pid, 0),
                                int(block.extra.get("ocr_model_load_ms") or 0),
                            )
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
                        completed_cost=max(0.0, job.estimated_cost),
                        service_seconds=max(
                            0.001,
                            outcome.parse_ms / 1000.0,
                        ),
                        completed_file=str(outcome.file_path),
                        worker_pid=outcome.worker_pid,
                        process_result_bytes=outcome.spool_bytes or None,
                        process_descriptor_bytes=descriptor_bytes,
                    )

            writer_summary = writer.finish()
            metrics.database_write_ms = writer_summary.write_ms
            cached_after_write = self.db.find_cached_documents(list(active_jobs))
            for identity, job in active_jobs.items():
                cached = cached_after_write.get(identity)
                if cached is None:
                    continue
                document_id, status = cached
                self.db.link_cached_document(
                    [job.file_id, *job.alias_file_ids],
                    document_id,
                    identity[0],
                    identity[1],
                    identity[2],
                    status,
                )
            self.db.record_file_metrics(run_id, file_timings)

            if defer_fts and (all_jobs or resumed_full_batch or requires_full_rebuild):
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
                if candidate_version_id is None:
                    fts_published = False
                    candidate_version_id = IndexVersionPublisher(
                        self.db
                    ).begin_candidate(
                        root_id=root_id,
                        run_id=run_id,
                        version_key=f"root:{root_id}:run:{run_id}",
                    )
                fts_started = time.perf_counter()
                try:
                    IndexVersionPublisher(self.db).publish(
                        candidate_version_id,
                        writer_idle=writer.is_idle(),
                        workers_idle=all(
                            not lane.pending and not lane.futures
                            for lane in lanes.values()
                        ),
                        golden_query_gate=_fast_publish_query_gate,
                    )
                except IndexPublishGateError as exc:
                    logger.error(
                        "Candidate FTS remains unpublished; failed gates=%s",
                        ",".join(exc.failed_gates),
                    )
                else:
                    fts_published = True
                metrics.fts_build_ms = int(
                    (time.perf_counter() - fts_started) * 1000
                )

            completion = self.db.root_completion(root_id)
            is_complete = completion["blocking"] == 0 and fts_published
            self.db.update_root_scan_time(root_id, "ready" if is_complete else "incomplete")
            if is_complete and full_batch:
                self.db.mark_full_batch_complete()
            if is_complete and requires_full_rebuild:
                self.db.mark_full_rebuild_complete()
            metrics.process_spawn_count = len(worker_pids)
            metrics.ocr_metrics["model_load_count"] = sum(
                ocr_model_loads_by_pid.values()
            )
            metrics.ocr_metrics["model_load_ms"] = sum(
                ocr_model_load_ms_by_pid.values()
            )
            metrics.ocr_metrics["ocr_model_load_count"] = int(
                metrics.ocr_metrics["model_load_count"]
            )
            metrics.ocr_metrics["ocr_model_load_ms"] = int(
                metrics.ocr_metrics["model_load_ms"]
            )
            ocr_worker_pids = {
                int(timing.worker_pid)
                for timing in file_timings
                if timing.queue_name == "ocr"
                and timing.worker_pid is not None
            }
            metrics.ocr_metrics["ocr_worker_count_peak"] = len(
                ocr_worker_pids
            )
            for runtime_key in (
                "detect_requests",
                "detect_inference_calls",
                "detect_batch_count",
                "detect_pixels",
                "recognize_requests",
                "recognize_inference_calls",
                "recognize_batch_count",
                "recognize_pixels",
                "oversize_single_count",
                "cancelled_before_batch_count",
            ):
                metrics.ocr_metrics[runtime_key] = int(
                    sum(
                        worker_metrics.get(runtime_key, 0)
                        for worker_metrics
                        in ocr_runtime_metrics_by_pid.values()
                    )
                )
            for public_key, runtime_key in (
                ("ocr_detect_requests", "detect_requests"),
                ("ocr_detect_calls", "detect_inference_calls"),
                ("ocr_detect_pixels", "detect_pixels"),
                ("ocr_recognize_requests", "recognize_requests"),
                (
                    "ocr_recognize_calls",
                    "recognize_inference_calls",
                ),
                ("ocr_recognize_pixels", "recognize_pixels"),
            ):
                runtime_value = int(
                    metrics.ocr_metrics.get(runtime_key) or 0
                )
                if runtime_value:
                    metrics.ocr_metrics[public_key] = runtime_value
            metrics.ocr_metrics["detect_average_batch_size"] = (
                round(
                    int(metrics.ocr_metrics["detect_requests"])
                    / max(
                        1,
                        int(
                            metrics.ocr_metrics[
                                "detect_batch_count"
                            ]
                        ),
                    ),
                    3,
                )
                if int(metrics.ocr_metrics["detect_batch_count"])
                else 0.0
            )
            metrics.ocr_metrics["recognize_average_batch_size"] = (
                round(
                    int(metrics.ocr_metrics["recognize_requests"])
                    / max(
                        1,
                        int(
                            metrics.ocr_metrics[
                                "recognize_batch_count"
                            ]
                        ),
                    ),
                    3,
                )
                if int(metrics.ocr_metrics["recognize_batch_count"])
                else 0.0
            )
            for wait_key in (
                "microbatch_wait_ms_p50",
                "microbatch_wait_ms_p95",
                "microbatch_wait_ms_max",
            ):
                metrics.ocr_metrics[wait_key] = round(
                    max(
                        [
                            0.0,
                            *[
                                worker_metrics.get(wait_key, 0.0)
                                for worker_metrics
                                in ocr_runtime_metrics_by_pid.values()
                            ],
                        ]
                    ),
                    3,
                )
            recognize_batches = int(
                metrics.ocr_metrics.get("recognizer_batches") or 0
            )
            recognize_requests = int(
                metrics.ocr_metrics.get("ocr_recognize_requests") or 0
            )
            metrics.ocr_metrics["ocr_microbatch_count"] = (
                recognize_batches
            )
            metrics.ocr_metrics["ocr_average_microbatch_size"] = (
                round(
                    recognize_requests / max(1, recognize_batches),
                    3,
                )
                if recognize_batches
                else 0.0
            )
            metrics.total_ms = int((time.perf_counter() - run_started) * 1000)
            if resource_monitor is not None:
                metrics.peak_rss_bytes = max(
                    metrics.peak_rss_bytes,
                    resource_monitor.peak_rss_bytes,
                )
            metrics.resource_metrics["rss_budget_exceeded"] = bool(
                int(metrics.peak_rss_bytes)
                > int(
                    metrics.resource_metrics.get(
                        "rss_budget_bytes",
                        0,
                    )
                    or 0
                )
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
            planning_runner.cancel_active()
            for key, value in planning_runner.metrics.items():
                metrics.hang_metrics[key] = (
                    int(metrics.hang_metrics.get(key) or 0)
                    + int(value)
                )
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
            with self._pause_lock:
                self._pause_lanes = None
                self._pause_writer = None
                self._pause_executors = None
                self._pause_process_executors = None
                self._pause_spool_dir = None
                self._pause_estimator = None
                self._current_metrics = None
            self._planning_runner = None
            self._planning_control_dir = None
            self._planning_metadata = {}
            self._pdf_document_jobs = {}
            self._pdf_failed_documents = set()
            self._pdf_confirmation_buffers = {}
            resume_processes(spool_dir)
            cleanup_registered_office_processes(spool_dir)
            if spool_dir.exists():
                shutil.rmtree(spool_dir, ignore_errors=True)
            metrics.total_ms = max(metrics.total_ms, int((time.perf_counter() - run_started) * 1000))
            metrics.eta_metrics.setdefault("replay_events", []).append(
                {
                    "at_seconds": round(
                        max(0.0, time.perf_counter() - run_started),
                        6,
                    ),
                    "event_type": "finish",
                    "remaining_cost_by_lane": {},
                    "active_elapsed_by_lane": {},
                    "workers_by_lane": dict(metrics.lane_worker_limits),
                    "run_status": run_status,
                }
            )
            metrics.eta_metrics.pop("_run_started_monotonic", None)
            metrics.eta_metrics.pop("_last_eta_seconds", None)
            metrics.eta_metrics.pop("_last_trace_at_seconds", None)
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
        return list(
            self._iter_discovered_files(
                root_path,
                include_subfolders,
                token,
                summary,
                progress_callback,
            )
        )

    def _iter_discovered_files(
        self,
        root_path: Path,
        include_subfolders: bool,
        token: CancelToken,
        summary: IndexSummary,
        progress_callback: ProgressCallback | None,
    ) -> Iterable[Path]:
        runner = self._planning_runner
        control_dir = self._planning_control_dir
        if runner is None or control_dir is None:
            raise RuntimeError("recoverable planning runner is not initialized")
        last_emit = 0.0
        discovered_bytes = 0
        discovered_count = 0

        def report_planning(progress: PlanningProgress) -> None:
            nonlocal last_emit
            now = time.monotonic()
            if now - last_emit < 0.2 and progress.completed > 1:
                return
            self._emit(
                progress_callback,
                "planning",
                summary,
                current_file=progress.detail,
                pending=0,
                total_files=discovered_count,
                total_bytes=discovered_bytes,
                planning_phase=progress.phase,
                planning_cursor=progress.cursor,
                planning_completed=progress.completed,
                planning_total=progress.total,
                planning_bytes_read=progress.bytes_read,
                planning_worker_pid=progress.worker_pid,
                phase_label=_planning_phase_label(progress.phase),
            )
            last_emit = now

        batches = runner.stream(
            "directory_enumeration",
            discover_file_batches,
            root_path,
            include_subfolders,
            self.settings.to_dict(),
            control_dir,
            max(1, int(self.settings.planning_discovery_batch_size or 128)),
            cancel_check=token.throw_if_cancelled,
            progress_callback=report_planning,
        )
        for path_batch in batches:
            token.wait_if_paused()
            token.throw_if_cancelled()
            metadata, metadata_errors = self._stat_paths_recoverable(
                [str(path) for path in path_batch],
                token,
                report_planning,
            )
            self._planning_metadata.update(
                {item.path: item for item in metadata}
            )
            for failed_path, exc in metadata_errors:
                logger.error("Failed to read metadata for %s: %s", failed_path, exc)
            for path_text in path_batch:
                file_path = Path(path_text)
                discovered_count += 1
                item = self._planning_metadata.get(str(file_path))
                if item is not None:
                    discovered_bytes += item.size_bytes
                summary.scanned += 1
                now = time.monotonic()
                if (
                    discovered_count == 1
                    or discovered_count % 100 == 0
                    or now - last_emit >= 0.2
                ):
                    self._emit(
                        progress_callback,
                        "discovering",
                        summary,
                        current_file=str(file_path),
                        pending=0,
                        total_files=discovered_count,
                        total_bytes=discovered_bytes,
                        planning_worker_pid=(
                            item.worker_pid if item is not None else 0
                        ),
                        planning_phase="file_stat",
                        phase_label="正在发现文件并读取元数据",
                    )
                    last_emit = now
                yield file_path

    def _stat_paths_recoverable(
        self,
        paths: list[str],
        token: CancelToken,
        progress_callback: Callable[[PlanningProgress], None] | None = None,
    ) -> tuple[list[PreparedFileMetadata], list[tuple[Path, Exception]]]:
        if not paths:
            return [], []
        runner = self._planning_runner
        if runner is None:
            raise RuntimeError("recoverable planning runner is not initialized")
        try:
            result = runner.run(
                "file_stat",
                stat_file_batch,
                paths,
                bool(self.settings.compute_full_hash),
                False,
                self._planning_control_dir,
                cancel_check=token.throw_if_cancelled,
                progress_callback=progress_callback,
            )
        except PlanningNoProgressError as exc:
            if len(paths) == 1:
                return [], [(Path(paths[0]), exc)]
            middle = max(1, len(paths) // 2)
            left_metadata, left_errors = self._stat_paths_recoverable(
                paths[:middle],
                token,
                progress_callback,
            )
            right_metadata, right_errors = self._stat_paths_recoverable(
                paths[middle:],
                token,
                progress_callback,
            )
            return (
                [*left_metadata, *right_metadata],
                [*left_errors, *right_errors],
            )
        if not isinstance(result, StatBatchResult):
            raise PlanningWorkerError("file_stat 返回了无效的元数据结果")
        errors = [
            (
                Path(item.path),
                OSError(f"{item.error_type}: {item.message}"),
            )
            for item in result.errors
        ]
        return list(result.metadata), errors

    def _prepare_jobs(
        self,
        root_id: int,
        file_paths: list[Path],
        run_id: str,
        summary: IndexSummary,
        metrics: IndexRunMetrics,
        token: CancelToken,
        active_jobs: dict[tuple[str, str, str], ParseJob] | None = None,
        planning_progress_callback: (
            Callable[[PlanningProgress], None] | None
        ) = None,
    ) -> list[ParseJob]:
        self._ensure_planning_runner(run_id)
        prepared_rows: list[PreparedSource] = []
        source_spool_dir = TEMP_DIR / "process_results" / run_id / "source_spool"
        physical_rows: dict[str, tuple[Path, int, bool, int]] = {}
        batch_size = max(64, min(1024, int(self.settings.index_write_batch_size or 32) * 8))
        for offset in range(0, len(file_paths), batch_size):
            token.throw_if_cancelled()
            chunk = file_paths[offset : offset + batch_size]
            versions = {
                str(path): parser_identity_for_path(path, self.settings)[1]
                for path in chunk
            }
            metadata = [
                self._planning_metadata[str(path)]
                for path in chunk
                if str(path) in self._planning_metadata
            ]
            missing_paths = [
                str(path)
                for path in chunk
                if str(path) not in self._planning_metadata
            ]
            stat_errors: list[tuple[Path, Exception]] = []
            if missing_paths:
                recovered_metadata, stat_errors = self._stat_paths_recoverable(
                    missing_paths,
                    token,
                )
                metadata.extend(recovered_metadata)
                self._planning_metadata.update(
                    {item.path: item for item in recovered_metadata}
                )
            prepared, errors = self.db.upsert_precomputed_file_metadata_many(
                root_id,
                metadata,
                retry_failed_files=self.settings.retry_failed_files,
                compute_full_hash=self.settings.compute_full_hash,
                mark_processing=False,
                parser_versions=versions,
            )
            for file_path, exc in [*stat_errors, *errors]:
                logger.error("Failed to read metadata for %s: %s", file_path, exc)
                summary.failed += 1
            for file_path, file_id, changed in prepared:
                metadata_item = self._planning_metadata[str(file_path)]
                physical_rows[str(file_path)] = (
                    file_path,
                    file_id,
                    changed,
                    metadata_item.size_bytes,
                )
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    if changed:
                        self.db.mark_video_excluded([file_id])
                    summary.excluded_video += 1
                elif file_path.suffix.lower() == ".zip" and (
                    changed or self.db.zip_container_requires_sync(file_id)
                ):
                    try:
                        planning_runner = self._planning_runner
                        planning_control_dir = self._planning_control_dir
                        if planning_runner is None or planning_control_dir is None:
                            raise RuntimeError("planning runner is unavailable")
                        manifest = planning_runner.run(
                            "zip_manifest",
                            scan_zip_manifest_task,
                            file_path,
                            self.settings.to_dict(),
                            planning_control_dir,
                            cancel_check=token.throw_if_cancelled,
                        )
                    except (
                        OSError,
                        ValueError,
                        zipfile.BadZipFile,
                        PlanningWorkerError,
                    ):
                        logger.info(
                            "ZIP manifest planning fell back to the container parser: %s",
                            file_path,
                            exc_info=True,
                        )
                        prepared_rows.append(
                            PreparedSource(
                                file_path,
                                file_id,
                                size_bytes=metadata_item.size_bytes,
                            )
                        )
                        continue
                    # Unsafe or encrypted members retain the existing container-level
                    # partial-failure semantics instead of being silently discarded.
                    if manifest.unsafe_members or manifest.encrypted_members:
                        prepared_rows.append(
                            PreparedSource(
                                file_path,
                                file_id,
                                size_bytes=metadata_item.size_bytes,
                            )
                        )
                        continue
                    versions = {
                        f"{file_path} > {member.internal_path}": parser_identity_for_path(
                            Path(member.internal_path), self.settings
                        )[1]
                        for member in manifest.members
                    }
                    member_rows = self.db.sync_zip_members(
                        root_id,
                        file_id,
                        file_path,
                        manifest.members,
                        versions,
                        retry_failed_files=self.settings.retry_failed_files,
                    )
                    for member, member_file_id, member_changed, display_path in member_rows:
                        if member_changed:
                            prepared_rows.append(
                                PreparedSource(
                                    file_path=Path(display_path),
                                    file_id=member_file_id,
                                    size_bytes=int(member.size_bytes),
                                    archive_path=file_path,
                                    archive_member_index=int(member.member_index),
                                    archive_member_name=str(member.member_name),
                                    archive_member_crc32=int(member.crc32),
                                    archive_internal_path=str(member.internal_path),
                                )
                            )
                        else:
                            summary.skipped += 1
                elif changed:
                    prepared_rows.append(
                        PreparedSource(
                            file_path,
                            file_id,
                            size_bytes=metadata_item.size_bytes,
                        )
                    )
                else:
                    summary.skipped += 1

        # A newly added directory file may match a ZIP member indexed during a
        # previous run. Only those extension/size candidates are promoted back
        # into planning and fully hashed; ordinary unchanged files stay cheap.
        known_zip_candidates = self.db.active_zip_member_candidate_keys()
        prepared_file_ids = {source.file_id for source in prepared_rows}
        physical_states = self.db.file_content_states(
            [file_id for _path, file_id, _changed, _size in physical_rows.values()]
        )
        for file_path, file_id, _changed, size_bytes in physical_rows.values():
            candidate = (file_path.suffix.lower(), size_bytes)
            if candidate not in known_zip_candidates or file_id in prepared_file_ids:
                continue
            content_hash_full, content_key = physical_states.get(file_id, (None, None))
            if content_hash_full and content_key == f"sha256:{content_hash_full}":
                continue
            prepared_rows.append(
                PreparedSource(
                    file_path,
                    file_id,
                    size_bytes=size_bytes,
                )
            )
            prepared_file_ids.add(file_id)

        physical_candidate_keys = {
            (file_path.suffix.lower(), int(size_bytes))
            for file_path, _file_id, _changed, size_bytes in physical_rows.values()
        }
        zip_group_counts: dict[tuple[str, int, int], int] = {}
        for source in prepared_rows:
            if source.archive_path is None or source.archive_member_crc32 is None:
                continue
            key = (
                source.file_path.suffix.lower(),
                _prepared_source_size(source),
                int(source.archive_member_crc32),
            )
            zip_group_counts[key] = zip_group_counts.get(key, 0) + 1
        candidate_hash_keys = known_zip_candidates | physical_candidate_keys
        for source in prepared_rows:
            if source.archive_path is None or source.archive_member_index is None or source.archive_member_crc32 is None:
                continue
            if source.exact_sha256 is not None:
                continue
            size_bytes = _prepared_source_size(source)
            ext_key = (source.file_path.suffix.lower(), size_bytes)
            triple_key = (source.file_path.suffix.lower(), size_bytes, int(source.archive_member_crc32))
            needs_ocr_spool = bool(
                self.settings.enable_ocr
                and (
                    (
                        self.settings.ocr_images
                        and source.file_path.suffix.lower()
                        in IMAGE_EXTENSIONS
                    )
                    or (
                        self.settings.ocr_scanned_pdf
                        and source.file_path.suffix.lower() == ".pdf"
                    )
                )
            )
            if (
                not needs_ocr_spool
                and ext_key not in candidate_hash_keys
                and zip_group_counts.get(triple_key, 0) <= 1
            ):
                continue
            if source.archive_path is None:
                continue
            try:
                planning_runner = self._planning_runner
                if planning_runner is None:
                    raise RuntimeError("planning runner is unavailable")
                prepared_member = planning_runner.run(
                    "zip_member_prepare",
                    prepare_zip_member_task,
                    source.archive_path,
                    int(source.archive_member_index),
                    source.archive_internal_path,
                    size_bytes,
                    int(source.archive_member_crc32),
                    source.file_path.suffix.lower(),
                    source_spool_dir,
                    False,
                    self._planning_control_dir,
                    cancel_check=token.throw_if_cancelled,
                )
                if not isinstance(prepared_member, PreparedZipMemberResult):
                    raise PlanningWorkerError(
                        "zip_member_prepare 返回了无效结果"
                    )
                source.exact_sha256 = prepared_member.sha256
                source.source_spool_path = prepared_member.spool_path
                source.image_width = prepared_member.image_width
                source.image_height = prepared_member.image_height
                metrics.source_open_count += 1
                metrics.source_bytes_read += prepared_member.bytes_read
                metrics.full_hash_count += 1
                metrics.full_hash_bytes += prepared_member.bytes_read
                metrics.spool_write_bytes += prepared_member.bytes_read
                metrics.zip_metrics["member_extract_count"] = (
                    metrics.zip_metrics.get("member_extract_count", 0) + 1
                )
                metrics.zip_metrics["member_extract_bytes"] = (
                    metrics.zip_metrics.get("member_extract_bytes", 0)
                    + prepared_member.bytes_read
                )
                self.db.set_content_hash_full(source.file_id, source.exact_sha256)
            except Exception:
                logger.info(
                    "ZIP member candidate hash skipped during planning: %s",
                    source.file_path,
                    exc_info=True,
                )

        candidate_sizes: dict[tuple[str, int], int] = {}
        for source in prepared_rows:
            key = (
                source.file_path.suffix.lower(),
                _prepared_source_size(source),
            )
            candidate_sizes[key] = candidate_sizes.get(key, 0) + 1
        metrics.dedup_candidate_count += sum(
            count for count in candidate_sizes.values() if count > 1
        )
        metrics.dedup_full_hash_count += sum(
            1 for source in prepared_rows if source.exact_sha256 is not None
        )
        exact_member_candidates = known_zip_candidates | {
            (source.file_path.suffix.lower(), int(source.size_bytes or 0))
            for source in prepared_rows
            if source.exact_sha256 is not None
        }
        physical_sources = [
            source
            for source in prepared_rows
            if source.exact_sha256 is None and source.archive_path is None
        ]
        fingerprint_results, fingerprint_errors = (
            self._fingerprint_paths_recoverable(
                [
                    (
                        str(source.file_path),
                        bool(
                            self.settings.enable_ocr
                            and self.settings.ocr_images
                            and source.file_path.suffix.lower()
                            in IMAGE_EXTENSIONS
                        ),
                    )
                    for source in physical_sources
                ],
                source_spool_dir,
                token,
                planning_progress_callback,
            )
        )
        fingerprint_by_path = {
            result.source_path: result for result in fingerprint_results
        }
        source_by_path = {
            str(source.file_path): source for source in physical_sources
        }
        for failed_path, exc in fingerprint_errors:
            failed_source = source_by_path.get(str(failed_path))
            if failed_source is not None:
                self.db.record_failure(
                    failed_source.file_id,
                    "PLANNING_CONTENT_HASH_FAILED",
                    str(exc),
                    parser_name="planning",
                    retryable=True,
                )
            summary.failed += 1
            logger.error("Failed to fingerprint %s: %s", failed_path, exc)
        fingerprinted: list[ParseJob] = []
        for source in prepared_rows:
            token.wait_if_paused()
            token.throw_if_cancelled()
            file_path = source.file_path
            file_id = source.file_id
            parser_name, parser_version = parser_identity_for_path(file_path, self.settings)
            source_spool_path: Path | None = source.source_spool_path
            source_modified_time_ns = 0
            image_width = int(source.image_width)
            image_height = int(source.image_height)
            if source.exact_sha256 is not None:
                fingerprint = ContentFingerprint(
                    f"sha256:{source.exact_sha256}",
                    int(source.size_bytes or 0),
                    "sha256",
                )
            elif source.archive_path is not None:
                size_bytes = _prepared_source_size(source)
                fingerprint = ContentFingerprint(
                    f"zip:{source.archive_path}:{source.archive_member_index}:{source.archive_member_crc32}:{size_bytes}",
                    size_bytes,
                    "zip_member",
                )
            else:
                if file_path.suffix.lower() in SUPPORTED_EXTENSIONS:
                    fingerprint_result = fingerprint_by_path.get(str(file_path))
                    if fingerprint_result is None:
                        continue
                    fingerprint = fingerprint_result.fingerprint
                    source_spool_path = fingerprint_result.spool_path
                    source_modified_time_ns = (
                        fingerprint_result.source_modified_time_ns
                    )
                    image_width = fingerprint_result.image_width
                    image_height = fingerprint_result.image_height
                    bytes_read = fingerprint_result.bytes_read
                    if bytes_read > 0:
                        metrics.source_open_count += 1
                        metrics.source_bytes_read += bytes_read
                    if fingerprint.method == "sha256":
                        metrics.full_hash_count += 1
                        metrics.full_hash_bytes += bytes_read
                    if source_spool_path is not None:
                        metrics.spool_write_bytes += bytes_read
                else:
                    fingerprint = self._fingerprint_for(file_path)
                candidate = (file_path.suffix.lower(), _prepared_source_size(source))
                if candidate in exact_member_candidates:
                    if fingerprint.key.startswith("sha256:"):
                        full_hash = fingerprint.key.partition(":")[2]
                        metrics.hash_reused_count += 1
                    else:
                        exact_results, exact_errors = (
                            self._fingerprint_paths_recoverable(
                                [(str(file_path), True)],
                                source_spool_dir,
                                token,
                                planning_progress_callback,
                            )
                        )
                        if exact_errors or not exact_results:
                            error = (
                                exact_errors[0][1]
                                if exact_errors
                                else PlanningWorkerError(
                                    "content_hash_full 没有返回结果"
                                )
                            )
                            self.db.record_failure(
                                file_id,
                                "PLANNING_CONTENT_HASH_FAILED",
                                str(error),
                                parser_name="planning",
                                retryable=True,
                            )
                            summary.failed += 1
                            continue
                        exact_result = exact_results[0]
                        full_hash = exact_result.fingerprint.key.partition(":")[2]
                        source_spool_path = exact_result.spool_path
                        source_modified_time_ns = (
                            exact_result.source_modified_time_ns
                        )
                        bytes_read = exact_result.bytes_read
                        metrics.source_open_count += 1
                        metrics.source_bytes_read += bytes_read
                        metrics.full_hash_count += 1
                        metrics.full_hash_bytes += bytes_read
                        if source_spool_path is not None:
                            metrics.spool_write_bytes += bytes_read
                    fingerprint = ContentFingerprint(
                        f"sha256:{full_hash}",
                        candidate[1],
                        "sha256",
                    )
                    self.db.set_content_hash_full(file_id, full_hash)
                    metrics.dedup_full_hash_count += 1
            if not self.settings.enable_parse_cache:
                content_key = f"path:{file_path}:{fingerprint.key}"
            else:
                content_key = fingerprint.key
            size_bytes = _prepared_source_size(source)
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
                    memory_estimate_bytes=estimate_job_memory_bytes(
                        file_path,
                        size_bytes,
                        fingerprint.relevant_bytes,
                    ),
                    estimated_cost=estimate_parse_cost(file_path, size_bytes, fingerprint.relevant_bytes),
                    queued_monotonic=time.perf_counter(),
                    archive_path=source.archive_path,
                    archive_member_index=source.archive_member_index,
                    archive_member_name=source.archive_member_name,
                    archive_member_crc32=source.archive_member_crc32,
                    archive_internal_path=source.archive_internal_path,
                    exact_sha256=source.exact_sha256 or "",
                    content_hash_full=source.exact_sha256,
                    source_spool_path=source_spool_path,
                    source_modified_time_ns=source_modified_time_ns,
                    source_size_bytes=size_bytes,
                    ocr_width=image_width,
                    ocr_height=image_height,
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
            if len(group) > 1:
                metrics.dedup_verified_source_count += len(group)
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
                metrics.dedup_parse_avoided_count += len(group)
                metrics.dedup_bytes_avoided += sum(job.size_bytes for job in group)
                if group[0].file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    summary.excluded_video += len(group)
                else:
                    summary.skipped += len(group)
                continue
            existing = active_jobs.get(identity) if active_jobs is not None else None
            if existing is not None:
                existing.alias_file_ids = tuple(
                    dict.fromkeys((*existing.alias_file_ids, *file_ids))
                )
                metrics.dedup_verified_source_count += len(group)
                metrics.dedup_parse_avoided_count += len(group)
                metrics.dedup_bytes_avoided += sum(job.size_bytes for job in group)
                continue
            primary = group[0]
            primary.alias_file_ids = tuple(job.file_id for job in group[1:])
            metrics.dedup_parse_avoided_count += max(0, len(group) - 1)
            metrics.dedup_bytes_avoided += sum(job.size_bytes for job in group[1:])
            metrics.cache_misses += 1
            if primary.file_path.suffix.lower() == ".pdf":
                try:
                    page_jobs = self._plan_pdf_page_jobs(
                        primary,
                        run_id,
                        token,
                    )
                except PlanningWorkerError:
                    logger.info(
                        "PDF page graph planning failed; parser will classify the document: %s",
                        primary.file_path,
                        exc_info=True,
                    )
                else:
                    jobs.extend(page_jobs)
                    if active_jobs is not None:
                        active_jobs[identity] = primary
                    continue
            jobs.append(primary)
            priority = max(1, min(1_000_000, int(primary.estimated_cost * 100)))
            task_specs.append((primary.file_id, run_id, primary.lane, priority))
            task_jobs.append(primary)

        task_ids = self.db.create_parse_tasks(task_specs)
        for job, task_id in zip(task_jobs, task_ids, strict=True):
            job.task_id = task_id
            if job.parser_name in {"pdf", "zip", "xlsx_stream", "image_ocr"}:
                job.checkpoint_path = persistent_checkpoint_path(job)
                checkpoint = load_partial_parse_checkpoint(
                    job,
                    job.checkpoint_path.parent,
                    consume=False,
                )
                if checkpoint is not None and checkpoint.resume_cursor > 0:
                    job.resume_cursor = checkpoint.resume_cursor
            if job.lane == "ocr":
                self._register_ocr_request(
                    job,
                    width=job.ocr_width,
                    height=job.ocr_height,
                )
            if active_jobs is not None:
                active_jobs[
                    (job.content_key, job.parser_name, job.parser_version)
                ] = job
        return _order_planned_jobs_fairly(jobs)

    def _plan_pdf_page_jobs(
        self,
        parent_job: ParseJob,
        run_id: str,
        token: CancelToken,
    ) -> list[ParseJob]:
        runner = self._planning_runner
        if runner is None:
            raise RuntimeError("recoverable planning runner is not initialized")
        if (
            parent_job.archive_path is not None
            and parent_job.source_spool_path is None
        ):
            if (
                parent_job.archive_member_index is None
                or parent_job.archive_member_crc32 is None
            ):
                raise PlanningWorkerError(
                    "ZIP 内 PDF 缺少稳定成员身份"
                )
            prepared_member = runner.run(
                "zip_pdf_member_prepare",
                prepare_zip_member_task,
                parent_job.archive_path,
                int(parent_job.archive_member_index),
                parent_job.archive_internal_path,
                parent_job.size_bytes,
                int(parent_job.archive_member_crc32),
                ".pdf",
                TEMP_DIR / "process_results" / run_id / "source_spool",
                False,
                self._planning_control_dir,
                cancel_check=token.throw_if_cancelled,
            )
            if not isinstance(prepared_member, PreparedZipMemberResult):
                raise PlanningWorkerError(
                    "zip_pdf_member_prepare 返回了无效结果"
                )
            parent_job.source_spool_path = prepared_member.spool_path
            parent_job.exact_sha256 = prepared_member.sha256
            parent_job.content_hash_full = prepared_member.sha256
        parse_path = parent_job.source_spool_path or parent_job.file_path
        scan = runner.run(
            "pdf_scan",
            scan_pdf_document_task,
            parse_path,
            cancel_check=token.throw_if_cancelled,
        )
        if not isinstance(scan, PdfDocumentScanResult):
            raise PlanningWorkerError("pdf_scan 返回了无效结果")
        plans = [
            PdfPagePlan(
                page.page_number,
                page.page_identity,
                page.width_points,
                page.height_points,
                bool(
                    page.requires_ocr
                    and self.settings.enable_ocr
                    and self.settings.ocr_scanned_pdf
                ),
            )
            for page in scan.pages
        ]
        with self._pause_lock:
            current_metrics = self._current_metrics
        if current_metrics is not None:
            current_metrics.pdf_metrics["pdf_documents_total"] = (
                current_metrics.pdf_metrics.get(
                    "pdf_documents_total",
                    0,
                )
                + 1
            )
            current_metrics.pdf_metrics["pdf_pages_total"] = (
                current_metrics.pdf_metrics.get("pdf_pages_total", 0)
                + len(plans)
            )
            current_metrics.pdf_metrics["pdf_ocr_candidate_pages"] = (
                current_metrics.pdf_metrics.get(
                    "pdf_ocr_candidate_pages",
                    0,
                )
                + sum(1 for page in plans if page.requires_ocr)
            )
        graph = PdfTaskGraphRepository(self.db)
        document_task_id = graph.plan_document(
            file_id=parent_job.file_id,
            run_id=run_id,
            source_digest=parent_job.content_key,
            parser_version=parent_job.parser_version,
            pages=plans,
            ocr_config_fingerprint=(
                f"{ADAPTIVE_OCR_VERSION}:{PDF_DYNAMIC_OCR_VERSION}:"
                f"{self.settings.ocr_language}:{self.settings.max_ocr_image_side}"
            ),
        )
        self._pdf_document_jobs[document_task_id] = parent_job
        page_jobs: list[ParseJob] = []
        page_count = max(1, len(plans))
        scheduled_claims = graph.scheduled_page_tasks(document_task_id)
        for claim in scheduled_claims:
            lane = "ocr" if claim.task_type == "pdf_ocr_page" else "pdf"
            page_job = ParseJob(
                    file_id=parent_job.file_id,
                    file_path=parent_job.file_path,
                    task_id=claim.task_id,
                    content_key=(
                        f"{parent_job.content_key}:"
                        f"{claim.task_type}:{claim.page_number}"
                    ),
                    parser_name="pdf",
                    parser_version=parent_job.parser_version,
                    lane=lane,
                    size_bytes=max(1, parent_job.size_bytes // page_count),
                    relevant_bytes=max(1, parent_job.relevant_bytes // page_count),
                    memory_estimate_bytes=max(
                        1,
                        parent_job.memory_estimate_bytes // page_count,
                    ),
                    estimated_cost=max(
                        0.1,
                        parent_job.estimated_cost / page_count,
                    ),
                    queued_monotonic=time.perf_counter(),
                    exact_sha256=parent_job.exact_sha256,
                    content_hash_full=parent_job.content_hash_full,
                    source_spool_path=parent_job.source_spool_path,
                    source_modified_time_ns=(
                        parent_job.source_modified_time_ns
                    ),
                    source_size_bytes=parent_job.source_size_bytes,
                    checkpoint_path=(
                        TEMP_DIR
                        / "checkpoints"
                        / (
                            f"pdf_page_{parent_job.file_id}_"
                            f"{claim.task_id}.pickle"
                        )
                    ),
                    pdf_document_task_id=document_task_id,
                    pdf_page_number=claim.page_number,
                    pdf_task_type=claim.task_type,
                    pdf_source_digest=parent_job.content_key,
                )
            if page_job.lane == "ocr":
                self._register_ocr_request(
                    page_job,
                    width=max(
                        1,
                        int(
                            float(claim.payload.get("width_points") or 0)
                            / 72.0
                            * 200
                        ),
                    ),
                    height=max(
                        1,
                        int(
                            float(claim.payload.get("height_points") or 0)
                            / 72.0
                            * 200
                        ),
                    ),
                )
            page_jobs.append(page_job)
        if not scheduled_claims and graph.merge_readiness(document_task_id).ready:
            merge_task_id = graph.merge_task_id(document_task_id)
            if merge_task_id is None:
                raise RuntimeError(
                    f"PDF graph is ready without merge task: {document_task_id}"
                )
            page_jobs.append(
                ParseJob(
                    file_id=parent_job.file_id,
                    file_path=parent_job.file_path,
                    task_id=merge_task_id,
                    content_key=f"{parent_job.content_key}:document_merge",
                    parser_name="pdf",
                    parser_version=parent_job.parser_version,
                    lane="pdf",
                    size_bytes=1,
                    relevant_bytes=1,
                    memory_estimate_bytes=1,
                    estimated_cost=0.01,
                    queued_monotonic=time.perf_counter(),
                    exact_sha256=parent_job.exact_sha256,
                    content_hash_full=parent_job.content_hash_full,
                    source_spool_path=parent_job.source_spool_path,
                    source_modified_time_ns=(
                        parent_job.source_modified_time_ns
                    ),
                    source_size_bytes=parent_job.source_size_bytes,
                    pdf_document_task_id=document_task_id,
                    pdf_page_number=0,
                    pdf_task_type="document_merge",
                    pdf_source_digest=parent_job.content_key,
                )
            )
        return page_jobs

    def _register_ocr_request(
        self,
        job: ParseJob,
        *,
        width: int = 0,
        height: int = 0,
    ) -> None:
        if job.task_id is None or job.lane != "ocr":
            return
        job.ocr_width = max(0, int(width))
        job.ocr_height = max(0, int(height))
        source_kind = (
            "pdf_page"
            if job.pdf_task_type == "pdf_ocr_page"
            else (
                "zip_image"
                if job.archive_path is not None
                else "image"
            )
        )
        source_unit = (
            f"page:{job.pdf_page_number}"
            if job.pdf_page_number is not None
            else (
                job.archive_internal_path
                if job.archive_internal_path
                else "image"
            )
        )
        content_sha256 = (
            job.exact_sha256
            or job.content_hash_full
            or (
                job.content_key.partition(":")[2]
                if job.content_key.startswith("sha256:")
                else hashlib.sha256(
                    job.content_key.encode("utf-8")
                ).hexdigest()
            )
        )
        config_payload = {
            "strategy": ADAPTIVE_OCR_VERSION,
            "pdf_strategy": PDF_DYNAMIC_OCR_VERSION,
            "language": self.settings.ocr_language,
            "detection_side": self.settings.max_ocr_image_side,
            "model": ocr_models_fingerprint(),
            "microbatch": {
                "requests": self.settings.ocr_microbatch_max_requests,
                "pixels": self.settings.ocr_microbatch_max_pixels,
                "memory_mb": self.settings.ocr_microbatch_memory_mb,
                "wait_ms": self.settings.ocr_microbatch_wait_ms,
            },
        }
        config_fingerprint = hashlib.sha256(
            json.dumps(
                config_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        pixel_cost = max(
            1,
            int(width) * int(height),
            int(job.memory_estimate_bytes or 0) // 4,
        )
        job.ocr_request_id = OcrRequestRepository(self.db).enqueue(
            file_id=job.file_id,
            parent_task_id=job.task_id,
            source_kind=source_kind,
            source_unit=source_unit,
            image_spool_path=(
                job.source_spool_path or job.file_path
            ),
            content_sha256=content_sha256,
            width=job.ocr_width,
            height=job.ocr_height,
            config_fingerprint=config_fingerprint,
            priority=max(
                1,
                min(1_000_000, int(job.estimated_cost * 100)),
            ),
            pixel_cost=pixel_cost,
            checkpoint_cursor=str(job.resume_cursor or ""),
        )

    def _claim_submitted_ocr_requests(
        self,
        jobs: Iterable[ParseJob],
        task_ids: Iterable[int],
        run_id: str,
    ) -> None:
        submitted = {
            int(task_id)
            for task_id in task_ids
        }
        if not submitted:
            return
        request_jobs = [
            job
            for job in jobs
            if job.task_id in submitted
            and job.ocr_request_id is not None
        ]
        request_ids = [
            int(job.ocr_request_id)
            for job in request_jobs
            if job.ocr_request_id is not None
        ]
        claimed = set(
            OcrRequestRepository(self.db).claim_specific(
                request_ids,
                worker_id=run_id,
                lease_seconds=max(
                    60,
                    int(
                        self.settings.ocr_no_progress_timeout_seconds
                    )
                    * 2,
                ),
            )
        )
        for job in request_jobs:
            if job.ocr_request_id in claimed:
                job.ocr_request_owner = run_id

    def _mark_submitted_tasks_running(
        self,
        task_ids: Iterable[int],
    ) -> None:
        submitted = [int(task_id) for task_id in task_ids]
        if not submitted:
            return
        if self.db.try_mark_tasks_running(submitted):
            return
        # Submission state is a correctness boundary for page confirmation and
        # attempt recovery. A busy zero-wait update may be skipped for progress
        # telemetry, but not for this durable lifecycle transition.
        self.db.mark_tasks_running(submitted)

    def _update_ocr_request_for_outcome(
        self,
        job: ParseJob,
        outcome: ParseOutcome,
        *,
        requeue: bool = False,
    ) -> None:
        request_id = job.ocr_request_id
        if request_id is None:
            return
        repository = OcrRequestRepository(self.db)
        if requeue:
            repository.requeue(
                request_id,
                checkpoint_cursor=str(
                    outcome.resume_cursor or job.resume_cursor or ""
                ),
            )
            job.ocr_request_owner = ""
            return
        if outcome.status in {
            "success",
            "partial_success",
            "metadata_only",
            "ocr_disabled",
        }:
            if (
                outcome.spool_path is None
                or not outcome.spool_checksum
            ):
                repository.fail(
                    request_id,
                    "OCR_RESULT_SPOOL_MISSING",
                    "OCR 结果没有可确认的 spool 或摘要",
                )
                return
            repository.confirm(
                request_id,
                worker_id=job.ocr_request_owner,
                result_spool_path=outcome.spool_path,
                result_digest=outcome.spool_checksum,
            )
            return
        repository.fail(
            request_id,
            outcome.error_code or "OCR_REQUEST_FAILED",
            outcome.error_message or "OCR 请求失败",
        )

    def _record_pdf_page_outcome(
        self,
        job: ParseJob,
        outcome: ParseOutcome,
    ) -> tuple[ParseOutcome, ParseJob] | None:
        document_task_id = int(job.pdf_document_task_id or 0)
        parent_job = self._pdf_document_jobs.get(document_task_id)
        if document_task_id <= 0 or parent_job is None or job.task_id is None:
            raise RuntimeError("PDF page task lost its parent graph identity")
        graph = PdfTaskGraphRepository(self.db)
        with self._pause_lock:
            current_metrics = self._current_metrics
        if current_metrics is not None:
            current_metrics.pdf_metrics[
                "pdf_max_page_queue_wait_ms"
            ] = max(
                current_metrics.pdf_metrics.get(
                    "pdf_max_page_queue_wait_ms",
                    0,
                ),
                max(0, int(outcome.queue_wait_ms)),
            )
        is_recovery_merge = job.pdf_task_type == "document_merge"
        if outcome.status not in {
            "success",
            "partial_success",
            "metadata_only",
        }:
            error_code = outcome.error_code or "PDF_PAGE_FAILED"
            error_message = outcome.error_message or (
                f"第 {job.pdf_page_number or 0} 页任务失败"
            )
            self._flush_pdf_page_confirmations(
                document_task_id,
                graph,
            )
            graph.fail_page_task(job.task_id, error_code, error_message)
            graph.fail_document(
                document_task_id,
                error_code,
                error_message,
            )
            self.db.record_failure(
                parent_job.file_id,
                error_code,
                error_message,
                parser_name="pdf_task_graph",
                retryable=outcome.status == "failed_retryable",
            )
            return None

        if not is_recovery_merge:
            if (
                outcome.spool_path is not None
                and outcome.spool_path.is_file()
                and outcome.spool_checksum
            ):
                result_path = outcome.spool_path
                result_digest = outcome.spool_checksum
            else:
                identity = hashlib.sha256(
                    (
                        f"{job.pdf_source_digest}|{job.parser_version}|"
                        f"{job.pdf_task_type}|{job.pdf_page_number}"
                    ).encode("utf-8")
                ).hexdigest()
                page_dir = TEMP_DIR / "pdf_page_results" / identity[:24]
                page_dir.mkdir(parents=True, exist_ok=True)
                result_path = page_dir / f"{job.task_id}.pickle"
                temporary = result_path.with_suffix(".tmp")
                with temporary.open("wb") as stream:
                    pickle.dump(outcome, stream, protocol=pickle.HIGHEST_PROTOCOL)
                temporary.replace(result_path)
                result_digest = sha256_path(result_path)
            self._pdf_confirmation_buffers.setdefault(
                document_task_id,
                [],
            ).append(
                (int(job.task_id), result_path, result_digest)
            )
            if not job.pdf_confirmation_batch_end:
                return None
            self._flush_pdf_page_confirmations(
                document_task_id,
                graph,
            )
        readiness = graph.merge_readiness(document_task_id)
        if not readiness.ready:
            return None

        merge_started = time.perf_counter()
        page_outcomes: list[tuple[int, str, ParseOutcome, Path]] = []
        for page_number, task_type, spool_path, expected_digest in (
            graph.confirmed_page_results(document_task_id)
        ):
            if not spool_path.is_file():
                raise FileNotFoundError(
                    f"PDF page result is missing: {spool_path}"
                )
            if sha256_path(spool_path) != expected_digest:
                raise ValueError(
                    f"PDF page result digest mismatch: {spool_path}"
                )
            with spool_path.open("rb") as stream:
                page_outcome = pickle.load(stream)
            if not isinstance(page_outcome, ParseOutcome):
                raise TypeError("Invalid PDF page result")
            page_outcomes.append(
                (page_number, task_type, page_outcome, spool_path)
            )
        page_outcomes.sort(
            key=lambda item: (
                item[0],
                0 if item[1] == "pdf_native_page" else 1,
            )
        )
        blocks: list[ContentBlock] = []
        for _page_number, _task_type, page_outcome, _path in page_outcomes:
            blocks.extend(page_outcome.blocks)
        blocks.sort(
            key=lambda block: (
                int(block.page_number or 0),
                0 if block.source_type != "ocr" else 1,
                int(block.block_index),
            )
        )
        for block_index, block in enumerate(blocks):
            block.block_index = block_index
        digest_payload = "\n".join(
            (
                f"{block.page_number}|{block.source_type}|"
                f"{block.block_type}|{block.raw_text}"
            )
            for block in blocks
        ).encode("utf-8")
        merge_digest = hashlib.sha256(digest_payload).hexdigest()
        graph.confirm_merge(document_task_id, merge_digest)
        merged_status = (
            "partial_success"
            if any(
                page_outcome.status == "partial_success"
                for _page, _type, page_outcome, _path in page_outcomes
            )
            else "success"
        )
        merged = ParseOutcome(
            file_id=parent_job.file_id,
            file_path=parent_job.file_path,
            blocks=blocks,
            parser_name="pdf_task_graph",
            status=merged_status,
            alias_file_ids=parent_job.alias_file_ids,
            content_key=parent_job.content_key,
            parser_version=parent_job.parser_version,
            lane="pdf",
            size_bytes=parent_job.size_bytes,
            estimated_cost=parent_job.estimated_cost,
            parse_ms=sum(
                page_outcome.parse_ms
                for _page, _type, page_outcome, _path in page_outcomes
            ),
            normalize_ms=sum(
                page_outcome.normalize_ms
                for _page, _type, page_outcome, _path in page_outcomes
            ),
            content_hash_full=parent_job.content_hash_full,
        )
        for _page, _type, _page_outcome, spool_path in page_outcomes:
            spool_path.unlink(missing_ok=True)
            try:
                spool_path.parent.rmdir()
            except OSError:
                pass
        if current_metrics is not None:
            current_metrics.pdf_metrics["pdf_merge_ms"] = (
                current_metrics.pdf_metrics.get("pdf_merge_ms", 0)
                + int((time.perf_counter() - merge_started) * 1000)
            )
        return merged, parent_job

    def _flush_pdf_page_confirmations(
        self,
        document_task_id: int,
        graph: PdfTaskGraphRepository | None = None,
    ) -> None:
        confirmations = self._pdf_confirmation_buffers.pop(
            int(document_task_id),
            [],
        )
        if not confirmations:
            return
        (graph or PdfTaskGraphRepository(self.db)).confirm_page_tasks(
            confirmations
        )

    def _ensure_planning_runner(self, run_id: str) -> RecoverablePlanningRunner:
        runner = self._planning_runner
        if runner is not None:
            return runner
        base = TEMP_DIR / "process_results" / str(run_id)
        planning_control_dir = base / "planning_run_control"
        runner = RecoverablePlanningRunner(
            base / "planning_tasks",
            no_progress_timeout_seconds=max(
                1,
                int(self.settings.planning_no_progress_timeout_seconds or 300),
            ),
            startup_timeout_seconds=max(
                1,
                int(self.settings.planning_startup_timeout_seconds or 30),
            ),
            pause_control_dir=planning_control_dir,
        )
        self._planning_runner = runner
        self._planning_control_dir = planning_control_dir
        return runner

    def _fingerprint_paths_recoverable(
        self,
        requests: list[tuple[str, bool]],
        spool_dir: Path,
        token: CancelToken,
        progress_callback: (
            Callable[[PlanningProgress], None] | None
        ) = None,
    ) -> tuple[list[FingerprintSourceResult], list[tuple[Path, Exception]]]:
        if not requests:
            return [], []
        runner = self._planning_runner
        if runner is None:
            raise RuntimeError("recoverable planning runner is not initialized")
        try:
            result = runner.run(
                "content_hash",
                fingerprint_source_batch,
                requests,
                spool_dir,
                False,
                self._planning_control_dir,
                False,
                cancel_check=token.throw_if_cancelled,
                progress_callback=progress_callback,
            )
        except PlanningNoProgressError as exc:
            if len(requests) == 1:
                return [], [(Path(requests[0][0]), exc)]
            middle = max(1, len(requests) // 2)
            left_results, left_errors = self._fingerprint_paths_recoverable(
                requests[:middle],
                spool_dir,
                token,
                progress_callback,
            )
            right_results, right_errors = self._fingerprint_paths_recoverable(
                requests[middle:],
                spool_dir,
                token,
                progress_callback,
            )
            return [*left_results, *right_results], [*left_errors, *right_errors]
        if not isinstance(result, FingerprintBatchResult):
            raise PlanningWorkerError("content_hash 返回了无效的批量结果")
        errors = [
            (
                Path(item.path),
                OSError(f"{item.error_type}: {item.message}"),
            )
            for item in result.errors
        ]
        return list(result.results), errors

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
        normal_executor = _new_process_executor(
            self.settings,
            self.settings.parser_workers,
            spool_dir,
        )
        zip_executor = _new_process_executor(self.settings, self.settings.slow_file_workers, spool_dir)
        ocr_initial_workers = (
            max(1, int(self.settings.ocr_workers or 1))
            if self.settings.index_performance_preset == "fastest"
            else 1
        )
        ocr_executor = _new_process_executor(
            self.settings,
            ocr_initial_workers,
            spool_dir,
            persistent=True,
        )
        pdf_executor = _new_process_executor(
            self.settings,
            self.settings.pdf_parser_workers,
            spool_dir,
            persistent=bool(self.settings.enable_ocr and self.settings.ocr_scanned_pdf),
        )
        office_executor = _new_process_executor(
            self.settings,
            self.settings.process_parser_workers,
            spool_dir,
            persistent=bool(self.settings.enable_ocr and self.settings.ocr_scanned_pdf),
        )
        legacy_executors = {
            lane_name: _new_process_executor(
                self.settings,
                1,
                spool_dir,
                persistent=True,
            )
            for lane_name in (
                "legacy_word",
                "legacy_excel",
                "legacy_powerpoint",
            )
        }
        lanes = {
            "normal": ParseLane(
                "normal",
                normal_executor,
                submission_window(
                    self.settings.normal_pending_tasks,
                    self.settings.parser_workers,
                ),
                effective_lane_budget(self.settings, "normal"),
                process_based=True,
                worker_count=max(1, int(self.settings.parser_workers or 1)),
            ),
            "zip": ParseLane(
                "zip",
                zip_executor,
                submission_window(self.settings.slow_pending_tasks, self.settings.slow_file_workers),
                effective_lane_budget(self.settings, "zip"),
                process_based=True,
                worker_count=max(1, int(self.settings.slow_file_workers or 1)),
            ),
            "ocr": ParseLane(
                "ocr",
                ocr_executor,
                submission_window(
                    self.settings.ocr_pending_tasks,
                    ocr_initial_workers,
                ),
                effective_lane_budget(self.settings, "ocr"),
                process_based=True,
                persistent_process=True,
                worker_count=ocr_initial_workers,
            ),
            "pdf": ParseLane(
                "pdf",
                pdf_executor,
                submission_window(
                    self.settings.pdf_pending_tasks,
                    self.settings.pdf_parser_workers,
                ),
                effective_lane_budget(self.settings, "pdf"),
                process_based=True,
                persistent_process=bool(
                    self.settings.enable_ocr and self.settings.ocr_scanned_pdf
                ),
                worker_count=max(1, int(self.settings.pdf_parser_workers or 1)),
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
                persistent_process=bool(
                    self.settings.enable_ocr and self.settings.ocr_scanned_pdf
                ),
                worker_count=max(1, int(self.settings.process_parser_workers or 1)),
            ),
        }
        for lane_name, executor in legacy_executors.items():
            lanes[lane_name] = ParseLane(
                lane_name,
                executor,
                1,
                effective_lane_budget(self.settings, lane_name),
                process_based=True,
                persistent_process=True,
                worker_count=1,
            )
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
            pdf_executor,
            office_executor,
            *legacy_executors.values(),
        ]
        return lanes, executors, [
            normal_executor,
            zip_executor,
            ocr_executor,
            pdf_executor,
            office_executor,
            *legacy_executors.values(),
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
                    impacted.extend(
                        (lane, submitted_job)
                        for submitted_job in _submission_jobs(job)
                    )
                    lane.futures.discard(future)
                    lane.jobs.pop(future, None)
                    lane.inflight_bytes = max(
                        0,
                        lane.inflight_bytes - job_memory_bytes(job),
                    )
                    job.batch_jobs = ()
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
            representative_job = next(
                (
                    job
                    for _lane, job in impacted
                    if job.watchdog_timed_out
                ),
                impacted[0][1] if impacted else None,
            )
            representative_lane = next(
                (
                    lane
                    for lane, job in impacted
                    if job is representative_job
                ),
                group[0] if group else None,
            )
            if representative_job is not None:
                self._record_scheduler_diagnostic(
                    {
                        "state": "rebuilding_pool",
                        "source": "index_manager",
                        "lane": (
                            representative_lane.name
                            if representative_lane is not None
                            else ""
                        ),
                        "file": str(representative_job.file_path),
                        "phase": representative_job.progress_phase,
                        "cursor": representative_job.progress_cursor,
                        "retry_count": int(representative_job.retry_count),
                        "reason": (
                            "无进展 worker 已退出，正在创建并健康检查替代解析进程池"
                            if representative_job.watchdog_timed_out
                            else "解析进程池异常退出，正在创建并健康检查替代进程池"
                        ),
                    }
                )
            replacement = _new_process_executor(
                self.settings,
                workers,
                spool_dir,
                persistent=any(lane.persistent_process for lane in group),
            )
            with self._pause_lock:
                current_metrics = self._current_metrics
            if current_metrics is not None:
                current_metrics.hang_metrics["pool_rebuild_count"] = (
                    int(
                        current_metrics.hang_metrics.get(
                            "pool_rebuild_count",
                            0,
                        )
                    )
                    + 1
                )
                _record_eta_replay_control(
                    current_metrics,
                    "worker_recycle",
                    {lane.name: lane for lane in group},
                )
            for lane in group:
                lane.executor = replacement
            process_executors.append(replacement)
            executors.append(replacement)
            with self._executor_lock:
                self._active_process_executors.add(replacement)
            for lane, job in reversed(impacted):
                job_timed_out = bool(job.watchdog_timed_out)
                if job.task_id is not None:
                    self.db.mark_task_attempt_interrupted(
                        int(job.task_id),
                        (
                            "PARSE_NO_PROGRESS"
                            if job_timed_out
                            else "PROCESS_WORKER_CRASH"
                        ),
                        (
                            f"{job.progress_phase or '解析'} 阶段无进展，"
                            "worker 已退出且进程池正在重建"
                            if job_timed_out
                            else "解析 worker 异常退出，进程池已重建并将任务重新排队"
                        ),
                    )
                if job_timed_out:
                    checkpoint = load_partial_parse_checkpoint(job, spool_dir, consume=False)
                    max_retries = max(0, int(self.settings.no_progress_max_retries))
                    repeated_stall_count = register_stall(job)
                    if current_metrics is not None:
                        current_metrics.hang_metrics[
                            "same_stall_signature_count"
                        ] = max(
                            int(
                                current_metrics.hang_metrics.get(
                                    "same_stall_signature_count",
                                    0,
                                )
                            ),
                            repeated_stall_count,
                        )
                    if job.retry_count < max_retries and repeated_stall_count < 2:
                        job.retry_count += 1
                        if (
                            checkpoint is not None
                            and checkpoint.resume_cursor > 0
                            and job.parser_name in {
                                "pdf",
                                "zip",
                                "xlsx_stream",
                                "image_ocr",
                            }
                        ):
                            job.resume_cursor = checkpoint.resume_cursor
                            self._record_scheduler_diagnostic(
                                {
                                    "state": "checkpoint_resumed",
                                    "source": "index_manager",
                                    "lane": lane.name,
                                    "file": str(job.file_path),
                                    "phase": job.progress_phase,
                                    "cursor": job.progress_cursor,
                                    "retry_count": int(job.retry_count),
                                    "reason": (
                                        "已确认检查点有效，将从安全单位 "
                                        f"{checkpoint.resume_cursor} 继续"
                                    ),
                                }
                            )
                            if current_metrics is not None:
                                current_metrics.hang_metrics[
                                    "checkpoint_resume_count"
                                ] = (
                                    int(
                                        current_metrics.hang_metrics.get(
                                            "checkpoint_resume_count",
                                            0,
                                        )
                                    )
                                    + 1
                                )
                                current_metrics.hang_metrics[
                                    "checkpoint_resume_units_avoided"
                                ] = (
                                    int(
                                        current_metrics.hang_metrics.get(
                                            "checkpoint_resume_units_avoided",
                                            0,
                                        )
                                    )
                                    + int(checkpoint.resume_cursor)
                                )
                        else:
                            partial_parse_path(job, spool_dir).unlink(missing_ok=True)
                            job.resume_cursor = 0
                        reset_job_for_retry(job, now)
                        lane.pending.appendleft(job)
                        continue
                    reason_code = "PARSE_NO_PROGRESS"
                    reason_text = (
                        f"在“{job.progress_phase or '解析'}”阶段连续 "
                        f"{max(1, repeated_stall_count)} 次停在同一安全游标且无有效进展，"
                        f"最近一次等待 {max(1, job.timeout_seconds)} 秒；完整索引尚未发布"
                    )
                    self._record_scheduler_diagnostic(
                        {
                            "state": "same_stall_retry_stopped",
                            "source": "index_manager",
                            "lane": lane.name,
                            "file": str(job.file_path),
                            "phase": job.progress_phase,
                            "cursor": job.progress_cursor,
                            "retry_count": int(job.retry_count),
                            "reason": reason_text,
                        }
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

    def _resize_idle_ocr_process_lane(
        self,
        lanes: dict[str, ParseLane],
        executors: list[Executor],
        process_executors: list[ProcessPoolExecutor],
        spool_dir: Path,
        *,
        target_workers: int,
        reason: str,
    ) -> tuple[bool, str]:
        """Replace the persistent OCR pool only at a confirmed idle boundary."""

        lane = lanes.get("ocr")
        if (
            lane is None
            or not lane.process_based
            or not isinstance(lane.executor, ProcessPoolExecutor)
        ):
            return False, "ocr_lane_unavailable"
        target = max(
            1,
            min(
                int(self.settings.ocr_workers or 1),
                int(target_workers or 1),
            ),
        )
        if target == lane.worker_count:
            lane.max_in_flight = min(
                target,
                max(1, int(lane.max_in_flight)),
            )
            return True, "unchanged"
        if lane.futures or lane.jobs or lane.inflight_bytes:
            return False, "active_tasks"

        old_executor = lane.executor
        old_workers = int(lane.worker_count)
        candidate: ProcessPoolExecutor | None = None
        try:
            candidate = _new_process_executor(
                self.settings,
                target,
                spool_dir,
                persistent=True,
            )
            if candidate.submit(_pool_health_probe).result(timeout=10) is not True:
                raise RuntimeError(
                    "OCR candidate pool health probe failed"
                )
        except Exception as exc:
            if candidate is not None:
                terminate_process_pool_workers(candidate)
                candidate.shutdown(
                    wait=False,
                    cancel_futures=True,
                )
            self._record_resource_pool_resize(
                old_workers=old_workers,
                new_workers=target,
                reason=reason,
                success=False,
                rollback_reason=f"{type(exc).__name__}: {exc}",
            )
            return False, "candidate_pool_failed"

        lane.executor = candidate
        lane.worker_count = target
        lane.max_in_flight = target
        if old_executor in executors:
            executors.remove(old_executor)
        if old_executor in process_executors:
            process_executors.remove(old_executor)
        executors.append(candidate)
        process_executors.append(candidate)
        with self._executor_lock:
            self._active_process_executors.discard(old_executor)
            self._active_process_executors.add(candidate)
        try:
            old_executor.shutdown(
                wait=True,
                cancel_futures=True,
            )
        except Exception:
            logger.debug(
                "Unable to close idle OCR pool after resize",
                exc_info=True,
            )
        self._record_resource_pool_resize(
            old_workers=old_workers,
            new_workers=target,
            reason=reason,
            success=True,
            rollback_reason="",
        )
        return True, "resized"

    def _record_resource_pool_resize(
        self,
        *,
        old_workers: int,
        new_workers: int,
        reason: str,
        success: bool,
        rollback_reason: str,
    ) -> None:
        with self._pause_lock:
            metrics = self._current_metrics
        if metrics is None:
            return
        sample = self._latest_resource_sample
        metrics.resource_metrics["ocr_pool_resize_count"] = (
            int(
                metrics.resource_metrics.get(
                    "ocr_pool_resize_count",
                    0,
                )
            )
            + 1
        )
        if not success:
            metrics.resource_metrics[
                "ocr_pool_resize_failure_count"
            ] = (
                int(
                    metrics.resource_metrics.get(
                        "ocr_pool_resize_failure_count",
                        0,
                    )
                )
                + 1
            )
        event: dict[str, object] = {
                "timestamp": time.time(),
                "type": "ocr_pool_resize",
                "from_workers": int(old_workers),
                "to_workers": int(new_workers),
                "reason": str(reason),
                "success": bool(success),
                "rollback_reason": str(rollback_reason),
                "safe_boundary": True,
                "cpu_percent": float(
                    sample.total_cpu_percent if sample is not None else 0.0
                ),
                "app_cpu_percent": float(
                    sample.app_cpu_percent if sample is not None else 0.0
                ),
                "rss_bytes": int(
                    sample.app_rss_bytes if sample is not None else 0
                ),
                "memory_available_bytes": int(
                    sample.memory_available_bytes
                    if sample is not None
                    else 0
                ),
                "disk_busy_percent": float(
                    sample.disk_busy_percent if sample is not None else 0.0
                ),
                "network_read_latency_ms": float(
                    sample.network_read_latency_ms
                    if sample is not None
                    else 0.0
                ),
                "queue_depth": int(
                    sample.queue_depth if sample is not None else 0
                ),
                "active_tasks": int(
                    sample.active_tasks if sample is not None else 0
                ),
                "writer_queue_depth": int(
                    sample.writer_queue_depth
                    if sample is not None
                    else 0
                ),
                "worker_rss_bytes": {
                    str(pid): int(rss)
                    for pid, rss in (
                        sample.worker_rss_bytes.items()
                        if sample is not None
                        else ()
                    )
                },
                "cooldown_seconds": float(
                    self._resource_resize_cooldown_seconds
                ),
            }
        metrics.profile_transitions.append(event)
        if not success:
            metrics.fallback_and_throttle_events.append(dict(event))

    def _outcome_from_result(
        self,
        job: ParseJob,
        result: ParseResult,
        spool_dir: Path,
    ) -> ParseOutcome:
        descriptor = result if isinstance(result, SpoolParseResult) else None
        if (
            descriptor is not None
            and job.task_id is not None
            and job.pdf_task_type not in {
                "pdf_native_page",
                "pdf_ocr_page",
            }
        ):
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

    def _record_scheduler_diagnostic(
        self,
        payload: dict[str, object],
    ) -> None:
        with self._diagnostic_lock:
            self._diagnostic_sequence += 1
            self._diagnostic_events.append(
                {
                    **dict(payload),
                    "sequence": self._diagnostic_sequence,
                    "recorded_at_epoch": time.time(),
                }
            )

    def _emit_pending_scheduler_diagnostics(
        self,
        progress_callback: ProgressCallback | None,
        summary: IndexSummary,
        lanes: dict[str, ParseLane],
        estimator: IndexTimeEstimator,
        total_files: int,
    ) -> None:
        events = self._take_scheduler_diagnostics()
        for event in events:
            self._emit_active_progress(
                progress_callback,
                summary,
                lanes,
                estimator,
                total_files,
                diagnostic_state=str(event.get("state") or ""),
                diagnostic_reason=str(event.get("reason") or ""),
                diagnostic_source=str(event.get("source") or ""),
                diagnostic_lane=str(event.get("lane") or ""),
                diagnostic_phase=str(event.get("phase") or ""),
                diagnostic_cursor=str(event.get("cursor") or ""),
                diagnostic_sequence=int(event.get("sequence") or 0),
            )

    def _take_scheduler_diagnostics(self) -> list[dict[str, object]]:
        with self._diagnostic_lock:
            events = list(self._diagnostic_events)
            self._diagnostic_events.clear()
        return events

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
        with self._pause_lock:
            current_metrics = self._current_metrics
        if current_metrics is not None:
            current_metrics.hang_metrics[
                "last_semantic_progress_age_seconds"
            ] = int(snapshot["no_progress_seconds"])
            eta_metrics = current_metrics.eta_metrics
            at_seconds = max(
                0.0,
                time.perf_counter()
                - float(
                    eta_metrics.get(
                        "_run_started_monotonic",
                        time.perf_counter(),
                    )
                ),
            )
            is_completion = bool(extra.get("completed_queue"))
            last_trace_at = float(
                eta_metrics.get("_last_trace_at_seconds") or -10.0
            )
            if is_completion or at_seconds - last_trace_at >= 10.0:
                replay_events = eta_metrics.setdefault("replay_events", [])
                if isinstance(replay_events, list):
                    replay_events.append(
                        {
                            "at_seconds": round(at_seconds, 6),
                            "event_type": (
                                "completion"
                                if is_completion
                                else "progress"
                            ),
                            "remaining_cost_by_lane": dict(
                                snapshot["remaining_cost_by_lane"]
                            ),
                            "active_elapsed_by_lane": dict(
                                snapshot["active_elapsed_by_lane"]
                            ),
                            "workers_by_lane": {
                                name: lane.worker_count
                                for name, lane in lanes.items()
                            },
                            "lane": str(
                                extra.get("completed_queue") or ""
                            ),
                            "completed_cost": float(
                                extra.get("completed_cost") or 0.0
                            ),
                            "service_seconds": float(
                                extra.get("service_seconds") or 0.0
                            ),
                            "eta_seconds": (
                                int(estimate.seconds) if estimate else 0
                            ),
                            "eta_ready": bool(
                                estimate is not None and estimate.ready
                            ),
                        }
                    )
                eta_metrics["_last_trace_at_seconds"] = at_seconds
            if estimate is not None and estimate.ready:
                if float(
                    eta_metrics.get("eta_first_ready_seconds") or 0
                ) <= 0:
                    eta_metrics["eta_first_ready_seconds"] = round(
                        time.perf_counter()
                        - float(
                            eta_metrics.get(
                                "_run_started_monotonic",
                                time.perf_counter(),
                            )
                        ),
                        3,
                    )
                eta_metrics["eta_update_count"] = (
                    int(eta_metrics.get("eta_update_count") or 0) + 1
                )
                previous_eta = int(eta_metrics.get("_last_eta_seconds") or 0)
                if previous_eta and abs(estimate.seconds - previous_eta) > max(
                    30,
                    int(previous_eta * 0.25),
                ):
                    eta_metrics["eta_jump_count"] = (
                        int(eta_metrics.get("eta_jump_count") or 0) + 1
                    )
                eta_metrics["_last_eta_seconds"] = int(estimate.seconds)
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
            representative_is_slowest=snapshot[
                "representative_is_slowest"
            ],
            other_active_lane_count=snapshot["other_active_lane_count"],
            other_recent_progress_seconds=snapshot[
                "other_recent_progress_seconds"
            ],
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
            eta_seconds=(estimate.seconds if estimate else 0),
            eta_ready=(estimate.ready if estimate else False),
            eta_confidence=(estimate.confidence if estimate else 0.0),
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
    return sum(
        len(lane.pending)
        + sum(
            len(_submission_jobs(job))
            for job in lane.jobs.values()
        )
        for lane in lanes
    )


def _apply_resource_decision(
    lanes: dict[str, ParseLane],
    baseline_limits: dict[str, int],
    decision: ResourceDecision,
    sample: ResourceSample,
    metrics: IndexRunMetrics,
) -> None:
    metrics.resource_metrics["sample_count"] = (
        int(metrics.resource_metrics.get("sample_count") or 0) + 1
    )
    for key, value in (
        ("peak_total_cpu_percent", float(sample.total_cpu_percent)),
        ("peak_app_cpu_percent", float(sample.app_cpu_percent)),
        ("peak_rss_bytes", int(sample.app_rss_bytes)),
        ("peak_disk_busy_percent", float(sample.disk_busy_percent)),
        (
            "peak_network_read_latency_ms",
            float(sample.network_read_latency_ms),
        ),
        ("max_queue_depth", int(sample.queue_depth)),
        ("max_active_tasks", int(sample.active_tasks)),
        ("max_ocr_pending_pixels", int(sample.ocr_pending_pixels)),
        ("max_writer_queue_depth", int(sample.writer_queue_depth)),
    ):
        metrics.resource_metrics[key] = max(
            float(metrics.resource_metrics.get(key) or 0),
            float(value),
        )
    metrics.resource_metrics["latest_worker_rss_bytes"] = {
        str(pid): int(rss)
        for pid, rss in sample.worker_rss_bytes.items()
    }
    previous_available = int(
        metrics.resource_metrics.get(
            "minimum_memory_available_bytes",
            0,
        )
        or 0
    )
    metrics.resource_metrics[
        "minimum_memory_available_bytes"
    ] = (
        int(sample.memory_available_bytes)
        if previous_available <= 0
        else min(
            previous_available,
            int(sample.memory_available_bytes),
        )
    )
    if decision.target_ocr_inflight is not None and "ocr" in lanes:
        lanes["ocr"].max_in_flight = max(
            1,
            min(
                lanes["ocr"].worker_count,
                int(decision.target_ocr_inflight),
            ),
        )
    for lane_name in (
        "normal",
        "zip",
        "pdf",
        "office_process",
        "legacy_word",
        "legacy_excel",
        "legacy_powerpoint",
    ):
        lane = lanes.get(lane_name)
        if lane is None:
            continue
        baseline = max(1, int(baseline_limits.get(lane_name, 1)))
        lane.max_in_flight = max(
            1,
            min(
                lane.worker_count,
                int(round(baseline * decision.target_read_inflight_scale)),
            ),
        )
    if not decision.changed and decision.target_ocr_inflight is None:
        return
    event = {
        "timestamp": float(sample.timestamp),
        "state": decision.state,
        "reason": decision.reason,
        "ocr_inflight": int(lanes["ocr"].max_in_flight),
        "cpu_percent": float(sample.total_cpu_percent),
        "app_cpu_percent": float(sample.app_cpu_percent),
        "rss_bytes": int(sample.app_rss_bytes),
        "memory_available_bytes": int(sample.memory_available_bytes),
        "disk_busy_percent": float(sample.disk_busy_percent),
        "network_read_latency_ms": float(
            sample.network_read_latency_ms
        ),
        "queue_depth": int(sample.queue_depth),
        "active_tasks": int(sample.active_tasks),
        "writer_queue_depth": int(sample.writer_queue_depth),
        "worker_rss_bytes": {
            str(pid): int(rss)
            for pid, rss in sample.worker_rss_bytes.items()
        },
        "read_inflight_scale": float(
            decision.target_read_inflight_scale
        ),
        "allow_active_tasks_to_finish": bool(
            decision.allow_active_tasks_to_finish
        ),
    }
    metrics.profile_transitions.append(event)
    if decision.state in {
        "memory_pressure",
        "cpu_pressure",
        "io_pressure",
        "recovery_cooldown",
    }:
        metrics.fallback_and_throttle_events.append(event)


def accumulate_required_ocr_block_metrics(
    metrics: IndexRunMetrics,
    block: ContentBlock,
) -> None:
    """Map one confirmed OCR block onto the public phase-two metric names."""

    extra = block.extra

    def add(
        section: dict[str, object],
        key: str,
        value: object,
    ) -> None:
        section[key] = int(section.get(key) or 0) + max(
            0,
            int(value or 0),
        )

    exact_cache_hits = int(
        extra.get("ocr_exact_cache_hits") or 0
    )
    add(
        metrics.ocr_metrics,
        "ocr_crop_cache_hits",
        exact_cache_hits,
    )
    add(
        metrics.ocr_metrics,
        "ocr_embedded_image_cache_hits",
        extra.get("ocr_embedded_image_cache_hits"),
    )

    preview_calls = int(extra.get("preview_detect_calls") or 0)
    tile_calls = int(extra.get("tiles_processed") or 0)
    detect_requests = int(
        extra.get("detect_requests")
        or (preview_calls + tile_calls)
    )
    detect_calls = int(
        extra.get("detect_inference_calls")
        or (preview_calls + tile_calls)
    )
    detect_pixels = int(
        extra.get("detect_pixels")
        or (
            int(extra.get("preview_detect_pixels") or 0)
            + int(extra.get("fallback_region_pixels") or 0)
        )
    )
    recognize_requests = int(
        extra.get("recognize_requests")
        or (
            int(extra.get("first_pass_regions") or 0)
            + int(extra.get("tile_regions_recognized") or 0)
        )
    )
    recognize_calls = int(
        extra.get("recognize_inference_calls")
        or extra.get("recognizer_batches")
        or 0
    )
    recognize_pixels = int(
        extra.get("recognize_pixels")
        or extra.get("original_region_pixels")
        or 0
    )
    add(
        metrics.ocr_metrics,
        "ocr_detect_requests",
        detect_requests,
    )
    add(
        metrics.ocr_metrics,
        "ocr_detect_calls",
        detect_calls,
    )
    add(
        metrics.ocr_metrics,
        "ocr_detect_pixels",
        detect_pixels,
    )
    add(
        metrics.ocr_metrics,
        "ocr_recognize_requests",
        recognize_requests,
    )
    add(
        metrics.ocr_metrics,
        "ocr_recognize_calls",
        recognize_calls,
    )
    add(
        metrics.ocr_metrics,
        "ocr_recognize_pixels",
        recognize_pixels,
    )
    add(
        metrics.ocr_metrics,
        "ocr_adaptive_split_count",
        extra.get("adaptive_regions_split"),
    )
    metrics.ocr_metrics["ocr_unresolved_regions_peak"] = max(
        int(
            metrics.ocr_metrics.get(
                "ocr_unresolved_regions_peak",
                0,
            )
            or 0
        ),
        int(extra.get("adaptive_regions_remaining_peak") or 0),
    )
    add(
        metrics.ocr_metrics,
        "ocr_regions_resumed",
        max(
            int(extra.get("checkpoint_regions_reused") or 0),
            int(
                extra.get(
                    "checkpoint_recognition_batches_reused"
                )
                or 0
            ),
        ),
    )
    if block.block_type != "pdf_page_ocr":
        return
    add(
        metrics.ocr_metrics,
        "ocr_page_cache_hits",
        exact_cache_hits,
    )
    add(
        metrics.pdf_metrics,
        "pdf_ocr_pages_completed",
        1,
    )
    add(
        metrics.pdf_metrics,
        "pdf_page_cache_hits",
        exact_cache_hits,
    )
    add(
        metrics.pdf_metrics,
        "pdf_full_page_fallback_count",
        1 if extra.get("pdf_full_page_fallback") else 0,
    )
    add(
        metrics.pdf_metrics,
        "pdf_200dpi_region_count",
        extra.get("pdf_detected_regions"),
    )
    add(
        metrics.pdf_metrics,
        "pdf_300dpi_upgrade_region_count",
        extra.get("pdf_upgraded_regions"),
    )


def lane_costs(jobs: Iterable[ParseJob]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for job in jobs:
        costs[job.lane] = costs.get(job.lane, 0.0) + max(0.0, job.estimated_cost)
    return costs


def _record_eta_replay_control(
    metrics: IndexRunMetrics,
    event_type: str,
    lanes: Mapping[str, ParseLane],
    *,
    mode: str = "",
) -> None:
    eta_metrics = metrics.eta_metrics
    started = float(
        eta_metrics.get("_run_started_monotonic")
        or time.perf_counter()
    )
    remaining: dict[str, float] = {}
    active_elapsed: dict[str, float] = {}
    now = time.perf_counter()
    for name, lane in lanes.items():
        remaining[name] = sum(
            max(0.0, job.estimated_cost)
            for job in lane.pending
        ) + sum(
            sum(
                max(0.0, submitted_job.estimated_cost)
                for submitted_job in _submission_jobs(job)
            )
            for job in lane.jobs.values()
        )
        elapsed = [
            max(0.0, now - submitted_job.started_monotonic)
            for job in lane.jobs.values()
            for submitted_job in _submission_jobs(job)
            if submitted_job.started_monotonic
        ]
        if elapsed:
            active_elapsed[name] = max(elapsed)
    replay_events = eta_metrics.setdefault("replay_events", [])
    if isinstance(replay_events, list):
        replay_events.append(
            {
                "at_seconds": round(max(0.0, now - started), 6),
                "event_type": str(event_type),
                "remaining_cost_by_lane": remaining,
                "active_elapsed_by_lane": active_elapsed,
                "workers_by_lane": {
                    name: lane.worker_count
                    for name, lane in lanes.items()
                },
                "mode": str(mode),
            }
        )


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
            sum(
                max(0.0, submitted_job.estimated_cost)
                for submitted_job in _submission_jobs(job)
            )
            for job in lane.jobs.values()
        )
        if first_pending is None and lane.pending:
            first_pending = (lane.name, lane.pending[0])
        for future, job in list(lane.jobs.items()):
            if future.done():
                continue
            for active_job in _submission_jobs(job):
                if not active_job.started_monotonic:
                    continue
                elapsed = max(
                    0.0,
                    now - active_job.started_monotonic,
                )
                active_count += 1
                active_elapsed[lane.name] = max(
                    active_elapsed.get(lane.name, 0.0),
                    elapsed,
                )
                active_jobs.append(
                    (elapsed, lane.name, active_job)
                )
    if active_jobs:
        elapsed, lane_name, current_job = max(active_jobs, key=lambda item: item[0])
    elif first_pending is not None:
        lane_name, current_job = first_pending
        elapsed = 0.0
    else:
        lane_name, current_job, elapsed = "", None, 0.0
    other_lane_jobs = [
        job
        for _elapsed, active_lane, job in active_jobs
        if active_lane != lane_name
    ]
    other_active_lane_count = len(
        {
            active_lane
            for _elapsed, active_lane, _job in active_jobs
            if active_lane != lane_name
        }
    )
    current_no_progress_seconds = (
        int(
            max(
                0.0,
                now
                - (
                    current_job.last_progress_monotonic
                    or current_job.started_monotonic
                ),
            )
        )
        if current_job is not None and current_job.started_monotonic
        else 0
    )
    other_recent_progress_seconds = (
        int(
            min(
                max(
                    0.0,
                    now
                    - (
                        job.last_progress_monotonic
                        or job.started_monotonic
                        or now
                    ),
                )
                for job in other_lane_jobs
            )
        )
        if other_lane_jobs
        else 0
    )
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
        "no_progress_seconds": current_no_progress_seconds,
        "retry_count": current_job.retry_count if current_job is not None else 0,
        "representative_is_slowest": bool(
            current_job is not None
            and current_no_progress_seconds >= 10
            and other_active_lane_count > 0
        ),
        "other_active_lane_count": other_active_lane_count,
        "other_recent_progress_seconds": other_recent_progress_seconds,
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
    *,
    persistent: bool = False,
) -> ProcessPoolExecutor:
    max_tasks = (
        None
        if persistent
        else max(
            1,
            int(
                settings.process_max_tasks_per_child
                or settings.process_recycle_max_tasks
                or 32
            ),
        )
    )
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

        if should_lower_process_priority(settings):
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


def should_lower_process_priority(settings: AppSettings) -> bool:
    return settings.index_performance_preset != "fastest"


def submission_window(configured: int, workers: int) -> int:
    """Bound executor prefetch while the lane deque owns remaining work."""

    worker_count = max(1, int(workers or 1))
    return max(1, min(int(configured or 1), worker_count * 2))


def _ocr_source_key(job: ParseJob) -> str:
    if job.pdf_source_digest:
        return f"pdf:{job.pdf_source_digest}"
    if job.archive_path is not None:
        return f"zip:{job.archive_path}:{job.archive_internal_path}"
    return f"file:{job.file_id}"


def _submission_jobs(job: ParseJob) -> tuple[ParseJob, ...]:
    return (job, *job.batch_jobs)


def _spool_result_key(
    file_id: int,
    task_id: int | None,
) -> tuple[str, int]:
    if task_id is not None:
        return "task", int(task_id)
    return "file", int(file_id)


def _claim_ocr_batch_extras(
    lane: ParseLane,
    leader: ParseJob,
    settings: AppSettings,
    *,
    maximum_bytes: int,
) -> tuple[ParseJob, ...]:
    limit = max(1, int(settings.ocr_microbatch_parent_jobs))
    if limit <= 1 or not lane.pending:
        return ()
    candidates = list(lane.pending)
    selected: list[ParseJob] = []
    used_bytes = job_memory_bytes(leader, include_batch=False)
    source_counts = {_ocr_source_key(leader): 1}
    # Prefer a different source first, then allow a second page from one PDF.
    for maximum_per_source in (1, 2):
        for candidate in candidates:
            if candidate in selected or len(selected) >= limit - 1:
                continue
            source_key = _ocr_source_key(candidate)
            if source_counts.get(source_key, 0) >= maximum_per_source:
                continue
            candidate_bytes = job_memory_bytes(
                candidate,
                include_batch=False,
            )
            if used_bytes + candidate_bytes > maximum_bytes:
                continue
            selected.append(candidate)
            used_bytes += candidate_bytes
            source_counts[source_key] = (
                source_counts.get(source_key, 0) + 1
            )
        if len(selected) >= limit - 1:
            break
    selected_ids = {id(candidate) for candidate in selected}
    lane.pending = deque(
        candidate
        for candidate in lane.pending
        if id(candidate) not in selected_ids
    )
    return tuple(selected)


def _claim_pdf_page_batch_extras(
    lane: ParseLane,
    leader: ParseJob,
    settings: AppSettings,
    *,
    maximum_bytes: int,
) -> tuple[ParseJob, ...]:
    limit = max(1, int(settings.pdf_page_batch_size or 1))
    if (
        limit <= 1
        or not lane.pending
        or leader.pdf_task_type != "pdf_native_page"
        or leader.pdf_document_task_id is None
        or leader.pdf_page_number is None
        or leader.validation_hang_stage
    ):
        return ()
    selected: list[ParseJob] = []
    used_bytes = job_memory_bytes(leader, include_batch=False)
    expected_page = int(leader.pdf_page_number) + 1
    for candidate in list(lane.pending):
        if len(selected) >= limit - 1:
            break
        if (
            candidate.pdf_task_type != "pdf_native_page"
            or candidate.pdf_document_task_id != leader.pdf_document_task_id
            or candidate.file_id != leader.file_id
            or candidate.validation_hang_stage
            or int(candidate.pdf_page_number or 0) != expected_page
        ):
            continue
        candidate_bytes = job_memory_bytes(
            candidate,
            include_batch=False,
        )
        if used_bytes + candidate_bytes > maximum_bytes:
            break
        selected.append(candidate)
        used_bytes += candidate_bytes
        expected_page += 1
    selected_ids = {id(candidate) for candidate in selected}
    lane.pending = deque(
        candidate
        for candidate in lane.pending
        if id(candidate) not in selected_ids
    )
    return tuple(selected)


def _pause_job_key(job: ParseJob) -> int:
    return int(job.task_id or job.file_id)


def _validation_safe_point_delay_ms() -> int:
    """Return the opt-in deterministic delay used only by validation commands."""

    value = os.environ.get("LFTS_VALIDATION_SAFE_POINT_DELAY_MS", "")
    if not value:
        return 0
    try:
        return max(0, min(5_000, int(value)))
    except ValueError:
        return 0


def schedule_parse_lanes(
    lanes: Iterable[ParseLane],
    settings: AppSettings,
    token: CancelToken,
    spool_dir: Path,
    *,
    metrics: IndexRunMetrics | None = None,
) -> list[int | None]:
    submitted_task_ids: list[int | None] = []
    lane_list = list(lanes)
    global_budget = max(128, int(settings.index_memory_budget_mb)) * 1024 * 1024
    cpu_budget = max(1, int(settings.index_cpu_token_budget or os.cpu_count() or 1))
    global_inflight = sum(lane.inflight_bytes for lane in lane_list)
    global_cpu_tokens = sum(
        cpu_tokens_for_job(job, settings)
        for lane in lane_list
        for job in lane.jobs.values()
    )
    if token.paused:
        return submitted_task_ids
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
            if token.paused:
                return submitted_task_ids
            token.throw_if_cancelled()
            job = lane.pending[0]
            job_bytes = job_memory_bytes(job)
            job_cpu_tokens = cpu_tokens_for_job(job, settings)
            if lane.futures and lane.inflight_bytes + job_bytes > lane.max_inflight_bytes:
                break
            if global_inflight and global_inflight + job_bytes > global_budget:
                break
            if global_cpu_tokens and global_cpu_tokens + job_cpu_tokens > cpu_budget:
                break
            lane.pending.popleft()
            job.batch_jobs = ()
            if lane.name == "ocr" and lane.process_based:
                remaining_lane_bytes = max(
                    job_bytes,
                    lane.max_inflight_bytes - lane.inflight_bytes,
                )
                remaining_global_bytes = max(
                    job_bytes,
                    global_budget - global_inflight,
                )
                job.batch_jobs = _claim_ocr_batch_extras(
                    lane,
                    job,
                    settings,
                    maximum_bytes=min(
                        remaining_lane_bytes,
                        remaining_global_bytes,
                    ),
                )
                job_bytes = job_memory_bytes(job)
            elif (
                lane.name == "pdf"
                and lane.process_based
                and job.pdf_task_type == "pdf_native_page"
            ):
                remaining_lane_bytes = max(
                    job_bytes,
                    lane.max_inflight_bytes - lane.inflight_bytes,
                )
                remaining_global_bytes = max(
                    job_bytes,
                    global_budget - global_inflight,
                )
                job.batch_jobs = _claim_pdf_page_batch_extras(
                    lane,
                    job,
                    settings,
                    maximum_bytes=min(
                        remaining_lane_bytes,
                        remaining_global_bytes,
                    ),
                )
                job_bytes = job_memory_bytes(job)
                batch_pages = len(_submission_jobs(job))
                if metrics is not None:
                    metrics.pdf_metrics["pdf_dispatch_batch_count"] = (
                        int(metrics.pdf_metrics.get("pdf_dispatch_batch_count", 0))
                        + 1
                    )
                    metrics.pdf_metrics["pdf_dispatched_page_count"] = (
                        int(metrics.pdf_metrics.get("pdf_dispatched_page_count", 0))
                        + batch_pages
                    )
                    metrics.pdf_metrics["pdf_max_batch_pages"] = max(
                        int(metrics.pdf_metrics.get("pdf_max_batch_pages", 0)),
                        batch_pages,
                    )
            started_at = time.perf_counter()
            for submitted_job in _submission_jobs(job):
                submitted_job.started_monotonic = started_at
                submitted_job.pdf_confirmation_batch_end = True
                submitted_job.watchdog_timed_out = False
                submitted_job.last_progress_monotonic = started_at
                submitted_job.timeout_seconds = no_progress_timeout(
                    settings,
                    submitted_job,
                )
            try:
                if lane.process_based:
                    worker = (
                        parse_ocr_batch_process_worker
                        if lane.name == "ocr" and job.batch_jobs
                        else (
                            parse_pdf_batch_process_worker
                            if lane.name == "pdf" and job.batch_jobs
                            else parse_file_process_worker
                        )
                    )
                    future = lane.executor.submit(
                        worker,
                        job,
                        settings,
                        spool_dir,
                    )
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
            global_cpu_tokens += job_cpu_tokens
            submitted_task_ids.extend(
                submitted_job.task_id
                for submitted_job in _submission_jobs(job)
            )
    return submitted_task_ids


def cpu_tokens_for_job(job: ParseJob, settings: AppSettings) -> int:
    if job.lane == "ocr" or (
        job.lane == "pdf"
        and not job.pdf_task_type
        and settings.enable_ocr
        and settings.ocr_scanned_pdf
    ):
        return max(1, int(settings.ocr_cpu_threads or 1))
    if job.file_path.suffix.lower() in {".xlsx", ".xlsm"}:
        return max(1, int(settings.xlsx_sheet_workers or 1))
    return 1


def job_memory_bytes(
    job: ParseJob,
    *,
    include_batch: bool = True,
) -> int:
    own_bytes = max(
        1,
        int(job.memory_estimate_bytes or job.size_bytes or 1),
    )
    if not include_batch or not job.batch_jobs:
        return own_bytes
    return own_bytes + sum(
        job_memory_bytes(child, include_batch=False)
        for child in job.batch_jobs
    )


def estimate_job_memory_bytes(
    file_path: Path,
    size_bytes: int,
    relevant_bytes: int = 0,
) -> int:
    suffix = file_path.suffix.lower()
    source_bytes = max(1, int(size_bytes or 0), int(relevant_bytes or 0))
    if suffix in {".xlsx", ".xlsm"}:
        return min(2 * 1024 * 1024 * 1024, source_bytes * 8)
    if suffix in IMAGE_EXTENSIONS:
        return min(1024 * 1024 * 1024, source_bytes * 4)
    if suffix == ".pdf":
        return min(1024 * 1024 * 1024, source_bytes * 2)
    return source_bytes


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
        token.wait_if_paused()
        token.throw_if_cancelled()
        return []
    if block:
        done, _ = wait(futures, timeout=0.2, return_when=FIRST_COMPLETED)
    else:
        done = {future for future in futures if future.done()}
    if not done:
        if token.paused:
            return []
        token.throw_if_cancelled()
        return []
    completed: list[tuple[str, ParseJob, ParseResult, int | None]] = []
    for lane in lane_list:
        lane_done = lane.futures.intersection(done)
        lane.futures.difference_update(lane_done)
        for future in lane_done:
            token.throw_if_cancelled()
            job = lane.jobs.pop(future)
            submitted_jobs = _submission_jobs(job)
            lane.inflight_bytes = max(
                0,
                lane.inflight_bytes - job_memory_bytes(job),
            )
            descriptor_bytes: int | None = None
            try:
                result = future.result()
                if isinstance(result, list):
                    descriptors = {
                        _spool_result_key(
                            descriptor.file_id,
                            descriptor.task_id,
                        ): descriptor
                        for descriptor in result
                    }
                    missing = [
                        int(submitted.task_id or submitted.file_id)
                        for submitted in submitted_jobs
                        if _spool_result_key(
                            submitted.file_id,
                            submitted.task_id,
                        ) not in descriptors
                    ]
                    if missing:
                        raise RuntimeError(
                            "Parser batch omitted result(s): "
                            + ",".join(str(value) for value in missing)
                        )
                    for submitted in submitted_jobs:
                        descriptor = descriptors[
                            _spool_result_key(
                                submitted.file_id,
                                submitted.task_id,
                            )
                        ]
                        item_descriptor_bytes = len(
                            pickle.dumps(
                                descriptor,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
                        )
                        completed.append(
                            (
                                lane.name,
                                submitted,
                                descriptor,
                                item_descriptor_bytes,
                            )
                        )
                    if (
                        lane.name == "pdf"
                        and len(submitted_jobs) > 1
                        and all(
                            submitted.pdf_task_type
                            == "pdf_native_page"
                            for submitted in submitted_jobs
                        )
                    ):
                        for submitted in submitted_jobs[:-1]:
                            submitted.pdf_confirmation_batch_end = False
                    continue
                if isinstance(result, SpoolParseResult):
                    descriptor_bytes = len(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
            except CancelledError:
                raise
            except Exception as exc:
                if token.cancelled:
                    raise CancelledError("任务已取消") from exc
                logger.error(
                    "Parse lane %s failed for %s\n%s",
                    lane.name,
                    job.file_path,
                    traceback.format_exc(),
                )
                for submitted in submitted_jobs:
                    if submitted.watchdog_timed_out:
                        failed_result: ParseResult = _diagnostic_outcome(
                            submitted,
                            "process_worker",
                            "failed_retryable",
                            "PARSE_NO_PROGRESS",
                            (
                                f"“{submitted.progress_phase or '解析'}”阶段连续 "
                                f"{max(1, submitted.timeout_seconds)} 秒无有效进展"
                            ),
                            submitted.started_monotonic
                            or time.perf_counter(),
                        )
                    else:
                        failed_result = failed_parse_outcome(
                            submitted,
                            exc,
                        )
                    completed.append(
                        (
                            lane.name,
                            submitted,
                            failed_result,
                            None,
                        )
                    )
                continue
            completed.append((lane.name, job, result, descriptor_bytes))
    return completed


def parse_file_worker(job: ParseJob, settings: AppSettings, cancel_token: CancelToken) -> ParseOutcome:
    return parse_file_with_registry(job, worker_registry(settings), cancel_token, settings)


def parse_file_process_worker(job: ParseJob, settings: AppSettings, spool_dir: Path) -> SpoolParseResult:
    registry = _process_registry or ParserRegistry(settings)
    return _parse_file_process_with_registry(
        job,
        settings,
        spool_dir,
        registry,
    )


def parse_ocr_batch_process_worker(
    job: ParseJob,
    settings: AppSettings,
    spool_dir: Path,
) -> list[SpoolParseResult]:
    """Parse several OCR parents concurrently around one shared live model."""

    jobs = _submission_jobs(job)
    process_registry = _process_registry or ParserRegistry(settings)
    shared_ocr = process_registry.shared_ocr

    def run_one(item: ParseJob) -> SpoolParseResult:
        isolated_registry = ParserRegistry(
            settings,
            shared_ocr=shared_ocr,
        )
        return _parse_file_process_with_registry(
            item,
            settings,
            spool_dir,
            isolated_registry,
        )

    with ThreadPoolExecutor(
        max_workers=min(
            len(jobs),
            max(1, int(settings.ocr_microbatch_parent_jobs)),
        ),
        thread_name_prefix="ocr-parent",
    ) as executor:
        futures = [executor.submit(run_one, item) for item in jobs]
        return [future.result() for future in futures]


def parse_pdf_batch_process_worker(
    job: ParseJob,
    settings: AppSettings,
    spool_dir: Path,
) -> list[SpoolParseResult]:
    """Run consecutive durable native pages with one live PDF registry."""

    process_registry = _process_registry or ParserRegistry(settings)
    return [
        _parse_file_process_with_registry(
            item,
            settings,
            spool_dir,
            process_registry,
        )
        for item in _submission_jobs(job)
    ]


def _parse_file_process_with_registry(
    job: ParseJob,
    settings: AppSettings,
    spool_dir: Path,
    registry: ParserRegistry,
) -> SpoolParseResult:
    if job.validation_hang_stage:
        write_process_progress(
            process_progress_path(job, spool_dir),
            {
                "file_id": job.file_id,
                "progress_sequence": 1,
                "phase": job.validation_hang_stage,
                "completed": 1,
                "total": 2,
                "cursor": "validation:1",
                "bytes_read": 1,
                "output_blocks": 0,
                "checkpoint_version": 1,
                "detail": "hang-recovery validation fault",
                "worker_pid": os.getpid(),
            },
        )
        while True:
            time.sleep(1)
    checkpoint_path = partial_parse_path(job, spool_dir)
    outcome = parse_file_with_registry(
        job,
        registry,
        ProcessRunControlToken(spool_dir),
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
    _replace_transiently_locked_file(temporary_path, spool_path)
    if outcome.status != "paused":
        checkpoint_path.unlink(missing_ok=True)
    # The coordinator may be reading this small JSON file at the exact moment
    # the worker finishes.  Windows denies unlink while another process still
    # has the file open, so cleanup must never turn a successful parse into a
    # failed parse result.
    try:
        process_progress_path(job, spool_dir).unlink(missing_ok=True)
    except OSError:
        logger.debug(
            "Unable to remove completed parser progress file for %s",
            job.file_path,
            exc_info=True,
        )
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
        task_id=job.task_id,
    )


def materialize_zip_member(job: ParseJob, cancel_token: CancelToken) -> tuple[Path, Path, str]:
    if job.archive_path is None or job.archive_member_index is None:
        raise ValueError("ZIP member descriptor is incomplete")
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    extracted_root = Path(tempfile.mkdtemp(prefix="zip_member_", dir=TEMP_DIR))
    extracted = extracted_root / f"member{job.file_path.suffix.lower()}"
    digest = hashlib.sha256()
    try:
        with zipfile.ZipFile(job.archive_path) as archive:
            infos = archive.infolist()
            if job.archive_member_index < 0 or job.archive_member_index >= len(infos):
                raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
            info = infos[job.archive_member_index]
            if info.is_dir():
                raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
            if info.flag_bits & 0x1:
                raise ZipMemberEncryptedError("ZIP member is encrypted")
            decoded_name = decoded_zip_member_name(info)
            safe_name = safe_zip_member_name(decoded_name)
            if safe_name != job.archive_internal_path:
                raise ZipMemberDirectoryChangedError("ZIP member directory changed during indexing")
            if int(info.file_size) != int(job.size_bytes):
                raise ZipMemberSizeChangedError("ZIP member size changed during indexing")
            if job.archive_member_crc32 is not None and int(info.CRC) != int(job.archive_member_crc32):
                raise ZipMemberContentChangedError("ZIP member content changed during indexing")
            with archive.open(info) as source, extracted.open("wb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    cancel_token.throw_if_cancelled()
                    digest.update(chunk)
                    target.write(chunk)
        content_hash = digest.hexdigest()
        if job.exact_sha256 and content_hash != job.exact_sha256:
            raise ZipMemberContentChangedError("ZIP member content changed during indexing")
        return extracted_root, extracted, content_hash
    except Exception:
        shutil.rmtree(extracted_root, ignore_errors=True)
        raise


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
    materialized_root: Path | None = None
    materialized_sha256: str | None = None
    try:
        cancel_token.wait_if_paused()
        cancel_token.throw_if_cancelled()
        parse_path = job.source_spool_path or job.file_path
        if job.archive_path is not None and job.source_spool_path is None:
            materialized_root, parse_path, materialized_sha256 = materialize_zip_member(job, cancel_token)
        elif job.archive_path is not None:
            materialized_sha256 = job.exact_sha256 or job.content_hash_full
        _assert_direct_source_identity(job, parse_path)
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
                "bytes_read": max(0, int(payload.get("bytes_read") or 0)),
                "output_blocks": len(logical_blocks),
                "checkpoint_version": cursor,
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
            validation_delay_ms = _validation_safe_point_delay_ms()
            if validation_delay_ms > 0:
                time.sleep(validation_delay_ms / 1000.0)

        parser.configure_runtime(
            resume_cursor=job.resume_cursor,
            content_digest=(
                job.content_hash_full
                or job.exact_sha256
                or (job.content_key if job.content_key.startswith("sha256:") else "")
            ),
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
        if job.pdf_task_type == "document_merge":
            parsed_blocks: Iterable[ContentBlock] = ()
        elif job.pdf_task_type:
            parse_scheduled_page = getattr(parser, "parse_scheduled_page", None)
            if not callable(parse_scheduled_page) or job.pdf_page_number is None:
                raise RuntimeError(
                    f"Parser does not support scheduled PDF pages: {parser.name}"
                )
            parsed_blocks = parse_scheduled_page(
                parse_path,
                job.pdf_page_number,
                job.pdf_task_type,
                cancel_token,
            )
        else:
            parsed_blocks = parser.parse(parse_path, cancel_token)
        for block in parsed_blocks:
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            if block.raw_text.strip():
                if parse_path != job.file_path:
                    block.file_path = str(job.file_path)
                if job.archive_path is not None:
                    block.extra["zip_internal_path"] = job.archive_internal_path
                    block.extra["zip_archive_path"] = str(job.archive_path)
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
        _assert_direct_source_identity(job, parse_path)
        parse_ms = int((time.perf_counter() - started) * 1000)
        normalize_started = time.perf_counter()
        effective_settings = settings or registry.settings
        blocks = BlockCoalescer(
            effective_settings.block_target_chars,
            effective_settings.block_max_chars,
        ).coalesce(logical_blocks)
        normalize_ms = int((time.perf_counter() - normalize_started) * 1000)
        status = parser.last_status or "success"
        content_hash_full = materialized_sha256 or (job.exact_sha256 or None)
        if job.archive_path is not None and content_hash_full:
            content_key = f"sha256:{content_hash_full}"
        else:
            content_key = job.content_key
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
            content_key=content_key,
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
            content_hash_full=content_hash_full,
            diagnostics=list(parser.last_diagnostics),
        )
    except UnsupportedFormatError as exc:
        return _diagnostic_outcome(job, "metadata", "unsupported", "UNSUPPORTED_FORMAT", str(exc), started)
    except PauseRequestedError:
        checkpoint = (
            load_partial_parse_checkpoint(job, checkpoint_path.parent, consume=False)
            if checkpoint_path is not None
            else None
        )
        outcome = _diagnostic_outcome(
            job,
            "run_control",
            "paused",
            "PAUSED_AT_SAFE_POINT",
            "任务已在安全检查点暂停",
            started,
        )
        outcome.resume_cursor = (
            checkpoint.resume_cursor
            if checkpoint is not None
            else max(0, int(job.resume_cursor))
        )
        outcome.progress_phase = (
            checkpoint.progress_phase
            if checkpoint is not None and checkpoint.progress_phase
            else job.progress_phase or "paused"
        )
        outcome.progress_completed = (
            checkpoint.progress_completed
            if checkpoint is not None
            else max(0, int(job.progress_completed))
        )
        outcome.progress_total = (
            checkpoint.progress_total
            if checkpoint is not None
            else max(0, int(job.progress_total))
        )
        return outcome
    except CancelledError:
        raise
    except Exception as exc:
        logger.error("Failed to parse %s\n%s", job.file_path, traceback.format_exc())
        retryable = _is_retryable_io_error(exc)
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
    finally:
        if materialized_root is not None:
            shutil.rmtree(materialized_root, ignore_errors=True)


def _assert_direct_source_identity(job: ParseJob, parse_path: Path) -> None:
    if (
        job.archive_path is not None
        or job.source_spool_path is not None
        or job.source_modified_time_ns <= 0
    ):
        return
    stat = parse_path.stat()
    expected_size = int(job.source_size_bytes or job.size_bytes)
    if (
        int(stat.st_size) != expected_size
        or int(stat.st_mtime_ns) != int(job.source_modified_time_ns)
    ):
        raise OSError("SOURCE_CHANGED_DURING_PARSE")


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
    outcome.parser_version = job.parser_version
    outcome.lane = job.lane
    outcome.size_bytes = job.size_bytes
    outcome.estimated_cost = job.estimated_cost
    if not outcome.content_hash_full:
        outcome.content_hash_full = job.content_hash_full or (job.exact_sha256 or None)
    if job.archive_path is not None and outcome.content_hash_full:
        outcome.content_key = f"sha256:{outcome.content_hash_full}"
    else:
        outcome.content_key = job.content_key
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
    retryable = _is_retryable_io_error(exc)
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


def _fast_publish_query_gate(connection: object) -> bool:
    """Exercise the candidate corpus with a deterministic pre-publication query."""

    invalid = int(
        connection.execute(
            """
            SELECT COUNT(*) FROM content_blocks
            WHERE normalized_text IS NULL
            """
        ).fetchone()[0]
    )
    if invalid:
        return False
    sample = connection.execute(
        """
        SELECT normalized_text FROM content_blocks
        WHERE normalized_text != ''
        ORDER BY id LIMIT 1
        """
    ).fetchone()
    if sample is None:
        return True
    token = str(sample["normalized_text"])[:32]
    return int(
        connection.execute(
            """
            SELECT COUNT(*) FROM content_blocks
            WHERE instr(normalized_text, ?) > 0
            """,
            (token,),
        ).fetchone()[0]
    ) > 0


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
        version += (
            f":ocr={int(settings.enable_ocr and settings.ocr_scanned_pdf)}:{settings.ocr_language}"
            f":adaptive={ADAPTIVE_OCR_VERSION}:dynamic={PDF_DYNAMIC_OCR_VERSION}:"
            "preview=150x2400:"
            "region=200:upgrade=300"
        )
    elif name == "image_ocr":
        version += (
            f":ocr={int(settings.enable_ocr and settings.ocr_images)}:{settings.ocr_language}"
            f":min={settings.min_ocr_image_pixels}:detect={settings.max_ocr_image_side}"
            f":adaptive={ADAPTIVE_OCR_VERSION}:tile=1280:overlap=160"
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
    if suffix in ARCHIVE_EXTENSIONS:
        return "zip"
    if suffix == ".doc":
        return "legacy_word"
    if suffix == ".xls":
        return "legacy_excel"
    if suffix == ".ppt":
        return "legacy_powerpoint"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".docx", ".xlsx", ".xlsm", ".pptx"}:
        return "office_process"
    return "normal"


def effective_lane_budget(settings: AppSettings, lane: str) -> int:
    configured = {
        "normal": settings.normal_inflight_bytes,
        "ocr": settings.ocr_inflight_bytes,
        "zip": settings.slow_inflight_bytes,
        "pdf": settings.pdf_inflight_bytes,
        "office_process": settings.office_inflight_bytes,
        "legacy_word": settings.slow_inflight_bytes,
        "legacy_excel": settings.slow_inflight_bytes,
        "legacy_powerpoint": settings.slow_inflight_bytes,
    }[lane]
    multiplier = {"low_resource": 0.5, "balanced": 1.0, "fastest": 1.5}.get(
        settings.index_performance_preset,
        1.0,
    )
    global_budget = max(128, int(settings.index_memory_budget_mb)) * 1024 * 1024
    if lane == "pdf":
        configured = min(
            configured,
            max(128, int(settings.process_memory_budget_mb))
            * 1024
            * 1024
            * max(1, int(settings.pdf_parser_workers)),
        )
    elif lane == "office_process":
        configured = min(
            configured,
            max(128, int(settings.process_memory_budget_mb))
            * 1024
            * 1024
            * max(1, int(settings.process_parser_workers)),
        )
    elif lane.startswith("legacy_"):
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
    if isinstance(exc, ZipMemberDirectoryChangedError):
        return "ZIP_DIRECTORY_CHANGED"
    if isinstance(exc, ZipMemberSizeChangedError):
        return "ZIP_MEMBER_SIZE_CHANGED"
    if isinstance(exc, ZipMemberContentChangedError):
        return "ZIP_MEMBER_CONTENT_CHANGED"
    if isinstance(exc, ZipMemberEncryptedError):
        return "ZIP_MEMBER_ENCRYPTED"
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {32, 33}:
        return "FILE_IN_USE"
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED"
    if isinstance(exc, FileNotFoundError):
        return "FILE_NOT_FOUND"
    if isinstance(exc, zipfile.BadZipFile):
        return "ZIP_CORRUPT"
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in {32, 33}:
            return "FILE_IN_USE"
        if getattr(exc, "errno", None) in {13}:
            return "PERMISSION_DENIED"
        return "IO_ERROR"
    return "PARSER_ERROR"


def _is_retryable_io_error(exc: Exception) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {32, 33}:
        return True
    return isinstance(exc, OSError) and not isinstance(
        exc,
        (PermissionError, FileNotFoundError),
    )


def user_message_for_exception(exc: Exception) -> str:
    if isinstance(exc, ParserDependencyError):
        return str(exc)
    if isinstance(exc, PasswordProtectedError):
        return "文件已加密，需要密码"
    if isinstance(exc, ZipMemberDirectoryChangedError):
        return "ZIP 内部目录结构发生变化"
    if isinstance(exc, ZipMemberSizeChangedError):
        return "ZIP 成员大小发生变化，请重新索引"
    if isinstance(exc, ZipMemberContentChangedError):
        return "ZIP 成员内容发生变化，请重新索引"
    if isinstance(exc, ZipMemberEncryptedError):
        return "ZIP 成员已加密，需要密码"
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) in {32, 33}:
        return "文件正被其他程序占用，请稍后重试"
    if isinstance(exc, PermissionError):
        return "没有权限读取该文件"
    if isinstance(exc, FileNotFoundError):
        return "文件不存在或已被移动"
    if isinstance(exc, zipfile.BadZipFile):
        return "ZIP 文件结构损坏或格式异常"
    if isinstance(exc, OSError):
        if getattr(exc, "winerror", None) in {32, 33}:
            return "文件正被其他程序占用"
        if getattr(exc, "errno", None) in {13}:
            return "没有权限读取该文件"
        return f"读取文件失败：{str(exc) or exc.__class__.__name__}"
    return str(exc) or exc.__class__.__name__


def partial_parse_path(job: ParseJob, spool_dir: Path) -> Path:
    """Return the per-job checkpoint path inside the current controlled run spool."""

    return job.checkpoint_path or (spool_dir / f"{job.file_id}.partial.pickle")


def persistent_checkpoint_path(job: ParseJob) -> Path:
    identity = (
        f"{job.file_id}\0{job.file_path}\0{job.content_key}\0"
        f"{job.parser_name}\0{job.parser_version}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return TEMP_DIR / "parse_checkpoints" / f"{job.file_id}_{digest}.partial.pickle"


def process_progress_path(job: ParseJob, spool_dir: Path) -> Path:
    return spool_dir / f"{job.file_id}.progress.json"


def _replace_transiently_locked_file(
    source: Path,
    target: Path,
    *,
    timeout_seconds: float = 3.0,
) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            source.replace(target)
            return
        except PermissionError as exc:
            if (
                getattr(exc, "winerror", None) not in {32, 33}
                or time.monotonic() >= deadline
            ):
                raise
            time.sleep(0.05)


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


def refresh_job_progress(job: ParseJob, spool_dir: Path, observed_at: float) -> bool:
    path = process_progress_path(job, spool_dir)
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("file_id") or -1) != job.file_id:
            return False
        sequence = max(0, int(payload.get("progress_sequence") or 0))
        if sequence <= job.progress_sequence:
            return False
        previous = SemanticProgress(
            phase=job.progress_phase,
            completed=job.progress_completed,
            total=job.progress_total,
            cursor=job.progress_cursor,
            bytes_read=job.progress_bytes_read,
            output_blocks=job.progress_output_blocks,
            checkpoint_version=job.progress_checkpoint_version,
        )
        current = SemanticProgress.from_mapping(payload)
        job.progress_sequence = sequence
        job.progress_phase = current.phase
        job.progress_completed = current.completed
        job.progress_total = current.total
        job.progress_detail = str(payload.get("detail") or "")
        job.progress_cursor = current.cursor
        job.progress_bytes_read = current.bytes_read
        job.progress_output_blocks = current.output_blocks
        job.progress_checkpoint_version = current.checkpoint_version
        worker_pid = payload.get("worker_pid")
        if isinstance(worker_pid, int):
            job.progress_worker_pid = worker_pid
        advanced = is_semantic_progress(previous, current)
        if advanced:
            job.last_progress_monotonic = observed_at
        return advanced
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.debug("Unable to read parser progress %s", path, exc_info=True)
        return False


def no_progress_timeout(settings: AppSettings, job: ParseJob) -> int:
    base = {
        "ocr": settings.ocr_no_progress_timeout_seconds,
        "zip": settings.archive_no_progress_timeout_seconds,
        "legacy_word": settings.legacy_no_progress_timeout_seconds,
        "legacy_excel": settings.legacy_no_progress_timeout_seconds,
        "legacy_powerpoint": settings.legacy_no_progress_timeout_seconds,
        "pdf": settings.process_no_progress_timeout_seconds,
        "office_process": settings.process_no_progress_timeout_seconds,
        "normal": settings.normal_no_progress_timeout_seconds,
    }.get(job.lane, settings.normal_no_progress_timeout_seconds)
    if "ocr" in job.progress_phase.lower():
        base = max(base, settings.ocr_no_progress_timeout_seconds)
    return max(1, int(base))


def pdf_child_progress(
    job: ParseJob,
) -> tuple[str, str, str] | None:
    if job.parser_name != "pdf" or job.pdf_task_type:
        return None
    try:
        page_number = int(job.progress_cursor)
    except (TypeError, ValueError):
        return None
    if page_number <= 0:
        return None
    phase = job.progress_phase.lower()
    if phase.startswith("pdf_ocr"):
        status = "complete" if phase == "pdf_ocr_page" else "running"
        return "pdf_ocr_page", f"page:{page_number}", status
    if phase in {"pdf_page", "pdf_native_page"}:
        return "pdf_native_page", f"page:{page_number}", "complete"
    return None


def register_stall(job: ParseJob) -> int:
    signature = (
        f"{job.parser_version}|"
        + progress_signature(
            SemanticProgress(
                phase=job.progress_phase,
                completed=job.progress_completed,
                total=job.progress_total,
                cursor=job.progress_cursor,
                bytes_read=job.progress_bytes_read,
                output_blocks=job.progress_output_blocks,
                checkpoint_version=job.progress_checkpoint_version,
            )
        )
    )
    if signature == job.stall_signature:
        job.repeated_stall_count += 1
    else:
        job.stall_signature = signature
        job.repeated_stall_count = 1
    return job.repeated_stall_count


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
        # Checkpoints retain the parser's original logical units. Coalescing
        # here and then coalescing again after resume changes block boundaries
        # compared with an uninterrupted run even when the text is identical.
        blocks = list(logical_blocks)
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
        if job.content_key and outcome.content_key != job.content_key:
            raise ValueError("Partial parse checkpoint content key mismatch")
        if job.parser_version and outcome.parser_version != job.parser_version:
            raise ValueError("Partial parse checkpoint parser version mismatch")
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


def _prepared_source_size(source: PreparedSource) -> int:
    if source.size_bytes is None:
        raise PlanningWorkerError(
            f"准备任务没有返回文件大小：{source.file_path}"
        )
    return max(0, int(source.size_bytes))


def _planning_phase_label(phase: str) -> str:
    return {
        "root_scan": "正在检查搜索范围",
        "directory_enumeration": "正在发现文件",
        "file_stat": "正在读取文件元数据",
        "source_prepare": "正在准备文件内容",
        "content_hash": "正在读取并计算文件指纹",
        "zip_manifest": "正在读取 ZIP 成员目录",
        "zip_member_prepare": "正在准备 ZIP 成员",
    }.get(str(phase or ""), "正在准备索引任务")


def _order_planned_jobs_fairly(jobs: list[ParseJob]) -> list[ParseJob]:
    """Round-robin sources inside each lane while keeping deterministic order."""

    by_lane: dict[str, dict[str, deque[ParseJob]]] = {}
    source_labels: dict[tuple[str, str], str] = {}
    for job in jobs:
        source_key = (
            f"pdf:{job.pdf_document_task_id}"
            if job.pdf_document_task_id is not None
            else f"file:{job.file_id}"
        )
        by_lane.setdefault(job.lane, {}).setdefault(source_key, deque()).append(job)
        source_labels[(job.lane, source_key)] = str(job.file_path).lower()
    ordered: list[ParseJob] = []
    for lane_name in sorted(by_lane):
        groups = by_lane[lane_name]
        for queue in groups.values():
            sorted_jobs = sorted(
                queue,
                key=lambda job: (
                    int(job.pdf_page_number or 0),
                    0 if job.pdf_task_type != "pdf_ocr_page" else 1,
                    -job.estimated_cost,
                    str(job.file_path).lower(),
                ),
            )
            queue.clear()
            queue.extend(sorted_jobs)
        source_keys = sorted(
            groups,
            key=lambda key: source_labels[(lane_name, key)],
        )
        while source_keys:
            next_keys: list[str] = []
            for source_key in source_keys:
                queue = groups[source_key]
                if queue:
                    ordered.append(queue.popleft())
                if queue:
                    next_keys.append(source_key)
            source_keys = next_keys
    return ordered
