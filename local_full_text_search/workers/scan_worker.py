from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from pathlib import Path
from collections.abc import Sequence

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager, utc_now
from local_full_text_search.core.hardware_profile import (
    build_performance_profile,
    detect_hardware_profile_for_roots,
    settings_for_profile,
)
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.index_generation_store import (
    CandidateGeneration,
    IndexGenerationStore,
)
from local_full_text_search.core.index_change_detector import check_published_index_changes
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.core.search_index_coordinator import (
    get_search_index_coordinator,
)
from local_full_text_search.services.startup_diagnostics import (
    frozen_runtime_error_message,
    frozen_runtime_issues,
)
from local_full_text_search.services.logging_service import log_exception, log_event

logger = logging.getLogger(__name__)


class ScanWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal(object)
    # This is the only lifecycle signal.  It is emitted after process-pool,
    # candidate and database cleanup has completed, so the UI can safely
    # enable a new scan without overlapping the previous one.
    ended = Signal(object)

    def __init__(
        self,
        db_path: Path,
        settings: AppSettings,
        *,
        control_db_path: Path | None = None,
        performance_mode: bool = False,
        root_ids: Sequence[int] | None = None,
        file_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.control_db_path = Path(control_db_path or db_path)
        self.generation_store = IndexGenerationStore(db_path)
        self._change_check = None
        self.candidate: CandidateGeneration | None = None
        self.candidate_published = False
        self.settings = settings
        self.performance_mode = performance_mode
        self.root_ids = tuple(int(root_id) for root_id in (root_ids or ()))
        self.file_ids = tuple(int(file_id) for file_id in (file_ids or ()))
        self.token = CancelToken()
        self.manager: IndexManager | None = None
        self.database: DatabaseManager | None = None
        self._pause_monitor_lock = threading.Lock()
        self._pause_monitor_generation = 0
        self._hardware: object | None = None
        self._root_disk_classes: dict[str, str] = {}
        self._profile_payload: dict[str, object] | None = None
        self.scan_id = f"scan-{uuid.uuid4().hex}"
        self.run_ids: list[str] = []
        self._last_error_message = ""
        self._last_readiness: dict[str, object] = {}
        self._last_blocking_files: list[dict[str, object]] = []
        self._last_unready_roots: list[dict[str, object]] = []
        self.partial_publish = False
        self._scan_started_at = ""
        self._terminal_status = "failed"
        self._terminal_payload: dict[str, object] = {}
        self._terminal_emitted = False
        self._cleanup_errors: list[dict[str, str]] = []
        self._failure_report_persisted = False

    def _phase_event(
        self,
        phase: str,
        status: str,
        **fields: object,
    ) -> None:
        """Record the worker boundary that was reached before a hard exit."""

        try:
            log_event(
                "index.scan.phase",
                component="scan_worker",
                scan_id=self.scan_id,
                phase=str(phase),
                phase_status=str(status),
                database_path=str(self.db_path),
                parent_pid=os.getppid(),
                **fields,
            )
        except BaseException:
            # Diagnostics must not turn an otherwise recoverable scan error
            # into a second failure when the log directory is unavailable.
            logger.debug("Unable to write scan phase event", exc_info=True)

    @Slot()
    def run(self) -> None:
        # run() 位于 MainWindow 创建的 scan QThread 中。这个线程负责整轮索引
        # 的“总调度”；具体 PDF、Office、OCR 解析还会下发给后面的进程池。
        # The scan identifier is durable diagnostic context even when startup
        # fails before a candidate or index run can be created.
        self._scan_started_at = utc_now()
        log_event(
            "index.scan.started",
            component="scan_worker",
            scan_id=self.scan_id,
            database_path=str(self.db_path),
            root_ids=list(self.root_ids),
            file_ids=list(self.file_ids),
            cancellation_source="未取消",
            parent_pid=os.getppid(),
        )
        try:
            self.progress.emit(
                {
                    "stage": "starting",
                    "phase_label": "正在启动更新并检查运行组件…",
                }
            )
            self._phase_event("runtime_check", "started")
            runtime_issues = frozen_runtime_issues()
            if runtime_issues:
                raise RuntimeError(frozen_runtime_error_message(runtime_issues))
            self._phase_event("runtime_check", "completed")
            self.progress.emit(
                {
                    "stage": "change_check",
                    "phase_label": "正在只读检查文件与索引版本是否有变化…",
                }
            )
            self._phase_event("change_check", "started")
            from local_full_text_search.core.index_manager import parser_identity_for_path

            change_check = check_published_index_changes(
                self.generation_store,
                self.control_db_path,
                self.settings,
                root_ids=self.root_ids,
                file_ids=self.file_ids,
                parser_identity=parser_identity_for_path,
                cancel_token=self.token,
            )
            self._change_check = change_check
            self._phase_event(
                "change_check",
                "completed",
                **change_check.to_dict(),
            )
            log_event(
                "index.scan.change_check",
                component="scan_worker",
                scan_id=self.scan_id,
                database_path=str(self.db_path),
                control_database_path=str(self.control_db_path),
                **change_check.to_dict(),
            )
            if change_check.unchanged:
                current_active = self.generation_store.active()
                message = (
                    f"索引未变化，继续使用第 {change_check.active_generation} 代；"
                    "未创建新的索引数据库"
                )
                self._terminal_status = "finished"
                self._terminal_payload = {
                    "status": "finished",
                    "unchanged": True,
                    "message": message,
                    "active_generation": change_check.active_generation,
                    "active_database_path": str(current_active.path if current_active else ""),
                    "scan_id": self.scan_id,
                    "run_ids": [],
                    "change_check": change_check.to_dict(),
                }
                self.progress.emit(
                    {
                        "stage": "unchanged",
                        "phase_label": message,
                        "change_check": change_check.to_dict(),
                    }
                )
                return
            self.progress.emit(
                {
                    "stage": "snapshot_candidate",
                    "phase_label": "正在准备独立索引数据库…",
                }
            )
            self._phase_event("candidate_begin", "started")
            try:
                # 创建 gen-(N+1).building.db。此时 SearchWorker 仍然读取已发布的
                # gen-N.db，因此索引和搜索可以并行进行而不争用同一 SQLite 文件。
                self.candidate = self.generation_store.begin_candidate(
                    scan_id=self.scan_id
                )
            except TypeError as exc:
                # Keep older/test doubles that expose begin_candidate() with
                # no keyword arguments compatible while the production store
                # records scan_id in its build lock.
                if "scan_id" not in str(exc):
                    raise
                self.candidate = self.generation_store.begin_candidate()
            self._phase_event(
                "candidate_begin",
                "completed",
                candidate_generation=self.candidate.generation,
                candidate_path=str(self.candidate.path),
            )
            self._phase_event("candidate_database_initialize", "started")
            # 本轮所有扫描、解析结果和数据库写入都指向候选库，不修改旧快照。
            db = DatabaseManager(self.candidate.path)
            self.database = db
            # A current-schema database is already usable by the GUI.  Avoid
            # rebuilding a large FTS table before an incremental run; legacy
            # databases still take the full migration path in this worker.
            if db._startup_fast_path_ready():
                db.initialize(startup_fast=True)
                # Normal GUI startup intentionally skips recovery work. Run
                # bounded cleanup here before new parse results are planned so
                # a force-closed large XLS cannot leave staging rows forever.
                db.cleanup_orphan_staging_documents()
            else:
                last_migration_progress = 0.0

                def report_migration(payload: dict[str, object]) -> None:
                    nonlocal last_migration_progress
                    stage = str(payload.get("stage") or "")
                    if stage != "database_migration_backup":
                        self.progress.emit(dict(payload))
                        return
                    now = time.monotonic()
                    if now - last_migration_progress < 0.2 and int(
                        payload.get("progress_percent") or 0
                    ) < 100:
                        return
                    last_migration_progress = now
                    percent = max(0, min(100, int(payload.get("progress_percent") or 0)))
                    copied = max(0, int(payload.get("completed_bytes") or 0))
                    total = max(0, int(payload.get("total_bytes") or 0))
                    if total <= 0:
                        self.progress.emit(dict(payload))
                        return
                    self.progress.emit(
                        {
                            "stage": "database_migration_backup",
                            "phase_label": (
                                f"正在安全备份索引数据库 {percent}%"
                                f"（{copied / (1024 * 1024 * 1024):.1f} / "
                                f"{total / (1024 * 1024 * 1024):.1f} GB）"
                            ),
                            "migration_progress_percent": percent,
                        }
                    )

                db.initialize(progress_callback=report_migration)
            self._phase_event(
                "candidate_database_initialize",
                "completed",
                candidate_generation=self.candidate.generation,
            )
            self._phase_event("control_roots_sync", "started")
            self._sync_control_roots(db)
            self._phase_event("control_roots_sync", "completed")
            self._phase_event("usn_state_restore", "started")
            self._restore_usn_state_from_control(db)
            self._phase_event("usn_state_restore", "completed")
            effective_settings = self.settings
            profile_payload: dict[str, object] | None = None
            roots = db.list_roots(enabled_only=True)
            root_paths = [Path(str(root["path"])) for root in roots]
            self._phase_event(
                "hardware_detection",
                "started",
                root_count=len(root_paths),
            )
            hardware, root_disk_classes = detect_hardware_profile_for_roots(root_paths)
            self._hardware = hardware
            self._root_disk_classes = dict(root_disk_classes)
            self._phase_event(
                "hardware_detection",
                "completed",
                root_count=len(root_paths),
                disk_class=hardware.disk_class,
            )
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
            self._phase_event("index_manager_create", "started")
            # IndexManager 负责枚举文件、分配解析通道、启动进程池并提交写库任务。
            self.manager = IndexManager(
                db,
                effective_settings,
                run_context={
                    "execution_mode": "performance" if self.performance_mode else "normal",
                    "scan_id": self.scan_id,
                    "hardware": hardware.to_dict(),
                    "root_disk_classes": root_disk_classes,
                    "effective_profile": effective_profile,
                    "run_id_callback": self._register_run_id,
                },
            )
            get_search_index_coordinator().register_index_manager(
                self.manager,
                pause_controller=self,
            )
            self._phase_event("index_manager_create", "completed")
            if self.token.paused:
                self.manager.request_pause()
            self._phase_event(
                "index_enabled_roots",
                "started",
                root_ids=list(self.root_ids),
                file_ids=list(self.file_ids),
            )
            # 这是本轮最主要的耗时步骤。虽然调用形式是同步的，但当前已经位于
            # scan QThread，因此不会阻塞界面；内部解析还会多进程并行。
            summary = self.manager.index_enabled_roots(
                self.token,
                self.progress.emit,
                root_ids=self.root_ids or None,
                file_ids=self.file_ids or None,
            )
            self._phase_event(
                "index_enabled_roots",
                "completed",
                summary=summary.to_dict(),
            )
            self._phase_event("readiness_check", "started")
            run_reports = db.recent_index_runs_since(self._scan_started_at)
            self.run_ids = [str(row.get("id") or "") for row in run_reports]
            if bool(getattr(summary, "cancelled", False)) or self.token.cancelled:
                raise CancelledError("本轮索引已取消，候选数据库不会发布")
            # 发布前必须确认候选库达到可搜索条件。未通过时保留旧 generation，
            # 不允许用户看到一份半成品索引。
            readiness = db.index_readiness(reconcile=False)
            self._last_blocking_files = [
                dict(row) for row in (readiness.get("blocking_file_samples") or [])
            ]
            if int(readiness.get("blocking_files") or 0) > len(
                self._last_blocking_files
            ):
                try:
                    # Readiness keeps a bounded sample for quick UI probes;
                    # the failure report must retain every row so selection,
                    # exclusion and force-complete operate on the real total.
                    self._last_blocking_files = [
                        dict(row)
                        for row in db.blocking_file_diagnostics(limit=None)
                    ]
                except Exception:
                    logger.exception(
                        "Unable to collect the complete blocker report; keeping the readiness sample"
                    )
            readiness = dict(readiness)
            readiness["blocking_file_samples"] = list(self._last_blocking_files)
            readiness["blocking_file_details"] = list(self._last_blocking_files)
            readiness["blocking_details_complete"] = (
                len(self._last_blocking_files)
                == int(readiness.get("blocking_files") or 0)
            )
            self._last_readiness = dict(readiness)
            self._last_unready_roots = [
                dict(row) for row in (readiness.get("unready_root_samples") or [])
            ]
            log_event(
                "index.candidate.readiness_checked",
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=self.candidate.generation,
                candidate_path=str(self.candidate.path),
                run_ids=list(self.run_ids),
                readiness=dict(readiness),
            )
            self._phase_event(
                "readiness_check",
                "completed",
                ready=bool(readiness.get("ready")),
                blocking_files=int(readiness.get("blocking_files") or 0),
            )
            if not bool(readiness.get("ready")):
                if self._can_publish_with_terminal_blockers(readiness):
                    self.partial_publish = True
                    details = self._format_blocking_details(
                        self._last_blocking_files
                    )
                    log_event(
                        "index.candidate.partial_publish_allowed",
                        level=logging.WARNING,
                        component="scan_worker",
                        scan_id=self.scan_id,
                        run_ids=list(self.run_ids),
                        candidate_generation=self.candidate.generation,
                        blocking_files=int(readiness.get("blocking_files") or 0),
                        blocking_status_counts=dict(
                            readiness.get("blocking_status_counts") or {}
                        ),
                        blocking_details=details,
                    )
                else:
                    details = self._format_blocking_details(
                        self._last_blocking_files
                    )
                    suffix = f"；阻断文件：{details}" if details else ""
                    raise RuntimeError(
                        "本轮索引未达到完整发布条件："
                        + ", ".join(
                            str(item)
                            for item in readiness.get("not_ready_reasons", [])
                        )
                        + suffix
                    )
            self._phase_event("candidate_publish", "started")
            candidate_root_rows = [
                dict(row) for row in db.list_roots(enabled_only=True)
            ]
            usn_root_ids = tuple(
                sorted(
                    {
                        int(root["id"])
                        for root in candidate_root_rows
                        if not self.root_ids
                        or int(root["id"]) in set(self.root_ids)
                    }
                )
            )
            usn_state = db.export_usn_state(usn_root_ids)
            published_at = utc_now()
            # 校验通过后原子发布候选 generation。manifest 切换后，新搜索会读取
            # 新版本；切换前已经开始的搜索仍可通过租约继续读取旧版本。
            published = self.generation_store.publish(
                self.candidate,
                file_count=int(readiness.get("file_count") or 0),
                document_count=int(readiness.get("document_count") or 0),
                block_count=int(readiness.get("block_count") or 0),
                published_at=published_at,
                partial=self.partial_publish,
                scan_signature=(
                    self._change_check.scan_signature
                    if self._change_check is not None
                    and not self.root_ids
                    and not self.file_ids
                    else None
                ),
            )
            self.candidate_published = True
            if self.partial_publish:
                self._persist_failure_report(
                    error_message=(
                        "本轮索引已发布，但终止性跳过文件未纳入正文索引"
                    ),
                    candidate_path=str(published.path),
                    candidate_generation=published.generation,
                    published_partial=True,
                )
            else:
                self.generation_store.clear_failure_report()
            self._persist_usn_state_to_control(
                usn_state,
                root_ids=usn_root_ids,
                candidate_generation=published.generation,
            )
            self._sync_published_root_state_to_control(
                candidate_root_rows,
                published_at,
            )
            log_event(
                "index.candidate.published",
                component="scan_worker",
                scan_id=self.scan_id,
                run_ids=list(self.run_ids),
                candidate_generation=published.generation,
                database_path=str(published.path),
                file_count=published.file_count,
                document_count=published.document_count,
                block_count=published.block_count,
            )
            self._phase_event(
                "candidate_publish",
                "completed",
                candidate_generation=published.generation,
                file_count=published.file_count,
            )
            self.generation_store.cleanup_retired()
            self._terminal_status = "finished"
            self._terminal_payload = {
                "status": "finished",
                "summary": summary.to_dict(),
                "runs": run_reports,
                "run_metrics": _aggregate_run_metrics(run_reports),
                "performance_mode": self.performance_mode,
                "performance_profile": self._profile_payload,
                "active_generation": published.generation,
                "active_database_path": str(published.path),
                "published_at": published.published_at,
                "partial_publish": self.partial_publish,
                "blocking_files": int(readiness.get("blocking_files") or 0),
                "scan_id": self.scan_id,
                "run_ids": list(self.run_ids),
            }
        except BaseException as exc:
            if self.token.cancelled or isinstance(exc, CancelledError):
                self._terminal_status = "cancelled"
                self._last_error_message = "本轮索引已取消"
                self._terminal_payload = {
                    "status": "cancelled",
                    "message": self._last_error_message,
                    "cancellation_source": self.token.cancel_reason,
                    "scan_id": self.scan_id,
                    "run_ids": list(self.run_ids),
                }
            elif (
                isinstance(exc, (KeyboardInterrupt, SystemExit))
                or "目录发现子进程意外中断" in str(exc)
                or "WORKER_INTERRUPTED" in str(exc)
            ):
                self._terminal_status = "failed"
                self._last_error_message = (
                    "目录发现子进程意外中断，已停止本轮候选发布"
                    f"（{type(exc).__name__}）"
                )
                self._terminal_payload = {
                    "status": "failed",
                    "message": self._last_error_message,
                    "cancellation_source": "子进程意外中断",
                    "scan_id": self.scan_id,
                    "run_ids": list(self.run_ids),
                }
            else:
                self._terminal_status = "failed"
                self._last_error_message = str(exc) or exc.__class__.__name__
                self._terminal_payload = {
                    "status": "failed",
                    "message": self._last_error_message,
                    "cancellation_source": "未取消",
                    "scan_id": self.scan_id,
                    "run_ids": list(self.run_ids),
                }
            if self.token.cancelled or isinstance(exc, CancelledError):
                logger.info(
                    "Index scan cancelled: source=%s",
                    self.token.cancel_reason,
                    exc_info=True,
                )
            else:
                logger.exception("Index scan failed")
            log_exception(
                "index.scan.cancelled"
                if self.token.cancelled or isinstance(exc, CancelledError)
                else "index.scan.failed",
                exc,
                component="scan_worker",
                run_id=self.run_ids[-1] if self.run_ids else None,
                scan_id=self.scan_id,
                run_ids=list(self.run_ids),
                candidate_generation=(
                    self.candidate.generation if self.candidate is not None else None
                ),
                database_path=str(self.db_path),
                candidate_path=str(self.candidate.path) if self.candidate else "",
                candidate_published=self.candidate_published,
                root_ids=list(self.root_ids),
                file_ids=list(self.file_ids),
                readiness=dict(self._last_readiness),
                blocking_files=list(self._last_blocking_files),
                unready_roots=list(self._last_unready_roots),
                cancellation_source=(
                    self.token.cancel_reason
                    if self.token.cancelled
                    else self._terminal_payload.get("cancellation_source", "未取消")
                ),
            )
            self._terminal_payload.update(
                {
                    "error": self._last_error_message or str(exc) or exc.__class__.__name__,
                    "original_exception_type": type(exc).__name__,
                    "original_exception": str(exc) or type(exc).__name__,
                }
            )
        finally:
            self._cleanup_errors = []
            if self.manager is not None:
                try:
                    get_search_index_coordinator().unregister_index_manager(self.manager)
                except BaseException as exc:
                    self._record_cleanup_error("unregister_index_manager", exc)
                try:
                    self.manager.force_terminate_processes()
                except BaseException as exc:
                    self._record_cleanup_error("force_terminate_processes", exc)
            self.manager = None
            if self.candidate is not None and not self.candidate_published:
                try:
                    self._capture_failure_state()
                except BaseException as exc:
                    self._record_cleanup_error("capture_failure_state", exc)
                report_saved = False
                try:
                    self._persist_failure_report(
                        error_message=self._last_error_message,
                        candidate_path=str(self.candidate.path),
                        candidate_generation=self.candidate.generation,
                    )
                    report_saved = True
                    self._failure_report_persisted = True
                except BaseException as exc:
                    self._record_cleanup_error("persist_failure_report", exc)
                    log_event(
                        "index.failure_report.preserve_candidate",
                        level=logging.ERROR,
                        component="scan_worker",
                        scan_id=self.scan_id,
                        candidate_generation=self.candidate.generation,
                        candidate_path=str(self.candidate.path),
                        reason="failure_report_persist_failed",
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:1000],
                    )
                log_event(
                    "index.candidate.aborted",
                    level=logging.WARNING,
                    component="scan_worker",
                    run_id=self.run_ids[-1] if self.run_ids else None,
                    scan_id=self.scan_id,
                    run_ids=list(self.run_ids),
                    candidate_generation=self.candidate.generation,
                    baseline_generation=(
                        self.candidate.baseline.generation
                        if self.candidate.baseline is not None
                        else None
                    ),
                    candidate_path=str(self.candidate.path),
                    reason=(
                        "scan_worker_exit_before_publish"
                        if report_saved
                        else "scan_worker_exit_report_persist_failed_candidate_preserved"
                    ),
                )
                if report_saved:
                    try:
                        self.generation_store.abort_candidate(self.candidate)
                    except BaseException as exc:
                        self._record_cleanup_error("abort_candidate", exc)
            self.candidate = None
            if self.database is not None:
                # Interrupt any statement that was still in flight when the
                # worker failed or was cancelled. The connection context then
                # rolls back and closes it, releasing SQLite locks before the
                # QThread is allowed to finish.
                try:
                    self.database.interrupt_active_connections()
                except BaseException as exc:
                    self._record_cleanup_error("interrupt_database_connections", exc)
            self.database = None
            self._terminal_payload.setdefault("status", self._terminal_status)
            self._terminal_payload.update(
                {
                    "scan_id": self.scan_id,
                    "run_ids": list(self.run_ids),
                    "cancellation_source": self._terminal_payload.get(
                        "cancellation_source",
                        self.token.cancel_reason if self.token.cancelled else "未取消",
                    ),
                    "cleanup_errors": list(self._cleanup_errors),
                    "failure_report_persisted": self._failure_report_persisted,
                }
            )
            self._emit_terminal_signals()

    def _register_run_id(self, run_id: str) -> None:
        value = str(run_id or "")
        if value and value not in self.run_ids:
            self.run_ids.append(value)
            log_event(
                "index.run.registered",
                component="scan_worker",
                scan_id=self.scan_id,
                run_id=value,
                database_path=str(self.db_path),
            )

    def _record_cleanup_error(self, operation: str, exc: BaseException) -> None:
        payload = {
            "operation": str(operation),
            "error_type": type(exc).__name__,
            "error": str(exc) or type(exc).__name__,
        }
        self._cleanup_errors.append(payload)
        log_exception(
            "index.scan.cleanup_failed",
            exc,
            component="scan_worker",
            scan_id=self.scan_id,
            operation=str(operation),
        )

    def _emit_terminal_signals(self) -> None:
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        payload = dict(self._terminal_payload)
        if self._terminal_status == "finished":
            self.finished.emit(payload)
        elif self._terminal_status == "cancelled":
            self.cancelled.emit(payload)
        else:
            self.failed.emit(str(payload.get("message") or self._last_error_message))
        self.ended.emit(payload)

    def _capture_failure_state(self) -> None:
        """Capture candidate state before aborting it after an abnormal exit."""

        if self.database is None:
            return
        capture_error: BaseException | None = None
        for attempt, delay in enumerate((0.0, 0.2, 0.5, 1.0), 1):
            if delay:
                time.sleep(delay)
            try:
                self.database.interrupt_active_connections()
            except Exception:
                logger.debug(
                    "Unable to interrupt candidate connections before failure capture",
                    exc_info=True,
                )
            try:
                readiness = dict(self.database.index_readiness(reconcile=False))
                blocking_files: list[dict[str, object]] = []
                blockers = int(readiness.get("blocking_files") or 0)
                if blockers:
                    blocking_files = [
                        dict(row)
                        for row in self.database.blocking_file_diagnostics(limit=None)
                    ]
                    readiness["blocking_file_samples"] = list(blocking_files)
                    readiness["blocking_file_details"] = list(blocking_files)
                    readiness["blocking_details_complete"] = (
                        len(blocking_files) == blockers
                    )
                self._last_readiness = readiness
                self._last_blocking_files = blocking_files
                self._last_unready_roots = [
                    dict(row)
                    for row in (readiness.get("unready_root_samples") or [])
                ]
                log_event(
                    "index.failure_state.captured",
                    component="scan_worker",
                    scan_id=self.scan_id,
                    candidate_generation=(
                        self.candidate.generation
                        if self.candidate is not None
                        else None
                    ),
                    attempt=attempt,
                    blocking_files=len(blocking_files),
                    readiness_blocking_files=blockers,
                    unfinished_tasks=int(readiness.get("unfinished_tasks") or 0),
                    active_runs=int(readiness.get("active_runs") or 0),
                )
                capture_error = None
                break
            except BaseException as exc:
                capture_error = exc
                if attempt == 4:
                    logger.warning(
                        "Unable to capture candidate failure state before abort: %s",
                        exc,
                        exc_info=True,
                    )
        if capture_error is not None:
            log_event(
                "index.failure_state.capture_failed",
                level=logging.ERROR,
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=(
                    self.candidate.generation if self.candidate is not None else None
                ),
                error_type=type(capture_error).__name__,
                error_message=str(capture_error)[:1000],
            )
        if self._scan_started_at:
            try:
                reports = self.database.recent_index_runs_since(self._scan_started_at)
                self.run_ids = [str(row.get("id") or "") for row in reports]
            except Exception:
                logger.debug("Unable to collect run IDs before candidate abort", exc_info=True)

    def _persist_failure_report(
        self,
        *,
        error_message: str,
        candidate_path: str,
        candidate_generation: int,
        published_partial: bool = False,
    ) -> None:
        self.generation_store.write_failure_report(
            {
                "schema_version": 1,
                "failed_at": utc_now(),
                "scan_id": self.scan_id,
                "run_ids": list(self.run_ids),
                "candidate_generation": int(candidate_generation),
                "baseline_generation": (
                    self.candidate.baseline.generation
                    if self.candidate is not None
                    and self.candidate.baseline is not None
                    else None
                ),
                "candidate_path": str(candidate_path),
                "database_path": str(self.db_path),
                "error": str(error_message),
                "status": self._terminal_status,
                "cancellation_source": self._terminal_payload.get(
                    "cancellation_source",
                    self.token.cancel_reason if self.token.cancelled else "未取消",
                ),
                "original_exception": self._terminal_payload.get(
                    "original_exception",
                    str(error_message),
                ),
                "original_exception_type": self._terminal_payload.get(
                    "original_exception_type",
                    "",
                ),
                "published_partial": bool(published_partial),
                "readiness": dict(self._last_readiness),
                "blocking_files": list(self._last_blocking_files),
                "unready_roots": list(self._last_unready_roots),
            }
        )

    @staticmethod
    def _can_publish_with_terminal_blockers(
        readiness: dict[str, object],
    ) -> bool:
        reasons = {
            str(item) for item in (readiness.get("not_ready_reasons") or [])
        }
        if not reasons.issubset({"blocking_files", "unready_root"}):
            return False
        blocker_count = int(readiness.get("blocking_files") or 0)
        samples = [
            dict(row)
            for row in (readiness.get("blocking_file_samples") or [])
        ]
        if blocker_count <= 0 or len(samples) < blocker_count:
            return False
        terminal_statuses = {
            "failed",
            "failed_retryable",
            "unsupported",
            "skipped",
            "ocr_disabled",
            "ocr_failed",
            "converter_missing",
            "partial_success",
            "password_protected",
        }
        return all(
            str(row.get("parse_status") or "") in terminal_statuses
            and str(row.get("task_status") or "complete")
            in {"", "complete", "failed", "failed_retryable"}
            for row in samples[:blocker_count]
        )

    @staticmethod
    def _format_blocking_details(
        rows: list[dict[str, object]],
        *,
        limit: int = 3,
    ) -> str:
        details = []
        for row in rows[: max(1, int(limit))]:
            path = str(row.get("path") or "")
            status = str(row.get("parse_status") or "unknown")
            code = str(row.get("parse_error_code") or "")
            if not path:
                continue
            details.append(f"{path}({status}{':' + code if code else ''})")
        return "、".join(details)

    def _sync_published_root_state_to_control(
        self,
        candidate_roots: list[dict[str, object]],
        published_at: str,
    ) -> None:
        """Persist display-only root completion state in the control DB."""

        if self.control_db_path.resolve() == self.db_path.resolve():
            return
        selected = set(self.root_ids)
        try:
            updates = []
            for root in candidate_roots:
                root_id = int(root["id"])
                if selected and root_id not in selected:
                    continue
                updates.append(
                    (
                        str(root["status"] or "ready"),
                        str(root["last_scan_at"] or published_at),
                        str(root["updated_at"] or published_at),
                        root_id,
                    )
                )
            if not updates:
                return
            control = DatabaseManager(self.control_db_path)
            with control.connect(timeout_seconds=2.0) as con:
                con.executemany(
                    """
                    UPDATE roots
                    SET status = ?, last_scan_at = ?, updated_at = ?
                    WHERE id = ? AND enabled = 1
                    """,
                    updates,
                )
            log_event(
                "index.control_root_state.synced",
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=self.candidate.generation if self.candidate else None,
                root_count=len(updates),
            )
        except Exception as exc:
            # The immutable snapshot is already published; stale control
            # metadata must not make a valid generation look like a failed
            # scan. The next UI refresh/startup can retry this synchronization.
            log_exception(
                "index.control_root_state.sync_failed",
                exc,
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=(
                    self.candidate.generation if self.candidate is not None else None
                ),
                control_db_path=str(self.control_db_path),
            )

    def _sync_control_roots(self, candidate_db: DatabaseManager) -> None:
        """Merge the current scope configuration into the candidate DB.

        A published snapshot is intentionally immutable. Scope additions,
        disables and removals are recorded in the control DB and must be
        reflected only in the next candidate, never by mutating the active
        search snapshot.
        """

        if not self.control_db_path.is_file():
            return
        source = DatabaseManager(self.control_db_path, read_only=True)
        control_roots = source.list_roots()
        with candidate_db.connect() as con:
            for root in control_roots:
                now = utc_now()
                con.execute(
                    """
                    INSERT INTO roots(
                        id, path, enabled, include_subfolders,
                        exclude_rules_json, created_at, updated_at,
                        last_scan_at, status
                    ) VALUES (?, ?, ?, ?, ?, COALESCE(?, ?), ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        path = excluded.path,
                        enabled = excluded.enabled,
                        include_subfolders = excluded.include_subfolders,
                        exclude_rules_json = excluded.exclude_rules_json,
                        updated_at = excluded.updated_at,
                        status = CASE
                            WHEN excluded.enabled = 0 THEN 'disabled'
                            ELSE status
                        END
                    """,
                    (
                        int(root["id"]),
                        str(root["path"]),
                        int(root["enabled"] or 0),
                        int(root["include_subfolders"] or 0),
                        root["exclude_rules_json"],
                        root["created_at"],
                        now,
                        now,
                        root["last_scan_at"],
                        root["status"] or ("ready" if root["enabled"] else "disabled"),
                    ),
                )
            if control_roots:
                ids = tuple(int(root["id"]) for root in control_roots)
                placeholders = ",".join("?" for _ in ids)
                con.execute(
                    f"UPDATE roots SET enabled = 0, status = 'removed' "
                    f"WHERE id NOT IN ({placeholders})",
                    ids,
                )
            else:
                con.execute("UPDATE roots SET enabled = 0, status = 'removed'")

    @classmethod
    def _usn_state_ready_root_ids(
        cls,
        state: dict[str, object],
    ) -> set[int]:
        """Return roots with both a cursor and a directory map."""

        checkpoint_roots: set[int] = set()
        directory_roots: set[int] = set()
        for row in state.get("checkpoints") or []:
            if not isinstance(row, dict):
                continue
            parts = str(row.get("key") or "").split(":", 2)
            if len(parts) < 2 or parts[0] != "usn_checkpoint":
                continue
            try:
                checkpoint_roots.add(int(parts[1]))
            except (TypeError, ValueError):
                pass
        for row in state.get("directories") or []:
            if isinstance(row, dict) and row.get("root_id") is not None:
                try:
                    directory_roots.add(int(row["root_id"]))
                except (TypeError, ValueError):
                    pass
        return checkpoint_roots & directory_roots

    def _restore_usn_state_from_control(
        self,
        candidate_db: DatabaseManager,
    ) -> None:
        """Hydrate candidate-only USN metadata from the control database."""

        if (
            self.control_db_path.resolve() == self.db_path.resolve()
            or not self.control_db_path.is_file()
        ):
            return
        try:
            source = DatabaseManager(self.control_db_path, read_only=True)
            state = source.export_usn_state()
            root_ids = self._usn_state_ready_root_ids(state)
            if not root_ids:
                return
            candidate_db.import_usn_state(state, root_ids=sorted(root_ids))
            log_event(
                "index.usn_state.restored",
                component="scan_worker",
                scan_id=self.scan_id,
                root_count=len(root_ids),
                checkpoint_count=len(state.get("checkpoints") or []),
                directory_count=len(state.get("directories") or []),
                control_db_path=str(self.control_db_path),
            )
        except Exception as exc:
            # Missing or stale USN metadata is safe: the index manager will
            # fall back to a complete directory comparison for that root.
            log_exception(
                "index.usn_state.restore_failed",
                exc,
                component="scan_worker",
                scan_id=self.scan_id,
                control_db_path=str(self.control_db_path),
            )

    def _persist_usn_state_to_control(
        self,
        state: dict[str, object],
        *,
        root_ids: Sequence[int],
        candidate_generation: int,
    ) -> None:
        """Persist USN metadata only after the candidate was published."""

        if (
            self.control_db_path.resolve() == self.db_path.resolve()
            or not self.control_db_path.is_file()
            or not root_ids
        ):
            return
        try:
            ready_root_ids = self._usn_state_ready_root_ids(state)
            ready_root_ids &= {int(value) for value in root_ids}
            if not ready_root_ids:
                return
            control = DatabaseManager(self.control_db_path)
            control.import_usn_state(state, root_ids=sorted(ready_root_ids))
            log_event(
                "index.usn_state.persisted",
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=candidate_generation,
                root_count=len(ready_root_ids),
                checkpoint_count=len(state.get("checkpoints") or []),
                directory_count=len(state.get("directories") or []),
                control_db_path=str(self.control_db_path),
            )
        except Exception as exc:
            # Publication is already durable. A future run can safely fall
            # back to directory comparison if this auxiliary synchronization
            # is temporarily blocked by another control-DB writer.
            log_exception(
                "index.usn_state.persist_failed",
                exc,
                component="scan_worker",
                scan_id=self.scan_id,
                candidate_generation=candidate_generation,
                root_ids=list(ready_root_ids),
                control_db_path=str(self.control_db_path),
            )

    @Slot()
    def cancel(
        self,
        *,
        force: bool = False,
        reason: str = "USER_CANCELLED",
    ) -> None:
        self.token.cancel(force=force, reason=reason)
        if force and self.manager is not None:
            self.manager.force_terminate_processes()
        if force and self.database is not None:
            self.database.interrupt_active_connections()

    @Slot()
    def pause(self) -> None:
        self.request_coordinator_pause()
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
        self.request_coordinator_resume()
        with self._pause_monitor_lock:
            self._pause_monitor_generation += 1
        self.progress.emit(
            {
                "stage": "resuming",
                "pause_state": "resuming",
                "phase_label": "正在从安全检查点继续",
            }
        )

    def request_coordinator_pause(self) -> None:
        """Pause scheduling without emitting a user-visible manual pause."""

        self.token.pause()
        if self.manager is not None:
            self.manager.request_pause()

    def request_coordinator_resume(self) -> None:
        """Resume only the pause created for a completed foreground search."""

        if self.manager is not None:
            self.manager.request_resume()
        self.token.resume()

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
