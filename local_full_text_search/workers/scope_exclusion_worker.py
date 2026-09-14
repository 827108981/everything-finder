from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken

logger = logging.getLogger(__name__)


class ScopeExclusionWorker(QObject):
    """Apply manual scope exclusions without blocking the Qt main thread."""

    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        db_path: Path,
        file_ids: list[int],
        *,
        reason: str,
        operation_source: str = "ui",
        force_complete: bool = False,
        generation_base_path: Path | None = None,
        report_only_file_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.file_ids = list(dict.fromkeys(int(value) for value in file_ids))
        self.reason = str(reason)
        self.operation_source = str(operation_source)
        self.force_complete = bool(force_complete)
        self.generation_base_path = Path(
            generation_base_path or self.db_path
        ).resolve()
        self.report_only_file_ids = {
            int(value) for value in (report_only_file_ids or ())
        }
        self.task_id = uuid.uuid4().hex
        self.token = CancelToken()
        self._database: DatabaseManager | None = None
        self._database_lock = threading.Lock()
        self._started_at = 0.0
        self._phase_started_at = 0.0
        self._current_stage = ""
        self._phase_durations_ms: dict[str, int] = {}
        self._triggered_full_fts_rebuild = False

    @Slot()
    def run(self) -> None:
        self._started_at = time.monotonic()
        self._phase_started_at = self._started_at
        try:
            self.token.throw_if_cancelled()
            database = DatabaseManager(self.db_path)
            with self._database_lock:
                self._database = database
            generation_store = IndexGenerationStore(self.generation_base_path)
            failure_report = generation_store.read_failure_report()
            report_rows = [
                dict(row)
                for row in (failure_report or {}).get("blocking_files", [])
                if isinstance(row, dict)
            ]
            report_ids = {
                int(row["id"])
                for row in report_rows
                if row.get("id") is not None
            }
            if self.report_only_file_ids:
                unknown = self.report_only_file_ids - report_ids
                if unknown:
                    raise ValueError(
                        "失败报告中的选项已变化："
                        + ", ".join(str(value) for value in sorted(unknown))
                    )
            else:
                self.report_only_file_ids = {
                    value for value in self.file_ids if value in report_ids
                }
            real_file_ids = [
                value for value in self.file_ids
                if value not in self.report_only_file_ids
            ]
            if self.report_only_file_ids:
                logger.info(
                    "scope_exclusion.report_only.started task_id=%s "
                    "report_path=%s ids=%s operation=%s",
                    self.task_id,
                    generation_store.failure_report_path,
                    sorted(self.report_only_file_ids),
                    "force_complete" if self.force_complete else "exclude",
                )
            self._emit_progress(
                {
                    "stage": "validating",
                    "phase_label": "正在校验选中文件",
                    "processed_files": 0,
                    "total_files": len(self.file_ids),
                    "large_fts_operation": False,
                    "can_cancel": True,
                }
            )
            if self.force_complete:
                if real_file_ids or not self.report_only_file_ids:
                    repair_result = database.force_complete_current_scope(
                        file_ids=real_file_ids,
                        reason=self.reason,
                        operation_source=self.operation_source,
                        progress_callback=self._emit_progress,
                        cancel_requested=lambda: self.token.cancelled,
                    )
                    changed = int(repair_result.get("excluded_files") or 0)
                else:
                    # An empty control-DB selection means "all remaining
                    # blockers" to force_complete_current_scope.  Do not
                    # invoke it for a report-only selection, or an unrelated
                    # control-DB blocker could be excluded accidentally.
                    repair_result = {}
                    changed = 0
            else:
                repair_result = {}
                changed = database.exclude_files_from_index(
                    real_file_ids,
                    reason=self.reason,
                    operation_source=self.operation_source,
                    progress_callback=self._emit_progress,
                    cancel_requested=lambda: self.token.cancelled,
                )
            report_result: dict[str, object] = {}
            if self.report_only_file_ids:
                self.token.throw_if_cancelled()
                report_result = generation_store.mark_failure_report_items_excluded(
                    self.report_only_file_ids,
                    reason=self.reason,
                    operation_source=self.operation_source,
                )
                changed += int(report_result.get("changed") or 0)
                if not real_file_ids and not int(
                    report_result.get("remaining_blocking_files") or 0
                ):
                    resolved_root_ids = {
                        int(row["root_id"])
                        for row in report_result.get("changed_rows", [])
                        if isinstance(row, dict) and row.get("root_id") is not None
                    }
                    for root_id in sorted(resolved_root_ids):
                        database.update_root_scan_time(root_id, "ready")
            self._finish_current_phase()
            readiness = database.index_readiness()
            current_report = generation_store.read_failure_report() or {}
            remaining_report_blockers = [
                row
                for row in (current_report.get("blocking_files") or [])
                if isinstance(row, dict)
            ]
            report_unready_roots = list(
                current_report.get("unready_roots") or []
            )
            real_ready = bool(readiness.get("ready"))
            if self.report_only_file_ids and not real_file_ids:
                # Report-only blockers are intentionally absent from the
                # control DB.  Its root/file readiness cannot veto an action
                # that only updates the already-published compact snapshot.
                active = generation_store.active()
                result_ready = bool(
                    active is not None
                    and not remaining_report_blockers
                    and not report_unready_roots
                )
            else:
                result_ready = bool(
                    real_ready and not remaining_report_blockers
                )
            with database.connect() as connection:
                counts = connection.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM content_blocks) AS content_blocks,
                        (SELECT COUNT(*) FROM content_fts) AS content_fts_rows,
                        (SELECT COUNT(*) FROM files_fts) AS files_fts_rows
                    """
                ).fetchone()
            result = {
                "task_id": self.task_id,
                "operation": "force_complete" if self.force_complete else "exclude",
                "selected_files": len(self.file_ids),
                "excluded_files": int(changed),
                "report_only_files": len(self.report_only_file_ids),
                "report_only_excluded_files": int(
                    report_result.get("changed") or 0
                ),
                "triggered_full_fts_rebuild": self._triggered_full_fts_rebuild,
                "ready": result_ready,
                "blocking_files": int(readiness.get("blocking_files") or 0)
                + len(remaining_report_blockers),
                "manual_excluded_files": int(
                    readiness.get("manual_excluded_files") or 0
                ),
                "database_size_bytes": (
                    self.db_path.stat().st_size if self.db_path.is_file() else 0
                ),
                "content_blocks": int(counts["content_blocks"] or 0),
                "content_fts_rows": int(counts["content_fts_rows"] or 0),
                "files_fts_rows": int(counts["files_fts_rows"] or 0),
                "phase_durations_ms": dict(self._phase_durations_ms),
                "total_ms": int((time.monotonic() - self._started_at) * 1000),
                **repair_result,
                **report_result,
            }
            if self.report_only_file_ids:
                logger.info(
                    "scope_exclusion.report_only.completed task_id=%s "
                    "report_path=%s changed=%s remaining=%s ready=%s",
                    self.task_id,
                    generation_store.failure_report_path,
                    int(report_result.get("changed") or 0),
                    len(remaining_report_blockers),
                    result_ready,
                )
            logger.info("Scope exclusion completed: %s", result)
            self.finished.emit(result)
        except CancelledError:
            self._finish_current_phase()
            logger.info(
                "Scope exclusion cancelled and rolled back: task_id=%s selected=%s total_ms=%s",
                self.task_id,
                len(self.file_ids),
                int((time.monotonic() - self._started_at) * 1000),
            )
            self.cancelled.emit()
        except Exception as exc:
            self._finish_current_phase()
            if self.token.cancelled:
                logger.info(
                    "Scope exclusion interrupted and rolled back: task_id=%s selected=%s total_ms=%s",
                    self.task_id,
                    len(self.file_ids),
                    int((time.monotonic() - self._started_at) * 1000),
                )
                self.cancelled.emit()
            else:
                logger.exception(
                    "Scope exclusion failed and rolled back: task_id=%s selected=%s database_locked=%s",
                    self.task_id,
                    len(self.file_ids),
                    "locked" in str(exc).lower(),
                )
                if self.report_only_file_ids:
                    logger.exception(
                        "scope_exclusion.report_only.failed task_id=%s "
                        "report_path=%s ids=%s",
                        self.task_id,
                        self.generation_base_path,
                        sorted(self.report_only_file_ids),
                    )
                self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            with self._database_lock:
                self._database = None

    @Slot()
    def cancel(self) -> None:
        self.token.cancel()
        with self._database_lock:
            database = self._database
        if database is not None:
            database.interrupt_active_connections()

    def _emit_progress(self, payload: dict[str, object]) -> None:
        stage = str(payload.get("stage") or "updating_index")
        now = time.monotonic()
        if stage != self._current_stage:
            self._finish_current_phase(now)
            self._current_stage = stage
            self._phase_started_at = now
            logger.info(
                "Scope exclusion phase started: task_id=%s stage=%s selected=%s",
                self.task_id,
                stage,
                len(self.file_ids),
            )
        if stage == "rebuilding_content_fts":
            self._triggered_full_fts_rebuild = True
        self.progress.emit(
            {
                **payload,
                "task_id": self.task_id,
                "elapsed_seconds": int(max(0.0, now - self._started_at)),
            }
        )

    def _finish_current_phase(self, now: float | None = None) -> None:
        if not self._current_stage:
            return
        finished_at = time.monotonic() if now is None else now
        elapsed_ms = int(max(0.0, finished_at - self._phase_started_at) * 1000)
        self._phase_durations_ms[self._current_stage] = (
            self._phase_durations_ms.get(self._current_stage, 0) + elapsed_ms
        )
        logger.info(
            "Scope exclusion phase finished: task_id=%s stage=%s elapsed_ms=%s",
            self.task_id,
            self._current_stage,
            elapsed_ms,
        )
        self._current_stage = ""
