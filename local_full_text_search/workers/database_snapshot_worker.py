from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from PySide6.QtCore import QThread, QObject, Signal, Slot

from local_full_text_search.config.constants import VIDEO_EXTENSIONS
from local_full_text_search.core.database import (
    DatabaseManager,
    FINAL_FAILURE_STATUSES,
)
from local_full_text_search.core.compact_snapshot_schema import is_compact_snapshot
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.services.logging_service import log_event, log_exception


logger = logging.getLogger(__name__)


def _format_snapshot_timestamp(value: object) -> str:
    """Return a stable ISO timestamp for compact metadata and legacy rows."""

    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return str(value)
    return str(value)


def collect_compact_database_snapshot(
    path: Path,
    *,
    page_offsets: dict[str, int],
) -> dict[str, object]:
    """Collect the small, published snapshot state without task-table scans."""

    con = sqlite3.connect(
        f"file:{Path(path).resolve().as_posix()}?mode=ro",
        uri=True,
        timeout=2.0,
    )
    con.row_factory = sqlite3.Row
    try:
        roots = [dict(row) for row in con.execute("SELECT * FROM roots ORDER BY id")]
        readiness_row = con.execute(
            "SELECT * FROM snapshot_meta WHERE id = 1"
        ).fetchone()
        if readiness_row is None or int(readiness_row["state"]) != 1:
            raise RuntimeError("发布快照尚未完成")
        published_at = _format_snapshot_timestamp(readiness_row["published_at"])
        for root in roots:
            # Compact snapshots intentionally omit mutable operational fields.
            # Supply stable display defaults so the management UI can consume
            # either the operational or published-root row contract.
            root.setdefault("last_scan_at", published_at)
            root.setdefault("updated_at", published_at)
            root.setdefault("created_at", published_at)
            root.setdefault("exclude_rules_json", None)
        files = int(con.execute("SELECT COUNT(*) FROM files").fetchone()[0])
        blocks = int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0])
        documents = int(con.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
        root_stats: dict[int, dict[str, int]] = {}
        for row in con.execute(
            """
            SELECT root_id, COALESCE(extension, '') AS extension, COUNT(*) AS n
            FROM files GROUP BY root_id, extension
            """
        ):
            root_stats.setdefault(int(row["root_id"]), {})[str(row["extension"])] = int(row["n"])
        readiness = {
            "quality": "complete",
            "stats_complete": True,
            "file_count": files,
            "eligible_files": files,
            "complete_files": files,
            "blocking_files": 0,
            "metadata_only_files": int(
                con.execute(
                    "SELECT COUNT(*) FROM files WHERE parse_status = 'metadata_only'"
                ).fetchone()[0]
            ),
            "video_excluded": 0,
            "manual_excluded_files": 0,
            "block_count": blocks,
            "document_count": documents,
            "enabled_roots": len(roots),
            "active_runs": 0,
            "unfinished_tasks": 0,
            "unpublished_candidates": 0,
            "content_fts_dirty": False,
            "full_batch_incomplete": False,
            "search_available": True,
            "search_is_partial": False,
            "index_complete": True,
            "ready": True,
            "active_generation": int(readiness_row["generation"]),
            "published_at": published_at,
        }
        return {
            "roots": roots,
            "stats": readiness,
            "readiness": readiness,
            "root_stats_by_id": root_stats,
            "failed_counts": {"blocking": 0, "excluded": 0, "metadata": 0},
            "failed_rows": [],
            "all_blocking_ids": [],
            "excluded_rows": [],
            "metadata_rows": [],
            "history": [],
            "timings_ms": {},
            "total_ms": 0,
            "compact_snapshot": True,
        }
    finally:
        con.close()


def collect_live_candidate_failure_snapshot(
    path: Path,
    *,
    page_size: int,
    page_offsets: dict[str, int],
    generation: int | None = None,
    scan_id: str = "",
    snapshot_sequence: int = 0,
) -> dict[str, object] | None:
    """Read current failure rows from the unpublished operational candidate.

    The published compact snapshot intentionally contains no mutable failure
    rows.  During an active scan the candidate is still safe to read through a
    WAL read-only connection, so the management page can show failures that
    have already been committed without exposing the candidate to search.
    """

    candidate_path = Path(path).resolve()
    if not candidate_path.is_file():
        return None
    started = time.perf_counter()
    try:
        candidate = DatabaseManager(candidate_path, read_only=True)
        limit = max(1, int(page_size))
        statuses = tuple(sorted(FINAL_FAILURE_STATUSES))
        in_progress_statuses = ("pending", "processing", "cancelled")
        status_placeholders = ",".join("?" for _ in statuses)
        in_progress_placeholders = ",".join("?" for _ in in_progress_statuses)
        video_placeholders = ",".join("?" for _ in VIDEO_EXTENSIONS)
        scope_exclusion = """
            NOT EXISTS (
                SELECT 1 FROM index_scope_exclusions e
                WHERE e.file_id = f.id
                  AND e.revoked_at IS NULL
                  AND e.invalidated_at IS NULL
            )
        """
        with candidate.connect_read_only(
            timeout_seconds=2.0,
            operation="candidate_failure_snapshot",
        ) as con:
            # One read transaction keeps counts, the current page and the
            # all-selection IDs on the same candidate version.
            con.execute("BEGIN")
            blocking = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (
                          f.extension IN ({video_placeholders})
                          AND f.parse_status = 'metadata_only'
                      )
                      AND {scope_exclusion}
                    """,
                    (*statuses, *sorted(VIDEO_EXTENSIONS)),
                ).fetchone()[0]
            )
            in_progress = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({in_progress_placeholders})
                      AND NOT (
                          f.extension IN ({video_placeholders})
                          AND f.parse_status = 'metadata_only'
                      )
                      AND {scope_exclusion}
                    """,
                    (*in_progress_statuses, *sorted(VIDEO_EXTENSIONS)),
                ).fetchone()[0]
            )
            rows = [
                dict(row)
                for row in con.execute(
                    f"""
                    SELECT f.id, f.path, f.filename, f.extension, f.parse_status,
                           f.parse_error_code, f.parse_error_message,
                           f.parse_diagnostics_json, f.parser_name, f.indexed_at,
                           'included' AS scope_state,
                           COALESCE(pt.progress_phase, '') AS progress_phase,
                           COALESCE(pt.progress_cursor, '') AS progress_cursor,
                           CASE
                               WHEN f.parse_error_code = 'PARSE_NO_PROGRESS'
                                   THEN '检查文件是否损坏或位于不稳定存储；修复后重新尝试'
                               WHEN f.parse_error_code = 'PROCESS_WORKER_CRASH'
                                   THEN '关闭占用该文件的程序后重试；若重复发生请保留诊断日志'
                               ELSE '根据错误原因修复源文件或依赖后重新尝试'
                           END AS recovery_advice
                    FROM files f
                    LEFT JOIN parse_tasks pt ON pt.id = (
                        SELECT latest.id FROM parse_tasks latest
                        WHERE latest.file_id = f.id
                        ORDER BY latest.id DESC LIMIT 1
                    )
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (
                          f.extension IN ({video_placeholders})
                          AND f.parse_status = 'metadata_only'
                      )
                      AND {scope_exclusion}
                    ORDER BY f.indexed_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (
                        *statuses,
                        *sorted(VIDEO_EXTENSIONS),
                        limit,
                        max(0, int(page_offsets.get("blocking", 0))),
                    ),
                ).fetchall()
            ]
            all_ids = [
                int(row[0])
                for row in con.execute(
                    f"""
                    SELECT f.id FROM files f
                    WHERE f.is_deleted = 0
                      AND f.parse_status IN ({status_placeholders})
                      AND NOT (
                          f.extension IN ({video_placeholders})
                          AND f.parse_status = 'metadata_only'
                      )
                      AND {scope_exclusion}
                    ORDER BY f.id
                    """,
                    (*statuses, *sorted(VIDEO_EXTENSIONS)),
                ).fetchall()
            ]
    except sqlite3.OperationalError as exc:
        log_event(
            "database.candidate_snapshot.unavailable",
            level=logging.WARNING,
            component="database_snapshot_worker",
            database_path=str(candidate_path),
            candidate_generation=generation,
            error_type=type(exc).__name__,
            error_message=str(exc)[:1000],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
        return None
    except Exception as exc:
        logger.warning(
            "Unable to read live candidate failure rows: path=%s",
            candidate_path,
            exc_info=True,
        )
        log_exception(
            "database.candidate_snapshot.failed",
            exc,
            component="database_snapshot_worker",
            database_path=str(candidate_path),
            candidate_generation=generation,
        )
        return None
    snapshot = {
        "candidate_live": True,
        "candidate_failure_source": "live_candidate_database",
        "candidate_database_path": str(candidate_path),
        "candidate_generation": generation,
        "candidate_scan_id": str(scan_id or ""),
        "candidate_snapshot_sequence": max(0, int(snapshot_sequence)),
        "candidate_snapshot_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "failed_counts": {"blocking": blocking, "in_progress": in_progress},
        "failed_rows": rows,
        "all_blocking_ids": all_ids,
        "candidate_snapshot_elapsed_ms": int(
            (time.perf_counter() - started) * 1000
        ),
    }
    log_event(
        "database.candidate_snapshot.completed",
        component="database_snapshot_worker",
        database_path=str(candidate_path),
        candidate_generation=generation,
        blocking_files=int(blocking),
        page_rows=len(rows),
        all_blocking_ids=len(all_ids),
        elapsed_ms=int((time.perf_counter() - started) * 1000),
    )
    return snapshot


def merge_control_snapshot_with_published(
    control_snapshot: dict[str, object],
    published_snapshot: dict[str, object],
) -> dict[str, object]:
    """Combine mutable scope/task state with immutable published statistics."""

    control_roots = [dict(row) for row in control_snapshot.get("roots") or []]
    published_roots = {
        int(row["id"]): dict(row)
        for row in (published_snapshot.get("roots") or [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    roots: list[dict[str, object]] = []
    for root in control_roots:
        published = published_roots.get(int(root["id"]))
        if published is not None:
            for field in ("created_at", "updated_at", "last_scan_at"):
                if not root.get(field) and published.get(field):
                    root[field] = published[field]
        roots.append(root)
    if not roots:
        roots = [dict(row) for row in (published_snapshot.get("roots") or [])]

    control_readiness = dict(control_snapshot.get("readiness") or {})
    published_readiness = dict(published_snapshot.get("readiness") or {})
    readiness = dict(published_readiness)
    # The control DB owns mutable task and scope state.  The published
    # snapshot owns file/document/block counts and search completeness.
    for key in (
        "active_runs",
        "unfinished_tasks",
        "orphan_pending_files",
        "unpublished_candidates",
        "background_task_state",
        "failure_alert_ready",
    ):
        if key in control_readiness:
            readiness[key] = control_readiness[key]
    enabled_roots = sum(1 for row in roots if int(row.get("enabled") or 0) == 1)
    unready_roots = sum(
        1
        for row in roots
        if int(row.get("enabled") or 0) == 1
        and str(row.get("status") or "") not in {"ready", ""}
    )
    readiness["enabled_roots"] = enabled_roots
    readiness["unready_roots"] = unready_roots
    readiness["search_available"] = True
    if unready_roots or int(readiness.get("active_runs") or 0) > 0:
        # The last published snapshot remains searchable, but the latest
        # control scope is not yet completely represented in it.
        readiness["ready"] = False
        readiness["index_complete"] = False

    merged = dict(control_snapshot)
    merged["roots"] = roots
    merged["root_stats_by_id"] = dict(
        published_snapshot.get("root_stats_by_id") or {}
    )
    merged["stats"] = dict(readiness)
    merged["readiness"] = readiness
    merged["active_generation"] = readiness.get("active_generation")
    merged["active_database_path"] = published_snapshot.get(
        "active_database_path"
    )
    merged["published_snapshot"] = True
    return merged


def merge_failure_report(
    snapshot: dict[str, object],
    failure_report: dict[str, object] | None,
) -> dict[str, object]:
    """Expose the last candidate blockers even after its DB was removed."""

    if not failure_report:
        return snapshot
    blocking_rows = [
        dict(row)
        for row in (failure_report.get("blocking_files") or [])
        if isinstance(row, dict)
    ]
    excluded_report_rows = [
        dict(row)
        for row in (failure_report.get("excluded_files") or [])
        if isinstance(row, dict)
    ]
    if not blocking_rows and not excluded_report_rows and not (
        failure_report.get("unready_roots") or []
    ):
        return snapshot

    merged = dict(snapshot)
    existing_rows = [dict(row) for row in (merged.get("failed_rows") or [])]
    existing_ids = {
        int(row["id"])
        for row in existing_rows
        if row.get("id") is not None
    }
    normalized_rows = []
    for row in blocking_rows:
        if row.get("id") is not None and int(row["id"]) in existing_ids:
            continue
        row.setdefault("scope_state", "included")
        row.setdefault("progress_phase", "")
        row.setdefault("progress_cursor", "")
        row.setdefault(
            "recovery_advice",
            "根据错误原因处理源文件或调整索引范围后重新尝试",
        )
        normalized_rows.append(row)
    merged["failed_rows"] = existing_rows + normalized_rows

    existing_excluded = [
        dict(row) for row in (merged.get("excluded_rows") or [])
    ]
    existing_excluded_ids = {
        int(row["file_id"] if row.get("file_id") is not None else row["id"])
        for row in existing_excluded
        if row.get("file_id") is not None or row.get("id") is not None
    }
    normalized_excluded = []
    for row in excluded_report_rows:
        row.setdefault("scope_state", "manual_excluded")
        row.setdefault("reason", row.get("exclusion_reason") or "用户确认排除")
        row.setdefault("current_parse_status", row.get("parse_status") or "")
        row.setdefault("current_error_code", row.get("parse_error_code") or "")
        row.setdefault("current_error_message", row.get("parse_error_message") or "")
        row.setdefault("file_id", row.get("id"))
        row.setdefault("created_at", row.get("excluded_at") or "")
        row_id = row.get("file_id")
        if row_id is not None and int(row_id) in existing_excluded_ids:
            continue
        normalized_excluded.append(row)
    merged["excluded_rows"] = existing_excluded + normalized_excluded

    counts = dict(merged.get("failed_counts") or {})
    report_readiness = dict(failure_report.get("readiness") or {})
    counts["blocking"] = max(
        int(counts.get("blocking") or 0),
        len(blocking_rows),
        int(report_readiness.get("blocking_files") or 0),
    )
    counts["excluded"] = max(
        int(counts.get("excluded") or 0),
        len(excluded_report_rows),
    )
    merged["failed_counts"] = counts
    merged["all_blocking_ids"] = sorted(
        {
            *(
                int(value)
                for value in (merged.get("all_blocking_ids") or [])
            ),
            *(
                int(row["id"])
                for row in blocking_rows
                if row.get("id") is not None
            ),
        }
    )
    merged["last_failure_report"] = dict(failure_report)

    readiness = dict(merged.get("readiness") or merged.get("stats") or {})
    readiness["latest_candidate_failure"] = True
    readiness["candidate_failure_generation"] = failure_report.get(
        "candidate_generation"
    )
    readiness["candidate_failure_reason"] = failure_report.get("error")
    readiness["blocking_files"] = max(
        int(readiness.get("blocking_files") or 0),
        len(blocking_rows),
        int(report_readiness.get("blocking_files") or 0),
    )
    if report_readiness:
        readiness["blocking_status_counts"] = dict(
            report_readiness.get("blocking_status_counts") or {}
        )
        readiness["blocking_file_samples"] = blocking_rows
        readiness["unready_root_samples"] = list(
            failure_report.get("unready_roots") or []
        )
        if bool(failure_report.get("published_partial")):
            # A partial generation is now the active searchable snapshot, so
            # its candidate counters describe the current scope better than
            # the compact file count alone (the skipped files are omitted
            # from the compact projection by design).
            for key in (
                "files",
                "discovered_files",
                "blocks",
                "documents",
                "eligible_files",
                "complete_files",
                "successful_files",
                "metadata_only_complete_files",
                "video_excluded",
                "manual_excluded_files",
            ):
                if key in report_readiness:
                    readiness[key] = report_readiness[key]
        if excluded_report_rows:
            readiness["manual_excluded_files"] = max(
                int(readiness.get("manual_excluded_files") or 0),
                int(report_readiness.get("manual_excluded_files") or 0),
                len(excluded_report_rows),
            )
    readiness["search_available"] = bool(
        readiness.get("search_available")
        or merged.get("published_snapshot")
    )
    if blocking_rows or report_readiness.get("blocking_files"):
        readiness["ready"] = False
        readiness["index_complete"] = False
    elif failure_report.get("report_resolved_at"):
        # The active snapshot is immutable and already omits the manually
        # excluded report-only files. Once the report decision is persisted,
        # it represents a complete searchable scope.
        readiness["latest_candidate_failure"] = False
        readiness["blocking_files"] = int(readiness.get("blocking_files") or 0)
        readiness["ready"] = bool(readiness.get("search_available"))
        readiness["index_complete"] = bool(readiness.get("search_available"))
    merged["readiness"] = readiness
    merged["stats"] = dict(readiness)
    return merged


def merge_live_candidate_failures(
    snapshot: dict[str, object],
    candidate_snapshot: dict[str, object] | None,
) -> dict[str, object]:
    """Overlay live candidate failures while preserving published statistics."""

    if not candidate_snapshot or not bool(candidate_snapshot.get("candidate_live")):
        return snapshot
    merged = dict(snapshot)
    candidate_counts = dict(candidate_snapshot.get("failed_counts") or {})
    counts = dict(merged.get("failed_counts") or {})
    # The candidate is the authoritative failure view for this active run;
    # do not retain rows from an older failure report over the live candidate.
    counts["blocking"] = int(candidate_counts.get("blocking") or 0)
    counts["in_progress"] = int(candidate_counts.get("in_progress") or 0)
    merged["failed_counts"] = counts
    merged["failed_rows"] = [
        dict(row)
        for row in (candidate_snapshot.get("failed_rows") or [])
        if isinstance(row, dict)
    ]
    merged["all_blocking_ids"] = [
        int(value) for value in (candidate_snapshot.get("all_blocking_ids") or [])
    ]
    merged["candidate_live_failures"] = True
    merged["candidate_failure_source"] = candidate_snapshot.get(
        "candidate_failure_source"
    )
    merged["candidate_database_path"] = candidate_snapshot.get(
        "candidate_database_path"
    )
    merged["candidate_generation"] = candidate_snapshot.get("candidate_generation")
    merged["candidate_scan_id"] = candidate_snapshot.get("candidate_scan_id")
    merged["candidate_snapshot_sequence"] = candidate_snapshot.get(
        "candidate_snapshot_sequence"
    )
    merged["candidate_snapshot_at"] = candidate_snapshot.get("candidate_snapshot_at")

    readiness = dict(merged.get("readiness") or merged.get("stats") or {})
    readiness["candidate_live"] = True
    readiness["candidate_live_failures"] = True
    readiness["candidate_failure_source"] = candidate_snapshot.get(
        "candidate_failure_source"
    )
    readiness["candidate_database_path"] = candidate_snapshot.get(
        "candidate_database_path"
    )
    readiness["candidate_generation"] = candidate_snapshot.get("candidate_generation")
    readiness["candidate_scan_id"] = candidate_snapshot.get("candidate_scan_id")
    readiness["candidate_snapshot_sequence"] = candidate_snapshot.get(
        "candidate_snapshot_sequence"
    )
    readiness["candidate_snapshot_at"] = candidate_snapshot.get("candidate_snapshot_at")
    readiness["candidate_snapshot_elapsed_ms"] = candidate_snapshot.get(
        "candidate_snapshot_elapsed_ms"
    )
    readiness["active_runs"] = max(1, int(readiness.get("active_runs") or 0))
    readiness["scan_active"] = True
    readiness["background_task_state"] = "running"
    readiness["blocking_files"] = int(candidate_counts.get("blocking") or 0)
    readiness["in_progress_files"] = int(candidate_counts.get("in_progress") or 0)
    readiness["blocking_file_samples"] = list(merged.get("failed_rows") or [])
    readiness["blocking_details_complete"] = len(
        merged.get("failed_rows") or []
    ) >= readiness["blocking_files"]
    readiness["latest_candidate_failure"] = bool(readiness["blocking_files"])
    readiness["ready"] = False
    readiness["index_complete"] = False
    reasons = [str(value) for value in (readiness.get("not_ready_reasons") or [])]
    if "active_run" not in reasons:
        reasons.append("active_run")
    if readiness["blocking_files"] and "blocking_files" not in reasons:
        reasons.append("blocking_files")
    readiness["not_ready_reasons"] = reasons
    merged["readiness"] = readiness
    merged["stats"] = dict(readiness)
    return merged


def collect_database_snapshot(
    db: DatabaseManager,
    *,
    page_size: int,
    page_offsets: dict[str, int],
    fallback_snapshot: dict[str, object] | None = None,
) -> dict[str, object]:
    """Collect the expensive management-page state away from the UI thread."""

    timings: dict[str, int] = {}

    def measured(name: str, operation: object) -> object:
        started = time.perf_counter()
        value = operation()  # type: ignore[operator]
        timings[name] = int((time.perf_counter() - started) * 1000)
        return value

    roots = [
        dict(row)
        for row in measured("roots", db.list_roots)  # type: ignore[arg-type]
    ]
    # Snapshot collection is strictly read-only. Reconciliation writes are
    # performed by the index lifecycle, never from the paging/statistics
    # worker, so a long FTS commit cannot deadlock the management page.
    def read_readiness() -> dict[str, object]:
        try:
            return dict(db.index_readiness(reconcile=False))
        except TypeError as exc:
            # Keep compatibility with test doubles and older plug-ins that
            # expose the pre-reconcile signature.
            if "reconcile" not in str(exc):
                raise
            return dict(db.index_readiness())

    readiness = dict(measured("readiness", read_readiness))
    # During an active index run, a detailed readiness scan competes with the
    # same large tables that FTS is writing. Reuse the last complete snapshot
    # and only refresh the cheap quick gate; the next idle refresh will collect
    # full counts again.
    if (
        fallback_snapshot
        and int(readiness.get("active_runs") or 0) > 0
    ):
        cached = dict(fallback_snapshot)
        cached["readiness"] = readiness
        cached["stats"] = readiness
        cached["readiness_degraded"] = True
        cached["timings_ms"] = timings
        cached["total_ms"] = sum(timings.values())
        return cached

    def root_stats() -> dict[int, dict[str, int]]:
        data: dict[int, dict[str, int]] = {}
        with db.connect_read_only() as connection:
            rows = connection.execute(
                """
                SELECT root_id, extension, COUNT(*) AS n
                FROM files
                WHERE is_deleted = 0
                GROUP BY root_id, extension
                """
            ).fetchall()
        for row in rows:
            root_id = int(row["root_id"])
            extension = str(row["extension"] or "")
            data.setdefault(root_id, {})[extension] = int(row["n"])
        return data

    root_stats_by_id = measured("root_stats", root_stats)
    limit = max(1, int(page_size))
    failed_snapshot_reader = getattr(db, "failed_snapshot", None)
    if callable(failed_snapshot_reader):
        failure_snapshot = dict(
            measured(
                "failed_snapshot",
                lambda: failed_snapshot_reader(
                    page_size=limit,
                    offset=max(0, int(page_offsets.get("blocking", 0))),
                ),
            )
        )
        failed_counts = dict(failure_snapshot.get("failed_counts") or {})
        failed_rows = [
            dict(row) for row in (failure_snapshot.get("failed_rows") or [])
        ]
        all_blocking_ids = [
            int(value)
            for value in (failure_snapshot.get("all_blocking_ids") or [])
        ]
    else:
        failed_counts = dict(
            measured("failed_counts", db.failed_file_counts)  # type: ignore[arg-type]
        )
        failed_rows = [
            dict(row)
            for row in measured(
                "failed_rows",
                lambda: db.failed_files(
                    limit=limit,
                    offset=max(0, int(page_offsets.get("blocking", 0))),
                ),
            )  # type: ignore[arg-type]
        ]
        all_blocking_ids = list(
            measured("failed_ids", db.failed_file_ids)  # type: ignore[arg-type]
        )
    excluded_rows = [
        dict(row)
        for row in measured(
            "excluded_rows",
            lambda: db.excluded_files(
                limit=limit,
                offset=max(0, int(page_offsets.get("excluded", 0))),
            ),
        )  # type: ignore[arg-type]
    ]
    metadata_rows = [
        dict(row)
        for row in measured(
            "metadata_rows",
            lambda: db.metadata_only_files(
                limit=limit,
                offset=max(0, int(page_offsets.get("metadata", 0))),
            ),
        )  # type: ignore[arg-type]
    ]
    history = list(
        measured("history", db.search_history)  # type: ignore[arg-type]
    )
    return {
        "roots": roots,
        # index_readiness already contains the aggregate stats. Avoid the
        # duplicate full-table scan previously performed by db.stats().
        "stats": readiness,
        "readiness": readiness,
        "root_stats_by_id": root_stats_by_id,
        "failed_counts": failed_counts,
        "failed_rows": failed_rows,
        "all_blocking_ids": all_blocking_ids,
        "excluded_rows": excluded_rows,
        "metadata_rows": metadata_rows,
        "history": history,
        "timings_ms": timings,
        "total_ms": sum(timings.values()),
    }


class DatabaseSnapshotWorker(QObject):
    quick_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        db_path: Path,
        *,
        page_size: int,
        page_offsets: dict[str, int],
        fallback_snapshot: dict[str, object] | None = None,
        published_snapshot_path: Path | None = None,
        generation_base_path: Path | None = None,
        snapshot_timeout_seconds: float = 30.0,
        scan_id: str = "",
        snapshot_sequence: int = 0,
    ) -> None:
        super().__init__()
        self.db = DatabaseManager(Path(db_path), read_only=True)
        self.published_snapshot_path = (
            Path(published_snapshot_path)
            if published_snapshot_path is not None
            else None
        )
        self.generation_base_path = (
            Path(generation_base_path)
            if generation_base_path is not None
            else None
        )
        self.page_size = max(1, int(page_size))
        self.page_offsets = dict(page_offsets)
        self.fallback_snapshot = dict(fallback_snapshot or {})
        self.snapshot_timeout_seconds = max(1.0, float(snapshot_timeout_seconds))
        self.scan_id = str(scan_id or "")
        self.snapshot_sequence = max(0, int(snapshot_sequence))
        self._cancelled = False
        self._timed_out = False
        self._quick_snapshot: dict[str, object] = {}
        self._quick_roots: list[dict[str, object]] = []

    @Slot()
    def run(self) -> None:
        timeout_timer = None
        # Unit tests intentionally call ``run()`` synchronously. Starting a
        # Python timer from the Qt main thread creates an unmanaged native
        # thread during pytest/Qt teardown and can surface as a process-level
        # KeyboardInterrupt. Production workers are moved to their QThread;
        # only that path needs the independent timeout supervisor.
        supervise_timeout = QThread.currentThread() != self.thread()
        try:
            published_snapshot: dict[str, object] | None = None
            failure_report: dict[str, object] | None = None
            generation_store: IndexGenerationStore | None = None
            candidate_live: dict[str, object] | None = None
            if self.generation_base_path is not None:
                generation_store = IndexGenerationStore(self.generation_base_path)
                candidate = generation_store.building_candidate()
                if candidate is not None:
                    candidate_live = collect_live_candidate_failure_snapshot(
                        candidate.path,
                        page_size=self.page_size,
                        page_offsets=self.page_offsets,
                        generation=candidate.generation,
                        scan_id=candidate.scan_id,
                        snapshot_sequence=self.snapshot_sequence,
                    )
                report_path = generation_store.failure_report_path
                if report_path.is_file():
                    try:
                        parsed_report = json.loads(
                            report_path.read_text(encoding="utf-8")
                        )
                        if isinstance(parsed_report, dict):
                            failure_report = parsed_report
                    except (OSError, ValueError, TypeError):
                        logger.warning(
                            "Unable to read candidate failure report: %s",
                            report_path,
                            exc_info=True,
                        )
            snapshot_lease = None
            try:
                snapshot_path = self.published_snapshot_path
                if self.generation_base_path is not None:
                    store = IndexGenerationStore(self.generation_base_path)
                    active = store.active()
                    if active is not None:
                        snapshot_path = active.path
                        snapshot_lease = store.lease_registry.acquire(
                            active.generation
                        )
                if (
                    snapshot_path is not None
                    and snapshot_path.resolve() != self.db.db_path.resolve()
                    and snapshot_path.is_file()
                    and is_compact_snapshot(snapshot_path)
                ):
                    published_snapshot = collect_compact_database_snapshot(
                        snapshot_path,
                        page_offsets=self.page_offsets,
                    )
            finally:
                if snapshot_lease is not None:
                    snapshot_lease.release()
            if is_compact_snapshot(self.db.db_path):
                snapshot = collect_compact_database_snapshot(
                    self.db.db_path,
                    page_offsets=self.page_offsets,
                )
                snapshot = merge_failure_report(snapshot, failure_report)
                snapshot = merge_live_candidate_failures(snapshot, candidate_live)
                quick = dict(snapshot["readiness"])
                self._quick_roots = list(snapshot["roots"])
                self._quick_snapshot = quick
                self.quick_ready.emit(
                    {
                        "roots": list(snapshot["roots"]),
                        "readiness": quick,
                        "failed_counts": dict(snapshot.get("failed_counts") or {}),
                        "failed_rows": list(snapshot.get("failed_rows") or []),
                        "all_blocking_ids": list(
                            snapshot.get("all_blocking_ids") or []
                        ),
                        "candidate_live_failures": bool(
                            snapshot.get("candidate_live_failures")
                        ),
                        "candidate_scan_id": snapshot.get("candidate_scan_id"),
                        "candidate_snapshot_sequence": snapshot.get(
                            "candidate_snapshot_sequence"
                        ),
                        "elapsed_ms": 0,
                    }
                )
                if not self._cancelled:
                    self.finished.emit(snapshot)
                else:
                    self.cancelled.emit()
                return
            quick_started = time.perf_counter()
            roots = [dict(row) for row in self.db.list_roots()]
            self._quick_roots = list(roots)
            quick = self.db.quick_index_readiness()
            quick_payload = {
                "roots": roots,
                "stats": quick,
                "readiness": quick,
            }
            if published_snapshot is not None:
                quick_payload = merge_control_snapshot_with_published(
                    quick_payload,
                    published_snapshot,
                )
            quick_payload = merge_failure_report(quick_payload, failure_report)
            quick_payload = merge_live_candidate_failures(quick_payload, candidate_live)
            self._quick_roots = list(quick_payload.get("roots") or [])
            self._quick_snapshot = dict(quick_payload.get("readiness") or quick)
            self.quick_ready.emit(
                {
                    "roots": list(quick_payload.get("roots") or []),
                    "readiness": dict(quick_payload.get("readiness") or quick),
                    "failed_counts": dict(
                        quick_payload.get("failed_counts") or {}
                    ),
                    "failed_rows": list(quick_payload.get("failed_rows") or []),
                    "all_blocking_ids": list(
                        quick_payload.get("all_blocking_ids") or []
                    ),
                    "candidate_live_failures": bool(
                        quick_payload.get("candidate_live_failures")
                    ),
                    "candidate_scan_id": quick_payload.get("candidate_scan_id"),
                    "candidate_snapshot_sequence": quick_payload.get(
                        "candidate_snapshot_sequence"
                    ),
                    "elapsed_ms": int(
                        (time.perf_counter() - quick_started) * 1000
                    ),
                }
            )
            if self._cancelled:
                self.cancelled.emit()
                return
            if supervise_timeout:
                timeout_timer = threading.Timer(
                    self.snapshot_timeout_seconds,
                    self._interrupt_for_timeout,
                )
                timeout_timer.daemon = True
                timeout_timer.start()
            if (
                self.fallback_snapshot
                and (
                    int(quick.get("active_runs") or 0) > 0
                    or bool(quick.get("content_fts_dirty"))
                )
            ):
                cached = dict(self.fallback_snapshot)
                cached["readiness"] = dict(
                    quick_payload.get("readiness") or quick
                )
                cached["stats"] = dict(cached["readiness"])
                cached["readiness_degraded"] = True
                cached["timings_ms"] = {"quick_readiness": int((time.perf_counter() - quick_started) * 1000)}
                cached["total_ms"] = cached["timings_ms"]["quick_readiness"]
                cached = merge_live_candidate_failures(cached, candidate_live)
                self.finished.emit(cached)
                return
            snapshot = collect_database_snapshot(
                self.db,
                page_size=self.page_size,
                page_offsets=self.page_offsets,
                fallback_snapshot=self.fallback_snapshot,
            )
            if published_snapshot is not None:
                snapshot = merge_control_snapshot_with_published(
                    snapshot,
                    published_snapshot,
                )
            snapshot = merge_failure_report(snapshot, failure_report)
            snapshot = merge_live_candidate_failures(snapshot, candidate_live)
            if self._cancelled:
                self.cancelled.emit()
                return
            logger.info(
                "Background database snapshot completed: total_ms=%s timings=%s",
                snapshot.get("total_ms"),
                snapshot.get("timings_ms"),
            )
            self.finished.emit(snapshot)
        except sqlite3.OperationalError as exc:
            if self._timed_out:
                cached = dict(self.fallback_snapshot)
                # A first-run database has no previous full snapshot. Return
                # the quick readiness probe as a minimal degraded snapshot so
                # the UI can continue normally while the next idle refresh
                # retries the expensive aggregates.
                if not cached:
                    cached = {
                        "roots": list(self._quick_roots),
                        "stats": dict(self._quick_snapshot),
                        "readiness": dict(self._quick_snapshot),
                        "root_stats_by_id": {},
                        "failed_counts": {},
                        "failed_rows": [],
                        "all_blocking_ids": [],
                        "excluded_rows": [],
                        "metadata_rows": [],
                        "history": [],
                    }
                else:
                    cached.setdefault("readiness", dict(self._quick_snapshot))
                    cached.setdefault("stats", dict(cached["readiness"]))
                cached["readiness_degraded"] = True
                cached["snapshot_timeout"] = True
                cached["snapshot_timeout_reason"] = str(exc) or "interrupted"
                self.finished.emit(cached)
                return
            if self._cancelled and "interrupt" in str(exc).lower():
                self.cancelled.emit()
                return
            logger.exception("Background database snapshot failed")
            log_exception(
                "database.snapshot.failed",
                exc,
                component="database_snapshot_worker",
                database_path=str(self.db.db_path),
                timed_out=self._timed_out,
                cancelled=self._cancelled,
            )
            self.failed.emit(str(exc) or exc.__class__.__name__)
        except Exception as exc:
            logger.exception("Background database snapshot failed")
            log_exception(
                "database.snapshot.failed",
                exc,
                component="database_snapshot_worker",
                database_path=str(self.db.db_path),
                timed_out=self._timed_out,
                cancelled=self._cancelled,
            )
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            if timeout_timer is not None:
                timeout_timer.cancel()

    def cancel(self) -> None:
        self._cancelled = True
        self.db.interrupt_active_connections()

    def _interrupt_for_timeout(self) -> None:
        if self._cancelled:
            return
        self._timed_out = True
        self.db.interrupt_active_connections()


class IndexReadinessWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db = DatabaseManager(Path(db_path))
        self.cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            started = time.perf_counter()
            try:
                readiness = self.db.index_readiness(reconcile=False)
            except TypeError as exc:
                if "reconcile" not in str(exc):
                    raise
                readiness = self.db.index_readiness()
            if self.cancelled:
                return
            self.finished.emit(
                {
                    "readiness": readiness,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        except sqlite3.OperationalError as exc:
            if self.cancelled and "interrupt" in str(exc).lower():
                return
            logger.debug("Background readiness refresh failed: %s", exc)
            self.failed.emit(str(exc) or exc.__class__.__name__)
        except Exception as exc:
            logger.exception("Background readiness refresh failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def cancel(self) -> None:
        self.cancelled = True
        self.db.interrupt_active_connections()
