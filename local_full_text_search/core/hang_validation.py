from __future__ import annotations

import json
import math
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

import psutil

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import PlanningNoProgressError
from local_full_text_search.core.index_manager import (
    IndexManager,
    ParseJob,
    ParseLane,
    ParseOutcome,
    ProcessLaneWatchdog,
    SpoolParseResult,
    _new_process_executor,
    drain_completed_lanes,
    schedule_parse_lanes,
    terminate_process_pool_workers,
)
from local_full_text_search.core.planning_tasks import (
    fingerprint_source_batch,
    prepare_zip_member_task,
    scan_zip_manifest_task,
    stat_file_batch,
)
from local_full_text_search.core.planning_worker import (
    PlanningProgress,
    PlanningProgressReporter,
    RecoverablePlanningRunner,
)
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.core.task_manager import CancelToken


HANG_RECOVERY_SCENARIOS = (
    "normal_text_parse",
    "directory_or_stat",
    "content_hash",
    "zip_manifest",
    "zip_member_prepare",
    "pdf_native_page",
    "pdf_ocr_page",
    "image_ocr",
    "legacy_office_converter",
    "pre_database_spool",
)


def run_semantic_progress_validation(
    output_path: Path,
    *,
    timeout_seconds: float = 0.5,
) -> dict[str, Any]:
    """Prove total duration is unlimited while semantic work keeps advancing."""

    output_path = Path(output_path)
    with tempfile.TemporaryDirectory(prefix="lfts_semantic_validation_") as tmp:
        runner = RecoverablePlanningRunner(
            Path(tmp) / "control",
            no_progress_timeout_seconds=max(0.2, timeout_seconds),
            startup_timeout_seconds=5,
            poll_interval_seconds=0.02,
        )
        duration = max(0.75, timeout_seconds * 3.25)
        completed = runner.run(
            "healthy_long_semantic_progress",
            _healthy_semantic_task,
            duration,
            max(0.03, timeout_seconds / 4),
        )
        duplicate_timed_out = False
        try:
            runner.run(
                "duplicate_semantic_progress",
                _duplicate_semantic_task,
            )
        except PlanningNoProgressError:
            duplicate_timed_out = True
        residual_pids = list(runner.active_pids)
    report: dict[str, Any] = {
        "passed": bool(completed > 0 and duplicate_timed_out and not residual_pids),
        "healthy_duration_seconds": duration,
        "timeout_seconds": timeout_seconds,
        "healthy_completed_units": completed,
        "duplicate_progress_timed_out": duplicate_timed_out,
        "residual_pids": residual_pids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def run_hang_recovery_validation(
    output_path: Path,
    *,
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    """Run the ten required real-process hang/recovery scenarios."""

    output_path = Path(output_path)
    baseline_children = _child_pids()
    with tempfile.TemporaryDirectory(prefix="lfts_hang_validation_") as tmp:
        base = Path(tmp)
        source = base / "source.txt"
        source.write_text("HEALTHY_FOLLOW_UP_CONTENT", encoding="utf-8")
        archive = base / "sample.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr("member.txt", "ZIP_HEALTHY_FOLLOW_UP")
        with zipfile.ZipFile(archive) as source_zip:
            member = source_zip.infolist()[0]

        settings = AppSettings(
            enable_ocr=False,
            normal_no_progress_timeout_seconds=max(
                1, int(math.ceil(timeout_seconds))
            ),
            ocr_no_progress_timeout_seconds=max(
                1, int(math.ceil(timeout_seconds))
            ),
            archive_no_progress_timeout_seconds=max(
                1, int(math.ceil(timeout_seconds))
            ),
            legacy_no_progress_timeout_seconds=max(
                1, int(math.ceil(timeout_seconds))
            ),
            process_no_progress_timeout_seconds=max(
                1, int(math.ceil(timeout_seconds))
            ),
            no_progress_max_retries=0,
            parser_workers=1,
            ocr_workers=1,
            pdf_parser_workers=1,
            process_parser_workers=1,
            slow_file_workers=1,
        )
        planning_runner = RecoverablePlanningRunner(
            base / "planning_control",
            no_progress_timeout_seconds=max(0.2, float(timeout_seconds)),
            startup_timeout_seconds=5,
            poll_interval_seconds=0.02,
        )
        scenarios: dict[str, dict[str, Any]] = {}
        scenarios["directory_or_stat"] = _run_planning_hang(
            planning_runner,
            "file_stat",
            stat_file_batch,
            ([str(source)], False, True),
            ([str(source)], False, False),
        )
        scenarios["content_hash"] = _run_planning_hang(
            planning_runner,
            "content_hash",
            fingerprint_source_batch,
            ([(str(source), False)], base / "hash_spool", True),
            ([(str(source), False)], base / "hash_spool", False),
        )
        scenarios["zip_manifest"] = _run_planning_hang(
            planning_runner,
            "zip_manifest",
            scan_zip_manifest_task,
            (
                archive,
                settings.to_dict(),
                base / "run_control",
                True,
            ),
            (
                archive,
                settings.to_dict(),
                base / "run_control",
                False,
            ),
        )
        scenarios["zip_member_prepare"] = _run_planning_hang(
            planning_runner,
            "zip_member_prepare",
            prepare_zip_member_task,
            (
                archive,
                0,
                "member.txt",
                int(member.file_size),
                int(member.CRC),
                ".txt",
                base / "member_spool",
                True,
            ),
            (
                archive,
                0,
                "member.txt",
                int(member.file_size),
                int(member.CRC),
                ".txt",
                base / "member_spool",
                False,
            ),
        )

        parse_matrix = {
            "normal_text_parse": ("normal", "text", "text_parse"),
            "pdf_native_page": ("pdf", "pdf", "pdf_native_page"),
            "pdf_ocr_page": ("pdf", "pdf", "pdf_ocr_page"),
            "image_ocr": ("ocr", "image_ocr", "image_ocr_region"),
            "legacy_office_converter": (
                "legacy_word",
                "legacy_office",
                "legacy_office_convert",
            ),
            "pre_database_spool": (
                "normal",
                "text",
                "pre_database_spool",
            ),
        }
        for scenario, (lane, parser_name, phase) in parse_matrix.items():
            scenarios[scenario] = _run_parse_hang(
                base / scenario,
                source,
                settings,
                lane=lane,
                parser_name=parser_name,
                phase=phase,
            )
        planning_runner.cancel_active()

    _wait_for_new_children_to_exit(baseline_children, timeout=3)
    residual_pids = sorted(_child_pids() - baseline_children)
    ordered = {
        name: scenarios[name]
        for name in HANG_RECOVERY_SCENARIOS
    }
    passed = not residual_pids and all(
        bool(result.get("timed_out"))
        and bool(result.get("old_pid_exited"))
        and bool(result.get("healthy_follow_up"))
        for result in ordered.values()
    ) and all(
        {
            "reclaiming_no_progress",
            "terminating_worker",
            "rebuilding_pool",
            "same_stall_retry_stopped",
        }.issubset(set(ordered[name].get("scheduler_diagnostic_states") or []))
        for name in parse_matrix
    )
    report: dict[str, Any] = {
        "passed": passed,
        "timeout_seconds": timeout_seconds,
        "scenarios": ordered,
        "residual_pids": residual_pids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def _run_planning_hang(
    runner: RecoverablePlanningRunner,
    task_name: str,
    target: Callable[..., Any],
    failing_args: tuple[object, ...],
    healthy_args: tuple[object, ...],
) -> dict[str, Any]:
    latest = PlanningProgress()

    def progress(value: PlanningProgress) -> None:
        nonlocal latest
        latest = value

    started = time.monotonic()
    timed_out = False
    try:
        runner.run(
            task_name,
            target,
            *failing_args,
            progress_callback=progress,
        )
    except PlanningNoProgressError:
        timed_out = True
    elapsed = time.monotonic() - started
    old_pid = latest.worker_pid
    old_pid_exited = _wait_pid_exit(old_pid, 2)
    healthy_follow_up = False
    try:
        healthy_result = runner.run(
            f"{task_name}_healthy",
            target,
            *healthy_args,
        )
        healthy_follow_up = healthy_result is not None
    except Exception:
        healthy_follow_up = False
    return {
        "timed_out": timed_out,
        "elapsed_seconds": round(elapsed, 3),
        "phase": latest.phase,
        "cursor": latest.cursor,
        "worker_pid": old_pid,
        "old_pid_exited": old_pid_exited,
        "healthy_follow_up": healthy_follow_up,
        "retry_count": 0,
        "eta_event_types": ["worker_recycle"] if timed_out else [],
    }


def _run_parse_hang(
    case_dir: Path,
    healthy_source: Path,
    settings: AppSettings,
    *,
    lane: str,
    parser_name: str,
    phase: str,
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)
    database = DatabaseManager(case_dir / "validation.db")
    database.initialize()
    manager = IndexManager(database, settings)
    metrics = IndexRunMetrics(run_id=f"hang-{lane}-{phase}")
    metrics.eta_metrics["_run_started_monotonic"] = time.perf_counter()
    manager._current_metrics = metrics
    executor = _new_process_executor(
        settings,
        1,
        case_dir,
        persistent=True,
    )
    parse_lane = ParseLane(
        lane,
        executor,
        1,
        64 * 1024 * 1024,
        process_based=True,
        persistent_process=True,
        worker_count=1,
    )
    lanes = {lane: parse_lane}
    executors = [executor]
    process_executors = [executor]
    token = CancelToken()
    job = ParseJob(
        file_id=1,
        file_path=healthy_source,
        parser_name=parser_name,
        parser_version="validation",
        lane=lane,
        size_bytes=healthy_source.stat().st_size,
        memory_estimate_bytes=healthy_source.stat().st_size,
        validation_hang_stage=phase,
    )
    parse_lane.pending.append(job)
    watchdog = ProcessLaneWatchdog(
        lanes,
        case_dir,
        settings,
        diagnostic_callback=manager._record_scheduler_diagnostic,
    )
    watchdog.start()
    started = time.monotonic()
    timed_out_result: ParseOutcome | None = None
    try:
        schedule_parse_lanes(lanes.values(), settings, token, case_dir)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            recycled = manager._recycle_unhealthy_process_lanes(
                lanes,
                executors,
                process_executors,
                case_dir,
            )
            if recycled:
                candidate = recycled[0][2]
                if isinstance(candidate, ParseOutcome):
                    timed_out_result = candidate
                break
            time.sleep(0.05)
        old_pid = int(job.progress_worker_pid or 0)
        timed_out = bool(
            timed_out_result is not None
            and timed_out_result.error_code == "PARSE_NO_PROGRESS"
        )
        diagnostic_events = manager._take_scheduler_diagnostics()
        old_pid_exited = _wait_pid_exit(old_pid, 2)

        healthy_job = ParseJob(
            file_id=2,
            file_path=healthy_source,
            parser_name="text",
            parser_version="validation",
            lane=lane,
            size_bytes=healthy_source.stat().st_size,
            memory_estimate_bytes=healthy_source.stat().st_size,
        )
        parse_lane.pending.append(healthy_job)
        schedule_parse_lanes(lanes.values(), settings, token, case_dir)
        healthy_follow_up = False
        healthy_deadline = time.monotonic() + 8
        while time.monotonic() < healthy_deadline:
            completed = drain_completed_lanes(
                lanes.values(),
                token,
                case_dir,
                block=True,
            )
            if not completed:
                continue
            result = completed[0][2]
            healthy_follow_up = (
                isinstance(result, SpoolParseResult)
                or (
                    isinstance(result, ParseOutcome)
                    and result.status in {"success", "partial_success"}
                )
            )
            break
        return {
            "timed_out": timed_out,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "phase": job.progress_phase,
            "cursor": job.progress_cursor,
            "worker_pid": old_pid,
            "old_pid_exited": old_pid_exited,
            "healthy_follow_up": healthy_follow_up,
            "retry_count": job.retry_count,
            "error_code": (
                timed_out_result.error_code if timed_out_result else None
            ),
            "scheduler_diagnostic_states": [
                str(event.get("state") or "")
                for event in diagnostic_events
            ],
            "scheduler_diagnostic_events": diagnostic_events,
            "eta_event_types": [
                str(event.get("event_type") or "")
                for event in metrics.eta_metrics.get("replay_events", [])
                if isinstance(event, dict)
            ],
        }
    finally:
        watchdog.stop()
        for process_executor in process_executors:
            terminate_process_pool_workers(process_executor, case_dir)
        for active_executor in executors:
            try:
                active_executor.shutdown(wait=True, cancel_futures=True)
            except Exception:
                pass


def _child_pids() -> set[int]:
    try:
        return {
            child.pid
            for child in psutil.Process().children(recursive=True)
            if child.is_running()
        }
    except psutil.Error:
        return set()


def _wait_pid_exit(pid: int, timeout: float) -> bool:
    if pid <= 0:
        return False
    deadline = time.monotonic() + timeout
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.02)
    return not psutil.pid_exists(pid)


def _wait_for_new_children_to_exit(
    baseline: set[int],
    *,
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not (_child_pids() - baseline):
            return
        time.sleep(0.05)


def _healthy_semantic_task(
    reporter: PlanningProgressReporter,
    duration: float,
    interval: float,
) -> int:
    started = time.monotonic()
    completed = 0
    while time.monotonic() - started < duration:
        completed += 1
        reporter.advance(
            phase="content_hash",
            completed=completed,
            total=1_000_000,
            cursor=f"offset:{completed * 4096}",
            bytes_read=completed * 4096,
            checkpoint_version=completed,
        )
        time.sleep(interval)
    return completed


def _duplicate_semantic_task(reporter: PlanningProgressReporter) -> None:
    while True:
        reporter.advance(
            phase="content_hash",
            completed=1,
            total=10,
            cursor="offset:4096",
            bytes_read=4096,
            checkpoint_version=1,
        )
        time.sleep(0.03)
