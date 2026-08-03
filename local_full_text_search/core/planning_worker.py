from __future__ import annotations

import json
import logging
import multiprocessing
import os
import shutil
import threading
import time
import traceback
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from local_full_text_search.core.errors import (
    PlanningNoProgressError,
    PlanningWorkerError,
)
from local_full_text_search.core.semantic_progress import (
    SemanticProgress,
    is_semantic_progress,
)
from local_full_text_search.core.task_manager import PAUSE_MARKER_NAME

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlanningProgress:
    phase: str = ""
    completed: int = 0
    total: int = 0
    cursor: str = ""
    bytes_read: int = 0
    output_blocks: int = 0
    checkpoint_version: int = 0
    detail: str = ""
    worker_pid: int = 0

    def semantic(self) -> SemanticProgress:
        return SemanticProgress(
            phase=self.phase,
            completed=max(0, int(self.completed)),
            total=max(0, int(self.total)),
            cursor=self.cursor,
            bytes_read=max(0, int(self.bytes_read)),
            output_blocks=max(0, int(self.output_blocks)),
            checkpoint_version=max(0, int(self.checkpoint_version)),
        )


class PlanningProgressReporter:
    """Atomically publish durable progress from a spawned planning process."""

    def __init__(self, progress_path: Path) -> None:
        self.progress_path = Path(progress_path)
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._sequence = 0

    def advance(
        self,
        *,
        phase: str,
        completed: int = 0,
        total: int = 0,
        cursor: str = "",
        bytes_read: int = 0,
        output_blocks: int = 0,
        checkpoint_version: int = 0,
        detail: str = "",
    ) -> None:
        self._sequence += 1
        snapshot = PlanningProgress(
            phase=str(phase or ""),
            completed=max(0, int(completed or 0)),
            total=max(0, int(total or 0)),
            cursor=str(cursor or ""),
            bytes_read=max(0, int(bytes_read or 0)),
            output_blocks=max(0, int(output_blocks or 0)),
            checkpoint_version=max(0, int(checkpoint_version or 0)),
            detail=str(detail or ""),
            worker_pid=os.getpid(),
        )
        payload = {**asdict(snapshot), "sequence": self._sequence, "updated_at": time.time()}
        # Keep internal progress paths bounded for deeply nested Windows app
        # data directories while retaining uniqueness for atomic replacement.
        temporary = self.progress_path.with_suffix(
            f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        # On Windows a concurrent reader can briefly deny rename/delete
        # sharing. Keep the atomic replace contract, but tolerate that bounded
        # sharing window instead of failing a healthy planning task.
        for attempt in range(20):
            try:
                temporary.replace(self.progress_path)
                break
            except PermissionError:
                if attempt >= 19:
                    raise
                time.sleep(0.005)


class RecoverablePlanningRunner:
    """Execute risky planning work in a killable spawned process.

    The runner deliberately creates an isolated process per planning unit. The
    units are expected to be batches (directory batch, stat batch, hash file,
    ZIP manifest), not individual cheap Python operations. A stopped unit is
    terminated together with descendants; the same runner can then execute the
    next healthy unit.
    """

    def __init__(
        self,
        control_dir: Path,
        *,
        no_progress_timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
        startup_timeout_seconds: float = 1.0,
        pause_control_dir: Path | None = None,
    ) -> None:
        self.control_dir = Path(control_dir)
        self.control_dir.mkdir(parents=True, exist_ok=True)
        self.no_progress_timeout_seconds = max(
            0.05, float(no_progress_timeout_seconds)
        )
        self.poll_interval_seconds = max(0.01, float(poll_interval_seconds))
        self.startup_timeout_seconds = max(0.25, float(startup_timeout_seconds))
        self.pause_control_dir = (
            Path(pause_control_dir)
            if pause_control_dir is not None
            else None
        )
        self._active_pids: set[int] = set()
        self._lock = threading.Lock()
        self.metrics: dict[str, int] = {
            "planning_task_count": 0,
            "planning_worker_timeout_count": 0,
            "worker_process_kill_count": 0,
        }

    @property
    def active_pids(self) -> tuple[int, ...]:
        with self._lock:
            return tuple(sorted(self._active_pids))

    def cancel_active(self) -> None:
        """Terminate only planning process trees owned by this runner."""

        for pid in self.active_pids:
            _terminate_process_tree(pid)
            self.metrics["worker_process_kill_count"] += 1

    def run(
        self,
        task_name: str,
        target: Callable[..., Any],
        *args: object,
        cancel_check: Callable[[], None] | None = None,
        progress_callback: Callable[[PlanningProgress], None] | None = None,
        no_progress_timeout_seconds: float | None = None,
    ) -> Any:
        self.metrics["planning_task_count"] += 1
        task_id = _planning_task_id(task_name)
        task_dir = self.control_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        progress_path = task_dir / "progress.json"
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_planning_process_entry,
            args=(sender, progress_path, target, args),
            name=f"lfts-plan-{_safe_name(task_name)[:24]}",
        )
        timeout_seconds = max(
            0.05,
            float(
                self.no_progress_timeout_seconds
                if no_progress_timeout_seconds is None
                else no_progress_timeout_seconds
            ),
        )
        process_started_at = time.monotonic()
        last_progress_at = process_started_at
        observed_progress = False
        previous = SemanticProgress()
        latest = PlanningProgress()
        process.start()
        sender.close()
        if process.pid is None:
            receiver.close()
            shutil.rmtree(task_dir, ignore_errors=True)
            raise PlanningWorkerError(f"准备任务无法启动：{task_name}")
        with self._lock:
            self._active_pids.add(process.pid)
        try:
            while True:
                if cancel_check is not None:
                    try:
                        cancel_check()
                    except BaseException:
                        _terminate_process_tree(process.pid)
                        process.join(timeout=1)
                        raise
                snapshot = _read_progress(progress_path)
                if snapshot is not None:
                    semantic = snapshot.semantic()
                    if is_semantic_progress(previous, semantic):
                        previous = semantic
                        latest = snapshot
                        last_progress_at = time.monotonic()
                        observed_progress = True
                        if progress_callback is not None:
                            progress_callback(snapshot)
                if receiver.poll():
                    message = receiver.recv()
                    process.join(timeout=1)
                    if not isinstance(message, tuple) or not message:
                        raise PlanningWorkerError(
                            f"准备任务返回了无效结果：{task_name}"
                        )
                    if message[0] == "ok":
                        final_snapshot = _read_progress(progress_path)
                        if (
                            final_snapshot is not None
                            and progress_callback is not None
                            and is_semantic_progress(
                                previous,
                                final_snapshot.semantic(),
                            )
                        ):
                            progress_callback(final_snapshot)
                        return message[1]
                    if message[0] == "error":
                        error_type = str(message[1] or "PlanningWorkerError")
                        error_message = str(message[2] or "")
                        remote_traceback = str(message[3] or "")
                        raise PlanningWorkerError(
                            f"{task_name} 失败：{error_type}: {error_message}\n"
                            f"{remote_traceback}"
                        )
                    raise PlanningWorkerError(
                        f"准备任务返回了未知状态：{task_name}: {message[0]}"
                    )
                if not process.is_alive():
                    process.join(timeout=1)
                    if receiver.poll():
                        continue
                    raise PlanningWorkerError(
                        f"准备进程异常退出：{task_name}，退出码 {process.exitcode}"
                    )
                now = time.monotonic()
                if self._pause_requested():
                    process_started_at = now
                    last_progress_at = now
                overdue = (
                    observed_progress
                    and now - last_progress_at >= timeout_seconds
                ) or (
                    not observed_progress
                    and now - process_started_at >= self.startup_timeout_seconds
                )
                if overdue:
                    _terminate_process_tree(process.pid)
                    self.metrics["planning_worker_timeout_count"] += 1
                    self.metrics["worker_process_kill_count"] += 1
                    process.join(timeout=1)
                    effective_timeout = (
                        timeout_seconds
                        if observed_progress
                        else self.startup_timeout_seconds
                    )
                    raise PlanningNoProgressError(
                        f"{task_name} 在“{latest.phase or '启动'}”阶段停在"
                        f"“{latest.cursor or '无游标'}”且连续 "
                        f"{effective_timeout:g} 秒无有效进展"
                    )
                time.sleep(self.poll_interval_seconds)
        finally:
            if process.is_alive():
                _terminate_process_tree(process.pid)
                process.join(timeout=1)
            receiver.close()
            with self._lock:
                if process.pid is not None:
                    self._active_pids.discard(process.pid)
            shutil.rmtree(task_dir, ignore_errors=True)

    def stream(
        self,
        task_name: str,
        target: Callable[..., Any],
        *args: object,
        cancel_check: Callable[[], None] | None = None,
        progress_callback: Callable[[PlanningProgress], None] | None = None,
        no_progress_timeout_seconds: float | None = None,
    ) -> Iterator[Any]:
        """Yield bounded batches while retaining the same semantic watchdog."""

        self.metrics["planning_task_count"] += 1
        task_id = _planning_task_id(task_name)
        task_dir = self.control_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=False)
        progress_path = task_dir / "progress.json"
        context = multiprocessing.get_context("spawn")
        receiver, sender = context.Pipe(duplex=False)
        process = context.Process(
            target=_planning_stream_process_entry,
            args=(sender, progress_path, target, args),
            name=f"lfts-plan-{_safe_name(task_name)[:24]}",
        )
        timeout_seconds = max(
            0.05,
            float(
                self.no_progress_timeout_seconds
                if no_progress_timeout_seconds is None
                else no_progress_timeout_seconds
            ),
        )
        process_started_at = time.monotonic()
        last_progress_at = process_started_at
        observed_progress = False
        previous = SemanticProgress()
        latest = PlanningProgress()
        process.start()
        sender.close()
        if process.pid is None:
            receiver.close()
            shutil.rmtree(task_dir, ignore_errors=True)
            raise PlanningWorkerError(f"准备任务无法启动：{task_name}")
        with self._lock:
            self._active_pids.add(process.pid)
        try:
            while True:
                if cancel_check is not None:
                    try:
                        cancel_check()
                    except BaseException:
                        _terminate_process_tree(process.pid)
                        process.join(timeout=1)
                        raise
                snapshot = _read_progress(progress_path)
                if snapshot is not None:
                    semantic = snapshot.semantic()
                    if is_semantic_progress(previous, semantic):
                        previous = semantic
                        latest = snapshot
                        last_progress_at = time.monotonic()
                        observed_progress = True
                        if progress_callback is not None:
                            progress_callback(snapshot)
                if receiver.poll():
                    try:
                        message = receiver.recv()
                    except EOFError as exc:
                        raise PlanningWorkerError(
                            f"准备流异常关闭：{task_name}"
                        ) from exc
                    if not isinstance(message, tuple) or not message:
                        raise PlanningWorkerError(
                            f"准备流返回了无效结果：{task_name}"
                        )
                    if message[0] == "item":
                        yield message[1]
                        continue
                    if message[0] == "done":
                        final_snapshot = _read_progress(progress_path)
                        if (
                            final_snapshot is not None
                            and progress_callback is not None
                            and is_semantic_progress(
                                previous,
                                final_snapshot.semantic(),
                            )
                        ):
                            progress_callback(final_snapshot)
                        process.join(timeout=1)
                        return
                    if message[0] == "error":
                        error_type = str(message[1] or "PlanningWorkerError")
                        error_message = str(message[2] or "")
                        remote_traceback = str(message[3] or "")
                        raise PlanningWorkerError(
                            f"{task_name} 失败：{error_type}: {error_message}\n"
                            f"{remote_traceback}"
                        )
                    raise PlanningWorkerError(
                        f"准备流返回了未知状态：{task_name}: {message[0]}"
                    )
                if not process.is_alive():
                    process.join(timeout=1)
                    if receiver.poll():
                        continue
                    raise PlanningWorkerError(
                        f"准备进程异常退出：{task_name}，退出码 {process.exitcode}"
                    )
                now = time.monotonic()
                if self._pause_requested():
                    process_started_at = now
                    last_progress_at = now
                overdue = (
                    observed_progress
                    and now - last_progress_at >= timeout_seconds
                ) or (
                    not observed_progress
                    and now - process_started_at >= self.startup_timeout_seconds
                )
                if overdue:
                    _terminate_process_tree(process.pid)
                    self.metrics["planning_worker_timeout_count"] += 1
                    self.metrics["worker_process_kill_count"] += 1
                    process.join(timeout=1)
                    effective_timeout = (
                        timeout_seconds
                        if observed_progress
                        else self.startup_timeout_seconds
                    )
                    raise PlanningNoProgressError(
                        f"{task_name} 在“{latest.phase or '启动'}”阶段停在"
                        f"“{latest.cursor or '无游标'}”且连续 "
                        f"{effective_timeout:g} 秒无有效进展"
                    )
                time.sleep(self.poll_interval_seconds)
        finally:
            if process.is_alive():
                _terminate_process_tree(process.pid)
                process.join(timeout=1)
            receiver.close()
            with self._lock:
                if process.pid is not None:
                    self._active_pids.discard(process.pid)
            shutil.rmtree(task_dir, ignore_errors=True)

    def _pause_requested(self) -> bool:
        return bool(
            self.pause_control_dir is not None
            and (
                self.pause_control_dir / PAUSE_MARKER_NAME
            ).is_file()
        )


def _planning_process_entry(
    sender: Any,
    progress_path: Path,
    target: Callable[..., Any],
    args: tuple[object, ...],
) -> None:
    reporter = PlanningProgressReporter(progress_path)
    try:
        result = target(reporter, *args)
        sender.send(("ok", result))
    except BaseException as exc:
        try:
            sender.send(
                (
                    "error",
                    exc.__class__.__name__,
                    str(exc),
                    traceback.format_exc(),
                )
            )
        except BaseException:
            pass
    finally:
        sender.close()


def _planning_stream_process_entry(
    sender: Any,
    progress_path: Path,
    target: Callable[..., Any],
    args: tuple[object, ...],
) -> None:
    reporter = PlanningProgressReporter(progress_path)
    try:
        for item in target(reporter, *args):
            sender.send(("item", item))
        sender.send(("done", None))
    except BaseException as exc:
        try:
            sender.send(
                (
                    "error",
                    exc.__class__.__name__,
                    str(exc),
                    traceback.format_exc(),
                )
            )
        except BaseException:
            pass
    finally:
        sender.close()


def _read_progress(path: Path) -> PlanningProgress | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return PlanningProgress(
            phase=str(payload.get("phase") or ""),
            completed=max(0, int(payload.get("completed") or 0)),
            total=max(0, int(payload.get("total") or 0)),
            cursor=str(payload.get("cursor") or ""),
            bytes_read=max(0, int(payload.get("bytes_read") or 0)),
            output_blocks=max(0, int(payload.get("output_blocks") or 0)),
            checkpoint_version=max(
                0, int(payload.get("checkpoint_version") or 0)
            ),
            detail=str(payload.get("detail") or ""),
            worker_pid=max(0, int(payload.get("worker_pid") or 0)),
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _terminate_process_tree(pid: int) -> None:
    try:
        parent = psutil.Process(int(pid))
    except (psutil.Error, ValueError):
        return
    descendants = parent.children(recursive=True)
    processes = [*descendants, parent]
    for process in reversed(processes):
        try:
            process.terminate()
        except psutil.Error:
            continue
    _gone, alive = psutil.wait_procs(processes, timeout=0.5)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            continue
    psutil.wait_procs(alive, timeout=0.5)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in {"-", "_"} else "-"
        for character in str(value or "planning")
    )
    return cleaned.strip("-") or "planning"


def _planning_task_id(task_name: str) -> str:
    """Create a unique, bounded directory name for a planning subprocess."""

    return f"{_safe_name(task_name)[:16]}-{uuid.uuid4().hex[:16]}"
