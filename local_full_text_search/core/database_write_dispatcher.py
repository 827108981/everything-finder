"""Bounded, observable process-local database write dispatching."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_full_text_search.core.errors import OperationCancelled
from local_full_text_search.core.search_index_coordinator import (
    SEARCH_ACTIVE,
    SEARCH_QUIESCING,
    SearchIndexCoordinator,
    get_search_index_coordinator,
)


QUEUED = "QUEUED"
RUNNING = "RUNNING"
COMMITTED = "COMMITTED"
DEFERRED = "DEFERRED"
CANCELLED = "CANCELLED"
FAILED_RETRYABLE = "FAILED_RETRYABLE"
FAILED_FATAL = "FAILED_FATAL"


@dataclass(slots=True)
class WriteOperation:
    operation_id: str
    operation: Callable[[], Any]
    priority: int = 50
    label: str = ""
    status: str = QUEUED
    attempts: int = 0
    error: str = ""
    result: Any = None


@dataclass(slots=True)
class _DatabaseWriterState:
    execution_lock: threading.Lock
    active_lock: threading.Lock
    active_operation_id: str | None = None
    active_started_at: float = 0.0


_WRITER_STATES_LOCK = threading.Lock()
_WRITER_STATES: dict[str, _DatabaseWriterState] = {}


def _writer_state_for(database_path: Path | str | None) -> _DatabaseWriterState:
    if database_path is None:
        key = f"anonymous:{uuid.uuid4().hex}"
    else:
        key = str(Path(database_path).resolve()).casefold()
    with _WRITER_STATES_LOCK:
        return _WRITER_STATES.setdefault(
            key,
            _DatabaseWriterState(
                execution_lock=threading.Lock(),
                active_lock=threading.Lock(),
            ),
        )


class DatabaseWriteDispatcher:
    """Serialize application writes and preserve deferred work explicitly.

    教学提示：上游可以并行解析多个文件，但 SQLite 最终写入不能毫无节制地
    并发执行。这个调度器用队列和 execution_lock 把写事务串行化，从而避免
    大量 ``database is locked`` 冲突。
    """

    def __init__(
        self,
        coordinator: SearchIndexCoordinator | None = None,
        *,
        database_path: Path | str | None = None,
        retry_seconds: float = 2.0,
    ) -> None:
        self.coordinator = coordinator or get_search_index_coordinator()
        self.retry_seconds = max(0.0, float(retry_seconds))
        self._lock = threading.RLock()
        # ``submit`` is synchronous for its caller, but it must still be a
        # real per-dispatcher queue boundary.  Previously two callers could
        # enter ``_run`` concurrently and both open SQLite write transactions,
        # defeating the purpose of this class and producing process-local
        # ``database is locked`` storms.
        self.database_path = Path(database_path).resolve() if database_path else None
        self._writer_state = _writer_state_for(database_path)
        self._pending: list[WriteOperation] = []
        self.coordinator.register_dispatcher(self)

    def submit(
        self,
        operation: Callable[[], Any],
        *,
        priority: int = 50,
        label: str = "",
        defer_if_search: bool = True,
        max_wait_seconds: float | None = None,
    ) -> WriteOperation:
        # 每次写库先包装为可追踪的 WriteOperation，记录优先级、名称和状态。
        item = WriteOperation(
            operation_id=f"db-write:{uuid.uuid4().hex}",
            operation=operation,
            priority=int(priority),
            label=str(label),
        )
        with self._lock:
            state = self.coordinator.snapshot().state
            if defer_if_search and state in {SEARCH_QUIESCING, SEARCH_ACTIVE}:
                # 旧的共享数据库模式下，前台搜索活跃时把写任务放回待处理队列，
                # 等搜索释放租约后由 drain() 继续执行。
                item.status = DEFERRED
                self._pending.append(item)
                self._sort_pending()
                return item
        return self._run(
            item,
            allow_defer=defer_if_search,
            max_wait_seconds=max_wait_seconds,
        )

    def drain(self) -> int:
        """Run deferred writes after the search lease has been released."""

        completed = 0
        while True:
            with self._lock:
                if not self._pending:
                    break
                item = self._pending.pop(0)
            result = self._run(item, allow_defer=True, max_wait_seconds=None)
            if result.status == COMMITTED:
                completed += 1
            elif result.status in {DEFERRED, FAILED_RETRYABLE}:
                with self._lock:
                    self._pending.append(result)
                    self._sort_pending()
                break
        return completed

    def cancel_pending(self) -> int:
        with self._lock:
            count = 0
            for item in self._pending:
                item.status = CANCELLED
                count += 1
            self._pending.clear()
            return count

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def pending_operations(self) -> tuple[WriteOperation, ...]:
        with self._lock:
            return tuple(self._pending)

    def active_operation(self) -> dict[str, object] | None:
        """Return a cheap in-memory snapshot for watchdog diagnostics."""

        with self._writer_state.active_lock:
            operation_id = self._writer_state.active_operation_id
            started_at = self._writer_state.active_started_at
            if not operation_id:
                return None
            return {
                "operation_id": operation_id,
                "database_path": str(self.database_path) if self.database_path else "",
                "elapsed_ms": int(max(0.0, time.monotonic() - started_at) * 1000),
            }

    def _run(
        self,
        item: WriteOperation,
        *,
        allow_defer: bool,
        max_wait_seconds: float | None,
    ) -> WriteOperation:
        # execution_lock 以数据库路径为范围，同一数据库同一时刻只允许一个
        # dispatcher 进入真实写事务。解析可以并行，提交数据库必须有序。
        if max_wait_seconds is None:
            acquired = self._writer_state.execution_lock.acquire()
        else:
            acquired = self._writer_state.execution_lock.acquire(
                timeout=max(0.0, float(max_wait_seconds))
            )
        if not acquired:
            item.status = DEFERRED
            item.error = "dispatcher_busy_timeout"
            return item
        with self._writer_state.active_lock:
            self._writer_state.active_operation_id = item.operation_id
            self._writer_state.active_started_at = time.monotonic()
        try:
            return self._run_serialized(item, allow_defer=allow_defer)
        finally:
            with self._writer_state.active_lock:
                self._writer_state.active_operation_id = None
                self._writer_state.active_started_at = 0.0
            self._writer_state.execution_lock.release()

    def _run_serialized(
        self,
        item: WriteOperation,
        *,
        allow_defer: bool,
    ) -> WriteOperation:
        state = self.coordinator.snapshot().state
        if allow_defer and state in {SEARCH_QUIESCING, SEARCH_ACTIVE}:
            item.status = DEFERRED
            return item
        deadline = time.monotonic() + self.retry_seconds
        delay = 0.05
        while True:
            item.status = RUNNING
            item.attempts += 1
            try:
                self.coordinator.begin_index_write()
                try:
                    item.result = item.operation()
                finally:
                    self.coordinator.end_index_write()
                item.status = COMMITTED
                item.error = ""
                return item
            except OperationCancelled as exc:
                item.status = CANCELLED
                item.error = str(exc) or exc.__class__.__name__
                return item
            except sqlite3.OperationalError as exc:
                item.error = str(exc) or exc.__class__.__name__
                if not _is_busy(exc) or time.monotonic() >= deadline:
                    item.status = FAILED_RETRYABLE if _is_busy(exc) else FAILED_FATAL
                    return item
                item.status = FAILED_RETRYABLE
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(0.5, delay * 1.7)
            except Exception as exc:
                item.error = str(exc) or exc.__class__.__name__
                item.status = FAILED_FATAL
                return item

    def _sort_pending(self) -> None:
        self._pending.sort(key=lambda item: (item.priority, item.operation_id))


def _is_busy(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message
