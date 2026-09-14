"""Bounded, observable process-local database write dispatching."""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
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


class DatabaseWriteDispatcher:
    """Serialize application writes and preserve deferred work explicitly."""

    def __init__(
        self,
        coordinator: SearchIndexCoordinator | None = None,
        *,
        retry_seconds: float = 2.0,
    ) -> None:
        self.coordinator = coordinator or get_search_index_coordinator()
        self.retry_seconds = max(0.0, float(retry_seconds))
        self._lock = threading.RLock()
        self._pending: list[WriteOperation] = []
        self.coordinator.register_dispatcher(self)

    def submit(
        self,
        operation: Callable[[], Any],
        *,
        priority: int = 50,
        label: str = "",
        defer_if_search: bool = True,
    ) -> WriteOperation:
        item = WriteOperation(
            operation_id=f"db-write:{uuid.uuid4().hex}",
            operation=operation,
            priority=int(priority),
            label=str(label),
        )
        with self._lock:
            state = self.coordinator.snapshot().state
            if defer_if_search and state in {SEARCH_QUIESCING, SEARCH_ACTIVE}:
                item.status = DEFERRED
                self._pending.append(item)
                self._sort_pending()
                return item
        return self._run(item, allow_defer=defer_if_search)

    def drain(self) -> int:
        """Run deferred writes after the search lease has been released."""

        completed = 0
        while True:
            with self._lock:
                if not self._pending:
                    break
                item = self._pending.pop(0)
            result = self._run(item, allow_defer=True)
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

    def _run(
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
