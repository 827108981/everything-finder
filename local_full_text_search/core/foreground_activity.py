"""Process-local coordination between foreground searches and index writes."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any


_condition = threading.Condition()
_active_searches = 0
_active_leases: dict[str, "SearchLease"] = {}
_active_index_writes = 0


@dataclass(slots=True)
class SearchLease:
    """A process-local foreground reservation held for one search query."""

    lease_id: str
    query_id: str
    acquired_at: float
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        _release_lease(self.lease_id)

    @property
    def released(self) -> bool:
        return self._released


def acquire_search_lease(
    query_id: str | None = None,
    *,
    cancel_event: threading.Event | None = None,
    max_wait_seconds: float = 2.0,
) -> SearchLease:
    """Acquire the single foreground-search slot without blocking the UI.

    Index workers observe the same condition through ``search_active`` and
    ``wait_for_search_idle``.  A second query waits briefly for the previous
    query to cancel, then the caller can retry or report cancellation.
    """

    global _active_searches
    query = str(query_id or uuid.uuid4().hex)
    deadline = time.monotonic() + max(0.0, float(max_wait_seconds))
    with _condition:
        while _active_searches or _active_index_writes:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("搜索租约已取消")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Do not serialize independent searches indefinitely. The UI
                # normally cancels the previous SearchWorker before retrying.
                raise TimeoutError("等待上一搜索完成超时")
            _condition.wait(timeout=min(0.1, remaining))
        lease = SearchLease(uuid.uuid4().hex, query, time.monotonic())
        _active_leases[lease.lease_id] = lease
        _active_searches += 1
        _condition.notify_all()
        return lease


def _release_lease(lease_id: str) -> None:
    global _active_searches
    with _condition:
        _active_leases.pop(str(lease_id), None)
        _active_searches = max(0, _active_searches - 1)
        if _active_searches == 0:
            _condition.notify_all()


def search_status() -> dict[str, Any]:
    with _condition:
        return {
            "active": bool(_active_leases),
            "active_count": len(_active_leases),
            "lease_ids": tuple(_active_leases),
        }


def begin_search() -> None:
    """Mark a foreground search as active so writers can yield to it."""

    global _active_searches
    with _condition:
        _active_searches += 1


def end_search() -> None:
    """Release a foreground search reservation."""

    global _active_searches
    with _condition:
        _active_searches = max(0, _active_searches - 1)
        if _active_searches == 0:
            _condition.notify_all()


def search_active() -> bool:
    with _condition:
        return _active_searches > 0


def begin_index_write(
    *,
    cancel_event: threading.Event | None = None,
    max_wait_seconds: float | None = None,
) -> bool:
    """Reserve the writer side of the foreground-search gate.

    The reservation is acquired atomically with the search lease condition,
    so a writer cannot start a SQLite commit after a foreground snapshot has
    been granted.
    """

    deadline = (
        None
        if max_wait_seconds is None
        else time.monotonic() + max(0.0, float(max_wait_seconds))
    )
    global _active_index_writes
    with _condition:
        while _active_searches:
            if cancel_event is not None and cancel_event.is_set():
                return False
            if deadline is None:
                _condition.wait(timeout=0.1)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _condition.wait(timeout=min(0.1, remaining))
        _active_index_writes += 1
        return True


def end_index_write() -> None:
    global _active_index_writes
    with _condition:
        _active_index_writes = max(0, _active_index_writes - 1)
        _condition.notify_all()


def index_write_active() -> bool:
    with _condition:
        return _active_index_writes > 0


def wait_for_search_idle(
    max_wait_seconds: float | None = 1.0,
    *,
    cancel_event: threading.Event | None = None,
) -> bool:
    """Wait until foreground searches finish.

    ``None`` gives the index scheduler explicit foreground priority and waits
    at a safe boundary for the whole search duration.
    """

    deadline = (
        None
        if max_wait_seconds is None
        else time.monotonic() + max(0.0, float(max_wait_seconds))
    )
    with _condition:
        while _active_searches:
            if cancel_event is not None and cancel_event.is_set():
                return False
            if deadline is None:
                _condition.wait(timeout=0.25)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _condition.wait(timeout=remaining)
        return True
