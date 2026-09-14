"""Process-local coordination between foreground search and indexing.

The coordinator owns the *application* pause protocol.  SQLite itself remains
responsible for reporting real busy/locked errors; callers must not use a
database connection as a proxy for this state machine.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
from weakref import ReferenceType, WeakSet, ref

from local_full_text_search.core.errors import (
    IndexNotReadyError,
    OperationCancelled,
    QuiesceTimeout,
    SearchLeaseBusy,
)
from local_full_text_search.core.foreground_activity import (
    SearchLease,
    acquire_search_lease,
)


IDLE = "IDLE"
INDEX_RUNNING = "INDEX_RUNNING"
SEARCH_QUIESCING = "SEARCH_QUIESCING"
SEARCH_ACTIVE = "SEARCH_ACTIVE"
WRITE_DRAINING = "WRITE_DRAINING"
USER_PAUSED = "USER_PAUSED"
STOPPING = "STOPPING"


@dataclass(frozen=True, slots=True)
class CoordinatorSnapshot:
    state: str
    request_id: str | None
    index_run_id: str | None
    active_tasks: int
    writer_queue_depth: int
    deferred_write_count: int
    unconfirmed: tuple[str, ...]
    waited_ms: int

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state,
            "request_id": self.request_id,
            "index_run_id": self.index_run_id,
            "active_tasks": self.active_tasks,
            "writer_queue_depth": self.writer_queue_depth,
            "deferred_write_count": self.deferred_write_count,
            "unconfirmed": self.unconfirmed,
            "waited_ms": self.waited_ms,
        }


class SearchIndexCoordinator:
    """Single process-wide owner of the search/index coordination protocol."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._state = IDLE
        self._manager_ref: ReferenceType[Any] | None = None
        self._pause_controller_ref: ReferenceType[Any] | None = None
        self._manager_user_paused = False
        self._request_id: str | None = None
        self._index_run_id: str | None = None
        self._lease: SearchLease | None = None
        self._deferred_writes: list[tuple[int, str, object]] = []
        self._active_index_writes = 0
        self._sequence = 0
        self._unconfirmed: tuple[str, ...] = ()
        self._waited_ms = 0
        self._dispatchers: WeakSet[Any] = WeakSet()
        self._draining_thread_id: int | None = None

    def register_dispatcher(self, dispatcher: Any) -> None:
        with self._condition:
            self._dispatchers.add(dispatcher)

    def unregister_dispatcher(self, dispatcher: Any) -> None:
        with self._condition:
            self._dispatchers.discard(dispatcher)

    def register_index_manager(
        self,
        manager: Any,
        *,
        index_run_id: str | None = None,
        pause_controller: Any | None = None,
    ) -> None:
        """Register the active index manager before it starts scheduling."""

        with self._condition:
            self._manager_ref = ref(manager)
            self._pause_controller_ref = (
                ref(pause_controller) if pause_controller is not None else None
            )
            self._index_run_id = str(index_run_id) if index_run_id else None
            if self._state == IDLE:
                self._state = INDEX_RUNNING
            self._condition.notify_all()

    def unregister_index_manager(self, manager: Any | None = None) -> None:
        with self._condition:
            current = self._manager_ref() if self._manager_ref else None
            if manager is not None and current is not None and current is not manager:
                return
            self._manager_ref = None
            self._pause_controller_ref = None
            self._index_run_id = None
            if self._state == INDEX_RUNNING:
                self._state = IDLE
            self._condition.notify_all()

    def mark_index_running(self, index_run_id: str | None = None) -> None:
        with self._condition:
            self._index_run_id = str(index_run_id) if index_run_id else self._index_run_id
            if self._state not in {SEARCH_ACTIVE, SEARCH_QUIESCING, STOPPING}:
                self._state = INDEX_RUNNING
            self._condition.notify_all()

    def request_search(
        self,
        query_id: str | None = None,
        *,
        cancel_event: threading.Event | None = None,
        readiness_checker: Any | None = None,
        cancel_checker: Any | None = None,
        max_wait_seconds: float = 10.0,
    ) -> SearchLease:
        """Wait for a complete projection, then grant one foreground lease."""

        request_id = str(query_id or f"search:{uuid.uuid4().hex}")
        started = time.perf_counter()

        # A search requested during indexing must wait for a complete,
        # atomically published projection.  Waiting here deliberately does not
        # start the query deadline and does not pause the index prematurely.
        if readiness_checker is not None:
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("搜索已取消")
                if callable(cancel_checker) and cancel_checker():
                    raise OperationCancelled("搜索已取消")
                try:
                    readiness = dict(readiness_checker() or {})
                except IndexNotReadyError:
                    raise
                except Exception:
                    readiness = {}
                if bool(readiness.get("ready")) and not bool(
                    readiness.get("active_runs")
                ):
                    break
                if (
                    readiness
                    and not bool(readiness.get("ready"))
                    and int(readiness.get("active_runs") or 0) == 0
                    and self.snapshot().state not in {
                        INDEX_RUNNING,
                        SEARCH_QUIESCING,
                        SEARCH_ACTIVE,
                        WRITE_DRAINING,
                    }
                ):
                    raise IndexNotReadyError(
                        "完整搜索投影尚未就绪，普通搜索已排队等待索引收敛"
                    )
                time.sleep(0.1)

        # Index wait is intentionally outside the quiesce deadline. Only the
        # short transition from an idle index to a search-safe point is timed.
        quiesce_started = time.perf_counter()
        with self._condition:
            if self._state == STOPPING:
                raise OperationCancelled("应用正在关闭，搜索已取消")
            while self._lease is not None or self._state == SEARCH_QUIESCING:
                if cancel_event is not None and cancel_event.is_set():
                    raise OperationCancelled("搜索已取消")
                if callable(cancel_checker) and cancel_checker():
                    raise OperationCancelled("搜索已取消")
                remaining = max(0.0, float(max_wait_seconds)) - (
                    time.perf_counter() - quiesce_started
                )
                if remaining <= 0:
                    raise SearchLeaseBusy("正在等待上一搜索或安全暂停结束")
                self._condition.wait(timeout=min(0.1, remaining))
            self._request_id = request_id
            self._state = SEARCH_QUIESCING
            manager = self._manager_ref() if self._manager_ref else None
            pause_controller = (
                self._pause_controller_ref()
                if self._pause_controller_ref
                else None
            )

        # The manager may be on a QThread. Its pause request is deliberately
        # outside the coordinator lock so progress callbacks cannot deadlock.
        manager_paused = False
        if manager is not None:
            try:
                current_status = dict(manager.pause_status() or {})
                already_paused = str(current_status.get("state") or "") in {
                    "paused",
                    "pause_requested",
                    "pausing",
                }
                if not already_paused:
                    if pause_controller is not None:
                        pause_controller.request_coordinator_pause()
                    else:
                        manager.request_pause()
                    manager_paused = True
                with self._condition:
                    self._manager_user_paused = already_paused
            except Exception as exc:
                self._reset_after_failed_request()
                raise SearchLeaseBusy(f"无法请求索引安全暂停：{exc}") from exc

        deadline = quiesce_started + max(0.0, float(max_wait_seconds))
        unconfirmed: tuple[str, ...] = ()
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self._resume_manager_if_needed(
                    manager,
                    pause_controller,
                    manager_paused,
                )
                self._reset_after_failed_request()
                raise OperationCancelled("搜索已取消")
            safe = True
            if manager is not None:
                try:
                    safe = bool(manager.is_safely_paused())
                    if not safe:
                        status = dict(manager.pause_status() or {})
                        unconfirmed = self._unconfirmed_from_status(status)
                except Exception as exc:
                    unconfirmed = (f"index_manager:{type(exc).__name__}",)
                    safe = False
            with self._condition:
                active_writes = self._active_index_writes
            if safe and active_writes == 0:
                break
            if active_writes:
                unconfirmed = (f"active_index_writes:{active_writes}",)
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                self._resume_manager_if_needed(
                    manager,
                    pause_controller,
                    manager_paused,
                )
                self._reset_after_failed_request(unconfirmed)
                detail = ", ".join(unconfirmed) or "索引安全检查点"
                raise QuiesceTimeout(f"等待索引安全暂停超时：{detail}")
            time.sleep(min(0.05, remaining))

        try:
            lease = acquire_search_lease(
                request_id,
                cancel_event=cancel_event,
                max_wait_seconds=max(0.0, deadline - time.perf_counter()),
            )
        except TimeoutError as exc:
            self._resume_manager_if_needed(
                manager,
                pause_controller,
                manager_paused,
            )
            self._reset_after_failed_request(unconfirmed)
            raise SearchLeaseBusy("等待索引写入器排空超时") from exc
        except RuntimeError as exc:
            self._resume_manager_if_needed(
                manager,
                pause_controller,
                manager_paused,
            )
            self._reset_after_failed_request(unconfirmed)
            raise OperationCancelled(str(exc) or "搜索已取消") from exc

        lease._release_callback = lambda: self._release_search(
            manager,
            pause_controller,
            manager_paused,
        )
        with self._condition:
            self._lease = lease
            self._state = SEARCH_ACTIVE
            self._waited_ms = int((time.perf_counter() - started) * 1000)
            self._unconfirmed = unconfirmed
            self._condition.notify_all()
        return lease

    def _release_search(
        self,
        manager: Any | None,
        pause_controller: Any | None,
        manager_paused: bool,
    ) -> None:
        with self._condition:
            was_stopping = self._state == STOPPING
            if was_stopping:
                self._lease = None
                self._deferred_writes.clear()
                self._request_id = None
                self._manager_user_paused = False
                self._unconfirmed = ()
                self._condition.notify_all()
                return
            self._state = WRITE_DRAINING
            self._lease = None
            self._draining_thread_id = threading.get_ident()
            self._condition.notify_all()
        self._resume_manager_if_needed(
            manager,
            pause_controller,
            manager_paused,
        )
        self.drain_deferred_writes()
        with self._condition:
            dispatchers = tuple(self._dispatchers)
        for dispatcher in dispatchers:
            try:
                dispatcher.drain()
            except Exception:
                continue
        with self._condition:
            self._draining_thread_id = None
            self._state = (
                USER_PAUSED
                if manager is not None and self._manager_user_paused
                else INDEX_RUNNING
                if manager is not None
                else IDLE
            )
            self._request_id = None
            self._manager_user_paused = False
            self._unconfirmed = ()
            self._condition.notify_all()

    @staticmethod
    def _resume_manager_if_needed(
        manager: Any | None,
        pause_controller: Any | None,
        paused: bool,
    ) -> None:
        if not paused or manager is None:
            return
        try:
            # A user pause is represented by the manager's own token.  Only
            # resume a pause requested by this coordinator.
            if pause_controller is not None:
                pause_controller.request_coordinator_resume()
            else:
                manager.request_resume()
        except Exception:
            # Search lease release must remain idempotent and never mask the
            # query's result with a cleanup exception.
            return

    def _reset_after_failed_request(self, unconfirmed: tuple[str, ...] = ()) -> None:
        with self._condition:
            self._state = INDEX_RUNNING if self._manager_ref and self._manager_ref() else IDLE
            self._request_id = None
            self._manager_user_paused = False
            self._unconfirmed = unconfirmed
            self._waited_ms = 0
            self._condition.notify_all()

    @staticmethod
    def _unconfirmed_from_status(status: dict[str, object]) -> tuple[str, ...]:
        required = int(status.get("required_acknowledgements") or 0)
        received = int(status.get("received_acknowledgements") or 0)
        items = status.get("acknowledgements")
        if isinstance(items, (list, tuple)) and required > received:
            return (f"safe_points:{received}/{required}",)
        state = str(status.get("state") or "pausing")
        return (state,)

    def defer_write(self, operation: object, *, priority: int = 50, label: str = "") -> str:
        """Queue a write while search owns the lease; return its operation id."""

        with self._condition:
            self._sequence += 1
            operation_id = f"write:{self._sequence}:{uuid.uuid4().hex[:8]}"
            self._deferred_writes.append((int(priority), operation_id, operation))
            self._deferred_writes.sort(key=lambda item: (item[0], item[1]))
            self._condition.notify_all()
            return operation_id

    def begin_index_write(self) -> None:
        """Wait for search release, then account for one write transaction."""

        with self._condition:
            while self._state in {SEARCH_QUIESCING, SEARCH_ACTIVE} or (
                self._state == WRITE_DRAINING
                and self._draining_thread_id != threading.get_ident()
            ):
                if self._state == STOPPING:
                    raise OperationCancelled("应用正在关闭，索引写入已取消")
                self._condition.wait(timeout=0.1)
            if self._state == STOPPING:
                raise OperationCancelled("应用正在关闭，索引写入已取消")
            self._active_index_writes += 1
            self._condition.notify_all()

    def try_begin_index_write(self) -> bool:
        """Reserve a write only when it cannot delay a search quiesce."""

        with self._condition:
            if self._state in {
                SEARCH_QUIESCING,
                SEARCH_ACTIVE,
                WRITE_DRAINING,
                STOPPING,
            }:
                return False
            self._active_index_writes += 1
            self._condition.notify_all()
            return True

    def end_index_write(self) -> None:
        with self._condition:
            self._active_index_writes = max(0, self._active_index_writes - 1)
            self._condition.notify_all()

    def drain_deferred_writes(self) -> int:
        """Run deferred callables in content/task/root/FTS/telemetry order."""

        with self._condition:
            pending = list(self._deferred_writes)
            self._deferred_writes.clear()
        completed = 0
        for priority, operation_id, operation in pending:
            if not callable(operation):
                continue
            try:
                operation()
            except Exception:
                # Put failed operations back for the next idle pass. This is
                # intentionally separate from real SQLite error reporting.
                with self._condition:
                    self._deferred_writes.append((priority, operation_id, operation))
                continue
            completed += 1
        return completed

    def cancel_deferred_writes(self) -> int:
        with self._condition:
            count = len(self._deferred_writes)
            self._deferred_writes.clear()
            return count

    def begin_stopping(self) -> None:
        with self._condition:
            self._state = STOPPING
            self._condition.notify_all()

    def snapshot(self) -> CoordinatorSnapshot:
        with self._condition:
            manager = self._manager_ref() if self._manager_ref else None
            active_tasks = 0
            writer_queue_depth = 0
            if manager is not None:
                try:
                    status = dict(manager.pause_status() or {})
                    active_tasks = max(0, int(status.get("active_tasks") or 0))
                    writer_queue_depth = max(0, int(status.get("writer_queue_depth") or 0))
                except Exception:
                    pass
            return CoordinatorSnapshot(
                self._state,
                self._request_id,
                self._index_run_id,
                max(active_tasks, self._active_index_writes),
                writer_queue_depth,
                len(self._deferred_writes),
                self._unconfirmed,
                self._waited_ms,
            )


_COORDINATOR = SearchIndexCoordinator()


def get_search_index_coordinator() -> SearchIndexCoordinator:
    return _COORDINATOR
