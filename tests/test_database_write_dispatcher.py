from __future__ import annotations

import threading
import time

from local_full_text_search.core.database_write_dispatcher import (
    CANCELLED,
    COMMITTED,
    DEFERRED,
    FAILED_FATAL,
    DatabaseWriteDispatcher,
)
from local_full_text_search.core.search_index_coordinator import (
    SEARCH_ACTIVE,
    SearchIndexCoordinator,
)


def test_dispatcher_defers_during_search_and_drains_after_release() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)
    lease = coordinator.request_search("dispatcher-search", max_wait_seconds=0.1)
    calls: list[str] = []
    try:
        item = dispatcher.submit(lambda: calls.append("content"), priority=10)
        assert item.status == DEFERRED
        assert calls == []
        assert coordinator.snapshot().state == SEARCH_ACTIVE
    finally:
        lease.release()

    assert calls == ["content"]
    assert dispatcher.pending_count() == 0


def test_dispatcher_marks_non_sql_errors_fatal() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)

    item = dispatcher.submit(
        lambda: (_ for _ in ()).throw(ValueError("bad payload")),
        label="invalid",
    )

    assert item.status == FAILED_FATAL
    assert item.error == "bad payload"


def test_dispatcher_commits_successful_operation() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)
    item = dispatcher.submit(lambda: 42)

    assert item.status == COMMITTED
    assert item.result == 42


def test_dispatcher_cancels_new_writes_during_shutdown() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)
    coordinator.begin_stopping()

    item = dispatcher.submit(lambda: 42)

    assert item.status == CANCELLED


def test_dispatcher_serializes_concurrent_operations() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def operation() -> None:
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            time.sleep(0.01)
        finally:
            with state_lock:
                state["active"] -= 1

    def submit_one() -> None:
        try:
            barrier.wait(timeout=2)
            item = dispatcher.submit(
                operation,
                priority=10,
                label="concurrent-test",
                defer_if_search=False,
            )
            assert item.status == COMMITTED
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=submit_one) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=3)

    assert not errors
    assert state["active"] == 0
    assert state["maximum"] == 1
    assert dispatcher.active_operation() is None


def test_dispatcher_can_drop_low_priority_write_when_writer_is_busy() -> None:
    coordinator = SearchIndexCoordinator()
    dispatcher = DatabaseWriteDispatcher(coordinator)
    started = threading.Event()
    release = threading.Event()

    def long_operation() -> None:
        started.set()
        assert release.wait(timeout=2)

    holder = threading.Thread(
        target=lambda: dispatcher.submit(
            long_operation,
            priority=10,
            label="long-critical-write",
            defer_if_search=False,
        )
    )
    holder.start()
    assert started.wait(timeout=1)

    telemetry = dispatcher.submit(
        lambda: None,
        priority=90,
        label="heartbeat",
        defer_if_search=False,
        max_wait_seconds=0.0,
    )
    assert telemetry.status == DEFERRED
    assert telemetry.error == "dispatcher_busy_timeout"

    release.set()
    holder.join(timeout=2)
    assert not holder.is_alive()


def test_dispatchers_for_same_database_share_writer_state(tmp_path) -> None:
    coordinator = SearchIndexCoordinator()
    database_path = tmp_path / "candidate.db"
    first = DatabaseWriteDispatcher(coordinator, database_path=database_path)
    second = DatabaseWriteDispatcher(coordinator, database_path=database_path)
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()
    barrier = threading.Barrier(2)

    def operation() -> None:
        with state_lock:
            state["active"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
        try:
            time.sleep(0.02)
        finally:
            with state_lock:
                state["active"] -= 1

    def submit(dispatcher: DatabaseWriteDispatcher) -> None:
        barrier.wait(timeout=2)
        assert dispatcher.submit(
            operation,
            priority=10,
            defer_if_search=False,
        ).status == COMMITTED

    threads = [
        threading.Thread(target=submit, args=(first,)),
        threading.Thread(target=submit, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert all(not thread.is_alive() for thread in threads)
    assert state["maximum"] == 1
