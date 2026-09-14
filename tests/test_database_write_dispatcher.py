from __future__ import annotations

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
