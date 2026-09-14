from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from local_full_text_search.core.errors import OperationCancelled, SearchLeaseBusy
from local_full_text_search.core.foreground_activity import search_status
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_index_coordinator import (
    IDLE,
    INDEX_RUNNING,
    SEARCH_ACTIVE,
    SEARCH_QUIESCING,
    STOPPING,
    WRITE_DRAINING,
    SearchIndexCoordinator,
    get_search_index_coordinator,
)


class FakeIndexManager:
    def __init__(self, *, safe_after: int = 0) -> None:
        self.pause_calls = 0
        self.resume_calls = 0
        self.safe_checks = 0
        self.safe_after = safe_after
        self.state = "running"

    def request_pause(self) -> None:
        self.pause_calls += 1
        self.state = "pausing"

    def request_resume(self) -> None:
        self.resume_calls += 1
        self.state = "running"

    def is_safely_paused(self) -> bool:
        self.safe_checks += 1
        if self.safe_checks > self.safe_after:
            self.state = "paused"
            return True
        return False

    def pause_status(self) -> dict[str, object]:
        return {
            "state": self.state,
            "required_acknowledgements": 1,
            "received_acknowledgements": 0,
            "acknowledgements": [],
        }


class FakePauseController:
    def __init__(self) -> None:
        self.pause_calls = 0
        self.resume_calls = 0

    def request_coordinator_pause(self) -> None:
        self.pause_calls += 1

    def request_coordinator_resume(self) -> None:
        self.resume_calls += 1


def test_search_lease_is_granted_only_after_index_safe_point() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager(safe_after=2)
    coordinator.register_index_manager(manager, index_run_id="run-1")

    lease = coordinator.request_search("query-1", max_wait_seconds=1)
    try:
        assert manager.pause_calls == 1
        assert manager.safe_checks >= 3
        assert coordinator.snapshot().state == SEARCH_ACTIVE
        assert search_status()["active_count"] == 1
    finally:
        lease.release()

    assert manager.resume_calls == 1
    assert coordinator.snapshot().state == INDEX_RUNNING
    assert search_status()["active_count"] == 0


def test_search_pause_timeout_has_specific_error_and_resumes_index() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager(safe_after=10_000)
    coordinator.register_index_manager(manager)

    with pytest.raises(SearchLeaseBusy, match="安全暂停超时"):
        coordinator.request_search("query-timeout", max_wait_seconds=0.03)

    assert manager.resume_calls == 1
    assert coordinator.snapshot().state == INDEX_RUNNING
    assert search_status()["active_count"] == 0


def test_cancel_during_quiescing_does_not_leave_a_lease() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager(safe_after=10_000)
    coordinator.register_index_manager(manager)
    cancelled = threading.Event()

    def cancel_later() -> None:
        time.sleep(0.02)
        cancelled.set()

    thread = threading.Thread(target=cancel_later)
    thread.start()
    with pytest.raises(OperationCancelled):
        coordinator.request_search(
            "query-cancel",
            cancel_event=cancelled,
            max_wait_seconds=1,
        )
    thread.join()
    assert manager.resume_calls == 1
    assert coordinator.snapshot().state == INDEX_RUNNING
    assert search_status()["active_count"] == 0


def test_deferred_writes_drain_in_priority_order() -> None:
    coordinator = SearchIndexCoordinator()
    completed: list[str] = []
    coordinator.defer_write(lambda: completed.append("telemetry"), priority=90)
    coordinator.defer_write(lambda: completed.append("content"), priority=10)
    coordinator.defer_write(lambda: completed.append("fts"), priority=70)

    assert coordinator.drain_deferred_writes() == 3
    assert completed == ["content", "fts", "telemetry"]
    assert coordinator.snapshot().deferred_write_count == 0


def test_search_uses_scan_controller_to_stop_new_submissions() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager()
    controller = FakePauseController()
    coordinator.register_index_manager(manager, pause_controller=controller)

    lease = coordinator.request_search("controlled-search", max_wait_seconds=1)
    try:
        assert controller.pause_calls == 1
        assert manager.pause_calls == 0
    finally:
        lease.release()

    assert controller.resume_calls == 1
    assert manager.resume_calls == 0


def test_write_reservation_is_rejected_while_search_is_quiescing() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager(safe_after=10_000)
    coordinator.register_index_manager(manager)
    started = threading.Event()
    completed = threading.Event()

    def request() -> None:
        started.set()
        with pytest.raises(SearchLeaseBusy):
            coordinator.request_search("slow-quiesce", max_wait_seconds=0.05)
        completed.set()

    thread = threading.Thread(target=request)
    thread.start()
    assert started.wait(timeout=1)
    deadline = time.monotonic() + 1
    while coordinator.snapshot().state != SEARCH_QUIESCING and time.monotonic() < deadline:
        time.sleep(0.005)
    assert coordinator.snapshot().state == SEARCH_QUIESCING
    assert coordinator.try_begin_index_write() is False
    thread.join(timeout=1)
    assert completed.is_set()


def test_stopping_releases_search_without_resuming_or_draining() -> None:
    coordinator = SearchIndexCoordinator()
    manager = FakeIndexManager()
    coordinator.register_index_manager(manager)
    lease = coordinator.request_search("stopping-search", max_wait_seconds=1)
    drained: list[bool] = []
    coordinator.defer_write(lambda: drained.append(True))

    coordinator.begin_stopping()
    lease.release()

    assert coordinator.snapshot().state == STOPPING
    assert manager.resume_calls == 0
    assert drained == []


def test_index_manager_unregisters_when_run_start_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "files"
    root.mkdir()
    (root / "one.txt").write_text("content", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    manager = IndexManager(database, AppSettings(enable_ocr=False))

    def fail_start(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("injected run start failure")

    monkeypatch.setattr(database, "start_index_run", fail_start)
    with pytest.raises(RuntimeError, match="injected run start failure"):
        manager.index_root(root_id)

    snapshot = get_search_index_coordinator().snapshot()
    assert snapshot.state == IDLE
    assert snapshot.index_run_id is None
    assert snapshot.active_tasks == 0
