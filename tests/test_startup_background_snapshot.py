from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.ui.main_window import MainWindow
from local_full_text_search.workers import database_snapshot_worker as snapshot_module
from local_full_text_search.workers.database_snapshot_worker import (
    DatabaseSnapshotWorker,
    collect_live_candidate_failure_snapshot,
    merge_failure_report,
    merge_control_snapshot_with_published,
    merge_live_candidate_failures,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module", autouse=True)
def _qt_module_lifecycle():
    """Do not leak QApplication state into the next pytest module."""

    app = _app()
    yield
    for widget in list(app.topLevelWidgets()):
        try:
            widget.close()
            widget.deleteLater()
        except RuntimeError:
            pass
    app.processEvents()
    app.sendPostedEvents(None, 0)
    app.quit()
    app.processEvents()
    try:
        import shiboken6

        shiboken6.delete(app)
    except (ImportError, RuntimeError):
        # Some PySide builds do not expose a deletable wrapper. The normal
        # event-queue cleanup above is still sufficient for those builds.
        pass


def _wait_until(
    predicate: object,
    *,
    timeout: float = 4.0,
) -> bool:
    app = _app()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():  # type: ignore[operator]
            return True
        time.sleep(0.01)
    app.processEvents()
    return bool(predicate())  # type: ignore[operator]


def _window(tmp_path: Path) -> MainWindow:
    _app()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root = tmp_path / "files"
    root.mkdir()
    database.add_root(root)
    settings = AppSettings(
        enable_ocr=False,
        monitor_file_changes=False,
    )
    settings_service = SettingsService(tmp_path / "settings.json")
    settings_service.save(settings)
    window = MainWindow(
        database,
        settings,
        settings_service,
        defer_initial_refresh=True,
    )
    window.show()
    return window


def _close_test_window(window: MainWindow, timer: QTimer | None = None) -> None:
    """Release Qt resources before pytest tears down QApplication.

    These tests run inside the desktop app's command host. Leaving a hidden
    top-level window or a QThread-owned worker for interpreter teardown can
    make Qt abort the native process instead of returning a pytest failure.
    """

    if timer is not None:
        timer.stop()
        timer.deleteLater()
    window.close()
    window.deleteLater()
    app = _app()
    app.processEvents()
    app.sendPostedEvents(None, 0)
    app.processEvents()


def test_deferred_startup_snapshot_keeps_qt_event_loop_responsive(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    original = DatabaseManager.index_readiness

    def deliberately_slow_readiness(self: DatabaseManager) -> dict[str, object]:
        time.sleep(0.5)
        return original(self)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        DatabaseManager,
        "index_readiness",
        deliberately_slow_readiness,
    )
    window = _window(tmp_path)
    heartbeat = {"count": 0}
    timer = QTimer()
    timer.setInterval(20)
    timer.timeout.connect(
        lambda: heartbeat.__setitem__("count", heartbeat["count"] + 1)
    )
    timer.start()

    assert _wait_until(lambda: heartbeat["count"] >= 5, timeout=0.35)
    assert window.database_snapshot_thread is not None
    assert _wait_until(
        lambda: window.database_snapshot_thread is None,
        timeout=4.0,
    )

    _close_test_window(window, timer)


def test_startup_incremental_scan_waits_for_snapshot_completion(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    original = DatabaseManager.index_readiness

    def deliberately_slow_readiness(self: DatabaseManager) -> dict[str, object]:
        time.sleep(0.35)
        return original(self)

    monkeypatch.setattr(  # type: ignore[attr-defined]
        DatabaseManager,
        "index_readiness",
        deliberately_slow_readiness,
    )
    window = _window(tmp_path)
    scan_calls: list[bool] = []
    monkeypatch.setattr(  # type: ignore[attr-defined]
        window,
        "start_scan",
        lambda *args, **kwargs: scan_calls.append(True),
    )

    window.schedule_startup_incremental_scan()
    _wait_until(lambda: False, timeout=0.15)
    assert scan_calls == []
    assert _wait_until(lambda: bool(scan_calls), timeout=4.0)

    _close_test_window(window)


def test_snapshot_timeout_without_cache_emits_degraded_finished(monkeypatch, tmp_path: Path) -> None:
    worker = DatabaseSnapshotWorker(
        tmp_path / "missing.db",
        page_size=20,
        page_offsets={},
        snapshot_timeout_seconds=1,
    )
    monkeypatch.setattr(worker.db, "list_roots", lambda: [])
    monkeypatch.setattr(
        worker.db,
        "quick_index_readiness",
        lambda: {"search_available": True, "ready": False, "active_runs": 1},
    )
    def interrupted(*_args, **_kwargs):
        raise sqlite3.OperationalError("interrupted")
    monkeypatch.setattr(snapshot_module, "collect_database_snapshot", interrupted)
    worker._timed_out = True
    finished: list[dict[str, object]] = []
    failures: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == []
    assert finished and finished[0]["readiness_degraded"] is True
    assert finished[0]["snapshot_timeout"] is True


def test_control_snapshot_uses_published_file_counts_and_keeps_scope_state() -> None:
    merged = merge_control_snapshot_with_published(
        {
            "roots": [
                {
                    "id": 1,
                    "path": r"E:\scope",
                    "enabled": 1,
                    "status": "pending",
                    "last_scan_at": None,
                }
            ],
            "stats": {"file_count": 0, "complete_files": 0},
            "readiness": {
                "file_count": 0,
                "active_runs": 0,
                "background_task_state": "converged",
            },
            "failed_counts": {"blocking": 0},
        },
        {
            "compact_snapshot": True,
            "roots": [{"id": 1, "path": r"E:\scope"}],
            "stats": {
                "file_count": 12,
                "complete_files": 12,
                "eligible_files": 12,
            },
            "readiness": {
                "file_count": 12,
                "complete_files": 12,
                "eligible_files": 12,
                "search_available": True,
                "ready": True,
            },
            "root_stats_by_id": {1: {".txt": 12}},
        },
    )

    assert merged["stats"]["file_count"] == 12
    assert merged["root_stats_by_id"] == {1: {".txt": 12}}
    assert merged["roots"][0]["status"] == "pending"
    assert merged["readiness"]["search_available"] is True
    assert merged["readiness"]["ready"] is False


def test_failure_report_is_visible_after_candidate_database_is_removed() -> None:
    merged = merge_failure_report(
        {
            "roots": [],
            "stats": {"file_count": 10},
            "readiness": {
                "search_available": True,
                "ready": True,
                "blocking_files": 0,
            },
            "failed_rows": [],
            "failed_counts": {"blocking": 0},
            "all_blocking_ids": [],
        },
        {
            "candidate_generation": 3,
            "error": "candidate blocker",
            "blocking_files": [
                {
                    "id": 2320,
                    "path": r"E:\blocked.zip",
                    "parse_status": "skipped",
                    "parse_error_code": "ZIP_FILE_COUNT_LIMIT",
                }
            ],
        },
    )

    assert merged["failed_counts"]["blocking"] == 1
    assert merged["all_blocking_ids"] == [2320]
    assert merged["failed_rows"][0]["path"] == r"E:\blocked.zip"
    assert merged["readiness"]["latest_candidate_failure"] is True
    assert merged["readiness"]["search_available"] is True
    assert merged["readiness"]["ready"] is False


def test_live_candidate_failure_rows_override_published_snapshot(
    tmp_path: Path,
) -> None:
    _app()
    control_path = tmp_path / "index_control.db"
    control = DatabaseManager(control_path)
    control.initialize()
    control_root = tmp_path / "control-root"
    control_root.mkdir()
    control.add_root(control_root)

    generation_base = tmp_path / "search_index.db"
    store = snapshot_module.IndexGenerationStore(generation_base)
    candidate = store.begin_candidate()
    candidate_db = DatabaseManager(candidate.path)
    candidate_db.initialize()
    candidate_root = tmp_path / "candidate-root"
    candidate_root.mkdir()
    root_id = candidate_db.add_root(candidate_root)
    source = candidate_root / "blocked.zip"
    source.write_bytes(b"not-a-zip")
    file_id, _ = candidate_db.upsert_file_metadata(root_id, source)
    with candidate_db.connect() as connection:
        connection.execute(
            """
            UPDATE files
            SET parse_status = 'failed_retryable',
                parse_error_code = 'TEST_CANDIDATE_FAILURE',
                parse_error_message = 'candidate failure'
            WHERE id = ?
            """,
            (file_id,),
        )

    live = collect_live_candidate_failure_snapshot(
        candidate.path,
        page_size=20,
        page_offsets={},
        generation=candidate.generation,
    )
    assert live is not None
    merged = merge_live_candidate_failures(
        {
            "failed_rows": [],
            "failed_counts": {"blocking": 0},
            "all_blocking_ids": [],
            "readiness": {"search_available": True, "ready": True},
        },
        live,
    )
    assert merged["candidate_live_failures"] is True
    assert merged["failed_counts"]["blocking"] == 1
    assert merged["all_blocking_ids"] == [file_id]
    assert merged["failed_rows"][0]["path"] == str(source)
    assert merged["readiness"]["candidate_generation"] == candidate.generation
    store.abort_candidate(candidate)


def test_snapshot_worker_reads_live_candidate_failures_while_scan_is_active(
    tmp_path: Path,
) -> None:
    _app()
    control_path = tmp_path / "index_control.db"
    control = DatabaseManager(control_path)
    control.initialize()
    control_root = tmp_path / "control-root"
    control_root.mkdir()
    control.add_root(control_root)

    generation_base = tmp_path / "search_index.db"
    store = snapshot_module.IndexGenerationStore(generation_base)
    candidate = store.begin_candidate()
    candidate_db = DatabaseManager(candidate.path)
    candidate_db.initialize()
    candidate_root = tmp_path / "candidate-root"
    candidate_root.mkdir()
    root_id = candidate_db.add_root(candidate_root)
    source = candidate_root / "live-failure.txt"
    source.write_text("live candidate", encoding="utf-8")
    file_id, _ = candidate_db.upsert_file_metadata(root_id, source)
    with candidate_db.connect() as connection:
        connection.execute(
            "UPDATE files SET parse_status = 'failed', "
            "parse_error_code = 'LIVE_FAILURE' WHERE id = ?",
            (file_id,),
        )

    worker = DatabaseSnapshotWorker(
        control_path,
        page_size=20,
        page_offsets={},
        generation_base_path=generation_base,
    )
    finished: list[dict[str, object]] = []
    worker.finished.connect(finished.append)
    worker.run()

    assert finished
    assert finished[0]["candidate_live_failures"] is True
    assert finished[0]["failed_counts"]["blocking"] == 1
    assert finished[0]["failed_rows"][0]["id"] == file_id
    assert finished[0]["readiness"]["candidate_database_path"] == str(
        candidate.path.resolve()
    )
    store.abort_candidate(candidate)
