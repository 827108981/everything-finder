from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QMessageBox

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.ui.main_window import FailedPage, MainWindow, PAGE_INDEX
from local_full_text_search.workers.scope_exclusion_worker import (
    ScopeExclusionWorker,
)


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _scope_database(
    tmp_path: Path,
) -> tuple[DatabaseManager, int, Path, Path]:
    root = tmp_path / "root"
    root.mkdir()
    included = root / "included.txt"
    blocked = root / "blocked.pdf"
    included.write_text("included source", encoding="utf-8")
    blocked.write_bytes(b"%PDF-1.4\nbroken source must stay unchanged")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    included_id, _ = database.upsert_file_metadata(root_id, included)
    blocked_id, _ = database.upsert_file_metadata(root_id, blocked)
    token = "BACKGROUND_EXCLUSION_INCLUDED_TOKEN"
    database.replace_document_blocks_many(
        [
            {
                "file_id": included_id,
                "file_ids": [included_id],
                "filename": included.name,
                "path": str(included),
                "blocks": [
                    ContentBlock(
                        file_path=str(included),
                        block_index=0,
                        block_type="paragraph",
                        location_text="正文",
                        raw_text=token,
                        normalized_text=token.lower(),
                    )
                ],
                "parser_name": "text",
                "parser_version": "scope-worker-test-v1",
                "status": "success",
                "content_key": "scope-worker-included",
                "task_id": None,
            }
        ]
    )
    database.record_failure(
        blocked_id,
        "PDF_CORRUPTED",
        "injected blocker",
        parser_name="pdf",
    )
    database.update_root_scan_time(root_id, "incomplete")
    database.begin_deferred_fts()
    with database.connect() as connection:
        connection.execute("DELETE FROM content_fts")
    return database, blocked_id, included, blocked


def _wait_for_exclusion(window: MainWindow, timeout: float = 5.0) -> None:
    app = _app()
    deadline = time.monotonic() + timeout
    while window.exclusion_thread is not None and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert window.exclusion_thread is None


def test_scope_exclusion_worker_reports_stages_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    _app()
    database, blocked_id, _included, blocked = _scope_database(tmp_path)
    original_bytes = blocked.read_bytes()
    progress: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    failures: list[str] = []
    worker = ScopeExclusionWorker(
        database.db_path,
        [blocked_id],
        reason="confirmed unrecoverable",
        operation_source="test",
    )
    worker.progress.connect(progress.append)
    worker.finished.connect(results.append)
    worker.failed.connect(failures.append)

    worker.run()

    assert failures == []
    assert len(results) == 1
    result = results[0]
    assert result["excluded_files"] == 1
    assert result["triggered_full_fts_rebuild"] is True
    assert result["ready"] is True
    stages = {str(item["stage"]) for item in progress}
    assert {
        "recording_exclusions",
        "cleaning_content_fts",
        "rebuilding_content_fts",
        "updating_filename_fts",
        "refreshing_index_state",
    }.issubset(stages)
    assert blocked.read_bytes() == original_bytes
    assert SearchEngine(database).search(
        SearchQuery(text="BACKGROUND_EXCLUSION_INCLUDED_TOKEN", mode="exact")
    ).total_confirmed == 1
    with database.connect() as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_scope_exclusion_worker_cancel_rolls_back_audit_and_fts(
    tmp_path: Path,
) -> None:
    _app()
    database, blocked_id, _included, _blocked = _scope_database(tmp_path)
    with database.connect() as connection:
        before = (
            int(connection.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]),
        )
    worker = ScopeExclusionWorker(
        database.db_path,
        [blocked_id],
        reason="cancelled operation",
        operation_source="test",
    )
    cancelled: list[bool] = []

    def cancel_after_audit(payload: dict[str, object]) -> None:
        if (
            payload.get("stage") == "recording_exclusions"
            and int(payload.get("processed_files") or 0) == 1
        ):
            worker.cancel()

    worker.progress.connect(cancel_after_audit)
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.run()

    assert cancelled == [True]
    assert database.excluded_files(include_history=True) == []
    with database.connect() as connection:
        after = (
            int(connection.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM files_fts").fetchone()[0]),
        )
    assert after == before
    assert database.index_readiness()["blocking_files"] == 1


def test_force_complete_worker_repairs_zero_blocker_residual_state(
    tmp_path: Path,
) -> None:
    _app()
    root = tmp_path / "root"
    root.mkdir()
    included = root / "included.txt"
    included.write_text("ZERO_BLOCKER_REPAIR_TOKEN", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    included_id, _ = database.upsert_file_metadata(root_id, included)
    database.replace_document_blocks_many(
        [
            {
                "file_id": included_id,
                "file_ids": [included_id],
                "filename": included.name,
                "path": str(included),
                "blocks": [
                    ContentBlock(
                        file_path=str(included),
                        block_index=0,
                        block_type="paragraph",
                        location_text="正文",
                        raw_text="ZERO_BLOCKER_REPAIR_TOKEN",
                        normalized_text="zero_blocker_repair_token",
                    )
                ],
                "parser_name": "text",
                "parser_version": "force-worker-test-v1",
                "status": "success",
                "content_key": "force-worker-included",
                "task_id": None,
            }
        ]
    )
    database.create_parse_task(included_id, "stale-run", "normal")
    database.update_root_scan_time(root_id, "incomplete")
    database.begin_deferred_fts()
    progress: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    worker = ScopeExclusionWorker(
        database.db_path,
        [],
        reason="repair residual state",
        operation_source="test",
        force_complete=True,
    )
    worker.progress.connect(progress.append)
    worker.finished.connect(results.append)

    worker.run()

    assert len(results) == 1
    assert results[0]["operation"] == "force_complete"
    assert results[0]["excluded_files"] == 0
    assert results[0]["invalidated_tasks"] == 1
    assert results[0]["ready"] is True
    assert "rebuilding_content_fts" in {str(item["stage"]) for item in progress}
    assert SearchEngine(database).search(
        SearchQuery(text="ZERO_BLOCKER_REPAIR_TOKEN", mode="exact")
    ).total_confirmed == 1


def test_failed_page_shows_busy_progress_and_disables_conflicting_actions() -> None:
    _app()
    page = FailedPage()
    page.set_rows(
        [
            {
                "id": 1,
                "path": r"E:\scope\blocked.pdf",
                "filename": "blocked.pdf",
                "extension": ".pdf",
                "parse_status": "failed",
            }
        ]
    )

    page.set_exclusion_running(True)
    page.set_exclusion_progress(
        {
            "stage": "recording_exclusions",
            "phase_label": "正在记录排除范围",
            "processed_files": 1,
            "total_files": 3,
            "large_fts_operation": False,
            "can_cancel": True,
        }
    )
    assert page.exclusion_progress.isHidden() is False
    assert page.exclusion_progress.maximum() == 3
    assert page.exclusion_progress.value() == 1
    assert not page.retry_button.isEnabled()
    assert not page.exclude_button.isEnabled()
    assert not page.restore_button.isEnabled()
    assert not page.force_complete_button.isEnabled()
    assert not page.table.isEnabled()

    page.set_exclusion_progress(
        {
            "stage": "rebuilding_content_fts",
            "phase_label": "正在重建全文索引",
            "large_fts_operation": True,
            "can_cancel": True,
        }
    )
    assert page.exclusion_progress.minimum() == 0
    assert page.exclusion_progress.maximum() == 0
    assert "大型 FTS 操作" in page.status.text()

    page.set_exclusion_running(False)
    assert page.exclusion_progress.isHidden() is True
    assert page.retry_button.isEnabled()
    assert page.exclude_button.isEnabled()


def test_main_window_background_exclusion_cleans_thread_state(
    tmp_path: Path,
) -> None:
    app = _app()
    database, blocked_id, _included, _blocked = _scope_database(tmp_path)
    window = MainWindow(
        database,
        AppSettings(monitor_file_changes=False),
        SettingsService(tmp_path / "settings.json"),
    )

    started = time.monotonic()
    window.start_scope_exclusion([blocked_id], "background UI test")
    assert time.monotonic() - started < 0.5
    assert window.exclusion_thread is not None
    assert window.failed_page.exclusion_running is True
    app.processEvents()
    _wait_for_exclusion(window)

    assert window.exclusion_worker is None
    assert window.failed_page.exclusion_running is False
    assert window.stack.currentIndex() == PAGE_INDEX["search"]
    assert window.search_page.index_ready() is True
    window.close()


def test_main_window_cancel_and_failure_restore_controls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    app = _app()
    database, blocked_id, _included, _blocked = _scope_database(tmp_path)
    window = MainWindow(
        database,
        AppSettings(monitor_file_changes=False),
        SettingsService(tmp_path / "settings.json"),
    )

    window.start_scope_exclusion([blocked_id], "cancel UI test")
    window.cancel_scope_exclusion()
    _wait_for_exclusion(window)
    assert database.excluded_files(include_history=True) == []
    assert window.failed_page.exclusion_running is False
    assert "已取消" in window.failed_page.status.text()

    critical_messages: list[str] = []
    monkeypatch.setattr(
        QMessageBox,
        "critical",
        lambda _parent, _title, message: critical_messages.append(str(message)),
    )
    window.start_scope_exclusion([999_999], "failure UI test")
    _wait_for_exclusion(window)
    assert window.exclusion_worker is None
    assert window.failed_page.exclusion_running is False
    assert critical_messages
    assert "回滚" in window.failed_page.status.text()
    window.close()
