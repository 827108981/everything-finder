from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.compact_snapshot_schema import (
    configure_new_database,
    create_compact_snapshot_schema,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_change_detector import check_published_index_changes
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.index_manager import parser_identity_for_path
from local_full_text_search.workers.scan_worker import ScanWorker


def _publish_one_file_index(tmp_path: Path) -> tuple[IndexGenerationStore, Path, Path]:
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "guide.txt"
    source.write_text("stable contents", encoding="utf-8")

    control_path = tmp_path / "control.db"
    control = DatabaseManager(control_path)
    control.initialize()
    root_id = control.add_root(root)

    candidate_store = IndexGenerationStore(tmp_path / "search_index.db")
    candidate = candidate_store.begin_candidate()
    stat = source.stat()
    parser_name, parser_version = parser_identity_for_path(
        source, AppSettings(enable_ocr=False)
    )
    con = sqlite3.connect(candidate.path)
    try:
        configure_new_database(con)
        create_compact_snapshot_schema(con)
        con.execute(
            "INSERT INTO roots(id, path, include_subfolders, enabled, status) VALUES (?, ?, 1, 1, 'ready')",
            (root_id, str(root)),
        )
        con.execute(
            """
            INSERT INTO files(
                id, root_id, path, filename, extension, size_bytes,
                modified_time, modified_ns, quick_fingerprint,
                parser_name, parser_version, parse_status
            ) VALUES (1, ?, ?, ?, '.txt', ?, ?, ?, ?, ?, ?, 'success')
            """,
            (
                root_id,
                str(source),
                source.name,
                stat.st_size,
                stat.st_mtime,
                stat.st_mtime_ns,
                f"{stat.st_size}:{stat.st_mtime_ns}",
                parser_name,
                parser_version,
            ),
        )
        con.commit()
    finally:
        con.close()
    candidate_store.publish(
        candidate,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-10-20T00:00:00+00:00",
        scan_signature=None,
    )
    return candidate_store, control_path, source


def _parser_identity(_path: Path, _settings: AppSettings) -> tuple[str, str]:
    return parser_identity_for_path(_path, _settings)


def test_unchanged_full_scan_does_not_create_a_new_generation(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)
    before = store.active()
    assert before is not None and before.generation == 1

    check = check_published_index_changes(
        store,
        control_path,
        AppSettings(),
        parser_identity=_parser_identity,
    )

    assert check.unchanged is True
    assert check.active_generation == 1
    assert check.checked_files == 1
    assert store.active() is not None and store.active().generation == 1
    assert store.scan_signature()
    assert not list(tmp_path.glob("*.building.db"))


def test_scan_worker_reuses_active_generation_when_nothing_changed(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)
    finished: list[dict[str, object]] = []
    worker = ScanWorker(
        store.database_path,
        AppSettings(enable_ocr=False),
        control_db_path=control_path,
    )
    worker.finished.connect(finished.append)

    worker.run()

    assert finished and finished[0]["unchanged"] is True
    assert finished[0]["active_generation"] == 1
    assert store.active() is not None and store.active().generation == 1
    assert not list(tmp_path.glob("*.building.db"))


def test_changed_file_forces_a_candidate_update(tmp_path: Path) -> None:
    store, control_path, source = _publish_one_file_index(tmp_path)
    source.write_text("changed contents", encoding="utf-8")

    check = check_published_index_changes(
        store,
        control_path,
        AppSettings(),
        parser_identity=_parser_identity,
    )

    assert check.unchanged is False
    assert check.reason == "filesystem_or_parser_changed"
    assert check.changed_files >= 1
    assert check.scan_signature


def test_explicit_file_retry_always_forces_indexing(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)

    check = check_published_index_changes(
        store,
        control_path,
        AppSettings(),
        file_ids=(101,),
        parser_identity=_parser_identity,
    )

    assert check.unchanged is False
    assert check.reason == "explicit_file_retry"


def test_retry_failed_files_setting_disables_noop_shortcut(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)

    check = check_published_index_changes(
        store,
        control_path,
        AppSettings(enable_ocr=False, retry_failed_files=True),
        parser_identity=_parser_identity,
    )

    assert check.unchanged is False
    assert check.reason == "retry_failed_files_enabled"


def test_indexing_setting_change_does_not_reuse_old_generation(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)
    original_settings = AppSettings(enable_ocr=False)

    initial_check = check_published_index_changes(
        store,
        control_path,
        original_settings,
        parser_identity=_parser_identity,
    )
    assert initial_check.unchanged is True

    changed_settings = replace(original_settings, block_target_chars=2048)
    changed_check = check_published_index_changes(
        store,
        control_path,
        changed_settings,
        parser_identity=_parser_identity,
    )

    assert changed_check.unchanged is False
    assert changed_check.reason == "scan_signature_changed"


def test_full_hash_setting_change_does_not_reuse_old_generation(tmp_path: Path) -> None:
    store, control_path, _source = _publish_one_file_index(tmp_path)
    original_settings = AppSettings(enable_ocr=False)
    check_published_index_changes(
        store,
        control_path,
        original_settings,
        parser_identity=_parser_identity,
    )

    changed_check = check_published_index_changes(
        store,
        control_path,
        replace(original_settings, compute_full_hash=True),
        parser_identity=_parser_identity,
    )

    assert changed_check.unchanged is False
    assert changed_check.reason == "scan_signature_changed"
