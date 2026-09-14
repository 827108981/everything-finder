from __future__ import annotations

import sqlite3
import json
import pytest
from types import SimpleNamespace
from pathlib import Path

from local_full_text_search.core.compact_snapshot_schema import (
    create_compact_snapshot_schema,
    configure_new_database,
    validate_compact_snapshot_schema,
)
from local_full_text_search.core.compact_snapshot_builder import (
    build_compact_snapshot,
    expand_compact_snapshot,
    expand_compact_snapshot_isolated,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.index_manager import (
    ExecutorLifecycleError,
    IndexManager,
    ParseJob,
    ParseLane,
    schedule_parse_lanes,
)
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.workers.scan_worker import ScanWorker
from local_full_text_search.workers.database_snapshot_worker import merge_failure_report
from local_full_text_search.core.usn_journal import UsnChangeSet, UsnCheckpoint


def _create_candidate(path: Path, generation: int) -> None:
    con = sqlite3.connect(path)
    try:
        configure_new_database(con)
        create_compact_snapshot_schema(con)
        con.execute(
            """
            INSERT INTO snapshot_meta(
                id, generation, schema_version, published_at,
                root_count, file_count, document_count, block_count,
                parser_version, state
            ) VALUES (1, ?, 1, 0, 0, 0, 0, 0, '', 0)
            """,
            (generation,),
        )
        con.commit()
    finally:
        con.close()


def test_compact_schema_has_no_runtime_or_short_token_tables(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.db"
    con = sqlite3.connect(path)
    try:
        configure_new_database(con)
        create_compact_snapshot_schema(con)
        con.commit()
    finally:
        con.close()

    result = validate_compact_snapshot_schema(path)
    assert result["valid"] is True
    assert result["missing"] == []
    assert result["forbidden"] == []


def test_generation_publish_switches_manifest_atomically(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    first = store.begin_candidate()
    _create_candidate(first.path, first.generation)
    published = store.publish(
        first,
        file_count=0,
        document_count=0,
        block_count=0,
        published_at="2026-09-03T10:00:00+00:00",
    )

    assert published.generation == 1
    active = store.active()
    assert active is not None
    assert active.path == published.path
    assert store.manifest_path.is_file()


def test_candidate_publish_rejects_stale_owner_nonce(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    candidate = store.begin_candidate(scan_id="scan-owner-test")
    _create_candidate(candidate.path, candidate.generation)
    store.build_lock_path.write_text(
        json.dumps(
            {
                "pid": candidate.generation,
                "owner_nonce": "different-owner",
                "scan_id": candidate.scan_id,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="所有权已失效"):
        store.publish(
            candidate,
            file_count=0,
            document_count=0,
            block_count=0,
            published_at="2026-09-04T01:00:00+00:00",
        )

    store.abort_candidate(candidate)


def test_candidate_failure_does_not_change_active_generation(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    first = store.begin_candidate()
    _create_candidate(first.path, first.generation)
    store.publish(
        first,
        file_count=0,
        document_count=0,
        block_count=0,
        published_at="2026-09-03T10:00:00+00:00",
    )
    before = store.active()
    assert before is not None

    second = store.begin_candidate()
    store.abort_candidate(second)
    after = store.active()
    assert after is not None
    assert after.generation == before.generation
    assert after.path == before.path


def test_dead_build_owner_is_recovered_without_touching_active(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    stale = tmp_path / "search_index.gen-000001.building.db"
    stale.write_bytes(b"stale")
    store.build_lock_path.write_text(
        '{"pid": 2147483647, "baseline_generation": 0}',
        encoding="utf-8",
    )
    candidate = store.begin_candidate()
    assert candidate.path == stale
    assert not candidate.path.exists()
    store.abort_candidate(candidate)


def test_active_snapshot_database_manager_rejects_writes(tmp_path: Path) -> None:
    path = tmp_path / "active.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE marker(value INTEGER)")
    con.commit()
    con.close()
    database = DatabaseManager(path, read_only=True)
    try:
        with database.connect():
            pass
    except sqlite3.OperationalError as exc:
        assert "只读" in str(exc)
    else:
        raise AssertionError("active snapshot unexpectedly exposed a writer connection")


def test_task_convergence_uses_temp_ids_under_sqlite_variable_limit(tmp_path: Path) -> None:
    db = DatabaseManager(tmp_path / "index.db")
    db.initialize()
    root = tmp_path / "tasks"
    root.mkdir()
    file_path = root / "pending.txt"
    file_path.write_text("pending", encoding="utf-8")
    root_id = db.add_root(root)
    file_id, _changed = db.upsert_file_metadata(root_id, file_path)
    run_id = "run-with-many-tasks"
    task_count = 2_000
    with db.connect() as con:
        con.executemany(
            "INSERT INTO parse_tasks(file_id, run_id, task_type, status, created_at) VALUES (?, ?, 'text', 'running', 'now')",
            ((file_id, run_id) for _ in range(task_count)),
        )
    with db.connect() as con:
        con.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, 999)
        converged = DatabaseManager._converge_index_run_tasks(con, run_id, "failed")
        remaining = int(
            con.execute(
                "SELECT COUNT(*) FROM parse_tasks WHERE run_id = ? AND status = 'running'",
                (run_id,),
            ).fetchone()[0]
        )
    assert converged == task_count
    assert remaining == 0


def test_scheduler_rejects_closed_executor_before_submit(tmp_path: Path) -> None:
    class ClosedExecutor:
        _shutdown = True

        def submit(self, *_args: object, **_kwargs: object) -> object:
            raise AssertionError("closed executor must not receive submit")

    lane = ParseLane("normal", ClosedExecutor(), 1, 1024)
    lane.pending.append(
        ParseJob(
            file_id=1,
            file_path=tmp_path / "closed.txt",
            lane="normal",
            size_bytes=1,
        )
    )
    with pytest.raises(ExecutorLifecycleError):
        schedule_parse_lanes(
            [lane],
            AppSettings(enable_ocr=False),
            CancelToken(),
            tmp_path,
        )


def test_compact_snapshot_searches_three_chars_and_scans_two_chars(tmp_path: Path) -> None:
    root = tmp_path / "files"
    root.mkdir()
    source_path = tmp_path / "working.db"
    db = DatabaseManager(source_path)
    db.initialize()
    root_id = db.add_root(root)
    file_path = root / "校准记录.txt"
    file_path.write_text("真空校准记录", encoding="utf-8")
    file_id, _changed = db.upsert_file_metadata(root_id, file_path)
    db.replace_file_blocks(
        file_id,
        file_path.name,
        str(file_path),
        [
            ContentBlock(
                file_path=str(file_path),
                block_index=0,
                block_type="text",
                location_text="正文",
                raw_text="真空校准记录",
                normalized_text="真空校准记录",
            )
        ],
        parser_name="text",
    )
    compact_path = tmp_path / "compact.db"
    build_compact_snapshot(source_path, compact_path, generation=1)

    engine = SearchEngine(DatabaseManager(compact_path), generation_isolated=True)
    two = engine.search(SearchQuery(text="真空", search_filename=False, search_path=False))
    three = engine.search(SearchQuery(text="真空校", search_filename=False, search_path=False))
    assert two.total_confirmed == 1
    assert three.total_confirmed == 1


def test_compact_two_character_search_hydrates_matching_file_beyond_first_page(
    tmp_path: Path,
) -> None:
    root = tmp_path / "many-block-files"
    root.mkdir()
    source_path = tmp_path / "working.db"
    db = DatabaseManager(source_path)
    db.initialize()
    root_id = db.add_root(root)
    file_path = root / "teacher.xls"
    file_path.write_text("placeholder", encoding="utf-8")
    file_id, _changed = db.upsert_file_metadata(root_id, file_path)
    blocks = [
        ContentBlock(
            file_path=str(file_path),
            block_index=index,
            block_type="sheet_row",
            location_text=f"第 {index + 1} 行",
            raw_text=f"填充数据 {index}",
            normalized_text=f"填充数据 {index}",
        )
        for index in range(120)
    ]
    blocks.append(
        ContentBlock(
            file_path=str(file_path),
            block_index=120,
            block_type="sheet_row",
            location_text="第 121 行",
            raw_text="教师 赵煜 课后服务",
            normalized_text="教师 赵煜 课后服务",
        )
    )
    db.replace_file_blocks(
        file_id,
        file_path.name,
        str(file_path),
        blocks,
        parser_name="text",
    )
    compact_path = tmp_path / "compact.db"
    build_compact_snapshot(source_path, compact_path, generation=1)

    page = SearchEngine(
        DatabaseManager(compact_path),
        generation_isolated=True,
    ).search(
        SearchQuery(
            text="赵煜",
            search_filename=False,
            search_path=False,
        )
    )

    assert page.total_confirmed == 1
    assert len(page.results) == 1
    assert page.results[0].file_path == str(file_path)


def test_generation_store_compacts_work_database_and_can_expand_it(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    candidate = store.begin_candidate()
    work = DatabaseManager(candidate.path)
    work.initialize()
    root = tmp_path / "work-files"
    root.mkdir()
    root_id = work.add_root(root)
    source = root / "sample.txt"
    source.write_text("真空校准新版本", encoding="utf-8")
    file_id, _changed = work.upsert_file_metadata(root_id, source)
    work.replace_file_blocks(
        file_id,
        source.name,
        str(source),
        [
            ContentBlock(
                file_path=str(source),
                block_index=0,
                block_type="text",
                location_text="正文",
                raw_text="真空校准新版本",
                normalized_text="真空校准新版本",
            )
        ],
        parser_name="text",
    )
    published = store.publish(
        candidate,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-09-03T10:00:00+00:00",
    )
    compact = DatabaseManager(published.path)
    with sqlite3.connect(published.path) as connection:
        connection.row_factory = sqlite3.Row
        compact_file = connection.execute(
            "SELECT quick_fingerprint, parser_name, parser_version, source_kind "
            "FROM files WHERE path = ?",
            (str(source),),
        ).fetchone()
    assert compact_file is not None
    assert compact_file["quick_fingerprint"]
    assert compact_file["parser_name"] == "text"
    assert compact_file["parser_version"]
    assert compact_file["source_kind"] == 4
    page = SearchEngine(compact, generation_isolated=True).search(
        SearchQuery(text="真空校", search_filename=False, search_path=False)
    )
    assert page.total_confirmed == 1

    next_candidate = store.begin_candidate()
    expanded_db = DatabaseManager(next_candidate.path)
    assert expanded_db.list_roots()[0]["path"] == str(root)
    expanded_readiness = expanded_db.search_projection_readiness()
    assert expanded_readiness["content_fts_dirty"] is False
    assert expanded_readiness["projection_ready"] is True
    store.abort_candidate(next_candidate)


def test_usn_state_round_trips_between_control_and_candidate_databases(
    tmp_path: Path,
) -> None:
    root = tmp_path / "usn-root"
    child = root / "nested"
    child.mkdir(parents=True)
    control = DatabaseManager(tmp_path / "control.db")
    control.initialize()
    root_id = control.add_root(root)
    control.save_usn_checkpoint(
        root_id,
        journal_id=123,
        next_usn=456,
        volume_serial=789,
    )
    control.refresh_usn_directories(
        root_id,
        root,
        include_subfolders=True,
    )
    state = control.export_usn_state([root_id])

    candidate = DatabaseManager(tmp_path / "candidate.db")
    candidate.initialize()
    candidate.add_root(root)
    candidate.import_usn_state(state, root_ids=[root_id])

    assert candidate.load_usn_checkpoint(root_id) == {
        "journal_id": "123",
        "next_usn": "456",
        "volume_serial": "789",
    }
    assert candidate.usn_directories_for_root(root_id) == control.usn_directories_for_root(
        root_id
    )


def test_restored_usn_state_selects_the_usn_incremental_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "usn-plan-root"
    root.mkdir()
    source = root / "source.txt"
    source.write_text("USN_PLAN_SOURCE", encoding="utf-8")

    control_path = tmp_path / "control.db"
    control = DatabaseManager(control_path)
    control.initialize()
    root_id = control.add_root(root)
    file_id, _changed = control.upsert_file_metadata(root_id, source)
    with control.connect() as connection:
        connection.execute(
            "UPDATE files SET file_reference = ? WHERE id = ?",
            (11, file_id),
        )
    control.save_usn_checkpoint(
        root_id,
        journal_id=123,
        next_usn=456,
        volume_serial=789,
    )
    control.refresh_usn_directories(
        root_id,
        root,
        include_subfolders=True,
    )

    candidate_path = tmp_path / "candidate.db"
    candidate = DatabaseManager(candidate_path)
    candidate.initialize()
    candidate_root_id = candidate.add_root(root)
    candidate_file_id, _changed = candidate.upsert_file_metadata(
        candidate_root_id,
        source,
    )
    with candidate.connect() as connection:
        connection.execute(
            "UPDATE files SET file_reference = ? WHERE id = ?",
            (11, candidate_file_id),
        )

    worker = ScanWorker(
        tmp_path / "search_index.db",
        AppSettings(enable_ocr=False),
        control_db_path=control_path,
    )
    worker._restore_usn_state_from_control(candidate)
    checkpoint = UsnCheckpoint(
        journal_id=123,
        next_usn=456,
        volume_serial=789,
    )
    monkeypatch.setattr(
        "local_full_text_search.core.index_manager.changes_since",
        lambda *_args, **_kwargs: UsnChangeSet(
            checkpoint=checkpoint,
            records=(),
        ),
    )
    manager = IndexManager(candidate, AppSettings(enable_ocr=False))

    usn_fast, changed_paths, deleted_paths, target_checkpoint = (
        manager._try_usn_incremental_plan(root_id, root, True)
    )

    assert usn_fast is True
    assert changed_paths == set()
    assert deleted_paths == set()
    assert target_checkpoint == checkpoint


def test_expanding_compact_snapshot_uses_publication_time_not_epoch(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact.db"
    _create_candidate(compact, 7)
    con = sqlite3.connect(compact)
    try:
        root = tmp_path / "timestamp-root"
        root.mkdir()
        con.execute(
            "INSERT INTO roots(id, path, include_subfolders) VALUES (1, ?, 1)",
            (str(root),),
        )
        con.execute(
            "UPDATE snapshot_meta SET published_at = ? WHERE id = 1",
            (1_700_000_000,),
        )
        con.commit()
    finally:
        con.close()

    expanded = tmp_path / "expanded.db"
    expand_compact_snapshot(compact, expanded)
    expanded_db = DatabaseManager(expanded, read_only=True)
    row = expanded_db.list_roots()[0]

    assert str(row["created_at"]).startswith("2023-11-14T")
    assert str(row["updated_at"]).startswith("2023-11-14T")
    assert str(row["created_at"]) != "1970-01-01T00:00:00+00:00"


def test_isolated_compact_expansion_reports_stages_and_replaces_target(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact-isolated.db"
    _create_candidate(compact, 7)
    expanded = tmp_path / "expanded-isolated.db"
    progress: list[dict[str, object]] = []

    expand_compact_snapshot_isolated(
        compact,
        expanded,
        scan_id="scan-isolated-test",
        attempt_id="attempt-isolated-test",
        progress_callback=progress.append,
        timeout_seconds=30,
        stall_seconds=10,
    )

    assert expanded.is_file()
    assert any(
        row.get("phase") == "candidate_expand"
        and row.get("phase_status") == "started"
        for row in progress
    )
    assert any(
        row.get("phase") == "candidate_commit"
        and row.get("committed") is True
        for row in progress
    )
    assert any(
        row.get("phase") == "replace"
        and row.get("phase_status") == "completed"
        for row in progress
    )


def test_isolated_compact_expansion_reports_unexpected_worker_exit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    compact = tmp_path / "compact-worker-exit.db"
    _create_candidate(compact, 7)
    expanded = tmp_path / "expanded-worker-exit.db"
    progress: list[dict[str, object]] = []
    monkeypatch.setenv("LFTS_TEST_CANDIDATE_EXPAND_EXIT_PHASE", "schema")
    monkeypatch.setenv("LFTS_TEST_CANDIDATE_EXPAND_EXIT_CODE", "97")

    with pytest.raises(RuntimeError, match="意外退出"):
        expand_compact_snapshot_isolated(
            compact,
            expanded,
            scan_id="scan-worker-exit-test",
            attempt_id="attempt-worker-exit-test",
            progress_callback=progress.append,
            timeout_seconds=30,
            stall_seconds=10,
        )

    assert not expanded.exists()
    assert any(row.get("phase") == "worker_exit" for row in progress)


def test_isolated_compact_expansion_stops_when_parent_is_gone(
    tmp_path: Path,
) -> None:
    compact = tmp_path / "compact-parent-exit.db"
    _create_candidate(compact, 7)
    expanded = tmp_path / "expanded-parent-exit.db"
    progress: list[dict[str, object]] = []

    with pytest.raises(RuntimeError, match="父进程已退出|子进程失败"):
        expand_compact_snapshot_isolated(
            compact,
            expanded,
            scan_id="scan-parent-exit-test",
            attempt_id="attempt-parent-exit-test",
            progress_callback=progress.append,
            timeout_seconds=30,
            stall_seconds=10,
            parent_pid=2_147_483_647,
        )

    assert not expanded.exists()
    assert any(row.get("phase") == "worker_exit" for row in progress)


def test_scan_worker_publishes_compact_generation_after_full_round(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    control = tmp_path / "index_control.db"
    db = DatabaseManager(control)
    db.initialize()
    root = tmp_path / "scan-files"
    root.mkdir()
    source = root / "scan.txt"
    source.write_text("GENERATION_ISOLATION_HIT", encoding="utf-8")
    db.add_root(root)

    worker = ScanWorker(
        base,
        AppSettings(enable_ocr=False),
        control_db_path=control,
    )
    worker.run()

    store = IndexGenerationStore(base)
    active = store.active()
    assert active is not None
    assert active.path.name.endswith(".db")
    assert ".building" not in active.path.name
    with db.connect_read_only() as control_connection:
        control_root = control_connection.execute(
            "SELECT status, last_scan_at FROM roots WHERE id = ?",
            (1,),
        ).fetchone()
    assert control_root["status"] == "ready"
    assert control_root["last_scan_at"]
    result = SearchEngine(
        DatabaseManager(active.path),
        generation_isolated=True,
    ).search(SearchQuery(text="GENERATION_ISOLATION_HIT"))
    assert result.total_confirmed == 1

    source.write_text("GENERATION_ISOLATION_NEW_HIT", encoding="utf-8")
    second_worker = ScanWorker(
        base,
        AppSettings(enable_ocr=False),
        control_db_path=control,
    )
    second_worker.run()
    active2 = store.active()
    assert active2 is not None
    assert active2.generation == 2
    result2 = SearchEngine(
        DatabaseManager(active2.path),
        generation_isolated=True,
    ).search(SearchQuery(text="GENERATION_ISOLATION_NEW_HIT"))
    assert result2.total_confirmed == 1


def test_snapshot_search_reports_snapshot_active_without_writer_pause(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    candidate = store.begin_candidate()
    work = DatabaseManager(candidate.path)
    work.initialize()
    root = tmp_path / "status-files"
    root.mkdir()
    source = root / "status.txt"
    source.write_text("SNAPSHOT_STATUS_HIT", encoding="utf-8")
    root_id = work.add_root(root)
    file_id, _changed = work.upsert_file_metadata(root_id, source)
    work.replace_file_blocks(
        file_id,
        source.name,
        str(source),
        [
            ContentBlock(
                file_path=str(source),
                block_index=0,
                block_type="text",
                location_text="正文",
                raw_text="SNAPSHOT_STATUS_HIT",
                normalized_text="snapshot_status_hit",
            )
        ],
        parser_name="text",
    )
    published = store.publish(
        candidate,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-09-04T01:00:00+00:00",
    )

    statuses: list[str] = []
    page = SearchEngine(
        DatabaseManager(published.path),
        generation_isolated=True,
        snapshot_generation=published.generation,
    ).search(
        SearchQuery(text="SNAPSHOT_STATUS_HIT"),
        status_callback=statuses.append,
    )

    assert page.total_confirmed == 1
    assert statuses == ["snapshot_active"]


def test_generation_lease_registry_is_shared_between_worker_store_objects(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    first = IndexGenerationStore(base)
    second = IndexGenerationStore(base)
    lease = first.lease_registry.acquire(7)
    try:
        assert second.lease_registry.active(7) == 1
    finally:
        lease.release()
    assert second.lease_registry.active(7) == 0


def test_candidate_failure_report_survives_candidate_cleanup(tmp_path: Path) -> None:
    store = IndexGenerationStore(tmp_path / "search_index.db")

    store.write_failure_report(
        {
            "scan_id": "scan-test",
            "candidate_generation": 3,
            "blocking_files": [{"path": r"E:\blocked.txt"}],
        }
    )

    payload = json.loads(store.failure_report_path.read_text(encoding="utf-8"))
    assert payload["scan_id"] == "scan-test"
    assert payload["blocking_files"][0]["path"] == r"E:\blocked.txt"
    store.clear_failure_report()
    assert not store.failure_report_path.exists()


def test_report_only_blocker_can_be_excluded_without_control_file_row(
    tmp_path: Path,
) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    store.write_failure_report(
        {
            "candidate_generation": 3,
            "published_partial": True,
            "readiness": {
                "blocking_files": 1,
                "blocking_status_counts": {"skipped": 1},
                "blocking_file_samples": [],
            },
            "blocking_files": [
                {
                    "id": 2320,
                    "path": r"E:\blocked.zip",
                    "parse_status": "skipped",
                    "parse_error_code": "ZIP_FILE_COUNT_LIMIT",
                }
            ],
        }
    )
    store.manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "active_generation": 3,
                "active_file": "search_index.gen-000003.db",
                "partial": True,
            }
        ),
        encoding="utf-8",
    )

    result = store.mark_failure_report_items_excluded(
        [2320],
        reason="安全策略跳过，人工确认排除",
        operation_source="test",
    )

    assert result["changed"] == 1
    payload = json.loads(store.failure_report_path.read_text(encoding="utf-8"))
    assert payload["blocking_files"] == []
    assert payload["excluded_files"][0]["id"] == 2320
    assert payload["excluded_files"][0]["scope_state"] == "manual_excluded"
    assert json.loads(store.manifest_path.read_text(encoding="utf-8"))["partial"] is False


def test_report_only_exclusion_is_visible_in_excluded_scope() -> None:
    merged = merge_failure_report(
        {
            "failed_rows": [],
            "excluded_rows": [],
            "failed_counts": {"blocking": 0, "excluded": 0},
            "all_blocking_ids": [],
            "readiness": {
                "search_available": True,
                "ready": True,
                "blocking_files": 0,
            },
        },
        {
            "published_partial": True,
            "report_resolved_at": "2026-09-04T08:00:00+00:00",
            "excluded_files": [
                {
                    "id": 2320,
                    "path": r"E:\blocked.zip",
                    "parse_status": "skipped",
                    "parse_error_code": "ZIP_FILE_COUNT_LIMIT",
                    "exclusion_reason": "人工排除",
                }
            ],
            "blocking_files": [],
            "readiness": {"blocking_files": 0},
        },
    )

    assert merged["failed_counts"]["blocking"] == 0
    assert merged["failed_counts"]["excluded"] == 1
    assert merged["failed_rows"] == []
    assert merged["excluded_rows"][0]["file_id"] == 2320
    assert merged["readiness"]["ready"] is True


def test_terminal_policy_skip_does_not_block_partial_generation_publish() -> None:
    assert ScanWorker._can_publish_with_terminal_blockers(
        {
            "ready": False,
            "not_ready_reasons": ["blocking_files", "unready_root"],
            "blocking_files": 1,
            "blocking_file_samples": [
                {
                    "parse_status": "skipped",
                    "parse_error_code": "ZIP_FILE_COUNT_LIMIT",
                    "task_status": "complete",
                }
            ],
        }
    ) is True
    assert ScanWorker._can_publish_with_terminal_blockers(
        {
            "ready": False,
            "not_ready_reasons": ["blocking_files", "unfinished_tasks"],
            "blocking_files": 1,
            "blocking_file_samples": [
                {"parse_status": "skipped", "task_status": "complete"}
            ],
        }
    ) is False


def test_scan_worker_publishes_other_files_when_one_terminal_skip_blocks_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import local_full_text_search.workers.scan_worker as scan_worker_module

    base = tmp_path / "search_index.db"
    control = tmp_path / "index_control.db"
    control_db = DatabaseManager(control)
    control_db.initialize()
    root = tmp_path / "scope"
    root.mkdir()
    control_db.add_root(root)

    class FakeIndexManager:
        def __init__(self, database, _settings, **_kwargs) -> None:
            self.database = database

        def request_pause(self) -> None:
            return None

        def force_terminate_processes(self) -> None:
            return None

        def index_enabled_roots(self, _token, _progress, **_kwargs):
            root_row = self.database.list_roots(enabled_only=True)[0]
            root_id = int(root_row["id"])
            skipped_path = root / "large-package.zip"
            skipped_path.write_bytes(b"zip placeholder")
            file_id, _changed = self.database.upsert_file_metadata(
                root_id,
                skipped_path,
            )
            task_id = self.database.create_parse_task(
                file_id,
                "fake-run",
                "zip",
            )
            with self.database.connect() as con:
                con.execute(
                    """
                    UPDATE files
                    SET parse_status = 'skipped',
                        parse_error_code = 'ZIP_FILE_COUNT_LIMIT',
                        parse_error_message = 'too many files'
                    WHERE id = ?
                    """,
                    (file_id,),
                )
                con.execute(
                    "UPDATE parse_tasks SET status = 'complete' WHERE id = ?",
                    (task_id,),
                )
                con.execute(
                    "UPDATE roots SET status = 'incomplete' WHERE id = ?",
                    (root_id,),
                )
            return SimpleNamespace(
                cancelled=False,
                to_dict=lambda: {"indexed": 0, "skipped": 1},
            )

    monkeypatch.setattr(scan_worker_module, "IndexManager", FakeIndexManager)
    worker = ScanWorker(
        base,
        AppSettings(enable_ocr=False),
        control_db_path=control,
    )
    finished: list[dict[str, object]] = []
    failed: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert finished and finished[0]["partial_publish"] is True
    active = IndexGenerationStore(base).active()
    assert active is not None and active.partial is True
    report = json.loads(
        IndexGenerationStore(base).failure_report_path.read_text(encoding="utf-8")
    )
    assert report["published_partial"] is True
    assert report["blocking_files"][0]["parse_error_code"] == "ZIP_FILE_COUNT_LIMIT"


def test_new_scope_keeps_old_snapshot_searchable_and_is_synced_to_candidate(
    tmp_path: Path,
) -> None:
    base = tmp_path / "search_index.db"
    control = tmp_path / "index_control.db"
    control_db = DatabaseManager(control)
    control_db.initialize()
    first_root = tmp_path / "first-root"
    first_root.mkdir()
    first_file = first_root / "old.txt"
    first_file.write_text("OLD_SCOPE_SEARCH_HIT", encoding="utf-8")
    control_db.add_root(first_root)

    ScanWorker(
        base,
        AppSettings(enable_ocr=False),
        control_db_path=control,
    ).run()
    store = IndexGenerationStore(base)
    active_before = store.active()
    assert active_before is not None

    second_root = tmp_path / "second-root"
    second_root.mkdir()
    second_file = second_root / "new.txt"
    second_file.write_text("NEW_SCOPE_SEARCH_HIT", encoding="utf-8")
    control_db.add_root(second_root)

    # Creating the next candidate must not affect the old published reader.
    candidate = store.begin_candidate()
    active_page = SearchEngine(
        DatabaseManager(active_before.path),
        generation_isolated=True,
    ).search(SearchQuery(text="OLD_SCOPE_SEARCH_HIT"))
    assert active_page.total_confirmed == 1
    candidate_db = DatabaseManager(candidate.path)
    ScanWorker(
        base,
        AppSettings(enable_ocr=False),
        control_db_path=control,
    )._sync_control_roots(candidate_db)
    assert [str(row["path"]) for row in candidate_db.list_roots(enabled_only=True)] == [
        str(first_root),
        str(second_root),
    ]
    store.abort_candidate(candidate)


def test_search_keeps_previous_generation_while_candidate_is_written(tmp_path: Path) -> None:
    base = tmp_path / "search_index.db"
    store = IndexGenerationStore(base)
    first = store.begin_candidate()
    work = DatabaseManager(first.path)
    work.initialize()
    root = tmp_path / "snapshot-files"
    root.mkdir()
    source = root / "versioned.txt"
    source.write_text("OLD_SNAPSHOT_TEXT", encoding="utf-8")
    root_id = work.add_root(root)
    file_id, _changed = work.upsert_file_metadata(root_id, source)
    work.replace_file_blocks(
        file_id,
        source.name,
        str(source),
        [
            ContentBlock(
                file_path=str(source),
                block_index=0,
                block_type="text",
                location_text="正文",
                raw_text="OLD_SNAPSHOT_TEXT",
                normalized_text="old_snapshot_text",
            )
        ],
        parser_name="text",
    )
    store.publish(
        first,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-09-03T10:00:00+00:00",
    )

    second = store.begin_candidate()
    candidate_db = DatabaseManager(second.path)
    source.write_text("NEW_SNAPSHOT_TEXT", encoding="utf-8")
    candidate_db.upsert_file_metadata(root_id, source)
    candidate_db.replace_file_blocks(
        file_id,
        source.name,
        str(source),
        [
            ContentBlock(
                file_path=str(source),
                block_index=0,
                block_type="text",
                location_text="正文",
                raw_text="NEW_SNAPSHOT_TEXT",
                normalized_text="new_snapshot_text",
            )
        ],
        parser_name="text",
    )

    active_db = DatabaseManager(store.active().path)  # type: ignore[union-attr]
    old_page = SearchEngine(active_db, generation_isolated=True).search(
        SearchQuery(text="OLD_SNAPSHOT_TEXT")
    )
    new_before_publish = SearchEngine(active_db, generation_isolated=True).search(
        SearchQuery(text="NEW_SNAPSHOT_TEXT")
    )
    assert old_page.total_confirmed == 1
    assert new_before_publish.total_confirmed == 0

    store.publish(
        second,
        file_count=1,
        document_count=1,
        block_count=1,
        published_at="2026-09-03T10:01:00+00:00",
    )
    new_db = DatabaseManager(store.active().path)  # type: ignore[union-attr]
    new_page = SearchEngine(new_db, generation_isolated=True).search(
        SearchQuery(text="NEW_SNAPSHOT_TEXT")
    )
    assert new_page.total_confirmed == 1
