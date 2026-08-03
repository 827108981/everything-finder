from __future__ import annotations

import importlib
import csv
import sqlite3
from pathlib import Path

import pytest

from app import seed_failure_fallback_demo
from local_full_text_search.core.atomic_fts_publish import IndexVersionPublisher
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.models.search_query import SearchQuery


def _database_with_file(
    tmp_path: Path,
    *,
    filename: str = "document.txt",
    text: str = "source text",
) -> tuple[DatabaseManager, int, int, Path]:
    root = tmp_path / "root"
    root.mkdir()
    source = root / filename
    source.write_text(text, encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)
    database.update_root_scan_time(root_id, "ready")
    return database, root_id, file_id, source


def _store_success(
    database: DatabaseManager,
    file_id: int,
    source: Path,
    *,
    token: str,
) -> None:
    database.replace_document_blocks_many(
        [
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": source.name,
                "path": str(source),
                "blocks": [
                    ContentBlock(
                        file_path=str(source),
                        block_index=0,
                        block_type="paragraph",
                        location_text="正文",
                        raw_text=token,
                        normalized_text=token.lower(),
                    )
                ],
                "parser_name": "text",
                "parser_version": "fts-02r-test-v1",
                "status": "success",
                "content_key": f"fts-02r:{file_id}:{token}",
                "task_id": None,
            }
        ]
    )


def test_fts_02r_schema_v8_has_scope_exclusion_history(tmp_path: Path) -> None:
    database, _root_id, _file_id, _source = _database_with_file(tmp_path)

    with database.connect() as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert version == 8
    assert "index_scope_exclusions" in tables


def test_fts_02r_clean_metadata_only_is_complete_and_not_a_failure(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, _source = _database_with_file(
        tmp_path,
        filename="software-package.zip",
    )
    database.set_file_error_status(
        file_id,
        "metadata_only",
        "ZIP_NO_SUPPORTED_MEMBER",
        "压缩包内没有可解析文件",
        parser_name="zip",
    )

    readiness = database.index_readiness()

    assert readiness["eligible_files"] == 1
    assert readiness["metadata_only_complete_files"] == 1
    assert readiness["complete_files"] == 1
    assert readiness["blocking_files"] == 0
    assert readiness["ready"] is True
    assert database.failed_files() == []


def test_fts_02r_manual_exclusion_preserves_failure_and_unblocks_readiness(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, _source = _database_with_file(tmp_path)
    database.record_failure(
        file_id,
        "PARSER_ERROR",
        "injected parser failure",
        parser_name="text",
    )
    assert database.index_readiness()["ready"] is False

    changed = database.exclude_files_from_index(
        [file_id],
        reason="现场文件确认无法解析",
        operation_source="test",
    )
    readiness = database.index_readiness()

    assert changed == 1
    assert readiness["manual_excluded_files"] == 1
    assert readiness["eligible_files"] == 0
    assert readiness["blocking_files"] == 0
    assert readiness["ready"] is True
    with database.connect() as connection:
        status = connection.execute(
            "SELECT parse_status FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()[0]
    assert status == "failed"
    assert database.failed_files() == []
    excluded = database.excluded_files()
    assert len(excluded) == 1
    assert excluded[0]["scope_state"] == "manual_excluded"
    assert excluded[0]["reason"] == "现场文件确认无法解析"


def test_manual_exclusion_of_last_blocker_publishes_incomplete_first_index(
    tmp_path: Path,
) -> None:
    database, root_id, failed_id, failed_source = _database_with_file(
        tmp_path,
        filename="blocked.pdf",
    )
    searchable_source = failed_source.parent / "searchable.txt"
    searchable_source.write_text("searchable source", encoding="utf-8")
    searchable_id, _changed = database.upsert_file_metadata(
        root_id,
        searchable_source,
    )
    _store_success(
        database,
        searchable_id,
        searchable_source,
        token="MANUAL_EXCLUSION_PUBLISH_TOKEN",
    )
    database.record_failure(
        failed_id,
        "PARSER_ERROR",
        "injected final blocker",
        parser_name="pdf",
    )
    database.update_root_scan_time(root_id, "incomplete")
    database.begin_deferred_fts()
    database.start_index_run(
        IndexRunMetrics("manual-exclusion-open-run", mode="full")
    )
    with database.connect() as connection:
        connection.execute("DELETE FROM content_fts")

    before = database.index_readiness()
    assert before["blocking_files"] == 1
    assert before["ready"] is False

    changed = database.exclude_files_from_index(
        [failed_id],
        reason="final blocker cannot be parsed",
        operation_source="test_first_index_publish",
    )

    after = database.index_readiness()
    assert changed == 1
    assert after["blocking_files"] == 0
    assert after["manual_excluded_files"] == 1
    assert after["active_runs"] == 0
    assert after["unready_roots"] == 0
    assert after["ready"] is True
    assert database.has_incomplete_full_batch() is False
    assert SearchEngine(database).search(
        SearchQuery(
            text="MANUAL_EXCLUSION_PUBLISH_TOKEN",
            mode="exact",
            search_filename=False,
            search_path=False,
            search_content=True,
        )
    ).total_confirmed == 1


def test_failure_fallback_demo_seeds_multiple_realistic_blockers(
    tmp_path: Path,
) -> None:
    database = DatabaseManager(tmp_path / "demo.db")
    database.initialize()

    seeded = seed_failure_fallback_demo(database, tmp_path / "demo-files")
    before = database.index_readiness()
    blockers = database.failed_files(limit=20)

    assert int(before["blocking_files"]) == 4
    assert int(before["metadata_only_complete_files"]) == 1
    assert {str(row["parse_error_code"]) for row in blockers} == {
        "PDF_CORRUPTED",
        "PASSWORD_PROTECTED",
        "OCR_FAILED",
        "ZIP_FILE_COUNT_LIMIT",
    }

    result = database.force_complete_current_scope(
        reason="test the hands-on demo fallback",
        operation_source="test_failure_demo",
    )

    assert result["excluded_files"] == 4
    assert database.index_readiness()["ready"] is True
    assert len(database.excluded_files(limit=20)) == 4
    assert SearchEngine(database).search(
        SearchQuery(text=str(seeded["search_token"]), mode="exact")
    ).total_confirmed == 1


def test_fts_02r_excluded_failure_cannot_match_filename_path_or_old_content(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, source = _database_with_file(
        tmp_path,
        filename="EXCLUDED_FILENAME_TOKEN.txt",
    )
    _store_success(database, file_id, source, token="EXCLUDED_OLD_CONTENT_TOKEN")
    database.record_failure(
        file_id,
        "PARSER_ERROR",
        "new version failed",
        parser_name="text",
    )
    database.exclude_files_from_index(
        [file_id],
        reason="exclude stale content",
        operation_source="test",
    )

    engine = SearchEngine(database)
    filename_hits = engine.search(
        SearchQuery(
            text="EXCLUDED_FILENAME_TOKEN",
            mode="exact",
            search_filename=True,
            search_path=True,
            search_content=False,
        )
    )
    content_hits = engine.search(
        SearchQuery(
            text="EXCLUDED_OLD_CONTENT_TOKEN",
            mode="exact",
            search_filename=False,
            search_path=False,
            search_content=True,
        )
    )

    assert filename_hits.total_confirmed == 0
    assert content_hits.total_confirmed == 0


def test_force_complete_converges_stale_pending_tasks_and_opens_search(
    tmp_path: Path,
) -> None:
    database, root_id, successful_id, successful_source = _database_with_file(
        tmp_path,
        filename="searchable.txt",
    )
    _store_success(
        database,
        successful_id,
        successful_source,
        token="FORCE_COMPLETE_VISIBLE_TOKEN",
    )
    blocked_source = successful_source.parent / "BLOCKED_FILENAME_TOKEN.pdf"
    blocked_source.write_bytes(b"%PDF-1.4\nblocked")
    blocked_id, _changed = database.upsert_file_metadata(root_id, blocked_source)
    task_id = database.create_parse_tasks(
        [(blocked_id, "stale-run", "pdf", 100)]
    )[0]
    database.mark_task_running(task_id)
    database.update_root_scan_time(root_id, "incomplete")
    database.begin_deferred_fts()

    result = database.force_complete_current_scope(
        reason="用户确认跳过无法完成的文件并开放搜索",
        operation_source="test_force_complete",
    )

    assert result["excluded_files"] == 1
    assert result["cancelled_tasks"] == 1
    assert result["ready_roots"] == 1
    readiness = database.index_readiness()
    assert readiness["ready"] is True
    assert readiness["manual_excluded_files"] == 1
    assert readiness["blocking_files"] == 0
    with database.connect() as connection:
        file_row = connection.execute(
            """
            SELECT parse_status, parse_error_code
            FROM files WHERE id = ?
            """,
            (blocked_id,),
        ).fetchone()
        task_row = connection.execute(
            "SELECT status, error_code FROM parse_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
    assert file_row["parse_status"] == "failed"
    assert file_row["parse_error_code"] == "FORCE_COMPLETED_EXCLUDED"
    assert task_row["status"] == "cancelled"
    assert task_row["error_code"] == "FORCE_COMPLETED_EXCLUDED"
    assert SearchEngine(database).search(
        SearchQuery(text="FORCE_COMPLETE_VISIBLE_TOKEN", mode="exact")
    ).total_confirmed == 1


def test_zero_blocker_force_complete_repairs_residual_state(
    tmp_path: Path,
) -> None:
    database, root_id, file_id, source = _database_with_file(
        tmp_path,
        filename="ready-content.txt",
    )
    _store_success(
        database,
        file_id,
        source,
        token="ZERO_BLOCKER_REPAIR_VISIBLE_TOKEN",
    )
    stale_task = database.create_parse_task(
        file_id,
        run_id="stale-success-run",
        task_type="text",
        priority=100,
    )
    database.begin_deferred_fts()
    database.update_root_scan_time(root_id, "incomplete")
    candidate = IndexVersionPublisher(database).begin_candidate(
        root_id=root_id,
        run_id="stale-success-run",
        version_key="zero-blocker-stale-candidate",
    )

    before = database.index_readiness()
    assert before["blocking_files"] == 0
    assert before["unfinished_tasks"] == 1
    assert before["content_fts_dirty"] is True
    assert before["full_batch_incomplete"] is True
    assert before["unpublished_candidates"] == 1
    assert before["repairable"] is True
    assert before["ready"] is False

    result = database.force_complete_current_scope(
        reason="repair zero-blocker residual state",
        operation_source="test_zero_blocker_repair",
    )

    after = database.index_readiness()
    assert result["excluded_files"] == 0
    assert result["invalidated_tasks"] == 1
    assert result["discarded_candidates"] == 1
    assert after["unfinished_tasks"] == 0
    assert after["unpublished_candidates"] == 0
    assert after["not_ready_reasons"] == []
    assert after["ready"] is True
    with database.connect() as connection:
        assert connection.execute(
            "SELECT status FROM parse_tasks WHERE id = ?",
            (stale_task,),
        ).fetchone()["status"] == "invalidated"
        assert connection.execute(
            "SELECT status FROM index_versions WHERE id = ?",
            (candidate,),
        ).fetchone()["status"] == "failed"
    assert SearchEngine(database).search(
        SearchQuery(text="ZERO_BLOCKER_REPAIR_VISIBLE_TOKEN", mode="exact")
    ).total_confirmed == 1


def test_last_exclusion_auto_converges_successful_file_stale_task(
    tmp_path: Path,
) -> None:
    database, root_id, successful_id, successful_source = _database_with_file(
        tmp_path,
        filename="included.txt",
    )
    _store_success(
        database,
        successful_id,
        successful_source,
        token="AUTO_CONVERGE_VISIBLE_TOKEN",
    )
    stale_task = database.create_parse_task(
        successful_id,
        run_id="stale-auto-converge",
        task_type="text",
        priority=100,
    )
    blocked_source = successful_source.parent / "blocked.pdf"
    blocked_source.write_bytes(b"%PDF-1.4\nbroken")
    blocked_id, _changed = database.upsert_file_metadata(root_id, blocked_source)
    database.record_failure(
        blocked_id,
        "PARSER_ERROR",
        "final blocker",
        parser_name="pdf",
    )
    database.begin_deferred_fts()
    database.update_root_scan_time(root_id, "incomplete")

    database.exclude_files_from_index(
        [blocked_id],
        reason="exclude final blocker and converge",
        operation_source="test_auto_converge",
    )

    readiness = database.index_readiness()
    assert readiness["blocking_files"] == 0
    assert readiness["unfinished_tasks"] == 0
    assert readiness["ready"] is True
    with database.connect() as connection:
        assert connection.execute(
            "SELECT status FROM parse_tasks WHERE id = ?",
            (stale_task,),
        ).fetchone()["status"] == "invalidated"
    assert SearchEngine(database).search(
        SearchQuery(
            text="BLOCKED_FILENAME_TOKEN",
            mode="exact",
            search_filename=True,
            search_path=True,
            search_content=False,
        )
    ).total_confirmed == 0


def test_pending_blocker_is_visible_before_force_completion(
    tmp_path: Path,
) -> None:
    database, _root_id, _file_id, source = _database_with_file(tmp_path)
    rows = database.failed_files()

    assert len(rows) == 1
    assert rows[0]["path"] == str(source)
    assert rows[0]["parse_status"] == "pending"


def test_fts_02r_source_change_invalidates_exact_version_exclusion(
    tmp_path: Path,
) -> None:
    database, root_id, file_id, source = _database_with_file(tmp_path)
    database.record_failure(file_id, "PARSER_ERROR", "old failure")
    database.exclude_files_from_index(
        [file_id],
        reason="old version only",
        operation_source="test",
    )

    source.write_text("changed source with a different size", encoding="utf-8")
    returned_id, changed = database.upsert_file_metadata(root_id, source)

    assert returned_id == file_id
    assert changed is True
    assert database.excluded_files() == []
    assert database.index_readiness()["manual_excluded_files"] == 0
    assert database.index_readiness()["blocking_files"] == 1
    history = database.excluded_files(include_history=True)
    assert len(history) == 1
    assert history[0]["invalidated_at"]
    assert history[0]["invalidation_reason"] == "source_identity_changed"


def test_fts_02r_restore_reincludes_file_and_keeps_audit_history(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, _source = _database_with_file(tmp_path)
    database.record_failure(file_id, "PARSER_ERROR", "failure")
    database.exclude_files_from_index(
        [file_id],
        reason="temporary exclusion",
        operation_source="test",
    )

    restored = database.restore_files_to_index(
        [file_id],
        reason="source repaired",
        operation_source="test",
    )

    assert restored == 1
    assert database.excluded_files() == []
    assert database.index_readiness()["blocking_files"] == 1
    history = database.excluded_files(include_history=True)
    assert len(history) == 1
    assert history[0]["revoked_at"]
    assert history[0]["revocation_reason"] == "source repaired"


def test_fts_02r_source_validator_exercises_atomic_publish_and_audit(
    tmp_path: Path,
) -> None:
    module = importlib.import_module(
        "local_full_text_search.core.manual_exclusion_validation"
    )

    result = module.run_manual_exclusion_validation(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["metadata_only_completion"] is True
    assert result["checks"]["excluded_publish"] is True
    assert result["checks"]["search_hidden"] is True
    assert result["checks"]["failure_preserved"] is True
    assert result["checks"]["restore_reblocks"] is True
    assert result["checks"]["source_change_invalidates"] is True
    assert result["checks"]["audit_history_preserved"] is True
    assert result["checks"]["force_complete_transaction"] is True
    assert result["force_complete"]["passed"] is True


class _FailingExclusionDatabase(DatabaseManager):
    def _after_scope_exclusion_audit(
        self,
        connection: sqlite3.Connection,
        file_ids: list[int],
    ) -> None:
        raise RuntimeError("injected exclusion interruption")


def test_fts_02r_exclusion_interruption_rolls_back_audit_and_fts(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, source = _database_with_file(tmp_path)
    _store_success(database, file_id, source, token="ROLLBACK_VISIBLE_TOKEN")
    database.record_failure(file_id, "PARSER_ERROR", "failure")
    failing = _FailingExclusionDatabase(database.db_path)

    with pytest.raises(RuntimeError, match="injected exclusion interruption"):
        failing.exclude_files_from_index(
            [file_id],
            reason="must roll back",
            operation_source="test",
        )

    assert database.excluded_files(include_history=True) == []
    with database.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM files_fts WHERE file_id = ?",
            (file_id,),
        ).fetchone()[0] == 1


def test_fts_02r_exclusion_rejects_file_with_active_parse_task(
    tmp_path: Path,
) -> None:
    database, _root_id, file_id, _source = _database_with_file(tmp_path)
    database.record_failure(file_id, "PARSER_ERROR", "failure")
    database.create_parse_task(
        file_id,
        run_id="active-run",
        task_type="parse_file",
        priority=1,
    )

    with pytest.raises(RuntimeError, match="活动解析任务"):
        database.exclude_files_from_index(
            [file_id],
            reason="must wait",
            operation_source="test",
        )


def test_fts_02r_failed_manifest_replay_matches_scope_semantics(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "failures.csv"
    with manifest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "路径",
                "扩展名",
                "状态",
                "错误码",
                "原因",
                "解析器",
                "时间",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "路径": r"D:\field\metadata.zip",
                    "扩展名": ".zip",
                    "状态": "metadata_only",
                    "错误码": "ZIP_NO_SUPPORTED_MEMBER",
                    "原因": "no member",
                    "解析器": "zip",
                    "时间": "2026-07-30T00:00:00Z",
                },
                {
                    "路径": r"D:\field\broken.zip",
                    "扩展名": ".zip",
                    "状态": "failed",
                    "错误码": "ZIP_CORRUPTED",
                    "原因": "bad zip",
                    "解析器": "zip",
                    "时间": "2026-07-30T00:00:01Z",
                },
                {
                    "路径": r"D:\field\large.zip",
                    "扩展名": ".zip",
                    "状态": "skipped",
                    "错误码": "ZIP_SIZE_LIMIT",
                    "原因": "too large",
                    "解析器": "zip",
                    "时间": "2026-07-30T00:00:02Z",
                },
            ]
        )
    module = importlib.import_module(
        "local_full_text_search.core.failed_manifest_validation"
    )

    result = module.replay_failed_manifest(
        manifest,
        tmp_path / "replay",
    )

    assert result["passed"] is True
    assert result["input_rows"] == 3
    assert result["nonblocking_metadata_only"] == 1
    assert result["blocking_before_exclusion"] == 2
    assert result["manual_excluded_after"] == 2
    assert result["blocking_after_exclusion"] == 0
    assert result["parse_statuses_preserved"] is True
