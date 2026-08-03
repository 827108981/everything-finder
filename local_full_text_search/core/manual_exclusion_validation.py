from __future__ import annotations

from pathlib import Path

from local_full_text_search.core.atomic_fts_publish import (
    IndexVersionPublisher,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery


def _store_text(
    database: DatabaseManager,
    file_id: int,
    source: Path,
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
                        location_text="body",
                        raw_text=token,
                        normalized_text=token.lower(),
                    )
                ],
                "parser_name": "text",
                "parser_version": "manual-exclusion-validation-v1",
                "status": "success",
                "content_key": f"validation:{file_id}:{token}",
                "task_id": None,
            }
        ]
    )


def _search_count(
    database: DatabaseManager,
    text: str,
    *,
    filename: bool,
    content: bool,
) -> int:
    result = SearchEngine(database).search(
        SearchQuery(
            text=text,
            mode="exact",
            search_filename=filename,
            search_path=filename,
            search_content=content,
        )
    )
    return int(result.total_confirmed)


def run_manual_exclusion_validation(base: Path) -> dict[str, object]:
    base.mkdir(parents=True, exist_ok=True)
    root = base / "scope"
    root.mkdir()
    success_source = root / "included.txt"
    failed_source = root / "EXCLUDED_NAME_TOKEN.txt"
    metadata_source = root / "metadata.zip"
    success_source.write_text("INCLUDED_TOKEN", encoding="utf-8")
    failed_source.write_text("failed-v1", encoding="utf-8")
    metadata_source.write_bytes(b"PK validation placeholder")

    database = DatabaseManager(base / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    success_id, _ = database.upsert_file_metadata(root_id, success_source)
    failed_id, _ = database.upsert_file_metadata(root_id, failed_source)
    metadata_id, _ = database.upsert_file_metadata(root_id, metadata_source)
    _store_text(database, success_id, success_source, "INCLUDED_TOKEN")
    _store_text(database, failed_id, failed_source, "EXCLUDED_OLD_TOKEN")
    database.record_failure(
        failed_id,
        "PARSER_ERROR",
        "injected field failure",
        parser_name="text",
    )
    database.set_file_error_status(
        metadata_id,
        "metadata_only",
        "ZIP_NO_SUPPORTED_MEMBER",
        "no supported member",
        parser_name="zip",
    )
    database.update_root_scan_time(root_id, "ready")

    before = database.index_readiness()
    database.exclude_files_from_index(
        [failed_id],
        reason="validation exclusion",
        operation_source="validation_tool",
    )
    excluded_readiness = database.index_readiness()
    publisher = IndexVersionPublisher(database)
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="manual-exclusion-validation",
        version_key="manual-exclusion-validation-v1",
    )
    published = publisher.publish(
        candidate,
        writer_idle=True,
        workers_idle=True,
        golden_query_gate=lambda connection: int(
            connection.execute(
                """
                SELECT COUNT(*) FROM content_blocks
                WHERE normalized_text = 'included_token'
                """
            ).fetchone()[0]
        )
        == 1,
    )
    after_publish = database.index_readiness()
    with database.connect() as connection:
        preserved_status = str(
            connection.execute(
                "SELECT parse_status FROM files WHERE id = ?",
                (failed_id,),
            ).fetchone()[0]
        )
    hidden = all(
        count == 0
        for count in (
            _search_count(
                database,
                "EXCLUDED_NAME_TOKEN",
                filename=True,
                content=False,
            ),
            _search_count(
                database,
                "EXCLUDED_OLD_TOKEN",
                filename=False,
                content=True,
            ),
        )
    )

    database.restore_files_to_index(
        [failed_id],
        reason="validation restore",
        operation_source="validation_tool",
    )
    restored_readiness = database.index_readiness()
    database.exclude_files_from_index(
        [failed_id],
        reason="validation exclusion v2",
        operation_source="validation_tool",
    )
    failed_source.write_text("failed-v2-source-changed", encoding="utf-8")
    database.upsert_file_metadata(root_id, failed_source)
    changed_readiness = database.index_readiness()
    history = database.excluded_files(include_history=True)
    force_checks = _validate_force_complete(base / "force-complete")

    checks = {
        "metadata_only_completion": (
            int(before["metadata_only_complete_files"]) == 1
            and int(before["blocking_files"]) == 1
        ),
        "excluded_publish": (
            published.version_id == candidate
            and bool(after_publish["ready"])
            and int(excluded_readiness["manual_excluded_files"]) == 1
        ),
        "search_hidden": hidden,
        "failure_preserved": preserved_status == "failed",
        "restore_reblocks": (
            int(restored_readiness["manual_excluded_files"]) == 0
            and int(restored_readiness["blocking_files"]) == 1
        ),
        "source_change_invalidates": (
            int(changed_readiness["manual_excluded_files"]) == 0
            and int(changed_readiness["blocking_files"]) == 1
        ),
        "audit_history_preserved": (
            len(history) == 2
            and any(row["revoked_at"] for row in history)
            and any(row["invalidated_at"] for row in history)
        ),
        "force_complete_transaction": bool(force_checks["passed"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "published_version_id": published.version_id,
        "excluded_readiness": dict(excluded_readiness),
        "restored_readiness": dict(restored_readiness),
        "changed_readiness": dict(changed_readiness),
        "audit_rows": len(history),
        "force_complete": force_checks,
    }


def _validate_force_complete(base: Path) -> dict[str, object]:
    base.mkdir(parents=True, exist_ok=True)
    root = base / "scope"
    root.mkdir()
    included = root / "included.txt"
    blocked = root / "FORCE_BLOCKED_NAME.pdf"
    included.write_text("FORCE_INCLUDED_TOKEN", encoding="utf-8")
    blocked.write_bytes(b"%PDF-1.4\nvalidation blocked")
    database = DatabaseManager(base / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    included_id, _ = database.upsert_file_metadata(root_id, included)
    blocked_id, _ = database.upsert_file_metadata(root_id, blocked)
    _store_text(database, included_id, included, "FORCE_INCLUDED_TOKEN")
    task_id = database.create_parse_tasks(
        [(blocked_id, "force-validation-run", "pdf", 100)]
    )[0]
    database.mark_task_running(task_id)
    database.update_root_scan_time(root_id, "incomplete")
    database.begin_deferred_fts()

    result = database.force_complete_current_scope(
        reason="validation force completion",
        operation_source="validation_tool",
    )
    readiness = database.index_readiness()
    with database.connect() as connection:
        file_row = connection.execute(
            "SELECT parse_status, parse_error_code FROM files WHERE id = ?",
            (blocked_id,),
        ).fetchone()
        task_row = connection.execute(
            "SELECT status, error_code FROM parse_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()
        integrity = [
            str(row[0]) for row in connection.execute("PRAGMA integrity_check")
        ]
        foreign_keys = list(connection.execute("PRAGMA foreign_key_check"))
    visible = _search_count(
        database,
        "FORCE_INCLUDED_TOKEN",
        filename=False,
        content=True,
    )
    hidden = _search_count(
        database,
        "FORCE_BLOCKED_NAME",
        filename=True,
        content=False,
    )
    passed = bool(
        result["excluded_files"] == 1
        and result["cancelled_tasks"] == 1
        and bool(readiness["ready"])
        and int(readiness["manual_excluded_files"]) == 1
        and str(file_row["parse_status"]) == "failed"
        and str(file_row["parse_error_code"]) == "FORCE_COMPLETED_EXCLUDED"
        and str(task_row["status"]) == "cancelled"
        and str(task_row["error_code"]) == "FORCE_COMPLETED_EXCLUDED"
        and visible == 1
        and hidden == 0
        and integrity == ["ok"]
        and not foreign_keys
    )
    return {
        "passed": passed,
        "result": result,
        "readiness": dict(readiness),
        "included_search_hits": visible,
        "excluded_search_hits": hidden,
        "integrity": integrity,
        "foreign_key_errors": len(foreign_keys),
    }
