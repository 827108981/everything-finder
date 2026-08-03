from __future__ import annotations

from pathlib import Path

import pytest

from local_full_text_search.core.atomic_fts_publish import (
    IndexPublishError,
    IndexPublishGateError,
    IndexVersionPublisher,
)
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.models.content_block import ContentBlock


def _database_with_candidate_text(
    tmp_path: Path,
) -> tuple[DatabaseManager, int, int]:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "document.txt"
    source.write_text("atomic publish source", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)
    database.begin_deferred_fts()
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
                        raw_text="ATOMIC_CANDIDATE_TEXT",
                        normalized_text="atomic_candidate_text",
                    )
                ],
                "parser_name": "text",
                "parser_version": "test-v1",
                "status": "success",
                "content_key": "sha256:candidate",
                "task_id": None,
            }
        ],
        update_fts=False,
    )
    return database, root_id, file_id


def _state(database: DatabaseManager, key: str) -> str | None:
    with database.connect() as con:
        row = con.execute(
            "SELECT value FROM index_state WHERE key = ?",
            (key,),
        ).fetchone()
    return str(row["value"]) if row is not None else None


def test_fts_01r_publish_requires_all_hard_gates(tmp_path: Path) -> None:
    database, root_id, file_id = _database_with_candidate_text(tmp_path)
    task_id = database.create_parse_task(
        file_id,
        run_id="candidate-run",
        task_type="pdf_native_page",
        priority=100,
    )
    publisher = IndexVersionPublisher(database)
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="candidate-run",
        version_key="candidate-v1",
    )

    with pytest.raises(IndexPublishGateError) as exc:
        publisher.publish(
            candidate,
            writer_idle=False,
            workers_idle=False,
            golden_query_gate=lambda _con: False,
        )

    assert {
        "writer_not_idle",
        "workers_not_idle",
        "unfinished_tasks",
        "golden_query_failed",
    }.issubset(set(exc.value.failed_gates))
    assert _state(database, "content_fts_dirty") == "1"
    assert _state(database, "active_index_version") is None
    with database.connect() as con:
        status = con.execute(
            "SELECT status FROM index_versions WHERE id = ?",
            (candidate,),
        ).fetchone()["status"]
        task_status = con.execute(
            "SELECT status FROM parse_tasks WHERE id = ?",
            (task_id,),
        ).fetchone()["status"]
    assert status == "staging"
    assert task_status == "queued"


def test_fts_01r_successful_publish_atomically_activates_uniform_version(
    tmp_path: Path,
) -> None:
    database, root_id, _file_id = _database_with_candidate_text(tmp_path)
    publisher = IndexVersionPublisher(database)
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="candidate-run",
        version_key="candidate-v1",
    )

    result = publisher.publish(
        candidate,
        writer_idle=True,
        workers_idle=True,
        golden_query_gate=lambda con: int(
            con.execute(
                """
                SELECT COUNT(*) FROM content_blocks
                WHERE normalized_text LIKE '%atomic_candidate_text%'
                """
            ).fetchone()[0]
        )
        == 1,
    )

    assert result.version_id == candidate
    assert result.block_count == 1
    assert result.fts_row_count == result.block_count
    assert _state(database, "active_index_version") == str(candidate)
    assert _state(database, "content_fts_dirty") == "0"
    assert _state(database, "full_batch_incomplete") == "0"
    with database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM content_fts"
        ).fetchone()[0] == 1
        block_versions = {
            int(version_row["index_version_id"])
            for version_row in con.execute(
                """
                SELECT DISTINCT index_version_id
                FROM content_blocks
                """
            )
            if version_row["index_version_id"] is not None
        }
        row = con.execute(
            "SELECT status, block_count, content_digest FROM index_versions WHERE id = ?",
            (candidate,),
        ).fetchone()
    assert row["status"] == "active"
    assert block_versions == {candidate}
    assert int(row["block_count"]) == 1
    assert len(str(row["content_digest"])) == 64


def test_fts_01r_crash_before_switch_keeps_dirty_and_never_activates_candidate(
    tmp_path: Path,
) -> None:
    database, root_id, _file_id = _database_with_candidate_text(tmp_path)
    publisher = IndexVersionPublisher(database)
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="candidate-run",
        version_key="candidate-v1",
    )

    with pytest.raises(IndexPublishError):
        publisher.publish(
            candidate,
            writer_idle=True,
            workers_idle=True,
            golden_query_gate=lambda _con: True,
            failpoint="after_candidate_build",
        )

    assert _state(database, "active_index_version") is None
    assert _state(database, "content_fts_dirty") == "1"
    with database.connect() as con:
        row = con.execute(
            "SELECT status FROM index_versions WHERE id = ?",
            (candidate,),
        ).fetchone()
        candidate_table = con.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'content_fts_candidate'
            """
        ).fetchone()[0]
    assert row["status"] == "failed"
    assert candidate_table == 0


def test_fts_01r_new_resume_candidate_safely_discards_stale_unpublished(
    tmp_path: Path,
) -> None:
    database, root_id, _file_id = _database_with_candidate_text(
        tmp_path
    )
    publisher = IndexVersionPublisher(database)
    stale = publisher.begin_candidate(
        root_id=root_id,
        run_id="interrupted-run",
        version_key="candidate-interrupted",
    )

    resumed = publisher.begin_candidate(
        root_id=root_id,
        run_id="resumed-run",
        version_key="candidate-resumed",
    )

    with database.connect() as con:
        rows = {
            int(row["id"]): (
                str(row["status"]),
                str(row["error_message"] or ""),
            )
            for row in con.execute(
                """
                SELECT id, status, error_message
                FROM index_versions WHERE id IN (?, ?)
                """,
                (stale, resumed),
            )
        }
    assert rows[stale][0] == "discarded"
    assert "resumed-run" in rows[stale][1]
    assert rows[resumed][0] == "staging"
    assert _state(database, "content_fts_dirty") == "1"


def test_fts_01r_failed_update_keeps_old_complete_fts_and_disables_search(
    tmp_path: Path,
) -> None:
    database, root_id, file_id = _database_with_candidate_text(
        tmp_path
    )
    publisher = IndexVersionPublisher(database)
    active = publisher.begin_candidate(
        root_id=root_id,
        run_id="active-run",
        version_key="active-v1",
    )
    publisher.publish(
        active,
        writer_idle=True,
        workers_idle=True,
        golden_query_gate=lambda _con: True,
    )
    with database.connect() as con:
        old_fts = [
            tuple(row)
            for row in con.execute(
                """
                SELECT rowid, block_id, file_id, normalized_text
                FROM content_fts ORDER BY rowid
                """
            )
        ]

    database.begin_deferred_fts()
    database.replace_document_blocks_many(
        [
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": "document.txt",
                "path": str(tmp_path / "root" / "document.txt"),
                "blocks": [
                    ContentBlock(
                        file_path=str(
                            tmp_path / "root" / "document.txt"
                        ),
                        block_index=0,
                        block_type="paragraph",
                        location_text="正文",
                        raw_text="UPDATED_CANDIDATE_TEXT",
                        normalized_text="updated_candidate_text",
                    )
                ],
                "parser_name": "text",
                "parser_version": "test-v2",
                "status": "success",
                "content_key": "sha256:updated",
                "task_id": None,
            }
        ],
        update_fts=False,
    )
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="update-run",
        version_key="candidate-v2",
    )

    with pytest.raises(IndexPublishError):
        publisher.publish(
            candidate,
            writer_idle=True,
            workers_idle=True,
            golden_query_gate=lambda _con: True,
            failpoint="before_activation",
        )

    with database.connect() as con:
        after_fts = [
            tuple(row)
            for row in con.execute(
                """
                SELECT rowid, block_id, file_id, normalized_text
                FROM content_fts ORDER BY rowid
                """
            )
        ]
        versions = {
            int(row["id"]): str(row["status"])
            for row in con.execute(
                """
                SELECT id, status FROM index_versions
                WHERE id IN (?, ?)
                """,
                (active, candidate),
            )
        }
    assert after_fts == old_fts
    assert versions[active] == "active"
    assert versions[candidate] == "failed"
    assert database.index_readiness()["ready"] is False
    assert _state(database, "content_fts_dirty") == "1"


def test_fts_02r_publish_records_exclusion_candidate_and_published_version(
    tmp_path: Path,
) -> None:
    database, root_id, file_id = _database_with_candidate_text(tmp_path)
    database.record_failure(
        file_id,
        "PARSER_ERROR",
        "candidate version cannot parse",
    )
    database.exclude_files_from_index(
        [file_id],
        reason="field exclusion",
        operation_source="test",
    )
    publisher = IndexVersionPublisher(database)
    candidate = publisher.begin_candidate(
        root_id=root_id,
        run_id="excluded-run",
        version_key="excluded-v1",
    )

    published = publisher.publish(
        candidate,
        writer_idle=True,
        workers_idle=True,
        golden_query_gate=lambda _connection: True,
    )

    exclusion = database.excluded_files()[0]
    assert published.block_count == 0
    assert exclusion["candidate_index_version_id"] == candidate
    assert exclusion["published_index_version_id"] == candidate
