from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import (
    SCHEMA_VERSION,
    DatabaseManager,
)
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.schema_validation import (
    run_schema_v7_validation,
)
import local_full_text_search.core.schema_validation as schema_validation
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


V7_TABLES = (
    "pdf_page_identities",
    "ocr_requests",
    "ocr_exact_cache",
    "index_versions",
    "resource_events",
    "backend_benchmarks",
)
V6_COLUMNS = (
    "parent_task_id",
    "unit_key",
    "payload_json",
    "progress_phase",
    "progress_completed",
    "progress_total",
    "progress_unit_type",
    "progress_cursor",
    "progress_bytes_read",
    "progress_output_blocks",
    "checkpoint_version",
    "last_semantic_progress_at",
    "worker_pid",
    "stall_signature",
    "checkpoint_path",
)
V7_COLUMNS = (
    "lease_owner",
    "lease_expires_at",
    "confirmed_at",
    "source_digest",
    "task_version",
    "result_digest",
    "index_version_id",
)


def _populated_database(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "files"
    root.mkdir()
    source = root / "preserved.txt"
    source.write_text("SCHEMA_V7_PRESERVES_GOLDEN_CONTENT", encoding="utf-8")
    database_path = tmp_path / "index.db"
    database = DatabaseManager(database_path)
    database.initialize()
    root_id = database.add_root(root)
    IndexManager(
        database,
        AppSettings(enable_ocr=False),
    ).index_root(root_id)
    database.add_search_history("SCHEMA_V7_HISTORY")
    return database_path, source


def _downgrade_layout(database_path: Path, version: int) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE content_blocks SET index_version_id = NULL"
        )
        for table in V7_TABLES:
            connection.execute(f"DROP TABLE IF EXISTS {table}")
        excluded = set(V7_COLUMNS)
        if version == 5:
            excluded.update(V6_COLUMNS)
            connection.execute("DROP TABLE IF EXISTS parse_task_attempts")
        column_rows = [
            row
            for row in connection.execute(
                "PRAGMA table_info(parse_tasks)"
            )
            if str(row[1]) not in excluded
        ]
        columns = [str(row[1]) for row in column_rows]
        selected = ", ".join(f'"{column}"' for column in columns)
        connection.execute(
            "CREATE TABLE parse_tasks_legacy ("
            + ", ".join(
                (
                    f'"{row[1]}" INTEGER PRIMARY KEY'
                    if int(row[5])
                    else (
                        f'"{row[1]}" {row[2] or "TEXT"}'
                        + (" NOT NULL" if int(row[3]) else "")
                        + (
                            f" DEFAULT {row[4]}"
                            if row[4] is not None
                            else ""
                        )
                    )
                )
                for row in column_rows
            )
            + ")"
        )
        connection.execute(
            f"INSERT INTO parse_tasks_legacy({selected}) "
            f"SELECT {selected} FROM parse_tasks"
        )
        connection.execute("DROP TABLE parse_tasks")
        connection.execute(
            "ALTER TABLE parse_tasks_legacy RENAME TO parse_tasks"
        )
        connection.execute(f"PRAGMA user_version = {int(version)}")
        connection.commit()
    finally:
        connection.close()


@pytest.mark.parametrize("version", [5, 6])
def test_schema_v5_and_v6_migrate_to_latest_without_content_loss(
    tmp_path: Path,
    version: int,
) -> None:
    database_path, _source = _populated_database(tmp_path)
    _downgrade_layout(database_path, version)

    migrated = DatabaseManager(database_path)
    migrated.initialize()
    report = migrated.integrity_report()

    with migrated.connect() as connection:
        current_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        root_count = int(
            connection.execute("SELECT COUNT(*) FROM roots").fetchone()[0]
        )
        file_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM files WHERE is_deleted = 0"
            ).fetchone()[0]
        )
        block_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_blocks"
            ).fetchone()[0]
        )
    hits = SearchEngine(migrated).search(
        SearchQuery(
            text="SCHEMA_V7_PRESERVES_GOLDEN_CONTENT",
            mode="exact",
        )
    ).total_confirmed

    assert current_version == SCHEMA_VERSION
    assert root_count == file_count == block_count == hits == 1
    assert migrated.search_history() == ["SCHEMA_V7_HISTORY"]
    assert report["integrity"] == ["ok"]
    assert report["foreign_key_errors"] == []
    assert (
        tmp_path / f"index.schema-v{version}.backup.db"
    ).is_file()


def test_latest_schema_repeated_initialization_is_idempotent(
    tmp_path: Path,
) -> None:
    database_path, _source = _populated_database(tmp_path)
    database = DatabaseManager(database_path)

    before = database.stats()
    database.initialize()
    database.initialize()
    after = database.stats()

    assert before == after
    assert database.integrity_report()["integrity"] == ["ok"]


def test_schema_v7_migrates_to_v8_with_content_and_history_preserved(
    tmp_path: Path,
) -> None:
    database_path, _source = _populated_database(tmp_path)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE index_scope_exclusions")
        connection.execute("PRAGMA user_version = 7")
        connection.commit()

    migrated = DatabaseManager(database_path)
    migrated.initialize()

    with migrated.connect() as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        table_exists = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'index_scope_exclusions'
                """
            ).fetchone()[0]
        )
    hits = SearchEngine(migrated).search(
        SearchQuery(
            text="SCHEMA_V7_PRESERVES_GOLDEN_CONTENT",
            mode="exact",
        )
    ).total_confirmed

    assert version == SCHEMA_VERSION == 8
    assert table_exists == 1
    assert hits == 1
    assert migrated.search_history() == ["SCHEMA_V7_HISTORY"]
    assert (tmp_path / "index.schema-v7.backup.db").is_file()


def test_failed_migration_restores_the_automatic_backup(
    tmp_path: Path,
) -> None:
    database_path, _source = _populated_database(tmp_path)
    _downgrade_layout(database_path, 6)
    database = DatabaseManager(database_path)

    def fail_after_write(
        self: DatabaseManager,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            "CREATE TABLE migration_damage(value TEXT)"
        )
        raise RuntimeError("injected migration failure")

    with patch.object(
        DatabaseManager,
        "_migrate_schema_v7",
        fail_after_write,
    ):
        with pytest.raises(RuntimeError, match="injected"):
            database.initialize()

    connection = sqlite3.connect(database_path)
    try:
        version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        damage = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'migration_damage'
            """
        ).fetchone()[0]
        block_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_blocks"
            ).fetchone()[0]
        )
    finally:
        connection.close()

    assert version == 6
    assert damage == 0
    assert block_count == 1
    assert (tmp_path / "index.schema-v6.backup.db").is_file()
    assert list(tmp_path.glob("index.migration-failed-*.db"))


def test_incomplete_latest_structural_repair_is_backed_up_and_restored(
    tmp_path: Path,
) -> None:
    database_path, _source = _populated_database(tmp_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("DROP INDEX idx_content_blocks_index_version")
        connection.commit()
    finally:
        connection.close()

    database = DatabaseManager(database_path)

    def fail_latest_repair(
        self: DatabaseManager,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute("CREATE TABLE migration_damage(value TEXT)")
        raise RuntimeError("injected v7 repair failure")

    with patch.object(
        DatabaseManager,
        "_migrate_schema_v7",
        fail_latest_repair,
    ):
        with pytest.raises(RuntimeError, match="injected v7 repair"):
            database.initialize()

    with sqlite3.connect(database_path) as connection:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        damage = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'migration_damage'
                """
            ).fetchone()[0]
        )
        missing_index = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'index' AND name = 'idx_content_blocks_index_version'
                """
            ).fetchone()[0]
        )

    assert version == SCHEMA_VERSION
    assert damage == 0
    assert missing_index == 0
    assert (
        tmp_path / f"index.schema-v{SCHEMA_VERSION}.backup.db"
    ).is_file()


def test_schema_v7_frozen_validator_includes_required_v4_upgrade(
    tmp_path: Path,
) -> None:
    result = run_schema_v7_validation(tmp_path)

    assert result["passed"] is True
    assert result["checks"]["v4_to_latest"]["passed"] is True


def test_schema_v8_validator_covers_v7_upgrade_and_failure_restore(
    tmp_path: Path,
) -> None:
    validator = getattr(schema_validation, "run_schema_v8_validation", None)

    assert callable(validator)
    result = validator(tmp_path)

    assert result["passed"] is True
    assert result["schema_version"] == 8
    assert result["checks"]["v7_to_latest"]["passed"] is True
    assert result["checks"]["failure_restore_v8"]["passed"] is True
    assert result["checks"]["v4_to_latest"]["from_version"] == 4
