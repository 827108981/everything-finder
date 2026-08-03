from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import (
    SCHEMA_VERSION,
    DatabaseManager,
)
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


_V7_TABLES = (
    "pdf_page_identities",
    "ocr_requests",
    "ocr_exact_cache",
    "index_versions",
    "resource_events",
    "backend_benchmarks",
)
_V8_TABLES = ("index_scope_exclusions",)
_V6_COLUMNS = (
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
_V7_COLUMNS = (
    "lease_owner",
    "lease_expires_at",
    "confirmed_at",
    "source_digest",
    "task_version",
    "result_digest",
    "index_version_id",
)
_V5_FILE_COLUMNS = (
    "source_kind",
    "container_file_id",
    "internal_path",
    "member_order",
    "member_crc32",
    "member_uncompressed_size",
    "content_hash_full",
)
_GOLDEN_TEXT = "SCHEMA_V7_PRESERVES_GOLDEN_CONTENT"
_GOLDEN_HISTORY = "SCHEMA_V7_PRESERVES_HISTORY"


def _create_populated_database(base: Path) -> Path:
    root = base / "files"
    root.mkdir(parents=True)
    (root / "preserved.txt").write_text(
        _GOLDEN_TEXT,
        encoding="utf-8",
    )
    database_path = base / "index.db"
    database = DatabaseManager(database_path)
    database.initialize()
    root_id = database.add_root(root)
    IndexManager(
        database,
        AppSettings(enable_ocr=False),
    ).index_root(root_id)
    database.add_search_history(_GOLDEN_HISTORY)
    return database_path


def _downgrade_layout(database_path: Path, version: int) -> None:
    connection = sqlite3.connect(database_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "UPDATE content_blocks SET index_version_id = NULL"
        )
        if version <= 7:
            for table in _V8_TABLES:
                connection.execute(f"DROP TABLE IF EXISTS {table}")
        for table in _V7_TABLES:
            if version <= 6:
                connection.execute(f"DROP TABLE IF EXISTS {table}")
        excluded = set(_V7_COLUMNS)
        if version <= 5:
            excluded.update(_V6_COLUMNS)
            connection.execute(
                "DROP TABLE IF EXISTS parse_task_attempts"
            )
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
        if version <= 4:
            file_column_rows = [
                row
                for row in connection.execute(
                    "PRAGMA table_info(files)"
                )
                if str(row[1]) not in set(_V5_FILE_COLUMNS)
            ]
            file_columns = [
                str(row[1]) for row in file_column_rows
            ]
            file_selected = ", ".join(
                f'"{column}"' for column in file_columns
            )
            connection.execute(
                "CREATE TABLE files_legacy ("
                + ", ".join(
                    (
                        f'"{row[1]}" INTEGER PRIMARY KEY'
                        if int(row[5])
                        else (
                            f'"{row[1]}" {row[2] or "TEXT"}'
                            + (
                                " NOT NULL"
                                if int(row[3])
                                else ""
                            )
                            + (
                                f" DEFAULT {row[4]}"
                                if row[4] is not None
                                else ""
                            )
                        )
                    )
                    for row in file_column_rows
                )
                + ")"
            )
            connection.execute(
                f"INSERT INTO files_legacy({file_selected}) "
                f"SELECT {file_selected} FROM files"
            )
            connection.execute("DROP TABLE files")
            connection.execute(
                "ALTER TABLE files_legacy RENAME TO files"
            )
        connection.execute(f"PRAGMA user_version = {int(version)}")
        connection.commit()
    finally:
        connection.close()


def _snapshot(database: DatabaseManager) -> dict[str, object]:
    digest = hashlib.sha256()
    with database.connect() as connection:
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            for table in (
                "roots",
                "files",
                "documents",
                "content_blocks",
                "content_fts",
                "search_history",
                "parse_tasks",
            )
        }
        for row in connection.execute(
            """
            SELECT file_id, block_index, block_type, location_text,
                   page_number, raw_text, normalized_text, source_type,
                   extra_json
            FROM content_blocks
            ORDER BY file_id, block_index, id
            """
        ):
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
    with database.connect() as connection:
        has_v8_scope = bool(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'index_scope_exclusions'
                """
            ).fetchone()[0]
        )
        if has_v8_scope:
            hits = SearchEngine(database).search(
                SearchQuery(text=_GOLDEN_TEXT, mode="exact")
            ).total_confirmed
        else:
            hits = int(
                connection.execute(
                    """
                    SELECT COUNT(*) FROM content_fts
                    WHERE normalized_text LIKE ?
                    """,
                    (f"%{_GOLDEN_TEXT.lower()}%",),
                ).fetchone()[0]
            )
    return {
        "user_version": user_version,
        "counts": counts,
        "content_digest": digest.hexdigest(),
        "golden_hits": hits,
        "search_history": database.search_history(),
    }


def _validate_upgrade(base: Path, version: int) -> dict[str, object]:
    database_path = _create_populated_database(base)
    _downgrade_layout(database_path, version)
    before = _snapshot(DatabaseManager(database_path))
    started = time.perf_counter()
    migrated = DatabaseManager(database_path)
    migrated.initialize()
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    after = _snapshot(migrated)
    integrity = migrated.integrity_report()
    backup = base / f"index.schema-v{version}.backup.db"
    passed = bool(
        after["user_version"] == SCHEMA_VERSION
        and before["counts"] == after["counts"]
        and before["content_digest"] == after["content_digest"]
        and before["golden_hits"] == after["golden_hits"] == 1
        and before["search_history"] == after["search_history"]
        and backup.is_file()
        and integrity["integrity"] == ["ok"]
        and not integrity["foreign_key_errors"]
    )
    return {
        "passed": passed,
        "from_version": version,
        "to_version": after["user_version"],
        "migration_ms": elapsed_ms,
        "backup_created": backup.is_file(),
        "before": before,
        "after": after,
        "integrity": integrity,
    }


def _validate_latest_idempotency(base: Path) -> dict[str, object]:
    database_path = _create_populated_database(base)
    database = DatabaseManager(database_path)
    before = _snapshot(database)
    database.initialize()
    database.initialize()
    after = _snapshot(database)
    integrity = database.integrity_report()
    passed = bool(
        before == after
        and integrity["integrity"] == ["ok"]
        and not integrity["foreign_key_errors"]
    )
    return {
        "passed": passed,
        "before": before,
        "after": after,
        "integrity": integrity,
    }


class _FailingMigrationDatabase(DatabaseManager):
    def _migrate_schema_v7(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            "CREATE TABLE migration_damage(value TEXT)"
        )
        raise RuntimeError("injected migration failure")


class _FailingV8MigrationDatabase(DatabaseManager):
    def _migrate_schema_v8(
        self,
        connection: sqlite3.Connection,
    ) -> None:
        connection.execute(
            "CREATE TABLE migration_v8_damage(value TEXT)"
        )
        raise RuntimeError("injected v8 migration failure")


def _validate_failure_restore(base: Path) -> dict[str, object]:
    database_path = _create_populated_database(base)
    _downgrade_layout(database_path, 6)
    failure = ""
    try:
        _FailingMigrationDatabase(database_path).initialize()
    except RuntimeError as exc:
        failure = str(exc)
    connection = sqlite3.connect(database_path)
    try:
        version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        damage = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'migration_damage'
                """
            ).fetchone()[0]
        )
        block_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_blocks"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    backup = base / "index.schema-v6.backup.db"
    failed_copies = list(
        base.glob("index.migration-failed-*.db")
    )
    passed = bool(
        failure == "injected migration failure"
        and version == 6
        and damage == 0
        and block_count == 1
        and backup.is_file()
        and failed_copies
    )
    return {
        "passed": passed,
        "failure": failure,
        "restored_version": version,
        "damage_table_count": damage,
        "content_block_count": block_count,
        "backup_created": backup.is_file(),
        "failed_copy_count": len(failed_copies),
    }


def _validate_v8_failure_restore(base: Path) -> dict[str, object]:
    database_path = _create_populated_database(base)
    _downgrade_layout(database_path, 7)
    failure = ""
    try:
        _FailingV8MigrationDatabase(database_path).initialize()
    except RuntimeError as exc:
        failure = str(exc)
    connection = sqlite3.connect(database_path)
    try:
        version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
        damage = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'migration_v8_damage'
                """
            ).fetchone()[0]
        )
        exclusion_table = int(
            connection.execute(
                """
                SELECT COUNT(*) FROM sqlite_master
                WHERE type = 'table' AND name = 'index_scope_exclusions'
                """
            ).fetchone()[0]
        )
        block_count = int(
            connection.execute(
                "SELECT COUNT(*) FROM content_blocks"
            ).fetchone()[0]
        )
    finally:
        connection.close()
    backup = base / "index.schema-v7.backup.db"
    failed_copies = list(base.glob("index.migration-failed-*.db"))
    passed = bool(
        failure == "injected v8 migration failure"
        and version == 7
        and damage == 0
        and exclusion_table == 0
        and block_count == 1
        and backup.is_file()
        and failed_copies
    )
    return {
        "passed": passed,
        "failure": failure,
        "restored_version": version,
        "damage_table_count": damage,
        "exclusion_table_count": exclusion_table,
        "content_block_count": block_count,
        "backup_created": backup.is_file(),
        "failed_copy_count": len(failed_copies),
    }


def run_schema_v8_validation(base: Path) -> dict[str, object]:
    base.mkdir(parents=True, exist_ok=True)
    checks = {
        "v4_to_latest": _validate_upgrade(
            base / "v4_to_latest",
            4,
        ),
        "v5_to_latest": _validate_upgrade(
            base / "v5_to_latest",
            5,
        ),
        "v6_to_latest": _validate_upgrade(
            base / "v6_to_latest",
            6,
        ),
        "v7_to_latest": _validate_upgrade(
            base / "v7_to_latest",
            7,
        ),
        "latest_idempotent": _validate_latest_idempotency(
            base / "latest_idempotent"
        ),
        "failure_restore": _validate_failure_restore(
            base / "failure_restore"
        ),
        "failure_restore_v8": _validate_v8_failure_restore(
            base / "failure_restore_v8"
        ),
    }
    return {
        "passed": all(
            bool(check["passed"]) for check in checks.values()
        ),
        "schema_version": SCHEMA_VERSION,
        "checks": checks,
    }


def run_schema_v7_validation(base: Path) -> dict[str, object]:
    """Backward-compatible name retained for old field scripts."""
    return run_schema_v8_validation(base)
