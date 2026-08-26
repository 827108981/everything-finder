from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.services.database_location_service import (
    DatabaseLocationError,
    DatabaseLocationService,
)
from local_full_text_search.workers.database_migration_worker import (
    DatabaseMigrationWorker,
)


def _initialized_database(path: Path) -> DatabaseManager:
    database = DatabaseManager(path)
    database.initialize()
    database.add_root(path.parent / "documents")
    return database


def test_verified_custom_location_survives_future_upgrade(tmp_path: Path) -> None:
    default_path = tmp_path / "default" / "search_index.db"
    custom_path = tmp_path / "custom" / "search_index.db"
    _initialized_database(custom_path)
    config_path = tmp_path / "config" / "database-location.json"

    first_version = DatabaseLocationService(config_path, default_path)
    first_version.save_verified(custom_path)
    next_version = DatabaseLocationService(config_path, default_path)

    resolution = next_version.resolve()

    assert resolution.path == custom_path.resolve()
    assert resolution.source == "configured"


def test_startup_database_probe_is_lightweight(tmp_path: Path) -> None:
    database_path = tmp_path / "search_index.db"
    database_path.write_bytes(b"SQLite format 3\x00" + b"0" * 64)
    executed: list[str] = []

    class Result:
        def fetchone(self) -> tuple[int]:
            return (1,)

    class Connection:
        def execute(self, statement: str) -> Result:
            executed.append(statement)
            return Result()

        def close(self) -> None:
            return None

    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        database_path,
    )
    with patch(
        "local_full_text_search.services.database_location_service.sqlite3.connect",
        return_value=Connection(),
    ) as connect:
        assert service._database_is_usable(database_path) is True

    assert not any("quick_check" in statement for statement in executed)
    assert any("sqlite_master" in statement for statement in executed)
    assert connect.call_args.kwargs["timeout"] <= 1


def test_corrupt_primary_location_uses_verified_config_backup(tmp_path: Path) -> None:
    default_path = tmp_path / "default" / "search_index.db"
    custom_path = tmp_path / "custom" / "search_index.db"
    _initialized_database(custom_path)
    config_path = tmp_path / "config" / "database-location.json"
    service = DatabaseLocationService(config_path, default_path)
    service.save_verified(custom_path)
    config_path.write_text("{broken", encoding="utf-8")

    resolution = service.resolve()

    assert resolution.path == custom_path.resolve()
    assert resolution.source == "config_backup"
    assert "上一次有效配置" in resolution.warning
    repaired = json.loads(config_path.read_text(encoding="utf-8"))
    assert Path(str(repaired["database_path"])) == custom_path.resolve()


def test_pending_migration_keeps_using_previous_database(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target = tmp_path / "target" / "search_index.db"
    _initialized_database(source)
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    service.save_pending(source, target)

    resolution = service.resolve()

    assert resolution.path == source.resolve()
    assert resolution.source == "pending_recovery"
    assert "未完成" in resolution.warning
    recovered = json.loads(service.config_path.read_text(encoding="utf-8"))
    assert recovered["migration_state"] == "verified"


def test_invalid_config_never_creates_silent_empty_database(tmp_path: Path) -> None:
    config = tmp_path / "config" / "database-location.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps(
            {
                "format_version": 1,
                "database_path": str(tmp_path / "missing" / "search_index.db"),
                "migration_state": "verified",
            }
        ),
        encoding="utf-8",
    )
    service = DatabaseLocationService(
        config,
        tmp_path / "default" / "search_index.db",
    )

    with pytest.raises(DatabaseLocationError):
        service.resolve()


def test_database_worker_migrates_and_persists_new_location(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target_directory = tmp_path / "target"
    original = _initialized_database(source)
    original_stats = original.stats()
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    worker = DatabaseMigrationWorker(
        source,
        target_directory,
        service,
        allow_other_instances=True,
    )
    finished: list[dict[str, object]] = []
    failed: list[str] = []
    worker.finished.connect(finished.append)
    worker.failed.connect(failed.append)

    worker.run()

    assert failed == []
    assert len(finished) == 1
    target = target_directory / "search_index.db"
    assert target.is_file()
    assert not source.exists()
    backup = Path(str(finished[0]["previous_database_backup"]))
    assert backup.is_file()
    assert ".migrated-" in backup.name
    assert DatabaseManager(target).stats() == original_stats
    assert service.resolve().path == target.resolve()
    assert json.loads(service.config_path.read_text(encoding="utf-8"))[
        "migration_state"
    ] == "verified"
    backup_payload = json.loads(service.backup_path.read_text(encoding="utf-8"))
    assert Path(str(backup_payload["database_path"])) == target.resolve()


def test_failed_migration_keeps_original_database_and_location(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target_directory = tmp_path / "target"
    _initialized_database(source)
    target_directory.mkdir(parents=True)
    (target_directory / "search_index.db").write_bytes(b"occupied")
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    worker = DatabaseMigrationWorker(
        source,
        target_directory,
        service,
        allow_other_instances=True,
    )
    failed: list[str] = []
    worker.failed.connect(failed.append)

    worker.run()

    assert failed
    assert "已经存在" in failed[0]
    assert source.is_file()
    assert service.resolve().path == source.resolve()


def test_cancelled_migration_before_copy_keeps_original_database(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target_directory = tmp_path / "target"
    _initialized_database(source)
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    worker = DatabaseMigrationWorker(
        source,
        target_directory,
        service,
        allow_other_instances=True,
    )
    cancelled: list[bool] = []
    worker.cancelled.connect(lambda: cancelled.append(True))
    worker.cancel()

    worker.run()

    assert cancelled == [True]
    assert source.is_file()
    assert not (target_directory / "search_index.db").exists()
    assert service.resolve().path == source.resolve()


def test_other_running_version_blocks_migration(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target_directory = tmp_path / "target"
    _initialized_database(source)
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    worker = DatabaseMigrationWorker(source, target_directory, service)
    failed: list[str] = []
    worker.failed.connect(failed.append)

    with patch(
        "local_full_text_search.workers.database_migration_worker._other_application_instances",
        return_value=[(1234, "本地多格式全文搜索工具.exe")],
    ):
        worker.run()

    assert failed and "其他版本或实例" in failed[0]
    assert source.is_file()
    assert service.resolve().path == source.resolve()


def test_database_write_lock_blocks_migration(tmp_path: Path) -> None:
    source = tmp_path / "source" / "search_index.db"
    target_directory = tmp_path / "target"
    _initialized_database(source)
    service = DatabaseLocationService(
        tmp_path / "config" / "database-location.json",
        source,
    )
    service.save_verified(source)
    locker = sqlite3.connect(source, timeout=1)
    locker.execute("BEGIN IMMEDIATE")
    try:
        worker = DatabaseMigrationWorker(
            source,
            target_directory,
            service,
            allow_other_instances=True,
        )
        failed: list[str] = []
        worker.failed.connect(failed.append)
        worker.run()
    finally:
        locker.rollback()
        locker.close()

    assert failed and "正在被其他进程写入" in failed[0]
    assert source.is_file()
    assert service.resolve().path == source.resolve()
