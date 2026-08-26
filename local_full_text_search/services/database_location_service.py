from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.config.constants import CONFIG_DIR, DB_PATH


LOCATION_FORMAT_VERSION = 1
STARTUP_PROBE_TIMEOUT_SECONDS = 0.75


class DatabaseLocationError(RuntimeError):
    """Raised when a configured database cannot be opened safely."""


@dataclass(frozen=True, slots=True)
class DatabaseLocationResolution:
    path: Path
    source: str
    warning: str = ""


class DatabaseLocationService:
    """Persist the active database path outside the application directory."""

    def __init__(
        self,
        config_path: Path | None = None,
        default_db_path: Path | None = None,
    ) -> None:
        self.config_path = Path(
            config_path or (CONFIG_DIR / "database-location.json")
        )
        self.backup_path = self.config_path.with_suffix(
            self.config_path.suffix + ".bak"
        )
        self.default_db_path = Path(default_db_path or DB_PATH)

    def resolve(self) -> DatabaseLocationResolution:
        primary_error = ""
        for candidate, source in (
            (self.config_path, "configured"),
            (self.backup_path, "config_backup"),
        ):
            if not candidate.is_file():
                continue
            try:
                payload = self._read_payload(candidate)
                path = self._path_for_payload(payload)
                state = str(payload.get("migration_state") or "")
                if state == "pending":
                    previous = self._optional_path(
                        payload.get("previous_database_path")
                    )
                    if previous is not None and self._database_is_usable(previous):
                        self.restore_verified(previous)
                        return DatabaseLocationResolution(
                            previous,
                            "pending_recovery",
                            "检测到未完成的数据库迁移，已继续使用迁移前数据库。",
                        )
                    raise DatabaseLocationError(
                        "数据库迁移尚未完成，且迁移前数据库不可用"
                    )
                if state != "verified":
                    raise DatabaseLocationError(
                        f"数据库位置配置状态无效：{state or 'missing'}"
                    )
                if not self._database_is_usable(path):
                    raise DatabaseLocationError(
                        f"配置的数据库不存在或校验失败：{path}"
                    )
                warning = ""
                if source == "config_backup":
                    self._write_payload(payload)
                    warning = "数据库位置配置已损坏，当前使用上一次有效配置。"
                return DatabaseLocationResolution(path, source, warning)
            except (OSError, ValueError, sqlite3.Error, DatabaseLocationError) as exc:
                primary_error = str(exc) or exc.__class__.__name__

        if self.default_db_path.is_file():
            if not self._database_is_usable(self.default_db_path):
                raise DatabaseLocationError(
                    f"默认数据库存在但校验失败：{self.default_db_path}"
                )
            warning = (
                f"数据库位置配置不可用，已回退默认数据库：{primary_error}"
                if primary_error
                else ""
            )
            return DatabaseLocationResolution(
                self.default_db_path,
                "legacy_default",
                warning,
            )

        if primary_error:
            raise DatabaseLocationError(
                "数据库位置配置不可用，且未找到可恢复的默认数据库："
                + primary_error
            )
        return DatabaseLocationResolution(
            self.default_db_path,
            "new_default",
        )

    def save_pending(self, source: Path, target: Path) -> None:
        self._write_payload(
            {
                "format_version": LOCATION_FORMAT_VERSION,
                "database_root": str(Path(target).parent),
                "database_file": Path(target).name,
                "database_path": str(Path(target)),
                "previous_database_path": str(Path(source)),
                "database_schema": self._database_version(source),
                "last_verified": self._now(),
                "migration_state": "pending",
            }
        )

    def save_verified(self, path: Path, *, schema_version: int | None = None) -> None:
        database_path = Path(path).resolve()
        self._write_payload(
            {
                "format_version": LOCATION_FORMAT_VERSION,
                "database_root": str(database_path.parent),
                "database_file": database_path.name,
                "database_path": str(database_path),
                "database_schema": (
                    int(schema_version)
                    if schema_version is not None
                    else self._database_version(database_path)
                ),
                "last_verified": self._now(),
                "migration_state": "verified",
            }
        )

    def restore_verified(self, path: Path) -> None:
        self.save_verified(path)

    def _write_payload(self, payload: dict[str, object]) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config_path.with_suffix(
            self.config_path.suffix + ".tmp"
        )
        encoded = json.dumps(payload, ensure_ascii=False, indent=2)
        temporary.write_text(encoded, encoding="utf-8")
        state = str(payload.get("migration_state") or "")
        if state == "pending" and self.config_path.is_file():
            backup_temporary = self.backup_path.with_suffix(
                self.backup_path.suffix + ".tmp"
            )
            backup_temporary.write_bytes(self.config_path.read_bytes())
            os.replace(backup_temporary, self.backup_path)
        os.replace(temporary, self.config_path)
        if state == "verified":
            backup_temporary = self.backup_path.with_suffix(
                self.backup_path.suffix + ".tmp"
            )
            backup_temporary.write_text(encoded, encoding="utf-8")
            os.replace(backup_temporary, self.backup_path)

    @staticmethod
    def _read_payload(path: Path) -> dict[str, object]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("数据库位置配置不是对象")
        version = int(payload.get("format_version") or 0)
        if version < 1 or version > LOCATION_FORMAT_VERSION:
            raise ValueError(f"不支持的数据库位置配置版本：{version}")
        return payload

    @staticmethod
    def _path_for_payload(payload: dict[str, object]) -> Path:
        explicit = str(payload.get("database_path") or "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve()
        root = str(payload.get("database_root") or "").strip()
        filename = str(payload.get("database_file") or "").strip()
        if not root or not filename:
            raise ValueError("数据库位置配置缺少路径")
        return (Path(root).expanduser() / filename).resolve()

    @staticmethod
    def _optional_path(value: object) -> Path | None:
        text = str(value or "").strip()
        return Path(text).expanduser().resolve() if text else None

    @staticmethod
    def _database_is_usable(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        # Startup only needs to prove that the configured path is a readable
        # SQLite database.  A full quick_check scans every page and can keep
        # the GUI hidden for minutes on a large index.  Schema/migration work
        # is performed by DatabaseManager after this bounded probe.
        connection = sqlite3.connect(
            f"file:{path.as_posix()}?mode=ro",
            uri=True,
            timeout=STARTUP_PROBE_TIMEOUT_SECONDS,
        )
        try:
            connection.execute(
                f"PRAGMA busy_timeout = {int(STARTUP_PROBE_TIMEOUT_SECONDS * 1000)}"
            )
            row = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'table' LIMIT 1"
            ).fetchone()
            if row is None:
                return False
            connection.execute("PRAGMA schema_version").fetchone()
            return True
        finally:
            connection.close()

    @staticmethod
    def _database_version(path: Path) -> int:
        if not Path(path).is_file():
            return 0
        connection = sqlite3.connect(
            f"file:{Path(path).as_posix()}?mode=ro",
            uri=True,
            timeout=5,
        )
        try:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()
