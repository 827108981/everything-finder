from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.config.constants import CONFIG_DIR, DB_PATH


LOCATION_FORMAT_VERSION = 2
STARTUP_PROBE_TIMEOUT_SECONDS = 0.75


class DatabaseLocationError(RuntimeError):
    """Raised when a configured database cannot be opened safely."""


@dataclass(frozen=True, slots=True)
class DatabaseLocationResolution:
    path: Path
    source: str
    warning: str = ""
    index_store_root: Path | None = None

    @property
    def store_root(self) -> Path:
        """The directory containing control DB, manifest and generations."""

        return self.index_store_root or (
            self.path.parent if self.path.suffix.lower() == ".db" else self.path
        )


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
                path, migrated_from_legacy_file = self._path_for_payload(payload)
                state = str(payload.get("migration_state") or "")
                if state == "pending":
                    previous = self._optional_path(
                        payload.get("previous_index_store_root")
                        or payload.get("previous_database_path")
                    )
                    if previous is not None:
                        if previous.suffix.lower() == ".db":
                            previous = previous.parent
                        if self._store_is_usable(previous):
                            self.restore_verified(previous)
                            return DatabaseLocationResolution(
                                previous / self.default_db_path.name,
                                "pending_recovery",
                                "检测到未完成的索引存储目录迁移，已继续使用迁移前目录。",
                                previous,
                            )
                    raise DatabaseLocationError(
                        "索引存储目录迁移尚未完成，且迁移前目录不可用"
                    )
                if state != "verified":
                    raise DatabaseLocationError(
                        f"数据库位置配置状态无效：{state or 'missing'}"
                    )
                configured_directory = bool(payload.get("index_store_root"))
                if not path.is_dir() or (
                    not configured_directory and not self._store_is_usable(path)
                ):
                    raise DatabaseLocationError(f"配置的索引存储目录不存在或不可用：{path}")
                warning = ""
                if migrated_from_legacy_file:
                    self.save_verified(path)
                    warning = "已将旧版数据库文件配置转换为索引存储目录配置。"
                if source == "config_backup":
                    self.save_verified(path)
                    warning = "数据库位置配置已损坏，当前使用上一次有效配置。"
                return DatabaseLocationResolution(
                    path / self.default_db_path.name,
                    source,
                    warning,
                    path,
                )
            except (OSError, ValueError, sqlite3.Error, DatabaseLocationError) as exc:
                primary_error = str(exc) or exc.__class__.__name__

        default_root = self.default_db_path.parent
        if self._store_is_usable(default_root):
            if not self._database_is_usable(self.default_db_path) and not (
                default_root / "index_control.db"
            ).is_file():
                raise DatabaseLocationError(
                    f"默认索引存储目录存在但校验失败：{default_root}"
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
                default_root,
            )

        if primary_error:
            raise DatabaseLocationError(
                "数据库位置配置不可用，且未找到可恢复的默认数据库："
                + primary_error
            )
        return DatabaseLocationResolution(
            self.default_db_path,
            "new_default",
            "",
            default_root,
        )

    def save_pending(self, source: Path, target: Path) -> None:
        source_root = self._store_root(source)
        target_root = self._store_root(target)
        self._write_payload(
            {
                "format_version": LOCATION_FORMAT_VERSION,
                "index_store_root": str(target_root),
                "previous_index_store_root": str(source_root),
                "last_verified": self._now(),
                "migration_state": "pending",
            }
        )

    def save_verified(self, path: Path, *, schema_version: int | None = None) -> None:
        store_root = self._store_root(path)
        self._write_payload(
            {
                "format_version": LOCATION_FORMAT_VERSION,
                "index_store_root": str(store_root),
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
    def _path_for_payload(payload: dict[str, object]) -> tuple[Path, bool]:
        store_root = str(payload.get("index_store_root") or "").strip()
        if store_root:
            return Path(store_root).expanduser().resolve(), False
        explicit = str(payload.get("database_path") or "").strip()
        if explicit:
            return Path(explicit).expanduser().resolve().parent, True
        root = str(payload.get("database_root") or "").strip()
        filename = str(payload.get("database_file") or "").strip()
        if not root or not filename:
            raise ValueError("数据库位置配置缺少路径")
        return (Path(root).expanduser() / filename).resolve().parent, True

    @staticmethod
    def _store_root(path: Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        if resolved.suffix.lower() == ".db":
            return resolved.parent
        return resolved

    @classmethod
    def _store_is_usable(cls, path: Path) -> bool:
        root = cls._store_root(path)
        if not root.is_dir():
            return False
        manifest = root / "search_index.manifest.json"
        control = root / "index_control.db"
        legacy = root / "search_index.db"
        return (
            manifest.is_file()
            or cls._database_is_usable(control)
            or cls._database_is_usable(legacy)
        )

    @staticmethod
    def _optional_path(value: object) -> Path | None:
        text = str(value or "").strip()
        return Path(text).expanduser().resolve() if text else None

    @staticmethod
    def _database_is_usable(path: Path) -> bool:
        if not path.is_file() or path.stat().st_size <= 0:
            return False
        try:
            # Do not open SQLite during GUI startup. Opening a WAL database can
            # replay/checkpoint a multi-gigabyte log before returning, which
            # blocks the main thread before QApplication can show a window.
            # DatabaseManager performs the real schema/integrity work in the
            # background scan after the window is visible.
            with path.open("rb") as handle:
                return handle.read(16) == b"SQLite format 3\x00"
        except OSError:
            return False

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
