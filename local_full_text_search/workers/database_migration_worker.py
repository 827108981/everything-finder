from __future__ import annotations

import ctypes
import logging
import os
import shutil
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Callable

from PySide6.QtCore import QObject, Signal, Slot
import psutil

from local_full_text_search.config.constants import APP_DISPLAY_NAME
from local_full_text_search.services.database_location_service import (
    DatabaseLocationService,
)
from local_full_text_search.services.logging_service import log_event, log_exception


logger = logging.getLogger(__name__)
MIGRATION_MARGIN_BYTES = 64 * 1024 * 1024
REQUIRED_TABLES = {
    "roots",
    "files",
    "documents",
    "content_blocks",
    "content_fts",
    "files_fts",
    "index_state",
}


class DatabaseMigrationCancelled(RuntimeError):
    pass


class DatabaseMigrationWorker(QObject):
    progress = Signal(object)
    finished = Signal(object)
    cancelled = Signal()
    failed = Signal(str)

    def __init__(
        self,
        source_path: Path,
        target_directory: Path,
        location_service: DatabaseLocationService,
        *,
        allow_other_instances: bool = False,
    ) -> None:
        super().__init__()
        self.source_path = Path(source_path).resolve()
        self.target_directory = Path(target_directory).resolve()
        self.target_path = self.target_directory / self.source_path.name
        self.location_service = location_service
        self.allow_other_instances = bool(allow_other_instances)
        self._cancelled = threading.Event()
        self._last_stage = ""

    @Slot()
    def run(self) -> None:
        if self.source_path.is_dir():
            self._run_store_directory_migration()
            return
        temporary = self.target_path.with_suffix(
            self.target_path.suffix + ".migration.tmp"
        )
        archived_source: Path | None = None
        location_pending = False
        location_switched = False
        try:
            self._emit("preparing", "正在准备迁移", 0)
            self._validate_paths()
            source_size = self.source_path.stat().st_size
            required = source_size + max(MIGRATION_MARGIN_BYTES, source_size // 4)
            free = shutil.disk_usage(self.target_directory).free
            if free < required:
                raise RuntimeError(
                    f"目标磁盘空间不足，需要至少 {_format_bytes(required)}，"
                    f"当前可用 {_format_bytes(free)}"
                )
            self._throw_if_cancelled()
            temporary.unlink(missing_ok=True)

            source_snapshot = _database_snapshot(
                self.source_path,
                full_check=False,
                cancel_requested=self._cancelled.is_set,
            )
            self.location_service.save_pending(
                self.source_path,
                self.target_path,
            )
            location_pending = True

            self._emit("transferring", "正在传输数据库", 5, total_bytes=source_size)
            source = sqlite3.connect(self.source_path, timeout=30)
            target = sqlite3.connect(temporary, timeout=30)
            try:
                source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                page_size = int(source.execute("PRAGMA page_size").fetchone()[0])

                def report(_status: int, remaining: int, total: int) -> None:
                    self._throw_if_cancelled()
                    processed_pages = max(0, int(total) - int(remaining))
                    total_pages = max(1, int(total))
                    processed_bytes = min(source_size, processed_pages * page_size)
                    percent = 5 + int(65 * processed_pages / total_pages)
                    self._emit(
                        "transferring",
                        "正在传输数据库",
                        percent,
                        processed_bytes=processed_bytes,
                        total_bytes=source_size,
                    )

                source.backup(target, pages=1024, progress=report, sleep=0.02)
                target.commit()
            finally:
                target.close()
                source.close()

            self._throw_if_cancelled()
            self._emit("verifying", "正在校验数据库完整性", 72)
            target_snapshot = _database_snapshot(
                temporary,
                full_check=True,
                cancel_requested=self._cancelled.is_set,
            )
            _compare_snapshots(source_snapshot, target_snapshot)

            self._throw_if_cancelled()
            self._emit("switching", "正在切换数据库路径", 84)
            if self.target_path.exists():
                raise RuntimeError(f"目标数据库已存在：{self.target_path}")
            os.replace(temporary, self.target_path)
            final_snapshot = _database_snapshot(
                self.target_path,
                full_check=True,
                cancel_requested=self._cancelled.is_set,
            )
            _compare_snapshots(source_snapshot, final_snapshot)

            try:
                self.location_service.save_verified(
                    self.target_path,
                    schema_version=int(final_snapshot["schema_version"]),
                )
                location_pending = False
                location_switched = True
                archived_source = self._archive_source()
            except Exception:
                if archived_source is not None:
                    self._restore_source(archived_source)
                    archived_source = None
                self.location_service.restore_verified(self.source_path)
                self.target_path.unlink(missing_ok=True)
                raise

            self._emit("reopening", "正在重新打开数据库", 94)
            reopened = _database_snapshot(
                self.target_path,
                full_check=False,
                cancel_requested=self._cancelled.is_set,
            )
            _compare_snapshots(source_snapshot, reopened)
            self._emit("completed", "数据库迁移完成", 100)
            self.finished.emit(
                {
                    "database_path": str(self.target_path),
                    "previous_database_backup": str(archived_source or ""),
                    "database_size_bytes": int(self.target_path.stat().st_size),
                    "schema_version": int(final_snapshot["schema_version"]),
                    "counts": dict(final_snapshot["counts"]),
                }
            )
            logger.info(
                "Database migration completed: source=%s target=%s backup=%s bytes=%s",
                self.source_path,
                self.target_path,
                archived_source,
                self.target_path.stat().st_size,
            )
        except DatabaseMigrationCancelled:
            temporary.unlink(missing_ok=True)
            if archived_source is not None:
                self._restore_source(archived_source)
            if location_pending or location_switched:
                self.location_service.restore_verified(self.source_path)
            if self.source_path.is_file():
                self.target_path.unlink(missing_ok=True)
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Database migration failed")
            temporary.unlink(missing_ok=True)
            if archived_source is not None:
                try:
                    self._restore_source(archived_source)
                except OSError:
                    logger.exception("Failed to restore archived source database")
            if location_pending:
                try:
                    self.location_service.restore_verified(self.source_path)
                except Exception:
                    logger.exception("Failed to restore database location configuration")
            elif location_switched:
                try:
                    self.location_service.restore_verified(self.source_path)
                except Exception:
                    logger.exception("Failed to roll back database location configuration")
            if self.source_path.is_file():
                self.target_path.unlink(missing_ok=True)
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def _run_store_directory_migration(self) -> None:
        """Move the complete generation store, not a single SQLite file."""

        source_root = self.source_path
        target_root = self.target_directory
        temporary_root = target_root.with_name(target_root.name + ".migration.tmp")
        archived_source: Path | None = None
        location_pending = False
        try:
            self._emit("preparing", "正在准备索引存储目录迁移", 0)
            self._validate_store_paths(source_root, target_root)
            files = [
                path
                for path in source_root.rglob("*")
                if path.is_file()
                and not path.name.endswith(("-wal", "-shm", "-journal"))
                and ".migration.tmp" not in path.name
            ]
            total_bytes = sum(path.stat().st_size for path in files)
            free = shutil.disk_usage(target_root.parent).free
            required = total_bytes + max(MIGRATION_MARGIN_BYTES, total_bytes // 4)
            if free < required:
                raise RuntimeError(
                    f"目标磁盘空间不足，需要至少 {_format_bytes(required)}，"
                    f"当前可用 {_format_bytes(free)}"
                )
            if temporary_root.exists():
                shutil.rmtree(temporary_root)
            self.location_service.save_pending(source_root, target_root)
            location_pending = True
            temporary_root.mkdir(parents=True, exist_ok=True)
            copied = 0
            self._emit("transferring", "正在传输索引存储目录", 5, total_bytes=total_bytes)
            for source_file in files:
                self._throw_if_cancelled()
                relative = source_file.relative_to(source_root)
                destination = temporary_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_file, destination)
                copied += source_file.stat().st_size
                percent = 5 + int(70 * copied / max(1, total_bytes))
                self._emit(
                    "transferring",
                    "正在传输索引存储目录",
                    percent,
                    processed_bytes=copied,
                    total_bytes=total_bytes,
                )
            self._throw_if_cancelled()
            self._emit("verifying", "正在校验索引存储目录", 80)
            if not self.location_service._store_is_usable(temporary_root) and files:
                raise RuntimeError("迁移后的索引存储目录缺少有效数据库或活动清单")
            if target_root.exists():
                target_root.rmdir()
            os.replace(temporary_root, target_root)
            self._emit("switching", "正在切换索引存储目录", 90)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            archived_source = source_root.with_name(
                f"{source_root.name}.migrated-{stamp}.backup"
            )
            os.replace(source_root, archived_source)
            self.location_service.save_verified(target_root)
            location_pending = False
            self._emit("completed", "索引存储目录迁移完成", 100)
            payload = {
                "database_path": str(target_root / "search_index.db"),
                "index_store_root": str(target_root),
                "previous_database_backup": str(archived_source),
                "database_size_bytes": sum(
                    path.stat().st_size for path in target_root.rglob("*") if path.is_file()
                ),
            }
            self.finished.emit(payload)
            log_event(
                "database.store_migration.completed",
                component="database_migration_worker",
                source_root=str(source_root),
                target_root=str(target_root),
                file_count=len(files),
                database_size_bytes=payload["database_size_bytes"],
            )
        except DatabaseMigrationCancelled:
            if temporary_root.exists():
                shutil.rmtree(temporary_root, ignore_errors=True)
            if location_pending:
                self.location_service.restore_verified(source_root)
            self.cancelled.emit()
        except Exception as exc:
            log_exception(
                "database.store_migration.failed",
                exc,
                component="database_migration_worker",
                source_root=str(source_root),
                target_root=str(target_root),
                location_pending=location_pending,
            )
            if temporary_root.exists():
                shutil.rmtree(temporary_root, ignore_errors=True)
            if archived_source is not None and not source_root.exists():
                os.replace(archived_source, source_root)
            if location_pending:
                try:
                    self.location_service.restore_verified(source_root)
                except Exception:
                    logger.exception("Failed to restore store location configuration")
            self.failed.emit(str(exc) or exc.__class__.__name__)

    @staticmethod
    def _validate_store_paths(source_root: Path, target_root: Path) -> None:
        if not source_root.is_dir():
            raise RuntimeError(f"当前索引存储目录不存在：{source_root}")
        if target_root == source_root:
            raise RuntimeError("目标目录与当前索引存储目录相同")
        if source_root in target_root.parents or target_root in source_root.parents:
            raise RuntimeError("索引存储目录不能互相嵌套")
        target_root.parent.mkdir(parents=True, exist_ok=True)
        if target_root.exists() and any(target_root.iterdir()):
            raise RuntimeError(f"目标索引存储目录不是空目录：{target_root}")
        install_directory = Path(sys.executable).resolve().parent
        if target_root == install_directory or install_directory in target_root.parents:
            raise RuntimeError("索引存储目录不能位于程序安装目录")

    @Slot()
    def cancel(self) -> None:
        self._cancelled.set()

    def _validate_paths(self) -> None:
        if not self.source_path.is_file():
            raise RuntimeError(f"当前数据库不存在：{self.source_path}")
        if not self.allow_other_instances:
            other_instances = _other_application_instances()
            if other_instances:
                details = "、".join(
                    f"PID {pid} ({name})" for pid, name in other_instances
                )
                raise RuntimeError(
                    "检测到其他版本或实例仍在运行："
                    + details
                    + "。请关闭所有本地全文搜索程序后重试。"
                )
        _assert_database_write_available(self.source_path)
        if self.target_path == self.source_path:
            raise RuntimeError("目标目录与当前数据库目录相同")
        self.target_directory.mkdir(parents=True, exist_ok=True)
        if self.target_path.exists():
            raise RuntimeError(f"目标目录已经存在同名数据库：{self.target_path}")
        install_directory = Path(sys.executable).resolve().parent
        if (
            self.target_directory == install_directory
            or install_directory in self.target_directory.parents
        ):
            raise RuntimeError("数据库不能迁移到程序安装目录")
        if os.name == "nt" and not _is_fixed_drive(self.target_directory):
            raise RuntimeError("数据库只能迁移到本机固定磁盘")
        probe = self.target_directory / ".lfts-write-test.tmp"
        try:
            probe.write_bytes(b"ok")
        finally:
            probe.unlink(missing_ok=True)

    def _archive_source(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self.source_path.with_name(
            f"{self.source_path.stem}.migrated-{stamp}.backup"
            f"{self.source_path.suffix}"
        )
        os.replace(self.source_path, backup)
        for suffix in ("-wal", "-shm"):
            Path(str(self.source_path) + suffix).unlink(missing_ok=True)
        return backup

    def _restore_source(self, archived_source: Path) -> None:
        if archived_source.is_file() and not self.source_path.exists():
            os.replace(archived_source, self.source_path)

    def _throw_if_cancelled(self) -> None:
        if self._cancelled.is_set():
            raise DatabaseMigrationCancelled()

    def _emit(
        self,
        stage: str,
        label: str,
        percent: int,
        *,
        processed_bytes: int = 0,
        total_bytes: int = 0,
    ) -> None:
        if stage != self._last_stage:
            self._last_stage = stage
            logger.info(
                "Database migration phase: stage=%s source=%s target=%s",
                stage,
                self.source_path,
                self.target_path,
            )
        self.progress.emit(
            {
                "stage": stage,
                "phase_label": label,
                "percent": max(0, min(100, int(percent))),
                "processed_bytes": max(0, int(processed_bytes)),
                "total_bytes": max(0, int(total_bytes)),
            }
        )


def _database_snapshot(
    path: Path,
    *,
    full_check: bool,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, object]:
    connection = sqlite3.connect(
        f"file:{Path(path).as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    try:
        if cancel_requested is not None:
            connection.set_progress_handler(
                lambda: 1 if cancel_requested() else 0,
                1000,
            )
        try:
            check = "integrity_check" if full_check else "quick_check(1)"
            integrity = [
                str(row[0]) for row in connection.execute(f"PRAGMA {check}")
            ]
        except sqlite3.OperationalError as exc:
            if cancel_requested is not None and cancel_requested():
                raise DatabaseMigrationCancelled() from exc
            raise
        if integrity != ["ok"]:
            raise RuntimeError(f"数据库完整性校验失败：{integrity[:3]}")
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
        missing = REQUIRED_TABLES - tables
        if missing:
            raise RuntimeError("数据库缺少关键表：" + ", ".join(sorted(missing)))
        counts = {
            table: int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            for table in (
                "roots",
                "files",
                "documents",
                "content_blocks",
                "content_fts",
                "files_fts",
            )
        }
        return {
            "schema_version": int(
                connection.execute("PRAGMA user_version").fetchone()[0]
            ),
            "counts": counts,
        }
    except sqlite3.OperationalError as exc:
        if cancel_requested is not None and cancel_requested():
            raise DatabaseMigrationCancelled() from exc
        raise
    finally:
        connection.set_progress_handler(None, 0)
        connection.close()


def _compare_snapshots(
    source: dict[str, object],
    target: dict[str, object],
) -> None:
    if int(source["schema_version"]) != int(target["schema_version"]):
        raise RuntimeError("迁移前后数据库版本不一致")
    if dict(source["counts"]) != dict(target["counts"]):
        raise RuntimeError("迁移前后关键记录数量不一致")


def _is_fixed_drive(path: Path) -> bool:
    anchor = Path(path).resolve().anchor
    if not anchor:
        return False
    drive_type = int(ctypes.windll.kernel32.GetDriveTypeW(str(anchor)))
    return drive_type == 3


def _format_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= 1024
    return f"{size:.1f} TB"


def _other_application_instances() -> list[tuple[int, str]]:
    current_pid = os.getpid()
    expected = APP_DISPLAY_NAME.lower()
    matches: list[tuple[int, str]] = []
    for process in psutil.process_iter(["pid", "name", "exe"]):
        try:
            pid = int(process.info.get("pid") or 0)
            if pid == current_pid:
                continue
            name = str(process.info.get("name") or "")
            executable = Path(str(process.info.get("exe") or "")).name
            identity = f"{name} {executable}".lower()
            if expected in identity:
                matches.append((pid, name or executable))
        except (psutil.Error, OSError, ValueError):
            continue
    return sorted(matches)


def _assert_database_write_available(path: Path) -> None:
    connection = sqlite3.connect(path, timeout=1)
    try:
        connection.execute("PRAGMA busy_timeout = 1000")
        connection.execute("BEGIN IMMEDIATE")
        connection.rollback()
    except sqlite3.OperationalError as exc:
        if "locked" in str(exc).lower() or "busy" in str(exc).lower():
            raise RuntimeError(
                "当前数据库正在被其他进程写入，请关闭其他实例后重试"
            ) from exc
        raise
    finally:
        connection.close()
