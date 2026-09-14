from __future__ import annotations

import json
import logging
import os
import threading
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from local_full_text_search.config.constants import LOG_DIR


_LOG_LOCK = threading.RLock()
_CONFIGURED = False


def _now() -> datetime:
    return datetime.now().astimezone()


def date_log_dir(value: datetime | None = None) -> Path:
    return LOG_DIR / (value or _now()).strftime("%Y-%m-%d")


class DatePartitionFileHandler(logging.Handler):
    """Write each record into the local-time directory for its date."""

    def __init__(self, filename: str, *, level: int = logging.NOTSET) -> None:
        super().__init__(level)
        self.filename = filename
        self._streams: dict[str, Any] = {}

    def emit(self, record: logging.LogRecord) -> None:
        try:
            current = _now()
            key = current.strftime("%Y-%m-%d")
            directory = date_log_dir(current)
            directory.mkdir(parents=True, exist_ok=True)
            with _LOG_LOCK:
                stream = self._streams.get(key)
                if stream is None:
                    stream = (directory / self.filename).open(
                        "a", encoding="utf-8", buffering=1
                    )
                    self._streams[key] = stream
                stream.write(self.format(record) + "\n")
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with _LOG_LOCK:
            for stream in self._streams.values():
                try:
                    stream.close()
                except Exception:
                    pass
            self._streams.clear()
        super().close()


class ErrorOnlyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.ERROR


def configure_logging() -> None:
    global _CONFIGURED
    with _LOG_LOCK:
        if _CONFIGURED:
            return
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        app_handler = DatePartitionFileHandler("app.log")
        app_handler.setFormatter(formatter)
        root.addHandler(app_handler)
        error_handler = DatePartitionFileHandler("errors.log", level=logging.ERROR)
        error_handler.addFilter(ErrorOnlyFilter())
        error_handler.setFormatter(formatter)
        root.addHandler(error_handler)
        if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            root.addHandler(console_handler)
        _CONFIGURED = True
    cleanup_logs()


def log_event(
    event: str,
    *,
    level: int = logging.INFO,
    component: str = "app",
    request_id: str | None = None,
    run_id: str | None = None,
    snapshot_id: str | None = None,
    **fields: object,
) -> dict[str, object]:
    """Record one searchable event without persisting raw query text."""

    current = _now()
    payload: dict[str, object] = {
        "timestamp": current.isoformat(timespec="milliseconds"),
        "level": logging.getLevelName(level),
        "event": str(event),
        "component": str(component),
        "pid": os.getpid(),
        "thread": threading.current_thread().name,
    }
    for key, value in (
        ("request_id", request_id),
        ("run_id", run_id),
        ("snapshot_id", snapshot_id),
    ):
        if value:
            payload[key] = str(value)
    payload.update({key: value for key, value in fields.items() if value is not None})
    directory = date_log_dir(current)
    directory.mkdir(parents=True, exist_ok=True)
    with _LOG_LOCK:
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    summary_keys = (
        "scan_id",
        "database_path",
        "root_id",
        "operation",
        "stage",
        "phase",
        "task_id",
        "cancellation_source",
        "error_type",
    )
    summary = " ".join(
        f"{key}={str(payload[key]).replace(chr(10), ' ')[:500]}"
        for key in summary_keys
        if key in payload
    )
    logging.getLogger(component).log(
        level,
        "%s request_id=%s run_id=%s%s",
        event,
        request_id or "-",
        run_id or "-",
        f" {summary}" if summary else "",
    )
    return payload


def log_exception(
    event: str,
    exc: BaseException,
    *,
    component: str,
    request_id: str | None = None,
    run_id: str | None = None,
    **fields: object,
) -> dict[str, object]:
    return log_event(
        event,
        level=logging.ERROR,
        component=component,
        request_id=request_id,
        run_id=run_id,
        error_type=type(exc).__name__,
        error_message=str(exc)[:1000],
        stacktrace="".join(traceback.format_exception(exc))[-8000:],
        **fields,
    )


def cleanup_logs(
    *,
    retention_days: int = 30,
    error_retention_days: int = 90,
    max_total_size_mb: int = 512,
) -> dict[str, int]:
    """Remove old log files, preserving today's directory."""

    started = _now()
    deleted = 0
    deleted_bytes = 0
    log_event("log.cleanup.started", component="logging")
    try:
        directories = sorted(path for path in LOG_DIR.glob("????-??-??") if path.is_dir())
        today = date_log_dir(started).name
        for directory in directories:
            if directory.name == today:
                continue
            try:
                day = datetime.strptime(directory.name, "%Y-%m-%d").replace(
                    tzinfo=started.tzinfo or timezone.utc
                )
            except ValueError:
                continue
            for path in directory.iterdir():
                if not path.is_file():
                    continue
                limit = error_retention_days if path.name == "errors.log" else retention_days
                if started - day <= timedelta(days=max(1, int(limit))):
                    continue
                try:
                    size = path.stat().st_size
                    path.unlink()
                    deleted += 1
                    deleted_bytes += size
                except OSError as exc:
                    log_event(
                        "log.cleanup.failed",
                        level=logging.WARNING,
                        component="logging",
                        path=str(path),
                        error_type=type(exc).__name__,
                        error_message=str(exc),
                    )
            try:
                if not any(directory.iterdir()):
                    directory.rmdir()
            except OSError:
                pass
        total_limit = max(1, int(max_total_size_mb)) * 1024 * 1024
        files = sorted(
            (
                path
                for directory in LOG_DIR.glob("????-??-??")
                if directory.is_dir()
                for path in directory.iterdir()
                if path.is_file()
            ),
            key=lambda path: path.stat().st_mtime,
        )
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total <= total_limit or path.parent.name == today or path.name == "errors.log":
                continue
            try:
                size = path.stat().st_size
                path.unlink()
                total -= size
                deleted += 1
                deleted_bytes += size
            except OSError:
                continue
        result = {"deleted_files": deleted, "deleted_bytes": deleted_bytes}
        log_event("log.cleanup.deleted", component="logging", **result)
        return result
    except Exception as exc:
        log_event(
            "log.cleanup.failed",
            level=logging.WARNING,
            component="logging",
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return {"deleted_files": deleted, "deleted_bytes": deleted_bytes}
