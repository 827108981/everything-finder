from __future__ import annotations

import ctypes
import faulthandler
import json
import os
import platform
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import TextIO

from local_full_text_search.version import __version__


class StartupDiagnostics:
    """Record startup progress before Qt and application dependencies are ready."""

    def __init__(
        self,
        *,
        app_name: str = "LocalFullTextSearch",
        app_version: str = __version__,
        timeout_seconds: float = 15.0,
        base_dir: Path | None = None,
        settings_path: Path | None = None,
        database_path: Path | None = None,
    ) -> None:
        root = base_dir or _default_base_dir(app_name)
        self.app_name = app_name
        self.app_version = app_version
        self.root = root
        self.log_path = root / "logs" / "startup.log"
        self.state_path = root / "logs" / "startup-state.json"
        self.settings_path = settings_path or root / "config" / "settings.json"
        self.database_path = database_path or root / "data" / "search_index.db"
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.current_stage = "Python 初始化"
        self._started_at = time.perf_counter()
        self._stage_started_at = self._started_at
        self._timer: threading.Timer | None = None
        self._complete = False
        self._lock = threading.Lock()
        self._fault_stream: TextIO | None = None
        self._last_status = "created"
        self.previous_failure: dict[str, object] | None = None

    def begin(self) -> None:
        self._prepare_paths()
        self._enable_faulthandler()
        self.previous_failure = self._read_incomplete_previous_state()
        self._write(
            "START",
            {
                "argv": sys.argv,
                "python": sys.version,
            },
        )
        self._write_state("running")
        self._install_exception_hooks()

    def stage_started(self, name: str) -> None:
        with self._lock:
            self._cancel_timer_locked()
            self.current_stage = name
            self._stage_started_at = time.perf_counter()
            self._write("STAGE_STARTED", {"stage": name})
            self._write_state("running")
            timer = threading.Timer(self.timeout_seconds, self._report_slow_stage, args=(name,))
            timer.daemon = True
            self._timer = timer
            timer.start()

    def stage_completed(self, name: str | None = None) -> None:
        with self._lock:
            stage = name or self.current_stage
            elapsed_ms = int((time.perf_counter() - self._stage_started_at) * 1000)
            self._cancel_timer_locked()
            self._write("STAGE_COMPLETED", {"stage": stage, "elapsed_ms": elapsed_ms})

    def stage_failed(self, name: str | None, exc: BaseException) -> None:
        with self._lock:
            stage = name or self.current_stage
            elapsed_ms = int((time.perf_counter() - self._stage_started_at) * 1000)
            self._cancel_timer_locked()
            self._write(
                "STAGE_FAILED",
                {
                    "stage": stage,
                    "elapsed_ms": elapsed_ms,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                    "error_code": classify_startup_error(exc, stage)[0],
                    "suggestion": classify_startup_error(exc, stage)[1],
                    "traceback": "".join(traceback.format_exception(exc)),
                },
            )
            self._write_state("failed", stage=stage, error=f"{exc.__class__.__name__}: {exc}")

    def mark_window_visible(self) -> None:
        with self._lock:
            self._complete = True
            self._cancel_timer_locked()
            self._write(
                "WINDOW_VISIBLE",
                {"elapsed_ms": int((time.perf_counter() - self._started_at) * 1000)},
            )
            self._write_state("complete")

    def mark_completed(self) -> None:
        """Finish a non-UI validation command without recording a false failure."""

        with self._lock:
            self._complete = True
            self._cancel_timer_locked()
            self._write(
                "STARTUP_COMPLETED",
                {"elapsed_ms": int((time.perf_counter() - self._started_at) * 1000)},
            )
            self._write_state("complete")

    def format_error_message(self, exc: BaseException) -> str:
        error_code, suggestion = classify_startup_error(exc, self.current_stage)
        return (
            f"启动失败：{self.current_stage}\n\n"
            f"错误分类：{error_code}\n"
            f"{exc.__class__.__name__}: {str(exc) or '未提供详细信息'}\n\n"
            f"建议：{suggestion}\n\n"
            f"诊断日志：{self.log_path}"
        )

    def show_native_error(self, title: str, message: str) -> None:
        try:
            if os.name == "nt":
                ctypes.windll.user32.MessageBoxW(None, message, title, 0x10)
                return
        except Exception:
            pass
        print(f"{title}\n{message}", file=sys.stderr)

    def _prepare_paths(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8"):
                pass
        except OSError:
            fallback = Path(tempfile.gettempdir()) / "LocalFullTextSearch"
            self.root = fallback
            self.log_path = fallback / "startup.log"
            self.state_path = fallback / "startup-state.json"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8"):
                pass

    def _read_incomplete_previous_state(self) -> dict[str, object] | None:
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if isinstance(data, dict) and data.get("status") in {"running", "failed"}:
            return data
        return None

    def _write_state(
        self,
        status: str,
        *,
        stage: str | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "status": status,
            "stage": stage or self.current_stage,
            "error": error,
            "updated_at": time.time(),
            "log_path": str(self.log_path),
        }
        try:
            temporary = self.state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(self.state_path)
        except OSError:
            pass

    def _write(self, event: str, payload: dict[str, object]) -> None:
        self._last_status = {
            "START": "running",
            "STAGE_STARTED": "running",
            "STAGE_COMPLETED": "running",
            "STAGE_SLOW": "slow",
            "STAGE_FAILED": "failed",
            "WINDOW_VISIBLE": "complete",
            "STARTUP_COMPLETED": "complete",
        }.get(event, self._last_status)
        record = {
            "timestamp": time.time(),
            "event": event,
            **self._common_metadata(),
            **payload,
        }
        try:
            with self.log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError:
            pass

    def _common_metadata(self) -> dict[str, object]:
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "executable": sys.executable,
            "cwd": os.getcwd(),
            "frozen": bool(getattr(sys, "frozen", False)),
            "platform": platform.platform(),
            "windows_version": platform.version() if os.name == "nt" else "",
            "settings_path": str(self.settings_path),
            "database_path": str(self.database_path),
            "log_path": str(self.log_path),
            "last_status": self._last_status,
        }

    def _enable_faulthandler(self) -> None:
        try:
            self._fault_stream = self.log_path.open("a", encoding="utf-8")
            faulthandler.enable(file=self._fault_stream, all_threads=True)
            self._write("FAULTHANDLER_ENABLED", {})
        except (OSError, RuntimeError):
            if self._fault_stream is not None:
                self._fault_stream.close()
                self._fault_stream = None

    def _install_exception_hooks(self) -> None:
        previous_sys_hook = sys.excepthook

        def report_sys_exception(exc_type: type[BaseException], exc: BaseException, tb: object) -> None:
            self.stage_failed(self.current_stage, exc)
            previous_sys_hook(exc_type, exc, tb)

        sys.excepthook = report_sys_exception
        if hasattr(threading, "excepthook"):
            previous_thread_hook = threading.excepthook

            def report_thread_exception(args: threading.ExceptHookArgs) -> None:
                self.stage_failed(self.current_stage, args.exc_value)
                previous_thread_hook(args)

            threading.excepthook = report_thread_exception

    def _report_slow_stage(self, expected_stage: str) -> None:
        with self._lock:
            if self._complete or self.current_stage != expected_stage:
                return
            elapsed_seconds = int(time.perf_counter() - self._stage_started_at)
            self._write(
                "STAGE_SLOW",
                {"stage": expected_stage, "elapsed_seconds": elapsed_seconds},
            )
        self.show_native_error(
            "本地全文搜索仍在启动",
            f"当前阶段：{expected_stage}\n"
            f"已等待约 {elapsed_seconds} 秒。\n\n"
            f"程序仍在尝试启动；如持续无窗口，请提供诊断日志：\n{self.log_path}",
        )

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def _default_base_dir(app_name: str) -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / app_name if base else Path.home() / f".{app_name}"


def classify_startup_error(
    exc: BaseException,
    stage: str = "",
) -> tuple[str, str]:
    message = str(exc).lower()
    stage_text = stage.lower()
    if isinstance(exc, ModuleNotFoundError) or (
        isinstance(exc, FileNotFoundError)
        and any(word in stage_text for word in ("模块", "图形", "module", "qt"))
    ):
        return (
            "CRITICAL_COMPONENT_MISSING",
            "请重新解压完整程序目录，确保 EXE 与 _internal 未被拆分或拦截。",
        )
    if isinstance(exc, PermissionError) or (
        any(word in message for word in ("permission denied", "access is denied", "只读"))
        and any(word in stage_text for word in ("设置", "数据库", "data", "config"))
    ):
        return (
            "USER_DATA_NOT_WRITABLE",
            "请确认本机用户数据目录可写，并避免从只读目录或压缩包预览中运行。",
        )
    if isinstance(exc, sqlite3.Error) or "database" in message:
        if any(word in message for word in ("locked", "busy", "占用")):
            return (
                "DATABASE_LOCKED",
                "请关闭其他程序实例后重试，并将启动日志提供给支持人员。",
            )
        if any(
            word in message
            for word in ("malformed", "corrupt", "not a database", "disk image")
        ):
            return (
                "DATABASE_CORRUPT",
                "请保留数据库和日志以便恢复；不要直接删除现有索引文件。",
            )
        return (
            "DATABASE_MIGRATION_FAILED",
            "请保留数据库备份和启动日志，由支持人员检查迁移失败原因。",
        )
    if isinstance(exc, (ImportError, OSError)) and any(
        word in message
        for word in ("dll", "pyside", "qt", "platform plugin", "module")
    ):
        return (
            "QT_OR_DLL_LOAD_FAILED",
            "请确认程序目录完整，并检查安全软件是否隔离了 DLL 或 Qt 文件。",
        )
    return (
        "UNKNOWN_STARTUP_ERROR",
        "请保留当前提示和启动日志，并提供给支持人员进一步定位。",
    )
