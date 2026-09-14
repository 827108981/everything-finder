from __future__ import annotations

import ctypes
import faulthandler
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import TextIO

from local_full_text_search.version import __version__


def frozen_runtime_issues() -> list[str]:
    """Return missing files that would prevent a frozen child process from starting.

    PyInstaller's multiprocessing child starts in the bootloader, before Python
    logging is available.  A partially copied or cleaned one-folder package can
    therefore only show the generic ``Failed to start embedded python
    interpreter`` dialog.  Checking the small set of boot-critical artifacts in
    the parent process lets the UI report an actionable error instead.
    """

    if not bool(getattr(sys, "frozen", False)):
        return []
    issues: list[str] = []
    executable = Path(sys.executable).resolve()
    if not executable.is_file():
        issues.append(f"主程序 {executable.name}")
    runtime_dir_value = getattr(sys, "_MEIPASS", None)
    runtime_dir = Path(runtime_dir_value).resolve() if runtime_dir_value else None
    if runtime_dir is None or not runtime_dir.is_dir():
        issues.append("冻结运行目录 _internal")
        return issues
    required = [
        (runtime_dir / "base_library.zip", "Python 标准库"),
        (runtime_dir / "local_full_text_search", "应用模块"),
        (runtime_dir / "PySide6", "Qt 运行库"),
    ]
    python_runtime = next(
        (
            candidate
            for candidate in (
                runtime_dir / "python311.dll",
                runtime_dir / "python3.dll",
            )
            if candidate.is_file()
        ),
        None,
    )
    if python_runtime is None:
        issues.append("Python 嵌入运行库")
    for path, label in required:
        if not path.exists():
            issues.append(label)
    return issues


def frozen_runtime_error_message(issues: list[str]) -> str:
    missing = "、".join(issues) if issues else "关键运行组件"
    return (
        "更新启动失败：冻结版运行组件不完整，缺少 "
        f"{missing}。请退出程序并重新安装完整发布目录，不要只复制 EXE。"
    )


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
        self.process_role = "gui"
        # The numeric parent PID alone is not enough on Windows: an installer,
        # an IDE launcher, Restart Manager, or another copy of the application
        # can all be the parent.  Capture the identity at startup so a later
        # exit record can be correlated with the process that launched us.
        self.parent_process = _process_snapshot(os.getppid())
        self._monitor_process: subprocess.Popen[bytes] | None = None

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

    def start_external_monitor(self) -> None:
        """Start a detached helper that can record a hard process termination."""

        if "--monitor-parent" in sys.argv or self._monitor_process is not None:
            return
        output_path = self.log_path.parent / "process-exit.jsonl"
        try:
            if bool(getattr(sys, "frozen", False)):
                command = [
                    sys.executable,
                    "--monitor-parent",
                    str(os.getpid()),
                    str(output_path),
                    str(self.state_path),
                ]
            else:
                command = [
                    sys.executable,
                    str(Path(sys.argv[0]).resolve()),
                    "--monitor-parent",
                    str(os.getpid()),
                    str(output_path),
                    str(self.state_path),
                ]
            creation_flags = 0
            if os.name == "nt":
                creation_flags = 0x00000008 | 0x08000000  # DETACHED_PROCESS | CREATE_NO_WINDOW
            self._monitor_process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                creationflags=creation_flags,
            )
            self.runtime_event(
                "PROCESS_MONITOR_STARTED",
                monitor_pid=self._monitor_process.pid,
                monitor_output=str(output_path),
            )
        except Exception as exc:
            self._monitor_process = None
            self.runtime_event(
                "PROCESS_MONITOR_START_FAILED",
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )

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
            # Showing the window is not the end of the application's
            # lifetime.  The first background scan is deliberately started
            # after this point, so marking startup complete here used to hide
            # the exact failure window when the process disappeared during
            # scan initialization.
            self.current_stage = "运行中"
            self._write(
                "WINDOW_VISIBLE",
                {
                    "elapsed_ms": int((time.perf_counter() - self._started_at) * 1000),
                    "startup_complete": False,
                },
            )
            self._write_state("running", stage=self.current_stage)

    def runtime_event(self, event: str, **fields: object) -> None:
        """Record a post-window lifecycle event without changing the stage timer."""

        with self._lock:
            if self._complete:
                return
            self._write(str(event), dict(fields))
            self._write_state("running", stage=self.current_stage)

    def verify_crash_dump_configuration(self) -> None:
        """Record whether Windows native dump capture is available."""

        dump_dir = (
            Path(os.environ.get("LOCALAPPDATA", tempfile.gettempdir()))
            / "LocalFullTextSearch"
            / "crash-dumps"
        )
        registry_values: dict[str, object] = {}
        try:
            dump_dir.mkdir(parents=True, exist_ok=True)
            probe = dump_dir / f".write-probe-{os.getpid()}"
            probe.write_bytes(b"ok")
            probe.unlink(missing_ok=True)
            if os.name == "nt":
                import winreg

                key_path = (
                    "SOFTWARE\\Microsoft\\Windows\\Windows Error Reporting\\LocalDumps\\"
                    + Path(sys.executable).name
                )
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                        for name in ("DumpFolder", "DumpType", "DumpCount"):
                            try:
                                registry_values[name] = winreg.QueryValueEx(key, name)[0]
                            except OSError:
                                registry_values[name] = None
                except OSError:
                    registry_values["key"] = "missing"
        except Exception as exc:
            self.runtime_event(
                "CRASH_DUMP_CONFIGURATION_FAILED",
                dump_dir=str(dump_dir),
                error_type=type(exc).__name__,
                error_message=str(exc)[:1000],
            )
            return
        self.runtime_event(
            "CRASH_DUMP_CONFIGURATION",
            dump_dir=str(dump_dir),
            executable_name=Path(sys.executable).name,
            registry_values=registry_values,
            writable=True,
        )

    def mark_interrupted(
        self,
        source: str = "IDE_OR_CONSOLE_INTERRUPT",
        *,
        stage: str | None = None,
        exit_code: int = 130,
    ) -> None:
        """Record an operator/IDE interrupt without calling it a clean exit."""

        with self._lock:
            if self._complete:
                return
            self._complete = True
            self._cancel_timer_locked()
            self._write(
                "APP_INTERRUPTED",
                {
                    "source": str(source),
                    "stage": stage or self.current_stage,
                    "exit_code": int(exit_code),
                    "elapsed_ms": int((time.perf_counter() - self._started_at) * 1000),
                },
            )
            self._write_state("interrupted", stage=stage or self.current_stage)

    def mark_clean_exit(
        self,
        reason: str = "event_loop_returned",
        *,
        exit_code: int | None = None,
    ) -> None:
        """Mark the process as normally terminated after all runtime work ended."""

        with self._lock:
            if self._complete:
                return
            self._complete = True
            self._cancel_timer_locked()
            payload: dict[str, object] = {
                "reason": str(reason),
                "elapsed_ms": int((time.perf_counter() - self._started_at) * 1000),
            }
            if exit_code is not None:
                payload["exit_code"] = int(exit_code)
            self._write("APP_CLEAN_EXIT", payload)
            self._write_state("complete", stage="已正常退出")

    def mark_completed(self) -> None:
        """Finish a non-UI validation command without recording a false failure."""

        self.mark_clean_exit("command_completed")

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
        if isinstance(data, dict) and data.get("status") in {
            "running",
            "failed",
            "interrupted",
        }:
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
            "pid": os.getpid(),
            "parent_pid": os.getppid(),
            "process_role": self.process_role,
            "parent_process": self.parent_process,
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
            "WINDOW_VISIBLE": "running",
            "APP_CLEAN_EXIT": "complete",
            "APP_INTERRUPTED": "interrupted",
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
            "parent_pid": os.getppid(),
            "parent_process": self.parent_process,
            "process_role": self.process_role,
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
        # A slow stage is diagnostic information, not a startup failure.  A
        # modal native dialog hides the GUI and makes a recoverable startup
        # look broken.  The UI reports the non-blocking status after it shows.

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def _default_base_dir(app_name: str) -> Path:
    isolated = os.environ.get("LFTS_APP_DATA_DIR")
    if isolated:
        return Path(isolated).resolve()
    base = os.environ.get("LOCALAPPDATA")
    return Path(base) / app_name if base else Path.home() / f".{app_name}"


def _process_snapshot(pid: int) -> dict[str, object]:
    """Return best-effort identity data for a process that launched us."""

    result: dict[str, object] = {"pid": int(pid)}
    if int(pid) <= 0:
        result["status"] = "unknown"
        return result
    try:
        import psutil

        process = psutil.Process(int(pid))
        result.update(
            {
                "name": process.name(),
                "exe": process.exe(),
                "create_time": process.create_time(),
                "cmdline": process.cmdline()[:8],
                "status": process.status(),
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "unknown",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:300],
            }
        )
    return result


def run_process_monitor(
    parent_pid: int,
    output_path: Path,
    state_path: Path,
) -> int:
    """Wait outside the GUI process and persist its real exit code."""

    exit_code: int | None = None
    handle = None
    try:
        if os.name == "nt":
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            process_synchronize = 0x00100000
            process_query_limited_information = 0x1000
            kernel32.OpenProcess.restype = ctypes.c_void_p
            handle = kernel32.OpenProcess(
                process_synchronize | process_query_limited_information,
                False,
                int(parent_pid),
            )
            if handle:
                kernel32.WaitForSingleObject(handle, 0xFFFFFFFF)
                value = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(value)):
                    exit_code = int(value.value)
        else:
            _, status = os.waitpid(int(parent_pid), 0)
            exit_code = os.waitstatus_to_exitcode(status)
    except Exception as exc:
        monitor_error = f"{type(exc).__name__}: {exc}"
    else:
        monitor_error = ""
    finally:
        if handle:
            try:
                ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)
            except Exception:
                pass

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        state = {}
    if not isinstance(state, dict):
        state = {}
    clean = state.get("status") == "complete"
    interrupted = state.get("status") == "interrupted"
    if clean and exit_code == 0:
        classification = "clean_exit"
    elif interrupted or exit_code in {0xC000013A, 130}:
        classification = "operator_or_ide_interrupt"
    else:
        classification = "abrupt_process_exit"
    record = {
        "timestamp": time.time(),
        "event": "PROCESS_EXIT_OBSERVED",
        "parent_pid": int(parent_pid),
        "exit_code": exit_code,
        "exit_code_hex": f"0x{exit_code:08X}" if exit_code is not None else None,
        "classification": classification,
        "clean_exit_marker": bool(clean),
        "last_state": state,
        "monitor_error": monitor_error,
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass
    return 0


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
