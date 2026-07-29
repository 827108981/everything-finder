from __future__ import annotations

import atexit
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from local_full_text_search.config.constants import CACHE_DIR, TEMP_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.ooxml.docx_stream_parser import DocxStreamParser
from local_full_text_search.parsers.ooxml.pptx_stream_parser import PptxStreamParser
from local_full_text_search.parsers.ooxml.xlsx_stream_parser import XlsxStreamParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ConversionResult:
    path: Path | None = None
    status: str = "success"
    error_code: str | None = None
    message: str | None = None


class LegacyOfficeParser(BaseParser):
    """Convert old Office formats in a reusable, process-local session."""

    name = "legacy_office"

    def __init__(
        self,
        *,
        conversion_cache: bool = True,
        fast_ooxml: bool = True,
    ) -> None:
        super().__init__()
        self.conversion_cache = bool(conversion_cache)
        self.fast_ooxml = bool(fast_ooxml)
        self._office_session = OfficeConversionSession()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".doc", ".xls", ".ppt"}

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        cancel_token.throw_if_cancelled()
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        if not _looks_like_ole(file_path):
            self.set_status(
                "failed",
                "LEGACY_INVALID_FORMAT",
                "文件扩展名是老版 Office 格式，但文件头不是有效的 OLE 文档",
            )
            return
        with tempfile.TemporaryDirectory(prefix="legacy_office_", dir=TEMP_DIR) as temp_name:
            self.report_progress(
                "legacy_cache_lookup",
                completed=0,
                total=max(1, file_path.stat().st_size),
                unit_type="bytes",
                detail=file_path.name,
            )
            converted = self._cached_conversion(file_path)
            if converted is None:
                result = self._convert(file_path, Path(temp_name))
                converted = result.path
                if converted is not None:
                    converted = self._store_cached_conversion(file_path, converted)
                else:
                    self.set_status(
                        result.status,
                        result.error_code,
                        result.message,
                    )
                    return
            else:
                self.report_progress(
                    "legacy_cache_hit",
                    completed=1,
                    total=1,
                    unit_type="file",
                    detail=converted.name,
                )
            self.report_progress(
                "legacy_parse_converted",
                completed=0,
                total=max(1, converted.stat().st_size),
                unit_type="bytes",
                detail=converted.name,
            )
            parser = self._parser_for_converted(converted)
            for block in parser.parse(converted, cancel_token):
                block.file_path = str(file_path)
                block.location_text = f"转换自 {file_path.suffix.lower()} > {block.location_text}"
                yield block
            if parser.last_status != "success":
                self.set_status(parser.last_status, parser.last_error_code, parser.last_error_message)

    def _convert(self, file_path: Path, temp_dir: Path) -> ConversionResult:
        target = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
        self.report_progress(
            "legacy_office_open",
            completed=0,
            total=1,
            unit_type="file",
            detail=file_path.name,
        )
        office_result = self._convert_with_office(file_path, target)
        if office_result.path is not None:
            return office_result
        self.report_progress(
            "legacy_libreoffice_convert",
            completed=0,
            total=1,
            unit_type="file",
            detail=file_path.name,
        )
        libre_result = self._convert_with_libreoffice(file_path, temp_dir)
        if libre_result.path is not None:
            candidate = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
            if candidate.exists():
                return ConversionResult(path=candidate)
        if office_result.error_code != "CONVERTER_MISSING":
            return office_result
        return libre_result

    def _convert_with_office(self, source: Path, target: Path) -> ConversionResult:
        return self._office_session.convert(source, target)

    def _convert_with_libreoffice(self, source: Path, output_dir: Path) -> ConversionResult:
        soffice = _find_soffice()
        if soffice is None:
            return ConversionResult(
                status="converter_missing",
                error_code="CONVERTER_MISSING",
                message="未检测到 Microsoft Office、WPS Office 或 LibreOffice 转换器",
            )
        process: subprocess.Popen[str] | None = None
        registry_path: Path | None = None
        try:
            process = subprocess.Popen(
                [
                    str(soffice),
                    "--headless",
                    "--convert-to",
                    self._libreoffice_target(source),
                    "--outdir",
                    str(output_dir),
                    str(source),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            registry_path = _register_office_process("soffice", int(process.pid))
            stdout, stderr = process.communicate()
        except OSError as exc:
            return ConversionResult(
                status="failed_retryable",
                error_code="LIBREOFFICE_START_FAILED",
                message=f"LibreOffice 启动失败：{exc}",
            )
        finally:
            if registry_path is not None:
                registry_path.unlink(missing_ok=True)
        candidate = output_dir / f"{source.stem}{self._target_suffix(source)}"
        if process is not None and process.returncode == 0 and candidate.is_file():
            return ConversionResult(path=candidate)
        detail = ((stderr if "stderr" in locals() else "") or (stdout if "stdout" in locals() else "")).strip()
        return ConversionResult(
            status="failed_retryable",
            error_code="LIBREOFFICE_CONVERSION_FAILED",
            message=f"LibreOffice 转换失败：{detail or '未生成目标文件'}",
        )

    @staticmethod
    def _target_suffix(source: Path) -> str:
        return {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}[source.suffix.lower()]

    @staticmethod
    def _libreoffice_target(source: Path) -> str:
        return {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}[source.suffix.lower()]

    def _parser_for_converted(self, converted: Path) -> BaseParser:
        suffix = converted.suffix.lower()
        if suffix == ".docx":
            fallback = DocxParser()
            return DocxStreamParser(fallback, defer_normalization=True) if self.fast_ooxml else fallback
        if suffix == ".xlsx":
            fallback = XlsxParser()
            return XlsxStreamParser(fallback, defer_normalization=True) if self.fast_ooxml else fallback
        if suffix == ".pptx":
            fallback = PptxParser()
            return PptxStreamParser(fallback, defer_normalization=True) if self.fast_ooxml else fallback
        raise ValueError(f"未知转换格式: {converted}")

    def _cache_path(self, source: Path) -> Path:
        digest = hashlib.sha256()
        total = max(1, source.stat().st_size)
        completed = 0
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                completed += len(chunk)
                self.report_progress(
                    "legacy_cache_hash",
                    completed=completed,
                    total=total,
                    unit_type="bytes",
                    detail=source.name,
                )
        return CACHE_DIR / "legacy_conversion" / f"{digest.hexdigest()}{self._target_suffix(source)}"

    def _cached_conversion(self, source: Path) -> Path | None:
        if not self.conversion_cache:
            return None
        target = self._cache_path(source)
        return target if target.is_file() else None

    def _store_cached_conversion(self, source: Path, converted: Path) -> Path:
        if not self.conversion_cache:
            return converted
        target = self._cache_path(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        shutil.copy2(converted, temporary)
        temporary.replace(target)
        return target


class OfficeConversionSession:
    """Own COM applications in the worker process that initialized them."""

    def __init__(self) -> None:
        self._apps: dict[tuple[str, str], object] = {}
        self._job_handles: dict[tuple[str, str], object] = {}
        self._registry_paths: dict[tuple[str, str], Path] = {}
        self._com_initialized = False
        atexit.register(self.close)

    def convert(self, source: Path, target: Path) -> ConversionResult:
        kind = source.suffix.lower()
        prog_ids = {
            ".doc": ("Word.Application", "KWPS.Application"),
            ".xls": ("Excel.Application", "KET.Application"),
            ".ppt": ("PowerPoint.Application", "KWPP.Application"),
        }.get(kind, ())
        application_errors: list[str] = []
        conversion_errors: list[ConversionResult] = []
        for prog_id in prog_ids:
            key = (kind, prog_id)
            try:
                app = self._application(kind, prog_id)
            except Exception as exc:
                application_errors.append(f"{prog_id}: {exc}")
                self._reset(key)
                continue
            result = self._convert_with_application(app, kind, source, target, prog_id)
            if result.path is not None:
                return result
            conversion_errors.append(result)
            self._reset(key)
        if conversion_errors:
            return conversion_errors[-1]
        return ConversionResult(
            status="converter_missing",
            error_code="CONVERTER_MISSING",
            message=(
                "Microsoft Office/WPS 自动化组件不可用"
                + (f"：{'；'.join(application_errors)}" if application_errors else "")
            ),
        )

    @staticmethod
    def _convert_with_application(
        app: object,
        kind: str,
        source: Path,
        target: Path,
        prog_id: str,
    ) -> ConversionResult:
        opened: object | None = None
        try:
            if kind == ".doc":
                opened = app.Documents.Open(str(source), ReadOnly=True)
            elif kind == ".xls":
                opened = app.Workbooks.Open(str(source), ReadOnly=True)
            elif kind == ".ppt":
                opened = app.Presentations.Open(str(source), WithWindow=False)
            else:
                return ConversionResult(
                    status="failed",
                    error_code="LEGACY_EXTENSION_UNSUPPORTED",
                    message=f"不支持转换格式：{kind}",
                )
        except Exception as exc:
            return ConversionResult(
                status="failed_retryable",
                error_code="LEGACY_OPEN_FAILED",
                message=f"{prog_id} 无法打开文档：{exc}",
            )
        try:
            if kind == ".doc":
                save_as = getattr(opened, "SaveAs2", None) or getattr(opened, "SaveAs")
                save_as(str(target), FileFormat=16)
            elif kind == ".xls":
                opened.SaveAs(str(target), FileFormat=51)
            else:
                opened.SaveAs(str(target), 24)
        except Exception as exc:
            return ConversionResult(
                status="failed_retryable",
                error_code="LEGACY_SAVE_FAILED",
                message=f"{prog_id} 无法保存转换文件：{exc}",
            )
        finally:
            if opened is not None:
                try:
                    opened.Close(False) if kind != ".ppt" else opened.Close()
                except Exception:
                    logger.debug("Unable to close converted legacy Office document", exc_info=True)
        if not target.is_file():
            return ConversionResult(
                status="failed_retryable",
                error_code="LEGACY_OUTPUT_MISSING",
                message=f"{prog_id} 已执行转换，但没有生成目标文件",
            )
        return ConversionResult(path=target)

    def _application(self, kind: str, prog_id: str) -> object:
        key = (kind, prog_id)
        if key in self._apps:
            return self._apps[key]
        import pythoncom
        import win32com.client

        if not self._com_initialized:
            pythoncom.CoInitialize()
            self._com_initialized = True
        previous_pids = _office_process_ids(kind, prog_id)
        app = win32com.client.DispatchEx(prog_id)
        if kind == ".doc":
            app.Visible = False
            app.DisplayAlerts = 0
        elif kind == ".xls":
            app.Visible = False
            app.DisplayAlerts = False
        self._bind_application_lifetime(key, app, previous_pids)
        self._apps[key] = app
        return app

    def _bind_application_lifetime(
        self,
        key: tuple[str, str],
        app: object,
        previous_pids: set[int],
    ) -> None:
        if os.name != "nt":
            return
        kind, prog_id = key
        try:
            process_id = _wait_for_new_office_process(kind, prog_id, previous_pids)
            if process_id is None:
                hwnd = 0
                for attribute in ("Hwnd", "HWND"):
                    try:
                        hwnd = int(getattr(app, attribute, 0) or 0)
                    except Exception:
                        continue
                    if hwnd:
                        break
                if not hwnd:
                    return
                import win32process

                _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            self._registry_paths[key] = _register_office_process(prog_id, int(process_id))
            job = _create_kill_on_close_job(int(process_id))
            if job is not None:
                self._job_handles[key] = job
        except Exception:
            logger.debug("Unable to bind Office process lifetime for %s", prog_id, exc_info=True)

    def _reset(self, key: tuple[str, str]) -> None:
        app = self._apps.pop(key, None)
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        job = self._job_handles.pop(key, None)
        if job is not None:
            try:
                import win32api

                win32api.CloseHandle(job)
            except Exception:
                logger.debug("Unable to close Office job handle", exc_info=True)
        registry_path = self._registry_paths.pop(key, None)
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)

    def close(self) -> None:
        for key in list(self._apps):
            self._reset(key)
        if self._com_initialized:
            try:
                import pythoncom

                pythoncom.CoUninitialize()
            except Exception:
                pass
            self._com_initialized = False


def _create_kill_on_close_job(process_id: int) -> object | None:
    """Attach one automation process to a private kill-on-close Windows job."""

    if os.name != "nt" or process_id <= 0:
        return None
    import win32api
    import win32con
    import win32job

    job = win32job.CreateJobObject(None, "")
    process_handle = None
    try:
        info = win32job.QueryInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
        )
        info["BasicLimitInformation"]["LimitFlags"] |= (
            win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        win32job.SetInformationJobObject(
            job,
            win32job.JobObjectExtendedLimitInformation,
            info,
        )
        access = (
            win32con.PROCESS_TERMINATE
            | win32con.PROCESS_SET_QUOTA
            | win32con.PROCESS_QUERY_INFORMATION
        )
        process_handle = win32api.OpenProcess(access, False, process_id)
        win32job.AssignProcessToJobObject(job, process_handle)
        return job
    except Exception:
        win32api.CloseHandle(job)
        raise
    finally:
        if process_handle is not None:
            win32api.CloseHandle(process_handle)


def _register_office_process(kind: str, process_id: int) -> Path:
    registry_dir = Path(os.environ.get("LFTS_PROCESS_REGISTRY_DIR", str(TEMP_DIR)))
    target_dir = registry_dir / "office_processes"
    target_dir.mkdir(parents=True, exist_ok=True)
    create_time = 0.0
    try:
        import psutil

        create_time = float(psutil.Process(process_id).create_time())
    except Exception:
        logger.debug("Unable to read Office process creation time", exc_info=True)
    target = target_dir / f"{process_id}.json"
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"pid": process_id, "create_time": create_time, "kind": kind},
            ensure_ascii=True,
        ),
        encoding="ascii",
    )
    temporary.replace(target)
    return target


def _office_process_ids(kind: str, prog_id: str) -> set[int]:
    is_wps = prog_id.upper().startswith("K")
    process_name = (
        {
            ".doc": "wps.exe",
            ".xls": "et.exe",
            ".ppt": "wpp.exe",
        }
        if is_wps
        else {
            ".doc": "winword.exe",
            ".xls": "excel.exe",
            ".ppt": "powerpnt.exe",
        }
    )[kind]
    try:
        import psutil

        return {
            int(process.pid)
            for process in psutil.process_iter(["name"])
            if str(process.info.get("name") or "").lower() == process_name
        }
    except Exception:
        logger.debug("Unable to snapshot Office processes", exc_info=True)
        return set()


def _wait_for_new_office_process(
    kind: str,
    prog_id: str,
    previous_pids: set[int],
    timeout_seconds: float = 5.0,
) -> int | None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        candidates = _office_process_ids(kind, prog_id) - previous_pids
        if candidates:
            try:
                import psutil

                return max(candidates, key=lambda pid: psutil.Process(pid).create_time())
            except Exception:
                return max(candidates)
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.05)


def cleanup_registered_office_processes(registry_dir: Path) -> None:
    target_dir = registry_dir / "office_processes"
    if not target_dir.is_dir():
        return
    try:
        import psutil
    except ImportError:
        return
    allowed_names = {
        "winword.exe",
        "excel.exe",
        "powerpnt.exe",
        "wps.exe",
        "et.exe",
        "wpp.exe",
        "soffice.exe",
        "soffice.bin",
    }
    for record in target_dir.glob("*.json"):
        try:
            payload = json.loads(record.read_text(encoding="ascii"))
            process = psutil.Process(int(payload["pid"]))
            expected_time = float(payload.get("create_time") or 0.0)
            if process.name().lower() not in allowed_names:
                continue
            if expected_time and abs(process.create_time() - expected_time) > 1.0:
                continue
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except psutil.TimeoutExpired:
                process.kill()
        except (OSError, ValueError, KeyError, json.JSONDecodeError, psutil.Error):
            pass
        finally:
            record.unlink(missing_ok=True)


def _looks_like_ole(source: Path) -> bool:
    try:
        with source.open("rb") as stream:
            return stream.read(8) == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
    except OSError:
        return False


def _find_soffice() -> Path | None:
    found = shutil.which("soffice")
    if found:
        return Path(found)
    candidates = [
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
    ]
    return next((path for path in candidates if path.exists()), None)
