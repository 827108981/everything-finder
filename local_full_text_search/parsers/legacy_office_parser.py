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


class LegacyOfficeParser(BaseParser):
    """Convert old Office formats in a reusable, process-local session."""

    name = "legacy_office"

    def __init__(
        self,
        timeout_seconds: int = 120,
        *,
        conversion_cache: bool = True,
        fast_ooxml: bool = True,
    ) -> None:
        super().__init__()
        self.timeout_seconds = max(1, int(timeout_seconds))
        self.conversion_cache = bool(conversion_cache)
        self.fast_ooxml = bool(fast_ooxml)
        self._office_session = OfficeConversionSession()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".doc", ".xls", ".ppt"}

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        cancel_token.throw_if_cancelled()
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="legacy_office_", dir=TEMP_DIR) as temp_name:
            converted = self._cached_conversion(file_path)
            if converted is None:
                converted = self._convert(file_path, Path(temp_name))
                if converted is not None:
                    converted = self._store_cached_conversion(file_path, converted)
            if converted is None:
                self.set_status(
                    "converter_missing",
                    "CONVERTER_MISSING",
                    "未找到可用转换器，或文件不是有效的老版 Office 文档",
                )
                return
            parser = self._parser_for_converted(converted)
            for block in parser.parse(converted, cancel_token):
                block.file_path = str(file_path)
                block.location_text = f"转换自 {file_path.suffix.lower()} > {block.location_text}"
                yield block
            if parser.last_status != "success":
                self.set_status(parser.last_status, parser.last_error_code, parser.last_error_message)

    def _convert(self, file_path: Path, temp_dir: Path) -> Path | None:
        target = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
        if self._convert_with_office(file_path, target):
            return target
        if self._convert_with_libreoffice(file_path, temp_dir):
            candidate = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
            if candidate.exists():
                return candidate
        return None

    def _convert_with_office(self, source: Path, target: Path) -> bool:
        if not _looks_like_ole(source):
            return False
        return self._office_session.convert(source, target)

    def _convert_with_libreoffice(self, source: Path, output_dir: Path) -> bool:
        soffice = _find_soffice()
        if soffice is None:
            return False
        try:
            result = subprocess.run(
                [
                    str(soffice),
                    "--headless",
                    "--convert-to",
                    self._libreoffice_target(source),
                    "--outdir",
                    str(output_dir),
                    str(source),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0

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
        with source.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
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
        self._apps: dict[str, object] = {}
        self._job_handles: dict[str, object] = {}
        self._registry_paths: dict[str, Path] = {}
        self._com_initialized = False
        atexit.register(self.close)

    def convert(self, source: Path, target: Path) -> bool:
        kind = source.suffix.lower()
        try:
            app = self._application(kind)
            if kind == ".doc":
                document = app.Documents.Open(str(source), ReadOnly=True)
                try:
                    document.SaveAs2(str(target), FileFormat=16)
                finally:
                    document.Close(False)
            elif kind == ".xls":
                workbook = app.Workbooks.Open(str(source), ReadOnly=True)
                try:
                    workbook.SaveAs(str(target), FileFormat=51)
                finally:
                    workbook.Close(False)
            elif kind == ".ppt":
                presentation = app.Presentations.Open(str(source), WithWindow=False)
                try:
                    presentation.SaveAs(str(target), 24)
                finally:
                    presentation.Close()
            else:
                return False
            return target.exists()
        except Exception:
            self._reset(kind)
            return False

    def _application(self, kind: str) -> object:
        if kind in self._apps:
            return self._apps[kind]
        import pythoncom
        import win32com.client

        if not self._com_initialized:
            pythoncom.CoInitialize()
            self._com_initialized = True
        prog_id = {
            ".doc": "Word.Application",
            ".xls": "Excel.Application",
            ".ppt": "PowerPoint.Application",
        }[kind]
        previous_pids = _office_process_ids(kind)
        app = win32com.client.DispatchEx(prog_id)
        if kind == ".doc":
            app.Visible = False
            app.DisplayAlerts = 0
        elif kind == ".xls":
            app.Visible = False
            app.DisplayAlerts = False
        self._bind_application_lifetime(kind, app, previous_pids)
        self._apps[kind] = app
        return app

    def _bind_application_lifetime(
        self,
        kind: str,
        app: object,
        previous_pids: set[int],
    ) -> None:
        if os.name != "nt":
            return
        try:
            process_id = _wait_for_new_office_process(kind, previous_pids)
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
            self._registry_paths[kind] = _register_office_process(kind, int(process_id))
            job = _create_kill_on_close_job(int(process_id))
            if job is not None:
                self._job_handles[kind] = job
        except Exception:
            logger.debug("Unable to bind Office process lifetime for %s", kind, exc_info=True)

    def _reset(self, kind: str) -> None:
        app = self._apps.pop(kind, None)
        if app is not None:
            try:
                app.Quit()
            except Exception:
                pass
        job = self._job_handles.pop(kind, None)
        if job is not None:
            try:
                import win32api

                win32api.CloseHandle(job)
            except Exception:
                logger.debug("Unable to close Office job handle", exc_info=True)
        registry_path = self._registry_paths.pop(kind, None)
        if registry_path is not None:
            registry_path.unlink(missing_ok=True)

    def close(self) -> None:
        for kind in list(self._apps):
            self._reset(kind)
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


def _office_process_ids(kind: str) -> set[int]:
    process_name = {
        ".doc": "winword.exe",
        ".xls": "excel.exe",
        ".ppt": "powerpnt.exe",
    }[kind]
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
    previous_pids: set[int],
    timeout_seconds: float = 5.0,
) -> int | None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        candidates = _office_process_ids(kind) - previous_pids
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
    allowed_names = {"winword.exe", "excel.exe", "powerpnt.exe"}
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
