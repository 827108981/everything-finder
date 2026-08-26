"""Restricted Windows service for reading NTFS change-journal metadata.

The service exposes three fixed operations through a local named pipe. It does
not read file contents, accept executable commands, or listen on the network.
"""

from __future__ import annotations

import json
import os
import struct
import sys
import threading
from dataclasses import asdict
from pathlib import Path

from local_full_text_search.core.usn_journal import (
    UsnCheckpoint,
    UsnJournalUnavailable,
    _changes_since_direct,
    _current_checkpoint_direct,
)


SERVICE_NAME = "LocalFullTextSearchUsn"
SERVICE_DISPLAY_NAME = "本地全文搜索 - NTFS 变更监控服务"
SERVICE_DESCRIPTION = (
    "仅读取本机 NTFS 文件变化记录，帮助本地全文搜索工具快速定位新增、修改和删除文件。"
)
PIPE_NAME = r"\\.\pipe\LocalFullTextSearch.Usn.v1"
MAX_REQUEST_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def dispatch_request(request: dict[str, object]) -> dict[str, object]:
    """Execute one allow-listed journal operation."""

    operation = str(request.get("operation") or "")
    if operation == "ping":
        return {"service": SERVICE_NAME, "version": 1}
    root_path = _validated_local_root(request.get("root_path"))
    if operation == "current_checkpoint":
        return asdict(_current_checkpoint_direct(root_path))
    if operation == "changes_since":
        raw_checkpoint = request.get("checkpoint")
        if not isinstance(raw_checkpoint, dict):
            raise ValueError("checkpoint is required")
        checkpoint = UsnCheckpoint(
            journal_id=int(raw_checkpoint["journal_id"]),
            next_usn=int(raw_checkpoint["next_usn"]),
            volume_serial=int(raw_checkpoint["volume_serial"]),
        )
        max_records = max(1, min(int(request.get("max_records", 20_000)), 20_000))
        result = _changes_since_direct(root_path, checkpoint, max_records=max_records)
        return {
            "checkpoint": asdict(result.checkpoint),
            "records": [asdict(record) for record in result.records],
        }
    raise ValueError("unsupported USN helper operation")


def _validated_local_root(value: object) -> Path:
    text = os.path.abspath(str(value or ""))
    root = Path(text)
    drive = root.drive
    if len(drive) != 2 or drive[1] != ":" or not drive[0].isalpha():
        raise ValueError("root_path must be on a local drive")
    return root


def _pipe_security_attributes():
    import win32security

    # SYSTEM and administrators retain full control. Interactive users can
    # only connect and exchange pipe data; the operation allow-list above is
    # the service's authorization boundary.
    descriptor = win32security.ConvertStringSecurityDescriptorToSecurityDescriptor(
        "D:P(A;;GA;;;SY)(A;;GA;;;BA)(A;;GRGW;;;IU)",
        win32security.SDDL_REVISION_1,
    )
    attributes = win32security.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR = descriptor
    return attributes


def _read_exact(handle, size: int) -> bytes:
    import win32file

    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        _status, data = win32file.ReadFile(handle, min(remaining, 64 * 1024))
        if not data:
            raise OSError("client closed the USN helper pipe")
        chunks.append(bytes(data))
        remaining -= len(data)
    return b"".join(chunks)


def _write_all(handle, data: bytes) -> None:
    import win32file

    offset = 0
    while offset < len(data):
        chunk = data[offset : offset + 64 * 1024]
        _status, written = win32file.WriteFile(handle, chunk)
        amount = int(written) if isinstance(written, int) else len(chunk)
        if amount <= 0:
            raise OSError("USN helper pipe write made no progress")
        offset += amount


def _serve_connection(handle) -> None:
    try:
        request_size = struct.unpack("<I", _read_exact(handle, 4))[0]
        if request_size <= 0 or request_size > MAX_REQUEST_BYTES:
            raise ValueError("invalid USN helper request size")
        request = json.loads(_read_exact(handle, request_size).decode("utf-8"))
        if not isinstance(request, dict):
            raise ValueError("USN helper request must be an object")
        result = dispatch_request(request)
        response: dict[str, object] = {"ok": True, "result": result}
    except (KeyError, TypeError, ValueError, OSError, UsnJournalUnavailable) as exc:
        response = {"ok": False, "error": f"{exc.__class__.__name__}: {exc}"}
    body = json.dumps(response, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if len(body) > MAX_RESPONSE_BYTES:
        body = json.dumps(
            {"ok": False, "error": "USN helper response exceeded the safety limit"},
            separators=(",", ":"),
        ).encode("utf-8")
    _write_all(handle, struct.pack("<I", len(body)) + body)


import servicemanager
import win32event
import win32pipe
import win32service
import win32serviceutil


class UsnJournalWindowsService(win32serviceutil.ServiceFramework):
    """Windows service host for the fixed local USN API."""

    _svc_name_ = SERVICE_NAME
    _svc_display_name_ = SERVICE_DISPLAY_NAME
    _svc_description_ = SERVICE_DESCRIPTION

    def __init__(self, args):
        super().__init__(args)
        self._stop_event = win32event.CreateEvent(None, True, False, None)
        self._stopping = threading.Event()

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        self._stopping.set()
        win32event.SetEvent(self._stop_event)
        _wake_pipe()

    def SvcDoRun(self):
        servicemanager.LogInfoMsg(f"{SERVICE_NAME} started")
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        while not self._stopping.is_set():
            pipe = win32pipe.CreateNamedPipe(
                PIPE_NAME,
                win32pipe.PIPE_ACCESS_DUPLEX,
                win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_READMODE_BYTE | win32pipe.PIPE_WAIT,
                4,
                64 * 1024,
                64 * 1024,
                1000,
                _pipe_security_attributes(),
            )
            try:
                try:
                    win32pipe.ConnectNamedPipe(pipe, None)
                except Exception as exc:
                    if getattr(exc, "winerror", None) != 535:
                        raise
                if not self._stopping.is_set():
                    _serve_connection(pipe)
            except Exception as exc:
                servicemanager.LogErrorMsg(f"{SERVICE_NAME} request failed: {exc}")
            finally:
                try:
                    win32pipe.DisconnectNamedPipe(pipe)
                except Exception:
                    pass
                import win32file

                win32file.CloseHandle(pipe)
        servicemanager.LogInfoMsg(f"{SERVICE_NAME} stopped")


def _wake_pipe() -> None:
    try:
        import win32file

        handle = win32file.CreateFile(
            PIPE_NAME,
            win32file.GENERIC_READ | win32file.GENERIC_WRITE,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None,
        )
        win32file.CloseHandle(handle)
    except Exception:
        pass


def run_service_dispatcher() -> int:
    import servicemanager

    service_class = UsnJournalService.service_class()
    servicemanager.Initialize()
    servicemanager.PrepareToHostSingle(service_class)
    servicemanager.StartServiceCtrlDispatcher()
    return 0


def install_or_update_service() -> int:
    import pywintypes
    import win32service
    import win32serviceutil

    executable = str(Path(sys.executable).resolve())
    try:
        win32serviceutil.InstallService(
            f"{__name__}.UsnJournalWindowsService",
            SERVICE_NAME,
            SERVICE_DISPLAY_NAME,
            startType=win32service.SERVICE_AUTO_START,
            exeName=executable,
            exeArgs="--run-usn-service",
            description=SERVICE_DESCRIPTION,
            delayedstart=False,
        )
    except pywintypes.error as exc:
        if exc.winerror != 1073:
            raise
        win32serviceutil.ChangeServiceConfig(
            f"{__name__}.UsnJournalWindowsService",
            SERVICE_NAME,
            startType=win32service.SERVICE_AUTO_START,
            exeName=executable,
            exeArgs="--run-usn-service",
            displayName=SERVICE_DISPLAY_NAME,
            description=SERVICE_DESCRIPTION,
            delayedstart=False,
        )
    try:
        win32serviceutil.StartService(SERVICE_NAME)
    except pywintypes.error as exc:
        if exc.winerror != 1056:
            raise
    return 0


def remove_service() -> int:
    import pywintypes
    import win32serviceutil

    try:
        win32serviceutil.StopService(SERVICE_NAME)
        try:
            win32serviceutil.WaitForServiceStatus(SERVICE_NAME, 1, 10)
        except pywintypes.error:
            pass
    except pywintypes.error as exc:
        if exc.winerror not in {1060, 1062}:
            raise
    try:
        win32serviceutil.RemoveService(SERVICE_NAME)
    except pywintypes.error as exc:
        if exc.winerror != 1060:
            raise
    return 0
