"""Best-effort NTFS USN journal access for fast incremental discovery.

The journal is only an optimization. Callers must fall back to a directory walk
when its continuity cannot be proved; that rule prevents missed files after a
journal reset, a volume change, or an unsupported filesystem.
"""

from __future__ import annotations

import os
import json
import struct
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path


class UsnJournalUnavailable(RuntimeError):
    """The volume cannot safely provide a continuous USN change set."""


class UsnChangeSetLimitExceeded(UsnJournalUnavailable):
    """The journal gap is too large for the bounded USN fast path."""

    error_code = "USN_CHANGE_SET_LIMIT"

    def __init__(
        self,
        limit: int = 20_000,
        records_seen: int | None = None,
        *,
        message: str | None = None,
    ) -> None:
        self.limit = max(1, int(limit))
        self.records_seen = (
            None if records_seen is None else max(0, int(records_seen))
        )
        if message:
            detail = str(message)
        else:
            detail = f"USN change set exceeds safe fast-path limit (limit={self.limit}"
            if self.records_seen is not None:
                detail += f", records_seen={self.records_seen}"
            detail += ")"
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class UsnCheckpoint:
    journal_id: int
    next_usn: int
    volume_serial: int


@dataclass(frozen=True, slots=True)
class UsnRecord:
    file_reference: int
    parent_reference: int
    usn: int
    reason: int
    attributes: int
    name: str

    @property
    def is_directory(self) -> bool:
        return bool(self.attributes & 0x10)  # FILE_ATTRIBUTE_DIRECTORY


@dataclass(frozen=True, slots=True)
class UsnChangeSet:
    checkpoint: UsnCheckpoint
    records: tuple[UsnRecord, ...]


@dataclass(frozen=True, slots=True)
class UsnChangePage:
    """One bounded page of records and the stable target cursor for the read."""

    checkpoint: UsnCheckpoint
    target_checkpoint: UsnCheckpoint
    records: tuple[UsnRecord, ...]


_FSCTL_QUERY_USN_JOURNAL = 0x000900F4
_FSCTL_READ_USN_JOURNAL = 0x000900BB
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_SERVICE_PIPE = r"\\.\pipe\LocalFullTextSearch.Usn.v1"
_SERVICE_MAX_RESPONSE_BYTES = 32 * 1024 * 1024
_DEFAULT_MAX_RECORDS = 20_000
_DEFAULT_USN_PAGE_BYTES = 1024 * 1024
_MIN_USN_PAGE_BYTES = 64 * 1024
_MAX_USN_PAGE_BYTES = 2 * 1024 * 1024


def _combine_fallback_errors(
    direct_error: UsnJournalUnavailable,
    service_error: UsnJournalUnavailable,
) -> UsnJournalUnavailable:
    """Keep both errors when direct access and the helper fail differently."""

    if isinstance(service_error, UsnChangeSetLimitExceeded):
        return service_error
    if isinstance(direct_error, UsnChangeSetLimitExceeded):
        return direct_error
    return UsnJournalUnavailable(
        f"USN direct access failed: {direct_error}; "
        f"USN helper failed: {service_error}"
    )


def current_checkpoint(root_path: Path) -> UsnCheckpoint:
    """Return the current NTFS journal cursor for a drive-rooted path."""

    try:
        return _current_checkpoint_direct(root_path)
    except UsnJournalUnavailable as direct_error:
        try:
            response = _service_request(
                {"operation": "current_checkpoint", "root_path": str(root_path)}
            )
            return UsnCheckpoint(
                journal_id=int(response["journal_id"]),
                next_usn=int(response["next_usn"]),
                volume_serial=int(response["volume_serial"]),
            )
        except (KeyError, TypeError, ValueError, OSError, UsnJournalUnavailable) as service_error:
            combined_error = _combine_fallback_errors(direct_error, service_error)
            if combined_error is service_error:
                raise
            raise combined_error from service_error


def _current_checkpoint_direct(root_path: Path) -> UsnCheckpoint:
    """Read a journal cursor in-process when the caller is elevated."""

    kernel32, volume = _open_volume(root_path)
    try:
        data = _query_journal(kernel32, volume)
        return UsnCheckpoint(
            journal_id=int(data[0]),
            next_usn=int(data[2]),
            volume_serial=_volume_serial(kernel32, root_path),
        )
    finally:
        kernel32.CloseHandle(volume)


def _checkpoint_payload(checkpoint: UsnCheckpoint) -> dict[str, int]:
    return {
        "journal_id": int(checkpoint.journal_id),
        "next_usn": int(checkpoint.next_usn),
        "volume_serial": int(checkpoint.volume_serial),
    }


def _checkpoint_from_payload(value: object, *, label: str) -> UsnCheckpoint:
    if not isinstance(value, dict):
        raise UsnJournalUnavailable(f"USN helper returned no {label} checkpoint")
    try:
        return UsnCheckpoint(
            journal_id=int(value["journal_id"]),
            next_usn=int(value["next_usn"]),
            volume_serial=int(value["volume_serial"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise UsnJournalUnavailable(
            f"USN helper returned an invalid {label} checkpoint"
        ) from exc


def _bounded_page_bytes(value: object) -> int:
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = _DEFAULT_USN_PAGE_BYTES
    return max(_MIN_USN_PAGE_BYTES, min(requested, _MAX_USN_PAGE_BYTES))


def _resolve_target_checkpoint(
    checkpoint: UsnCheckpoint,
    journal_data: tuple[int, int, int],
    volume_serial: int,
    requested_target: UsnCheckpoint | None,
) -> UsnCheckpoint:
    journal_id, first_usn, current_target_usn = (
        int(journal_data[0]),
        int(journal_data[1]),
        int(journal_data[2]),
    )
    if int(checkpoint.journal_id) != journal_id:
        raise UsnJournalUnavailable("USN journal identifier changed")
    if int(checkpoint.volume_serial) != int(volume_serial):
        raise UsnJournalUnavailable("volume serial changed")
    target_usn = current_target_usn
    if requested_target is not None:
        if int(requested_target.journal_id) != journal_id:
            raise UsnJournalUnavailable("USN target journal identifier changed")
        if int(requested_target.volume_serial) != int(volume_serial):
            raise UsnJournalUnavailable("USN target volume serial changed")
        target_usn = int(requested_target.next_usn)
        if target_usn > current_target_usn:
            raise UsnJournalUnavailable("USN target checkpoint is ahead of the journal")
    if not (first_usn <= int(checkpoint.next_usn) <= target_usn):
        raise UsnJournalUnavailable("USN checkpoint is outside the retained journal")
    return UsnCheckpoint(journal_id, target_usn, int(volume_serial))


def _read_changes_page_from_volume(
    kernel32,
    volume,
    checkpoint: UsnCheckpoint,
    target_checkpoint: UsnCheckpoint,
    *,
    page_bytes: int,
) -> UsnChangePage:
    start_usn = int(checkpoint.next_usn)
    target_usn = int(target_checkpoint.next_usn)
    if start_usn >= target_usn:
        return UsnChangePage(
            checkpoint=target_checkpoint,
            target_checkpoint=target_checkpoint,
            records=(),
        )
    output = _read_journal(
        kernel32,
        volume,
        start_usn,
        int(target_checkpoint.journal_id),
        output_bytes=page_bytes,
    )
    if len(output) < 8:
        raise UsnJournalUnavailable("USN read returned an incomplete cursor")
    next_usn = int(struct.unpack_from("<q", output, 0)[0])
    if next_usn <= start_usn:
        raise UsnJournalUnavailable("USN journal made no cursor progress")
    next_usn = min(next_usn, target_usn)
    records = tuple(
        record
        for record in _parse_records(output[8:])
        if int(record.usn) < target_usn
    )
    return UsnChangePage(
        checkpoint=UsnCheckpoint(
            int(target_checkpoint.journal_id),
            next_usn,
            int(target_checkpoint.volume_serial),
        ),
        target_checkpoint=target_checkpoint,
        records=records,
    )


def _changes_since_page_direct(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    target_checkpoint: UsnCheckpoint | None = None,
    page_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> UsnChangePage:
    """Read one bounded page in-process or inside the privileged service."""

    kernel32, volume = _open_volume(root_path)
    try:
        journal_data = _query_journal(kernel32, volume)
        serial = _volume_serial(kernel32, root_path)
        target = _resolve_target_checkpoint(
            checkpoint,
            journal_data,
            serial,
            target_checkpoint,
        )
        return _read_changes_page_from_volume(
            kernel32,
            volume,
            checkpoint,
            target,
            page_bytes=_bounded_page_bytes(page_bytes),
        )
    finally:
        kernel32.CloseHandle(volume)


def _iter_changes_since_direct(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    page_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> Iterator[UsnChangePage]:
    """Read a stable journal window as bounded pages while holding one volume handle."""

    kernel32, volume = _open_volume(root_path)
    try:
        journal_data = _query_journal(kernel32, volume)
        serial = _volume_serial(kernel32, root_path)
        target = _resolve_target_checkpoint(checkpoint, journal_data, serial, None)
        current = checkpoint
        while True:
            page = _read_changes_page_from_volume(
                kernel32,
                volume,
                current,
                target,
                page_bytes=_bounded_page_bytes(page_bytes),
            )
            yield page
            if int(page.checkpoint.next_usn) >= int(target.next_usn):
                return
            current = page.checkpoint
    finally:
        kernel32.CloseHandle(volume)


def _iter_changes_since_service(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    page_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> Iterator[UsnChangePage]:
    current = checkpoint
    target: UsnCheckpoint | None = None
    while True:
        payload: dict[str, object] = {
            "operation": "changes_since_page",
            "root_path": str(root_path),
            "checkpoint": _checkpoint_payload(current),
            "page_bytes": _bounded_page_bytes(page_bytes),
        }
        if target is not None:
            payload["target_checkpoint"] = _checkpoint_payload(target)
        response = _service_request(payload)
        page_checkpoint = _checkpoint_from_payload(
            response.get("checkpoint"),
            label="page",
        )
        target_checkpoint = _checkpoint_from_payload(
            response.get("target_checkpoint"),
            label="target",
        )
        if int(page_checkpoint.next_usn) < int(current.next_usn):
            raise UsnJournalUnavailable("USN helper page cursor moved backwards")
        if int(page_checkpoint.next_usn) > int(target_checkpoint.next_usn):
            raise UsnJournalUnavailable("USN helper page cursor passed its target")
        records = tuple(
            UsnRecord(
                file_reference=int(item["file_reference"]),
                parent_reference=int(item["parent_reference"]),
                usn=int(item["usn"]),
                reason=int(item["reason"]),
                attributes=int(item["attributes"]),
                name=str(item["name"]),
            )
            for item in response.get("records", ())
            if isinstance(item, dict)
        )
        page = UsnChangePage(
            checkpoint=page_checkpoint,
            target_checkpoint=target_checkpoint,
            records=records,
        )
        yield page
        if int(page_checkpoint.next_usn) >= int(target_checkpoint.next_usn):
            return
        if int(page_checkpoint.next_usn) == int(current.next_usn):
            raise UsnJournalUnavailable("USN helper page made no cursor progress")
        current = page_checkpoint
        target = target_checkpoint


def iter_changes_since(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    page_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> Iterator[UsnChangePage]:
    """Yield bounded USN pages, preferring direct access and then the service."""

    try:
        yield from _iter_changes_since_direct(
            root_path,
            checkpoint,
            page_bytes=page_bytes,
        )
        return
    except UsnJournalUnavailable as direct_error:
        try:
            yield from _iter_changes_since_service(
                root_path,
                checkpoint,
                page_bytes=page_bytes,
            )
            return
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            UsnJournalUnavailable,
        ) as service_error:
            combined_error = _combine_fallback_errors(direct_error, service_error)
            if combined_error is service_error:
                raise
            raise combined_error from service_error


def changes_since(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    max_records: int | None = None,
    on_page: Callable[[tuple[UsnRecord, ...]], None] | None = None,
    page_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> UsnChangeSet:
    """Read a stable USN window in pages and optionally consume each page."""

    latest_by_reference: dict[int, UsnRecord] = {}
    records_seen = 0
    final_checkpoint = checkpoint
    for page in iter_changes_since(
        root_path,
        checkpoint,
        page_bytes=page_bytes,
    ):
        final_checkpoint = page.target_checkpoint
        if on_page is not None:
            on_page(page.records)
        else:
            for record in page.records:
                records_seen += 1
                if max_records is not None and records_seen > max(1, int(max_records)):
                    raise UsnChangeSetLimitExceeded(
                        limit=max_records,
                        records_seen=records_seen,
                    )
                latest_by_reference[int(record.file_reference)] = record
    return UsnChangeSet(
        checkpoint=final_checkpoint,
        records=tuple(latest_by_reference.values()),
    )


def _changes_since_direct(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    max_records: int | None = None,
) -> UsnChangeSet:
    """Read the complete journal window, retaining only the latest record per reference."""

    latest_by_reference: dict[int, UsnRecord] = {}
    records_seen = 0
    final_checkpoint = checkpoint
    for page in _iter_changes_since_direct(root_path, checkpoint):
        final_checkpoint = page.target_checkpoint
        for record in page.records:
            records_seen += 1
            if max_records is not None and records_seen > max(1, int(max_records)):
                raise UsnChangeSetLimitExceeded(
                    limit=max_records,
                    records_seen=records_seen,
                )
            latest_by_reference[int(record.file_reference)] = record
    return UsnChangeSet(
        checkpoint=final_checkpoint,
        records=tuple(latest_by_reference.values()),
    )


def _service_request(payload: dict[str, object], *, timeout_ms: int = 750) -> dict[str, object]:
    """Call the installed privileged helper through a local-only named pipe."""

    if os.name != "nt":
        raise UsnJournalUnavailable("USN helper service is only available on Windows")
    import pywintypes
    import win32file
    import win32pipe

    deadline = time.monotonic() + max(0.05, float(timeout_ms) / 1000.0)
    handle = None
    while handle is None:
        try:
            handle = win32file.CreateFile(
                _SERVICE_PIPE,
                win32file.GENERIC_READ | win32file.GENERIC_WRITE,
                0,
                None,
                win32file.OPEN_EXISTING,
                0,
                None,
            )
        except pywintypes.error as exc:
            if exc.winerror not in {2, 231} or time.monotonic() >= deadline:
                raise UsnJournalUnavailable(
                    f"USN helper service is unavailable ({exc.winerror})"
                ) from exc
            remaining_ms = max(1, int((deadline - time.monotonic()) * 1000))
            try:
                win32pipe.WaitNamedPipe(_SERVICE_PIPE, min(remaining_ms, 200))
            except pywintypes.error:
                time.sleep(0.02)
    try:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(body) > 64 * 1024:
            raise UsnJournalUnavailable("USN helper request is too large")
        _write_all(handle, struct.pack("<I", len(body)) + body)
        response_size = struct.unpack("<I", _read_exact(handle, 4))[0]
        if response_size <= 0 or response_size > _SERVICE_MAX_RESPONSE_BYTES:
            raise UsnJournalUnavailable("USN helper returned an invalid response size")
        response = json.loads(_read_exact(handle, response_size).decode("utf-8"))
        if not isinstance(response, dict):
            raise UsnJournalUnavailable("USN helper returned an invalid response")
        if not response.get("ok"):
            error_text = str(response.get("error") or "USN helper failed")
            error_code = str(response.get("error_code") or "")
            if (
                error_code == UsnChangeSetLimitExceeded.error_code
                or error_text.startswith("UsnChangeSetLimitExceeded:")
                or "USN change set exceeds safe fast-path limit" in error_text
            ):
                details = response.get("details")
                if not isinstance(details, dict):
                    details = {}
                try:
                    limit = int(details.get("limit", _DEFAULT_MAX_RECORDS))
                except (TypeError, ValueError):
                    limit = _DEFAULT_MAX_RECORDS
                try:
                    records_seen = (
                        None
                        if details.get("records_seen") is None
                        else int(details["records_seen"])
                    )
                except (TypeError, ValueError):
                    records_seen = None
                raise UsnChangeSetLimitExceeded(
                    limit=limit,
                    records_seen=records_seen,
                    message=error_text,
                )
            raise UsnJournalUnavailable(error_text)
        result = response.get("result")
        if not isinstance(result, dict):
            raise UsnJournalUnavailable("USN helper returned no result")
        return result
    except UsnJournalUnavailable:
        raise
    except (pywintypes.error, OSError, ValueError, TypeError, struct.error) as exc:
        error_code = getattr(exc, "winerror", None)
        suffix = f" ({error_code})" if error_code is not None else ""
        raise UsnJournalUnavailable(
            f"USN helper request failed{suffix}: {exc}"
        ) from exc
    finally:
        if handle is not None:
            try:
                win32file.CloseHandle(handle)
            except Exception:
                pass


def _read_exact(handle, size: int) -> bytes:
    import win32file

    chunks: list[bytes] = []
    remaining = int(size)
    while remaining > 0:
        _status, data = win32file.ReadFile(handle, min(remaining, 64 * 1024))
        if not data:
            raise UsnJournalUnavailable("USN helper closed the pipe unexpectedly")
        chunks.append(bytes(data))
        remaining -= len(data)
    return b"".join(chunks)


def _write_all(handle, data: bytes) -> None:
    import win32file

    offset = 0
    while offset < len(data):
        _status, written = win32file.WriteFile(handle, data[offset : offset + 64 * 1024])
        amount = int(written) if isinstance(written, int) else len(data[offset : offset + 64 * 1024])
        if amount <= 0:
            raise UsnJournalUnavailable("USN helper pipe write made no progress")
        offset += amount


def _open_volume(root_path: Path):
    if os.name != "nt":
        raise UsnJournalUnavailable("USN is only available on Windows NTFS volumes")
    import ctypes
    from ctypes import wintypes

    drive = Path(root_path).drive
    if len(drive) != 2 or not drive[0].isalpha() or drive[1] != ":":
        raise UsnJournalUnavailable("search root is not a local drive path")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.restype = wintypes.HANDLE
    volume = kernel32.CreateFileW(
        f"\\\\.\\{drive}",
        _GENERIC_READ,
        _FILE_SHARE_READ | _FILE_SHARE_WRITE,
        None,
        _OPEN_EXISTING,
        0,
        None,
    )
    if volume == wintypes.HANDLE(-1).value:
        error = ctypes.get_last_error()
        raise UsnJournalUnavailable(f"cannot open NTFS volume ({error})")
    return kernel32, volume


def _query_journal(kernel32, volume):
    import ctypes
    from ctypes import wintypes

    output = ctypes.create_string_buffer(56)
    returned = wintypes.DWORD()
    ok = kernel32.DeviceIoControl(
        volume,
        _FSCTL_QUERY_USN_JOURNAL,
        None,
        0,
        output,
        len(output),
        ctypes.byref(returned),
        None,
    )
    if not ok or returned.value < 24:
        raise UsnJournalUnavailable(f"cannot query NTFS USN journal ({ctypes.get_last_error()})")
    return struct.unpack_from("<Qqq", output.raw, 0)


def _read_journal(
    kernel32,
    volume,
    start_usn: int,
    journal_id: int,
    *,
    output_bytes: int = _DEFAULT_USN_PAGE_BYTES,
) -> bytes:
    import ctypes
    from ctypes import wintypes

    request = struct.pack(
        # Windows 8+ uses READ_USN_JOURNAL_DATA_V1:
        # StartUsn, ReasonMask, ReturnOnlyOnClose, Timeout (DWORDLONG),
        # BytesToWaitFor (DWORDLONG), UsnJournalID (DWORDLONG),
        # MinMajorVersion, MaxMajorVersion.
        # The two 64-bit wait fields are required on Windows 10; using a
        # 32-bit field here shifts the journal id and makes DeviceIoControl
        # return ERROR_INVALID_USER_BUFFER/ERROR_INVALID_PARAMETER.
        "<qIIQQQHH",
        int(start_usn),
        0xFFFFFFFF,
        0,
        0,
        0,
        int(journal_id),
        2,
        2,
    )
    input_buffer = ctypes.create_string_buffer(request)
    output = ctypes.create_string_buffer(_bounded_page_bytes(output_bytes))
    returned = wintypes.DWORD()
    ok = kernel32.DeviceIoControl(
        volume,
        _FSCTL_READ_USN_JOURNAL,
        input_buffer,
        len(request),
        output,
        len(output),
        ctypes.byref(returned),
        None,
    )
    if not ok:
        raise UsnJournalUnavailable(f"cannot read NTFS USN journal ({ctypes.get_last_error()})")
    return output.raw[: int(returned.value)]


def _volume_serial(kernel32, root_path: Path) -> int:
    import ctypes
    from ctypes import wintypes

    serial = wintypes.DWORD()
    root = f"{Path(root_path).drive}\\"
    ok = kernel32.GetVolumeInformationW(
        root,
        None,
        0,
        ctypes.byref(serial),
        None,
        None,
        None,
        0,
    )
    if not ok:
        raise UsnJournalUnavailable(f"cannot read volume serial ({ctypes.get_last_error()})")
    return int(serial.value)


def _parse_records(data: bytes) -> list[UsnRecord]:
    records: list[UsnRecord] = []
    offset = 0
    size = len(data)
    while offset + 8 <= size:
        record_length, major, _minor = struct.unpack_from("<IHH", data, offset)
        if record_length <= 0 or offset + record_length > size:
            raise UsnJournalUnavailable("malformed USN record")
        if major != 2 or record_length < 60:
            raise UsnJournalUnavailable(f"unsupported USN record version {major}")
        file_reference, parent_reference, usn = struct.unpack_from("<QQq", data, offset + 8)
        reason, attributes = struct.unpack_from("<II", data, offset + 40)
        name_length, name_offset = struct.unpack_from("<HH", data, offset + 56)
        name_end = offset + name_offset + name_length
        if name_offset < 60 or name_end > offset + record_length:
            raise UsnJournalUnavailable("malformed USN file name")
        name = data[offset + name_offset : name_end].decode("utf-16-le", errors="replace")
        records.append(
            UsnRecord(
                file_reference=int(file_reference),
                parent_reference=int(parent_reference),
                usn=int(usn),
                reason=int(reason),
                attributes=int(attributes),
                name=name,
            )
        )
        offset += record_length
    return records
