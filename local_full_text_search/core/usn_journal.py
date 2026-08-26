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
from dataclasses import dataclass
from pathlib import Path


class UsnJournalUnavailable(RuntimeError):
    """The volume cannot safely provide a continuous USN change set."""


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


_FSCTL_QUERY_USN_JOURNAL = 0x000900F4
_FSCTL_READ_USN_JOURNAL = 0x000900BB
_GENERIC_READ = 0x80000000
_FILE_SHARE_READ = 0x00000001
_FILE_SHARE_WRITE = 0x00000002
_OPEN_EXISTING = 3
_SERVICE_PIPE = r"\\.\pipe\LocalFullTextSearch.Usn.v1"
_SERVICE_MAX_RESPONSE_BYTES = 32 * 1024 * 1024


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
        except (KeyError, TypeError, ValueError, OSError, UsnJournalUnavailable):
            raise direct_error


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


def changes_since(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    max_records: int = 20_000,
) -> UsnChangeSet:
    """Return journal records after ``checkpoint`` up to a stable cursor.

    The returned checkpoint is captured before the read loop. Changes arriving
    during the loop are intentionally left for the next run rather than being
    lost behind a moving cursor.
    """

    try:
        return _changes_since_direct(root_path, checkpoint, max_records=max_records)
    except UsnJournalUnavailable as direct_error:
        try:
            response = _service_request(
                {
                    "operation": "changes_since",
                    "root_path": str(root_path),
                    "checkpoint": {
                        "journal_id": int(checkpoint.journal_id),
                        "next_usn": int(checkpoint.next_usn),
                        "volume_serial": int(checkpoint.volume_serial),
                    },
                    "max_records": max(1, min(int(max_records), 20_000)),
                }
            )
            next_checkpoint = response["checkpoint"]
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
            )
            return UsnChangeSet(
                checkpoint=UsnCheckpoint(
                    journal_id=int(next_checkpoint["journal_id"]),
                    next_usn=int(next_checkpoint["next_usn"]),
                    volume_serial=int(next_checkpoint["volume_serial"]),
                ),
                records=records,
            )
        except (KeyError, TypeError, ValueError, OSError, UsnJournalUnavailable):
            raise direct_error


def _changes_since_direct(
    root_path: Path,
    checkpoint: UsnCheckpoint,
    *,
    max_records: int = 20_000,
) -> UsnChangeSet:
    """Read journal records in-process when the caller is elevated."""

    kernel32, volume = _open_volume(root_path)
    try:
        data = _query_journal(kernel32, volume)
        journal_id, first_usn, target_usn = (int(data[0]), int(data[1]), int(data[2]))
        serial = _volume_serial(kernel32, root_path)
        if int(checkpoint.journal_id) != journal_id:
            raise UsnJournalUnavailable("USN journal identifier changed")
        if int(checkpoint.volume_serial) != serial:
            raise UsnJournalUnavailable("volume serial changed")
        if not (first_usn <= int(checkpoint.next_usn) <= target_usn):
            raise UsnJournalUnavailable("USN checkpoint is outside the retained journal")

        records: list[UsnRecord] = []
        start_usn = int(checkpoint.next_usn)
        while start_usn < target_usn:
            output = _read_journal(kernel32, volume, start_usn, journal_id)
            if len(output) < 8:
                raise UsnJournalUnavailable("USN read returned an incomplete cursor")
            next_usn = int(struct.unpack_from("<q", output, 0)[0])
            if next_usn <= start_usn:
                raise UsnJournalUnavailable("USN journal made no cursor progress")
            for record in _parse_records(output[8:]):
                if record.usn >= target_usn:
                    continue
                records.append(record)
                if len(records) > max(1, int(max_records)):
                    raise UsnJournalUnavailable("USN change set exceeds safe fast-path limit")
            start_usn = next_usn
        return UsnChangeSet(
            checkpoint=UsnCheckpoint(journal_id, target_usn, serial),
            records=tuple(records),
        )
    finally:
        kernel32.CloseHandle(volume)


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
            raise UsnJournalUnavailable(str(response.get("error") or "USN helper failed"))
        result = response.get("result")
        if not isinstance(result, dict):
            raise UsnJournalUnavailable("USN helper returned no result")
        return result
    finally:
        win32file.CloseHandle(handle)


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


def _read_journal(kernel32, volume, start_usn: int, journal_id: int) -> bytes:
    import ctypes
    from ctypes import wintypes

    request = struct.pack(
        "<qIIIIQHH",
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
    output = ctypes.create_string_buffer(1024 * 1024)
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
