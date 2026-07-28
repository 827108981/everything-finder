from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

from local_full_text_search.parsers.legacy_office_parser import _create_kill_on_close_job


def test_office_job_uses_kill_on_close_and_closes_process_handle() -> None:
    calls: list[tuple[object, ...]] = []
    job = object()
    process_handle = object()
    win32api = SimpleNamespace(
        OpenProcess=lambda access, inherit, pid: (
            calls.append(("open", access, inherit, pid)) or process_handle
        ),
        CloseHandle=lambda handle: calls.append(("close", handle)),
    )
    win32con = SimpleNamespace(
        PROCESS_TERMINATE=1,
        PROCESS_SET_QUOTA=2,
        PROCESS_QUERY_INFORMATION=4,
    )
    win32job = SimpleNamespace(
        JobObjectExtendedLimitInformation=9,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE=0x2000,
        CreateJobObject=lambda security, name: job,
        QueryInformationJobObject=lambda handle, info_class: {
            "BasicLimitInformation": {"LimitFlags": 0}
        },
        SetInformationJobObject=lambda handle, info_class, info: calls.append(
            ("limits", handle, info["BasicLimitInformation"]["LimitFlags"])
        ),
        AssignProcessToJobObject=lambda handle, process: calls.append(
            ("assign", handle, process)
        ),
    )
    with patch.dict(
        sys.modules,
        {"win32api": win32api, "win32con": win32con, "win32job": win32job},
    ):
        result = _create_kill_on_close_job(4321)

    assert result is job
    assert ("limits", job, 0x2000) in calls
    assert ("assign", job, process_handle) in calls
    assert ("close", process_handle) in calls
    assert ("close", job) not in calls
