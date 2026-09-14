from pathlib import Path

import pytest

import local_full_text_search.core.usn_journal as usn_journal


def _checkpoint() -> usn_journal.UsnCheckpoint:
    return usn_journal.UsnCheckpoint(
        journal_id=1,
        next_usn=2,
        volume_serial=3,
    )


def test_changes_since_preserves_helper_change_set_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direct_error = usn_journal.UsnJournalUnavailable("cannot open NTFS volume (5)")

    def fail_direct(*_args, **_kwargs):
        raise direct_error

    helper_error = usn_journal.UsnChangeSetLimitExceeded(
        limit=20_000,
        records_seen=20_001,
    )

    def fail_helper(*_args, **_kwargs):
        raise helper_error

    monkeypatch.setattr(usn_journal, "_iter_changes_since_direct", fail_direct)
    monkeypatch.setattr(usn_journal, "_service_request", fail_helper)

    with pytest.raises(usn_journal.UsnChangeSetLimitExceeded) as raised:
        usn_journal.changes_since(Path("E:/scope"), _checkpoint())

    assert raised.value is helper_error
    assert raised.value.limit == 20_000
    assert raised.value.records_seen == 20_001
    assert "USN change set exceeds safe fast-path limit" in str(raised.value)


def test_changes_since_reports_both_direct_and_helper_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_direct(*_args, **_kwargs):
        raise usn_journal.UsnJournalUnavailable("cannot open NTFS volume (5)")

    def fail_helper(*_args, **_kwargs):
        raise usn_journal.UsnJournalUnavailable("USN helper service is unavailable (2)")

    monkeypatch.setattr(usn_journal, "_iter_changes_since_direct", fail_direct)
    monkeypatch.setattr(usn_journal, "_service_request", fail_helper)

    with pytest.raises(usn_journal.UsnJournalUnavailable) as raised:
        usn_journal.changes_since(Path("E:/scope"), _checkpoint())

    message = str(raised.value)
    assert "cannot open NTFS volume (5)" in message
    assert "USN helper service is unavailable (2)" in message


def test_service_error_payload_is_decoded_as_change_set_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePipeError(Exception):
        winerror = 0

    class FakePyWinTypes:
        error = FakePipeError

    class FakeWin32File:
        GENERIC_READ = 0x80000000
        GENERIC_WRITE = 0x40000000
        OPEN_EXISTING = 3

        @staticmethod
        def CreateFile(*_args, **_kwargs):
            return object()

        @staticmethod
        def CloseHandle(_handle):
            return None

        @staticmethod
        def WriteFile(*_args, **_kwargs):
            return 0, 0

    class FakeWin32Pipe:
        @staticmethod
        def WaitNamedPipe(*_args, **_kwargs):
            return None

    response = (
        b'{"ok":false,"error_code":"USN_CHANGE_SET_LIMIT",'
        b'"error":"UsnChangeSetLimitExceeded: USN change set exceeds safe fast-path limit",'
        b'"details":{"limit":20000,"records_seen":20001}}'
    )
    chunks = [
        len(response).to_bytes(4, "little"),
        response,
    ]

    def read_exact(_handle, _size):
        return chunks.pop(0)

    monkeypatch.setitem(__import__("sys").modules, "pywintypes", FakePyWinTypes)
    monkeypatch.setitem(__import__("sys").modules, "win32file", FakeWin32File)
    monkeypatch.setitem(__import__("sys").modules, "win32pipe", FakeWin32Pipe)
    monkeypatch.setattr(usn_journal, "_write_all", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(usn_journal, "_read_exact", read_exact)
    monkeypatch.setattr(usn_journal.time, "monotonic", lambda: 0.0)

    with pytest.raises(usn_journal.UsnChangeSetLimitExceeded) as raised:
        usn_journal._service_request({"operation": "changes_since"})

    assert raised.value.limit == 20_000
    assert raised.value.records_seen == 20_001


def test_changes_since_reduces_each_page_to_the_latest_file_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _checkpoint()
    middle = usn_journal.UsnCheckpoint(1, 10, 3)
    target = usn_journal.UsnCheckpoint(1, 20, 3)
    record_old = usn_journal.UsnRecord(11, 101, 4, 1, 0, "old.txt")
    record_new = usn_journal.UsnRecord(11, 102, 5, 2, 0, "new.txt")
    record_other = usn_journal.UsnRecord(12, 101, 6, 1, 0, "other.txt")

    def pages(*_args, **_kwargs):
        yield usn_journal.UsnChangePage(
            checkpoint=middle,
            target_checkpoint=target,
            records=(record_old, record_other),
        )
        yield usn_journal.UsnChangePage(
            checkpoint=target,
            target_checkpoint=target,
            records=(record_new,),
        )

    monkeypatch.setattr(usn_journal, "_iter_changes_since_direct", pages)

    result = usn_journal.changes_since(Path("E:/scope"), first)

    assert result.checkpoint == target
    assert {
        record.file_reference: record.name for record in result.records
    } == {11: "new.txt", 12: "other.txt"}


def test_service_pagination_keeps_one_stable_target_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _checkpoint()
    middle = usn_journal.UsnCheckpoint(1, 10, 3)
    target = usn_journal.UsnCheckpoint(1, 20, 3)
    calls: list[dict[str, object]] = []
    responses = [
        {
            "checkpoint": {
                "journal_id": 1,
                "next_usn": 10,
                "volume_serial": 3,
            },
            "target_checkpoint": {
                "journal_id": 1,
                "next_usn": 20,
                "volume_serial": 3,
            },
            "records": [],
        },
        {
            "checkpoint": {
                "journal_id": 1,
                "next_usn": 20,
                "volume_serial": 3,
            },
            "target_checkpoint": {
                "journal_id": 1,
                "next_usn": 20,
                "volume_serial": 3,
            },
            "records": [],
        },
    ]

    def fail_direct(*_args, **_kwargs):
        raise usn_journal.UsnJournalUnavailable("direct access unavailable")

    def page_request(payload):
        calls.append(payload)
        return responses.pop(0)

    monkeypatch.setattr(usn_journal, "_iter_changes_since_direct", fail_direct)
    monkeypatch.setattr(usn_journal, "_service_request", page_request)

    pages = list(usn_journal.iter_changes_since(Path("E:/scope"), first))

    assert [page.checkpoint for page in pages] == [middle, target]
    assert calls[0]["operation"] == "changes_since_page"
    assert "target_checkpoint" not in calls[0]
    assert calls[1]["target_checkpoint"] == {
        "journal_id": 1,
        "next_usn": 20,
        "volume_serial": 3,
    }


def test_service_dispatches_a_bounded_change_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from local_full_text_search.services import usn_service

    page = usn_journal.UsnChangePage(
        checkpoint=usn_journal.UsnCheckpoint(1, 10, 3),
        target_checkpoint=usn_journal.UsnCheckpoint(1, 20, 3),
        records=(usn_journal.UsnRecord(11, 12, 9, 1, 0, "page.txt"),),
    )
    monkeypatch.setattr(
        usn_service,
        "_changes_since_page_direct",
        lambda *_args, **_kwargs: page,
    )

    result = usn_service.dispatch_request(
        {
            "operation": "changes_since_page",
            "root_path": "E:/scope",
            "checkpoint": {
                "journal_id": 1,
                "next_usn": 2,
                "volume_serial": 3,
            },
        }
    )

    assert result["checkpoint"]["next_usn"] == 10
    assert result["target_checkpoint"]["next_usn"] == 20
    assert result["records"][0]["name"] == "page.txt"
