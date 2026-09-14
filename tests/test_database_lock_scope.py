from __future__ import annotations

import threading
from pathlib import Path

from local_full_text_search.core.database import DatabaseManager


def test_process_writer_lock_is_scoped_to_normalized_database_path(
    tmp_path: Path,
) -> None:
    first = DatabaseManager(tmp_path / "first.db")
    second = DatabaseManager(tmp_path / "second.db")
    first.initialize()
    second.initialize()
    entered = threading.Event()
    release = threading.Event()

    def hold_first_database() -> None:
        with first.connect(timeout_seconds=1.0, operation="test_hold"):
            entered.set()
            release.wait(timeout=2.0)

    thread = threading.Thread(target=hold_first_database)
    thread.start()
    assert entered.wait(timeout=1.0)
    try:
        with second.connect(timeout_seconds=0.1, operation="test_other_db") as con:
            con.execute("SELECT 1")
    finally:
        release.set()
        thread.join(timeout=2.0)

    assert not thread.is_alive()
