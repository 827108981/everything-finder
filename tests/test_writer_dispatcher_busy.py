from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from local_full_text_search.core.index_writer import IndexWriter


def test_retryable_dispatcher_busy_returns_to_writer_busy_circuit() -> None:
    writer = IndexWriter.__new__(IndexWriter)
    writer.write_dispatcher = SimpleNamespace(
        submit=lambda *args, **kwargs: SimpleNamespace(
            status="FAILED_RETRYABLE",
            error="database is locked",
        )
    )

    with pytest.raises(sqlite3.OperationalError, match="database is locked"):
        writer._run_database_write(lambda: None, label="test_busy")
