from __future__ import annotations

from pathlib import Path

from local_full_text_search.core.database import DatabaseManager


def test_failure_snapshot_keeps_failed_and_in_progress_counts_consistent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    failed_path = root / "failed.txt"
    pending_path = root / "pending.txt"
    failed_path.write_text("failed", encoding="utf-8")
    pending_path.write_text("pending", encoding="utf-8")
    failed_id, _ = database.upsert_file_metadata(root_id, failed_path)
    pending_id, _ = database.upsert_file_metadata(root_id, pending_path)
    with database.connect() as connection:
        connection.executemany(
            "UPDATE files SET parse_status = ? WHERE id = ?",
            [("failed", failed_id), ("processing", pending_id)],
        )

    snapshot = database.failed_snapshot(page_size=20)

    assert snapshot["failed_counts"] == {
        "blocking": 1,
        "in_progress": 1,
        "excluded": 0,
        "metadata": 0,
    }
    assert [int(row["id"]) for row in snapshot["failed_rows"]] == [failed_id]
    assert snapshot["all_blocking_ids"] == [failed_id]
