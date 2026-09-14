from __future__ import annotations

from pathlib import Path
import threading

import pytest

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.errors import PlanningWorkerError
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.core.index_manager import ParseOutcome
from local_full_text_search.core.index_writer import IndexWriter
from local_full_text_search.models.content_block import ContentBlock


def test_large_artifact_defers_fts_to_bounded_sync(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "large.txt"
    source.write_text("large artifact", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)

    replace_calls: list[tuple[bool, bool]] = []
    sync_calls: list[tuple[tuple[int, ...], int, int]] = []
    original_replace = database.replace_document_blocks_many
    original_sync = database.sync_content_fts_for_files

    def record_replace(items, *, update_fts=True, remove_old_fts=False):
        replace_calls.append((bool(update_fts), bool(remove_old_fts)))
        return original_replace(
            items,
            update_fts=update_fts,
            remove_old_fts=remove_old_fts,
        )

    def record_sync(file_ids, *, batch_blocks=64, batch_bytes=512 * 1024):
        ids = tuple(int(value) for value in file_ids)
        sync_calls.append((ids, int(batch_blocks), int(batch_bytes)))
        return original_sync(
            ids,
            batch_blocks=batch_blocks,
            batch_bytes=batch_bytes,
        )

    monkeypatch.setattr(database, "replace_document_blocks_many", record_replace)
    monkeypatch.setattr(database, "sync_content_fts_for_files", record_sync)

    blocks = [
        ContentBlock(
            file_path=str(source),
            block_index=index,
            block_type="paragraph",
            location_text=f"line {index}",
            raw_text=f"LARGE_ARTIFACT_{index}",
            normalized_text=f"large_artifact_{index}",
        )
        for index in range(6)
    ]
    writer = IndexWriter(
        database,
        update_fts=True,
        batch_blocks=500,
        batch_bytes=1024 * 1024,
        max_delay_ms=10,
        large_artifact_fts_threshold_blocks=4,
    )
    writer.submit(
        ParseOutcome(
            file_id=file_id,
            file_path=source,
            blocks=blocks,
            parser_name="text",
            parser_version="test-v1",
            status="success",
            content_key="large-artifact-v1",
        )
    )
    summary = writer.finish()

    assert replace_calls == [(False, True)]
    assert sync_calls == [((file_id,), 4, 512 * 1024)]
    assert summary.artifacts == 1
    assert summary.blocks == 6
    with database.connect_read_only() as con:
        assert int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]) == 6
        assert int(con.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0]) == 6
        dirty = con.execute(
            "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
        ).fetchone()
    assert dirty is not None and str(dirty["value"]) == "0"


def test_combined_medium_artifacts_do_not_form_one_large_fts_transaction(
    tmp_path: Path,
) -> None:
    class RecordingDatabase:
        def __init__(self) -> None:
            self.replace_calls: list[tuple[int, bool, bool]] = []
            self.sync_calls: list[tuple[tuple[int, ...], int, int]] = []

        def replace_document_blocks_many(
            self,
            items,
            *,
            update_fts=True,
            remove_old_fts=False,
        ) -> None:
            self.replace_calls.append(
                (len(items), bool(update_fts), bool(remove_old_fts))
            )

        def sync_content_fts_for_files(
            self,
            file_ids,
            *,
            batch_blocks=64,
            batch_bytes=512 * 1024,
        ) -> int:
            ids = tuple(sorted(int(value) for value in file_ids))
            self.sync_calls.append(
                (ids, int(batch_blocks), int(batch_bytes))
            )
            return 6

        def interrupt_active_connections(self) -> int:
            return 0

    database = RecordingDatabase()
    writer = IndexWriter(
        database,  # type: ignore[arg-type]
        update_fts=True,
        batch_blocks=500,
        batch_bytes=1024 * 1024,
        max_delay_ms=500,
        large_artifact_fts_threshold_blocks=5,
    )
    for file_id in (1, 2):
        source = tmp_path / f"medium-{file_id}.txt"
        blocks = [
            ContentBlock(
                file_path=str(source),
                block_index=index,
                block_type="paragraph",
                location_text=f"line {index}",
                raw_text=f"medium {file_id} {index}",
                normalized_text=f"medium {file_id} {index}",
            )
            for index in range(3)
        ]
        writer.submit(
            ParseOutcome(
                file_id=file_id,
                file_path=source,
                blocks=blocks,
                parser_name="text",
                status="success",
                content_key=f"medium-{file_id}",
            )
        )

    writer.finish()

    assert database.replace_calls == [(2, False, True)]
    assert database.sync_calls == [((1, 2), 5, 512 * 1024)]


def test_explicit_deferred_fts_keeps_large_excel_out_of_write_transaction(
    tmp_path: Path,
) -> None:
    class RecordingDatabase:
        def __init__(self) -> None:
            self.replace_calls: list[tuple[bool, bool]] = []
            self.sync_calls = 0

        def replace_document_blocks_many(
            self,
            _items,
            *,
            update_fts=True,
            remove_old_fts=False,
        ) -> None:
            self.replace_calls.append(
                (bool(update_fts), bool(remove_old_fts))
            )

        def sync_content_fts_for_files(self, _file_ids, **_kwargs) -> int:
            self.sync_calls += 1
            return 0

        def interrupt_active_connections(self) -> int:
            return 0

    source = tmp_path / "large.xls"
    blocks = [
        ContentBlock(
            file_path=str(source),
            block_index=index,
            block_type="cell",
            location_text=f"row {index}",
            raw_text=f"EXCEL_CELL_{index}",
            normalized_text=f"excel_cell_{index}",
        )
        for index in range(4000)
    ]
    database = RecordingDatabase()
    writer = IndexWriter(
        database,  # type: ignore[arg-type]
        update_fts=True,
        defer_fts=True,
        batch_blocks=500,
        batch_bytes=1024 * 1024,
        max_delay_ms=10,
    )
    writer.submit(
        ParseOutcome(
            file_id=62025,
            file_path=source,
            blocks=blocks,
            parser_name="legacy_excel",
            status="success",
            content_key="legacy-excel-62025",
        )
    )

    summary = writer.finish(sync_deferred_fts=False)

    assert summary.blocks == 4000
    assert database.replace_calls == [(False, True)]
    assert database.sync_calls == 0
    assert writer.deferred_fts_file_ids == frozenset({62025})


def test_fts_replay_reports_bounded_progress_without_global_projection_scan(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "small.txt"
    source.write_text("FTS_PROGRESS", encoding="utf-8")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)
    database.replace_document_blocks_many(
        [
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": source.name,
                "path": str(source),
                "blocks": [
                    ContentBlock(
                        file_path=str(source),
                        block_index=index,
                        block_type="paragraph",
                        location_text=f"line {index}",
                        raw_text=f"FTS_PROGRESS_{index}",
                        normalized_text=f"fts_progress_{index}",
                    )
                    for index in range(5)
                ],
                "parser_name": "text",
                "parser_version": "test-v1",
                "status": "success",
                "content_key": "fts-progress-v1",
            }
        ],
        update_fts=False,
        remove_old_fts=True,
    )
    progress: list[tuple[int, int]] = []
    cursors: list[int] = []
    synced = database.sync_content_fts_for_files(
        [file_id],
        batch_blocks=2,
        cursor_callback=cursors.append,
        progress_callback=lambda completed, total: progress.append(
            (completed, total)
        ),
    )

    assert synced == 5
    assert progress[-1] == (5, 5)
    assert cursors == sorted(cursors)
    assert cursors
    with database.connect_read_only() as con:
        dirty = con.execute(
            "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
        ).fetchone()
        marker = con.execute(
            "SELECT 1 FROM index_state WHERE key = ?",
            (f"content_fts_pending:{file_id}",),
        ).fetchone()
    assert dirty is not None and str(dirty["value"]) == "0"
    assert marker is None


def test_file_stat_worker_crash_falls_back_to_parent_stat(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    paths = []
    for index in range(3):
        path = root / f"file-{index}.txt"
        path.write_text(f"file {index}", encoding="utf-8")
        paths.append(str(path))

    class CrashingRunner:
        calls = 0

        def run(self, *_args, **_kwargs):
            self.calls += 1
            raise PlanningWorkerError("file_stat child exited with code 1")

    database = DatabaseManager(tmp_path / "index.db")
    manager = IndexManager(database, AppSettings(enable_ocr=False))
    runner = CrashingRunner()
    manager._planning_runner = runner  # type: ignore[assignment]
    progress = []

    metadata, errors = manager._stat_paths_recoverable(
        paths,
        CancelToken(),
        progress.append,
    )

    assert runner.calls == 2
    assert errors == []
    assert [Path(item.path).name for item in metadata] == [
        "file-0.txt",
        "file-1.txt",
        "file-2.txt",
    ]
    assert progress[-1].completed == 3
    assert progress[-1].total == 3


def test_large_excel_blocks_commit_in_chunks_then_switch_atomically(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "large.xls"
    source.write_bytes(b"legacy excel")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)
    old_block = ContentBlock(
        file_path=str(source),
        block_index=0,
        block_type="cell",
        location_text="row 0",
        raw_text="OLD_VISIBLE_CONTENT",
        normalized_text="old_visible_content",
    )
    database.replace_document_blocks_many(
        [
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": source.name,
                "path": str(source),
                "blocks": [old_block],
                "parser_name": "legacy_excel",
                "parser_version": "test-v1",
                "status": "success",
                "content_key": "large-excel-v1",
            }
        ]
    )
    with database.connect_read_only() as con:
        old_document_id = int(
            con.execute(
                "SELECT document_id FROM files WHERE id = ?", (file_id,)
            ).fetchone()["document_id"]
        )

    blocks = [
        ContentBlock(
            file_path=str(source),
            block_index=index,
            block_type="cell",
            location_text=f"row {index}",
            raw_text=f"NEW_CELL_{index}",
            normalized_text=f"new_cell_{index}",
        )
        for index in range(650)
    ]
    progress: list[int] = []

    def observe_committed_chunk(completed: int, total: int) -> None:
        progress.append(completed)
        assert total == 650
        # A task/telemetry writer can commit between body chunks. This is the
        # lock-starvation regression that the production logs exposed.
        with database.connect(timeout_seconds=0.1) as con:
            con.execute(
                """
                INSERT INTO index_state(key, value) VALUES ('chunk_heartbeat', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (str(completed),),
            )
        with database.connect_read_only() as con:
            file_row = con.execute(
                "SELECT document_id, parse_status FROM files WHERE id = ?",
                (file_id,),
            ).fetchone()
            assert int(file_row["document_id"]) == old_document_id
            assert str(file_row["parse_status"]) == "success"

    original_chunked = database.replace_document_blocks_chunked

    def observed_chunked(item, **kwargs) -> None:
        kwargs["progress_callback"] = observe_committed_chunk
        original_chunked(item, **kwargs)

    monkeypatch.setattr(
        database,
        "replace_document_blocks_chunked",
        observed_chunked,
    )
    writer = IndexWriter(
        database,
        update_fts=True,
        defer_fts=True,
        batch_blocks=500,
        batch_bytes=1024 * 1024,
        max_delay_ms=10,
        artifact_write_chunk_blocks=100,
    )
    writer.submit(
        ParseOutcome(
            file_id=file_id,
            file_path=source,
            blocks=blocks,
            parser_name="legacy_excel",
            parser_version="test-v1",
            status="success",
            content_key="large-excel-v1",
        )
    )
    summary = writer.finish(sync_deferred_fts=False)

    assert progress == [100, 200, 300, 400, 500, 600, 650]
    assert summary.blocks == 650
    with database.connect_read_only() as con:
        file_row = con.execute(
            "SELECT document_id, parse_status FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        assert int(file_row["document_id"]) != old_document_id
        assert str(file_row["parse_status"]) == "success"
        assert int(
            con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
        ) == 650
        assert int(
            con.execute(
                "SELECT COUNT(*) FROM documents WHERE content_key LIKE '%#staging:%'"
            ).fetchone()[0]
        ) == 0
        assert int(
            con.execute(
                "SELECT COUNT(*) FROM documents d WHERE NOT EXISTS "
                "(SELECT 1 FROM files f WHERE f.document_id = d.id)"
            ).fetchone()[0]
        ) == 0
        pending = con.execute(
            "SELECT value FROM index_state WHERE key = ?",
            (f"content_fts_pending:{file_id}",),
        ).fetchone()
    assert pending is not None


def test_cancelled_chunked_write_preserves_old_document_and_cleans_staging(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = root / "cancelled.xls"
    source.write_bytes(b"legacy excel")
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    root_id = database.add_root(root)
    file_id, _changed = database.upsert_file_metadata(root_id, source)
    database.replace_document_blocks_many(
        [
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": source.name,
                "path": str(source),
                "blocks": [
                    ContentBlock(
                        file_path=str(source),
                        block_index=0,
                        block_type="cell",
                        location_text="row 0",
                        raw_text="OLD_CONTENT_SURVIVES",
                        normalized_text="old_content_survives",
                    )
                ],
                "parser_name": "legacy_excel",
                "parser_version": "test-v1",
                "status": "success",
                "content_key": "cancelled-excel-v1",
            }
        ]
    )
    with database.connect_read_only() as con:
        old_document_id = int(
            con.execute(
                "SELECT document_id FROM files WHERE id = ?", (file_id,)
            ).fetchone()["document_id"]
        )

    cancel_event = threading.Event()
    replacement = [
        ContentBlock(
            file_path=str(source),
            block_index=index,
            block_type="cell",
            location_text=f"row {index}",
            raw_text=f"PARTIAL_{index}",
            normalized_text=f"partial_{index}",
        )
        for index in range(400)
    ]

    with pytest.raises(CancelledError):
        database.replace_document_blocks_chunked(
            {
                "file_id": file_id,
                "file_ids": [file_id],
                "filename": source.name,
                "path": str(source),
                "blocks": replacement,
                "parser_name": "legacy_excel",
                "parser_version": "test-v1",
                "status": "success",
                "content_key": "cancelled-excel-v1",
            },
            chunk_blocks=100,
            cancel_event=cancel_event,
            progress_callback=lambda _completed, _total: cancel_event.set(),
        )

    with database.connect_read_only() as con:
        file_row = con.execute(
            "SELECT document_id, parse_status FROM files WHERE id = ?",
            (file_id,),
        ).fetchone()
        assert int(file_row["document_id"]) == old_document_id
        assert str(file_row["parse_status"]) == "success"
        rows = con.execute(
            "SELECT raw_text FROM content_blocks WHERE document_id = ?",
            (old_document_id,),
        ).fetchall()
        assert [str(row["raw_text"]) for row in rows] == ["OLD_CONTENT_SURVIVES"]
        assert int(
            con.execute(
                "SELECT COUNT(*) FROM documents WHERE content_key LIKE '%#staging:%'"
            ).fetchone()[0]
        ) == 0


def test_active_run_without_parse_rows_is_finishing_not_converged(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    root.mkdir()
    database = DatabaseManager(tmp_path / "index.db")
    database.initialize()
    database.add_root(root)
    with database.connect() as con:
        con.execute(
            """
            INSERT INTO index_runs(id, started_at, mode, status)
            VALUES ('finishing-run', '2026-08-28T17:00:00+00:00', 'incremental', 'running')
            """
        )

    readiness = database.index_readiness(reconcile=False)

    assert readiness["active_runs"] == 1
    assert readiness["unfinished_tasks"] == 0
    assert readiness["background_task_state"] == "finishing"
    assert readiness["ready"] is False
