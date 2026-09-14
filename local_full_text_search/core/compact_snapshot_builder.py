"""Build a compact immutable search snapshot from the operational index DB."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import sqlite3
import time
import traceback
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.core.compact_snapshot_schema import (
    COMPACT_SNAPSHOT_SCHEMA_VERSION,
    configure_new_database,
    create_compact_snapshot_schema,
)
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.text_blob_codec import DEFAULT_TEXT_CODEC


def _process_is_alive(pid: int) -> bool:
    """Check a Windows PID without sending a signal to it."""

    if int(pid) <= 0:
        return False
    try:
        import psutil

        return bool(psutil.pid_exists(int(pid)))
    except Exception:
        # This is only a fallback for environments without psutil.  Windows
        # does not support POSIX signal probing consistently, so treat an
        # uncertain result as alive and let the parent watchdog decide.
        return True


def _code(value: object) -> int:
    digest = hashlib.blake2s(str(value or "").encode("utf-8"), digest_size=2).digest()
    return max(1, int.from_bytes(digest, "big"))


def _source_code(value: object) -> int:
    known = {
        "native_text": 1,
        "ocr": 2,
        "metadata": 3,
        "file": 4,
        "zip_member": 5,
    }
    return known.get(str(value or "").lower(), 1)


def _source_kind(value: object) -> str:
    """Restore the operational text source kind from compact integer codes."""

    try:
        code = int(value)
    except (TypeError, ValueError):
        return str(value or "native_text")
    return {
        1: "native_text",
        2: "ocr",
        3: "metadata",
        4: "file",
        5: "zip_member",
    }.get(code, "native_text")


def _file_source_code(row: sqlite3.Row) -> int:
    """Preserve the file-table meaning of source_kind independently of block source."""

    # Older snapshots encoded every unknown file source as 1.  The container
    # relationship is authoritative for file rows and lets us repair those
    # snapshots while keeping the compact representation numeric.
    if row["container_file_id"] is not None:
        return 5
    return 4


def _file_source_kind(value: object, container_file_id: object) -> str:
    if container_file_id is not None:
        return "zip_member"
    # A top-level compact file is always a physical file, even when an older
    # snapshot stored its block source code (1) in this column.
    return "file"


def _hash_blob(value: object) -> bytes:
    text = str(value or "")
    try:
        return bytes.fromhex(text) if len(text) == 64 else hashlib.sha256(text.encode()).digest()
    except ValueError:
        return hashlib.sha256(text.encode()).digest()


def _snapshot_timestamp(source: sqlite3.Connection) -> str:
    """Return the publication time used for reconstructed operational rows."""

    try:
        value = source.execute(
            "SELECT published_at FROM snapshot_meta WHERE id = 1"
        ).fetchone()[0]
    except (IndexError, TypeError, sqlite3.Error):
        value = None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(
                float(value), timezone.utc
            ).isoformat()
        except (OverflowError, OSError, ValueError):
            pass
    text = str(value or "").strip()
    if text and not text.startswith("1970-01-01"):
        return text
    return datetime.now(timezone.utc).isoformat()


def build_compact_snapshot(
    source_path: Path,
    target_path: Path,
    *,
    generation: int,
    published_at: int | None = None,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, int]:
    """Copy only searchable data, compressing text and rebuilding small FTS."""

    source = sqlite3.connect(
        f"file:{Path(source_path).resolve().as_posix()}?mode=ro",
        uri=True,
        timeout=5,
    )
    source.row_factory = sqlite3.Row
    target_path = Path(target_path)
    target_path.unlink(missing_ok=True)
    target = sqlite3.connect(target_path)
    target.row_factory = sqlite3.Row
    target.create_function("decode_text", 2, lambda payload, codec: "")
    try:
        configure_new_database(target)
        create_compact_snapshot_schema(target)
        target.execute("BEGIN")
        roots = source.execute(
            "SELECT id, path, include_subfolders FROM roots WHERE enabled = 1 ORDER BY id"
        ).fetchall()
        root_ids = {int(row["id"]) for row in roots}
        for row in roots:
            target.execute(
                "INSERT INTO roots(id, path, include_subfolders) VALUES (?, ?, ?)",
                (int(row["id"]), str(row["path"]), int(row["include_subfolders"] or 0)),
            )

        files = source.execute(
            """
            SELECT f.*
            FROM files f
            WHERE f.is_deleted = 0
              AND f.root_id IN ({roots})
              AND f.parse_status IN ('success', 'metadata_only')
              AND NOT EXISTS (
                  SELECT 1 FROM index_scope_exclusions e
                  WHERE e.file_id = f.id
                    AND e.revoked_at IS NULL
                    AND e.invalidated_at IS NULL
              )
            ORDER BY f.id
            """.format(roots=",".join("?" for _ in root_ids) or "NULL"),
            tuple(sorted(root_ids)),
        ).fetchall()
        extension_ids: dict[str, int] = {}
        for row in files:
            extension = str(row["extension"] or "").lower()
            if extension not in extension_ids:
                extension_ids[extension] = len(extension_ids) + 1
                target.execute(
                    "INSERT INTO extensions(id, value) VALUES (?, ?)",
                    (extension_ids[extension], extension),
                )
        file_ids = {int(row["id"]) for row in files}
        document_ids = {
            int(row["document_id"])
            for row in files
            if row["document_id"] is not None
        }
        documents = source.execute(
            "SELECT * FROM documents WHERE id IN ({})".format(
                ",".join("?" for _ in document_ids) or "NULL"
            ),
            tuple(sorted(document_ids)),
        ).fetchall()
        document_map = {int(row["id"]): row for row in documents}
        for row in files:
            if row["document_id"] is not None:
                continue
            # Metadata-only/file-owned blocks still need a stable document
            # owner in the compact snapshot.
            document_id = -int(row["id"])
            document_ids.add(document_id)
            document_map[document_id] = row

        for document_id, row in sorted(document_map.items()):
            target.execute(
                """
                INSERT INTO documents(
                    id, content_hash, parser_kind, parser_version,
                    block_count, text_chars
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    document_id,
                    _hash_blob(row["content_hash"] if "content_hash" in row.keys() else row["content_key"]),
                    _code(row["parser_name"] if "parser_name" in row.keys() else "file"),
                    0,
                    int(row["block_count"] or 0) if "block_count" in row.keys() else 0,
                    int(row["text_chars"] or 0) if "text_chars" in row.keys() else 0,
                ),
            )
        for row in files:
            target.execute(
                """
                INSERT INTO files(
                    id, root_id, document_id, path, filename, extension,
                    extension_id, size_bytes, modified_time, modified_ns,
                    quick_fingerprint, parser_name, parser_version, content_hash,
                    parse_status, source_kind, container_file_id,
                    member_order, file_reference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(row["id"]), int(row["root_id"]),
                    int(row["document_id"]) if row["document_id"] is not None else -int(row["id"]),
                    str(row["path"]), str(row["filename"]), row["extension"],
                    extension_ids[str(row["extension"] or "").lower()],
                    int(row["size_bytes"] or 0), float(row["modified_time"] or 0),
                    int(float(row["modified_time"] or 0) * 1_000_000_000),
                    str(
                        row["quick_fingerprint"]
                        if "quick_fingerprint" in row.keys()
                        else f"{int(row['size_bytes'] or 0)}:{int(float(row['modified_time'] or 0) * 1_000_000_000)}"
                    ),
                    str(row["parser_name"] or "") if "parser_name" in row.keys() else "",
                    str(row["parser_version"] or "") if "parser_version" in row.keys() else "",
                    str(row["content_hash"] or "") if "content_hash" in row.keys() and row["content_hash"] else None,
                    str(row["parse_status"]), _file_source_code(row),
                    row["container_file_id"], row["member_order"], row["file_reference"],
                ),
            )
            target.execute(
                "INSERT INTO files_fts(rowid, normalized_path) VALUES (?, ?)",
                (int(row["id"]), normalize_text(str(row["path"]))),
            )

        selected_documents = set(document_ids)
        blocks = source.execute(
            "SELECT * FROM content_blocks ORDER BY id"
        ).fetchall()
        location_ids: dict[tuple[object, ...], int] = {}
        next_location_id = 1
        block_count = 0
        for row in blocks:
            source_document = row["document_id"]
            if source_document is None:
                if int(row["file_id"] or 0) not in file_ids:
                    continue
                document_id = -int(row["file_id"])
            else:
                document_id = int(source_document)
                if document_id not in selected_documents:
                    continue
            key = (
                document_id, str(row["location_text"] or ""), row["page_number"],
                row["slide_number"], row["sheet_name"], row["cell_start"],
                row["cell_end"], row["line_start"], row["line_end"],
            )
            location_id = location_ids.get(key)
            if location_id is None:
                location_id = next_location_id
                next_location_id += 1
                location_ids[key] = location_id
                target.execute(
                    """
                    INSERT INTO locations(
                        id, document_id, kind, ordinal, label, page_number,
                        slide_number, sheet_name, cell_start, cell_end,
                        line_start, line_end
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        location_id, document_id, _code(row["block_type"]),
                        int(row["block_index"] or 0), row["location_text"],
                        row["page_number"], row["slide_number"], row["sheet_name"],
                        row["cell_start"], row["cell_end"], row["line_start"], row["line_end"],
                    ),
                )
            raw = str(row["raw_text"] or "")
            codec, payload, raw_bytes = DEFAULT_TEXT_CODEC.encode(raw)
            block_id = int(row["id"])
            owner = target.execute(
                "SELECT id FROM files WHERE document_id = ? ORDER BY id LIMIT 1",
                (document_id,),
            ).fetchone()
            if owner is None:
                continue
            target.execute(
                """
                INSERT INTO content_blocks(
                    id, file_id, document_id, location_id, block_order, block_kind,
                    source_kind, codec, raw_chars, text_blob, ocr_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block_id, int(owner[0]), document_id, location_id, int(row["block_index"] or 0),
                    _code(row["block_type"]), _source_code(row["source_type"]), codec,
                    len(raw), payload,
                    int(float(row["ocr_confidence"] or 0) * 10000)
                    if row["ocr_confidence"] is not None else None,
                ),
            )
            normalized = str(row["normalized_text"] or normalize_text(raw))
            target.execute(
                "INSERT INTO content_fts(rowid, normalized_text) VALUES (?, ?)",
                (block_id, normalized),
            )
            block_count += 1
            if progress_callback is not None and block_count % 256 == 0:
                progress_callback({"processed_blocks": block_count})
        timestamp = int(published_at or time.time())
        target.execute(
            """
            INSERT INTO snapshot_meta(
                id, generation, schema_version, published_at, root_count,
                file_count, document_count, block_count, parser_version, state
            ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                int(generation), COMPACT_SNAPSHOT_SCHEMA_VERSION, timestamp,
                len(roots), len(files), len(document_ids), block_count, "compact-v1",
            ),
        )
        target.commit()
        target.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        target.execute("PRAGMA journal_mode=DELETE")
        target.commit()
        return {
            "root_count": len(roots),
            "file_count": len(files),
            "document_count": len(document_ids),
            "block_count": block_count,
        }
    finally:
        target.close()
        source.close()


def expand_compact_snapshot(
    source_path: Path,
    target_path: Path,
    *,
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    scan_id: str = "",
    attempt_id: str = "",
    parent_pid: int | None = None,
) -> None:
    """Materialize a compact active snapshot as the indexer's work schema."""

    # The current parser/index writer still needs task and diagnostic tables.
    # They are created only in the candidate work file and are never published.
    from local_full_text_search.core.database import DatabaseManager
    from local_full_text_search.core.text_blob_codec import decode_text_blob

    source = sqlite3.connect(
        f"file:{Path(source_path).resolve().as_posix()}?mode=ro",
        uri=True,
        timeout=5,
    )
    source.row_factory = sqlite3.Row
    target_path = Path(target_path)
    expanded = target_path.with_name(target_path.name + ".expanded")
    expanded.unlink(missing_ok=True)
    started_at = time.monotonic()

    def parent_is_alive() -> bool:
        if parent_pid is None:
            return True
        return _process_is_alive(int(parent_pid))

    def require_parent_alive() -> None:
        if not parent_is_alive():
            raise RuntimeError(
                "候选库扩展所属父进程已退出，禁止孤儿进程继续写入候选库"
            )

    def progress(phase: str, status: str, **fields: object) -> None:
        # This check is deliberately in the child, not only in the parent
        # watchdog.  If the GUI is terminated by an IDE/installer, the parent
        # can no longer reap us; the child must stop before the next commit or
        # atomic replace instead of continuing to grow an orphan candidate.
        if not (phase == "candidate_expand" and status == "failed"):
            require_parent_alive()
        if progress_callback is None:
            return
        payload: dict[str, object] = {
            "phase": str(phase),
            "phase_status": str(status),
            "scan_id": str(scan_id or ""),
            "attempt_id": str(attempt_id or ""),
            "elapsed_ms": int((time.monotonic() - started_at) * 1000),
        }
        payload.update(fields)
        try:
            progress_callback(payload)
        except BaseException:
            # A dead parent must not turn a disposable candidate into a
            # second error while the parent watchdog is reaping this worker.
            pass
        fault_phase = os.environ.get("LFTS_TEST_CANDIDATE_EXPAND_EXIT_PHASE", "")
        if (
            fault_phase == phase
            and status in {"progress", "completed"}
            and multiprocessing.current_process().name != "MainProcess"
        ):
            os._exit(int(os.environ.get("LFTS_TEST_CANDIDATE_EXPAND_EXIT_CODE", "97")))

    source_counts = {
        "roots": int(source.execute("SELECT COUNT(*) FROM roots").fetchone()[0]),
        "documents": int(source.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
        "files": int(source.execute("SELECT COUNT(*) FROM files").fetchone()[0]),
        "blocks": int(source.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]),
    }
    progress(
        "candidate_expand",
        "started",
        source_path=str(Path(source_path).resolve()),
        target_path=str(target_path.resolve()),
        expanded_path=str(expanded.resolve()),
        source_bytes=Path(source_path).stat().st_size,
        **source_counts,
    )
    target = DatabaseManager(expanded)
    target.initialize()
    progress(
        "schema",
        "completed",
        database_bytes=expanded.stat().st_size if expanded.exists() else 0,
    )
    target_con = sqlite3.connect(expanded)
    target_con.row_factory = sqlite3.Row
    target_con.execute("PRAGMA wal_autocheckpoint = 1000")
    source.create_function("decode_text", 2, decode_text_blob)
    published_at = _snapshot_timestamp(source)
    source_file_columns = {
        str(row[1])
        for row in source.execute("PRAGMA table_info(files)").fetchall()
    }
    quick_fingerprint_expr = (
        "COALESCE(quick_fingerprint, "
        "CAST(size_bytes AS TEXT) || ':' || CAST(modified_ns AS TEXT))"
        if "quick_fingerprint" in source_file_columns
        else "CAST(size_bytes AS TEXT) || ':' || CAST(modified_ns AS TEXT)"
    )
    parser_name_expr = (
        "COALESCE(parser_name, '')"
        if "parser_name" in source_file_columns
        else "''"
    )
    parser_version_expr = (
        "COALESCE(parser_version, '')"
        if "parser_version" in source_file_columns
        else "''"
    )
    content_hash_expr = (
        "content_hash" if "content_hash" in source_file_columns else "NULL"
    )
    document_keys: dict[int, str] = {}
    try:
        target_con.execute("PRAGMA foreign_keys = ON")
        target_con.execute("BEGIN")

        commit_sequence = 0

        def commit_stage(
            source_phase: str,
            *,
            completed: int,
            total: int,
            begin_next: bool = True,
        ) -> None:
            nonlocal commit_sequence
            require_parent_alive()
            target_con.commit()
            commit_sequence += 1
            checkpoint = target_con.execute(
                "PRAGMA wal_checkpoint(PASSIVE)"
            ).fetchone()
            progress(
                "candidate_commit",
                "completed",
                source_phase=source_phase,
                completed=completed,
                total=total,
                committed=True,
                commit_sequence=commit_sequence,
                checkpoint_busy=int(checkpoint[0]) if checkpoint else None,
                checkpoint_log_frames=int(checkpoint[1]) if checkpoint else None,
                checkpointed_frames=int(checkpoint[2]) if checkpoint else None,
                database_bytes=expanded.stat().st_size if expanded.exists() else 0,
                wal_bytes=(
                    Path(str(expanded) + "-wal").stat().st_size
                    if Path(str(expanded) + "-wal").exists()
                    else 0
                ),
            )
            if begin_next:
                target_con.execute("BEGIN")

        inserted_block_count = 0
        for completed, row in enumerate(
            source.execute("SELECT id, path, include_subfolders FROM roots"),
            1,
        ):
            target_con.execute(
                """
                INSERT INTO roots(
                    id, path, enabled, include_subfolders, exclude_rules_json,
                    created_at, updated_at, status
                ) VALUES (?, ?, 1, ?, NULL, ?, ?, 'ready')
                """,
                (
                    int(row["id"]),
                    str(row["path"]),
                    int(row["include_subfolders"] or 0),
                    published_at,
                    published_at,
                ),
            )
            if completed % 64 == 0 or completed == source_counts["roots"]:
                progress(
                    "roots",
                    "progress",
                    completed=completed,
                    total=source_counts["roots"],
                    committed=False,
                )
        progress(
            "roots",
            "completed",
            completed=source_counts["roots"],
            total=source_counts["roots"],
            committed=False,
        )
        commit_stage(
            "roots",
            completed=source_counts["roots"],
            total=source_counts["roots"],
        )
        for completed, row in enumerate(
            source.execute(
                "SELECT id, content_hash, parser_kind, parser_version, block_count, text_chars FROM documents"
            ),
            1,
        ):
            content_hash = bytes(row["content_hash"] or b"").hex()
            document_id = int(row["id"])
            # The compact projection can contain documents with the same
            # content hash but different parser kinds/versions. The
            # operational schema's UNIQUE key is not the compact hash alone;
            # retain the original document identity when expanding it.
            content_key = f"compact:{document_id}:{content_hash}"
            document_keys[document_id] = content_key
            target_con.execute(
                """
                INSERT INTO documents(
                    id, content_key, parser_name, parser_version, parse_status,
                    block_count, text_chars, created_at, updated_at
                ) VALUES (?, ?, 'compact', ?, 'success', ?, ?, ?, ?)
                """,
                (
                    document_id, content_key,
                    str(int(row["parser_version"] or 0)),
                    int(row["block_count"] or 0), int(row["text_chars"] or 0),
                    published_at,
                    published_at,
                ),
            )
            if completed % 128 == 0 or completed == source_counts["documents"]:
                progress(
                    "documents",
                    "progress",
                    completed=completed,
                    total=source_counts["documents"],
                    committed=False,
                )
        progress(
            "documents",
            "completed",
            completed=source_counts["documents"],
            total=source_counts["documents"],
            committed=False,
        )
        commit_stage(
            "documents",
            completed=source_counts["documents"],
            total=source_counts["documents"],
        )
        for completed, row in enumerate(
            source.execute(
                f"""
                SELECT id, root_id, document_id, path, filename, extension,
                       size_bytes, modified_time, modified_ns,
                       {quick_fingerprint_expr} AS quick_fingerprint,
                       {parser_name_expr} AS parser_name,
                       {parser_version_expr} AS parser_version,
                       {content_hash_expr} AS content_hash,
                       parse_status,
                       source_kind, container_file_id, member_order, file_reference
                FROM files ORDER BY id
                """
            ),
            1,
        ):
            document_id = (
                int(row["document_id"])
                if row["document_id"] is not None
                else -int(row["id"])
            )
            target_con.execute(
                """
                INSERT INTO files(
                    id, root_id, document_id, path, filename, extension,
                    size_bytes, modified_time, created_time, quick_fingerprint,
                    content_hash, content_key, parse_status, parser_name,
                    parser_version, indexed_at, last_seen_at, is_deleted,
                    source_kind, container_file_id, member_order, file_reference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                          ?, ?, ?, ?)
                """,
                (
                    int(row["id"]), int(row["root_id"]), document_id,
                    str(row["path"]), str(row["filename"]), row["extension"],
                    int(row["size_bytes"] or 0), float(row["modified_time"] or 0),
                    float(row["modified_time"] or 0),
                    str(row["quick_fingerprint"] or ""),
                    str(row["content_hash"] or "") if row["content_hash"] else None,
                    document_keys.get(
                        document_id,
                        f"compact:file:{int(row['id'])}",
                    ), str(row["parse_status"]),
                    str(row["parser_name"] or "compact"),
                    str(row["parser_version"] or "0"),
                    published_at,
                    published_at,
                    0,
                    _file_source_kind(
                        row["source_kind"],
                        row["container_file_id"],
                    ), row["container_file_id"],
                    row["member_order"], row["file_reference"],
                ),
            )
            if completed % 128 == 0 or completed == source_counts["files"]:
                progress(
                    "files",
                    "progress",
                    completed=completed,
                    total=source_counts["files"],
                    committed=False,
                )
        progress(
            "files",
            "completed",
            completed=source_counts["files"],
            total=source_counts["files"],
            committed=False,
        )
        commit_stage(
            "files",
            completed=source_counts["files"],
            total=source_counts["files"],
        )
        for completed, row in enumerate(
            source.execute("SELECT id, path, filename FROM files"),
            1,
        ):
            target_con.execute(
                "INSERT INTO files_fts(rowid, file_id, filename, path) VALUES (?, ?, ?, ?)",
                (int(row["id"]), int(row["id"]), str(row["filename"]), str(row["path"])),
            )
            if completed % 128 == 0 or completed == source_counts["files"]:
                progress(
                    "files_fts",
                    "progress",
                    completed=completed,
                    total=source_counts["files"],
                    committed=False,
                )
        progress(
            "files_fts",
            "completed",
            completed=source_counts["files"],
            total=source_counts["files"],
            committed=False,
        )
        commit_stage(
            "files_fts",
            completed=source_counts["files"],
            total=source_counts["files"],
        )
        for completed, row in enumerate(
            source.execute(
                """
                SELECT cb.id, cb.document_id, cb.block_order, cb.block_kind,
                       cb.file_id, cb.source_kind, cb.codec, cb.raw_chars, cb.text_blob,
                       cb.ocr_confidence, loc.label AS location_text,
                       loc.page_number, loc.slide_number, loc.sheet_name,
                       loc.cell_start, loc.cell_end, loc.line_start, loc.line_end
                FROM content_blocks cb
                LEFT JOIN locations loc ON loc.id = cb.location_id
                ORDER BY cb.id
                """
            ),
            1,
        ):
            raw = decode_text_blob(row["text_blob"], int(row["codec"]))
            normalized = normalize_text(raw)
            owner = target_con.execute(
                "SELECT id FROM files WHERE document_id = ? ORDER BY id LIMIT 1",
                (row["document_id"],),
            ).fetchone()
            if owner is None:
                continue
            target_con.execute(
                """
                INSERT INTO content_blocks(
                    id, file_id, document_id, block_index, block_type,
                    location_text, page_number, slide_number, sheet_name,
                    cell_start, cell_end, line_start, line_end, raw_text,
                    normalized_text, source_type, ocr_confidence, extra_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '{}', ?)
                """,
                (
                    int(row["id"]), int(owner[0]), row["document_id"], int(row["block_order"]),
                    str(row["block_kind"]), row["location_text"], row["page_number"],
                    row["slide_number"], row["sheet_name"], row["cell_start"],
                    row["cell_end"], row["line_start"], row["line_end"], raw,
                    normalized, _source_kind(row["source_kind"]),
                    (float(row["ocr_confidence"]) / 10000.0)
                    if row["ocr_confidence"] is not None else None,
                    published_at,
                ),
            )
            target_con.execute(
                """
                INSERT INTO content_fts(
                    rowid, block_id, file_id, filename, path,
                    location_text, normalized_text
                )
                SELECT ?, ?, f.id, f.filename, f.path, ?, ?
                FROM files f
                WHERE f.document_id = ? OR f.id = (
                    SELECT file_id FROM content_blocks WHERE id = ?
                )
                LIMIT 1
                """,
                (
                    int(row["id"]), int(row["id"]), row["location_text"], normalized,
                    row["document_id"], int(row["id"]),
                ),
            )
            inserted_block_count += 1
            if completed % 256 == 0 or completed == source_counts["blocks"]:
                progress(
                    "content_blocks_fts",
                    "progress",
                    completed=completed,
                    total=source_counts["blocks"],
                    inserted=inserted_block_count,
                    committed=False,
                )
            if completed % 512 == 0:
                commit_stage(
                    "content_blocks_fts",
                    completed=completed,
                    total=source_counts["blocks"],
                )
        copied_block_count = int(
            target_con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0]
        )
        progress(
            "content_blocks_fts",
            "completed",
            completed=source_counts["blocks"],
            total=source_counts["blocks"],
            inserted=copied_block_count,
            committed=False,
        )
        # The compact source already contains the complete immutable FTS
        # projection.  Mark the reconstructed operational candidate clean;
        # otherwise initialization leaves generation=0 and the readiness gate
        # incorrectly reports content_fts_dirty/projection_incomplete.
        target._set_index_state(target_con, "content_fts_dirty", "0")
        target._set_index_state(target_con, "full_batch_incomplete", "0")
        target._set_index_state(
            target_con,
            "partial_search_projection_active",
            "0",
        )
        target._set_index_state(
            target_con,
            "search_projection_block_count",
            str(copied_block_count),
        )
        if copied_block_count:
            target._advance_search_projection(
                target_con,
                updated_at=published_at,
            )
        progress(
            "candidate_commit",
            "started",
            completed=copied_block_count,
            total=source_counts["blocks"],
            wal_bytes=(
                Path(str(expanded) + "-wal").stat().st_size
                if Path(str(expanded) + "-wal").exists()
                else 0
            ),
        )
        commit_stage(
            "content_blocks_fts_final",
            completed=copied_block_count,
            total=source_counts["blocks"],
            begin_next=False,
        )
        target_con.close()
        target_con = None  # type: ignore[assignment]
        require_parent_alive()
        progress("rebuild_counters", "started", committed=True)
        target.rebuild_index_counters()
        progress("rebuild_counters", "completed", committed=True)
        progress("replace", "started", committed=True)
        require_parent_alive()
        os.replace(expanded, target_path)
        progress(
            "replace",
            "completed",
            committed=True,
            database_bytes=target_path.stat().st_size if target_path.exists() else 0,
        )
    except BaseException as exc:
        progress(
            "candidate_expand",
            "failed",
            error_type=type(exc).__name__,
            error_message=str(exc)[:2000],
            traceback=traceback.format_exc()[-8000:],
        )
        raise
    finally:
        if target_con is not None:
            target_con.rollback()
            target_con.close()
        source.close()


def _expand_compact_snapshot_process_entry(
    source_path: str,
    target_path: str,
    scan_id: str,
    attempt_id: str,
    parent_pid: int,
    sender: object,
) -> None:
    """Child-process entry point; isolate candidate expansion from the GUI."""

    def send(kind: str, payload: dict[str, object]) -> None:
        try:
            getattr(sender, "send")((kind, payload))
        except BaseException:
            pass

    try:
        expand_compact_snapshot(
            Path(source_path),
            Path(target_path),
            progress_callback=lambda payload: send("progress", payload),
            scan_id=scan_id,
            attempt_id=attempt_id,
            parent_pid=parent_pid,
        )
    except BaseException as exc:
        send(
            "error",
            {
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:2000],
                "traceback": traceback.format_exc()[-8000:],
            },
        )
        raise
    else:
        send(
            "completed",
            {"phase": "candidate_expand", "phase_status": "completed"},
        )
    finally:
        try:
            getattr(sender, "close")()
        except BaseException:
            pass


def expand_compact_snapshot_isolated(
    source_path: Path,
    target_path: Path,
    *,
    scan_id: str = "",
    attempt_id: str = "",
    progress_callback: Callable[[dict[str, object]], None] | None = None,
    timeout_seconds: float = 900.0,
    stall_seconds: float = 180.0,
    parent_pid: int | None = None,
) -> None:
    """Run expansion behind a parent-owned timeout and child reaper."""

    context = multiprocessing.get_context("spawn")
    receiver, sender = context.Pipe(False)
    process = context.Process(
        target=_expand_compact_snapshot_process_entry,
        args=(
            str(source_path),
            str(target_path),
            str(scan_id),
            str(attempt_id),
            int(parent_pid if parent_pid is not None else os.getpid()),
            sender,
        ),
        name="lfts-candidate-expand",
    )
    process.daemon = False
    started_at = time.monotonic()
    last_actual_progress = started_at
    last_payload: dict[str, object] = {}
    worker_error: dict[str, object] | None = None

    def has_message() -> bool:
        try:
            return bool(receiver.poll())
        except (EOFError, BrokenPipeError, OSError):
            # Windows reports ERROR_BROKEN_PIPE after the child closes the
            # send endpoint. The process exit code below remains authoritative.
            return False

    try:
        process.start()
        sender.close()
        if progress_callback is not None:
            progress_callback(
                {
                    "phase": "worker_start",
                    "phase_status": "completed",
                    "worker_pid": process.pid,
                    "elapsed_ms": 0,
                }
            )
        while process.is_alive() or has_message():
            try:
                message_available = receiver.poll(0.25)
            except (EOFError, BrokenPipeError, OSError):
                message_available = False
            if message_available:
                while has_message():
                    kind, payload = receiver.recv()
                    if isinstance(payload, dict):
                        last_payload = dict(payload)
                    if kind == "progress":
                        last_actual_progress = time.monotonic()
                        if progress_callback is not None and isinstance(payload, dict):
                            progress_callback(
                                {
                                    **payload,
                                    "worker_pid": process.pid,
                                    "worker_elapsed_ms": int(
                                        (time.monotonic() - started_at) * 1000
                                    ),
                                }
                            )
                    elif kind == "error" and isinstance(payload, dict):
                        worker_error = dict(payload)
            now = time.monotonic()
            if now - started_at > max(1.0, float(timeout_seconds)):
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                raise TimeoutError(
                    "候选库扩展超过父进程总预算；"
                    f"last_phase={last_payload.get('phase', '')}"
                )
            if now - last_actual_progress > max(1.0, float(stall_seconds)):
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)
                raise TimeoutError(
                    "候选库扩展长时间没有实际进度；"
                    f"last_phase={last_payload.get('phase', '')}"
                )
        process.join(5)
        if worker_error is not None:
            raise RuntimeError(
                "候选库扩展子进程失败："
                f"{worker_error.get('error_type', 'UnknownError')}: "
                f"{worker_error.get('error_message', '未知错误')}"
            )
        if process.exitcode != 0:
            raise RuntimeError(
                "候选库扩展子进程意外退出："
                f"exit_code={process.exitcode};"
                f"last_phase={last_payload.get('phase', '')}"
            )
    except BaseException:
        if progress_callback is not None:
            try:
                progress_callback(
                    {
                        "phase": "worker_exit",
                        "phase_status": "failed",
                        "worker_pid": process.pid,
                        "worker_exit_code": process.exitcode,
                        "last_phase": last_payload.get("phase", ""),
                        "last_actual_progress": last_payload,
                        "elapsed_ms": int((time.monotonic() - started_at) * 1000),
                    }
                )
            except BaseException:
                pass
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
        raise
    finally:
        try:
            receiver.close()
        except BaseException:
            pass
        try:
            sender.close()
        except BaseException:
            pass
