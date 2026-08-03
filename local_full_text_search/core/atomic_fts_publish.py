from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass

from local_full_text_search.config.constants import VIDEO_EXTENSIONS
from local_full_text_search.core.database import (
    TERMINAL_PARSE_TASK_STATUSES,
    DatabaseManager,
    utc_now,
)


class IndexPublishError(RuntimeError):
    pass


class IndexPublishGateError(IndexPublishError):
    def __init__(self, failed_gates: list[str]) -> None:
        self.failed_gates = tuple(dict.fromkeys(failed_gates))
        super().__init__(
            "Candidate index cannot be published: "
            + ", ".join(self.failed_gates)
        )


@dataclass(frozen=True, slots=True)
class IndexPublishResult:
    version_id: int
    document_count: int
    block_count: int
    fts_row_count: int
    content_digest: str


class IndexVersionPublisher:
    """Build and atomically activate one complete, internally consistent FTS."""

    CANDIDATE_TABLE = "content_fts_candidate"

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def begin_candidate(
        self,
        *,
        root_id: int,
        run_id: str,
        version_key: str,
    ) -> int:
        now = utc_now()
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE index_versions
                SET status = 'discarded', failed_at = ?,
                    error_message = ?
                WHERE root_id = ?
                  AND version_key != ?
                  AND status IN ('staging', 'failed')
                """,
                (
                    now,
                    (
                        "Safely discarded before resumed candidate "
                        f"run {run_id}"
                    ),
                    int(root_id),
                    str(version_key),
                ),
            )
            # The candidate FTS table is never active. A process crash may
            # leave one behind, so a new or resumed run always rebuilds it
            # from the complete candidate content snapshot.
            self._drop_candidate(con)
            row = con.execute(
                "SELECT id, status FROM index_versions WHERE version_key = ?",
                (str(version_key),),
            ).fetchone()
            if row is not None:
                if str(row["status"]) not in {"staging", "failed"}:
                    raise IndexPublishError(
                        f"Index version is not recoverable: {version_key}"
                    )
                con.execute(
                    """
                    UPDATE index_versions
                    SET root_id = ?, run_id = ?, status = 'staging',
                        failed_at = NULL, error_message = NULL
                    WHERE id = ?
                    """,
                    (int(root_id), str(run_id), int(row["id"])),
                )
                return int(row["id"])
            cursor = con.execute(
                """
                INSERT INTO index_versions(
                    root_id, run_id, version_key, status, created_at
                ) VALUES (?, ?, ?, 'staging', ?)
                """,
                (int(root_id), str(run_id), str(version_key), now),
            )
            return int(cursor.lastrowid)

    def publish(
        self,
        version_id: int,
        *,
        writer_idle: bool,
        workers_idle: bool,
        golden_query_gate: Callable[[sqlite3.Connection], bool],
        failpoint: str = "",
    ) -> IndexPublishResult:
        failed_gates = self._failed_gates(
            int(version_id),
            writer_idle=writer_idle,
            workers_idle=workers_idle,
            golden_query_gate=golden_query_gate,
        )
        if failed_gates:
            raise IndexPublishGateError(failed_gates)
        try:
            with self.database.connect() as con:
                version = con.execute(
                    """
                    SELECT id, root_id, run_id, status
                    FROM index_versions WHERE id = ?
                    """,
                    (int(version_id),),
                ).fetchone()
                if version is None or str(version["status"]) != "staging":
                    raise IndexPublishError(
                        f"Candidate index is unavailable: {version_id}"
                    )
                con.execute(
                    """
                    UPDATE content_blocks
                    SET index_version_id = ?
                    """,
                    (int(version_id),),
                )
                self._drop_candidate(con)
                self._create_candidate(con)
                con.execute(
                    f"""
                    INSERT INTO {self.CANDIDATE_TABLE}(
                        rowid, block_id, file_id, filename, path,
                        location_text, normalized_text
                    )
                    SELECT cb.id, cb.id, cb.file_id, COALESCE(f.filename, ''),
                           COALESCE(f.path, ''), cb.location_text,
                           cb.normalized_text
                    FROM content_blocks cb
                    LEFT JOIN files f ON f.id = cb.file_id
                    WHERE cb.index_version_id = ?
                      AND EXISTS (
                          SELECT 1
                          FROM files visible
                          WHERE visible.is_deleted = 0
                            AND (
                                (cb.document_id IS NOT NULL AND visible.document_id = cb.document_id)
                                OR (cb.document_id IS NULL AND visible.id = cb.file_id)
                            )
                            AND NOT EXISTS (
                                SELECT 1 FROM index_scope_exclusions excluded
                                WHERE excluded.file_id = visible.id
                                  AND excluded.revoked_at IS NULL
                                  AND excluded.invalidated_at IS NULL
                            )
                      )
                    ORDER BY cb.id
                    """,
                    (int(version_id),),
                )
                block_count = int(
                    con.execute(
                        """
                        SELECT COUNT(*) FROM content_blocks
                        WHERE index_version_id = ?
                          AND EXISTS (
                              SELECT 1
                              FROM files visible
                              WHERE visible.is_deleted = 0
                                AND (
                                    (content_blocks.document_id IS NOT NULL
                                     AND visible.document_id = content_blocks.document_id)
                                    OR (content_blocks.document_id IS NULL
                                        AND visible.id = content_blocks.file_id)
                                )
                                AND NOT EXISTS (
                                    SELECT 1 FROM index_scope_exclusions excluded
                                    WHERE excluded.file_id = visible.id
                                      AND excluded.revoked_at IS NULL
                                      AND excluded.invalidated_at IS NULL
                                )
                          )
                        """,
                        (int(version_id),),
                    ).fetchone()[0]
                )
                fts_row_count = int(
                    con.execute(
                        f"SELECT COUNT(*) FROM {self.CANDIDATE_TABLE}"
                    ).fetchone()[0]
                )
                if fts_row_count != block_count:
                    raise IndexPublishError(
                        "Candidate FTS row count does not match content blocks"
                    )
                unversioned_count = int(
                    con.execute(
                        """
                        SELECT COUNT(*) FROM content_blocks
                        WHERE index_version_id IS NULL
                           OR index_version_id != ?
                        """,
                        (int(version_id),),
                    ).fetchone()[0]
                )
                if unversioned_count:
                    raise IndexPublishError(
                        "Candidate contains mixed content versions"
                    )
                content_digest = _content_digest(
                    con,
                    version_id=int(version_id),
                )
                if failpoint == "after_candidate_build":
                    raise IndexPublishError(
                        "Injected failure after candidate FTS build"
                    )
                con.execute("DELETE FROM content_fts")
                con.execute(
                    f"""
                    INSERT INTO content_fts(
                        rowid, block_id, file_id, filename, path,
                        location_text, normalized_text
                    )
                    SELECT rowid, block_id, file_id, filename, path,
                           location_text, normalized_text
                    FROM {self.CANDIDATE_TABLE}
                    """
                )
                if failpoint == "before_activation":
                    raise IndexPublishError(
                        "Injected failure before active version switch"
                    )
                document_count = int(
                    con.execute(
                        """
                        SELECT COUNT(*) FROM documents d
                        WHERE EXISTS (
                            SELECT 1 FROM files visible
                            WHERE visible.document_id = d.id
                              AND visible.is_deleted = 0
                              AND NOT EXISTS (
                                  SELECT 1 FROM index_scope_exclusions excluded
                                  WHERE excluded.file_id = visible.id
                                    AND excluded.revoked_at IS NULL
                                    AND excluded.invalidated_at IS NULL
                              )
                        )
                        """
                    ).fetchone()[0]
                )
                now = utc_now()
                con.execute(
                    """
                    UPDATE index_versions
                    SET status = 'superseded'
                    WHERE status = 'active' AND id != ?
                    """,
                    (int(version_id),),
                )
                con.execute(
                    """
                    UPDATE index_versions
                    SET status = 'active', document_count = ?,
                        block_count = ?, content_digest = ?,
                        activated_at = ?, failed_at = NULL,
                        error_message = NULL
                    WHERE id = ?
                    """,
                    (
                        document_count,
                        block_count,
                        content_digest,
                        now,
                        int(version_id),
                    ),
                )
                con.execute(
                    """
                    UPDATE index_scope_exclusions
                    SET candidate_index_version_id = COALESCE(
                            candidate_index_version_id, ?
                        ),
                        published_index_version_id = ?
                    WHERE root_id = ?
                      AND revoked_at IS NULL
                      AND invalidated_at IS NULL
                    """,
                    (
                        int(version_id),
                        int(version_id),
                        int(version["root_id"]),
                    ),
                )
                _set_state(con, "active_index_version", str(int(version_id)))
                _set_state(con, "content_fts_dirty", "0")
                _set_state(con, "full_batch_incomplete", "0")
                if version["root_id"] is not None:
                    con.execute(
                        """
                        UPDATE roots SET status = ?, last_scan_at = ?
                        WHERE id = ?
                        """,
                        ("ready", now, int(version["root_id"])),
                    )
                self._drop_candidate(con)
                return IndexPublishResult(
                    version_id=int(version_id),
                    document_count=document_count,
                    block_count=block_count,
                    fts_row_count=fts_row_count,
                    content_digest=content_digest,
                )
        except IndexPublishGateError:
            raise
        except Exception as exc:
            with self.database.connect() as con:
                con.execute(
                    """
                    UPDATE index_versions
                    SET status = 'failed', failed_at = ?, error_message = ?
                    WHERE id = ? AND status != 'active'
                    """,
                    (utc_now(), str(exc), int(version_id)),
                )
                self._drop_candidate(con)
            if isinstance(exc, IndexPublishError):
                raise
            raise IndexPublishError(str(exc)) from exc

    def _failed_gates(
        self,
        version_id: int,
        *,
        writer_idle: bool,
        workers_idle: bool,
        golden_query_gate: Callable[[sqlite3.Connection], bool],
    ) -> list[str]:
        failed: list[str] = []
        if not writer_idle:
            failed.append("writer_not_idle")
        if not workers_idle:
            failed.append("workers_not_idle")
        video_extensions = tuple(sorted(VIDEO_EXTENSIONS))
        placeholders = ",".join("?" for _ in video_extensions)
        with self.database.connect() as con:
            version = con.execute(
                "SELECT root_id, run_id, status FROM index_versions WHERE id = ?",
                (int(version_id),),
            ).fetchone()
            if version is None or str(version["status"]) != "staging":
                failed.append("candidate_not_staging")
                return failed
            root_id = int(version["root_id"])
            blocking_files = int(
                con.execute(
                    f"""
                    SELECT COUNT(*) FROM files
                    WHERE root_id = ? AND is_deleted = 0
                      AND extension NOT IN ({placeholders})
                      AND parse_status NOT IN ('success', 'metadata_only')
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions excluded
                          WHERE excluded.file_id = files.id
                            AND excluded.revoked_at IS NULL
                            AND excluded.invalidated_at IS NULL
                      )
                    """,
                    (root_id, *video_extensions),
                ).fetchone()[0]
            )
            if blocking_files:
                failed.append("incomplete_files")
            terminal_placeholders = ",".join(
                "?" for _ in TERMINAL_PARSE_TASK_STATUSES
            )
            unfinished_tasks = int(
                con.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM parse_tasks pt
                    JOIN files f ON f.id = pt.file_id
                    WHERE f.root_id = ?
                      AND pt.run_id = ?
                      AND NOT EXISTS (
                          SELECT 1 FROM index_scope_exclusions excluded
                          WHERE excluded.file_id = f.id
                            AND excluded.revoked_at IS NULL
                            AND excluded.invalidated_at IS NULL
                      )
                      AND pt.status NOT IN ({terminal_placeholders})
                    """,
                    (
                        root_id,
                        str(version["run_id"]),
                        *sorted(TERMINAL_PARSE_TASK_STATUSES),
                    ),
                ).fetchone()[0]
            )
            if unfinished_tasks:
                failed.append("unfinished_tasks")
            integrity = [
                str(row[0]) for row in con.execute("PRAGMA integrity_check")
            ]
            if integrity != ["ok"]:
                failed.append("integrity_check_failed")
            if list(con.execute("PRAGMA foreign_key_check")):
                failed.append("foreign_key_check_failed")
            try:
                golden_ok = bool(golden_query_gate(con))
            except Exception:
                golden_ok = False
            if not golden_ok:
                failed.append("golden_query_failed")
        return failed

    def _create_candidate(self, con: sqlite3.Connection) -> None:
        source = con.execute(
            """
            SELECT sql FROM sqlite_master
            WHERE type = 'table' AND name = 'content_fts'
            """
        ).fetchone()
        if source is None or not source["sql"]:
            raise IndexPublishError("Active content FTS schema is unavailable")
        create_sql = str(source["sql"]).replace(
            "content_fts",
            self.CANDIDATE_TABLE,
            1,
        )
        con.execute(create_sql)

    def _drop_candidate(self, con: sqlite3.Connection) -> None:
        con.execute(f"DROP TABLE IF EXISTS {self.CANDIDATE_TABLE}")


def _content_digest(
    con: sqlite3.Connection,
    *,
    version_id: int | None = None,
) -> str:
    digest = hashlib.sha256()
    where = (
        """WHERE cb.index_version_id = ?
        AND EXISTS (
            SELECT 1 FROM files visible
            WHERE visible.is_deleted = 0
              AND (
                  (cb.document_id IS NOT NULL AND visible.document_id = cb.document_id)
                  OR (cb.document_id IS NULL AND visible.id = cb.file_id)
              )
              AND NOT EXISTS (
                  SELECT 1 FROM index_scope_exclusions excluded
                  WHERE excluded.file_id = visible.id
                    AND excluded.revoked_at IS NULL
                    AND excluded.invalidated_at IS NULL
              )
        )"""
        if version_id is not None
        else """WHERE EXISTS (
            SELECT 1 FROM files visible
            WHERE visible.is_deleted = 0
              AND (
                  (cb.document_id IS NOT NULL AND visible.document_id = cb.document_id)
                  OR (cb.document_id IS NULL AND visible.id = cb.file_id)
              )
              AND NOT EXISTS (
                  SELECT 1 FROM index_scope_exclusions excluded
                  WHERE excluded.file_id = visible.id
                    AND excluded.revoked_at IS NULL
                    AND excluded.invalidated_at IS NULL
              )
        )"""
    )
    params = (
        (int(version_id),)
        if version_id is not None
        else ()
    )
    for row in con.execute(
        f"""
        SELECT cb.id, cb.file_id, cb.block_index, cb.location_text,
               cb.normalized_text
        FROM content_blocks cb
        {where}
        ORDER BY cb.id
        """,
        params,
    ):
        for value in row:
            digest.update(str(value if value is not None else "").encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


def _set_state(con: sqlite3.Connection, key: str, value: str) -> None:
    con.execute(
        """
        INSERT INTO index_state(key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
