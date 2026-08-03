from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_full_text_search.core.database import DatabaseManager, utc_now


@dataclass(frozen=True, slots=True)
class PdfPagePlan:
    page_number: int
    page_identity: str
    width_points: float
    height_points: float
    requires_ocr: bool


@dataclass(frozen=True, slots=True)
class PdfPageTaskClaim:
    task_id: int
    parent_task_id: int
    file_id: int
    task_type: str
    page_number: int
    source_digest: str
    task_version: str
    lease_owner: str
    lease_expires_at: str
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class PdfMergeReadiness:
    ready: bool
    total_pages: int
    confirmed_pages: int
    failed_pages: int
    pending_pages: int


class PdfTaskGraphRepository:
    """Persistent PDF page task graph with leases and idempotent merge."""

    PAGE_TASK_TYPES = ("pdf_native_page", "pdf_ocr_page")

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def plan_document(
        self,
        *,
        file_id: int,
        run_id: str,
        source_digest: str,
        parser_version: str,
        pages: list[PdfPagePlan],
        ocr_config_fingerprint: str,
    ) -> int:
        now = utc_now()
        with self.database.connect() as con:
            existing = con.execute(
                """
                SELECT id FROM parse_tasks
                WHERE file_id = ? AND parent_task_id IS NULL
                  AND task_type = 'pdf_document'
                  AND source_digest = ? AND task_version = ?
                  AND status NOT IN ('invalidated', 'cancelled')
                ORDER BY id DESC LIMIT 1
                """,
                (file_id, source_digest, parser_version),
            ).fetchone()
            if existing is not None:
                document_task_id = int(existing["id"])
                self._recover_document_in_connection(
                    con,
                    document_task_id=document_task_id,
                    run_id=run_id,
                    now=now,
                )
                return document_task_id
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'invalidated', finished_at = ?
                WHERE file_id = ? AND task_type IN (
                    'pdf_document', 'pdf_scan', 'pdf_native_page',
                    'pdf_ocr_page', 'document_merge'
                ) AND status NOT IN ('invalidated', 'cancelled')
                """,
                (now, file_id),
            )
            con.execute(
                """
                UPDATE pdf_page_identities
                SET invalidated_at = ?
                WHERE file_id = ? AND invalidated_at IS NULL
                """,
                (now, file_id),
            )
            parent = con.execute(
                """
                INSERT INTO parse_tasks(
                    file_id, run_id, task_type, unit_key, payload_json,
                    status, priority, created_at, queued_at,
                    source_digest, task_version
                ) VALUES (?, ?, 'pdf_document', 'document', ?, 'running',
                          100, ?, ?, ?, ?)
                """,
                (
                    file_id,
                    run_id,
                    json.dumps(
                        {
                            "page_count": len(pages),
                            "ocr_config_fingerprint": ocr_config_fingerprint,
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    now,
                    now,
                    source_digest,
                    parser_version,
                ),
            )
            parent_task_id = int(parent.lastrowid)
            scan = con.execute(
                """
                INSERT INTO parse_tasks(
                    file_id, parent_task_id, run_id, task_type, unit_key,
                    payload_json, status, priority, created_at, queued_at,
                    started_at, finished_at, confirmed_at, source_digest,
                    task_version, progress_phase, progress_completed,
                    progress_total, checkpoint_version
                ) VALUES (?, ?, ?, 'pdf_scan', 'scan', ?, 'complete', 1000,
                          ?, ?, ?, ?, ?, ?, ?, 'pdf_scan', ?, ?, ?)
                """,
                (
                    file_id,
                    parent_task_id,
                    run_id,
                    json.dumps(
                        {"page_count": len(pages)},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    now,
                    now,
                    now,
                    now,
                    now,
                    source_digest,
                    parser_version,
                    len(pages),
                    len(pages),
                    len(pages),
                ),
            )
            _ = scan.lastrowid
            for page in pages:
                native_payload = {
                    "page_number": page.page_number,
                    "page_identity": page.page_identity,
                    "width_points": page.width_points,
                    "height_points": page.height_points,
                    "requires_ocr": page.requires_ocr,
                }
                native = self._insert_child(
                    con,
                    file_id=file_id,
                    parent_task_id=parent_task_id,
                    run_id=run_id,
                    task_type="pdf_native_page",
                    unit_key=f"page:{page.page_number}",
                    payload=native_payload,
                    priority=500,
                    now=now,
                    source_digest=source_digest,
                    task_version=parser_version,
                )
                ocr_task_id: int | None = None
                if page.requires_ocr:
                    ocr_payload = {
                        **native_payload,
                        "ocr_config_fingerprint": ocr_config_fingerprint,
                    }
                    ocr_task_id = self._insert_child(
                        con,
                        file_id=file_id,
                        parent_task_id=parent_task_id,
                        run_id=run_id,
                        task_type="pdf_ocr_page",
                        unit_key=f"page:{page.page_number}",
                        payload=ocr_payload,
                        priority=400,
                        now=now,
                        source_digest=source_digest,
                        task_version=parser_version,
                    )
                con.execute(
                    """
                    INSERT INTO pdf_page_identities(
                        file_id, source_digest, parser_version, page_number,
                        page_identity, width_points, height_points,
                        classification, native_task_id, ocr_task_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_id,
                        source_digest,
                        parser_version,
                        page.page_number,
                        page.page_identity,
                        page.width_points,
                        page.height_points,
                        "ocr_candidate" if page.requires_ocr else "native",
                        native,
                        ocr_task_id,
                        now,
                    ),
                )
            self._insert_child(
                con,
                file_id=file_id,
                parent_task_id=parent_task_id,
                run_id=run_id,
                task_type="document_merge",
                unit_key="merge",
                payload={"page_count": len(pages)},
                priority=10,
                now=now,
                source_digest=source_digest,
                task_version=parser_version,
                status="blocked",
            )
        return parent_task_id

    def claim_page_tasks(
        self,
        worker_id: str,
        *,
        limit: int,
        lease_seconds: int,
    ) -> list[PdfPageTaskClaim]:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires = (
            now_dt + timedelta(seconds=max(1, int(lease_seconds)))
        ).isoformat()
        with self.database.connect() as con:
            rows = con.execute(
                """
                WITH candidates AS (
                    SELECT pt.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY pt.parent_task_id
                               ORDER BY pt.priority DESC, pt.id
                           ) AS source_rank
                    FROM parse_tasks pt
                    WHERE pt.task_type IN ('pdf_native_page', 'pdf_ocr_page')
                      AND (
                          pt.status = 'queued'
                          OR (
                              pt.status = 'running'
                              AND pt.lease_expires_at IS NOT NULL
                              AND pt.lease_expires_at <= ?
                          )
                      )
                )
                SELECT * FROM candidates
                ORDER BY source_rank, parent_task_id, priority DESC, id
                LIMIT ?
                """,
                (now, max(1, int(limit))),
            ).fetchall()
            claims: list[PdfPageTaskClaim] = []
            for row in rows:
                updated = con.execute(
                    """
                    UPDATE parse_tasks
                    SET status = 'running', lease_owner = ?,
                        lease_expires_at = ?, started_at = COALESCE(started_at, ?)
                    WHERE id = ? AND (
                        status = 'queued'
                        OR (
                            status = 'running'
                            AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                        )
                    )
                    """,
                    (worker_id, expires, now, int(row["id"]), now),
                )
                if updated.rowcount != 1:
                    continue
                payload = json.loads(str(row["payload_json"] or "{}"))
                claims.append(
                    PdfPageTaskClaim(
                        task_id=int(row["id"]),
                        parent_task_id=int(row["parent_task_id"]),
                        file_id=int(row["file_id"]),
                        task_type=str(row["task_type"]),
                        page_number=int(payload.get("page_number") or 0),
                        source_digest=str(row["source_digest"] or ""),
                        task_version=str(row["task_version"] or ""),
                        lease_owner=worker_id,
                        lease_expires_at=expires,
                        payload=payload,
                    )
                )
        return claims

    def scheduled_page_tasks(
        self,
        document_task_id: int,
    ) -> list[PdfPageTaskClaim]:
        with self.database.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM parse_tasks
                WHERE parent_task_id = ?
                  AND task_type IN ('pdf_native_page', 'pdf_ocr_page')
                  AND status IN ('queued', 'running')
                ORDER BY
                    CAST(substr(unit_key, 6) AS INTEGER),
                    CASE task_type
                        WHEN 'pdf_native_page' THEN 0 ELSE 1
                    END,
                    id
                """,
                (int(document_task_id),),
            ).fetchall()
        claims: list[PdfPageTaskClaim] = []
        for row in rows:
            payload = json.loads(str(row["payload_json"] or "{}"))
            claims.append(
                PdfPageTaskClaim(
                    task_id=int(row["id"]),
                    parent_task_id=int(row["parent_task_id"]),
                    file_id=int(row["file_id"]),
                    task_type=str(row["task_type"]),
                    page_number=int(payload.get("page_number") or 0),
                    source_digest=str(row["source_digest"] or ""),
                    task_version=str(row["task_version"] or ""),
                    lease_owner=str(row["lease_owner"] or ""),
                    lease_expires_at=str(row["lease_expires_at"] or ""),
                    payload=payload,
                )
            )
        return claims

    def confirmed_page_results(
        self,
        document_task_id: int,
    ) -> list[tuple[int, str, Path, str]]:
        with self.database.connect() as con:
            rows = con.execute(
                """
                SELECT task_type, unit_key, spool_path, result_digest
                FROM parse_tasks
                WHERE parent_task_id = ?
                  AND task_type IN ('pdf_native_page', 'pdf_ocr_page')
                  AND status = 'complete' AND confirmed_at IS NOT NULL
                ORDER BY
                    CAST(substr(unit_key, 6) AS INTEGER),
                    CASE task_type
                        WHEN 'pdf_native_page' THEN 0 ELSE 1
                    END,
                    id
                """,
                (int(document_task_id),),
            ).fetchall()
        return [
            (
                int(str(row["unit_key"]).partition(":")[2]),
                str(row["task_type"]),
                Path(str(row["spool_path"])),
                str(row["result_digest"] or ""),
            )
            for row in rows
        ]

    def confirm_page_task(
        self,
        task_id: int,
        *,
        result_spool_path: Path,
        result_digest: str,
    ) -> None:
        self.confirm_page_tasks(
            [(task_id, result_spool_path, result_digest)]
        )

    def confirm_page_tasks(
        self,
        confirmations: list[tuple[int, Path, str]],
    ) -> None:
        """Confirm one worker batch with a single durable transaction."""

        if not confirmations:
            return
        now = utc_now()
        with self.database.connect() as con:
            for task_id, result_spool_path, result_digest in confirmations:
                updated = con.execute(
                    """
                    UPDATE parse_tasks
                    SET status = 'complete', confirmed_at = ?, finished_at = ?,
                        spool_path = ?, result_digest = ?, lease_owner = NULL,
                        lease_expires_at = NULL, progress_completed = 1,
                        progress_total = 1, checkpoint_version = 1
                    WHERE id = ? AND task_type IN (
                        'pdf_native_page', 'pdf_ocr_page'
                    ) AND status IN ('running', 'spooled')
                    """,
                    (
                        now,
                        now,
                        str(result_spool_path),
                        result_digest,
                        int(task_id),
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(
                        f"PDF page task cannot be confirmed: {task_id}"
                    )
                con.execute(
                    """
                    UPDATE parse_task_attempts
                    SET status = 'complete', finished_at = ?,
                        error_code = NULL, error_message = NULL
                    WHERE id = (
                        SELECT id FROM parse_task_attempts
                        WHERE task_id = ?
                        ORDER BY attempt_no DESC
                        LIMIT 1
                    ) AND status IN ('running', 'spooled')
                    """,
                    (now, int(task_id)),
                )

    def fail_page_task(self, task_id: int, error_code: str, message: str) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'failed', finished_at = ?, error_code = ?,
                    error_message = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ?
                """,
                (utc_now(), error_code, message, int(task_id)),
            )

    def fail_document(
        self,
        document_task_id: int,
        error_code: str,
        message: str,
    ) -> None:
        now = utc_now()
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'failed', finished_at = ?, error_code = ?,
                    error_message = ?, lease_owner = NULL,
                    lease_expires_at = NULL
                WHERE id = ? OR (
                    parent_task_id = ? AND task_type = 'document_merge'
                )
                """,
                (
                    now,
                    error_code,
                    message,
                    int(document_task_id),
                    int(document_task_id),
                ),
            )

    def requeue_page_task(self, task_id: int) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'queued', queued_at = ?, started_at = NULL,
                    finished_at = NULL, confirmed_at = NULL,
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_code = NULL, error_message = NULL
                WHERE id = ?
                """,
                (utc_now(), int(task_id)),
            )

    def expire_all_leases_for_validation(self) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE parse_tasks
                SET lease_expires_at = '1970-01-01T00:00:00+00:00'
                WHERE task_type IN ('pdf_native_page', 'pdf_ocr_page')
                  AND status = 'running'
                """
            )

    def recover_document(
        self,
        document_task_id: int,
        *,
        run_id: str,
    ) -> dict[str, int]:
        with self.database.connect() as con:
            return self._recover_document_in_connection(
                con,
                document_task_id=int(document_task_id),
                run_id=str(run_id),
                now=utc_now(),
            )

    @staticmethod
    def _recover_document_in_connection(
        con: object,
        *,
        document_task_id: int,
        run_id: str,
        now: str,
    ) -> dict[str, int]:
        """Recover a graph after the owning application process disappeared."""

        requeued_leases = 0
        requeued_corrupt = 0
        requeued_failed = 0
        page_rows = con.execute(
            """
            SELECT id, run_id, status, spool_path, result_digest
            FROM parse_tasks
            WHERE parent_task_id = ?
              AND task_type IN ('pdf_native_page', 'pdf_ocr_page')
            ORDER BY id
            """,
            (int(document_task_id),),
        ).fetchall()
        for row in page_rows:
            task_id = int(row["id"])
            status = str(row["status"])
            if status == "complete":
                spool_text = str(row["spool_path"] or "")
                expected_digest = str(row["result_digest"] or "")
                spool_path = Path(spool_text) if spool_text else None
                valid = bool(
                    spool_path is not None
                    and spool_path.is_file()
                    and expected_digest
                    and _sha256_file(spool_path) == expected_digest
                )
                if valid:
                    continue
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET run_id = ?, status = 'queued', queued_at = ?,
                        started_at = NULL, finished_at = NULL,
                        confirmed_at = NULL, lease_owner = NULL,
                        lease_expires_at = NULL, spool_path = NULL,
                        result_digest = NULL, error_code = NULL,
                        error_message = NULL
                    WHERE id = ?
                    """,
                    (run_id, now, task_id),
                )
                requeued_corrupt += 1
                continue
            if status == "running" and str(row["run_id"] or "") != run_id:
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET run_id = ?, status = 'queued', queued_at = ?,
                        started_at = NULL, finished_at = NULL,
                        lease_owner = NULL, lease_expires_at = NULL,
                        error_code = NULL, error_message = NULL
                    WHERE id = ?
                    """,
                    (run_id, now, task_id),
                )
                requeued_leases += 1
            elif status in {"queued", "failed", "failed_retryable", "paused"}:
                con.execute(
                    """
                    UPDATE parse_tasks
                    SET run_id = ?, status = 'queued', queued_at = ?,
                        started_at = NULL, finished_at = NULL,
                        lease_owner = NULL, lease_expires_at = NULL,
                        error_code = NULL, error_message = NULL
                    WHERE id = ?
                    """,
                    (run_id, now, task_id),
                )
                if status in {"failed", "failed_retryable"}:
                    requeued_failed += 1
        con.execute(
            """
            UPDATE parse_tasks
            SET run_id = ?,
                status = CASE WHEN status = 'complete' THEN status ELSE 'running' END,
                error_code = NULL, error_message = NULL
            WHERE id = ?
            """,
            (run_id, int(document_task_id)),
        )
        con.execute(
            """
            UPDATE parse_tasks
            SET run_id = ?,
                status = CASE WHEN status = 'complete' THEN status ELSE 'blocked' END,
                error_code = NULL, error_message = NULL
            WHERE parent_task_id = ? AND task_type = 'document_merge'
            """,
            (run_id, int(document_task_id)),
        )
        return {
            "requeued_leases": requeued_leases,
            "requeued_corrupt": requeued_corrupt,
            "requeued_failed": requeued_failed,
        }

    def ready_document_task_ids(self) -> list[int]:
        with self.database.connect() as con:
            rows = con.execute(
                """
                SELECT parent.id
                FROM parse_tasks parent
                JOIN parse_tasks child ON child.parent_task_id = parent.id
                WHERE parent.task_type = 'pdf_document'
                  AND parent.status = 'running'
                  AND child.task_type IN ('pdf_native_page', 'pdf_ocr_page')
                GROUP BY parent.id
                HAVING COUNT(*) > 0
                   AND SUM(
                       CASE WHEN child.status = 'complete'
                                  AND child.confirmed_at IS NOT NULL
                            THEN 1 ELSE 0 END
                   ) = COUNT(*)
                   AND SUM(CASE WHEN child.status = 'failed' THEN 1 ELSE 0 END) = 0
                ORDER BY parent.id
                """
            ).fetchall()
        return [int(row["id"]) for row in rows]

    def merge_task_id(self, document_task_id: int) -> int | None:
        with self.database.connect() as con:
            row = con.execute(
                """
                SELECT id FROM parse_tasks
                WHERE parent_task_id = ? AND task_type = 'document_merge'
                ORDER BY id DESC LIMIT 1
                """,
                (int(document_task_id),),
            ).fetchone()
        return int(row["id"]) if row is not None else None

    def merge_readiness(self, document_task_id: int) -> PdfMergeReadiness:
        placeholders = ",".join("?" for _ in self.PAGE_TASK_TYPES)
        with self.database.connect() as con:
            row = con.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'complete'
                                     AND confirmed_at IS NOT NULL
                                THEN 1 ELSE 0 END) AS confirmed,
                       SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed
                FROM parse_tasks
                WHERE parent_task_id = ?
                  AND task_type IN ({placeholders})
                """,
                (int(document_task_id), *self.PAGE_TASK_TYPES),
            ).fetchone()
        total = int(row["total"] or 0)
        confirmed = int(row["confirmed"] or 0)
        failed = int(row["failed"] or 0)
        pending = max(0, total - confirmed - failed)
        return PdfMergeReadiness(
            ready=total > 0 and confirmed == total and failed == 0,
            total_pages=total,
            confirmed_pages=confirmed,
            failed_pages=failed,
            pending_pages=pending,
        )

    def confirm_merge(self, document_task_id: int, result_digest: str) -> bool:
        readiness = self.merge_readiness(document_task_id)
        if not readiness.ready:
            return False
        now = utc_now()
        with self.database.connect() as con:
            merge = con.execute(
                """
                UPDATE parse_tasks
                SET status = 'complete', confirmed_at = ?, finished_at = ?,
                    result_digest = ?, progress_completed = ?,
                    progress_total = ?, checkpoint_version = ?
                WHERE parent_task_id = ? AND task_type = 'document_merge'
                  AND status != 'complete'
                """,
                (
                    now,
                    now,
                    result_digest,
                    readiness.confirmed_pages,
                    readiness.total_pages,
                    readiness.confirmed_pages,
                    int(document_task_id),
                ),
            )
            if merge.rowcount != 1:
                return False
            con.execute(
                """
                UPDATE parse_tasks
                SET status = 'complete', confirmed_at = ?, finished_at = ?,
                    result_digest = ?
                WHERE id = ? AND status != 'complete'
                """,
                (now, now, result_digest, int(document_task_id)),
            )
        return True

    @staticmethod
    def _insert_child(
        con: object,
        *,
        file_id: int,
        parent_task_id: int,
        run_id: str,
        task_type: str,
        unit_key: str,
        payload: dict[str, object],
        priority: int,
        now: str,
        source_digest: str,
        task_version: str,
        status: str = "queued",
    ) -> int:
        cursor = con.execute(
            """
            INSERT INTO parse_tasks(
                file_id, parent_task_id, run_id, task_type, unit_key,
                payload_json, status, priority, created_at, queued_at,
                source_digest, task_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_id,
                parent_task_id,
                run_id,
                task_type,
                unit_key,
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                status,
                priority,
                now,
                now,
                source_digest,
                task_version,
            ),
        )
        return int(cursor.lastrowid)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
