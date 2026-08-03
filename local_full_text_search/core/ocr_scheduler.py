from __future__ import annotations

import hashlib
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_full_text_search.core.database import DatabaseManager, utc_now


@dataclass(frozen=True, slots=True)
class OcrRequest:
    request_id: str
    source_id: str
    source_kind: str
    pixel_cost: int
    payload: dict[str, object]
    priority: int = 100
    submitted_at: float = field(default_factory=time.monotonic)


@dataclass(frozen=True, slots=True)
class DurableOcrRequestClaim:
    request_id: int
    file_id: int
    parent_task_id: int
    source_kind: str
    source_unit: str
    image_spool_path: Path
    content_sha256: str
    width: int
    height: int
    config_fingerprint: str
    priority: int
    pixel_cost: int
    checkpoint_cursor: str
    lease_owner: str
    lease_expires_at: str


class OcrRequestRepository:
    """Durable OCR parent queue with source-fair leased claims."""

    def __init__(self, database: DatabaseManager) -> None:
        self.database = database

    def enqueue(
        self,
        *,
        file_id: int,
        parent_task_id: int,
        source_kind: str,
        source_unit: str,
        image_spool_path: Path,
        content_sha256: str,
        width: int,
        height: int,
        config_fingerprint: str,
        priority: int,
        pixel_cost: int,
        checkpoint_cursor: str = "",
    ) -> int:
        now = utc_now()
        with self.database.connect() as con:
            con.execute(
                """
                INSERT OR IGNORE INTO ocr_requests(
                    file_id, parent_task_id, source_kind, source_unit,
                    image_spool_path, content_sha256, width, height,
                    config_fingerprint, priority, pixel_cost, status,
                    checkpoint_cursor, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    int(file_id),
                    int(parent_task_id),
                    str(source_kind),
                    str(source_unit),
                    str(image_spool_path),
                    str(content_sha256),
                    max(0, int(width)),
                    max(0, int(height)),
                    str(config_fingerprint),
                    int(priority),
                    max(1, int(pixel_cost)),
                    str(checkpoint_cursor),
                    now,
                ),
            )
            row = con.execute(
                """
                SELECT id FROM ocr_requests
                WHERE parent_task_id = ? AND source_kind = ?
                  AND source_unit = ? AND config_fingerprint = ?
                """,
                (
                    int(parent_task_id),
                    str(source_kind),
                    str(source_unit),
                    str(config_fingerprint),
                ),
            ).fetchone()
        if row is None:
            raise RuntimeError("Unable to persist OCR request")
        return int(row["id"])

    def claim(
        self,
        worker_id: str,
        *,
        limit: int,
        max_pixels: int,
        lease_seconds: int,
    ) -> list[DurableOcrRequestClaim]:
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires = (
            now_dt + timedelta(seconds=max(1, int(lease_seconds)))
        ).isoformat()
        request_limit = max(1, int(limit))
        pixel_limit = max(1, int(max_pixels))
        with self.database.connect() as con:
            rows = con.execute(
                """
                WITH candidates AS (
                    SELECT request.*,
                           ROW_NUMBER() OVER (
                               PARTITION BY request.file_id
                               ORDER BY request.priority DESC,
                                        request.created_at,
                                        request.id
                           ) AS source_rank
                    FROM ocr_requests request
                    WHERE request.status = 'queued'
                       OR (
                           request.status = 'running'
                           AND request.lease_expires_at IS NOT NULL
                           AND request.lease_expires_at <= ?
                       )
                )
                SELECT * FROM candidates
                ORDER BY source_rank, priority DESC, created_at, id
                """,
                (now,),
            ).fetchall()
            claims: list[DurableOcrRequestClaim] = []
            used_pixels = 0
            for row in rows:
                if len(claims) >= request_limit:
                    break
                pixels = max(1, int(row["pixel_cost"] or 0))
                if claims and used_pixels + pixels > pixel_limit:
                    continue
                updated = con.execute(
                    """
                    UPDATE ocr_requests
                    SET status = 'running', lease_owner = ?,
                        lease_expires_at = ?
                    WHERE id = ? AND (
                        status = 'queued'
                        OR (
                            status = 'running'
                            AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                        )
                    )
                    """,
                    (
                        str(worker_id),
                        expires,
                        int(row["id"]),
                        now,
                    ),
                )
                if updated.rowcount != 1:
                    continue
                used_pixels += pixels
                claims.append(
                    DurableOcrRequestClaim(
                        request_id=int(row["id"]),
                        file_id=int(row["file_id"]),
                        parent_task_id=int(row["parent_task_id"] or 0),
                        source_kind=str(row["source_kind"]),
                        source_unit=str(row["source_unit"]),
                        image_spool_path=Path(
                            str(row["image_spool_path"])
                        ),
                        content_sha256=str(row["content_sha256"]),
                        width=int(row["width"]),
                        height=int(row["height"]),
                        config_fingerprint=str(
                            row["config_fingerprint"]
                        ),
                        priority=int(row["priority"]),
                        pixel_cost=pixels,
                        checkpoint_cursor=str(
                            row["checkpoint_cursor"] or ""
                        ),
                        lease_owner=str(worker_id),
                        lease_expires_at=expires,
                    )
                )
        return claims

    def confirm(
        self,
        request_id: int,
        *,
        worker_id: str,
        result_spool_path: Path,
        result_digest: str,
    ) -> None:
        if not result_spool_path.is_file():
            raise ValueError("OCR result spool is missing")
        actual_digest = hashlib.sha256(
            result_spool_path.read_bytes()
        ).hexdigest()
        if actual_digest != str(result_digest):
            raise ValueError("OCR result spool digest mismatch")
        with self.database.connect() as con:
            updated = con.execute(
                """
                UPDATE ocr_requests
                SET status = 'confirmed', confirmed_at = ?,
                    result_spool_path = ?, result_digest = ?,
                    lease_owner = NULL, lease_expires_at = NULL,
                    error_code = NULL, error_message = NULL
                WHERE id = ? AND status = 'running'
                  AND lease_owner = ?
                """,
                (
                    utc_now(),
                    str(result_spool_path),
                    actual_digest,
                    int(request_id),
                    str(worker_id),
                ),
            )
            if updated.rowcount != 1:
                raise ValueError(
                    "OCR request is not owned by this worker"
                )

    def claim_specific(
        self,
        request_ids: list[int],
        *,
        worker_id: str,
        lease_seconds: int,
    ) -> list[int]:
        if not request_ids:
            return []
        now_dt = datetime.now(timezone.utc)
        now = now_dt.isoformat()
        expires = (
            now_dt + timedelta(seconds=max(1, int(lease_seconds)))
        ).isoformat()
        claimed: list[int] = []
        with self.database.connect() as con:
            for request_id in request_ids:
                updated = con.execute(
                    """
                    UPDATE ocr_requests
                    SET status = 'running', lease_owner = ?,
                        lease_expires_at = ?
                    WHERE id = ? AND (
                        status = 'queued'
                        OR (
                            status = 'running'
                            AND lease_expires_at IS NOT NULL
                            AND lease_expires_at <= ?
                        )
                    )
                    """,
                    (
                        str(worker_id),
                        expires,
                        int(request_id),
                        now,
                    ),
                )
                if updated.rowcount == 1:
                    claimed.append(int(request_id))
        return claimed

    def requeue(
        self,
        request_id: int,
        *,
        checkpoint_cursor: str = "",
    ) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE ocr_requests
                SET status = 'queued', lease_owner = NULL,
                    lease_expires_at = NULL, checkpoint_cursor = ?,
                    error_code = NULL, error_message = NULL
                WHERE id = ?
                """,
                (str(checkpoint_cursor), int(request_id)),
            )

    def fail(
        self,
        request_id: int,
        error_code: str,
        message: str,
    ) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE ocr_requests
                SET status = 'failed', lease_owner = NULL,
                    lease_expires_at = NULL, error_code = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (
                    str(error_code),
                    str(message),
                    int(request_id),
                ),
            )

    def expire_all_leases_for_validation(self) -> None:
        with self.database.connect() as con:
            con.execute(
                """
                UPDATE ocr_requests
                SET lease_expires_at = '1970-01-01T00:00:00+00:00'
                WHERE status = 'running'
                """
            )

    def get(self, request_id: int) -> dict[str, object]:
        with self.database.connect() as con:
            row = con.execute(
                "SELECT * FROM ocr_requests WHERE id = ?",
                (int(request_id),),
            ).fetchone()
        if row is None:
            raise KeyError(request_id)
        payload = dict(row)
        payload["lease_owner"] = str(payload.get("lease_owner") or "")
        payload["lease_expires_at"] = str(
            payload.get("lease_expires_at") or ""
        )
        return payload


class FairOcrScheduler:
    """Source-fair bounded OCR request scheduler."""

    def __init__(self, *, max_inflight_per_source: int = 0) -> None:
        self.max_inflight_per_source = max(0, int(max_inflight_per_source))
        self._queues: dict[str, deque[OcrRequest]] = {}
        self._source_order: deque[str] = deque()
        self._inflight: dict[str, OcrRequest] = {}
        self._inflight_by_source: dict[str, int] = {}

    @property
    def pending_count(self) -> int:
        return sum(len(queue) for queue in self._queues.values())

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def submit(self, request: OcrRequest) -> None:
        if request.request_id in self._inflight or any(
            queued.request_id == request.request_id
            for queue in self._queues.values()
            for queued in queue
        ):
            raise ValueError(f"Duplicate OCR request: {request.request_id}")
        queue = self._queues.get(request.source_id)
        if queue is None:
            queue = deque()
            self._queues[request.source_id] = queue
            self._source_order.append(request.source_id)
        queue.append(request)

    def claim_batch(
        self,
        *,
        max_requests: int,
        max_pixels: int,
    ) -> list[OcrRequest]:
        request_limit = max(1, int(max_requests))
        pixel_limit = max(1, int(max_pixels))
        claimed: list[OcrRequest] = []
        used_pixels = 0
        stalled_sources = 0
        while self._source_order and len(claimed) < request_limit:
            source_id = self._source_order.popleft()
            queue = self._queues[source_id]
            inflight = self._inflight_by_source.get(source_id, 0)
            at_inflight_limit = (
                self.max_inflight_per_source > 0
                and inflight >= self.max_inflight_per_source
            )
            request = queue[0] if queue else None
            exceeds_pixels = bool(
                request is not None
                and claimed
                and used_pixels + max(1, request.pixel_cost) > pixel_limit
            )
            if request is None:
                self._queues.pop(source_id, None)
                stalled_sources = 0
                continue
            if at_inflight_limit or exceeds_pixels:
                self._source_order.append(source_id)
                stalled_sources += 1
                if stalled_sources >= len(self._source_order):
                    break
                continue
            stalled_sources = 0
            queue.popleft()
            claimed.append(request)
            used_pixels += max(1, int(request.pixel_cost))
            self._inflight[request.request_id] = request
            self._inflight_by_source[source_id] = inflight + 1
            if queue:
                self._source_order.append(source_id)
            else:
                self._queues.pop(source_id, None)
        return claimed

    def confirm(self, request_id: str) -> OcrRequest:
        request = self._inflight.pop(request_id)
        remaining = max(
            0,
            self._inflight_by_source.get(request.source_id, 0) - 1,
        )
        if remaining:
            self._inflight_by_source[request.source_id] = remaining
        else:
            self._inflight_by_source.pop(request.source_id, None)
        return request

    def requeue(self, request_id: str) -> None:
        request = self.confirm(request_id)
        queue = self._queues.get(request.source_id)
        if queue is None:
            queue = deque()
            self._queues[request.source_id] = queue
            self._source_order.appendleft(request.source_id)
        queue.appendleft(request)
