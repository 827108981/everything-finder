from __future__ import annotations

import queue
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_full_text_search.core.database import (
    DatabaseManager,
    SQLITE_WRITER_BUSY_CIRCUIT_MAX_ATTEMPTS,
    SQLITE_WRITER_BUSY_CIRCUIT_MAX_SECONDS,
    SQLiteBusyCircuit,
    SQLiteBusyCircuitOpen,
)
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.foreground_activity import (
    begin_index_write,
    end_index_write,
)
from local_full_text_search.core.task_manager import CancelToken

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WriterSummary:
    artifacts: int = 0
    blocks: int = 0
    text_bytes: int = 0
    batches: int = 0
    write_ms: int = 0


class WriterFinalizationTimeout(RuntimeError):
    """Raised when the writer cannot reach a bounded shutdown point."""


class IndexWriter:
    """One bounded writer that overlaps SQLite batches with parsing."""

    _STOP = object()

    def __init__(
        self,
        db: DatabaseManager,
        *,
        update_fts: bool,
        batch_blocks: int,
        batch_bytes: int,
        max_delay_ms: int,
        queue_size: int = 64,
        large_artifact_fts_threshold_blocks: int = 128,
        large_artifact_fts_threshold_bytes: int = 512 * 1024,
        artifact_write_chunk_blocks: int = 256,
        artifact_write_chunk_bytes: int = 512 * 1024,
        defer_fts: bool = False,
        on_written: Callable[[list[Any], int], None] | None = None,
        write_dispatcher: Any | None = None,
    ) -> None:
        self.db = db
        self.update_fts = bool(update_fts)
        self.defer_fts = bool(defer_fts)
        # Keep the writer's transaction bounded even when old settings carry
        # multi-thousand-block batches. Large FTS transactions were the main
        # source of search/index contention; the parser queue still provides
        # throughput between these small commits.
        self.batch_blocks = max(1, min(500, int(batch_blocks)))
        self.batch_bytes = max(1024, min(1024 * 1024, int(batch_bytes)))
        self.max_delay = max(0.01, min(0.5, int(max_delay_ms) / 1000))
        self.large_artifact_fts_threshold_blocks = max(
            1, int(large_artifact_fts_threshold_blocks)
        )
        self.large_artifact_fts_threshold_bytes = max(
            1024, int(large_artifact_fts_threshold_bytes)
        )
        self.artifact_write_chunk_blocks = max(
            1, min(512, int(artifact_write_chunk_blocks))
        )
        self.artifact_write_chunk_bytes = max(
            64 * 1024, int(artifact_write_chunk_bytes)
        )
        self.on_written = on_written
        self.write_dispatcher = write_dispatcher
        self.summary = WriterSummary()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(2, int(queue_size)))
        self._thread = threading.Thread(target=self._run, name="lfts-index-writer", daemon=True)
        self._error: BaseException | None = None
        self._started = False
        self._aborted = threading.Event()
        self._flushing = threading.Event()
        self._deferred_fts_file_ids: set[int] = set()
        self._busy_circuit = SQLiteBusyCircuit(
            "index_writer_batch",
            max_attempts=SQLITE_WRITER_BUSY_CIRCUIT_MAX_ATTEMPTS,
            max_seconds=SQLITE_WRITER_BUSY_CIRCUIT_MAX_SECONDS,
            database_path=getattr(db, "db_path", None),
        )
        # Short-token generation is published once after all parser results
        # are committed. Rebuilding it for every batch defeats the bounded
        # writer design.
        try:
            self.db._short_tokens_deferred = True
        except Exception:
            pass

    @property
    def deferred_fts_file_ids(self) -> frozenset[int]:
        """Files whose FTS update was moved out of the content write transaction."""

        return frozenset(self._deferred_fts_file_ids)

    def start(self) -> None:
        if not self._started:
            self._started = True
            self._thread.start()

    def submit(
        self,
        artifact: Any,
        *,
        cancel_token: CancelToken | None = None,
    ) -> None:
        self.start()
        while True:
            self._raise_if_failed()
            if self._aborted.is_set() or (cancel_token is not None and cancel_token.cancelled):
                raise CancelledError("索引写入已取消")
            try:
                self._queue.put(artifact, timeout=0.1)
                return
            except queue.Full:
                continue

    def abort(self, timeout: float = 0.5) -> None:
        """Discard queued writes and return promptly during cancellation."""

        self._aborted.set()
        self.db.interrupt_active_connections()
        if not self._started:
            return
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not self._STOP:
                # Unwritten tasks remain non-complete and are retried on the next run.
                continue
        try:
            self._queue.put_nowait(self._STOP)
        except queue.Full:
            pass
        self._thread.join(timeout=max(0.0, float(timeout)))
        try:
            self.db._short_tokens_deferred = False
        except Exception:
            pass

    def finish(
        self,
        *,
        sync_deferred_fts: bool = True,
        max_wait_seconds: float = 30.0,
        progress_check: Callable[[], None] | None = None,
    ) -> WriterSummary:
        if not self._started:
            try:
                self.db._short_tokens_deferred = False
            except Exception:
                pass
            return self.summary
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(self._STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        deadline = time.monotonic() + max(0.1, float(max_wait_seconds))
        while self._thread.is_alive():
            self._thread.join(timeout=0.25)
            if progress_check is not None:
                try:
                    progress_check()
                except BaseException:
                    # A watchdog timeout must switch to the bounded abort
                    # path. Do not leave the worker alive and let an outer
                    # exception handler call finish() a second time.
                    self.abort(timeout=0.5)
                    raise
            if time.monotonic() >= deadline and self._thread.is_alive():
                self.abort(timeout=0.5)
                raise WriterFinalizationTimeout(
                    "索引写入器在有界收尾时间内未退出"
                )
        self._raise_if_failed()
        if sync_deferred_fts and self._deferred_fts_file_ids:
            sync_started = time.perf_counter()
            deferred_ids = tuple(self._deferred_fts_file_ids)
            try:
                synced_blocks = self._run_database_write(
                    lambda: self.db.sync_content_fts_for_files(
                        deferred_ids,
                        batch_blocks=min(
                            64,
                            self.large_artifact_fts_threshold_blocks,
                        ),
                        batch_bytes=self.large_artifact_fts_threshold_bytes,
                    ),
                    label="writer_deferred_fts",
                )
            except sqlite3.OperationalError as exc:
                if not _is_busy_error(exc):
                    raise
                # FTS is a derived projection.  Keep the durable pending
                # markers and let the next idle/index pass replay them; a
                # busy search must not turn a successful content write into a
                # failed root.
                logger.warning(
                    "Deferred FTS synchronization postponed because SQLite is busy: files=%s",
                    len(deferred_ids),
                )
                return self.summary
            sync_ms = int((time.perf_counter() - sync_started) * 1000)
            self.summary.write_ms += sync_ms
            logger.info(
                "Deferred large-artifact FTS synchronized: files=%s blocks=%s write_ms=%s",
                len(deferred_ids),
                synced_blocks,
                sync_ms,
            )
            self._deferred_fts_file_ids.clear()
        try:
            self.db._short_tokens_deferred = False
        except Exception:
            pass
        return self.summary

    def is_idle(self) -> bool:
        return self._queue.empty() and not self._flushing.is_set()

    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _run(self) -> None:
        pending: list[Any] = []
        blocks = 0
        text_bytes = 0
        deadline = time.monotonic() + self.max_delay
        try:
            while True:
                timeout = max(0.0, deadline - time.monotonic()) if pending else None
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    if self._flush(pending):
                        pending = []
                        blocks = 0
                        text_bytes = 0
                    else:
                        # A foreground search temporarily owns the read lease.
                        # Keep the same artifacts for retry instead of dropping
                        # them or waiting forever in one SQLite call.
                        time.sleep(0.05)
                    deadline = time.monotonic() + self.max_delay
                    continue
                if item is self._STOP:
                    if not self._aborted.is_set():
                        while pending and not self._aborted.is_set():
                            if self._flush(pending):
                                pending = []
                                break
                            time.sleep(0.05)
                    return
                if self._aborted.is_set():
                    pending = []
                    continue
                pending.append(item)
                item_blocks = list(getattr(item, "blocks", []) or [])
                blocks += len(item_blocks)
                text_bytes += sum(len(block.raw_text.encode("utf-8")) for block in item_blocks)
                if len(pending) == 1:
                    deadline = time.monotonic() + self.max_delay
                if blocks >= self.batch_blocks or text_bytes >= self.batch_bytes:
                    if self._flush(pending):
                        pending = []
                        blocks = 0
                        text_bytes = 0
                    else:
                        time.sleep(0.05)
                    deadline = time.monotonic() + self.max_delay
        except BaseException as exc:
            self._error = exc

    def _flush(self, artifacts: list[Any]) -> bool:
        if not artifacts:
            return True
        started = time.perf_counter()
        items = [self._to_item(artifact) for artifact in artifacts]
        artifact_sizes = [
            (
                len(getattr(artifact, "blocks", []) or []),
                sum(
                    len(str(block.raw_text or "").encode("utf-8"))
                    for block in (getattr(artifact, "blocks", []) or [])
                ),
            )
            for artifact in artifacts
        ]
        chunked_indexes = {
            index
            for index, (block_count, text_bytes) in enumerate(artifact_sizes)
            if block_count > self.artifact_write_chunk_blocks
            or text_bytes > self.artifact_write_chunk_bytes
        }
        reserved = False
        if not chunked_indexes:
            # Reserve the writer side atomically with the search lease. This
            # closes the race where a lease is granted between the idle check
            # and the SQLite transaction start.  Chunked documents deliberately
            # skip this outer reservation: each short DB transaction acquires
            # and releases the gate independently so telemetry can progress.
            if not begin_index_write(
                cancel_event=self._aborted,
                max_wait_seconds=0.5,
            ):
                return False
            reserved = True
        self._flushing.set()
        try:
            if chunked_indexes:
                chunk_writer = getattr(
                    self.db,
                    "replace_document_blocks_chunked",
                    None,
                )
                for index, item in enumerate(items):
                    should_defer = (
                        self.defer_fts
                        or self.update_fts
                        and index in chunked_indexes
                    )
                    if index in chunked_indexes and callable(chunk_writer):
                        self._run_database_write(
                            lambda item=item: chunk_writer(
                                item,
                                chunk_blocks=self.artifact_write_chunk_blocks,
                                chunk_bytes=self.artifact_write_chunk_bytes,
                                update_fts=False,
                                remove_old_fts=True,
                                cancel_event=self._aborted,
                            ),
                            label="writer_chunked_document",
                        )
                    elif should_defer:
                        # Keep compatibility with small test doubles and
                        # older database adapters; production uses the
                        # chunked method above for large artifacts.
                        self._run_database_write(
                            lambda item=item: self.db.replace_document_blocks_many(
                                [item],
                                update_fts=False,
                                remove_old_fts=True,
                            ),
                            label="writer_document_deferred",
                        )
                    else:
                        self._run_database_write(
                            lambda item=item: self.db.replace_document_blocks_many(
                                [item],
                                update_fts=True,
                            ),
                            label="writer_document",
                        )
                    if should_defer and item.get("status") in {
                        "success",
                        "metadata_only",
                    }:
                        self._deferred_fts_file_ids.update(
                            int(file_id)
                            for file_id in item.get("file_ids") or [item["file_id"]]
                        )
            elif self.defer_fts:
                # Commit parsed content and task state without touching the
                # FTS5 table.  FTS is published later in short transactions,
                # after page/task confirmations have had a chance to commit.
                self._run_database_write(
                    lambda: self.db.replace_document_blocks_many(
                        items,
                        update_fts=False,
                        remove_old_fts=True,
                    ),
                    label="writer_batch_deferred",
                )
                for item in items:
                    if item.get("status") not in {"success", "metadata_only"}:
                        continue
                    self._deferred_fts_file_ids.update(
                        int(file_id)
                        for file_id in item.get("file_ids") or [item["file_id"]]
                    )
            elif self.update_fts:
                large_items: list[dict[str, Any]] = []
                regular_items: list[dict[str, Any]] = []
                defer_entire_batch = (
                    sum(size[0] for size in artifact_sizes)
                    >= self.large_artifact_fts_threshold_blocks
                    or sum(size[1] for size in artifact_sizes)
                    >= self.large_artifact_fts_threshold_bytes
                )
                for item, (artifact_blocks, artifact_bytes) in zip(
                    items,
                    artifact_sizes,
                ):
                    target = (
                        large_items
                        if defer_entire_batch
                        or artifact_blocks
                        >= self.large_artifact_fts_threshold_blocks
                        or artifact_bytes
                        >= self.large_artifact_fts_threshold_bytes
                        else regular_items
                    )
                    target.append(item)
                if regular_items:
                    self._run_database_write(
                        lambda: self.db.replace_document_blocks_many(
                            regular_items,
                            update_fts=True,
                        ),
                        label="writer_batch_regular",
                    )
                if large_items:
                    self._run_database_write(
                        lambda: self.db.replace_document_blocks_many(
                            large_items,
                            update_fts=False,
                            remove_old_fts=True,
                        ),
                        label="writer_batch_large",
                    )
                    for item in large_items:
                        if item.get("status") not in {"success", "metadata_only"}:
                            continue
                        self._deferred_fts_file_ids.update(
                            int(file_id)
                            for file_id in item.get("file_ids") or [item["file_id"]]
                        )
            else:
                self._run_database_write(
                    lambda: self.db.replace_document_blocks_many(
                        items,
                        update_fts=False,
                    ),
                    label="writer_batch_content",
                )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            batch_block_count = sum(len(getattr(item, "blocks", []) or []) for item in artifacts)
            self.summary.artifacts += len(artifacts)
            self.summary.blocks += batch_block_count
            self.summary.text_bytes += sum(
                len(block.raw_text.encode("utf-8"))
                for artifact in artifacts
                for block in (getattr(artifact, "blocks", []) or [])
            )
            self.summary.batches += 1
            self.summary.write_ms += elapsed_ms
            if elapsed_ms >= 5000:
                logger.warning(
                    "Slow SQLite index batch: artifacts=%s blocks=%s write_ms=%s update_fts=%s",
                    len(artifacts),
                    batch_block_count,
                    elapsed_ms,
                    self.update_fts,
                )
            for artifact in artifacts:
                spool_path = getattr(artifact, "spool_path", None)
                if spool_path:
                    Path(spool_path).unlink(missing_ok=True)
            if self.on_written is not None:
                self.on_written(artifacts, elapsed_ms)
            self._busy_circuit.reset_after_success(stage="writer_batch")
            return True
        except sqlite3.OperationalError as exc:
            if _is_busy_error(exc):
                self._busy_circuit.note_busy(
                    exc,
                    stage="writer_batch",
                    batch_size=len(artifacts),
                    wait_timeout_seconds=0.5,
                )
                return False
            raise
        finally:
            self._flushing.clear()
            if reserved:
                end_index_write()

    def _run_database_write(self, operation: Callable[[], Any], *, label: str) -> Any:
        if self.write_dispatcher is None:
            return operation()
        item = self.write_dispatcher.submit(
            operation,
            priority=10,
            label=label,
            defer_if_search=False,
        )
        if item.status != "COMMITTED":
            message = item.error or f"数据库写入未提交：{label} ({item.status})"
            if item.status == "FAILED_RETRYABLE" and _is_busy_error_message(
                item.error
            ):
                # Let the writer's SQLite busy circuit own the retry budget.
                # Converting a transient dispatcher result to RuntimeError
                # kills the writer thread and leaves every queued task
                # without a terminal result.
                raise sqlite3.OperationalError(message)
            raise RuntimeError(message)
        return item.result
    @staticmethod
    def _to_item(artifact: Any) -> dict[str, Any]:
        file_id = int(artifact.file_id)
        file_ids = [file_id, *[int(value) for value in (getattr(artifact, "alias_file_ids", ()) or ())]]
        return {
            "file_id": file_id,
            "file_ids": file_ids,
            "filename": artifact.file_path.name,
            "path": str(artifact.file_path),
            "blocks": artifact.blocks,
            "parser_name": artifact.parser_name,
            "parser_version": getattr(artifact, "parser_version", None),
            "status": artifact.status,
            "error_code": artifact.error_code,
            "error_message": artifact.error_message,
            "diagnostics": list(
                getattr(artifact, "diagnostics", []) or []
            ),
            "content_key": getattr(artifact, "content_key", None),
            "content_hash_full": getattr(artifact, "content_hash_full", None),
            "task_id": getattr(artifact, "task_id", None),
        }

    def _raise_if_failed(self) -> None:
        if self._error is not None:
            if isinstance(self._error, SQLiteBusyCircuitOpen):
                raise self._error
            raise RuntimeError("Index writer failed") from self._error


def _is_busy_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "database is locked" in message or "database is busy" in message


def _is_busy_error_message(message: str) -> bool:
    text = str(message or "").lower()
    return "database is locked" in text or "database is busy" in text
