from __future__ import annotations

import queue
import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class WriterSummary:
    artifacts: int = 0
    blocks: int = 0
    text_bytes: int = 0
    batches: int = 0
    write_ms: int = 0


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
        on_written: Callable[[list[Any], int], None] | None = None,
    ) -> None:
        self.db = db
        self.update_fts = bool(update_fts)
        self.batch_blocks = max(1, int(batch_blocks))
        self.batch_bytes = max(1024, int(batch_bytes))
        self.max_delay = max(0.01, int(max_delay_ms) / 1000)
        self.on_written = on_written
        self.summary = WriterSummary()
        self._queue: queue.Queue[object] = queue.Queue(maxsize=max(2, int(queue_size)))
        self._thread = threading.Thread(target=self._run, name="lfts-index-writer", daemon=True)
        self._error: BaseException | None = None
        self._started = False
        self._aborted = threading.Event()
        self._flushing = threading.Event()

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

    def finish(self) -> WriterSummary:
        if not self._started:
            return self.summary
        while True:
            self._raise_if_failed()
            try:
                self._queue.put(self._STOP, timeout=0.1)
                break
            except queue.Full:
                continue
        self._thread.join()
        self._raise_if_failed()
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
                    self._flush(pending)
                    pending = []
                    blocks = 0
                    text_bytes = 0
                    deadline = time.monotonic() + self.max_delay
                    continue
                if item is self._STOP:
                    if not self._aborted.is_set():
                        self._flush(pending)
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
                    self._flush(pending)
                    pending = []
                    blocks = 0
                    text_bytes = 0
                    deadline = time.monotonic() + self.max_delay
        except BaseException as exc:
            self._error = exc

    def _flush(self, artifacts: list[Any]) -> None:
        if not artifacts:
            return
        started = time.perf_counter()
        self._flushing.set()
        try:
            items = [self._to_item(artifact) for artifact in artifacts]
            self.db.replace_document_blocks_many(items, update_fts=self.update_fts)
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
        finally:
            self._flushing.clear()

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
            raise RuntimeError("Index writer failed") from self._error
