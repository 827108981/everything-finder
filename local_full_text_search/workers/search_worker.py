from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import (
    CancelledError,
    OperationCancelled,
    QuiesceTimeout,
    SearchTimeout,
    SearchLeaseBusy,
)
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.logging_service import log_event, log_exception

logger = logging.getLogger(__name__)


class SearchWorker(QObject):
    finished = Signal(object)
    partial = Signal(object)
    progress = Signal(object)
    cancelled = Signal()
    quiesce_timeout = Signal(str)
    timed_out = Signal(str)
    failed = Signal(str)
    lease_status = Signal(str)

    def __init__(self, db_path: Path, query: SearchQuery) -> None:
        super().__init__()
        self.db_path = db_path
        self.query = query
        self.token = CancelToken()
        self._database: DatabaseManager | None = None
        self._database_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.request_id = f"search-{uuid.uuid4().hex}"

    @Slot()
    def run(self) -> None:
        active_lease = None
        active = None
        try:
            if self.token.cancelled:
                raise CancelledError("搜索已取消")
            store = IndexGenerationStore(self.db_path)
            active = store.active()
            search_path = active.path if active is not None else self.db_path
            if active is not None:
                active_lease = store.lease_registry.acquire(active.generation)
                log_event(
                    "search.snapshot.lease.acquired",
                    component="search_worker",
                    request_id=self.request_id,
                    generation=active.generation,
                    database_path=str(active.path),
                )
            db = DatabaseManager(search_path, read_only=active is not None)
            with self._database_lock:
                self._database = db
            if active is not None:
                log_event(
                    "search.snapshot.selected",
                    component="search_worker",
                    request_id=self.request_id,
                    generation=active.generation,
                    database_path=str(active.path),
                )
            engine = SearchEngine(
                db,
                generation_isolated=active is not None,
                snapshot_generation=active.generation if active is not None else None,
            )
            self.finished.emit(
                engine.search(
                    self.query,
                    self.token,
                    request_id=self.request_id,
                    status_callback=self.lease_status.emit,
                    progress_callback=self.progress.emit,
                    partial_callback=self.partial.emit,
                )
            )
        except QuiesceTimeout as exc:
            self.quiesce_timeout.emit(str(exc) or "索引安全暂停超时")
        except (TimeoutError, SearchLeaseBusy) as exc:
            self.quiesce_timeout.emit(str(exc) or "等待索引安全暂停超时")
        except SearchTimeout as exc:
            self.timed_out.emit(str(exc) or "搜索达到时间上限")
        except CancelledError:
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Search failed")
            log_exception(
                "search.worker.failed",
                exc,
                component="search_worker",
                request_id=self.request_id,
                database_path=str(active.path if active is not None else self.db_path),
                generation=active.generation if active is not None else None,
                generation_isolated=active is not None,
                query_mode=str(self.query.mode),
                query_length=len(self.query.text),
            )
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            self.lease_status.emit("released")
            if active_lease is not None:
                log_event(
                    "search.snapshot.lease.released",
                    component="search_worker",
                    request_id=self.request_id,
                    generation=active.generation if active is not None else None,
                    database_path=str(active.path) if active is not None else "",
                )
                active_lease.release()
            with self._database_lock:
                self._database = None

    @Slot()
    def cancel(self) -> None:
        self._cancel_event.set()
        self.token.cancel(reason="USER_CANCELLED")
        log_event(
            "search.cancel.requested",
            component="search_worker",
            request_id=self.request_id,
            query_length=len(self.query.text),
        )
        with self._database_lock:
            database = self._database
        if database is not None:
            database.interrupt_active_connections()
