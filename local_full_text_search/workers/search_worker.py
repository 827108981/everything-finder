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
        # run() 在 MainWindow 创建的搜索 QThread 中执行，而不是在界面主线程执行。
        # active_lease 用来保护本次正在读取的旧 generation，避免搜索尚未结束时
        # 清理程序把对应的只读数据库文件删除。
        active_lease = None
        active = None
        try:
            # 真正打开数据库之前先响应取消，避免创建无意义的连接。
            if self.token.cancelled:
                raise CancelledError("搜索已取消")
            # IndexGenerationStore 是“索引版本管理器”，负责从 manifest 找到
            # 当前已经完整发布、可以放心读取的数据库版本。
            store = IndexGenerationStore(self.db_path)
            active = store.active()
            # generation 模式下搜索已发布快照；没有快照时才兼容旧数据库路径。
            search_path = active.path if active is not None else self.db_path
            if active is not None:
                # 登记“这个 generation 正在被搜索使用”。release 前不能清理它。
                active_lease = store.lease_registry.acquire(active.generation)
                log_event(
                    "search.snapshot.lease.acquired",
                    component="search_worker",
                    request_id=self.request_id,
                    generation=active.generation,
                    database_path=str(active.path),
                )
            # 已发布 generation 是不可变快照，所以必须只读打开。
            db = DatabaseManager(search_path, read_only=active is not None)
            with self._database_lock:
                # 保存连接管理器，使另一个线程调用 cancel() 时可以中断 SQLite。
                self._database = db
            if active is not None:
                log_event(
                    "search.snapshot.selected",
                    component="search_worker",
                    request_id=self.request_id,
                    generation=active.generation,
                    database_path=str(active.path),
                )
            # SearchWorker 只负责线程、生命周期和信号；真正的检索算法交给
            # SearchEngine。generation_isolated=True 表示索引器正在写另一份文件，
            # 搜索不需要暂停后台索引。
            engine = SearchEngine(
                db,
                generation_isolated=active is not None,
                snapshot_generation=active.generation if active is not None else None,
            )
            # engine.search() 是同步函数，但这里位于后台 QThread，所以不会卡住界面。
            # finished 是跨线程 Qt 信号，MainWindow 收到后再在主线程更新结果列表。
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
            # 无论成功、失败还是取消，都必须解除状态并归还 generation 租约。
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
                # 清除引用，表示本次后台查询已经不再持有数据库管理器。
                self._database = None

    @Slot()
    def cancel(self) -> None:
        # cancel() 可能由界面线程调用：先设置线程安全取消标记，再通知搜索算法。
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
            # SQLite 正在执行耗时查询时，仅设置 Python 标记不够；这里直接请求
            # 中断活动连接，让后台线程尽快从 SQL 调用中返回。
            database.interrupt_active_connections()
