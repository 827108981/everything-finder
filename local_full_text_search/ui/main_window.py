from __future__ import annotations

import csv
import logging
import os
import shutil
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer, QThread, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.config.constants import APP_DISPLAY_NAME, FILE_TYPE_GROUPS, LOG_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_generation_store import IndexGenerationStore
from local_full_text_search.core.file_monitor import FileMonitor
from local_full_text_search.core.open_location import (
    open_file,
    open_file_at_location,
    open_parent_folder,
)
from local_full_text_search.core.search_time_estimator import SearchEstimateContext, SearchTimeEstimator
from local_full_text_search.core.search_index_coordinator import (
    get_search_index_coordinator,
)
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.services.logging_service import log_event
from local_full_text_search.services.database_location_service import (
    DatabaseLocationService,
)
from local_full_text_search.ui.preview_panel import PreviewPanel
from local_full_text_search.ui.result_view import ResultView, result_hits
from local_full_text_search.version import __build_timestamp__, __version__
from local_full_text_search.workers.scan_worker import ScanWorker
from local_full_text_search.workers.scope_exclusion_worker import ScopeExclusionWorker
from local_full_text_search.workers.root_removal_worker import RootRemovalWorker
from local_full_text_search.workers.database_migration_worker import (
    DatabaseMigrationWorker,
)
from local_full_text_search.workers.database_snapshot_worker import (
    DatabaseSnapshotWorker,
)
from local_full_text_search.workers.search_worker import SearchWorker
from local_full_text_search.workers.search_history_worker import SearchHistoryWorker


PAGE_INDEX = {"search": 0, "index": 1, "failed": 2, "settings": 3}
logger = logging.getLogger(__name__)


def _log_ui_lifecycle_event(event: str, **fields: object) -> None:
    """Best-effort lifecycle logging that cannot break Qt shutdown."""

    try:
        log_event(event, component="ui.main_window", **fields)
    except BaseException:
        logger.debug("Unable to write UI lifecycle event: %s", event, exc_info=True)


def performance_mode_notice() -> str:
    return (
        "性能模式适合在电脑空闲时使用。程序会根据本机配置尽可能使用 "
        "CPU、内存和磁盘带宽来缩短索引时间；检测到资源压力时会自动降低并发。"
    )


def format_index_scope_status(readiness: dict[str, object]) -> str:
    stats_complete = bool(readiness.get("stats_complete", True))
    unknown = any(
        readiness.get(key) is None
        for key in ("complete_files", "eligible_files", "blocking_files")
    ) or not stats_complete
    complete = readiness.get("complete_files")
    eligible = readiness.get("eligible_files")
    blocking = readiness.get("blocking_files")
    excluded = readiness.get("manual_excluded_files")
    video = readiness.get("video_excluded")
    complete_text = (
        "统计更新中"
        if unknown
        else f"{int(complete or 0)}/{int(eligible or 0)}"
    )
    blocking_count = int(blocking or 0) if blocking is not None else 0
    details = []
    if excluded is not None and int(excluded or 0):
        details.append(f"已人工排除 {int(excluded):,} 个")
    if video is not None and int(video or 0):
        details.append(f"排除视频 {int(video):,} 个")
    task_state = str(readiness.get("background_task_state") or "")
    unfinished = int(readiness.get("unfinished_tasks") or 0)
    if bool(readiness.get("scan_active")):
        # Database snapshots can briefly observe no parse-task rows while the
        # scan worker is still draining writers or publishing FTS.  Keep this
        # lifecycle state visible instead of claiming that the whole run has
        # converged.
        details.append("索引扫描收尾中")
    in_progress = int(
        readiness.get("in_progress_files")
        or readiness.get("in_progress")
        or 0
    )
    if in_progress:
        details.append(f"待处理/处理中 {in_progress:,} 个")
    if not bool(readiness.get("scan_active")) and not in_progress:
        if task_state == "running":
            details.append(f"后台任务运行中 {unfinished} 个")
        elif task_state == "residual":
            details.append(f"后台任务有残留 {unfinished} 个")
        elif task_state == "finishing":
            details.append("索引收尾中")
        elif task_state == "converged":
            details.append("后台任务已收敛")
    orphan_pending = int(readiness.get("orphan_pending_files") or 0)
    if orphan_pending:
        details.append(f"待调度 {orphan_pending} 个")
    if unknown:
        text = f"当前范围 {complete_text}"
    elif bool(readiness.get("ready")):
        text = f"当前范围完成 {complete_text}"
    elif bool(readiness.get("search_available")):
        text = f"索引可搜索 {complete_text} · 待处理 {blocking_count}"
    else:
        if blocking_count:
            text = f"索引未完成 {complete_text} · 待处理 {blocking_count}"
        else:
            residual = []
            unfinished = int(readiness.get("unfinished_tasks") or 0)
            unpublished = int(readiness.get("unpublished_candidates") or 0)
            if unfinished:
                residual.append(f"残留任务 {unfinished}")
            if unpublished:
                residual.append(f"未发布索引 {unpublished}")
            if bool(readiness.get("content_fts_dirty")):
                residual.append("全文索引待发布")
            text = f"文件已完成 {complete_text} · 索引状态待修复"
            if residual:
                text += " · " + " · ".join(residual)
    if details:
        text += " · " + " · ".join(details)
    return text


class MainWindow(QMainWindow):
    """应用级编排窗口：连接页面、后台任务和全局生命周期。

    MainWindow知道SearchPage和SearchWorker的存在，是为了把“用户意图”安排成
    一次完整后台任务，并在结束后把结果送回页面。它持有Worker不等于它执行
    搜索算法；真正检索仍由SearchWorker和SearchEngine负责。
    """

    file_change_detected = Signal(str)

    def __init__(
        self,
        db: DatabaseManager,
        settings: AppSettings,
        settings_service: SettingsService,
        *,
        database_location_service: DatabaseLocationService | None = None,
        generation_base_path: Path | None = None,
        defer_initial_refresh: bool = False,
    ) -> None:
        super().__init__()
        self.db = db
        # db 仍然是供索引和界面管理功能使用的可写控制数据库。搜索会通过
        # generation 管理器选择不可变的已发布快照，不会在扫描写入时读取 db。
        self.generation_base_path = Path(generation_base_path or db.db_path).resolve()
        self.index_generation_store = IndexGenerationStore(self.generation_base_path)
        self.settings = settings
        self.settings_service = settings_service
        self.database_location_service = (
            database_location_service or DatabaseLocationService()
        )
        # 搜索尚未开始时两者都是None；任务开始后，MainWindow持有QThread和
        # SearchWorker，便于取消、判断是否仍在运行、等待退出并防止对象过早销毁。
        self.search_thread: QThread | None = None
        self.search_worker: SearchWorker | None = None
        self.search_priority_active = False
        self.search_history_thread: QThread | None = None
        self.search_history_worker: SearchHistoryWorker | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.exclusion_thread: QThread | None = None
        self.exclusion_worker: ScopeExclusionWorker | None = None
        self.database_migration_thread: QThread | None = None
        self.database_migration_worker: DatabaseMigrationWorker | None = None
        self.database_snapshot_thread: QThread | None = None
        self.database_snapshot_worker: DatabaseSnapshotWorker | None = None
        self._snapshot_refresh_pending = False
        self.root_removal_thread: QThread | None = None
        self.root_removal_worker: RootRemovalWorker | None = None
        self.database_migration_outcome: tuple[str, object] | None = None
        self.exclusion_outcome: tuple[str, object] | None = None
        self.exclusion_progress_payload: dict[str, object] = {}
        self.exclusion_started_at = 0.0
        # 用户在旧搜索尚未结束时又提交新条件，先将pending_search设为True，
        # 取消旧任务；等旧线程真正清理完成后再启动最新搜索，避免两个结果竞争UI。
        self.pending_search = False
        self.pending_monitor_scan = False
        self.force_complete_after_retry = False
        self.pending_force_complete_confirmation = False
        self.force_complete_file_ids: list[int] = []
        self.last_search_readiness_refresh = 0.0
        self._scan_refresh_pending = False
        self._scan_progress_text = "正在准备索引"
        self._scan_terminal_payload: dict[str, object] = {}
        self._active_scan_id = ""
        self._failed_snapshot_sequence = 0
        self._latest_candidate_snapshot_sequence = 0
        self._startup_snapshot_ready = not defer_initial_refresh
        self._startup_scan_requested = False
        # A snapshot completion also occurs after every scan. Keep the startup
        # request one-shot so that completion snapshots cannot start another
        # full enabled-root scan on every launch cycle.
        self._startup_scan_in_flight = False
        self._startup_scan_consumed = False
        self._scan_generation = 0
        self._active_scan_generation = 0
        self._pending_scan_root_ids: set[int] = set()
        self._pending_scan_all_roots = False
        self._pending_scan_performance = False
        self._latest_readiness: dict[str, object] = {}
        self._latest_snapshot: dict[str, object] = {}
        self._last_snapshot_completed_monotonic = 0.0
        self.closing = False
        self._interrupt_shutdown_in_progress = False
        self.page = 1
        self.total_confirmed = 0
        self.file_monitor = FileMonitor(lambda path: self.file_change_detected.emit(str(path)))
        self.force_close_timer = QTimer(self)
        self.force_close_timer.setSingleShot(True)
        self.force_close_timer.setInterval(2_500)
        self.force_close_timer.timeout.connect(
            lambda: self._force_exit("close_timeout")
        )
        self.exclusion_heartbeat_timer = QTimer(self)
        self.exclusion_heartbeat_timer.setInterval(1_000)
        self.exclusion_heartbeat_timer.timeout.connect(
            self.on_exclusion_heartbeat
        )
        self.startup_scan_timer = QTimer(self)
        self.startup_scan_timer.setSingleShot(True)
        self.startup_scan_timer.timeout.connect(self._start_startup_incremental_scan)
        self.monitor_scan_timer = QTimer(self)
        self.monitor_scan_timer.setSingleShot(True)
        self.monitor_scan_timer.timeout.connect(self._start_pending_monitor_scan)
        self.failed_snapshot_timer = QTimer(self)
        self.failed_snapshot_timer.setInterval(1_500)
        self.failed_snapshot_timer.timeout.connect(
            self._refresh_failed_page_during_scan
        )

        self.setWindowTitle(
            f"{APP_DISPLAY_NAME} {__version__} {__build_timestamp__}"
        )
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        # 开始组装整个窗口。_build_shell() 会创建 SearchPage；SearchPage 的
        # __init__() 又会创建 SearchBox。调用返回后，搜索框才算初始化完成。
        self._build_shell()
        self.settings_page.set_database_info(self.generation_base_path)
        self._install_shortcuts()
        self.file_change_detected.connect(self.on_monitored_file_changed)
        if defer_initial_refresh:
            # 首次汇总查询可能扫描较大的SQLite数据库，因此等窗口可以显示后，
            # 再把它安排到Qt界面线程之外执行，避免启动阶段界面迟迟不出现。
            QTimer.singleShot(0, self.start_database_snapshot)
        else:
            self.refresh_all()

    def _build_shell(self) -> None:
        central = QWidget()
        central.setObjectName("AppRoot")
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.top_bar = TopBar()
        self.top_bar.index_requested.connect(lambda: self.switch_page("index"))
        root_layout.addWidget(self.top_bar)

        body = QFrame()
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.nav_requested.connect(self.on_nav_requested)
        body_layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        # 创建搜索页面会进入 SearchPage.__init__(settings)。SearchBox 不是由
        # MainWindow 直接创建，而是作为 SearchPage 的内部子组件被创建。
        self.search_page = SearchPage(self.settings)
        self.index_page = IndexPage()
        self.failed_page = FailedPage()
        self.settings_page = SettingsPage(self.settings)

        self.search_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.search_splitter.setObjectName("SearchSplitter")
        self.search_splitter.addWidget(self.search_page)
        self.preview_panel = PreviewPanel()
        self.preview_panel.setVisible(False)
        self.preview_panel.open_file_requested.connect(self.open_path)
        self.preview_panel.open_folder_requested.connect(self.open_folder_path)
        self.preview_panel.open_location_requested.connect(self.open_location)
        self.preview_panel.close_requested.connect(self.hide_preview)
        self.search_splitter.addWidget(self.preview_panel)
        self.search_splitter.setSizes([980, 0])

        self.stack.addWidget(self.search_splitter)
        self.stack.addWidget(self.index_page)
        self.stack.addWidget(self.failed_page)
        self.stack.addWidget(self.settings_page)
        body_layout.addWidget(self.stack, 1)
        root_layout.addWidget(body, 1)
        self.setCentralWidget(central)

        # SearchPage 把内部 SearchBox 的搜索信号转发到这里；MainWindow 收到后
        # 才调用 request_search()。初始化阶段只建立连接，不会立即执行搜索。
        self.search_page.search_requested.connect(self.request_search)
        self.search_page.stop_requested.connect(self.cancel_search)
        self.search_page.add_root_requested.connect(self.add_root)
        self.search_page.previous_page_requested.connect(self.previous_page)
        self.search_page.next_page_requested.connect(self.next_page_search)
        self.search_page.open_requested.connect(self.open_path)
        self.search_page.open_folder_requested.connect(self.open_folder_path)
        self.search_page.open_location_requested.connect(self.open_location)
        self.search_page.reindex_requested.connect(self.reindex_file)
        self.search_page.result_selected.connect(self.show_preview)
        self.search_page.clear_history_requested.connect(self.clear_search_history)

        self.index_page.add_root_requested.connect(self.add_root)
        self.index_page.scan_requested.connect(self.start_scan)
        self.index_page.performance_scan_requested.connect(
            self.confirm_performance_scan
        )
        self.index_page.scope_performance_scan_requested.connect(
            self.confirm_performance_scan
        )
        self.index_page.pause_requested.connect(lambda: self.scan_worker.pause() if self.scan_worker else None)
        self.index_page.resume_requested.connect(lambda: self.scan_worker.resume() if self.scan_worker else None)
        self.index_page.mode_switch_requested.connect(
            self.request_scan_mode_switch
        )
        self.index_page.cancel_requested.connect(self.cancel_scan)
        self.index_page.toggle_root_requested.connect(self.toggle_root)
        self.index_page.remove_root_requested.connect(self.remove_root)
        self.index_page.open_folder_requested.connect(self.open_path)
        self.index_page.failed_requested.connect(lambda: self.switch_page("failed"))

        self.failed_page.retry_requested.connect(self.retry_failed_files)
        self.failed_page.open_folder_requested.connect(self.open_folder_path)
        self.failed_page.refresh_requested.connect(self.refresh_failed_page)
        self.failed_page.export_requested.connect(self.export_failed_rows)
        self.failed_page.exclude_requested.connect(
            self.exclude_failed_files
        )
        self.failed_page.restore_requested.connect(
            self.restore_excluded_files
        )
        self.failed_page.force_complete_requested.connect(
            self.request_force_complete
        )
        self.failed_page.cancel_exclusion_requested.connect(
            self.cancel_scope_exclusion
        )

        self.settings_page.save_requested.connect(self.save_settings)
        self.settings_page.database_migration_requested.connect(
            self.request_database_migration
        )
        self.settings_page.database_migration_cancel_requested.connect(
            self.cancel_database_migration
        )

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search_page.focus_search)
        QShortcut(QKeySequence("Esc"), self, activated=self.handle_escape)

    def handle_escape(self) -> None:
        if self.preview_panel.isVisible():
            self.hide_preview()
        elif self.search_page.text():
            self.search_page.clear_search()

    def on_nav_requested(self, key: str) -> None:
        self.switch_page(key)

    def switch_page(self, key: str) -> None:
        self.stack.setCurrentIndex(PAGE_INDEX[key])
        title_map = {
            "search": "本地全文搜索",
            "index": "索引管理",
            "failed": "未成功索引",
            "settings": "设置",
        }
        subtitle_map = {
            "search": "搜索文件名、正文、表格、幻灯片和图片文字",
            "index": "管理需要搜索的本地文件夹和索引状态",
            "failed": "查看无法解析、仅元数据或需要处理的文件",
            "settings": "调整搜索、索引和 OCR 默认行为",
        }
        self.top_bar.set_title(title_map[key], subtitle_map[key])
        self.sidebar.set_active(key)
        if (
            key == "failed"
            and self.exclusion_thread is None
        ):
            self.request_failed_snapshot_refresh()

    def refresh_all(self) -> None:
        if (
            self._latest_snapshot
            and time.monotonic() - self._last_snapshot_completed_monotonic < 5.0
        ):
            self.on_database_snapshot_finished(self._latest_snapshot)
            return
        # Once a generation has been published, file statistics live in that
        # immutable snapshot while scope/task configuration lives in the
        # control DB.  Do not fall back to the control DB's intentionally empty
        # file tables after the cached UI snapshot expires.
        if self.index_generation_store.active() is not None:
            if self.database_snapshot_thread is None:
                self.start_database_snapshot()
            return
        roots = self.db.list_roots()
        stats = self.db.stats()
        readiness = self.db.index_readiness(reconcile=False)
        self._latest_readiness = dict(readiness)
        self.search_page.set_roots(roots)
        self.search_page.set_stats(stats, has_roots=bool(roots))
        search_readiness = self._search_readiness_from_published_snapshot(readiness)
        self.search_page.set_index_ready(
            bool(search_readiness.get("search_available")),
            search_readiness,
        )
        self.sidebar.set_failed_alert(
            0
            if self.scan_thread is not None
            or not bool(readiness.get("failure_alert_ready", False))
            else int(stats.get("blocking_files") or 0)
        )
        self.index_page.set_roots(roots, self.root_stats_by_id())
        self.index_page.set_readiness(readiness)
        if self.stack.currentIndex() == PAGE_INDEX["failed"]:
            self.request_failed_snapshot_refresh()
        self.update_index_status()
        self.search_page.set_history(self.db.search_history())
        self.refresh_file_monitor()

    def _active_snapshot_path(self) -> Path:
        active = self.index_generation_store.active()
        return active.path if active is not None else self.db.db_path

    def _search_readiness_from_published_snapshot(
        self,
        readiness: dict[str, object],
    ) -> dict[str, object]:
        """Base search availability on the last published snapshot only."""

        active = self.index_generation_store.active()
        if active is None:
            return {
                **readiness,
                "search_available": False,
                "index_complete": False,
                "ready": False,
                "search_is_partial": False,
            }
        partial = bool(active.partial)
        return {
            **readiness,
            "search_available": True,
            "index_complete": not partial,
            "ready": not partial,
            "search_is_partial": partial,
            "active_partial": partial,
            "active_generation": active.generation,
            "active_database_path": str(active.path),
            "active_published_at": active.published_at,
        }

    def start_database_snapshot(self) -> None:
        if self.closing or self.database_snapshot_thread is not None:
            return
        active_snapshot = self.index_generation_store.active()
        self._failed_snapshot_sequence += 1
        self.database_snapshot_thread = QThread()
        self.database_snapshot_worker = DatabaseSnapshotWorker(
            self.db.db_path,
            page_size=self.failed_page.page_size,
            page_offsets={
                "blocking": self.failed_page.page_offset("blocking"),
                "excluded": self.failed_page.page_offset("excluded"),
                "metadata": self.failed_page.page_offset("metadata"),
            },
            fallback_snapshot=self._latest_snapshot,
            published_snapshot_path=(
                active_snapshot.path if active_snapshot is not None else None
            ),
            generation_base_path=self.generation_base_path,
            scan_id=self._active_scan_id,
            snapshot_sequence=self._failed_snapshot_sequence,
        )
        self.database_snapshot_worker.moveToThread(self.database_snapshot_thread)
        self.database_snapshot_thread.started.connect(
            self.database_snapshot_worker.run
        )
        self.database_snapshot_worker.quick_ready.connect(
            self.on_database_snapshot_quick_ready
        )
        self.database_snapshot_worker.finished.connect(
            self.on_database_snapshot_finished
        )
        self.database_snapshot_worker.failed.connect(
            self.on_database_snapshot_failed
        )
        self.database_snapshot_worker.cancelled.connect(
            self.database_snapshot_worker.deleteLater
        )
        self.database_snapshot_worker.finished.connect(
            self.database_snapshot_worker.deleteLater
        )
        self.database_snapshot_worker.failed.connect(
            self.database_snapshot_worker.deleteLater
        )
        self.database_snapshot_worker.cancelled.connect(
            self.database_snapshot_thread.quit
        )
        self.database_snapshot_worker.finished.connect(
            self.database_snapshot_thread.quit
        )
        self.database_snapshot_worker.failed.connect(
            self.database_snapshot_thread.quit
        )
        self.database_snapshot_thread.finished.connect(
            self.cleanup_database_snapshot
        )
        self.database_snapshot_thread.finished.connect(
            self.database_snapshot_thread.deleteLater
        )
        self.database_snapshot_thread.start()

    def request_failed_snapshot_refresh(self) -> None:
        """Refresh failure-page data without running database queries on UI."""

        if self.closing:
            return
        if (
            self.database_snapshot_thread is None
            and self._latest_snapshot
            and time.monotonic() - self._last_snapshot_completed_monotonic < 2.0
        ):
            self.failed_page.set_snapshot_loading(False)
            return
        self.failed_page.set_snapshot_loading(True)
        self.failed_page.set_status("正在加载未成功索引…")
        if self.database_snapshot_thread is not None:
            self._snapshot_refresh_pending = True
            return
        self._failed_snapshot_sequence += 1
        self.start_database_snapshot()

    def on_database_snapshot_quick_ready(self, payload: object) -> None:
        if not isinstance(payload, dict) or self.closing:
            return
        candidate_scan_id = str(payload.get("candidate_scan_id") or "")
        candidate_sequence = int(payload.get("candidate_snapshot_sequence") or 0)
        if candidate_scan_id and candidate_scan_id != self._active_scan_id:
            return
        if candidate_sequence and candidate_sequence < self._latest_candidate_snapshot_sequence:
            return
        if candidate_sequence:
            self._latest_candidate_snapshot_sequence = candidate_sequence
        roots = list(payload.get("roots") or [])
        readiness = dict(payload.get("readiness") or {})
        self._latest_readiness = readiness
        self.search_page.set_roots(roots)
        self.search_page.set_stats(readiness, has_roots=bool(roots))
        self.sidebar.set_failed_alert(
            0
            if self.scan_thread is not None
            or not bool(readiness.get("failure_alert_ready", False))
            else int(readiness.get("blocking_files") or 0)
        )
        search_readiness = self._search_readiness_from_published_snapshot(readiness)
        self.search_page.set_index_ready(
            bool(search_readiness.get("search_available")),
            search_readiness,
        )
        self.index_page.set_roots(roots, {})
        self.index_page.set_readiness(readiness)
        if bool(payload.get("candidate_live_failures")):
            failed_counts = dict(payload.get("failed_counts") or {})
            self.failed_page.set_rows(
                list(payload.get("failed_rows") or []),
                totals=failed_counts,
                all_blocking_ids=[
                    int(value)
                    for value in (payload.get("all_blocking_ids") or [])
                ],
            )
            self.failed_page.set_readiness(readiness)
            self.failed_page.set_snapshot_loading(False)
            in_progress_total = int(failed_counts.get("in_progress") or 0)
            self.failed_page.set_status(
                "已加载候选库实时失败项"
                + (
                    f"；另有 {in_progress_total:,} 个待处理/处理中"
                    if in_progress_total
                    else ""
                )
                + "；当前仍使用上一版已发布数据库搜索"
            )
        self.update_index_status(readiness)
        self.top_bar.set_index_status("正在加载索引统计…", is_running=True)

    def on_database_snapshot_finished(self, payload: object) -> None:
        if not isinstance(payload, dict) or self.closing:
            return
        candidate_scan_id = str(payload.get("candidate_scan_id") or "")
        candidate_sequence = int(payload.get("candidate_snapshot_sequence") or 0)
        if candidate_scan_id and candidate_scan_id != self._active_scan_id:
            log_event = logger.debug
            log_event(
                "Discarding stale candidate snapshot scan_id=%s active_scan_id=%s",
                candidate_scan_id,
                self._active_scan_id,
            )
            return
        if candidate_sequence and candidate_sequence < self._latest_candidate_snapshot_sequence:
            return
        if candidate_sequence:
            self._latest_candidate_snapshot_sequence = candidate_sequence
        roots = list(payload.get("roots") or [])
        stats = dict(payload.get("stats") or {})
        readiness = dict(payload.get("readiness") or stats)
        failed_rows = list(payload.get("failed_rows") or [])
        excluded_rows = list(payload.get("excluded_rows") or [])
        metadata_rows = list(payload.get("metadata_rows") or [])
        failed_counts = dict(payload.get("failed_counts") or {})
        self._latest_snapshot = dict(payload)
        self._last_snapshot_completed_monotonic = time.monotonic()
        self._latest_readiness = readiness
        self.search_page.set_roots(roots)
        self.search_page.set_stats(stats, has_roots=bool(roots))
        self.sidebar.set_failed_alert(
            0
            if self.scan_thread is not None
            or not bool(readiness.get("failure_alert_ready", False))
            else int(stats.get("blocking_files") or 0)
        )
        search_readiness = self._search_readiness_from_published_snapshot(readiness)
        self.search_page.set_index_ready(
            bool(search_readiness.get("search_available")),
            search_readiness,
        )
        self.index_page.set_roots(
            roots,
            dict(payload.get("root_stats_by_id") or {}),
        )
        self.index_page.set_readiness(readiness)
        self.failed_page.set_rows(
            failed_rows,
            excluded_rows,
            metadata_rows,
            totals=failed_counts,
            all_blocking_ids=[
                int(value) for value in (payload.get("all_blocking_ids") or [])
            ],
        )
        self.failed_page.set_readiness(readiness)
        self.failed_page.set_snapshot_loading(False)
        if bool(payload.get("snapshot_timeout")):
            self.failed_page.set_status("统计仍在后台更新，当前显示快速状态")
        elif bool(payload.get("candidate_live_failures")):
            blocker_total = int(failed_counts.get("blocking") or 0)
            in_progress_total = int(failed_counts.get("in_progress") or 0)
            loaded_rows = len(failed_rows)
            self.failed_page.set_status(
                f"已加载候选库实时失败项 {loaded_rows:,}/{blocker_total:,}；"
                + (
                    f"另有 {in_progress_total:,} 个待处理/处理中；"
                    if in_progress_total
                    else ""
                )
                + "当前仍使用上一版已发布数据库搜索"
            )
        else:
            blocker_total = int(failed_counts.get("blocking") or 0)
            in_progress_total = int(failed_counts.get("in_progress") or 0)
            if blocker_total and not failed_rows:
                self.failed_page.set_status(
                    f"统计显示 {blocker_total:,} 个阻断项，但当前未取得明细；请重新更新索引"
                )
            elif blocker_total:
                text = f"已加载 {len(failed_rows):,} 个已失败明细"
                if in_progress_total:
                    text += f"；另有 {in_progress_total:,} 个待处理/处理中"
                self.failed_page.set_status(text)
            elif in_progress_total:
                self.failed_page.set_status(
                    f"暂无已失败文件；当前有 {in_progress_total:,} 个待处理/处理中"
                )
            else:
                self.failed_page.set_status("未发现未成功索引项")
        self.search_page.set_history(list(payload.get("history") or []))
        self.sidebar.set_failed_alert(
            0
            if self.scan_thread is not None
            or not bool(readiness.get("failure_alert_ready", False))
            else int((payload.get("failed_counts") or {}).get("blocking", 0))
        )
        self.update_index_status(readiness)
        self.refresh_file_monitor()
        self._startup_snapshot_ready = True
        if (
            self._startup_scan_requested
            and not self._startup_scan_consumed
            and not self._startup_scan_in_flight
            and not self.startup_scan_timer.isActive()
            and not self.closing
        ):
            self.startup_scan_timer.start(0)

    def on_database_snapshot_failed(self, message: str) -> None:
        if self.closing:
            return
        logger.warning("后台加载索引统计失败：%s", message)
        self.failed_page.set_snapshot_loading(False)
        self.failed_page.set_status(
            f"未成功索引数据加载失败：{message}。请再次打开此页面重试。"
        )
        self._startup_snapshot_ready = True
        self.update_index_status(self._latest_readiness or None)
        if (
            self._startup_scan_requested
            and not self._startup_scan_consumed
            and not self._startup_scan_in_flight
            and not self.startup_scan_timer.isActive()
        ):
            self.startup_scan_timer.start(0)

    def cleanup_database_snapshot(self) -> None:
        self.database_snapshot_thread = None
        self.database_snapshot_worker = None
        if self._snapshot_refresh_pending and not self.closing:
            self._snapshot_refresh_pending = False
            QTimer.singleShot(0, self.start_database_snapshot)
        if (
            self.pending_monitor_scan
            and self.scan_thread is None
            and self.exclusion_thread is None
            and not self.closing
        ):
            self.monitor_scan_timer.start(0)
        self._finish_close_if_idle()

    def refresh_file_monitor(self) -> None:
        self.file_monitor.stop()
        if (
            self.closing
            or self.database_migration_thread is not None
            or not self.settings.monitor_file_changes
        ):
            return
        roots = [
            Path(str(row["path"]))
            for row in self.db.list_roots(enabled_only=True)
            if Path(str(row["path"])).exists()
        ]
        if roots:
            self.file_monitor.start(roots)

    def on_monitored_file_changed(self, _path: str) -> None:
        if self.closing:
            return
        self.pending_monitor_scan = True
        if self.scan_thread is None and self.exclusion_thread is None:
            self.top_bar.set_index_status("检测到文件变化，后台准备增量更新", is_pending=True)
            self.monitor_scan_timer.start(800)

    def _start_startup_incremental_scan(self) -> None:
        if (
            self.closing
            or self.scan_thread is not None
            or not self._startup_snapshot_ready
            or self._startup_scan_in_flight
            or self._startup_scan_consumed
        ):
            return
        # Consume the one-shot request before any database or scheduling work.
        # A later snapshot callback can therefore never replay the same launch
        # scan even if it arrives while the worker is being created.
        self._startup_scan_requested = False
        self._startup_scan_consumed = True
        if not self.db.list_roots(enabled_only=True):
            return
        self._startup_scan_in_flight = True
        # Changes already observed before this scan are covered by its own
        # discovery/USN pass and must not immediately create a second scan.
        self.pending_monitor_scan = False
        self.start_scan()

    def schedule_startup_incremental_scan(self) -> None:
        """Queue the first background refresh after the window is visible."""

        if (
            self.closing
            or not self.settings.auto_scan_on_start
            or self._startup_scan_in_flight
            or self._startup_scan_consumed
        ):
            return
        self._startup_scan_requested = True
        if (
            self._startup_snapshot_ready
            and self.db.list_roots(enabled_only=True)
            and not self.startup_scan_timer.isActive()
        ):
            self.startup_scan_timer.start(0)

    def _start_pending_monitor_scan(self) -> None:
        if self.closing or not self.pending_monitor_scan:
            return
        if (
            self.scan_thread is not None
            or self.exclusion_thread is not None
            or self.database_snapshot_thread is not None
        ):
            return
        root_ids = (
            None
            if self._pending_scan_all_roots or not self._pending_scan_root_ids
            else tuple(sorted(self._pending_scan_root_ids))
        )
        performance_mode = self._pending_scan_performance
        self._pending_scan_root_ids.clear()
        self._pending_scan_all_roots = False
        self._pending_scan_performance = False
        self.pending_monitor_scan = False
        self.start_scan(root_ids, performance_mode=performance_mode)

    def clear_search_history(self) -> None:
        if self.database_migration_thread is not None:
            self.search_page.set_status("数据库正在迁移，请等待完成")
            return
        self._start_search_history_update(clear=True)
        self.search_page.set_status("正在清理搜索历史...")

    def _start_search_history_update(
        self,
        query_text: str = "",
        *,
        clear: bool = False,
    ) -> None:
        if self.closing or self.search_history_thread is not None:
            return
        thread = QThread()
        worker = SearchHistoryWorker(
            self.db.db_path,
            query_text=query_text,
            clear=clear,
        )
        self.search_history_thread = thread
        self.search_history_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.on_search_history_finished)
        worker.failed.connect(self.on_search_history_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.cleanup_search_history_thread)
        thread.start()

    def on_search_history_finished(self, histories: object) -> None:
        if isinstance(histories, list):
            self.search_page.set_history([str(item) for item in histories])

    def on_search_history_failed(self, message: str) -> None:
        # Search history is ancillary state. It must never make a completed
        # foreground search look failed or show a blocking error dialog.
        logger.warning("Search history update failed: %s", message)

    def cleanup_search_history_thread(self) -> None:
        self.search_history_thread = None
        self.search_history_worker = None
        self._finish_close_if_idle()

    def refresh_failed_page(self) -> None:
        self.request_failed_snapshot_refresh()

    def update_index_status(
        self,
        readiness: dict[str, object] | None = None,
    ) -> None:
        if self.database_migration_thread is not None:
            self.top_bar.set_index_status("正在迁移索引数据库...", is_running=True)
            return
        if self.exclusion_thread is not None:
            self.top_bar.set_index_status("正在更新搜索索引...", is_running=True)
            return
        readiness = dict(readiness or self._latest_readiness)
        if not readiness:
            readiness = self.db.index_readiness(reconcile=False)
            self._latest_readiness = dict(readiness)
        if self.pending_monitor_scan and self.scan_thread is None:
            self.top_bar.set_index_status("检测到文件变化，点击更新", is_pending=True)
            return
        if self.scan_thread is not None:
            # ``background_task_state=converged`` is a database task-row
            # snapshot, not the foreground scan lifecycle.  An active scan
            # must never be shown as converged while its writer/FTS finalizer
            # is still running.
            readiness["scan_active"] = True
            readiness["background_task_state"] = "running"
            readiness["ready"] = False
            readiness["index_complete"] = False
        blocking = int(readiness.get("blocking_files") or 0)
        task_state_active = str(
            readiness.get("background_task_state") or ""
        ) in {"running", "finishing"}
        if bool(readiness.get("ready")):
            self.top_bar.set_index_status(
                format_index_scope_status(readiness)
            )
            return
        self.top_bar.set_index_status(
            format_index_scope_status(readiness),
            is_running=(
                self.scan_thread is not None
                or task_state_active
            ),
            is_error=(
                self.scan_thread is None
                and blocking > 0
                and not task_state_active
            ),
            is_pending=(
                self.scan_thread is None
                and blocking == 0
                and not task_state_active
            ),
        )

    def _resolve_selected_failed_rows(
        self,
        file_ids: list[int],
    ) -> tuple[list[dict[str, object]], list[int]]:
        """Resolve control-DB rows and discarded-candidate report rows.

        A failed candidate is removed after a failed publish attempt, so its
        files are deliberately not present in ``index_control.db``.  The
        failure report is the authoritative source for those visible rows.
        """

        selected_ids = list(dict.fromkeys(int(value) for value in file_ids))
        control_rows = {
            int(row["id"]): dict(row)
            for row in self.db.failed_files(limit=None)
        }
        report = self._latest_snapshot.get("last_failure_report")
        if not isinstance(report, dict):
            report = self.index_generation_store.read_failure_report() or {}
        report_rows = {
            int(row["id"]): dict(row)
            for row in (report.get("blocking_files") or [])
            if isinstance(row, dict) and row.get("id") is not None
        }
        resolved: list[dict[str, object]] = []
        report_only: list[int] = []
        for file_id in selected_ids:
            row = control_rows.get(file_id)
            if row is not None:
                resolved.append(row)
                continue
            row = report_rows.get(file_id)
            if row is not None:
                resolved.append(row)
                report_only.append(file_id)
        return resolved, report_only

    def exclude_failed_files(self, file_ids: list[int]) -> None:
        if self.database_migration_thread is not None:
            self.failed_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请勿重复操作")
            return
        selected, report_only_ids = self._resolve_selected_failed_rows(file_ids)
        if len(selected) != len(set(file_ids)):
            self.failed_page.set_status("所选阻断项已变化，请刷新后重试")
            return
        reason, accepted = QInputDialog.getText(
            self,
            "人工排除原因",
            "排除原因（可留空）：",
        )
        if not accepted:
            return
        reason = reason.strip() or "用户确认当前文件暂时无法解析"
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认从当前索引范围排除")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"将排除 {len(selected)} 个文件。其文件名、路径和正文均不会出现在搜索结果中。"
        )
        dialog.setInformativeText(
            "源文件不会被删除，之后可以在“已人工排除”中恢复纳入。"
            "后台更新期间搜索和索引管理操作会暂时受限。"
        )
        dialog.setDetailedText(
            "\n".join(
                f"{row['path']}\n  {row.get('parse_error_code') or row.get('parse_status')}: "
                f"{row.get('parse_error_message') or ''}"
                for row in selected
            )
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        self.start_scope_exclusion(
            [int(row["id"]) for row in selected],
            reason,
            report_only_file_ids=report_only_ids,
        )

    def start_scope_exclusion(
        self,
        file_ids: list[int],
        reason: str,
        *,
        report_only_file_ids: list[int] | None = None,
    ) -> None:
        self._start_scope_update(
            file_ids,
            reason=reason,
            force_complete=False,
            report_only_file_ids=report_only_file_ids,
        )

    def start_scope_repair(
        self,
        file_ids: list[int],
        reason: str,
        *,
        report_only_file_ids: list[int] | None = None,
    ) -> None:
        self._start_scope_update(
            file_ids,
            reason=reason,
            force_complete=True,
            report_only_file_ids=report_only_file_ids,
        )

    def _start_scope_update(
        self,
        file_ids: list[int],
        *,
        reason: str,
        force_complete: bool,
        report_only_file_ids: list[int] | None = None,
    ) -> None:
        if self.database_migration_thread is not None:
            self.failed_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请勿重复操作")
            return
        if self.search_thread is not None:
            self.failed_page.set_status("搜索正在进行，请等待搜索结束后再更新索引")
            return
        self.pending_search = False
        self.hide_preview()
        self.exclusion_outcome = None
        self.exclusion_progress_payload = {
            "stage": "preparing",
            "phase_label": (
                "正在准备修复索引状态"
                if force_complete
                else "正在准备后台更新"
            ),
            "processed_files": 0,
            "total_files": len(file_ids),
            "large_fts_operation": False,
            "can_cancel": True,
        }
        self.exclusion_started_at = time.monotonic()
        self.exclusion_thread = QThread()
        self.exclusion_worker = ScopeExclusionWorker(
            self.db.db_path,
            file_ids,
            reason=reason,
            operation_source=("ui_force_complete" if force_complete else "ui"),
            force_complete=force_complete,
            generation_base_path=self.generation_base_path,
            report_only_file_ids=report_only_file_ids,
        )
        self.exclusion_worker.moveToThread(self.exclusion_thread)
        self.exclusion_thread.started.connect(self.exclusion_worker.run)
        self.exclusion_worker.progress.connect(self.on_exclusion_progress)
        self.exclusion_worker.finished.connect(self.on_exclusion_finished)
        self.exclusion_worker.cancelled.connect(self.on_exclusion_cancelled)
        self.exclusion_worker.failed.connect(self.on_exclusion_failed)
        self.exclusion_worker.finished.connect(self.exclusion_thread.quit)
        self.exclusion_worker.cancelled.connect(self.exclusion_thread.quit)
        self.exclusion_worker.failed.connect(self.exclusion_thread.quit)
        self.exclusion_thread.finished.connect(self.cleanup_exclusion_thread)
        readiness = self.db.index_readiness(reconcile=False)
        self.search_page.set_index_ready(
            False,
            {**readiness, "index_update_running": True},
        )
        self.failed_page.set_exclusion_running(True)
        self.failed_page.set_exclusion_progress(self.exclusion_progress_payload)
        self.index_page.set_database_update_running(True)
        self.top_bar.set_index_status("正在更新搜索索引...", is_running=True)
        self.exclusion_heartbeat_timer.start()
        self.exclusion_thread.start()

    def on_exclusion_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self.exclusion_progress_payload = dict(payload)
        self.failed_page.set_exclusion_progress(payload)
        phase_label = str(payload.get("phase_label") or "")
        self.index_page.set_database_update_progress(phase_label)
        self.top_bar.set_index_status(
            "正在更新搜索索引..."
            + (f" · {phase_label}" if phase_label else ""),
            is_running=True,
        )

    def on_exclusion_heartbeat(self) -> None:
        if self.exclusion_thread is None:
            self.exclusion_heartbeat_timer.stop()
            return
        payload = dict(self.exclusion_progress_payload)
        payload["elapsed_seconds"] = int(
            max(0.0, time.monotonic() - self.exclusion_started_at)
        )
        self.failed_page.set_exclusion_progress(payload)

    def cancel_scope_exclusion(self) -> None:
        if self.exclusion_worker is None:
            return
        self.failed_page.set_exclusion_cancel_pending()
        self.top_bar.set_index_status("正在取消索引更新并回滚...", is_running=True)
        self.exclusion_worker.cancel()

    def on_exclusion_finished(self, result: object) -> None:
        self.exclusion_outcome = ("finished", result)

    def on_exclusion_cancelled(self) -> None:
        self.exclusion_outcome = ("cancelled", None)

    def on_exclusion_failed(self, message: str) -> None:
        self.exclusion_outcome = ("failed", message)

    def cleanup_exclusion_thread(self) -> None:
        outcome = self.exclusion_outcome or (
            "failed",
            "后台任务异常结束，请查看日志",
        )
        self.exclusion_heartbeat_timer.stop()
        self.exclusion_thread = None
        self.exclusion_worker = None
        self.exclusion_progress_payload = {}
        self.exclusion_outcome = None
        self.failed_page.set_exclusion_running(False)
        self.index_page.set_database_update_running(False)
        if self.closing:
            self._finish_close_if_idle()
            return
        # Scope actions may update the generation manifest and the failure
        # report without changing control-DB file rows.  Invalidate the short
        # UI snapshot cache so the newly resolved report-only item disappears
        # immediately and the active snapshot's partial flag is re-read.
        self._last_snapshot_completed_monotonic = 0.0
        self.refresh_all()
        kind, payload = outcome
        if kind == "finished" and isinstance(payload, dict):
            changed = int(payload.get("excluded_files") or 0)
            remaining = int(payload.get("blocking_files") or 0)
            operation = str(payload.get("operation") or "exclude")
            if bool(payload.get("ready")):
                if operation == "force_complete":
                    self.failed_page.set_status(
                        f"索引状态修复完成，另排除 {changed} 个文件，当前已开放搜索"
                    )
                else:
                    self.failed_page.set_status(
                        f"已人工排除 {changed} 个文件，当前索引已开放搜索"
                    )
                self.switch_page("search")
            else:
                self.failed_page.scope_tabs.setCurrentIndex(0)
                self.failed_page.set_status(
                    f"已人工排除 {changed} 个文件，仍有 {remaining} 个阻断项"
                )
        elif kind == "cancelled":
            self.failed_page.set_status("后台更新已取消，本次变更已回滚")
        else:
            message = str(payload or "未知错误")
            self.failed_page.set_status(f"排除失败，所有变更已回滚：{message}")
            QMessageBox.critical(self, "排除失败", message)
        self._finish_close_if_idle()

    def retry_failed_files(self, file_ids: list[int]) -> None:
        if self.database_migration_thread is not None:
            self.failed_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None or self.scan_thread is not None:
            self.failed_page.set_status("当前索引任务仍在运行，请等待完成或安全暂停")
            return
        if not file_ids:
            self._prompt_select_failed_files()
            return
        queued = self.db.queue_files_for_retry(file_ids)
        if queued <= 0:
            self.failed_page.set_status("所选文件当前不需要重新尝试")
            self.refresh_failed_page()
            return
        self.failed_page.set_status(f"已选择 {queued} 个文件，正在重新尝试")
        self.start_scan(
            self.db.root_ids_for_files(file_ids),
            file_ids=file_ids,
        )

    def restore_excluded_files(self, file_ids: list[int]) -> None:
        if self.database_migration_thread is not None:
            self.failed_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请稍候")
            return
        if self.scan_thread is not None:
            self.failed_page.set_status(
                "索引任务仍在运行，请先暂停并等待安全点"
            )
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("恢复纳入并重试")
        dialog.setIcon(QMessageBox.Icon.Question)
        dialog.setText(
            f"将 {len(set(file_ids))} 个文件恢复到当前索引范围并重新尝试解析。"
        )
        dialog.setInformativeText(
            "恢复后，这些文件会重新成为阻断项，直到解析完成。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        selected_ids = list(dict.fromkeys(int(value) for value in file_ids))
        control_excluded = {
            int(row["file_id"])
            for row in self.db.excluded_files(limit=None)
            if row["file_id"] is not None
            and int(row["file_id"]) in set(selected_ids)
        }
        report = self._latest_snapshot.get("last_failure_report")
        if not isinstance(report, dict):
            report = self.index_generation_store.read_failure_report() or {}
        report_excluded_rows = [
            dict(row)
            for row in (report.get("excluded_files") or [])
            if isinstance(row, dict) and row.get("id") is not None
        ]
        report_excluded = {
            int(row["id"]): row
            for row in report_excluded_rows
            if int(row["id"]) in set(selected_ids)
        }
        if len(control_excluded) + len(report_excluded) != len(set(selected_ids)):
            self.failed_page.set_status("所选排除项已变化，请刷新后重试")
            return
        try:
            changed = self.db.restore_files_to_index(
                sorted(control_excluded),
                reason="用户恢复纳入并重试",
                operation_source="ui",
            ) if control_excluded else 0
            report_changed = (
                self.index_generation_store.restore_failure_report_items(
                    sorted(report_excluded)
                )
                if report_excluded
                else {}
            )
            if control_excluded:
                self.db.queue_files_for_retry(sorted(control_excluded))
        except Exception as exc:
            self.failed_page.set_status(f"恢复失败：{exc}")
            return
        root_ids = set(self.db.root_ids_for_files(sorted(control_excluded)))
        root_ids.update(
            int(row["root_id"])
            for row in report_changed.get("restored_rows", [])
            if isinstance(row, dict) and row.get("root_id") is not None
        )
        self._last_snapshot_completed_monotonic = 0.0
        self.refresh_all()
        self.failed_page.scope_tabs.setCurrentIndex(0)
        restored_count = int(changed) + len(report_excluded)
        self.failed_page.set_status(f"已恢复 {restored_count} 个文件，正在重新尝试")
        if root_ids:
            # Report-only files are absent from the control DB and therefore
            # cannot be passed as file_ids.  Re-scan their owning roots so the
            # planner can discover them again.
            self.start_scan(tuple(sorted(root_ids)))
        else:
            self.failed_page.set_status(
                f"已恢复 {restored_count} 个文件，但所属搜索范围已不存在"
            )

    def request_force_complete(self) -> None:
        if self.database_migration_thread is not None:
            self.failed_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请稍候")
            return
        if self.scan_thread is not None:
            self.failed_page.set_status("索引任务仍在运行，请先等待当前任务结束")
            return
        if self.search_thread is not None:
            self.failed_page.set_status("搜索正在进行，请等待搜索结束后再执行保底完成")
            return
        selected_ids = self.failed_page.selected_file_ids()
        if not selected_ids:
            self._prompt_select_failed_files()
            return
        readiness = dict(self._latest_readiness or self.db.quick_index_readiness())
        if bool(readiness.get("ready")):
            self.refresh_all()
            self.failed_page.set_status("当前索引已经就绪，无需修复")
            return
        blockers, report_only_ids = self._resolve_selected_failed_rows(selected_ids)
        if not blockers:
            self.failed_page.set_status("所选文件当前不是未成功索引项，请刷新后重新选择")
            return
        if report_only_ids and len(report_only_ids) == len(blockers):
            dialog = QMessageBox(self)
            dialog.setWindowTitle("保底完成并开放搜索")
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setText(
                f"所选 {len(blockers)} 个文件已在候选索引中完成终止性跳过。"
                "确认排除后立即开放上一版搜索？"
            )
            dialog.setInformativeText(
                "这些文件不会出现在当前搜索结果中，源文件不会被删除；"
                "之后可以在“已人工排除”中恢复并重新尝试。"
            )
            dialog.setDetailedText(
                "\n".join(
                    f"{row.get('path') or ''}\n  "
                    f"{row.get('parse_error_code') or row.get('parse_status') or ''}: "
                    f"{row.get('parse_error_message') or ''}"
                    for row in blockers
                )
            )
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel
            )
            dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if dialog.exec() != QMessageBox.StandardButton.Yes:
                return
            self.start_scope_repair(
                selected_ids,
                "用户确认终止性跳过并开放搜索",
                report_only_file_ids=report_only_ids,
            )
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("保底完成并开放搜索")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"将先对剩余 {len(blockers)} 个阻断项进行最后一次重试，不会立即排除文件。"
        )
        dialog.setInformativeText(
            "如果仍无法完成，程序会再次列出文件并请你确认排除；源文件不会被删除。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        self.force_complete_after_retry = True
        self.force_complete_file_ids = list(selected_ids)
        retry_ids = [
            int(row["id"])
            for row in blockers
            if int(row["id"]) not in set(report_only_ids)
        ]
        try:
            self.db.queue_files_for_retry(retry_ids)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                raise
            self.failed_page.set_status("数据库正忙，搜索结束后将自动重试保底完成")
            QTimer.singleShot(
                500,
                lambda ids=retry_ids: self._retry_force_complete_queue(ids),
            )
            return
        self.failed_page.set_status("正在执行最后一次恢复重试")
        self.start_scan(
            self.db.root_ids_for_files(retry_ids),
            file_ids=retry_ids,
        )

    def _retry_force_complete_queue(self, file_ids: list[int]) -> None:
        if self.closing:
            return
        if self.search_thread is not None:
            QTimer.singleShot(
                500,
                lambda ids=list(file_ids): self._retry_force_complete_queue(ids),
            )
            return
        try:
            self.db.queue_files_for_retry(file_ids)
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() and "busy" not in str(exc).lower():
                self.failed_page.set_status(f"保底完成排队失败：{exc}")
                return
            QTimer.singleShot(
                500,
                lambda ids=list(file_ids): self._retry_force_complete_queue(ids),
            )
            return
        self.failed_page.set_status("正在执行最后一次恢复重试")
        self.start_scan(
            self.db.root_ids_for_files(file_ids),
            file_ids=file_ids,
        )

    def confirm_force_complete_remaining(self) -> None:
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请稍候")
            return
        selected_ids = list(self.force_complete_file_ids or self.failed_page.selected_file_ids())
        if not selected_ids:
            self._prompt_select_failed_files()
            return
        blockers, report_only_ids = self._resolve_selected_failed_rows(selected_ids)
        readiness = dict(self._latest_readiness or self.db.quick_index_readiness())
        if bool(readiness["ready"]):
            self.force_complete_file_ids = []
            self.refresh_all()
            self.failed_page.set_status("最后一次恢复重试已成功，索引现已完成")
            return
        if not blockers:
            self.start_scope_repair(selected_ids, "最后一次恢复后修复选中项状态")
            return
        if report_only_ids and len(report_only_ids) == len(blockers):
            self.start_scope_repair(
                selected_ids,
                "用户确认终止性跳过后排除选中项并开放搜索",
                report_only_file_ids=report_only_ids,
            )
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认排除并开放搜索")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"最后一次恢复后仍有 {len(blockers)} 个文件无法完成。是否排除这些文件并开放搜索？"
        )
        dialog.setInformativeText(
            "确认排除后将开放其他文件的搜索；源文件不会删除，但这些文件的文件名、路径和正文"
            "不会出现在搜索结果中。可在“已人工排除”中恢复重试。"
        )
        dialog.setDetailedText(
            "\n".join(
                f"{row['path']}\n  {row['parse_status']}: "
                f"{row['parse_error_message'] or row['parse_error_code'] or '任务未完成'}"
                for row in blockers
            )
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            self.failed_page.set_status("已取消保底完成，索引仍保持未完成状态")
            return
        self.start_scope_repair(
            selected_ids,
            "用户确认多次恢复后排除选中项并开放搜索",
            report_only_file_ids=report_only_ids,
        )

    def _prompt_select_failed_files(self) -> None:
        QMessageBox.information(
            self,
            "需要选择未成功索引项",
            "请先勾选需要处理的未成功索引项。",
        )

    def root_stats_by_id(self) -> dict[int, dict[str, int]]:
        data: dict[int, dict[str, int]] = {}
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT root_id, extension, COUNT(*) AS n
                FROM files
                WHERE is_deleted = 0
                GROUP BY root_id, extension
                """
            ).fetchall()
        for row in rows:
            root_id = int(row["root_id"])
            extension = str(row["extension"] or "")
            data.setdefault(root_id, {})[extension] = int(row["n"])
        return data

    def add_root(self) -> None:
        if self.database_migration_thread is not None:
            self.index_page.task_label.setText("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            return
        directory = QFileDialog.getExistingDirectory(self, "选择搜索目录")
        if not directory:
            return
        self.db.add_root(Path(directory))
        self.refresh_all()
        self.switch_page("index")
        self.index_page.task_label.setText("已添加搜索范围，请选择更新方式")

    def remove_root(self, root_id: int) -> None:
        if self.database_migration_thread is not None:
            self.index_page.task_label.setText("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            return
        if self.scan_thread is not None:
            self.index_page.task_label.setText("索引正在运行，请先安全暂停后再删除搜索范围")
            return
        if self.root_removal_thread is not None:
            return
        if QMessageBox.question(self, "删除搜索范围", "只删除索引和配置，不会删除原文件。确认继续？") != QMessageBox.StandardButton.Yes:
            return
        self.file_monitor.stop()
        self.root_removal_thread = QThread()
        self.root_removal_worker = RootRemovalWorker(self.db.db_path, root_id)
        self.root_removal_worker.moveToThread(self.root_removal_thread)
        self.root_removal_thread.started.connect(self.root_removal_worker.run)
        self.root_removal_worker.progress.connect(self.on_root_removal_progress)
        self.root_removal_worker.finished.connect(self.on_root_removal_finished)
        self.root_removal_worker.failed.connect(self.on_root_removal_failed)
        self.root_removal_worker.finished.connect(self.root_removal_thread.quit)
        self.root_removal_worker.failed.connect(self.root_removal_thread.quit)
        self.root_removal_thread.finished.connect(self.cleanup_root_removal)
        self.index_page.set_database_update_running(True)
        self.top_bar.set_index_status("正在删除搜索范围索引...", is_running=True)
        self.root_removal_thread.start()

    def on_root_removal_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            label = str(payload.get("phase_label") or "正在删除搜索范围索引")
            self.index_page.set_database_update_progress(label)
            self.top_bar.set_index_status(label, is_running=True)

    def on_root_removal_finished(self, _payload: object) -> None:
        self.refresh_all()

    def on_root_removal_failed(self, message: str) -> None:
        self.index_page.set_task_running(False)
        self.index_page.task_label.setText("删除搜索范围失败，原索引已保留")
        self.index_page.task_label.setToolTip(str(message))

    def cleanup_root_removal(self) -> None:
        self.root_removal_thread = None
        self.root_removal_worker = None
        self.index_page.set_database_update_running(False)
        self.refresh_all()

    def toggle_root(self, root_id: int, enabled: bool) -> None:
        if self.database_migration_thread is not None:
            self.index_page.task_label.setText("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            return
        self.db.set_root_enabled(root_id, enabled)
        self.refresh_all()

    def start_scan(
        self,
        root_ids: object = None,
        *,
        performance_mode: bool = False,
        file_ids: object = None,
    ) -> None:
        # start_scan() 在界面主线程中只做“任务编排”，不亲自遍历和解析文件。
        # 如果迁移或另一项索引维护正在进行，先阻止创建相互冲突的新任务。
        if self.database_migration_thread is not None:
            self.index_page.task_label.setText("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            self.top_bar.set_index_status("正在更新搜索索引...", is_running=True)
            return
        if root_ids is None:
            selected_root_ids: tuple[int, ...] = ()
        elif isinstance(root_ids, int):
            selected_root_ids = (int(root_ids),)
        else:
            try:
                selected_root_ids = tuple(int(root_id) for root_id in root_ids)  # type: ignore[union-attr]
            except TypeError:
                selected_root_ids = ()
        if self.scan_thread is not None:
            # Manual, startup and monitor refreshes share one scheduler. Merge
            # a request that arrives during the active generation instead of
            # losing it or starting another worker for the same root.
            # 教学提示：已有索引线程时不再创建第二个 ScanWorker，而是合并请求，
            # 等当前任务结束后再检查变化。这可以避免两份候选 generation 同时写入。
            self.pending_monitor_scan = True
            if selected_root_ids:
                self._pending_scan_root_ids.update(selected_root_ids)
            else:
                self._pending_scan_all_roots = True
                self._pending_scan_root_ids.clear()
            self._pending_scan_performance = (
                self._pending_scan_performance or performance_mode
            )
            self.index_page.task_label.setText(
                "更新请求已合并，将在当前任务完成后检查变化"
            )
            self.switch_page("index")
            return
        self.pending_monitor_scan = False
        self._pending_scan_root_ids.clear()
        self._pending_scan_all_roots = False
        self._pending_scan_performance = False
        # 这里的 _scan_generation 只是界面层的“第几次扫描任务”流水号；
        # 它不同于 IndexGenerationStore 中代表数据库版本的 generation。
        self._scan_generation += 1
        self._active_scan_generation = self._scan_generation
        self._scan_progress_text = "正在准备索引"
        readiness = dict(self._latest_readiness)
        if not readiness:
            readiness = self.db.quick_index_readiness()
            self._latest_readiness = dict(readiness)
        search_readiness = self._search_readiness_from_published_snapshot(readiness)
        self.search_page.set_index_ready(
            bool(search_readiness.get("search_available")),
            {
                **search_readiness,
                "active_runs": 1,
            },
        )
        self.hide_preview()
        # 创建独立的索引调度线程，使耗时扫描不会阻塞 Qt 界面事件循环。
        self.scan_thread = QThread()
        # ScanWorker 负责一整轮候选数据库的创建、索引、校验和发布。
        self.scan_worker = ScanWorker(
            self.generation_base_path,
            self.settings,
            control_db_path=self.db.db_path,
            performance_mode=performance_mode,
            root_ids=selected_root_ids,
            file_ids=tuple(int(file_id) for file_id in (file_ids or ())),
        )
        # QObject 默认属于创建它的主线程；moveToThread 后，run() 会在 scan_thread 执行。
        self.scan_worker.moveToThread(self.scan_thread)
        # QThread 发出 started 信号时调用 ScanWorker.run，正式开始后台索引。
        self.scan_thread.started.connect(self.scan_worker.run)
        # 以下信号把后台进度和最终状态安全地送回界面主线程。
        self.scan_worker.progress.connect(self.on_scan_progress)
        self.scan_worker.finished.connect(self.on_scan_finished)
        self.scan_worker.failed.connect(self.on_scan_failed)
        self.scan_worker.cancelled.connect(self.on_scan_cancelled)
        self.scan_worker.ended.connect(self.on_scan_ended)
        # 无论成功、失败还是取消，ScanWorker 都会发 ended，让线程事件循环退出。
        self.scan_worker.ended.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.cleanup_scan_thread)
        self._active_scan_id = self.scan_worker.scan_id
        self._failed_snapshot_sequence += 1
        self.failed_snapshot_timer.start()
        self.index_page.set_performance_mode(performance_mode)
        self.top_bar.set_index_status(self._scan_progress_text, is_running=True)
        self.index_page.set_task_running(True)
        self.failed_page.set_index_running(True)
        # start() 只负责启动线程；真正的索引入口是上面连接的 ScanWorker.run。
        self.scan_thread.start()

    def confirm_performance_scan(self, root_ids: object = None) -> None:
        answer = QMessageBox.question(
            self,
            "启用性能模式",
            performance_mode_notice() + "\n\n确认开始性能模式索引？",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.start_scan(root_ids, performance_mode=True)

    def request_scan_mode_switch(self, performance_mode: bool) -> None:
        if self.scan_worker is None:
            return
        if performance_mode:
            answer = QMessageBox.question(
                self,
                "切换到性能模式",
                performance_mode_notice() + "\n\n确认切换？",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.scan_worker.switch_mode(performance_mode)

    def cancel_scan(self, *, force: bool = False) -> None:
        if self.scan_worker is not None:
            self.scan_worker.cancel(
                force=force,
                reason="应用退出" if self.closing else "界面取消",
            )

    def _refresh_failed_page_during_scan(self) -> None:
        if (
            self.scan_thread is not None
            and self.stack.currentIndex() == PAGE_INDEX["failed"]
        ):
            self.request_failed_snapshot_refresh()

    def refresh_search_availability_during_scan(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self.last_search_readiness_refresh < 1.5:
            return
        self.last_search_readiness_refresh = now
        readiness = dict(self._latest_readiness)
        if not readiness:
            return
        readiness = {
            **readiness,
            "active_runs": max(1, int(readiness.get("active_runs") or 0)),
            "scan_active": True,
            "background_task_state": "running",
            "ready": False,
            "index_complete": False,
            "search_is_partial": bool(readiness.get("search_available")),
        }
        self.search_page.set_stats(
            {
                key: readiness.get(key)
                for key in (
                    "complete_files",
                    "eligible_files",
                    "blocking_files",
                    "video_excluded",
                    "manual_excluded_files",
                )
            },
            has_roots=bool(readiness.get("enabled_roots")),
        )
        search_readiness = self._search_readiness_from_published_snapshot(readiness)
        self.search_page.set_index_ready(
            bool(search_readiness.get("search_available")),
            {
                **search_readiness,
                "active_runs": max(1, int(readiness.get("active_runs") or 0)),
                "scan_active": True,
                "background_task_state": "running",
            },
        )

    def on_scan_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        payload_scan_id = str(payload.get("scan_id") or "")
        if payload_scan_id and payload_scan_id != self._active_scan_id:
            logger.debug(
                "Discarding stale scan progress scan_id=%s active_scan_id=%s stage=%s",
                payload_scan_id,
                self._active_scan_id,
                payload.get("stage"),
            )
            return
        self.refresh_search_availability_during_scan()
        stage = str(payload.get("stage") or "")
        if stage == "starting":
            phase_label = str(payload.get("phase_label") or "正在启动更新…")
            self._scan_progress_text = phase_label
            self.top_bar.set_index_status(phase_label, is_running=True)
            self.index_page.set_scan_starting(phase_label)
            return
        if stage.startswith("database_migration_"):
            phase_label = str(payload.get("phase_label") or "正在升级索引数据库")
            self._scan_progress_text = phase_label
            self.top_bar.set_index_status(phase_label, is_running=True)
            self.index_page.set_scan_starting(phase_label)
            return
        if stage == "writer_commit":
            self.index_page.set_writer_commit_progress(
                int(payload.get("writer_committed_batches") or 0),
                int(payload.get("writer_committed_files") or 0),
                int(payload.get("writer_queue_depth") or 0),
            )
            return
        if stage == "performance_profile":
            profile = payload.get("performance_profile")
            if isinstance(profile, dict):
                self.index_page.set_performance_profile(profile)
                self.top_bar.set_index_status(
                    "正在检查文件变化…", is_running=True
                )
            return
        pause_state = str(payload.get("pause_state") or "")
        if pause_state:
            if "performance_mode" in payload:
                self.index_page.set_performance_mode(
                    bool(payload.get("performance_mode"))
                )
            profile = payload.get("performance_profile")
            if isinstance(profile, dict) and bool(payload.get("performance_mode")):
                self.index_page.set_performance_profile(profile)
            phase_label = str(payload.get("phase_label") or "")
            self.index_page.set_pause_state(pause_state, phase_label)
            self.top_bar.set_index_status(self._scan_progress_text, is_running=True)
            return
        indexed = int(payload.get("indexed") or 0)
        scanned = int(payload.get("scanned") or 0)
        total = int(payload.get("total_files") or scanned)
        completed = int(payload.get("completed_files") or indexed)
        failed = int(payload.get("failed") or 0)
        current = str(payload.get("current_file") or "")
        stage = stage or "indexing"
        phase_label = str(payload.get("phase_label") or "正在索引")
        eta_seconds = int(payload.get("eta_seconds") or 0)
        eta_ready = bool(payload.get("eta_ready"))
        eta_text = format_remaining_single(eta_seconds, eta_ready)
        active_elapsed = int(payload.get("active_elapsed_seconds") or 0)
        if not eta_ready and eta_seconds > 0 and active_elapsed >= 20:
            # The estimator already has a conservative static prior before its
            # first samples. Expose that single-value estimate after a short
            # warm-up instead of leaving the user on "正在估算…" indefinitely.
            eta_text = format_remaining_single(eta_seconds, True)
        active_queue = str(
            payload.get("queue") or payload.get("active_queue") or ""
        )
        active_count = int(payload.get("active_file_count") or 0)
        active_phase = str(payload.get("active_phase") or "")
        active_completed = int(payload.get("active_completed_units") or 0)
        active_total = int(payload.get("active_total_units") or 0)
        planning_phase = str(payload.get("planning_phase") or "")
        planning_completed = int(payload.get("planning_completed") or 0)
        planning_total = int(payload.get("planning_total") or 0)
        discovered_files = int(
            payload.get("discovered_files")
            or payload.get("scanned")
            or 0
        )
        pending_files = int(payload.get("pending_files") or 0)
        processing_files = int(payload.get("processing_files") or 0)
        reused_files = int(payload.get("reused_files") or 0)
        no_progress = int(payload.get("no_progress_seconds") or 0)
        retry_count = int(payload.get("retry_count") or 0)
        excluded_video = int(payload.get("excluded_video") or 0)
        diagnostic_state = str(payload.get("diagnostic_state") or "")
        diagnostic_reason = str(payload.get("diagnostic_reason") or "")
        representative_is_slowest = bool(
            payload.get("representative_is_slowest")
        )
        other_active_lane_count = int(
            payload.get("other_active_lane_count") or 0
        )
        other_recent_progress_seconds = int(
            payload.get("other_recent_progress_seconds") or 0
        )
        dynamic_total = max(total, discovered_files, completed, 1)
        text = f"已完成 {completed:,}/{dynamic_total:,}"
        if failed:
            text += f" · 失败 {failed:,}"
        if excluded_video:
            text += f" · 排除视频 {excluded_video:,}"
        if processing_files:
            text += f" · 处理中 {processing_files:,}"
        if pending_files:
            text += f" · 待处理 {pending_files:,}"
        self._scan_progress_text = text
        self.top_bar.set_index_status(text, is_running=True)
        self.index_page.set_scan_progress(
            completed,
            dynamic_total,
            failed,
            current,
            phase_label=phase_label,
            eta_text=eta_text,
            active_elapsed_seconds=active_elapsed,
            active_queue=active_queue,
            active_file_count=active_count,
            active_phase=active_phase,
            active_completed_units=active_completed,
            active_total_units=active_total,
            planning_phase=planning_phase,
            planning_completed=planning_completed,
            planning_total=planning_total,
            discovered_files=discovered_files,
            reused_files=reused_files,
            no_progress_seconds=no_progress,
            retry_count=retry_count,
            excluded_video=excluded_video,
            diagnostic_state=diagnostic_state,
            diagnostic_reason=diagnostic_reason,
            representative_is_slowest=representative_is_slowest,
            other_active_lane_count=other_active_lane_count,
            other_recent_progress_seconds=other_recent_progress_seconds,
            indeterminate=stage in {"discovering", "planning", "fts"},
        )

    def on_scan_finished(self, summary: object) -> None:
        payload = summary if isinstance(summary, dict) else {"summary": summary}
        # The worker emits this outcome only after candidate and process
        # cleanup, but the QThread itself has not yet emitted finished. Keep
        # controls disabled until cleanup_scan_thread runs.
        if self.closing:
            return
        self._scan_refresh_pending = True
        summary_text, summary_tooltip = format_index_run_summary(payload)
        if summary_text:
            self.index_page.show_run_summary(summary_text, summary_tooltip)
        if self.force_complete_after_retry:
            self.force_complete_after_retry = False
            self.pending_force_complete_confirmation = True

    def on_scan_failed(self, message: str) -> None:
        # Lifecycle/UI reset is handled by cleanup_scan_thread after ended.
        if self.closing:
            return
        lock_failure = (
            "database is locked" in str(message).lower()
            or "database is busy" in str(message).lower()
            or "数据库被其他进程" in str(message)
        )
        self._scan_refresh_pending = not lock_failure
        if lock_failure:
            text = "数据库被其他进程或残留诊断连接占用，索引已停止；关闭占用进程后请重新更新"
            self.top_bar.set_index_status(text, is_error=True)
            self.index_page.task_label.setText(text)
            self.index_page.task_label.setToolTip(str(message))
            return
        if "KeyboardInterrupt" in str(message) or "解析进程被中断" in str(message):
            text = "解析进程被中断，本轮候选索引已回滚；上一版索引仍可搜索"
            self.top_bar.set_index_status(text, is_error=True)
            self.index_page.task_label.setText(text)
            self.index_page.task_label.setToolTip(str(message))
            return
        if str(message).startswith("更新启动失败："):
            self.index_page.show_run_summary(str(message), str(message))
            self.top_bar.set_index_status(str(message), is_error=True)
            return
        if "阻断文件：" in str(message):
            self.top_bar.set_index_status(
                "本轮索引未发布：存在阻断文件，上一轮索引仍可搜索",
                is_error=True,
            )
            self.index_page.task_label.setText(
                "本轮索引未发布，上一轮索引仍可搜索；请查看提示中的阻断文件"
            )
            self.index_page.task_label.setToolTip(str(message))
            return
        self.top_bar.set_index_status("索引已保留进度，等待继续", is_error=True)
        self.index_page.task_label.setText(
            "索引遇到基础设施错误，已保留完成进度；请点击更新继续"
        )
        self.index_page.task_label.setToolTip(str(message))
        if self.force_complete_after_retry:
            self.force_complete_after_retry = False
            self.pending_force_complete_confirmation = True

    def on_scan_cancelled(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._scan_terminal_payload = dict(payload)
        if not self.closing:
            self.top_bar.set_index_status(
                "索引已取消，上一版索引仍可搜索",
                is_error=True,
            )
            self.index_page.task_label.setText("正在完成取消清理…")

    def on_scan_ended(self, payload: object) -> None:
        if isinstance(payload, dict):
            self._scan_terminal_payload = dict(payload)
        else:
            self._scan_terminal_payload = {"status": "failed"}
        # Start a fresh snapshot only after the worker has persisted the
        # report and removed/released its candidate resources.
        self._scan_refresh_pending = True

    def cleanup_scan_thread(self) -> None:
        # The one-shot guard is cleared when the worker exits; no second scan
        # is scheduled from this cleanup path.
        self._startup_scan_in_flight = False
        self._active_scan_generation = 0
        self._active_scan_id = ""
        self.scan_thread = None
        self.scan_worker = None
        self.failed_snapshot_timer.stop()
        self.index_page.set_task_running(False)
        self.failed_page.set_index_running(False)
        if self._scan_refresh_pending and not self.closing:
            self._scan_refresh_pending = False
            self.start_database_snapshot()
        else:
            self.update_index_status()
        if self.pending_force_complete_confirmation and not self.closing:
            self.pending_force_complete_confirmation = False
            QTimer.singleShot(0, self.confirm_force_complete_remaining)
        self._finish_close_if_idle()

    def request_search(self) -> None:
        # request_search() 是界面层入口：检查能否搜索并安排任务，不执行 SQL。
        # 第一层业务门禁：数据库迁移期间文件位置和状态可能变化，不能启动搜索。
        if self.database_migration_thread is not None:
            self.search_page.set_status("数据库正在迁移，请等待完成")
            return
        # 范围变更写入索引/控制数据库；已发布快照在此期间仍保持只读可用。
        readiness = self.search_page.index_readiness()
        # 第二层业务门禁：只有完整发布的generation才能对用户提供一致结果。
        if self.index_generation_store.active() is None:
            # 没有完整发布的 generation 时，候选 building 数据库不能拿来搜索。
            self.search_page.show_index_not_ready_state()
            return
        if not self.search_page.text():
            # 空关键词不是搜索任务，直接恢复第一页和空闲展示。
            self.page = 1
            self.total_confirmed = 0
            self.search_page.show_idle_state()
            self.hide_preview()
            return
        self.page = 1
        if self.search_thread is not None:
            # 用户在旧搜索尚未结束时再次搜索：先取消旧任务，cleanup 后会启动新任务。
            self.pending_search = True
            self.cancel_search()
            return
        self._run_search()

    def _run_search(self) -> None:
        # _run_search()负责把“页面上的当前状态”编排成“一次后台搜索任务”。
        # 它不写SQL，也不实现关键词匹配和排序。
        if not self.search_page.text():
            return
        # SearchQuery 是本次搜索条件的“快照式申请单”。创建后，后台线程只依赖
        # 这组固定下来的值，不再反复读取可能被用户修改的界面控件。
        query = SearchQuery(
            text=self.search_page.text(),
            mode=self.search_page.mode_value(),
            root_ids=self.search_page.root_ids(),
            extensions=self.search_page.extensions(),
            search_filename=self.search_page.search_filename(),
            search_path=self.search_page.search_path(),
            search_content=self.search_page.search_content(),
            include_ocr=self.search_page.include_ocr(),
            include_ocr_fuzzy=self.search_page.include_ocr_fuzzy(),
            ocr_min_confidence=self.settings.ocr_min_confidence,
            case_sensitive=self.search_page.case_sensitive(),
            page_size=min(max(self.settings.page_size, 10), 100),
            max_results=self.settings.max_results,
            page=self.page,
            sort_by=self.search_page.sort_value(),
        )
        # 搜索使用独立 QThread，因此 SQLite 查询和结果确认不会卡住界面。
        self.search_thread = QThread()
        # SearchWorker 负责线程适配、generation 选择、取消和错误信号。
        # Worker必须保存为MainWindow字段而不是临时局部变量：任务运行期间还要
        # 接收它的信号、响应停止按钮，并保证Qt对象不会过早失去引用。
        self.search_worker = SearchWorker(self.generation_base_path, query)
        # 将 Worker 的线程归属从界面主线程迁移到搜索线程。
        self.search_worker.moveToThread(self.search_thread)
        # 线程启动后调用 run()；run() 内部再创建 DatabaseManager 和 SearchEngine。
        self.search_thread.started.connect(self.search_worker.run)
        # 搜索结果、部分结果和进度通过信号回到 MainWindow，由主线程更新界面。
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.partial.connect(self.on_search_partial)
        self.search_worker.progress.connect(self.on_search_progress)
        self.search_worker.cancelled.connect(self.on_search_cancelled)
        self.search_worker.failed.connect(self.on_search_failed)
        self.search_worker.lease_status.connect(self.on_search_lease_status)
        self.search_worker.quiesce_timeout.connect(self.on_search_quiesce_timeout)
        self.search_worker.timed_out.connect(self.on_search_timed_out)
        # 所有终止路径都必须让 QThread 退出，防止后台线程残留。
        self.search_worker.finished.connect(self.search_thread.quit)
        self.search_worker.cancelled.connect(self.search_thread.quit)
        self.search_worker.failed.connect(self.search_thread.quit)
        self.search_worker.quiesce_timeout.connect(self.search_thread.quit)
        self.search_worker.timed_out.connect(self.search_thread.quit)
        # 线程完全结束后清理引用；如果 pending_search=True，会继续下一次搜索。
        self.search_thread.finished.connect(self.cleanup_search_thread)
        self.search_page.set_running(True)
        self.failed_page.set_search_running(True)
        # 从这里开始后台执行；本函数随后返回，Qt 主线程继续响应用户操作。
        self.search_thread.start()

    def cancel_search(self) -> None:
        # MainWindow只转达取消意图；具体怎样设置CancelToken和中断SQLite连接，
        # 由SearchWorker.cancel()负责。
        if self.search_worker is not None:
            self.search_worker.cancel()

    def on_search_lease_status(self, status: str) -> None:
        if self.closing:
            return
        state = str(status or "")
        if state == "waiting":
            self.search_priority_active = False
            self.search_page.set_status(
                "搜索已排队，正在等待索引完成"
            )
            self.top_bar.set_index_status(
                "搜索已排队，索引完成后自动开始",
                is_running=True,
            )
        elif state == "waiting_for_index":
            self.search_priority_active = False
            self.search_page.set_status("搜索已排队，正在等待索引完成")
        elif state == "snapshot_active":
            self.search_priority_active = False
            self.search_page.set_status(
                "正在使用上一轮完整索引搜索，后台增量索引继续进行"
            )
            self.top_bar.set_index_status(
                "正在使用上一轮索引搜索，后台索引继续进行",
                is_running=True,
            )
        elif state == "active":
            self.search_priority_active = True
            self.search_page.set_status(
                "搜索进行中，索引写入已暂缓"
            )
            self.top_bar.set_index_status(
                "搜索进行中，索引写入已暂缓",
                is_running=True,
            )
        elif state == "released":
            self.search_priority_active = False
            if self.scan_thread is None:
                self.update_index_status()

    def on_search_finished(self, page: object) -> None:
        # 该槽函数由 SearchWorker.finished 触发。Qt 跨线程连接会让界面更新
        # 回到 MainWindow 所属的主线程执行，而不是在搜索线程直接操作控件。
        self.total_confirmed = page.available_results
        # MainWindow不自己绘制结果，只把后台返回的数据交给SearchPage展示。
        self.search_page.set_results(page)
        if self.settings.save_search_history and self.search_worker is not None:
            self._start_search_history_update(self.search_worker.query.text)
        if page.total_confirmed <= 0:
            self.hide_preview()

    def on_search_partial(self, page: object) -> None:
        # 文件名/路径结果可能先于正文搜索返回；仍由MainWindow把跨线程信号
        # 转交给页面，Worker不直接操作Qt界面控件。
        if self.search_thread is None or page is None:
            return
        self.search_page.set_partial_results(page)

    def on_search_progress(self, payload: object) -> None:
        self.search_page.set_search_progress(payload)

    def on_search_cancelled(self) -> None:
        self.search_page.clear_timing()
        if self.pending_search:
            self.search_page.set_status("正在准备新搜索...")
        elif self.search_page.partial_results_visible:
            self.search_page.set_status("搜索已停止，当前显示部分结果")
        else:
            self.search_page.set_status("搜索已取消")
            self.search_page.show_interrupted_state("搜索已取消")

    def on_search_quiesce_timeout(self, message: str) -> None:
        self.search_page.clear_timing()
        self.search_page.set_status(f"索引安全暂停超时：{message}")
        self.search_page.show_interrupted_state("搜索尚未开始")

    def on_search_timed_out(self, message: str) -> None:
        self.search_page.clear_timing()
        self.search_page.set_status(f"搜索达到时间上限：{message}")
        self.search_page.show_interrupted_state("搜索超时，结果未完成")

    def on_search_failed(self, message: str) -> None:
        self.search_page.clear_timing()
        self.search_page.set_status(f"搜索失败：{message}")
        self.search_page.show_interrupted_state("搜索未完成")

    def cleanup_search_thread(self) -> None:
        # thread.finished说明后台线程已经真正退出。此时才能清除运行状态和引用，
        # 不能在刚调用cancel()时就假设任务已经结束。
        self.search_page.set_running(False)
        self.failed_page.set_search_running(False)
        self.search_thread = None
        self.search_worker = None
        if self.pending_search and not self.closing:
            # 旧任务清理完成后，从页面“此刻”的最新条件创建下一份SearchQuery。
            self.pending_search = False
            self._run_search()
        else:
            self.pending_search = False
        self._finish_close_if_idle()

    def previous_page(self) -> None:
        if self.page <= 1:
            return
        self.page -= 1
        self._run_search()

    def next_page_search(self) -> None:
        if self.page * min(max(self.settings.page_size, 10), 100) >= self.total_confirmed:
            return
        self.page += 1
        self._run_search()

    def show_preview(self, result: object) -> None:
        if result is None:
            return
        self.preview_panel.show_result(result, self.search_page.text())
        if not self.preview_panel.isVisible():
            self.preview_panel.setVisible(True)
            self.search_splitter.setSizes([860, 420])

    def hide_preview(self) -> None:
        self.preview_panel.setVisible(False)
        self.preview_panel.show_result(None)
        self.search_splitter.setSizes([1, 0])

    def open_path(self, path: str) -> None:
        try:
            open_file(path)
        except Exception as exc:
            self.search_page.set_status(f"打开失败：{exc}")

    def open_folder_path(self, path: str) -> None:
        try:
            open_parent_folder(path)
        except Exception as exc:
            self.search_page.set_status(f"打开文件夹失败：{exc}")

    def open_location(self, path: str, hit: object) -> None:
        try:
            message = open_file_at_location(path, hit)
            self.search_page.set_status(message)
        except Exception as exc:
            self.search_page.set_status(f"打开命中位置失败：{exc}")

    def reindex_file(self, path: str) -> None:
        if self.database_migration_thread is not None:
            self.search_page.set_status("数据库正在迁移，请等待完成")
            return
        if self.exclusion_thread is not None:
            self.search_page.set_status("索引正在更新，请稍候")
            return
        self.db.invalidate_file(path)
        self.start_scan()

    def export_failed_rows(
        self,
        scope: str = "blocking",
        status_filter: str = "",
        extension_filter: str = "",
    ) -> None:
        scope = scope if scope in {"blocking", "excluded", "metadata"} else "blocking"
        if scope == "excluded":
            source_rows = self.db.excluded_files(limit=None)
        elif scope == "metadata":
            source_rows = self.db.metadata_only_files(limit=None)
        else:
            source_rows = self.db.failed_files(limit=None)
        rows = [FailedPage._normalize_row(row, scope) for row in source_rows]
        if status_filter:
            rows = [row for row in rows if str(row.get("parse_status") or "") == status_filter]
        if extension_filter:
            rows = [row for row in rows if str(row.get("extension") or "") == extension_filter]
        if not rows:
            self.failed_page.set_status("当前没有可导出的记录")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出失败清单",
            str(Path.home() / "Desktop" / "失败文件清单.csv"),
            "CSV 文件 (*.csv)",
        )
        if not target:
            return
        with Path(target).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "路径",
                    "扩展名",
                    "范围状态",
                    "状态",
                    "错误码",
                    "任务阶段",
                    "安全游标",
                    "原因",
                    "恢复建议",
                    "解析器",
                    "成员级诊断(JSON)",
                    "人工排除原因",
                    "时间",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        str(row["path"]),
                        str(row["extension"] or ""),
                        str(row.get("scope_category") or ""),
                        str(row["parse_status"]),
                        str(row["parse_error_code"] or ""),
                        str(row["progress_phase"] or ""),
                        str(row["progress_cursor"] or ""),
                        str(row["parse_error_message"] or ""),
                        str(row["recovery_advice"] or ""),
                        str(row["parser_name"] or ""),
                        str(row.get("parse_diagnostics_json") or ""),
                        str(row.get("reason") or ""),
                        str(row["indexed_at"] or ""),
                    ]
                )
        self.failed_page.set_status(f"已导出全部 {len(rows):,} 条明细：{target}")

    def save_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings_service.save(settings)
        self.search_page.apply_settings(settings)
        self.refresh_file_monitor()
        self.search_page.set_status("设置已保存")

    def request_database_migration(self, target_directory: str) -> None:
        if self.database_migration_thread is not None:
            return
        if (
            self.scan_thread is not None
            or self.search_thread is not None
            or self.exclusion_thread is not None
        ):
            QMessageBox.information(
                self,
                "暂时不能迁移",
                "请先停止当前搜索、索引或失败文件处理任务，等待任务完全结束后再迁移数据库。",
            )
            return
        target = Path(target_directory).expanduser().resolve()
        source = self.generation_base_path.parent.resolve()
        if target == source:
            QMessageBox.information(self, "无需迁移", "目标目录与当前索引存储目录相同。")
            return
        source_size = sum(
            path.stat().st_size
            for path in source.rglob("*")
            if path.is_file()
        )
        target_path = target
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认迁移索引数据库")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"将索引存储目录迁移到：\n{target_path}"
        )
        dialog.setInformativeText(
            f"当前索引存储目录：{source}\n"
            f"索引存储大小：{format_bytes(source_size)}\n\n"
            "迁移期间搜索和索引将暂停。目标数据库完成完整性校验后才会切换；"
            "原数据库会改名为带时间戳的回滚备份。旧版本程序不得与新版本同时运行。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return

        self.file_monitor.stop()
        self.hide_preview()
        self.pending_search = False
        self.database_migration_outcome = None
        self.database_migration_thread = QThread()
        self.database_migration_worker = DatabaseMigrationWorker(
            source,
            target,
            self.database_location_service,
        )
        self.database_migration_worker.moveToThread(
            self.database_migration_thread
        )
        self.database_migration_thread.started.connect(
            self.database_migration_worker.run
        )
        self.database_migration_worker.progress.connect(
            self.on_database_migration_progress
        )
        self.database_migration_worker.finished.connect(
            self.on_database_migration_finished
        )
        self.database_migration_worker.cancelled.connect(
            self.on_database_migration_cancelled
        )
        self.database_migration_worker.failed.connect(
            self.on_database_migration_failed
        )
        self.database_migration_worker.finished.connect(
            self.database_migration_thread.quit
        )
        self.database_migration_worker.cancelled.connect(
            self.database_migration_thread.quit
        )
        self.database_migration_worker.failed.connect(
            self.database_migration_thread.quit
        )
        self.database_migration_thread.finished.connect(
            self.cleanup_database_migration_thread
        )
        self.settings_page.set_database_migration_running(True)
        self.failed_page.set_database_migration_running(True)
        self.search_page.set_index_ready(
            False,
            {
                **self.db.index_readiness(reconcile=False),
                "index_update_running": True,
                "ready": False,
            },
        )
        self.top_bar.set_index_status("正在迁移索引数据库...", is_running=True)
        self.database_migration_thread.start()

    def on_database_migration_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        self.settings_page.set_database_migration_progress(payload)
        label = str(payload.get("phase_label") or "正在迁移索引数据库")
        self.top_bar.set_index_status(label, is_running=True)

    def on_database_migration_finished(self, payload: object) -> None:
        self.database_migration_outcome = ("finished", payload)

    def on_database_migration_cancelled(self) -> None:
        self.database_migration_outcome = ("cancelled", None)

    def on_database_migration_failed(self, message: str) -> None:
        self.database_migration_outcome = ("failed", message)

    def cancel_database_migration(self) -> None:
        if self.database_migration_worker is not None:
            self.settings_page.set_database_migration_cancel_pending()
            self.database_migration_worker.cancel()

    def cleanup_database_migration_thread(self) -> None:
        outcome = self.database_migration_outcome or (
            "failed",
            "数据库迁移任务异常结束，请查看日志",
        )
        self.database_migration_thread = None
        self.database_migration_worker = None
        self.database_migration_outcome = None
        self.settings_page.set_database_migration_running(False)
        self.failed_page.set_database_migration_running(False)
        if self.closing:
            self._finish_close_if_idle()
            return

        kind, payload = outcome
        if kind == "finished" and isinstance(payload, dict):
            source_path = self.generation_base_path.parent.resolve()
            try:
                new_root = Path(
                    str(payload.get("index_store_root") or payload["database_path"])
                ).resolve()
                new_path = new_root / "index_control.db"
                new_database = DatabaseManager(new_path)
                new_database.initialize()
                self.db = new_database
                self.generation_base_path = new_root / "search_index.db"
                self.index_generation_store = IndexGenerationStore(
                    self.generation_base_path
                )
                self.settings_page.set_database_info(self.generation_base_path)
                self.settings_page.set_database_migration_completed(payload)
                self.refresh_all()
            except Exception as exc:
                backup_path = Path(
                    str(payload.get("previous_database_backup") or "")
                )
                try:
                    if backup_path.is_dir() and not source_path.exists():
                        os.replace(backup_path, source_path)
                    self.database_location_service.restore_verified(source_path)
                    self.db = DatabaseManager(source_path)
                    self.db.initialize()
                except Exception:
                    pass
                self.settings_page.set_database_migration_failed(str(exc))
                QMessageBox.critical(
                    self,
                    "数据库重新打开失败",
                    "迁移文件已经生成，但程序未能重新打开目标数据库。"
                    f"\n\n{exc}\n\n程序已尝试恢复迁移前数据库，请保留迁移日志。",
                )
                self.settings_page.set_database_info(self.db.db_path)
                self.refresh_all()
                self.refresh_file_monitor()
                return
            QMessageBox.information(
                self,
                "数据库迁移完成",
                f"当前数据库：{self.db.db_path}\n"
                f"大小：{format_bytes(int(payload.get('database_size_bytes') or 0))}\n"
                "完整性和关键记录数量校验已通过，现有索引可以继续使用。",
            )
        elif kind == "cancelled":
            self.settings_page.set_database_migration_cancelled()
            self.refresh_all()
        else:
            message = str(payload or "未知错误")
            self.settings_page.set_database_migration_failed(message)
            self.refresh_all()
            QMessageBox.critical(
                self,
                "数据库迁移失败",
                message + "\n\n程序继续使用迁移前数据库。",
            )
        self.refresh_file_monitor()
        self._finish_close_if_idle()

    def closeEvent(self, event: QCloseEvent) -> None:
        _log_ui_lifecycle_event(
            "app.window.close_requested",
            scan_id=self._active_scan_id or None,
            closing=self.closing,
            active_threads={
                "scan": self.scan_thread is not None,
                "search": self.search_thread is not None,
                "search_history": self.search_history_thread is not None,
                "exclusion": self.exclusion_thread is not None,
                "database_migration": self.database_migration_thread is not None,
                "database_snapshot": self.database_snapshot_thread is not None,
                "root_removal": self.root_removal_thread is not None,
            },
        )
        get_search_index_coordinator().begin_stopping()
        self.file_monitor.stop()
        if (
            self.scan_thread is None
            and self.search_thread is None
            and self.search_history_thread is None
            and self.exclusion_thread is None
            and self.database_migration_thread is None
            and self.database_snapshot_thread is None
            and self.root_removal_thread is None
        ):
            _log_ui_lifecycle_event("app.window.close_completed", mode="immediate")
            event.accept()
            return
        if self.closing:
            _log_ui_lifecycle_event("app.window.close_ignored", reason="already_closing")
            event.ignore()
            return
        self.closing = True
        _log_ui_lifecycle_event(
            "app.shutdown_started",
            scan_id=self._active_scan_id or None,
            cancellation_source="应用退出",
        )
        self.pending_search = False
        self.pending_monitor_scan = False
        self.cancel_scan(force=True)
        self.cancel_search()
        self.cancel_scope_exclusion()
        self.cancel_database_migration()
        if self.database_snapshot_worker is not None:
            self.database_snapshot_worker.cancel()
        self.hide()
        self.force_close_timer.start()
        event.ignore()

    def _finish_close_if_idle(self) -> None:
        if (
            self.closing
            and not self._interrupt_shutdown_in_progress
            and self.scan_thread is None
            and self.search_thread is None
            and self.search_history_thread is None
            and self.exclusion_thread is None
            and self.database_migration_thread is None
            and self.database_snapshot_thread is None
            and self.root_removal_thread is None
        ):
            _log_ui_lifecycle_event(
                "app.shutdown_workers_idle",
                scan_id=self._active_scan_id or None,
            )
            self._force_exit("workers_idle")

    def shutdown_for_interrupt(self, timeout_ms: int = 3000) -> None:
        """Synchronously stop background work after a console interrupt."""

        _log_ui_lifecycle_event(
            "app.shutdown_interrupt_started",
            scan_id=self._active_scan_id or None,
            timeout_ms=int(timeout_ms),
            cancellation_source="应用退出",
        )
        get_search_index_coordinator().begin_stopping()
        self.closing = True
        self._interrupt_shutdown_in_progress = True
        self.pending_search = False
        self.pending_monitor_scan = False
        self.file_monitor.stop()
        self.cancel_scan(force=True)
        self.cancel_search()
        self.cancel_scope_exclusion()
        self.cancel_database_migration()
        if self.database_snapshot_worker is not None:
            self.database_snapshot_worker.cancel()
        self.db.interrupt_active_connections()
        deadline = time.monotonic() + max(0.1, timeout_ms / 1000.0)
        threads = (
            self.scan_thread,
            self.search_thread,
            self.search_history_thread,
            self.exclusion_thread,
            self.database_migration_thread,
            self.database_snapshot_thread,
            self.root_removal_thread,
        )
        for thread in threads:
            if thread is None:
                continue
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            if remaining <= 0:
                break
            thread.quit()
            thread.wait(remaining)
        self._interrupt_shutdown_in_progress = False
        _log_ui_lifecycle_event(
            "app.shutdown_interrupt_completed",
            scan_id=self._active_scan_id or None,
        )

    def _force_exit(self, reason: str = "unspecified") -> None:
        if self.closing:
            _log_ui_lifecycle_event(
                "app.process.force_exit",
                scan_id=self._active_scan_id or None,
                reason=str(reason),
            )
            os._exit(0)


class TopBar(QFrame):
    index_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TopBar")
        self.title = QLabel("本地全文搜索")
        self.title.setObjectName("TopTitle")
        self.subtitle = QLabel("搜索文件名、正文、表格、幻灯片和图片文字")
        self.subtitle.setObjectName("TopSubtitle")
        title_layout = QVBoxLayout()
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(2)
        title_layout.addWidget(self.title)
        title_layout.addWidget(self.subtitle)

        self.index_button = QPushButton("已索引 0 个文件")
        self.index_button.setObjectName("IndexStatusButton")
        self.index_button.setToolTip("打开索引管理")
        self.index_button.clicked.connect(self.index_requested.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.addLayout(title_layout)
        layout.addStretch(1)
        layout.addWidget(self.index_button)

    def set_title(self, title: str, subtitle: str) -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)

    def set_index_status(
        self,
        text: str,
        *,
        is_running: bool = False,
        is_error: bool = False,
        is_pending: bool = False,
    ) -> None:
        self.index_button.setText(text)
        state = "error" if is_error else "running" if is_running else "pending" if is_pending else "ready"
        self.index_button.setProperty("state", state)
        self.index_button.style().unpolish(self.index_button)
        self.index_button.style().polish(self.index_button)


class Sidebar(QFrame):
    nav_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(184)
        self.buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(6)
        self._add_button(layout, "搜索", "search")
        self._add_button(layout, "索引管理", "index")
        self._add_button(layout, "未成功索引", "failed")
        layout.addStretch(1)
        layout.addWidget(separator())
        self._add_button(layout, "设置", "settings")
        self.set_active("search")

    def _add_button(self, layout: QVBoxLayout, text: str, key: str) -> None:
        button = QPushButton(text)
        button.setObjectName("NavButton")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda _checked=False, value=key: self.nav_requested.emit(value))
        layout.addWidget(button)
        self.buttons[key] = button

    def set_active(self, key: str) -> None:
        for button_key, button in self.buttons.items():
            active = button_key == key
            button.setProperty("active", active)
            button.style().unpolish(button)
            button.style().polish(button)

    def set_failed_alert(self, count: int) -> None:
        button = self.buttons.get("failed")
        if button is None:
            return
        count = max(0, int(count))
        button.setText(f"未成功索引  ·  {count:,}" if count else "未成功索引")
        button.setProperty("alert", bool(count))
        button.setToolTip(
            f"有 {count:,} 个文件未成功索引，打开查看处理方式"
            if count
            else "查看未成功索引"
        )
        button.style().unpolish(button)
        button.style().polish(button)


class SearchPage(QWidget):
    """完整的搜索功能页面，负责聚合输入、筛选、状态和结果展示。

    SearchBox 只是一个输入组件；SearchPage 才是页面级边界。它把回车、
    自动搜索、筛选变化和翻页等不同控件事件，整理成供 MainWindow 监听的
    统一业务信号，同时把页面上的各项条件通过 getter 方法对外提供。
    """

    # 页面级输出信号。MainWindow 只依赖这些公开信号，不需要了解页面内部
    # 使用了哪些输入框、下拉框、复选框和结果控件。
    search_requested = Signal()
    stop_requested = Signal()
    add_root_requested = Signal()
    previous_page_requested = Signal()
    next_page_requested = Signal()
    open_requested = Signal(str)
    open_folder_requested = Signal(str)
    open_location_requested = Signal(str, object)
    reindex_requested = Signal(str)
    result_selected = Signal(object)
    clear_history_requested = Signal()

    def __init__(self, settings: AppSettings) -> None:
        # MainWindow._build_shell() 执行 SearchPage(self.settings) 时进入这里。
        super().__init__()
        self.setObjectName("Page")
        # 保存页面展示需要的索引统计和就绪状态；这些状态不执行搜索算法。
        self._stats: dict[str, int] = {}
        self._has_roots = False
        self._index_ready = False
        self._readiness: dict[str, object] = {}
        # debounce 是“防抖计时器”：用户连续输入时不断重新计时，停止输入一段
        # 时间后只发出一次搜索请求，避免每敲一个字符都启动后台搜索。
        self.debounce = QTimer(self)
        # SingleShot=True 表示每次启动后只触发一次 timeout，不会周期性重复搜索。
        self.debounce.setSingleShot(True)
        # 计时结束后统一发出页面级 search_requested，MainWindow 无需认识计时器。
        self.debounce.timeout.connect(self.search_requested.emit)
        self.auto_search_enabled = False
        self._history: list[str] = []
        self.partial_results_visible = False
        self._estimator = SearchTimeEstimator()
        self._active_estimate_context: SearchEstimateContext | None = None
        # _build() 负责创建 SearchPage 内部的 SearchBox、筛选器和结果列表。
        self._build()
        # 所有控件创建完成后，再把应用设置同步到这些控件。
        self.apply_settings(settings)

    def _build(self) -> None:
        # 创建搜索页面的纵向布局；后面创建的 SearchBox 会加入这个布局。
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        # 这里才真正创建 SearchBox，并立即进入 SearchBox.__init__()。
        # __init__() 返回后，self.search_box 才引用一个初始化完成的搜索框对象。
        self.search_box = SearchBox()
        # 用户每输入或删除一个字符，QLineEdit.textChanged 都会调用
        # SearchPage._on_text_changed()，用于更新按钮和自动搜索计时器。
        self.search_box.input.textChanged.connect(self._on_text_changed)
        # SearchBox 只知道“用户按了回车”；SearchPage 在这里把这个内部信号
        # 原样转发为自己的 search_requested，供 MainWindow 统一监听。
        self.search_box.search_requested.connect(self.search_requested.emit)
        # 停止信号采用同样的转发方式，SearchBox 不直接接触后台线程。
        self.search_box.stop_requested.connect(self.stop_requested.emit)
        # 历史按钮由 SearchPage 自己处理，所以直接连接到 _show_history()。
        self.search_box.history_requested.connect(self._show_history)
        # 把已经初始化完成的 SearchBox 放到搜索页面最上方。
        layout.addWidget(self.search_box)

        # 下面开始创建基础筛选区。SearchPage 拥有这些控件，所以它能够把一组
        # 分散的界面值整理成 MainWindow 构造 SearchQuery 所需的统一读取接口。
        filters = QHBoxLayout()
        filters.setSpacing(8)
        # mode_combo 保存关键词匹配方式；显示文字和内部值分开存储。
        self.mode_combo = chip_combo()
        self.mode_combo.addItem("完全包含", "exact")
        self.mode_combo.addItem("完整短语", "phrase")
        self.mode_combo.addItem("全部关键词", "all")
        self.mode_combo.addItem("任意关键词", "any")
        self.mode_combo.addItem("正则表达式", "regex")
        # file_type_combo 保存文件类型分组，extensions() 会把分组转换成扩展名列表。
        self.file_type_combo = chip_combo()
        for label in ("全部", "PDF", "Word", "Excel", "PowerPoint", "图片", "文本/日志", "压缩包"):
            self.file_type_combo.addItem("全部格式" if label == "全部" else label, label)
        # scope_combo 保存搜索范围；None 代表全部根目录，整数代表一个 root_id。
        self.scope_combo = chip_combo()
        self.scope_combo.addItem("全部范围", None)
        # sort_combo 只决定结果排序方式，不会修改索引数据。
        self.sort_combo = chip_combo()
        for label, value in (
            ("相关性", "relevance"),
            ("修改时间（最新）", "modified_desc"),
            ("修改时间（最早）", "modified_asc"),
            ("文件大小（从大到小）", "size_desc"),
            ("文件大小（从小到大）", "size_asc"),
            ("文件名（升序）", "filename_asc"),
            ("文件名（降序）", "filename_desc"),
        ):
            self.sort_combo.addItem(label, value)
        self.sort_combo.setToolTip("仅对当前搜索结果排序，不改变索引内容")
        self.more_button = QPushButton("筛选")
        self.more_button.setObjectName("FilterButton")
        self.more_button.setCheckable(True)
        self.more_button.clicked.connect(self._toggle_advanced)
        filters.addWidget(self.mode_combo)
        filters.addWidget(self.file_type_combo)
        filters.addWidget(self.scope_combo)
        filters.addWidget(self.more_button)
        filters.addStretch(1)
        layout.addLayout(filters)

        # 高级筛选单独封装成 AdvancedFilters，默认折叠，需要时再展开。
        self.advanced = AdvancedFilters()
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        self.status_label = QLabel("输入关键词开始搜索")
        self.status_label.setObjectName("InlineStatus")
        self.timing_label = QLabel("")
        self.timing_label.setObjectName("TimingLabel")
        self.timing_label.setVisible(False)
        status_row = QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(QLabel("排序"))
        status_row.addWidget(self.sort_combo)
        status_row.addWidget(self.timing_label)
        layout.addLayout(status_row)

        self.search_progress = QProgressBar()
        self.search_progress.setTextVisible(False)
        self.search_progress.setFixedHeight(4)
        self.search_progress.setVisible(False)
        layout.addWidget(self.search_progress)

        # content_stack 在“空状态”和“结果列表”之间切换，同一时刻只显示一个。
        self.content_stack = QStackedWidget()
        self.empty_state = EmptyState()
        self.result_view = ResultView()
        # ResultView 内部产生的打开文件、定位和重新索引事件，也由 SearchPage
        # 转发成页面级信号，保持 MainWindow 不依赖 ResultView 的内部结构。
        self.result_view.open_requested.connect(self.open_requested.emit)
        self.result_view.open_folder_requested.connect(self.open_folder_requested.emit)
        self.result_view.open_location_requested.connect(self.open_location_requested.emit)
        self.result_view.reindex_requested.connect(self.reindex_requested.emit)
        self.result_view.selected_result_changed.connect(self.result_selected.emit)
        self.content_stack.addWidget(self.empty_state)
        self.content_stack.addWidget(self.result_view)
        layout.addWidget(self.content_stack, 1)

        # 翻页控件只发出上一页/下一页意图；真正页码和重新搜索由 MainWindow 管理。
        self.pager = QFrame()
        self.pager.setObjectName("Pager")
        pager_layout = QHBoxLayout(self.pager)
        pager_layout.setContentsMargins(0, 0, 0, 0)
        self.prev_button = QPushButton("上一页")
        self.next_button = QPushButton("下一页")
        self.page_label = QLabel("第 1 页")
        self.page_label.setObjectName("MutedText")
        self.prev_button.clicked.connect(self.previous_page_requested.emit)
        self.next_button.clicked.connect(self.next_page_requested.emit)
        pager_layout.addWidget(self.prev_button)
        pager_layout.addWidget(self.next_button)
        pager_layout.addWidget(self.page_label)
        pager_layout.addStretch(1)
        self.pager.setVisible(False)
        layout.addWidget(self.pager)

        # 基础筛选发生变化时统一进入 _filter_changed()；只要已有关键词，就重新
        # 发出 search_requested，让同一条后台搜索流程重新读取最新页面条件。
        self.file_type_combo.currentIndexChanged.connect(self._filter_changed)
        self.mode_combo.currentIndexChanged.connect(self._filter_changed)
        self.scope_combo.currentIndexChanged.connect(self._filter_changed)
        self.sort_combo.currentIndexChanged.connect(self._filter_changed)
        for checkbox in (
            self.advanced.search_filename,
            self.advanced.search_path,
            self.advanced.search_content,
            self.advanced.include_ocr,
            self.advanced.ocr_fuzzy,
            self.advanced.case_sensitive,
        ):
            checkbox.toggled.connect(self._update_filter_button)
        self._update_filter_button()
        self.show_idle_state()

    def _on_text_changed(self, text: str) -> None:
        # 该函数由 QLineEdit.textChanged 触发，每输入或删除一个字符都会进入。
        self.search_box.update_clear_button()
        if not text.strip():
            # 输入为空时取消尚未触发的自动搜索，并恢复页面空闲状态。
            self.debounce.stop()
            self.show_idle_state()
            return
        if self.auto_search_enabled:
            # start() 会重新开始单次计时；连续输入时旧计时不会提前触发搜索。
            self.debounce.start()

    def _filter_changed(self) -> None:
        # 没有关键词时，修改筛选只保存界面状态，不发起一次无意义的空搜索。
        if self.text():
            self.search_requested.emit()

    def _toggle_advanced(self) -> None:
        self.advanced.setVisible(self.more_button.isChecked())
        self._update_filter_button()

    def _update_filter_button(self) -> None:
        active_count = sum(
            (
                not self.advanced.search_filename.isChecked(),
                not self.advanced.search_path.isChecked(),
                not self.advanced.search_content.isChecked(),
                not self.advanced.include_ocr.isChecked(),
                self.advanced.ocr_fuzzy.isChecked(),
                self.advanced.case_sensitive.isChecked(),
            )
        )
        label = "收起筛选" if self.more_button.isChecked() else "筛选"
        self.more_button.setText(f"{label} · {active_count}" if active_count else label)
        self.more_button.setProperty("active", bool(active_count or self.more_button.isChecked()))
        self.more_button.style().unpolish(self.more_button)
        self.more_button.style().polish(self.more_button)

    def set_roots(self, roots: list[object]) -> None:
        # 外层刷新搜索目录时，临时屏蔽下拉框信号，防止重建选项的中间过程
        # 被误认为用户修改范围，从而意外触发搜索。
        current = self.scope_combo.currentData()
        self.scope_combo.blockSignals(True)
        self.scope_combo.clear()
        self.scope_combo.addItem("全部范围", None)
        for row in roots:
            self.scope_combo.addItem(Path(str(row["path"])).name or str(row["path"]), int(row["id"]))
        restore = self.scope_combo.findData(current)
        self.scope_combo.setCurrentIndex(restore if restore >= 0 else 0)
        self.scope_combo.blockSignals(False)
        self._has_roots = bool(roots)

    def set_stats(self, stats: dict[str, object], *, has_roots: bool) -> None:
        self._stats = stats
        self._has_roots = has_roots
        if not self.text():
            self.show_idle_state()

    def set_index_ready(
        self,
        ready: bool,
        readiness: dict[str, object] | None = None,
    ) -> None:
        self._index_ready = bool(ready)
        self._readiness = dict(readiness or {})
        can_queue_while_indexing = bool(
            self._readiness.get("active_runs")
            or self._readiness.get("index_update_running")
            or self._readiness.get("scan_active")
        )
        for widget in (
            self.search_box,
            self.mode_combo,
            self.file_type_combo,
            self.scope_combo,
            self.sort_combo,
            self.more_button,
            self.advanced,
        ):
            widget.setEnabled(self._index_ready or can_queue_while_indexing)
        if not self._index_ready:
            self.debounce.stop()
            self.show_index_not_ready_state()
        elif self.content_stack.currentWidget() is self.empty_state:
            self.show_idle_state()

    def index_ready(self) -> bool:
        return self._index_ready

    def index_complete(self) -> bool:
        return bool(self._readiness.get("index_complete", self._readiness.get("ready")))

    def index_readiness(self) -> dict[str, object]:
        return dict(self._readiness)

    def focus_search(self) -> None:
        self.search_box.input.setFocus()
        self.search_box.input.selectAll()

    def clear_search(self) -> None:
        self.search_box.input.clear()
        self.show_idle_state()

    def text(self) -> str:
        # MainWindow 需要关键词时，通过 SearchPage.text() 读取内部输入框。
        # strip() 去掉用户不小心输入在关键词首尾的空格。
        return self.search_box.input.text().strip()

    def mode_value(self) -> str:
        # getter 方法把控件内部值转换成稳定的普通 Python 类型。MainWindow
        # 通过这些公开方法读取条件，不直接访问 mode_combo 等页面内部控件。
        return str(self.mode_combo.currentData())

    def root_ids(self) -> list[int]:
        value = self.scope_combo.currentData()
        return [int(value)] if value is not None else []

    def extensions(self) -> list[str]:
        # 页面显示“PDF”等业务分组，SearchQuery 实际需要的是扩展名列表。
        label = str(self.file_type_combo.currentData())
        return sorted(FILE_TYPE_GROUPS.get(label, set()))

    def sort_value(self) -> str:
        return str(self.sort_combo.currentData() or "relevance")

    def search_filename(self) -> bool:
        return self.advanced.search_filename.isChecked()

    def search_path(self) -> bool:
        return self.advanced.search_path.isChecked()

    def search_content(self) -> bool:
        return self.advanced.search_content.isChecked()

    def include_ocr(self) -> bool:
        return self.advanced.include_ocr.isChecked()

    def include_ocr_fuzzy(self) -> bool:
        return self.advanced.ocr_fuzzy.isChecked()

    def case_sensitive(self) -> bool:
        return self.advanced.case_sensitive.isChecked()

    def set_running(self, running: bool) -> None:
        # MainWindow 在任务开始/结束时调用此方法。SearchPage 只改变展示状态，
        # 不创建线程，也不判断 SearchWorker 是否真的已经结束。
        self.search_box.set_running(running)
        if running:
            self.partial_results_visible = False
            self.search_progress.setVisible(True)
            self.search_progress.setRange(0, 0)
            self._active_estimate_context = self._estimate_context()
            estimate = self._estimator.estimate(self._active_estimate_context)
            self.set_status("正在检索索引...")
            self.timing_label.setText(estimate.display_text())
            self.timing_label.setVisible(True)
            if self.content_stack.currentWidget() is self.empty_state:
                self.empty_state.set_content(
                    "正在检索",
                    "正在匹配已建立的全文索引",
                    "",
                    None,
                )
        else:
            self.search_progress.setVisible(False)
            self.search_progress.setRange(0, 1)
            self.search_progress.setValue(0)

    def set_search_progress(self, payload: object) -> None:
        # SearchWorker 的进度最终由 MainWindow 转交到这里；页面把普通字典转换成
        # 进度条、阶段文字、命中数量和耗时提示。
        if not isinstance(payload, dict):
            return
        phase = str(payload.get("phase_label") or "正在搜索...")
        checked = int(payload.get("checked_candidates") or 0)
        total = int(payload.get("total_candidates") or 0)
        confirmed = int(payload.get("confirmed_files") or 0)
        elapsed_ms = int(payload.get("elapsed_ms") or 0)
        progress_kind = str(payload.get("progress_kind") or "busy")
        if progress_kind == "determinate" and total > 0:
            self.search_progress.setRange(0, total)
            self.search_progress.setValue(min(checked, total))
            detail = f"校验索引记录 {checked:,}/{total:,} · 已找到 {confirmed:,} 个文件"
        else:
            self.search_progress.setRange(0, 0)
            detail = f"校验索引记录 {checked:,} · 已找到 {confirmed:,} 个文件" if checked else ""
        if elapsed_ms >= 2_000:
            slow_reason = str(payload.get("slow_reason") or "")
            if "关键词较短" in slow_reason:
                # 通用估时器基于普通搜索训练，对两字关键词兼容扫描可能显示过时的
                # 数分钟范围，因此这里只展示实时阶段，不展示没有依据的预计时间。
                self.timing_label.setText("两字关键词兼容检索中")
                self.timing_label.setVisible(True)
            suffix = f" · {detail}" if detail else ""
            if slow_reason:
                suffix += f" · {slow_reason}"
            suffix += f" · 已用时 {format_elapsed(elapsed_ms)}"
            self.set_status(phase + suffix)
        else:
            self.set_status(phase + (f" · {detail}" if detail else ""))

    def set_partial_results(self, page: object) -> None:
        # 正文搜索尚未结束时，可以先展示文件名/路径命中。这里只更新界面，
        # 不代表整次搜索已经完成，所以暂时隐藏分页。
        if page is None or int(page.total_confirmed) <= 0:
            return
        self.partial_results_visible = True
        self.page_label.setText(f"第 {page.page} 页")
        self.content_stack.setCurrentWidget(self.result_view)
        self.result_view.set_results(page.results, self.text())
        self.pager.setVisible(False)
        scope_notice = self._partial_scope_notice()
        self.set_status(
            f"已先找到 {page.total_confirmed} 个文件名/路径结果，正在继续搜索正文..."
            f"{scope_notice}"
        )

    def set_results(self, page: object) -> None:
        # 完整 SearchPage 结果到达后，更新耗时模型、状态文字、结果列表和分页。
        # 参数 page 是搜索结果数据对象，与当前这个 UI 类 SearchPage 同名但含义不同。
        self.partial_results_visible = False
        if self._active_estimate_context is not None:
            self._estimator.observe(self._active_estimate_context, int(page.elapsed_ms))
            self._active_estimate_context = None
        self.page_label.setText(f"第 {page.page} 页")
        if page.truncated:
            count_text = (
                f"找到 {page.total_confirmed} 个文件，仅显示前 {page.available_results} 条，"
                "可能还有更多"
            )
        else:
            count_text = f"找到 {page.total_confirmed} 个文件"
        text_hits, ocr_hits, fuzzy_hits = self._hit_breakdown(page.results)
        breakdown = (
            f"当前页正文命中 {text_hits:,} · OCR 命中 {ocr_hits:,} · "
            f"OCR 疑似 {fuzzy_hits:,}"
        )
        self.set_status(
            f"{count_text} · {breakdown} · 校验索引记录 {page.total_candidates:,} 条"
            f"{self._partial_scope_notice()}"
        )
        self.timing_label.setText(f"实际用时 {format_elapsed(int(page.elapsed_ms))}")
        self.timing_label.setVisible(True)
        if page.total_confirmed <= 0:
            self.show_no_results()
            return
        self.content_stack.setCurrentWidget(self.result_view)
        self.result_view.set_results(page.results, self.text())
        self.pager.setVisible(page.available_results > page.page_size)

    def _partial_scope_notice(self) -> str:
        if not bool(self._readiness.get("search_is_partial")):
            return ""
        complete = int(self._readiness.get("complete_files") or 0)
        eligible = int(self._readiness.get("eligible_files") or 0)
        return f" · 部分索引 {complete:,}/{eligible:,}，结果可能不完整"

    @staticmethod
    def _hit_breakdown(results: list[object]) -> tuple[int, int, int]:
        text_hits = 0
        ocr_hits = 0
        fuzzy_hits = 0
        for result in results:
            for hit in result_hits(result):
                count = max(1, int(hit.hit_count or 0))
                if hit.source_type == "ocr":
                    ocr_hits += count
                    if hit.is_fuzzy:
                        fuzzy_hits += count
                elif hit.source_type != "metadata":
                    text_hits += count
        return text_hits, ocr_hits, fuzzy_hits

    def apply_settings(self, settings: AppSettings) -> None:
        # 初始化末尾统一应用设置，使“创建控件”和“用配置填充控件”分成两个步骤。
        self.auto_search_enabled = bool(settings.auto_search)
        self.debounce.setInterval(max(100, int(settings.search_debounce_ms)))
        mode_index = self.mode_combo.findData(settings.default_search_mode)
        if mode_index >= 0:
            self.mode_combo.setCurrentIndex(mode_index)
        self.advanced.include_ocr.setChecked(settings.enable_ocr)

    def _estimate_context(self) -> SearchEstimateContext:
        return SearchEstimateContext(
            mode=self.mode_value(),
            file_count=int(self._stats.get("files", 0)),
            scoped=bool(self.root_ids()),
            extension_filtered=bool(self.extensions()),
            searches_content=self.search_content(),
            ocr_fuzzy=self.include_ocr_fuzzy(),
            case_sensitive=self.case_sensitive(),
        )

    def clear_timing(self) -> None:
        self._active_estimate_context = None
        self.timing_label.clear()
        self.timing_label.setVisible(False)

    def set_history(self, items: list[str]) -> None:
        self._history = items
        self.search_box.history_button.setEnabled(bool(items))
        self.search_box.history_button.setVisible(bool(items))

    def _show_history(self) -> None:
        if not self._history:
            return
        menu = QMenu(self)
        for text in self._history:
            action = menu.addAction(text)
            action.triggered.connect(lambda _checked=False, value=text: self._select_history(value))
        menu.addSeparator()
        clear_action = menu.addAction("清空搜索历史")
        clear_action.triggered.connect(self.clear_history_requested.emit)
        button = self.search_box.history_button
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))

    def _select_history(self, text: str) -> None:
        self.search_box.input.setText(text)
        self.search_requested.emit()

    def show_idle_state(self) -> None:
        if not self._has_roots:
            self.empty_state.set_content(
                "尚未添加搜索范围",
                "添加一个本地文件夹后，即可建立全文索引并开始搜索",
                "添加搜索范围",
                self.add_root_requested.emit,
            )
        elif not self._index_ready:
            self.show_index_not_ready_state()
            return
        else:
            complete = self._stats.get("complete_files")
            eligible = self._stats.get("eligible_files")
            blocking = self._stats.get("blocking_files")
            video = self._stats.get("video_excluded")
            excluded = self._stats.get("manual_excluded_files")
            known_counts = all(value is not None for value in (complete, eligible, blocking))
            scope_text = (
                f"{int(complete or 0):,}/{int(eligible or 0):,}"
                if known_counts
                else "统计更新中"
            )
            video_text = f"，另排除视频 {int(video):,} 个" if video is not None and video else ""
            excluded_text = f"，已人工排除 {int(excluded):,} 个" if excluded is not None and excluded else ""
            self.empty_state.set_content(
                "输入关键词开始搜索",
                "支持搜索 PDF、Word、Excel、PowerPoint、文本、日志和图片中的文字\n"
                + f"当前范围完成 {scope_text}"
                + f"{excluded_text}{video_text}",
                "",
                None,
            )
        self.content_stack.setCurrentWidget(self.empty_state)
        self.pager.setVisible(False)
        self.set_status("输入关键词开始搜索")
        self.clear_timing()

    def show_index_not_ready_state(self) -> None:
        if not self._has_roots:
            self.show_idle_state()
            return
        if bool(self._readiness.get("index_update_running")):
            self.empty_state.set_content(
                "正在更新搜索索引...",
                "人工排除任务正在后台处理；完成或回滚后会自动恢复搜索。",
                "",
                None,
            )
            self.content_stack.setCurrentWidget(self.empty_state)
            self.pager.setVisible(False)
            self.set_status("索引正在更新，请稍候")
            self.clear_timing()
            return
        complete = self._readiness.get("complete_files")
        eligible = self._readiness.get("eligible_files")
        blocking = self._readiness.get("blocking_files")
        active = int(self._readiness.get("active_runs") or 0)
        unready_roots = int(self._readiness.get("unready_roots") or 0)
        excluded = self._readiness.get("manual_excluded_files")
        excluded_text = (
            f"，已人工排除 {int(excluded):,} 个"
            if excluded is not None and excluded
            else ""
        )
        scope_text = (
            f"{int(complete or 0):,}/{int(eligible or 0):,}"
            if complete is not None and eligible is not None
            else "统计更新中"
        )
        if active:
            subtitle = (
                f"正在建立当前范围索引：已完成 {scope_text}{excluded_text}。"
                "范围内文件达到允许终态后会自动开放搜索。"
            )
        elif blocking:
            subtitle = (
                f"当前范围已完成 {scope_text}{excluded_text}，仍有 {int(blocking or 0):,} 个阻断项。"
                "请在“未成功索引”中查看原因并重新更新。"
            )
        elif unready_roots:
            unfinished = int(self._readiness.get("unfinished_tasks") or 0)
            unpublished = int(self._readiness.get("unpublished_candidates") or 0)
            if not blocking and (unfinished or unpublished or bool(self._readiness.get("content_fts_dirty"))):
                details = []
                if unfinished:
                    details.append(f"残留解析任务 {unfinished} 个")
                if unpublished:
                    details.append(f"未发布全文索引 {unpublished} 个")
                if bool(self._readiness.get("content_fts_dirty")):
                    details.append("全文索引待发布")
                subtitle = (
                    "文件已经全部处理，但"
                    + "、".join(details)
                    + "。请在“未成功索引”中执行保底完成并开放搜索。"
                )
            else:
                subtitle = "搜索范围尚未完成首次索引，请先在“索引管理”中更新全部。"
        else:
            subtitle = "完整索引尚未就绪，请先更新全部。"
        self.empty_state.set_content("完整索引尚未完成", subtitle, "", None)
        self.content_stack.setCurrentWidget(self.empty_state)
        self.pager.setVisible(False)
        self.set_status("完整索引完成前不可搜索")
        self.clear_timing()

    def show_no_results(self) -> None:
        self.empty_state.set_content(
            f"没有找到“{self.text()}”",
            "可以尝试检查搜索范围、切换为全部关键词，或确认 OCR 结果已完成索引",
            "",
            None,
        )
        self.content_stack.setCurrentWidget(self.empty_state)
        self.pager.setVisible(False)

    def show_interrupted_state(self, title: str) -> None:
        if self.content_stack.currentWidget() is self.empty_state:
            self.empty_state.set_content(title, "检索条件已保留", "", None)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class SearchBox(QFrame):
    """搜索输入区域，只负责收集输入和发出用户操作信号。

    它是整条搜索链路的起点，但不是搜索控制器：
    1. 它不知道数据库在哪里；
    2. 它不知道怎样创建 SearchQuery；
    3. 它不知道怎样启动 SearchWorker；
    4. 它只把“搜索、停止、查看历史”这三种用户意图通知给外层。
    """

    # SearchBox 是纯界面组件：它只发出“搜索/停止/历史”意图，不创建
    # SearchQuery，也不知道 SearchWorker、SearchEngine 或数据库的存在。
    # Signal() 可以理解成一个“通知出口”：SearchBox 负责发出，外层负责接收。
    search_requested = Signal()
    stop_requested = Signal()
    history_requested = Signal()

    def __init__(self) -> None:
        # __init__ 只在创建 SearchBox 时执行一次，用来组装输入框和按钮。
        super().__init__()
        # objectName 主要供 Qt 样式表定位这个控件，不参与搜索逻辑。
        self.setObjectName("SearchBox")
        # 创建左侧的搜索图标。
        icon = QLabel("⌕")
        icon.setObjectName("SearchIcon")
        # QLineEdit 只负责接收和显示用户输入。
        self.input = QLineEdit()
        self.input.setObjectName("SearchInput")
        # 输入框为空时显示提示文字；这段文字不属于真正的搜索关键词。
        self.input.setPlaceholderText("搜索文件名、正文、表格、幻灯片和图片文字")
        # 用户按回车时只发信号；MainWindow 收到信号后才统一安排搜索。
        # returnPressed 是 QLineEdit 自带的信号，search_requested 是本类对外信号。
        # 这行完成了一次“信号转发”，中间没有执行数据库查询。
        self.input.returnPressed.connect(self.search_requested.emit)
        # 创建“清空”按钮，并设置它的样式名、文字、尺寸和提示。
        self.clear_button = QPushButton()
        self.clear_button.setObjectName("SearchActionButton")
        self.clear_button.setText("清空")
        self.clear_button.setFixedSize(48, 30)
        self.clear_button.setToolTip("清空搜索内容")
        # 点击清空按钮时，直接调用 QLineEdit.clear() 清除输入文字。
        self.clear_button.clicked.connect(self.input.clear)
        # 创建搜索历史按钮。当前功能未开放，所以按钮被禁用并隐藏。
        self.history_button = QPushButton()
        self.history_button.setObjectName("SearchActionButton")
        self.history_button.setText("历史")
        self.history_button.setFixedSize(48, 30)
        self.history_button.setToolTip("搜索历史")
        self.history_button.setEnabled(False)
        self.history_button.setVisible(False)
        # 将来按钮显示后，点击它只发出“查看历史”信号。
        self.history_button.clicked.connect(self.history_requested.emit)
        # 创建停止按钮。点击时只发出停止意图，不直接操作后台线程。
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        # 初始没有搜索任务，所以停止按钮不可见。
        self.stop_button.setVisible(False)
        # 创建输入区和按钮区之间的竖直分隔线。
        self.action_separator = QFrame()
        self.action_separator.setObjectName("SearchActionSeparator")
        self.action_separator.setFixedSize(1, 24)
        # 使用水平布局，从左到右排列图标、输入框、分隔线和三个按钮。
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 10, 0)
        layout.setSpacing(8)
        layout.addWidget(icon)
        # 第二个参数 1 表示输入框可以占用剩余的可伸缩空间。
        layout.addWidget(self.input, 1)
        layout.addWidget(self.action_separator)
        layout.addWidget(self.history_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.stop_button)
        # 根据输入框当前是否有文字，设置“清空”按钮的初始显示状态。
        self.update_clear_button()

    def update_clear_button(self) -> None:
        # self.input.text() 返回字符串；bool(字符串) 把它转换成是否有内容。
        # 有文字时显示“清空”，空字符串时隐藏。
        self.clear_button.setVisible(bool(self.input.text()))

    def set_running(self, running: bool) -> None:
        # 外层开始搜索时传入 True，显示停止按钮；搜索结束时传入 False。
        # SearchBox 只改变按钮显示，不负责判断任务究竟是否结束。
        self.stop_button.setVisible(running)


class AdvancedFilters(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("AdvancedFilters")
        self.search_filename = QCheckBox("搜索文件名")
        self.search_path = QCheckBox("搜索路径")
        self.search_content = QCheckBox("搜索正文")
        self.include_ocr = QCheckBox("包含 OCR")
        self.case_sensitive = QCheckBox("大小写敏感")
        self.ocr_fuzzy = QCheckBox("显示 OCR 疑似命中")
        self.search_filename.setToolTip("匹配文件名；文件名命中会优先快速显示")
        self.search_path.setToolTip("匹配文件夹路径和文件名所在路径")
        self.search_content.setToolTip("匹配已建立的正文索引")
        self.include_ocr.setToolTip("包含图片和扫描 PDF 的 OCR 文字")
        self.ocr_fuzzy.setToolTip("纳入低置信度 OCR 候选；结果会标记为疑似命中")
        self.case_sensitive.setToolTip("区分英文字母大小写")
        for checkbox in (self.search_filename, self.search_path, self.search_content, self.include_ocr):
            checkbox.setChecked(True)
        layout = QGridLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(16)
        layout.setVerticalSpacing(10)
        for index, checkbox in enumerate(
            (
                self.search_filename,
                self.search_path,
                self.search_content,
                self.include_ocr,
                self.ocr_fuzzy,
                self.case_sensitive,
            )
        ):
            layout.addWidget(checkbox, index // 4, index % 4)


class EmptyState(QFrame):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("EmptyState")
        self._callback: Callable[[], None] | None = None
        self.title = QLabel()
        self.title.setObjectName("EmptyTitle")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle = QLabel()
        self.subtitle.setObjectName("EmptySubtitle")
        self.subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.subtitle.setWordWrap(True)
        self.action = QPushButton()
        self.action.setObjectName("PrimaryButton")
        self.action.setVisible(False)
        self.action.clicked.connect(self._run_callback)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(12)
        layout.addStretch(1)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.action, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

    def set_content(self, title: str, subtitle: str, button_text: str, callback: Callable[[], None] | None) -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)
        self._callback = callback
        if button_text and callback:
            self.action.setText(button_text)
            self.action.setVisible(True)
        else:
            self.action.setVisible(False)

    def _run_callback(self) -> None:
        if self._callback is not None:
            self._callback()


class IndexPage(QWidget):
    add_root_requested = Signal()
    scan_requested = Signal(object)
    performance_scan_requested = Signal()
    scope_performance_scan_requested = Signal(object)
    pause_requested = Signal()
    resume_requested = Signal()
    mode_switch_requested = Signal(bool)
    cancel_requested = Signal()
    toggle_root_requested = Signal(int, bool)
    remove_root_requested = Signal(int)
    open_folder_requested = Signal(str)
    failed_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.running = False
        self.database_update_running = False
        self.readiness: dict[str, object] = {"ready": False}
        self.pause_state = "idle"
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        header = QHBoxLayout()
        section_title = QLabel("搜索范围")
        section_title.setObjectName("ContentHeading")
        self.add_button = QPushButton("添加搜索范围")
        self.add_button.setObjectName("PrimaryButton")
        self.add_button.clicked.connect(self.add_root_requested.emit)
        header.addWidget(section_title)
        header.addStretch(1)
        header.addWidget(self.add_button)
        layout.addLayout(header)

        self.task_strip = QFrame()
        self.task_strip.setObjectName("TaskStrip")
        task_layout = QVBoxLayout(self.task_strip)
        task_layout.setContentsMargins(14, 10, 14, 10)
        task_layout.setSpacing(7)
        task_overview = QHBoxLayout()
        task_overview.setSpacing(8)
        self.task_label = QLabel("索引已就绪")
        self.task_label.setObjectName("MutedText")
        self.task_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.task_eta = QLabel("")
        self.task_eta.setObjectName("IndexEta")
        self.task_eta.setMinimumWidth(140)
        self.task_eta.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.task_progress = QProgressBar()
        self.task_progress.setTextVisible(False)
        self.task_progress.setFixedWidth(160)
        self.start_button = QPushButton("更新全部范围")
        self.start_button.setToolTip("使用普通模式更新所有已启用搜索范围")
        self.performance_button = QPushButton("性能模式 · 更新全部范围")
        self.performance_button.setToolTip("使用性能模式更新所有已启用搜索范围")
        self.pause_button = QPushButton("暂停")
        self.cancel_button = QPushButton("取消")
        self.start_button.clicked.connect(self._normal_mode_clicked)
        self.performance_button.clicked.connect(self._performance_mode_clicked)
        self.pause_button.clicked.connect(self._toggle_pause)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        task_overview.addWidget(self.task_label, 1)
        task_overview.addWidget(self.task_eta)
        task_overview.addWidget(self.task_progress)
        task_overview.addWidget(self.start_button)
        task_overview.addWidget(self.performance_button)
        task_overview.addWidget(self.pause_button)
        task_overview.addWidget(self.cancel_button)
        task_layout.addLayout(task_overview)

        self.task_detail_row = QWidget()
        detail_layout = QHBoxLayout(self.task_detail_row)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(12)
        self.task_file = QLabel("")
        self.task_file.setObjectName("MutedText")
        self.task_file.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        self.task_file.setMinimumWidth(80)
        self.task_file.setWordWrap(True)
        self.task_file.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.task_runtime = QLabel("")
        self.task_runtime.setObjectName("MutedText")
        self.task_runtime.setMinimumWidth(118)
        self.task_phase = QLabel("")
        self.task_phase.setObjectName("MutedText")
        self.task_phase.setMinimumWidth(96)
        self.task_units = QLabel("")
        self.task_units.setObjectName("MutedText")
        self.task_units.setMinimumWidth(84)
        detail_layout.addWidget(self.task_file, 1)
        detail_layout.addWidget(self.task_runtime)
        detail_layout.addWidget(self.task_phase)
        detail_layout.addWidget(self.task_units)
        task_layout.addWidget(self.task_detail_row)
        layout.addWidget(self.task_strip)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_host = QWidget()
        self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(0)
        self.scroll.setWidget(self.list_host)
        layout.addWidget(self.scroll, 1)
        self.last_run_summary = ""
        self.set_task_running(False)

    def set_database_update_running(self, running: bool) -> None:
        self.database_update_running = bool(running)
        self.add_button.setEnabled(not running)
        self.scroll.setEnabled(not running)
        self.start_button.setEnabled(not running)
        self.performance_button.setEnabled(not running)
        if running:
            self.task_progress.setVisible(True)
            self.task_progress.setRange(0, 0)
            self.task_detail_row.setVisible(False)
            self.pause_button.setVisible(False)
            self.cancel_button.setVisible(False)
            self.task_label.setText("正在更新搜索索引...")
            self.task_eta.setText("后台事务进行中")
        else:
            self.set_task_running(False)

    def set_database_update_progress(self, phase_label: str) -> None:
        if not self.database_update_running:
            return
        self.task_label.setText(phase_label or "正在更新搜索索引...")

    def set_readiness(self, readiness: dict[str, object]) -> None:
        self.readiness = dict(readiness)
        if not self.running and not self.database_update_running:
            self.task_label.setText(self._idle_status_text())

    def _idle_status_text(self) -> str:
        if bool(self.readiness.get("ready")):
            return "索引已就绪"
        blockers = int(self.readiness.get("blocking_files") or 0)
        unfinished = int(self.readiness.get("unfinished_tasks") or 0)
        unpublished = int(self.readiness.get("unpublished_candidates") or 0)
        if blockers:
            return f"索引未完成 · {blockers} 个阻断项"
        details = []
        if unfinished:
            details.append(f"残留任务 {unfinished}")
        if unpublished:
            details.append(f"未发布索引 {unpublished}")
        if bool(self.readiness.get("content_fts_dirty")):
            details.append("全文索引待发布")
        return "索引状态待修复" + (" · " + " · ".join(details) if details else "")

    def set_performance_mode(self, enabled: bool) -> None:
        self.performance_mode = enabled
        if enabled and not self.running:
            self.task_label.setText("正在探测本机硬件并生成性能模式配置...")
        elif not self.running:
            self.task_label.setText(self._idle_status_text())

    def set_performance_profile(self, profile: dict[str, object]) -> None:
        normal = int(profile.get("normal_workers") or 0)
        office = int(profile.get("office_workers") or 0)
        pdf = int(profile.get("pdf_workers") or profile.get("pdf_parser_workers") or 0)
        zip_workers = int(profile.get("zip_member_workers") or 0)
        ocr = int(profile.get("ocr_workers") or 0)
        memory = int(profile.get("memory_budget_mb") or 0)
        disk = str(profile.get("disk_class") or "unknown")
        self.task_label.setText("正在检查文件变化…")
        self.task_label.setToolTip(
            "性能模式配置已应用："
            f"普通 {normal} / Office {office} / PDF {pdf} / ZIP {zip_workers} / OCR {ocr} "
            f"· 内存 {memory} MB · {disk}"
        )

    def set_scan_starting(self, label: str = "正在启动更新…") -> None:
        self.task_label.setText(label)
        self.task_file.clear()
        self.task_runtime.setText("启动检查")
        self.task_phase.setText("等待扫描")
        self.task_units.clear()
        self.task_progress.setRange(0, 0)

    def set_roots(self, roots: list[object], stats: dict[int, dict[str, int]]) -> None:
        clear_layout(self.list_layout)
        if not roots:
            empty = EmptyState()
            empty.set_content("尚未添加搜索范围", "添加文件夹后即可建立索引并开始搜索", "添加搜索范围", self.add_root_requested.emit)
            self.list_layout.addWidget(empty)
        else:
            for row in roots:
                card = RootCard(row, stats.get(int(row["id"]), {}))
                card.scan_requested.connect(self.scan_requested.emit)
                card.performance_scan_requested.connect(
                    self.scope_performance_scan_requested.emit
                )
                card.toggle_requested.connect(self.toggle_root_requested.emit)
                card.remove_requested.connect(self.remove_root_requested.emit)
                card.open_requested.connect(self.open_folder_requested.emit)
                card.failed_requested.connect(self.failed_requested.emit)
                self.list_layout.addWidget(card)
        self.list_layout.addStretch(1)

    def set_task_running(self, running: bool) -> None:
        was_running = self.running
        self.running = running
        if running:
            self.last_run_summary = ""
        self.task_progress.setVisible(running)
        self.task_detail_row.setVisible(running)
        self.pause_button.setVisible(running)
        self.cancel_button.setVisible(running)
        mode_enabled = not running or self.pause_state == "paused"
        self.start_button.setEnabled(mode_enabled and not self.database_update_running)
        self.performance_button.setEnabled(
            mode_enabled and not self.database_update_running
        )
        if not running:
            self.task_progress.setValue(0)
            self.task_label.setText(self._idle_status_text())
            self.task_eta.clear()
            self.task_file.clear()
            self.task_runtime.clear()
            self.task_phase.clear()
            self.task_units.clear()
            if self.last_run_summary:
                self.task_label.setText(self.last_run_summary)
            self.pause_button.setText("暂停")
            self.paused = False
            self.pause_state = "idle"
            self.performance_mode = False
            self.start_button.setText("更新全部范围")
            self.performance_button.setText("性能模式 · 更新全部范围")
        elif not was_running:
            self.paused = False
            self.pause_state = "running"
            self.pause_button.setText("暂停")
            self.pause_button.setEnabled(True)
            self.start_button.setText("普通模式")
            self.performance_button.setText("性能模式")

    def show_run_summary(self, text: str, tooltip: str = "") -> None:
        self.last_run_summary = text
        self.task_label.setText(text)
        self.task_label.setToolTip(tooltip or text)

    def _toggle_pause(self) -> None:
        if not self.running:
            return
        if self.pause_state == "paused":
            self.set_pause_state("resuming", "正在从安全检查点继续")
            self.resume_requested.emit()
        elif self.pause_state == "running":
            self.set_pause_state("pausing", "正在暂停，等待活动任务到达安全检查点")
            self.pause_requested.emit()

    def _normal_mode_clicked(self) -> None:
        if self.running:
            if self.pause_state == "paused":
                self.mode_switch_requested.emit(False)
            return
        self.scan_requested.emit(None)

    def _performance_mode_clicked(self) -> None:
        if self.running:
            if self.pause_state == "paused":
                self.mode_switch_requested.emit(True)
            return
        self.performance_scan_requested.emit()

    def set_pause_state(self, state: str, label: str = "") -> None:
        self.pause_state = state
        self.paused = state == "paused"
        if state == "pausing":
            self.pause_button.setText("正在暂停")
            self.pause_button.setEnabled(False)
        elif state == "paused":
            self.pause_button.setText("继续")
            self.pause_button.setEnabled(True)
        elif state == "resuming":
            self.pause_button.setText("正在继续")
            self.pause_button.setEnabled(False)
        elif state == "running":
            self.pause_button.setText("暂停")
            self.pause_button.setEnabled(True)
        mode_switch_enabled = self.running and state == "paused"
        self.start_button.setEnabled(mode_switch_enabled or not self.running)
        self.performance_button.setEnabled(mode_switch_enabled or not self.running)
        if label:
            self.task_label.setText(label)
        if state == "paused":
            self.task_eta.setText("已暂停 · 继续后重新估算")

    def pause_state_text(self) -> str:
        return {
            "pausing": "正在暂停",
            "paused": "已暂停",
            "resuming": "正在继续",
        }.get(self.pause_state, "正在索引")

    def set_scan_progress(
        self,
        completed: int,
        total: int,
        failed: int,
        current: str,
        *,
        phase_label: str = "正在索引",
        eta_text: str = "",
        active_elapsed_seconds: int = 0,
        active_queue: str = "",
        active_file_count: int = 0,
        active_phase: str = "",
        active_completed_units: int = 0,
        active_total_units: int = 0,
        planning_phase: str = "",
        planning_completed: int = 0,
        planning_total: int = 0,
        discovered_files: int = 0,
        reused_files: int = 0,
        no_progress_seconds: int = 0,
        retry_count: int = 0,
        excluded_video: int = 0,
        diagnostic_state: str = "",
        diagnostic_reason: str = "",
        representative_is_slowest: bool = False,
        other_active_lane_count: int = 0,
        other_recent_progress_seconds: int = 0,
        indeterminate: bool = False,
    ) -> None:
        self.set_task_running(True)
        if self.pause_state not in {"pausing", "paused"}:
            self.set_pause_state("running")
        current_display = str(current or "")
        self.task_file.setText(current_display)
        self.task_file.setToolTip(current_display)
        runtime_text = ""
        phase_text = ""
        units_text = ""
        if active_elapsed_seconds > 0:
            queue_labels = {
                "normal": "普通",
                "ocr": "OCR",
                "zip": "ZIP",
                "office_process": "Office",
                "legacy_office": "旧版 Office",
                "legacy_word": "旧版 Word",
                "legacy_excel": "旧版 Excel",
                "legacy_powerpoint": "旧版 PowerPoint",
            }
            queue_label = queue_labels.get(active_queue, active_queue)
            if "ocr" in active_phase.lower():
                queue_label = "OCR"
            runtime_text = (
                f"{queue_label} 已运行 {format_active_duration(active_elapsed_seconds)}"
            )
            if active_file_count > 1:
                runtime_text += f" · 活动 {active_file_count}"
        active_phase_text = format_active_phase(active_phase) if active_phase else ""
        diagnostic_text = scheduler_diagnostic_label(diagnostic_state)
        if diagnostic_text:
            phase_text = diagnostic_text
        elif active_phase_text:
            phase_text = active_phase_text
            if active_total_units > 0:
                units_text = f"{active_completed_units:,}/{active_total_units:,}"
        if no_progress_seconds > 0:
            phase_text += (
                f" · 距上次进展 {format_active_duration(no_progress_seconds)}"
            )
        if retry_count > 0:
            phase_text += f" · 重试 {retry_count}"
        self.task_runtime.setText(runtime_text)
        phase_tooltip_parts = [part for part in (phase_text, active_phase_text) if part]
        if diagnostic_reason:
            phase_tooltip_parts.append(diagnostic_reason)
        if representative_is_slowest and other_active_lane_count > 0:
            parallel_text = (
                "当前显示最慢任务；"
                f"另有 {other_active_lane_count} 个车道仍在处理，"
                f"最近有效进展 {max(0, other_recent_progress_seconds)} 秒前"
            )
            phase_tooltip_parts.append(parallel_text)
            phase_text += f" · 另 {other_active_lane_count} 车道有进展"
        self.task_phase.setText(phase_text)
        self.task_phase.setToolTip("；".join(dict.fromkeys(phase_tooltip_parts)))
        mode_label = "性能模式" if getattr(self, "performance_mode", False) else "普通模式"
        video_suffix = f"·排除视频 {excluded_video}" if excluded_video else ""
        # Once a parser has reported a real active phase, show that file and
        # phase. A late planning event must not overwrite an active parse and
        # make the UI look stuck on discovery.
        planning_active = bool(planning_phase) and not active_phase and not diagnostic_state
        if planning_active:
            dynamic_total = max(total, discovered_files, completed, 1)
            self.task_label.setText(
                f"{mode_label} · 已完成 {completed:,}/{dynamic_total:,}"
                f" · 失败 {failed}{video_suffix}"
            )
            if planning_total > 0:
                units_text = f"规划 {planning_completed:,}/{planning_total:,}"
            elif discovered_files > 0:
                units_text = f"发现 {discovered_files:,}"
        else:
            self.task_label.setText(
                f"{mode_label}·总体 {completed:,}/{max(total, completed):,}"
                f"·失败 {failed}{video_suffix}"
            )
        if reused_files > 0:
            units_text = (
                f"{units_text} · " if units_text else ""
            ) + f"复用 {reused_files:,}"
        self.task_units.setText(units_text)
        self.task_eta.setText(eta_text)
        if indeterminate:
            self.task_progress.setRange(0, 0)
        else:
            self.task_progress.setRange(0, max(total, completed, 1))
            self.task_progress.setValue(completed)

    def set_writer_commit_progress(
        self,
        committed_batches: int,
        committed_files: int,
        queue_depth: int,
    ) -> None:
        """Show durable SQLite progress without overwriting parser progress."""

        if not self.running:
            return
        text = (
            f"数据库已提交 {max(0, committed_batches):,} 批"
            f" / {max(0, committed_files):,} 个结果"
        )
        if queue_depth > 0:
            text += f" · 待写 {queue_depth:,}"
        self.task_units.setText(text)


class RootCard(QFrame):
    scan_requested = Signal(object)
    performance_scan_requested = Signal(object)
    toggle_requested = Signal(int, bool)
    remove_requested = Signal(int)
    open_requested = Signal(str)
    failed_requested = Signal()

    def __init__(self, row: object, stats: dict[str, int]) -> None:
        super().__init__()
        self.setObjectName("ScopeCard")
        def value(name: str, default: object = None) -> object:
            if isinstance(row, dict):
                return row.get(name, default)
            keys = getattr(row, "keys", None)
            if callable(keys):
                try:
                    if name in keys():
                        return row[name]  # type: ignore[index]
                except (KeyError, TypeError):
                    pass
            return default

        self.root_id = int(value("id", 0) or 0)
        self.path = str(value("path", "" ) or "")
        enabled = bool(value("enabled", True))
        last_scan = str(
            value("last_scan_at", value("updated_at", "未索引"))
            or "未索引"
        )
        total = sum(stats.values())

        title = QLabel(self.path)
        title.setObjectName("ScopeTitle")
        title.setToolTip(self.path)
        title.setWordWrap(True)
        status = QLabel("已启用" if enabled else "已禁用")
        status.setObjectName("SuccessBadge" if enabled else "StatusBadge")
        meta = QLabel(f"{total:,} 个文件 · 最后更新 {last_scan}")
        meta.setObjectName("MutedText")
        detail = QLabel(scope_detail(stats))
        detail.setObjectName("MutedText")

        update_button = QPushButton("更新此范围")
        update_button.setEnabled(enabled)
        update_button.setToolTip("只扫描并更新当前搜索范围" if enabled else "已禁用的范围不会自动或手动更新")
        update_button.clicked.connect(lambda: self.scan_requested.emit(self.root_id))
        performance_button = QPushButton("性能模式 · 更新此范围")
        performance_button.setEnabled(enabled)
        performance_button.setToolTip(
            "使用性能模式只更新当前搜索范围"
            if enabled
            else "已禁用的范围不会自动或手动更新"
        )
        performance_button.clicked.connect(
            lambda: self.performance_scan_requested.emit(self.root_id)
        )
        more_button = QPushButton("更多")
        more_button.clicked.connect(lambda: self._show_menu(more_button, enabled))

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(title, 1)
        top.addWidget(status)
        top.addWidget(update_button)
        top.addWidget(performance_button)
        top.addWidget(more_button)
        secondary = QHBoxLayout()
        secondary.setContentsMargins(0, 0, 0, 0)
        secondary.addWidget(meta)
        secondary.addSpacing(16)
        secondary.addWidget(detail, 1)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addLayout(secondary)

    def _show_menu(self, button: QPushButton, enabled: bool) -> None:
        menu = QMenu(self)
        open_action = QAction("打开文件夹", self)
        toggle_action = QAction("禁用范围" if enabled else "启用范围", self)
        failed_action = QAction("查看失败文件", self)
        delete_action = QAction("删除搜索范围", self)
        open_action.triggered.connect(lambda: self.open_requested.emit(self.path))
        toggle_action.triggered.connect(lambda: self.toggle_requested.emit(self.root_id, not enabled))
        failed_action.triggered.connect(self.failed_requested.emit)
        delete_action.triggered.connect(lambda: self.remove_requested.emit(self.root_id))
        for action in (open_action, toggle_action, failed_action, delete_action):
            menu.addAction(action)
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))


class FailedPage(QWidget):
    retry_requested = Signal(list)
    refresh_requested = Signal()
    export_requested = Signal(str, str, str)
    open_folder_requested = Signal(str)
    exclude_requested = Signal(list)
    restore_requested = Signal(list)
    force_complete_requested = Signal()
    cancel_exclusion_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.rows: list[dict[str, object]] = []
        self.rows_by_scope: dict[str, list[dict[str, object]]] = {
            "blocking": [],
            "excluded": [],
            "metadata": [],
        }
        self.page_size = 200
        self.page_by_scope: dict[str, int] = {
            "blocking": 0,
            "excluded": 0,
            "metadata": 0,
        }
        self.total_by_scope: dict[str, int] = {
            "blocking": 0,
            "excluded": 0,
            "metadata": 0,
        }
        self.all_matching_selected: dict[str, bool] = {
            "blocking": False,
            "excluded": False,
            "metadata": False,
        }
        self.selected_ids_by_scope: dict[str, set[int]] = {
            "blocking": set(),
            "excluded": set(),
            "metadata": set(),
        }
        self.all_blocking_ids: set[int] = set()
        self.index_running = False
        self.exclusion_running = False
        self.database_migration_running = False
        self.snapshot_loading = False
        self.search_running = False
        self.readiness: dict[str, object] = {"ready": False, "repairable": False}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        self.subtitle = QLabel("0 个文件需要处理")
        self.subtitle.setObjectName("ContentHeading")

        self.scope_tabs = QTabBar()
        self.scope_tabs.setObjectName("ScopeTabs")
        self.scope_tabs.addTab("阻断项 0")
        self.scope_tabs.addTab("已人工排除 0")
        self.scope_tabs.addTab("仅元数据完成 0")
        layout.addWidget(self.scope_tabs)

        self.guide = QFrame()
        self.guide.setObjectName("FailedActionGuide")
        guide_layout = QGridLayout(self.guide)
        guide_layout.setContentsMargins(14, 10, 14, 10)
        guide_layout.setHorizontalSpacing(18)
        guide_layout.setVerticalSpacing(4)
        guide_title = QLabel("如何处理未成功索引")
        guide_title.setObjectName("SectionTitle")
        guide_layout.addWidget(guide_title, 0, 0, 1, 3)
        guide_items = (
            (
                "1. 重新尝试",
                "文件已修复、解密或相关软件已恢复时使用；只重新解析，不会排除文件。",
            ),
            (
                "2. 从当前索引范围排除",
                "确认选中的文件不需要搜索时使用；源文件不会被删除，之后可以恢复纳入。",
            ),
            (
                "3. 保底完成并开放搜索",
                "保底完成并开放搜索针对整个未完成索引：先做最后一次重试，仍失败时确认排除，开放其他文件搜索。",
            ),
        )
        self.guide_labels: list[QLabel] = []
        for column, (title, body) in enumerate(guide_items):
            item = QWidget()
            item_layout = QVBoxLayout(item)
            item_layout.setContentsMargins(0, 2, 0, 2)
            item_layout.setSpacing(3)
            title_label = QLabel(title)
            title_label.setObjectName("FailedActionGuideTitle")
            body_label = QLabel(body)
            body_label.setObjectName("FailedActionGuideText")
            body_label.setWordWrap(True)
            body_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            body_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            item_layout.addWidget(title_label)
            item_layout.addWidget(body_label)
            guide_layout.addWidget(item, 1, column)
            guide_layout.setColumnStretch(column, 1)
            self.guide_labels.append(body_label)
        layout.addWidget(self.guide)

        self.risk_banner = QLabel("")
        self.risk_banner.setObjectName("FailedRiskBanner")
        self.risk_banner.setWordWrap(True)
        self.risk_banner.setVisible(False)
        layout.addWidget(self.risk_banner)

        self.recommendation_label = QLabel("")
        self.recommendation_label.setObjectName("FailedActionRecommendation")
        self.recommendation_label.setWordWrap(True)
        self.status = QLabel("")
        self.status.setObjectName("InlineStatus")
        recommendation_row = QHBoxLayout()
        recommendation_row.setContentsMargins(0, 0, 0, 0)
        recommendation_row.setSpacing(12)
        recommendation_row.addWidget(self.recommendation_label, 1)
        recommendation_row.addWidget(
            self.status, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addLayout(recommendation_row)

        self.filter_group = QWidget()
        filters = QHBoxLayout(self.filter_group)
        filters.setContentsMargins(0, 0, 0, 0)
        filters.setSpacing(10)
        self.status_filter = chip_combo()
        self.extension_filter = chip_combo()
        self.select_all = QCheckBox("全选当前页")
        self.select_all_matching = QPushButton("选择全部阻断项")
        self.select_all_matching.setToolTip("选择当前阻断项列表的全部记录，可跨页操作")
        self.clear_selection = QPushButton("取消全部选择")
        self.clear_selection.setToolTip("清除当前页面和其他页面的选择")
        self.retry_button = QPushButton("重新尝试")
        self.exclude_button = QPushButton("从当前索引范围排除")
        self.restore_button = QPushButton("恢复纳入并重试")
        self.force_complete_button = QPushButton("保底完成并开放搜索")
        self.export_button = QPushButton("导出全部明细")
        self.open_log_button = QPushButton("打开日志目录")
        self.exclusion_progress = QProgressBar()
        self.exclusion_progress.setTextVisible(False)
        self.exclusion_progress.setFixedWidth(150)
        self.exclusion_progress.setVisible(False)
        self.cancel_exclusion_button = QPushButton("取消索引更新")
        self.cancel_exclusion_button.setVisible(False)
        self.force_complete_button.setObjectName("PrimaryButton")
        self.retry_button.clicked.connect(self._emit_retry)
        self.exclude_button.clicked.connect(self._emit_exclude)
        self.restore_button.clicked.connect(self._emit_restore)
        self.force_complete_button.clicked.connect(self._emit_force_complete)
        self.cancel_exclusion_button.clicked.connect(
            self.cancel_exclusion_requested.emit
        )
        self.export_button.clicked.connect(self._emit_export)
        self.open_log_button.clicked.connect(self.open_log_dir)
        filters.addWidget(self.subtitle)
        filters.addSpacing(12)
        filters.addWidget(self.status_filter)
        filters.addWidget(self.extension_filter)
        filters.addWidget(self.select_all)
        filters.addWidget(self.select_all_matching)
        filters.addWidget(self.clear_selection)
        self.filter_group.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )

        self.action_group = QWidget()
        actions = QHBoxLayout(self.action_group)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(10)
        actions.addWidget(self.exclusion_progress)
        actions.addWidget(self.cancel_exclusion_button)
        actions.addWidget(self.retry_button)
        actions.addWidget(self.exclude_button)
        actions.addWidget(self.restore_button)
        actions.addWidget(self.force_complete_button)
        actions.addSpacing(10)
        action_separator = QFrame()
        action_separator.setObjectName("FailedActionSeparator")
        action_separator.setFrameShape(QFrame.Shape.VLine)
        action_separator.setFrameShadow(QFrame.Shadow.Sunken)
        actions.addWidget(action_separator)
        actions.addWidget(self.export_button)
        actions.addWidget(self.open_log_button)
        self.action_group.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )

        self.action_toolbar = QFrame()
        self.action_toolbar.setObjectName("FailedActionToolbar")
        self.action_toolbar_layout = QGridLayout(self.action_toolbar)
        self.action_toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self.action_toolbar_layout.setHorizontalSpacing(16)
        self.action_toolbar_layout.setVerticalSpacing(10)
        self.action_toolbar_layout.setColumnStretch(0, 1)
        self.action_toolbar_layout.setColumnStretch(1, 0)
        self._toolbar_compact: bool | None = None
        self._set_action_toolbar_compact(True)
        layout.addWidget(self.action_toolbar)

        self.table = QTableWidget(0, 11)
        self.table.setObjectName("FailedTable")
        self.table.setHorizontalHeaderLabels(
            [
                "选择",
                "文件名",
                "路径",
                "类型",
                "状态",
                "任务阶段",
                "安全游标",
                "原因",
                "恢复建议",
                "最后尝试",
                "操作",
            ]
        )
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(10, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)

        self.page_toolbar = QWidget()
        page_layout = QHBoxLayout(self.page_toolbar)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        self.previous_page_button = QPushButton("上一页")
        self.next_page_button = QPushButton("下一页")
        self.page_label = QLabel("第 1/1 页")
        self.page_label.setObjectName("MutedText")
        self.previous_page_button.clicked.connect(lambda: self._change_page(-1))
        self.next_page_button.clicked.connect(lambda: self._change_page(1))
        page_layout.addStretch(1)
        page_layout.addWidget(self.previous_page_button)
        page_layout.addWidget(self.page_label)
        page_layout.addWidget(self.next_page_button)
        layout.addWidget(self.page_toolbar)
        self.scope_tabs.currentChanged.connect(self._scope_changed)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        self.extension_filter.currentIndexChanged.connect(self.apply_filters)
        self.select_all.toggled.connect(self._select_visible)
        self.select_all_matching.clicked.connect(self._select_all_matching)
        self.clear_selection.clicked.connect(self._clear_all_selection)
        self.table.itemChanged.connect(self._table_item_changed)

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        if hasattr(self, "action_toolbar_layout"):
            self._set_action_toolbar_compact(self._toolbar_should_compact())

    def _toolbar_should_compact(self) -> bool:
        available_width = max(0, self.width() - 48)
        required_width = (
            self.filter_group.sizeHint().width()
            + self.action_group.sizeHint().width()
            + self.action_toolbar_layout.horizontalSpacing()
        )
        return available_width < required_width

    def _set_action_toolbar_compact(self, compact: bool) -> None:
        compact = bool(compact)
        if self._toolbar_compact == compact:
            return
        self.action_toolbar_layout.removeWidget(self.filter_group)
        self.action_toolbar_layout.removeWidget(self.action_group)
        if compact:
            self.action_toolbar_layout.addWidget(
                self.filter_group,
                0,
                0,
                1,
                2,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self.action_toolbar_layout.addWidget(
                self.action_group,
                1,
                0,
                1,
                2,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        else:
            self.action_toolbar_layout.addWidget(
                self.filter_group,
                0,
                0,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )
            self.action_toolbar_layout.addWidget(
                self.action_group,
                0,
                1,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
            )
        self._toolbar_compact = compact
        self.action_toolbar.updateGeometry()

    def set_rows(
        self,
        blocking: list[object],
        excluded: list[object] | None = None,
        metadata: list[object] | None = None,
        *,
        totals: dict[str, int] | None = None,
        all_blocking_ids: list[int] | None = None,
    ) -> None:
        self.rows_by_scope = {
            "blocking": [self._normalize_row(row, "blocking") for row in blocking],
            "excluded": [self._normalize_row(row, "excluded") for row in (excluded or [])],
            "metadata": [self._normalize_row(row, "metadata") for row in (metadata or [])],
        }
        self.total_by_scope = {
            scope: max(
                len(self.rows_by_scope[scope]),
                int((totals or {}).get(scope, len(self.rows_by_scope[scope]))),
            )
            for scope in ("blocking", "excluded", "metadata")
        }
        self.all_blocking_ids = {
            int(value)
            for value in (
                all_blocking_ids
                if all_blocking_ids is not None
                else [row["id"] for row in self.rows_by_scope["blocking"]]
            )
        }
        excluded_ids = {
            int(row["id"])
            for row in self.rows_by_scope["excluded"]
            if row.get("id") is not None
        }
        self.all_blocking_ids.difference_update(excluded_ids)
        self.selected_ids_by_scope["blocking"].difference_update(excluded_ids)
        for scope in ("blocking", "excluded", "metadata"):
            available_ids = {
                int(row["id"])
                for row in self.rows_by_scope[scope]
                if row.get("id") is not None
            }
            if scope == "blocking":
                available_ids.update(self.all_blocking_ids)
                if self.all_matching_selected[scope]:
                    self.selected_ids_by_scope[scope] = set(
                        self.all_blocking_ids
                    )
                else:
                    self.selected_ids_by_scope[scope].intersection_update(
                        available_ids
                    )
            else:
                self.selected_ids_by_scope[scope].intersection_update(
                    available_ids
                )
            if (
                not self.selected_ids_by_scope[scope]
                and not self.all_matching_selected[scope]
            ):
                self.all_matching_selected[scope] = False
        for scope in self.total_by_scope:
            last_page = self._page_count(scope) - 1
            self.page_by_scope[scope] = min(self.page_by_scope[scope], max(0, last_page))
        self.scope_tabs.setTabText(0, f"阻断项 {self.total_by_scope['blocking']}")
        self.scope_tabs.setTabText(1, f"已人工排除 {self.total_by_scope['excluded']}")
        self.scope_tabs.setTabText(2, f"仅元数据完成 {self.total_by_scope['metadata']}")
        self._scope_changed(self.scope_tabs.currentIndex())

    def _page_count(self, scope: str) -> int:
        total = max(0, int(self.total_by_scope.get(scope, 0)))
        return max(1, (total + self.page_size - 1) // self.page_size)

    def page_offset(self, scope: str) -> int:
        return max(0, int(self.page_by_scope.get(scope, 0))) * self.page_size

    def _change_page(self, delta: int) -> None:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        current = int(self.page_by_scope.get(scope, 0))
        target = max(0, min(current + int(delta), self._page_count(scope) - 1))
        if target != current:
            self.page_by_scope[scope] = target
            self.refresh_requested.emit()

    @staticmethod
    def _normalize_row(row: object, scope: str) -> dict[str, object]:
        data = dict(row)
        if scope == "excluded":
            data["exclusion_id"] = data.get("id")
            data["id"] = data.get("file_id")
            data["parse_status"] = data.get("current_parse_status") or data.get("parse_status")
            data["parse_error_code"] = data.get("current_error_code") or data.get("parse_error_code")
            data["parse_error_message"] = data.get("current_error_message") or data.get("parse_error_message")
            data["indexed_at"] = data.get("created_at") or data.get("indexed_at")
            data["recovery_advice"] = "恢复纳入后重新解析"
        data.setdefault("filename", Path(str(data.get("path") or "")).name)
        for key in (
            "path",
            "extension",
            "parse_status",
            "parse_error_code",
            "parse_error_message",
            "parse_diagnostics_json",
            "parser_name",
            "indexed_at",
            "progress_phase",
            "progress_cursor",
            "recovery_advice",
            "reason",
        ):
            data.setdefault(key, "")
        data["scope_category"] = scope
        return data

    def _scope_changed(self, index: int) -> None:
        scope = ("blocking", "excluded", "metadata")[max(0, min(index, 2))]
        self.rows = self.rows_by_scope[scope]
        total = self.total_by_scope.get(scope, len(self.rows))
        self.subtitle.setText(
            f"{total} 个阻断项"
            if scope == "blocking"
            else f"{total} 个已人工排除文件"
            if scope == "excluded"
            else f"{total} 个仅元数据完成文件"
        )
        self.subtitle.setProperty(
            "state",
            "error"
            if (
                scope == "blocking"
                and total
                and not self.index_running
                and not self.snapshot_loading
                and bool(self.readiness.get("failure_alert_ready", True))
            )
            else "",
        )
        self.subtitle.style().unpolish(self.subtitle)
        self.subtitle.style().polish(self.subtitle)
        current_page = int(self.page_by_scope.get(scope, 0)) + 1
        self.page_label.setText(f"第 {current_page}/{self._page_count(scope)} 页")
        self.previous_page_button.setEnabled(current_page > 1)
        self.next_page_button.setEnabled(current_page < self._page_count(scope))
        self.status_filter.blockSignals(True)
        self.extension_filter.blockSignals(True)
        self.status_filter.clear()
        self.extension_filter.clear()
        self.status_filter.addItem("全部原因", "")
        self.extension_filter.addItem("全部格式", "")
        for value in sorted({str(row["parse_status"]) for row in self.rows}):
            self.status_filter.addItem(value, value)
        for value in sorted({str(row["extension"] or "") for row in self.rows}):
            self.extension_filter.addItem(value or "<无扩展名>", value)
        self.status_filter.blockSignals(False)
        self.extension_filter.blockSignals(False)
        selectable = scope in {"blocking", "excluded"}
        self.select_all.setVisible(selectable)
        self.select_all_matching.setVisible(scope == "blocking")
        self.clear_selection.setVisible(selectable)
        self.select_all.blockSignals(True)
        self.select_all.setChecked(False)
        self.select_all.blockSignals(False)
        self.retry_button.setVisible(scope == "blocking")
        self.exclude_button.setVisible(scope == "blocking")
        self.restore_button.setVisible(scope == "excluded")
        self.force_complete_button.setVisible(scope == "blocking")
        self._update_action_states()
        self.apply_filters()
        self.action_group.layout().activate()
        self._set_action_toolbar_compact(self._toolbar_should_compact())

    def set_index_running(self, running: bool) -> None:
        self.index_running = bool(running)
        if running:
            self.risk_banner.clear()
            self.risk_banner.setVisible(False)
        self._update_action_states()

    def set_search_running(self, running: bool) -> None:
        self.search_running = bool(running)
        self._update_action_states()
        self._update_scope_alert_state()

    def set_readiness(self, readiness: dict[str, object]) -> None:
        self.readiness = dict(readiness)
        self._update_action_states()

    def set_snapshot_loading(self, loading: bool) -> None:
        self.snapshot_loading = bool(loading)
        self._update_action_states()
        self._update_scope_alert_state()

    def _update_scope_alert_state(self) -> None:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        total = self.total_by_scope.get(scope, len(self.rows_by_scope[scope]))
        visible = bool(
            scope == "blocking"
            and total
            and not self.index_running
            and not self.snapshot_loading
            and bool(self.readiness.get("failure_alert_ready", True))
            and str(self.readiness.get("background_task_state") or "")
            not in {"running", "finishing"}
        )
        self.subtitle.setProperty("state", "error" if visible else "")
        self.subtitle.style().unpolish(self.subtitle)
        self.subtitle.style().polish(self.subtitle)

    def set_database_migration_running(self, running: bool) -> None:
        self.database_migration_running = bool(running)
        self._update_action_states()

    def set_exclusion_running(self, running: bool) -> None:
        self.exclusion_running = bool(running)
        self.exclusion_progress.setVisible(running)
        self.cancel_exclusion_button.setVisible(running)
        self.table.setEnabled(not running)
        self._update_action_states()
        if not running:
            self.exclusion_progress.setRange(0, 1)
            self.exclusion_progress.setValue(0)

    def set_exclusion_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        phase_label = str(payload.get("phase_label") or "正在后台处理")
        processed = int(payload.get("processed_files") or 0)
        total = int(payload.get("total_files") or 0)
        elapsed = int(payload.get("elapsed_seconds") or 0)
        large_fts = bool(payload.get("large_fts_operation"))
        can_cancel = bool(payload.get("can_cancel", True))
        if total > 0 and str(payload.get("stage") or "") == "recording_exclusions":
            self.exclusion_progress.setRange(0, max(total, 1))
            self.exclusion_progress.setValue(min(processed, total))
            count_text = f" · {processed}/{total} 个文件"
        else:
            self.exclusion_progress.setRange(0, 0)
            count_text = ""
        fts_text = " · 正在执行大型 FTS 操作" if large_fts else ""
        elapsed_text = f" · 已用时 {elapsed} 秒" if elapsed > 0 else ""
        self.set_status(
            f"正在更新搜索索引... · {phase_label}{count_text}{fts_text}{elapsed_text}"
        )
        self.cancel_exclusion_button.setEnabled(can_cancel)
        self.cancel_exclusion_button.setToolTip(
            "取消后将回滚本次后台更新"
            if can_cancel
            else "正在完成事务安全边界，请稍候"
        )

    def set_exclusion_cancel_pending(self) -> None:
        self.cancel_exclusion_button.setEnabled(False)
        self.set_status("正在取消索引更新并回滚，请稍候...")

    def _update_action_states(self) -> None:
        busy = (
            self.index_running
            or self.search_running
            or self.exclusion_running
            or self.database_migration_running
            or self.snapshot_loading
            or bool(self.readiness.get("candidate_live"))
        )
        scope = self.scope_tabs.currentIndex()
        blockers = self.total_by_scope.get("blocking", len(self.rows_by_scope["blocking"]))
        ready = bool(self.readiness.get("ready", blockers == 0))
        has_selection = bool(self.selected_file_ids())
        self.retry_button.setEnabled(
            scope == 0 and blockers > 0 and not ready and not busy
        )
        # Exclusion is scoped to the selected files. A directory scan in
        # another root must not globally disable this action; the database
        # worker settles only real tasks belonging to the selection.
        self.exclude_button.setEnabled(
            scope == 0
            and blockers > 0
            and not ready
            and not busy
        )
        self.restore_button.setEnabled(scope == 1 and not busy)
        if busy:
            retry_reason = "当前任务正在处理，请等待完成"
        elif ready:
            retry_reason = "当前索引已经就绪，请从索引管理页发起更新"
        elif blockers == 0:
            retry_reason = "当前没有需要重新解析的阻断文件"
        elif has_selection:
            retry_reason = "只重新解析当前选中的阻断文件"
        else:
            retry_reason = "请先勾选至少一个未成功索引项"
        self.retry_button.setToolTip(retry_reason)
        self.exclude_button.setToolTip(
            "排除选中的阻断文件；不会删除源文件"
            if has_selection
            else "请先勾选至少一个阻断文件"
        )
        self.select_all.setEnabled(
            not self.exclusion_running
            and not bool(self.readiness.get("candidate_live"))
        )
        self.table.setEnabled(
            not self.exclusion_running
            and not bool(self.readiness.get("candidate_live"))
        )
        self._update_force_complete_state()
        self._update_recommendation()
        self._update_risk_banner()

    def _update_force_complete_state(self) -> None:
        busy = (
            self.index_running
            or self.search_running
            or self.exclusion_running
            or self.database_migration_running
            or self.snapshot_loading
            or bool(self.readiness.get("candidate_live"))
        )
        blockers = self.total_by_scope.get("blocking", len(self.rows_by_scope["blocking"]))
        ready = bool(self.readiness.get("ready", blockers == 0))
        repairable = bool(self.readiness.get("repairable", blockers > 0))
        reasons = [
            str(value) for value in self.readiness.get("not_ready_reasons", [])
        ]
        has_unready_reason = bool(reasons)
        enabled = (
            self.scope_tabs.currentIndex() == 0
            and not ready
            and blockers > 0
            and not busy
        )
        self.force_complete_button.setEnabled(enabled)
        if self.database_migration_running:
            tooltip = "数据库正在迁移，完成后可使用保底完成"
        elif self.exclusion_running:
            tooltip = "搜索索引正在后台更新，完成后可使用保底完成"
        elif bool(self.readiness.get("candidate_live")):
            tooltip = "候选索引正在构建，当前显示实时失败项；完成后可处理"
        elif self.index_running:
            tooltip = "索引任务正在运行，结束后可使用保底完成"
        elif self.search_running:
            tooltip = "搜索正在进行，结束后可使用保底完成"
        elif blockers:
            tooltip = f"仍有 {blockers} 个阻断项，可执行最终重试并排除后开放搜索"
        elif not blockers and repairable:
            labels = {
                "unfinished_tasks": "残留解析任务",
                "content_fts_dirty": "全文索引待发布",
                "full_batch_incomplete": "完整批次未收敛",
                "unready_root": "搜索范围未就绪",
                "unpublished_candidate": "候选索引未发布",
            }
            detail = "、".join(labels.get(reason, reason) for reason in reasons)
            tooltip = f"没有可选择的未成功索引项：{detail or '索引状态未收敛'}"
        elif ready:
            tooltip = "当前索引已经就绪"
        elif not blockers and not ready and not repairable:
            tooltip = "当前没有未成功索引项，无法执行此操作"
        else:
            tooltip = "当前状态暂不可修复，请查看未就绪原因"
        self.force_complete_button.setToolTip(tooltip)

    def _update_risk_banner(self) -> None:
        blockers = int(
            self.total_by_scope.get(
                "blocking",
                len(self.rows_by_scope["blocking"]),
            )
        )
        ready = bool(self.readiness.get("ready", blockers == 0))
        if (
            self.index_running
            or self.snapshot_loading
            or str(self.readiness.get("background_task_state") or "")
            in {"running", "finishing"}
        ):
            self.risk_banner.clear()
            self.risk_banner.setVisible(False)
            return
        if blockers > 0 and not self.snapshot_loading and bool(
            self.readiness.get("failure_alert_ready", True)
        ):
            self.risk_banner.setText(
                f"需要处理：当前有 {blockers:,} 个文件未成功索引。"
                "请从下方三种方式中选择一种；源文件不会被自动删除。"
            )
            state = "error"
            visible = True
        elif not ready:
            self.risk_banner.setText(
                "索引状态尚未收敛：文件可能已处理完成，但搜索索引仍需修复。"
                "可使用“保底完成并开放搜索”修复并开放搜索。"
            )
            state = "warning"
            visible = True
        else:
            self.risk_banner.clear()
            state = ""
            visible = False
        self.risk_banner.setProperty("state", state)
        self.risk_banner.setVisible(visible)
        self.risk_banner.style().unpolish(self.risk_banner)
        self.risk_banner.style().polish(self.risk_banner)

    def _update_recommendation(self) -> None:
        blockers = self.total_by_scope.get("blocking", len(self.rows_by_scope["blocking"]))
        ready = bool(self.readiness.get("ready", blockers == 0))
        if self.database_migration_running:
            text = "正在迁移数据库，完成后可继续操作。"
        elif self.exclusion_running:
            text = "正在更新搜索索引，完成后可继续操作。"
        elif bool(self.readiness.get("candidate_live_failures")):
            text = (
                f"候选库已实时记录 {blockers:,} 个未成功索引项；"
                "当前搜索仍使用上一版已发布数据库。"
            )
        elif self.index_running:
            text = "正在重新解析文件，完成后可继续操作。"
        elif ready:
            text = "当前索引已完成，可以正常搜索，无需处理。"
        elif blockers:
            text = (
                f"仍有 {blockers} 个文件无法解析。建议先重新尝试；确认无法修复后再排除，"
                "或使用保底完成。"
            )
        else:
            text = (
                "没有失败文件，但索引发布状态未完成，请使用“保底完成并开放搜索”修复。"
            )
        self.recommendation_label.setText(text)
        self.recommendation_label.setProperty(
            "state",
            "error"
            if (
                blockers > 0
                and not self.snapshot_loading
                and bool(self.readiness.get("failure_alert_ready", True))
            )
            else "warning"
            if not ready
            else "ready",
        )
        self.recommendation_label.style().unpolish(self.recommendation_label)
        self.recommendation_label.style().polish(self.recommendation_label)

    def visible_rows(self) -> list[dict[str, object]]:
        status = str(self.status_filter.currentData() or "")
        extension = str(self.extension_filter.currentData() or "")
        visible = []
        for row in self.rows:
            if status and row["parse_status"] != status:
                continue
            if extension and (row["extension"] or "") != extension:
                continue
            visible.append(row)
        return visible

    def _emit_export(self) -> None:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        self.export_requested.emit(
            scope,
            str(self.status_filter.currentData() or ""),
            str(self.extension_filter.currentData() or ""),
        )

    def apply_filters(self) -> None:
        rows = self.visible_rows()
        self.table.blockSignals(True)
        self.table.setRowCount(len(rows))
        selectable = self.scope_tabs.currentIndex() in {0, 1}
        for row_index, row in enumerate(rows):
            path = str(row["path"])
            checkbox = QTableWidgetItem("")
            checkbox.setData(Qt.ItemDataRole.UserRole, int(row["id"]))
            if selectable:
                checkbox.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsUserCheckable
                    | Qt.ItemFlag.ItemIsSelectable
                )
                scope = ("blocking", "excluded")[
                    self.scope_tabs.currentIndex() == 1
                ]
                selected = int(row["id"]) in self.selected_ids_by_scope.get(
                    scope, set()
                ) or (
                    scope == "blocking"
                    and self.all_matching_selected[scope]
                    and int(row["id"]) in self.all_blocking_ids
                )
                checkbox.setCheckState(
                    Qt.CheckState.Checked if selected else Qt.CheckState.Unchecked
                )
            else:
                checkbox.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.table.setItem(row_index, 0, checkbox)
            reason = (
                str(row.get("reason") or "")
                if row.get("scope_category") == "excluded"
                else str(row.get("parse_error_message") or row.get("parse_status") or "")
            )
            values = [
                str(row.get("filename") or Path(path).name),
                path,
                str(row["extension"] or ""),
                "人工排除" if row.get("scope_category") == "excluded" else str(row["parse_status"] or ""),
                str(row["progress_phase"] or ""),
                str(row["progress_cursor"] or ""),
                reason,
                str(row["recovery_advice"] or ""),
                str(row["indexed_at"] or ""),
            ]
            for col, value in enumerate(values, start=1):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row_index, col, item)
            action_button = QPushButton("打开文件夹")
            action_button.clicked.connect(lambda _checked=False, value=path: self.open_folder_requested.emit(value))
            self.table.setCellWidget(row_index, 10, action_button)
        self.table.blockSignals(False)
        self._update_action_states()

    def _sync_visible_selection(self) -> None:
        """Refresh visible row checkboxes from the cross-page selection state."""

        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        selected_ids = self.selected_ids_by_scope.get(scope, set())
        all_matching = (
            scope == "blocking" and self.all_matching_selected[scope]
        )
        self.table.blockSignals(True)
        try:
            for row_index in range(self.table.rowCount()):
                item = self.table.item(row_index, 0)
                if item is None:
                    continue
                file_id = int(item.data(Qt.ItemDataRole.UserRole) or 0)
                checked = file_id in selected_ids or (
                    all_matching and file_id in self.all_blocking_ids
                )
                item.setCheckState(
                    Qt.CheckState.Checked
                    if checked
                    else Qt.CheckState.Unchecked
                )
        finally:
            self.table.blockSignals(False)

    def _table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == 0:
            scope = ("blocking", "excluded", "metadata")[
                max(0, min(self.scope_tabs.currentIndex(), 2))
            ]
            row_id = item.data(Qt.ItemDataRole.UserRole)
            if row_id is not None:
                if item.checkState() == Qt.CheckState.Checked:
                    self.selected_ids_by_scope[scope].add(int(row_id))
                else:
                    self.selected_ids_by_scope[scope].discard(int(row_id))
            self.all_matching_selected[scope] = False
            self._update_action_states()

    def selected_file_ids(self) -> list[int]:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        selected = set(self.selected_ids_by_scope.get(scope, set()))
        if self.all_matching_selected.get(scope) and scope == "blocking":
            selected.update(self.all_blocking_ids)
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.add(int(item.data(Qt.ItemDataRole.UserRole)))
        return sorted(selected)

    def selection_is_all_matching(self) -> bool:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        return bool(self.all_matching_selected.get(scope))

    def selection_filters(self) -> tuple[str, str]:
        return (
            str(self.status_filter.currentData() or ""),
            str(self.extension_filter.currentData() or ""),
        )

    def _select_visible(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)
        if not checked:
            scope = ("blocking", "excluded", "metadata")[
                max(0, min(self.scope_tabs.currentIndex(), 2))
            ]
            self.all_matching_selected[scope] = False

    def _select_all_matching(self) -> None:
        scope = ("blocking", "excluded", "metadata")[
            max(0, min(self.scope_tabs.currentIndex(), 2))
        ]
        if scope != "blocking":
            return
        self.all_matching_selected[scope] = True
        self.selected_ids_by_scope[scope] = set(self.all_blocking_ids)
        self.select_all.blockSignals(True)
        self.select_all.setChecked(True)
        self.select_all.blockSignals(False)
        self._sync_visible_selection()
        self.set_status(f"已选择全部 {len(self.all_blocking_ids):,} 个阻断项")
        self._update_action_states()

    def _clear_all_selection(self) -> None:
        for scope in self.selected_ids_by_scope:
            self.selected_ids_by_scope[scope].clear()
            self.all_matching_selected[scope] = False
        self.select_all.blockSignals(True)
        self.select_all.setChecked(False)
        self.select_all.blockSignals(False)
        self.apply_filters()

    def _emit_exclude(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids:
            QMessageBox.information(
                self,
                "需要选择未成功索引项",
                "请先勾选需要处理的未成功索引项。",
            )
            return
        self.exclude_requested.emit(file_ids)

    def _emit_retry(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids:
            QMessageBox.information(
                self,
                "需要选择未成功索引项",
                "请先勾选需要处理的未成功索引项。",
            )
            return
        self.retry_requested.emit(file_ids)

    def _emit_restore(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids:
            QMessageBox.information(
                self,
                "需要选择未成功索引项",
                "请先勾选需要处理的未成功索引项。",
            )
            return
        self.restore_requested.emit(file_ids)

    def _emit_force_complete(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids and not self.selection_is_all_matching():
            QMessageBox.information(
                self,
                "需要选择未成功索引项",
                "请先勾选需要处理的未成功索引项。",
            )
            return
        self.force_complete_requested.emit()

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def open_log_dir(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        open_file(LOG_DIR)


class SettingsPage(QWidget):
    save_requested = Signal(object)
    database_migration_requested = Signal(str)
    database_migration_cancel_requested = Signal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll_host = QWidget()
        scroll_host_layout = QHBoxLayout(scroll_host)
        scroll_host_layout.setContentsMargins(0, 0, 0, 0)
        scroll_host_layout.addStretch(1)
        host = QWidget()
        host.setObjectName("SettingsContent")
        host.setMaximumWidth(1100)
        self.settings_content = host
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(12)
        scroll_host_layout.addWidget(host, 1)
        scroll_host_layout.addStretch(1)

        self.current_database_path = Path()
        self.database_migration_running = False
        self.database_path_value = QLineEdit("-")
        self.database_path_value.setReadOnly(True)
        self.database_path_value.setObjectName("DatabasePath")
        self.database_size_value = QLabel("-")
        self.database_schema_value = QLabel("-")
        self.database_current_space_value = QLabel("-")
        self.database_target = QLineEdit()
        self.database_target.setReadOnly(True)
        self.database_target.setPlaceholderText("选择本机固定磁盘上的目标目录")
        self.database_target_space_value = QLabel("尚未选择目标目录")
        self.database_choose_button = QPushButton("选择目录")
        self.database_migrate_button = QPushButton("迁移数据库")
        self.database_migrate_button.setObjectName("PrimaryButton")
        self.database_cancel_button = QPushButton("取消迁移")
        self.database_cancel_button.setVisible(False)
        self.database_copy_button = QPushButton("复制路径")
        self.database_migration_status = QLabel("数据库未迁移")
        self.database_migration_status.setWordWrap(True)
        self.database_migration_progress = QProgressBar()
        self.database_migration_progress.setRange(0, 100)
        self.database_migration_progress.setValue(0)
        self.database_migration_progress.setVisible(False)
        database_progress = QWidget()
        database_progress_layout = QVBoxLayout(database_progress)
        database_progress_layout.setContentsMargins(0, 0, 0, 0)
        database_progress_layout.setSpacing(6)
        database_progress_layout.addWidget(self.database_migration_status)
        database_progress_layout.addWidget(self.database_migration_progress)
        database_card = QFrame()
        database_card.setObjectName("SettingsCard")
        database_layout = QVBoxLayout(database_card)
        database_layout.setContentsMargins(16, 14, 16, 14)
        database_layout.setSpacing(9)
        database_heading = QLabel("索引数据库")
        database_heading.setObjectName("SectionTitle")
        database_layout.addWidget(database_heading)

        database_path_header = QHBoxLayout()
        database_path_header.addWidget(QLabel("当前索引存储目录"))
        database_path_header.addStretch(1)
        database_path_header.addWidget(self.database_copy_button)
        database_layout.addLayout(database_path_header)
        database_layout.addWidget(self.database_path_value)

        database_summary = QWidget()
        database_summary_layout = QGridLayout(database_summary)
        database_summary_layout.setContentsMargins(0, 2, 0, 2)
        database_summary_layout.setHorizontalSpacing(24)
        summary_items = (
            ("索引存储大小", self.database_size_value),
            ("数据库版本", self.database_schema_value),
            ("当前磁盘", self.database_current_space_value),
        )
        for column, (caption, value) in enumerate(summary_items):
            caption_label = QLabel(caption)
            caption_label.setObjectName("MutedText")
            database_summary_layout.addWidget(caption_label, 0, column)
            database_summary_layout.addWidget(value, 1, column)
            database_summary_layout.setColumnStretch(column, 1)
        database_layout.addWidget(database_summary)

        target_caption = QLabel("目标位置")
        target_caption.setObjectName("MutedText")
        database_layout.addWidget(target_caption)
        database_target_row = QHBoxLayout()
        database_target_row.setSpacing(8)
        database_target_row.addWidget(self.database_target, 1)
        database_target_row.addWidget(self.database_choose_button)
        database_target_row.addWidget(self.database_migrate_button)
        database_target_row.addWidget(self.database_cancel_button)
        database_layout.addLayout(database_target_row)
        self.database_target_space_value.setObjectName("InlineStatus")
        database_layout.addWidget(self.database_target_space_value)
        database_layout.addWidget(database_progress)
        host_layout.addWidget(database_card)

        self.page_size = QSpinBox()
        self.page_size.setRange(10, 100)
        self.page_size.setValue(settings.page_size)
        self.auto_search = QCheckBox("输入后自动搜索")
        self.auto_search.setChecked(settings.auto_search)
        self.search_delay = QSpinBox()
        self.search_delay.setRange(100, 2000)
        self.search_delay.setSuffix(" ms")
        self.search_delay.setValue(settings.search_debounce_ms)
        self.max_results = QSpinBox()
        self.max_results.setRange(100, 100_000)
        self.max_results.setValue(settings.max_results)
        self.search_history = QCheckBox("保存搜索历史")
        self.search_history.setChecked(settings.save_search_history)
        host_layout.addWidget(
            settings_card(
                "搜索体验",
                [
                    ("", self.auto_search),
                    ("自动搜索延迟", self.search_delay),
                    ("", self.search_history),
                    ("每页结果数量", self.page_size),
                    ("最大展示结果", self.max_results),
                ],
            )
        )

        self.enable_ocr = QCheckBox("启用 OCR")
        self.enable_ocr.setChecked(settings.enable_ocr)
        self.ocr_images = QCheckBox("索引图片 OCR")
        self.ocr_images.setChecked(settings.ocr_images)
        self.ocr_pdf = QCheckBox("索引扫描 PDF OCR")
        self.ocr_pdf.setChecked(settings.ocr_scanned_pdf)
        self.ocr_workers = QSpinBox()
        self.ocr_workers.setRange(1, 4)
        self.ocr_workers.setValue(settings.ocr_workers)
        self.ocr_min_confidence = QDoubleSpinBox()
        self.ocr_min_confidence.setRange(0.0, 1.0)
        self.ocr_min_confidence.setSingleStep(0.05)
        self.ocr_min_confidence.setDecimals(2)
        self.ocr_min_confidence.setValue(settings.ocr_min_confidence)
        self.parser_workers = QSpinBox()
        self.parser_workers.setRange(1, 16)
        self.parser_workers.setValue(settings.parser_workers)
        self.slow_file_workers = QSpinBox()
        self.slow_file_workers.setRange(1, 4)
        self.slow_file_workers.setValue(settings.slow_file_workers)
        self.process_parser_workers = QSpinBox()
        self.process_parser_workers.setRange(1, 4)
        self.process_parser_workers.setValue(settings.process_parser_workers)
        self.large_office_threshold_mb = QSpinBox()
        self.large_office_threshold_mb.setRange(1, 1024)
        self.large_office_threshold_mb.setValue(
            max(1, settings.large_office_process_min_bytes // (1024 * 1024))
        )
        self.write_batch_size = QSpinBox()
        self.write_batch_size.setRange(1, 200)
        self.write_batch_size.setValue(settings.index_write_batch_size)
        self.min_ocr_pixels = QSpinBox()
        self.min_ocr_pixels.setRange(0, 2_000_000)
        self.min_ocr_pixels.setValue(settings.min_ocr_image_pixels)
        self.max_ocr_side = QSpinBox()
        self.max_ocr_side.setRange(800, 6000)
        self.max_ocr_side.setValue(settings.max_ocr_image_side)
        host_layout.addWidget(
            settings_card(
                "OCR",
                [
                    ("", self.enable_ocr),
                    ("", self.ocr_images),
                    ("", self.ocr_pdf),
                ],
            )
        )

        self.monitor_changes = QCheckBox("检测文件变化并提示更新")
        self.monitor_changes.setChecked(settings.monitor_file_changes)
        self.hidden_files = QCheckBox("索引隐藏文件")
        self.hidden_files.setChecked(settings.include_hidden_files)
        self.exclude_dirs = QTextEdit("\n".join(settings.excluded_dirs))
        self.exclude_dirs.setMinimumHeight(90)
        self.performance_preset = QComboBox()
        self.performance_preset.addItem("平衡（推荐）", "balanced")
        self.performance_preset.addItem("低资源", "low_resource")
        self.performance_preset.addItem("最快完成", "fastest")
        preset_index = self.performance_preset.findData(settings.index_performance_preset)
        self.performance_preset.setCurrentIndex(preset_index if preset_index >= 0 else 0)
        host_layout.addWidget(
            settings_card(
                "索引",
                [
                    ("性能模式", self.performance_preset),
                    ("", self.monitor_changes),
                    ("", self.hidden_files),
                ],
            )
        )

        self.advanced_toggle = QPushButton("显示高级性能设置")
        self.advanced_toggle.setObjectName("DisclosureButton")
        self.advanced_toggle.setCheckable(True)
        host_layout.addWidget(self.advanced_toggle)

        self.advanced_content = QWidget()
        advanced_layout = QVBoxLayout(self.advanced_content)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(12)
        advanced_layout.addWidget(
            settings_card(
                "性能与资源",
                [
                    ("普通解析线程数", self.parser_workers),
                    ("OCR 工作线程数", self.ocr_workers),
                    ("OCR 最低置信度", self.ocr_min_confidence),
                    ("ZIP/老版 Office 慢任务线程数", self.slow_file_workers),
                    ("大型 XLSX/DOCX 进程数", self.process_parser_workers),
                    ("进程池启用阈值（MB）", self.large_office_threshold_mb),
                    ("批量写库文件数", self.write_batch_size),
                    ("小图片 OCR 跳过像素", self.min_ocr_pixels),
                    ("OCR 首轮检测边长", self.max_ocr_side),
                ],
            )
        )
        advanced_layout.addWidget(settings_card("排除目录", [("每行一个目录名", self.exclude_dirs)]))
        self.advanced_content.setVisible(False)
        host_layout.addWidget(self.advanced_content)

        self.save_button = QPushButton("保存设置")
        self.save_button.setObjectName("PrimaryButton")
        self.save_button.clicked.connect(self.apply)
        save_row = QHBoxLayout()
        save_row.addStretch(1)
        save_row.addWidget(self.save_button)
        host_layout.addLayout(save_row)
        host_layout.addStretch(1)
        scroll.setWidget(scroll_host)
        layout.addWidget(scroll, 1)

        self.auto_search.toggled.connect(self.search_delay.setEnabled)
        self.search_delay.setEnabled(self.auto_search.isChecked())
        self.enable_ocr.toggled.connect(self._update_ocr_controls)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.database_choose_button.clicked.connect(
            self.choose_database_directory
        )
        self.database_copy_button.clicked.connect(self.copy_database_path)
        self.database_migrate_button.clicked.connect(
            self.emit_database_migration
        )
        self.database_cancel_button.clicked.connect(
            self.database_migration_cancel_requested.emit
        )
        self._update_ocr_controls(self.enable_ocr.isChecked())
        self._refresh_database_actions()

    def set_database_info(self, path: Path) -> None:
        resolved = Path(path).resolve()
        self.current_database_path = (
            resolved.parent if resolved.suffix.lower() == ".db" else resolved
        )
        self.database_path_value.setText(str(self.current_database_path))
        self.database_path_value.setToolTip(str(self.current_database_path))
        size = sum(
            item.stat().st_size
            for item in self.current_database_path.rglob("*")
            if item.is_file() and item.suffix.lower() in {".db", ".json"}
        ) if self.current_database_path.is_dir() else 0
        self.database_size_value.setText(format_bytes(size))
        schema = 0
        control = (
            resolved
            if resolved.is_file() and resolved.suffix.lower() == ".db"
            else self.current_database_path / "index_control.db"
        )
        if control.is_file():
            try:
                connection = sqlite3.connect(
                    f"file:{control.as_posix()}?mode=ro",
                    uri=True,
                    timeout=5,
                )
                try:
                    schema = int(
                        connection.execute("PRAGMA user_version").fetchone()[0]
                    )
                finally:
                    connection.close()
            except sqlite3.Error:
                schema = 0
        self.database_schema_value.setText(
            f"schema v{schema}" if schema else "尚未初始化"
        )
        try:
            free = shutil.disk_usage(self.current_database_path.parent).free
            self.database_current_space_value.setText(
                f"可用 {format_bytes(free)}"
            )
        except OSError:
            self.database_current_space_value.setText("无法读取磁盘空间")
        self._refresh_database_actions()

    def copy_database_path(self) -> None:
        QGuiApplication.clipboard().setText(str(self.current_database_path))
        self.database_migration_status.setText("已复制当前索引存储目录")

    def choose_database_directory(self) -> None:
        initial = (
            str(self.current_database_path.parent)
            if str(self.current_database_path)
            else ""
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "选择数据库目标目录",
            initial,
        )
        if not directory:
            return
        self.database_target.setText(str(Path(directory).resolve()))
        self._update_target_space()
        self._refresh_database_actions()

    def emit_database_migration(self) -> None:
        target = self.database_target.text().strip()
        if target and not self.database_migration_running:
            self.database_migration_requested.emit(target)

    def set_database_migration_running(self, running: bool) -> None:
        self.database_migration_running = bool(running)
        self.database_choose_button.setEnabled(not running)
        self.database_cancel_button.setVisible(running)
        self.database_cancel_button.setEnabled(running)
        self.save_button.setEnabled(not running)
        self.database_migration_progress.setVisible(running)
        if running:
            self.database_migration_progress.setValue(0)
            self.database_migration_status.setText("正在准备数据库迁移")
        self._refresh_database_actions()

    def set_database_migration_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        percent = int(payload.get("percent") or 0)
        label = str(payload.get("phase_label") or "正在迁移数据库")
        processed = int(payload.get("processed_bytes") or 0)
        total = int(payload.get("total_bytes") or 0)
        if total > 0:
            label += f" · {format_bytes(processed)} / {format_bytes(total)}"
        self.database_migration_status.setText(label)
        self.database_migration_progress.setValue(percent)

    def set_database_migration_completed(self, payload: object) -> None:
        self.database_migration_progress.setVisible(True)
        self.database_migration_progress.setValue(100)
        backup = ""
        if isinstance(payload, dict):
            backup = str(payload.get("previous_database_backup") or "")
        text = "数据库迁移完成，当前索引可以继续使用。"
        if backup:
            text += f" 回滚备份：{backup}"
        self.database_migration_status.setText(text)
        self.database_target.clear()
        self.database_target_space_value.setText("尚未选择目标目录")
        self._refresh_database_actions()

    def set_database_migration_cancel_pending(self) -> None:
        self.database_cancel_button.setEnabled(False)
        self.database_migration_status.setText("正在取消迁移并恢复原数据库")

    def set_database_migration_cancelled(self) -> None:
        self.database_migration_progress.setVisible(False)
        self.database_migration_status.setText(
            "数据库迁移已取消，程序继续使用原数据库。"
        )

    def set_database_migration_failed(self, message: str) -> None:
        self.database_migration_progress.setVisible(False)
        self.database_migration_status.setText(
            f"迁移失败：{message}。程序继续使用原数据库。"
        )

    def _update_target_space(self) -> None:
        text = self.database_target.text().strip()
        if not text:
            self.database_target_space_value.setText("尚未选择目标目录")
            return
        try:
            free = shutil.disk_usage(Path(text)).free
            current_size = (
                sum(
                    item.stat().st_size
                    for item in self.current_database_path.rglob("*")
                    if item.is_file()
                )
                if self.current_database_path.is_dir()
                else 0
            )
            required = current_size + max(
                64 * 1024 * 1024,
                current_size // 4,
            )
            status = "空间充足" if free >= required else "空间不足"
            self.database_target_space_value.setText(
                f"可用 {format_bytes(free)} · 预计需要 {format_bytes(required)} · {status}"
            )
        except OSError:
            self.database_target_space_value.setText("无法读取目标磁盘空间")

    def _refresh_database_actions(self) -> None:
        target = self.database_target.text().strip()
        target_is_different = bool(target) and (
            Path(target).resolve() != self.current_database_path.parent
        )
        self.database_migrate_button.setEnabled(
            target_is_different and not self.database_migration_running
        )

    def _toggle_advanced(self, visible: bool) -> None:
        self.advanced_content.setVisible(visible)
        self.advanced_toggle.setText("收起高级性能设置" if visible else "显示高级性能设置")

    def _update_ocr_controls(self, enabled: bool) -> None:
        for widget in (
            self.ocr_images,
            self.ocr_pdf,
            self.ocr_workers,
            self.ocr_min_confidence,
            self.min_ocr_pixels,
            self.max_ocr_side,
        ):
            widget.setEnabled(enabled)

    def apply(self) -> None:
        self.settings.page_size = int(self.page_size.value())
        self.settings.max_results = int(self.max_results.value())
        self.settings.auto_search = self.auto_search.isChecked()
        self.settings.search_debounce_ms = int(self.search_delay.value())
        self.settings.save_search_history = self.search_history.isChecked()
        self.settings.enable_ocr = self.enable_ocr.isChecked()
        self.settings.ocr_images = self.ocr_images.isChecked()
        self.settings.ocr_scanned_pdf = self.ocr_pdf.isChecked()
        self.settings.parser_workers = int(self.parser_workers.value())
        self.settings.ocr_workers = int(self.ocr_workers.value())
        self.settings.ocr_min_confidence = float(self.ocr_min_confidence.value())
        self.settings.slow_file_workers = int(self.slow_file_workers.value())
        self.settings.process_parser_workers = int(self.process_parser_workers.value())
        self.settings.large_office_process_min_bytes = int(self.large_office_threshold_mb.value()) * 1024 * 1024
        self.settings.index_write_batch_size = int(self.write_batch_size.value())
        self.settings.min_ocr_image_pixels = int(self.min_ocr_pixels.value())
        self.settings.max_ocr_image_side = int(self.max_ocr_side.value())
        self.settings.index_performance_preset = str(self.performance_preset.currentData())
        self.settings.monitor_file_changes = self.monitor_changes.isChecked()
        self.settings.include_hidden_files = self.hidden_files.isChecked()
        self.settings.excluded_dirs = [line.strip() for line in self.exclude_dirs.toPlainText().splitlines() if line.strip()]
        self.save_requested.emit(self.settings)


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    line.setFixedHeight(1)
    return line


def chip_combo() -> QComboBox:
    combo = QComboBox()
    combo.setObjectName("FilterControl")
    combo.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    return combo


def clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()


def scope_detail(stats: dict[str, int]) -> str:
    groups = {
        "PDF": {".pdf"},
        "Word": {".doc", ".docx"},
        "Excel": {".xls", ".xlsx", ".xlsm"},
        "PowerPoint": {".ppt", ".pptx"},
        "图片": {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"},
    }
    pieces = []
    for label, extensions in groups.items():
        count = sum(stats.get(ext, 0) for ext in extensions)
        if count:
            pieces.append(f"{label} {count:,}")
    return " · ".join(pieces) if pieces else "等待索引或仅有其他格式文件"


def format_elapsed(elapsed_ms: int) -> str:
    if elapsed_ms < 1_000:
        return f"{max(1, elapsed_ms)} 毫秒"
    if elapsed_ms < 60_000:
        return f"{elapsed_ms / 1_000:.1f} 秒"
    return f"{elapsed_ms / 60_000:.1f} 分钟"


def format_remaining_single(seconds: int, ready: bool) -> str:
    if not ready:
        return "正在估算…"
    seconds = max(1, int(seconds))
    if seconds < 60:
        return f"预计剩余约 {seconds} 秒"
    return f"预计剩余约 {(seconds + 59) // 60} 分钟"


def format_active_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds} 秒"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes} 分 {remainder:02d} 秒"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} 小时 {minutes:02d} 分"


def scheduler_diagnostic_label(state: str) -> str:
    return {
        "reclaiming_no_progress": "正在回收无进展任务",
        "terminating_worker": "正在终止 worker",
        "rebuilding_pool": "正在重建解析进程池",
        "checkpoint_resumed": "已从检查点恢复",
        "same_stall_retry_stopped": "同一卡点重复发生，已停止自动重试",
    }.get(str(state or ""), "")


def format_active_phase(phase: str) -> str:
    normalized = str(phase or "").strip().lower()
    if not normalized:
        return ""
    rules = (
        ("recognize_microbatch", "OCR 批量识别文字区域"),
        ("recognize_original_regions", "OCR 识别原图文字区域"),
        ("tile_detect", "OCR 分块检测文字区域"),
        ("tile_recognize", "OCR 分块识别文字"),
        ("ocr_detect", "OCR 检测文字区域"),
        ("ocr_recognize", "OCR 识别文字"),
        ("model_load", "正在加载 OCR 模型"),
        ("pdf_native", "PDF 提取可复制正文"),
        ("pdf_preview", "PDF 低分辨率预检"),
        ("region_300dpi", "PDF 300 DPI 精细识别"),
        ("region_200dpi", "PDF 200 DPI 区域识别"),
        ("legacy_cache", "检查旧版 Office 转换缓存"),
        ("legacy_convert", "转换旧版 Office 文件"),
        ("xlsx_shared", "Excel 读取共享字符串"),
        ("xlsx_sheet", "Excel 解析工作表"),
        ("zip_member", "ZIP 解析成员文件"),
    )
    for marker, label in rules:
        if marker in normalized:
            return label
    return normalized.replace("_", " ")


def format_bytes(value: int) -> str:
    value = max(0, int(value))
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} B"


def format_index_run_summary(payload: dict[str, object]) -> tuple[str, str]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    run_metrics = payload.get("run_metrics") if isinstance(payload.get("run_metrics"), dict) else {}
    performance_profile = (
        payload.get("performance_profile")
        if isinstance(payload.get("performance_profile"), dict)
        else {}
    )
    performance_mode = bool(payload.get("performance_mode"))
    if not isinstance(summary, dict):
        summary = {}
    if not isinstance(run_metrics, dict):
        run_metrics = {}
    if not isinstance(performance_profile, dict):
        performance_profile = {}

    def _int(source: dict[str, object], key: str) -> int:
        try:
            return int(source.get(key) or 0)
        except (TypeError, ValueError):
            return 0

    indexed = _int(summary, "indexed")
    skipped = _int(summary, "skipped")
    failed = _int(summary, "failed")
    metadata_only = _int(summary, "metadata_only")
    partial_success = _int(summary, "partial_success")
    excluded_video = _int(summary, "excluded_video")
    deleted = _int(summary, "deleted")
    cancelled = bool(summary.get("cancelled"))
    partial_publish = bool(payload.get("partial_publish"))
    published_blocking_files = _int(payload, "blocking_files")

    dedup_candidate_count = _int(run_metrics, "dedup_candidate_count")
    dedup_full_hash_count = _int(run_metrics, "dedup_full_hash_count")
    dedup_verified_source_count = _int(run_metrics, "dedup_verified_source_count")
    dedup_parse_avoided_count = _int(run_metrics, "dedup_parse_avoided_count")
    dedup_bytes_avoided = _int(run_metrics, "dedup_bytes_avoided")
    discovered_files = _int(run_metrics, "discovered_files")
    discovered_bytes = _int(run_metrics, "discovered_bytes")
    scan_ms = _int(run_metrics, "scan_ms")
    parse_ms = _int(run_metrics, "parse_ms")
    write_ms = _int(run_metrics, "write_ms")
    fts_ms = _int(run_metrics, "fts_ms")
    total_ms = _int(run_metrics, "total_ms")
    process_spawn_count = _int(run_metrics, "process_spawn_count")
    cache_hits = _int(run_metrics, "cache_hits")
    cache_misses = _int(run_metrics, "cache_misses")

    label_prefix = "性能模式" if performance_mode else "索引"
    label_bits = [
        f"{label_prefix}部分完成" if partial_publish else f"{label_prefix}完成"
    ]
    if dedup_parse_avoided_count or dedup_bytes_avoided:
        label_bits.append(f"省略解析 {dedup_parse_avoided_count:,}")
        if dedup_bytes_avoided:
            label_bits.append(f"省略 {format_bytes(dedup_bytes_avoided)}")
    elif indexed:
        label_bits.append(f"已索引 {indexed:,}")

    tooltip_lines = [
        f"本次：已索引 {indexed:,}，跳过 {skipped:,}，失败 {failed:,}"
        f"{'，元数据 ' + format_count(metadata_only) if metadata_only else ''}"
        f"{'，部分成功 ' + format_count(partial_success) if partial_success else ''}"
        f"{'，排除视频 ' + format_count(excluded_video) if excluded_video else ''}"
        f"{'，删除 ' + format_count(deleted) if deleted else ''}"
        f"{'，已取消' if cancelled else ''}",
        f"总量：发现 {discovered_files:,} 个文件，{format_bytes(discovered_bytes)}",
        f"耗时：扫描 {format_elapsed(scan_ms)} · 解析 {format_elapsed(parse_ms)} · 写库 {format_elapsed(write_ms)} · FTS {format_elapsed(fts_ms)} · 总计 {format_elapsed(total_ms)}",
    ]
    if partial_publish:
        tooltip_lines.append(
            f"本轮已发布可搜索快照，但仍有 {published_blocking_files:,} 个终止性跳过/失败文件未纳入正文索引；"
            "可在“未成功索引”中查看和处理。"
        )
    if dedup_candidate_count or dedup_full_hash_count or dedup_verified_source_count:
        tooltip_lines.append(
            f"去重：候选 {dedup_candidate_count:,} · 完整哈希 {dedup_full_hash_count:,} · 已验证来源 {dedup_verified_source_count:,}"
        )
    if dedup_parse_avoided_count or dedup_bytes_avoided:
        tooltip_lines.append(
            f"节省：省略解析 {dedup_parse_avoided_count:,} · 省略 {format_bytes(dedup_bytes_avoided)}"
        )
    if process_spawn_count or cache_hits or cache_misses:
        tooltip_lines.append(
            f"运行：进程 {process_spawn_count:,} · 命中 {cache_hits:,} · 未命中 {cache_misses:,}"
        )
    if performance_profile:
        tooltip_lines.append(
            "性能配置："
            f"普通 {performance_profile.get('normal_workers', 0)} · "
            f"Office {performance_profile.get('office_workers', 0)} · "
            f"PDF {performance_profile.get('pdf_workers', performance_profile.get('pdf_parser_workers', 0))} · "
            f"ZIP {performance_profile.get('zip_member_workers', 0)} · "
            f"OCR {performance_profile.get('ocr_workers', 0)} · "
            f"内存 {performance_profile.get('memory_budget_mb', 0)} MB · "
            f"磁盘 {performance_profile.get('disk_class', 'unknown')}"
        )
    return " · ".join(label_bits), "\n".join(tooltip_lines)


def format_count(value: int) -> str:
    return f"{max(0, int(value)):,}"


def compact_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    left = max(1, max_chars // 2 - 1)
    right = max(1, max_chars - left - 1)
    return f"{text[:left]}…{text[-right:]}"


def settings_card(title: str, rows: list[tuple[str, QWidget]]) -> QFrame:
    card = QFrame()
    card.setObjectName("SettingsCard")
    layout = QGridLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setHorizontalSpacing(16)
    layout.setVerticalSpacing(10)
    layout.setColumnMinimumWidth(0, 160)
    layout.setColumnStretch(0, 0)
    layout.setColumnStretch(1, 1)
    heading = QLabel(title)
    heading.setObjectName("SectionTitle")
    layout.addWidget(heading, 0, 0, 1, 2)
    for index, (label, widget) in enumerate(rows, start=1):
        if label:
            text = QLabel(label)
            text.setObjectName("MutedText")
            if isinstance(widget, (QSpinBox, QDoubleSpinBox, QComboBox)):
                widget.setMinimumWidth(180)
                widget.setMaximumWidth(220)
            layout.addWidget(text, index, 0)
            layout.addWidget(widget, index, 1)
        else:
            layout.addWidget(widget, index, 0, 1, 2)
    return card
