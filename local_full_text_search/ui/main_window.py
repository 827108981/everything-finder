from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer, QThread, Qt, Signal
from PySide6.QtGui import QAction, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.config.constants import APP_DISPLAY_NAME, FILE_TYPE_GROUPS, LOG_DIR
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.open_location import open_file, open_parent_folder
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.ui.preview_panel import PreviewPanel
from local_full_text_search.ui.result_view import ResultView
from local_full_text_search.workers.scan_worker import ScanWorker
from local_full_text_search.workers.search_worker import SearchWorker


SEARCH_NAV_ITEMS = ("全部文件", "PDF", "Word", "Excel", "PowerPoint", "图片", "文本/日志")
PAGE_INDEX = {"search": 0, "index": 1, "failed": 2, "settings": 3}


class MainWindow(QMainWindow):
    """Product-style shell: navigation first, search central, management behind pages."""

    def __init__(self, db: DatabaseManager, settings: AppSettings, settings_service: SettingsService) -> None:
        super().__init__()
        self.db = db
        self.settings = settings
        self.settings_service = settings_service
        self.search_thread: QThread | None = None
        self.search_worker: SearchWorker | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.pending_search = False
        self.page = 1
        self.total_confirmed = 0

        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self._build_shell()
        self._install_shortcuts()
        self.refresh_all()

    def _build_shell(self) -> None:
        central = QWidget()
        central.setObjectName("AppRoot")
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.top_bar = TopBar()
        self.top_bar.settings_requested.connect(lambda: self.switch_page("settings"))
        self.top_bar.task_panel_requested.connect(self.toggle_task_panel)
        root_layout.addWidget(self.top_bar)

        self.task_panel = TaskPanel()
        self.task_panel.setVisible(False)
        self.task_panel.pause_resume_requested.connect(self.toggle_scan_pause)
        self.task_panel.cancel_requested.connect(self.cancel_scan)
        self.task_panel.failed_requested.connect(lambda: self.switch_page("failed"))
        root_layout.addWidget(self.task_panel)

        body = QFrame()
        body.setObjectName("Body")
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = Sidebar()
        self.sidebar.nav_requested.connect(self.on_nav_requested)
        body_layout.addWidget(self.sidebar)

        self.stack = QStackedWidget()
        self.search_page = SearchPage()
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

        self.search_page.search_requested.connect(self.request_search)
        self.search_page.stop_requested.connect(self.cancel_search)
        self.search_page.add_root_requested.connect(self.add_root)
        self.search_page.previous_page_requested.connect(self.previous_page)
        self.search_page.next_page_requested.connect(self.next_page_search)
        self.search_page.open_requested.connect(self.open_path)
        self.search_page.open_folder_requested.connect(self.open_folder_path)
        self.search_page.reindex_requested.connect(self.reindex_file)
        self.search_page.result_selected.connect(self.show_preview)

        self.index_page.add_root_requested.connect(self.add_root)
        self.index_page.scan_requested.connect(self.start_scan)
        self.index_page.pause_requested.connect(lambda: self.scan_worker.pause() if self.scan_worker else None)
        self.index_page.resume_requested.connect(lambda: self.scan_worker.resume() if self.scan_worker else None)
        self.index_page.cancel_requested.connect(self.cancel_scan)
        self.index_page.toggle_root_requested.connect(self.toggle_root)
        self.index_page.remove_root_requested.connect(self.remove_root)
        self.index_page.open_folder_requested.connect(self.open_path)
        self.index_page.failed_requested.connect(lambda: self.switch_page("failed"))

        self.failed_page.retry_requested.connect(self.start_scan)
        self.failed_page.open_folder_requested.connect(self.open_folder_path)
        self.failed_page.refresh_requested.connect(self.refresh_failed_page)
        self.failed_page.export_requested.connect(self.export_failed_rows)

        self.settings_page.save_requested.connect(self.save_settings)

    def _install_shortcuts(self) -> None:
        QShortcut(QKeySequence("Ctrl+F"), self, activated=self.search_page.focus_search)
        QShortcut(QKeySequence("Esc"), self, activated=self.handle_escape)

    def handle_escape(self) -> None:
        if self.preview_panel.isVisible():
            self.hide_preview()
        elif self.search_page.text():
            self.search_page.clear_search()

    def on_nav_requested(self, key: str) -> None:
        if key in SEARCH_NAV_ITEMS:
            self.switch_page("search")
            self.search_page.set_file_type_from_nav(key)
            self.sidebar.set_active(key)
        else:
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
        self.sidebar.set_active("全部文件" if key == "search" else key)
        if key == "failed":
            self.refresh_failed_page()

    def toggle_task_panel(self) -> None:
        self.task_panel.setVisible(not self.task_panel.isVisible())

    def refresh_all(self) -> None:
        roots = self.db.list_roots()
        stats = self.db.stats()
        self.search_page.set_roots(roots)
        self.search_page.set_stats(stats, has_roots=bool(roots))
        self.index_page.set_roots(roots, self.root_stats_by_id())
        self.refresh_failed_page()
        self.update_index_status()

    def refresh_failed_page(self) -> None:
        self.failed_page.set_rows(self.db.failed_files(limit=2000))

    def update_index_status(self) -> None:
        stats = self.db.stats()
        self.top_bar.set_index_status(
            f"已索引 {stats['files']} 个文件",
            is_running=self.scan_thread is not None,
        )
        self.task_panel.set_idle(stats)

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
        directory = QFileDialog.getExistingDirectory(self, "选择搜索目录")
        if not directory:
            return
        self.db.add_root(Path(directory))
        self.refresh_all()
        answer = QMessageBox.question(self, "建立索引", "已添加搜索范围，是否立即开始建立完整索引？")
        if answer == QMessageBox.StandardButton.Yes:
            self.start_scan()

    def remove_root(self, root_id: int) -> None:
        if QMessageBox.question(self, "删除搜索范围", "只删除索引和配置，不会删除原文件。确认继续？") != QMessageBox.StandardButton.Yes:
            return
        self.db.remove_root(root_id)
        self.refresh_all()

    def toggle_root(self, root_id: int, enabled: bool) -> None:
        self.db.set_root_enabled(root_id, enabled)
        self.refresh_all()

    def start_scan(self) -> None:
        if self.scan_thread is not None:
            self.task_panel.setVisible(True)
            return
        self.scan_thread = QThread()
        self.scan_worker = ScanWorker(self.db.db_path, self.settings)
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.on_scan_progress)
        self.scan_worker.finished.connect(self.on_scan_finished)
        self.scan_worker.failed.connect(self.on_scan_failed)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.cleanup_scan_thread)
        self.top_bar.set_index_status("正在索引...", is_running=True)
        self.task_panel.set_running("准备索引", 0, 0)
        self.task_panel.setVisible(True)
        self.index_page.set_task_running(True)
        self.scan_thread.start()

    def toggle_scan_pause(self) -> None:
        if self.scan_worker is None:
            return
        if self.task_panel.paused:
            self.scan_worker.resume()
            self.task_panel.set_paused(False)
        else:
            self.scan_worker.pause()
            self.task_panel.set_paused(True)

    def cancel_scan(self) -> None:
        if self.scan_worker is not None:
            self.scan_worker.cancel()

    def on_scan_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        indexed = int(payload.get("indexed") or 0)
        scanned = int(payload.get("scanned") or 0)
        failed = int(payload.get("failed") or 0)
        current = str(payload.get("current_file") or "")
        text = f"正在索引 {indexed:,} / {scanned:,}"
        self.top_bar.set_index_status(text, is_running=True)
        self.task_panel.set_running(current or text, indexed, scanned, failed)
        self.index_page.set_scan_progress(indexed, scanned, failed, current)

    def on_scan_finished(self, summary: object) -> None:
        self.index_page.set_task_running(False)
        self.task_panel.set_finished(str(summary))
        self.refresh_all()

    def on_scan_failed(self, message: str) -> None:
        self.index_page.set_task_running(False)
        self.top_bar.set_index_status("索引任务失败", is_error=True)
        self.task_panel.set_error("索引任务失败，详细信息已写入日志")
        QMessageBox.critical(self, "索引任务失败", message)

    def cleanup_scan_thread(self) -> None:
        self.scan_thread = None
        self.scan_worker = None
        self.index_page.set_task_running(False)
        self.update_index_status()

    def request_search(self) -> None:
        if not self.search_page.text():
            self.page = 1
            self.total_confirmed = 0
            self.search_page.show_idle_state()
            self.hide_preview()
            return
        self.page = 1
        if self.search_thread is not None:
            self.pending_search = True
            self.cancel_search()
            return
        self._run_search()

    def _run_search(self) -> None:
        if not self.search_page.text():
            return
        query = SearchQuery(
            text=self.search_page.text(),
            mode=self.search_page.mode_value(),
            root_ids=self.search_page.root_ids(),
            extensions=self.search_page.extensions(),
            search_filename=self.search_page.search_filename(),
            search_path=self.search_page.search_path(),
            search_content=self.search_page.search_content(),
            include_ocr=self.search_page.include_ocr(),
            case_sensitive=self.search_page.case_sensitive(),
            page_size=min(max(self.settings.page_size, 50), 100),
            page=self.page,
        )
        self.search_thread = QThread()
        self.search_worker = SearchWorker(self.db.db_path, query)
        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.cancelled.connect(self.on_search_cancelled)
        self.search_worker.failed.connect(self.on_search_failed)
        self.search_worker.finished.connect(self.search_thread.quit)
        self.search_worker.cancelled.connect(self.search_thread.quit)
        self.search_worker.failed.connect(self.search_thread.quit)
        self.search_thread.finished.connect(self.cleanup_search_thread)
        self.search_page.set_running(True)
        self.search_thread.start()

    def cancel_search(self) -> None:
        if self.search_worker is not None:
            self.search_worker.cancel()

    def on_search_finished(self, page: object) -> None:
        self.total_confirmed = page.total_confirmed
        self.search_page.set_results(page)
        if page.total_confirmed <= 0:
            self.hide_preview()

    def on_search_cancelled(self) -> None:
        if self.pending_search:
            self.search_page.set_status("正在准备新搜索...")
        else:
            self.search_page.set_status("搜索已取消")

    def on_search_failed(self, message: str) -> None:
        self.search_page.set_status("搜索失败，详细信息已写入日志")
        QMessageBox.critical(self, "搜索失败", message)

    def cleanup_search_thread(self) -> None:
        self.search_page.set_running(False)
        self.search_thread = None
        self.search_worker = None
        if self.pending_search:
            self.pending_search = False
            self._run_search()

    def previous_page(self) -> None:
        if self.page <= 1:
            return
        self.page -= 1
        self._run_search()

    def next_page_search(self) -> None:
        if self.page * min(max(self.settings.page_size, 50), 100) >= self.total_confirmed:
            return
        self.page += 1
        self._run_search()

    def show_preview(self, result: object) -> None:
        if result is None:
            return
        self.preview_panel.show_result(result)
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

    def reindex_file(self, _path: str) -> None:
        self.start_scan()

    def export_failed_rows(self, rows: list[object]) -> None:
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
            writer.writerow(["路径", "扩展名", "状态", "错误码", "原因", "解析器", "时间"])
            for row in rows:
                writer.writerow(
                    [
                        str(row["path"]),
                        str(row["extension"] or ""),
                        str(row["parse_status"]),
                        str(row["parse_error_code"] or ""),
                        str(row["parse_error_message"] or ""),
                        str(row["parser_name"] or ""),
                        str(row["indexed_at"] or ""),
                    ]
                )
        self.failed_page.set_status(f"已导出：{target}")

    def save_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings_service.save(settings)
        self.search_page.set_status("设置已保存")


class TopBar(QFrame):
    settings_requested = Signal()
    task_panel_requested = Signal()

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
        self.index_button.clicked.connect(self.task_panel_requested.emit)
        self.settings_button = QPushButton("设置")
        self.settings_button.setObjectName("TextButton")
        self.settings_button.setToolTip("打开设置")
        self.settings_button.clicked.connect(self.settings_requested.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 14, 24, 14)
        layout.addLayout(title_layout)
        layout.addStretch(1)
        layout.addWidget(self.index_button)
        layout.addWidget(self.settings_button)

    def set_title(self, title: str, subtitle: str) -> None:
        self.title.setText(title)
        self.subtitle.setText(subtitle)

    def set_index_status(self, text: str, *, is_running: bool = False, is_error: bool = False) -> None:
        self.index_button.setText(text)
        self.index_button.setProperty("state", "error" if is_error else "running" if is_running else "ready")
        self.index_button.style().unpolish(self.index_button)
        self.index_button.style().polish(self.index_button)


class Sidebar(QFrame):
    nav_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Sidebar")
        self.setFixedWidth(220)
        self.buttons: dict[str, QPushButton] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 18, 12, 18)
        layout.setSpacing(6)
        layout.addWidget(section_label("搜索"))
        for label in SEARCH_NAV_ITEMS:
            self._add_button(layout, label, label)
        layout.addSpacing(12)
        layout.addWidget(separator())
        layout.addSpacing(12)
        layout.addWidget(section_label("管理"))
        self._add_button(layout, "索引管理", "index")
        self._add_button(layout, "未成功索引", "failed")
        self._add_button(layout, "设置", "settings")
        layout.addStretch(1)
        self.set_active("全部文件")

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


class SearchPage(QWidget):
    search_requested = Signal()
    stop_requested = Signal()
    add_root_requested = Signal()
    previous_page_requested = Signal()
    next_page_requested = Signal()
    open_requested = Signal(str)
    open_folder_requested = Signal(str)
    reindex_requested = Signal(str)
    result_selected = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self._stats: dict[str, int] = {}
        self._has_roots = False
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.setInterval(300)
        self.debounce.timeout.connect(self.search_requested.emit)
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        title = QLabel("本地全文搜索")
        title.setObjectName("PageTitle")
        subtitle = QLabel("搜索文件名、正文、表格、幻灯片和图片文字")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        self.search_box = SearchBox()
        self.search_box.input.textChanged.connect(self._on_text_changed)
        self.search_box.search_requested.connect(self.search_requested.emit)
        self.search_box.stop_requested.connect(self.stop_requested.emit)
        layout.addWidget(self.search_box)

        filters = QHBoxLayout()
        filters.setSpacing(8)
        self.mode_combo = chip_combo()
        self.mode_combo.addItem("完全包含", "exact")
        self.mode_combo.addItem("完整短语", "phrase")
        self.mode_combo.addItem("全部关键词", "all")
        self.mode_combo.addItem("任意关键词", "any")
        self.mode_combo.addItem("正则表达式", "regex")
        self.file_type_combo = chip_combo()
        for label in ("全部", "PDF", "Word", "Excel", "PowerPoint", "图片", "文本/日志"):
            self.file_type_combo.addItem("全部格式" if label == "全部" else label, label)
        self.scope_combo = chip_combo()
        self.scope_combo.addItem("全部范围", None)
        self.more_button = QPushButton("更多筛选")
        self.more_button.setObjectName("ChipButton")
        self.more_button.setCheckable(True)
        self.more_button.clicked.connect(self._toggle_advanced)
        filters.addWidget(self.mode_combo)
        filters.addWidget(self.file_type_combo)
        filters.addWidget(self.scope_combo)
        filters.addWidget(self.more_button)
        filters.addStretch(1)
        layout.addLayout(filters)

        self.advanced = AdvancedFilters()
        self.advanced.setVisible(False)
        layout.addWidget(self.advanced)

        self.status_label = QLabel("输入关键词开始搜索")
        self.status_label.setObjectName("InlineStatus")
        layout.addWidget(self.status_label)

        self.content_stack = QStackedWidget()
        self.empty_state = EmptyState()
        self.result_view = ResultView()
        self.result_view.open_requested.connect(self.open_requested.emit)
        self.result_view.open_folder_requested.connect(self.open_folder_requested.emit)
        self.result_view.reindex_requested.connect(self.reindex_requested.emit)
        self.result_view.selected_result_changed.connect(self.result_selected.emit)
        self.content_stack.addWidget(self.empty_state)
        self.content_stack.addWidget(self.result_view)
        layout.addWidget(self.content_stack, 1)

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

        self.file_type_combo.currentIndexChanged.connect(self._filter_changed)
        self.mode_combo.currentIndexChanged.connect(self._filter_changed)
        self.scope_combo.currentIndexChanged.connect(self._filter_changed)
        self.show_idle_state()

    def _on_text_changed(self, text: str) -> None:
        self.search_box.update_clear_button()
        if not text.strip():
            self.debounce.stop()
            self.show_idle_state()
            return
        self.debounce.start()

    def _filter_changed(self) -> None:
        if self.text():
            self.search_requested.emit()

    def _toggle_advanced(self) -> None:
        self.advanced.setVisible(self.more_button.isChecked())

    def set_roots(self, roots: list[object]) -> None:
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

    def set_stats(self, stats: dict[str, int], *, has_roots: bool) -> None:
        self._stats = stats
        self._has_roots = has_roots
        if not self.text():
            self.show_idle_state()

    def set_file_type_from_nav(self, label: str) -> None:
        data = "全部" if label == "全部文件" else label
        index = self.file_type_combo.findData(data)
        if index >= 0:
            self.file_type_combo.setCurrentIndex(index)

    def focus_search(self) -> None:
        self.search_box.input.setFocus()
        self.search_box.input.selectAll()

    def clear_search(self) -> None:
        self.search_box.input.clear()
        self.show_idle_state()

    def text(self) -> str:
        return self.search_box.input.text().strip()

    def mode_value(self) -> str:
        return str(self.mode_combo.currentData())

    def root_ids(self) -> list[int]:
        value = self.scope_combo.currentData()
        return [int(value)] if value is not None else []

    def extensions(self) -> list[str]:
        label = str(self.file_type_combo.currentData())
        return sorted(FILE_TYPE_GROUPS.get(label, set()))

    def search_filename(self) -> bool:
        return self.advanced.search_filename.isChecked()

    def search_path(self) -> bool:
        return self.advanced.search_path.isChecked()

    def search_content(self) -> bool:
        return self.advanced.search_content.isChecked()

    def include_ocr(self) -> bool:
        return self.advanced.include_ocr.isChecked()

    def case_sensitive(self) -> bool:
        return self.advanced.case_sensitive.isChecked()

    def set_running(self, running: bool) -> None:
        self.search_box.set_running(running)
        if running:
            self.set_status("正在搜索...")

    def set_results(self, page: object) -> None:
        self.page_label.setText(f"第 {page.page} 页")
        self.set_status(f"找到 {page.total_confirmed} 条结果 · 用时 {page.elapsed_ms / 1000:.2f} 秒 · 候选 {page.total_candidates}")
        if page.total_confirmed <= 0:
            self.show_no_results()
            return
        self.content_stack.setCurrentWidget(self.result_view)
        self.result_view.set_results(page.results, self.text())
        self.pager.setVisible(page.total_confirmed > page.page_size)

    def show_idle_state(self) -> None:
        if not self._has_roots:
            self.empty_state.set_content(
                "尚未添加搜索范围",
                "添加一个本地文件夹后，即可建立全文索引并开始搜索",
                "添加搜索范围",
                self.add_root_requested.emit,
            )
        else:
            count = self._stats.get("files", 0)
            self.empty_state.set_content(
                "输入关键词开始搜索",
                f"支持搜索 PDF、Word、Excel、PowerPoint、文本、日志和图片中的文字\n已索引 {count:,} 个文件",
                "",
                None,
            )
        self.content_stack.setCurrentWidget(self.empty_state)
        self.pager.setVisible(False)
        self.set_status("输入关键词开始搜索")

    def show_no_results(self) -> None:
        self.empty_state.set_content(
            f"没有找到“{self.text()}”",
            "可以尝试检查搜索范围、切换为全部关键词，或确认 OCR 结果已完成索引",
            "",
            None,
        )
        self.content_stack.setCurrentWidget(self.empty_state)
        self.pager.setVisible(False)

    def set_status(self, text: str) -> None:
        self.status_label.setText(text)


class SearchBox(QFrame):
    search_requested = Signal()
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("SearchBox")
        icon = QLabel("⌕")
        icon.setObjectName("SearchIcon")
        self.input = QLineEdit()
        self.input.setObjectName("SearchInput")
        self.input.setPlaceholderText("搜索文件名、正文、表格、幻灯片和图片文字")
        self.input.returnPressed.connect(self.search_requested.emit)
        self.clear_button = QPushButton("×")
        self.clear_button.setObjectName("IconButton")
        self.clear_button.setFixedSize(30, 30)
        self.clear_button.clicked.connect(self.input.clear)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setVisible(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 10, 0)
        layout.setSpacing(8)
        layout.addWidget(icon)
        layout.addWidget(self.input, 1)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.stop_button)
        self.update_clear_button()

    def update_clear_button(self) -> None:
        self.clear_button.setVisible(bool(self.input.text()))

    def set_running(self, running: bool) -> None:
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
        self.exact_only = QCheckBox("仅精确命中")
        self.ocr_fuzzy = QCheckBox("显示 OCR 疑似命中")
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
                self.exact_only,
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
    scan_requested = Signal()
    pause_requested = Signal()
    resume_requested = Signal()
    cancel_requested = Signal()
    toggle_root_requested = Signal(int, bool)
    remove_root_requested = Signal(int)
    open_folder_requested = Signal(str)
    failed_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.running = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("索引管理")
        title.setObjectName("PageTitle")
        subtitle = QLabel("管理需要搜索的本地文件夹和索引状态")
        subtitle.setObjectName("PageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.add_button = QPushButton("添加搜索范围")
        self.add_button.setObjectName("PrimaryButton")
        self.add_button.clicked.connect(self.add_root_requested.emit)
        header.addLayout(title_box)
        header.addStretch(1)
        header.addWidget(self.add_button)
        layout.addLayout(header)

        self.task_strip = QFrame()
        self.task_strip.setObjectName("TaskStrip")
        task_layout = QHBoxLayout(self.task_strip)
        task_layout.setContentsMargins(14, 10, 14, 10)
        self.task_label = QLabel("当前没有索引任务")
        self.task_label.setObjectName("MutedText")
        self.task_progress = QProgressBar()
        self.task_progress.setTextVisible(False)
        self.task_progress.setFixedWidth(180)
        self.start_button = QPushButton("更新全部")
        self.pause_button = QPushButton("暂停")
        self.cancel_button = QPushButton("取消")
        self.start_button.clicked.connect(self.scan_requested.emit)
        self.pause_button.clicked.connect(self.pause_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        task_layout.addWidget(self.task_label, 1)
        task_layout.addWidget(self.task_progress)
        task_layout.addWidget(self.start_button)
        task_layout.addWidget(self.pause_button)
        task_layout.addWidget(self.cancel_button)
        layout.addWidget(self.task_strip)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.list_host = QWidget()
        self.list_layout = QVBoxLayout(self.list_host)
        self.list_layout.setContentsMargins(0, 0, 0, 0)
        self.list_layout.setSpacing(10)
        self.scroll.setWidget(self.list_host)
        layout.addWidget(self.scroll, 1)
        self.set_task_running(False)

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
                card.toggle_requested.connect(self.toggle_root_requested.emit)
                card.remove_requested.connect(self.remove_root_requested.emit)
                card.open_requested.connect(self.open_folder_requested.emit)
                card.failed_requested.connect(self.failed_requested.emit)
                self.list_layout.addWidget(card)
        self.list_layout.addStretch(1)

    def set_task_running(self, running: bool) -> None:
        self.running = running
        self.pause_button.setVisible(running)
        self.cancel_button.setVisible(running)
        self.start_button.setEnabled(not running)
        if not running:
            self.task_progress.setValue(0)
            self.task_label.setText("当前没有索引任务")

    def set_scan_progress(self, indexed: int, scanned: int, failed: int, current: str) -> None:
        self.set_task_running(True)
        self.task_label.setText(f"正在索引 {indexed:,} / {scanned:,} · 失败 {failed} · {Path(current).name if current else ''}")
        self.task_progress.setRange(0, max(scanned, indexed, 1))
        self.task_progress.setValue(indexed)


class RootCard(QFrame):
    scan_requested = Signal()
    toggle_requested = Signal(int, bool)
    remove_requested = Signal(int)
    open_requested = Signal(str)
    failed_requested = Signal()

    def __init__(self, row: object, stats: dict[str, int]) -> None:
        super().__init__()
        self.setObjectName("ScopeCard")
        self.root_id = int(row["id"])
        self.path = str(row["path"])
        enabled = bool(row["enabled"])
        last_scan = str(row["last_scan_at"] or "未索引")
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

        update_button = QPushButton("更新")
        update_button.clicked.connect(self.scan_requested.emit)
        more_button = QPushButton("更多")
        more_button.clicked.connect(lambda: self._show_menu(more_button, enabled))

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.addWidget(title, 1)
        top.addWidget(status)
        actions = QHBoxLayout()
        actions.addStretch(1)
        actions.addWidget(update_button)
        actions.addWidget(more_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(meta)
        layout.addWidget(detail)
        layout.addLayout(actions)

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
    retry_requested = Signal()
    refresh_requested = Signal()
    export_requested = Signal(list)
    open_folder_requested = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.rows: list[object] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)
        title = QLabel("未成功索引")
        title.setObjectName("PageTitle")
        self.subtitle = QLabel("0 个文件需要处理")
        self.subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(self.subtitle)

        filters = QHBoxLayout()
        self.status_filter = chip_combo()
        self.extension_filter = chip_combo()
        self.retry_button = QPushButton("重新尝试")
        self.export_button = QPushButton("导出明细")
        self.open_log_button = QPushButton("打开日志目录")
        self.retry_button.setObjectName("PrimaryButton")
        self.retry_button.clicked.connect(self.retry_requested.emit)
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.visible_rows()))
        self.open_log_button.clicked.connect(self.open_log_dir)
        filters.addWidget(self.status_filter)
        filters.addWidget(self.extension_filter)
        filters.addStretch(1)
        filters.addWidget(self.retry_button)
        filters.addWidget(self.export_button)
        filters.addWidget(self.open_log_button)
        layout.addLayout(filters)

        self.table = QTableWidget(0, 6)
        self.table.setObjectName("FailedTable")
        self.table.setHorizontalHeaderLabels(["文件名", "路径", "类型", "原因", "最后尝试", "操作"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table, 1)
        self.status = QLabel("")
        self.status.setObjectName("InlineStatus")
        layout.addWidget(self.status)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        self.extension_filter.currentIndexChanged.connect(self.apply_filters)

    def set_rows(self, rows: list[object]) -> None:
        self.rows = rows
        self.subtitle.setText(f"{len(rows)} 个文件需要处理")
        self.status_filter.blockSignals(True)
        self.extension_filter.blockSignals(True)
        self.status_filter.clear()
        self.extension_filter.clear()
        self.status_filter.addItem("全部原因", "")
        self.extension_filter.addItem("全部格式", "")
        for value in sorted({str(row["parse_status"]) for row in rows}):
            self.status_filter.addItem(value, value)
        for value in sorted({str(row["extension"] or "") for row in rows}):
            self.extension_filter.addItem(value or "<无扩展名>", value)
        self.status_filter.blockSignals(False)
        self.extension_filter.blockSignals(False)
        self.apply_filters()

    def visible_rows(self) -> list[object]:
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

    def apply_filters(self) -> None:
        rows = self.visible_rows()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            path = str(row["path"])
            values = [
                Path(path).name,
                path,
                str(row["extension"] or ""),
                str(row["parse_error_message"] or row["parse_status"]),
                str(row["indexed_at"] or ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row_index, col, item)
            action_button = QPushButton("打开文件夹")
            action_button.clicked.connect(lambda _checked=False, value=path: self.open_folder_requested.emit(value))
            self.table.setCellWidget(row_index, 5, action_button)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def open_log_dir(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        open_file(LOG_DIR)


class SettingsPage(QWidget):
    save_requested = Signal(object)

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.setObjectName("Page")
        self.settings = settings
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)
        title = QLabel("设置")
        title.setObjectName("PageTitle")
        subtitle = QLabel("调整搜索、索引和 OCR 默认行为")
        subtitle.setObjectName("PageSubtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(12)

        self.page_size = QSpinBox()
        self.page_size.setRange(50, 1000)
        self.page_size.setValue(settings.page_size)
        self.auto_search = QCheckBox("输入后自动搜索")
        self.auto_search.setChecked(True)
        self.search_history = QCheckBox("保存搜索历史")
        self.search_history.setChecked(settings.save_search_history)
        host_layout.addWidget(settings_card("搜索", [("默认结果数量", self.page_size), ("", self.auto_search), ("", self.search_history)]))

        self.enable_ocr = QCheckBox("启用 OCR")
        self.enable_ocr.setChecked(settings.enable_ocr)
        self.ocr_images = QCheckBox("索引图片 OCR")
        self.ocr_images.setChecked(settings.ocr_images)
        self.ocr_pdf = QCheckBox("索引扫描 PDF OCR")
        self.ocr_pdf.setChecked(settings.ocr_scanned_pdf)
        self.ocr_workers = QSpinBox()
        self.ocr_workers.setRange(1, 4)
        self.ocr_workers.setValue(settings.ocr_workers)
        self.parser_workers = QSpinBox()
        self.parser_workers.setRange(1, 16)
        self.parser_workers.setValue(settings.parser_workers)
        self.slow_file_workers = QSpinBox()
        self.slow_file_workers.setRange(1, 4)
        self.slow_file_workers.setValue(settings.slow_file_workers)
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
                "索引与 OCR",
                [
                    ("", self.enable_ocr),
                    ("", self.ocr_images),
                    ("", self.ocr_pdf),
                    ("普通解析线程数", self.parser_workers),
                    ("OCR 工作线程数", self.ocr_workers),
                    ("ZIP/老版 Office 慢任务线程数", self.slow_file_workers),
                    ("批量写库文件数", self.write_batch_size),
                    ("小图片 OCR 跳过像素", self.min_ocr_pixels),
                    ("图片 OCR 最大边长", self.max_ocr_side),
                ],
            )
        )

        self.monitor_changes = QCheckBox("自动监控文件变化")
        self.monitor_changes.setChecked(settings.monitor_file_changes)
        self.hidden_files = QCheckBox("索引隐藏文件")
        self.hidden_files.setChecked(settings.include_hidden_files)
        self.exclude_dirs = QTextEdit("\n".join(settings.excluded_dirs))
        self.exclude_dirs.setMinimumHeight(90)
        host_layout.addWidget(settings_card("高级", [("", self.monitor_changes), ("", self.hidden_files), ("默认排除目录", self.exclude_dirs)]))
        host_layout.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

        save = QPushButton("保存设置")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self.apply)
        layout.addWidget(save, alignment=Qt.AlignmentFlag.AlignRight)

    def apply(self) -> None:
        self.settings.page_size = int(self.page_size.value())
        self.settings.save_search_history = self.search_history.isChecked()
        self.settings.enable_ocr = self.enable_ocr.isChecked()
        self.settings.ocr_images = self.ocr_images.isChecked()
        self.settings.ocr_scanned_pdf = self.ocr_pdf.isChecked()
        self.settings.parser_workers = int(self.parser_workers.value())
        self.settings.ocr_workers = int(self.ocr_workers.value())
        self.settings.slow_file_workers = int(self.slow_file_workers.value())
        self.settings.index_write_batch_size = int(self.write_batch_size.value())
        self.settings.min_ocr_image_pixels = int(self.min_ocr_pixels.value())
        self.settings.max_ocr_image_side = int(self.max_ocr_side.value())
        self.settings.monitor_file_changes = self.monitor_changes.isChecked()
        self.settings.include_hidden_files = self.hidden_files.isChecked()
        self.settings.excluded_dirs = [line.strip() for line in self.exclude_dirs.toPlainText().splitlines() if line.strip()]
        self.save_requested.emit(self.settings)


class TaskPanel(QFrame):
    pause_resume_requested = Signal()
    cancel_requested = Signal()
    failed_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TaskPanel")
        self.paused = False
        title = QLabel("索引任务")
        title.setObjectName("SectionTitle")
        self.detail = QLabel("当前没有索引任务")
        self.detail.setObjectName("MutedText")
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.pause_resume_button = QPushButton("暂停")
        self.cancel_button = QPushButton("取消当前任务")
        self.failed_button = QPushButton("查看失败文件")
        self.pause_resume_button.clicked.connect(self.pause_resume_requested.emit)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.failed_button.clicked.connect(self.failed_requested.emit)
        layout = QGridLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)
        layout.setHorizontalSpacing(12)
        layout.addWidget(title, 0, 0)
        layout.addWidget(self.detail, 1, 0)
        layout.addWidget(self.progress, 2, 0, 1, 4)
        layout.addWidget(self.pause_resume_button, 0, 2)
        layout.addWidget(self.cancel_button, 0, 3)
        layout.addWidget(self.failed_button, 1, 3)

    def set_idle(self, stats: dict[str, int]) -> None:
        self.detail.setText(f"索引已更新 · 文件 {stats.get('files', 0):,} · 失败 {stats.get('failed', 0):,}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1)
        self.pause_resume_button.setEnabled(False)
        self.cancel_button.setEnabled(False)

    def set_running(self, current: str, indexed: int, scanned: int, failed: int = 0) -> None:
        self.detail.setText(f"{current}\n已处理 {indexed:,} / {scanned:,} · 失败 {failed}")
        self.progress.setRange(0, max(scanned, indexed, 1))
        self.progress.setValue(indexed)
        self.pause_resume_button.setEnabled(True)
        self.cancel_button.setEnabled(True)

    def set_paused(self, paused: bool) -> None:
        self.paused = paused
        self.pause_resume_button.setText("继续" if paused else "暂停")

    def set_finished(self, summary: str) -> None:
        self.detail.setText(f"索引完成：{summary}")
        self.pause_resume_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.set_paused(False)

    def set_error(self, message: str) -> None:
        self.detail.setText(message)
        self.pause_resume_button.setEnabled(False)
        self.cancel_button.setEnabled(False)


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setObjectName("SidebarSection")
    return label


def separator() -> QFrame:
    line = QFrame()
    line.setObjectName("Separator")
    line.setFixedHeight(1)
    return line


def chip_combo() -> QComboBox:
    combo = QComboBox()
    combo.setObjectName("FilterChip")
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


def settings_card(title: str, rows: list[tuple[str, QWidget]]) -> QFrame:
    card = QFrame()
    card.setObjectName("SettingsCard")
    layout = QGridLayout(card)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setHorizontalSpacing(16)
    layout.setVerticalSpacing(10)
    heading = QLabel(title)
    heading.setObjectName("SectionTitle")
    layout.addWidget(heading, 0, 0, 1, 2)
    for index, (label, widget) in enumerate(rows, start=1):
        if label:
            text = QLabel(label)
            text.setObjectName("MutedText")
            layout.addWidget(text, index, 0)
            layout.addWidget(widget, index, 1)
        else:
            layout.addWidget(widget, index, 0, 1, 2)
    return card
