from __future__ import annotations

import csv
import os
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QTimer, QThread, Qt, Signal
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QShortcut
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
from local_full_text_search.core.file_monitor import FileMonitor
from local_full_text_search.core.open_location import open_file, open_parent_folder
from local_full_text_search.core.search_time_estimator import SearchEstimateContext, SearchTimeEstimator
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.settings_service import SettingsService
from local_full_text_search.ui.preview_panel import PreviewPanel
from local_full_text_search.ui.result_view import ResultView
from local_full_text_search.workers.scan_worker import ScanWorker
from local_full_text_search.workers.scope_exclusion_worker import ScopeExclusionWorker
from local_full_text_search.workers.search_worker import SearchWorker


PAGE_INDEX = {"search": 0, "index": 1, "failed": 2, "settings": 3}


def performance_mode_notice() -> str:
    return (
        "性能模式适合在电脑空闲时使用。程序会根据本机配置尽可能使用 "
        "CPU、内存和磁盘带宽来缩短索引时间；检测到资源压力时会自动降低并发。"
    )


def format_index_scope_status(readiness: dict[str, object]) -> str:
    complete = int(readiness.get("complete_files") or 0)
    eligible = int(readiness.get("eligible_files") or 0)
    blocking = int(readiness.get("blocking_files") or 0)
    excluded = int(readiness.get("manual_excluded_files") or 0)
    video = int(readiness.get("video_excluded") or 0)
    details = []
    if excluded:
        details.append(f"已人工排除 {excluded} 个")
    if video:
        details.append(f"排除视频 {video} 个")
    if bool(readiness.get("ready")):
        text = f"当前范围完成 {complete}/{eligible}"
    else:
        if blocking:
            text = f"索引未完成 {complete}/{eligible} · 待处理 {blocking}"
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
            text = f"文件已完成 {complete}/{eligible} · 索引状态待修复"
            if residual:
                text += " · " + " · ".join(residual)
    if details:
        text += " · " + " · ".join(details)
    return text


class MainWindow(QMainWindow):
    """Product-style shell: navigation first, search central, management behind pages."""

    file_change_detected = Signal(str)

    def __init__(self, db: DatabaseManager, settings: AppSettings, settings_service: SettingsService) -> None:
        super().__init__()
        self.db = db
        self.settings = settings
        self.settings_service = settings_service
        self.search_thread: QThread | None = None
        self.search_worker: SearchWorker | None = None
        self.scan_thread: QThread | None = None
        self.scan_worker: ScanWorker | None = None
        self.exclusion_thread: QThread | None = None
        self.exclusion_worker: ScopeExclusionWorker | None = None
        self.exclusion_outcome: tuple[str, object] | None = None
        self.exclusion_progress_payload: dict[str, object] = {}
        self.exclusion_started_at = 0.0
        self.pending_search = False
        self.pending_monitor_scan = False
        self.force_complete_after_retry = False
        self.pending_force_complete_confirmation = False
        self.closing = False
        self.page = 1
        self.total_confirmed = 0
        self.file_monitor = FileMonitor(lambda path: self.file_change_detected.emit(str(path)))
        self.force_close_timer = QTimer(self)
        self.force_close_timer.setSingleShot(True)
        self.force_close_timer.setInterval(2_500)
        self.force_close_timer.timeout.connect(self._force_exit)
        self.exclusion_heartbeat_timer = QTimer(self)
        self.exclusion_heartbeat_timer.setInterval(1_000)
        self.exclusion_heartbeat_timer.timeout.connect(
            self.on_exclusion_heartbeat
        )

        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1440, 900)
        self.setMinimumSize(1100, 700)
        self._build_shell()
        self._install_shortcuts()
        self.file_change_detected.connect(self.on_monitored_file_changed)
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
        self.search_page.clear_history_requested.connect(self.clear_search_history)

        self.index_page.add_root_requested.connect(self.add_root)
        self.index_page.scan_requested.connect(self.start_scan)
        self.index_page.performance_scan_requested.connect(
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

        self.failed_page.retry_requested.connect(self.start_scan)
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
        if key == "failed" and self.exclusion_thread is None:
            self.refresh_failed_page()

    def refresh_all(self) -> None:
        roots = self.db.list_roots()
        stats = self.db.stats()
        readiness = self.db.index_readiness()
        self.search_page.set_roots(roots)
        self.search_page.set_stats(stats, has_roots=bool(roots))
        self.search_page.set_index_ready(bool(readiness["ready"]), readiness)
        self.index_page.set_roots(roots, self.root_stats_by_id())
        self.index_page.set_readiness(readiness)
        self.refresh_failed_page()
        self.update_index_status()
        self.search_page.set_history(self.db.search_history())
        self.refresh_file_monitor()

    def refresh_file_monitor(self) -> None:
        self.file_monitor.stop()
        if self.closing or not self.settings.monitor_file_changes:
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
            self.top_bar.set_index_status("检测到文件变化，点击更新", is_pending=True)

    def clear_search_history(self) -> None:
        self.db.clear_search_history()
        self.search_page.set_history([])

    def refresh_failed_page(self) -> None:
        readiness = self.db.index_readiness()
        self.failed_page.set_rows(
            self.db.failed_files(limit=2000),
            self.db.excluded_files(limit=2000),
            self.db.metadata_only_files(limit=2000),
        )
        self.failed_page.set_readiness(readiness)

    def update_index_status(self) -> None:
        if self.exclusion_thread is not None:
            self.top_bar.set_index_status("正在更新搜索索引...", is_running=True)
            return
        readiness = self.db.index_readiness()
        if self.pending_monitor_scan and self.scan_thread is None:
            self.top_bar.set_index_status("检测到文件变化，点击更新", is_pending=True)
            return
        blocking = int(readiness["blocking_files"])
        if bool(readiness["ready"]):
            self.top_bar.set_index_status(
                format_index_scope_status(readiness)
            )
            return
        self.top_bar.set_index_status(
            format_index_scope_status(readiness),
            is_running=self.scan_thread is not None,
            is_error=self.scan_thread is None and blocking > 0,
            is_pending=self.scan_thread is None and blocking == 0,
        )

    def exclude_failed_files(self, file_ids: list[int]) -> None:
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请勿重复操作")
            return
        if self.scan_thread is not None:
            self.failed_page.set_status(
                "索引任务仍在运行，请先暂停并等待安全点"
            )
            return
        selected = {
            int(row["id"]): row
            for row in self.failed_page.rows_by_scope["blocking"]
            if int(row["id"]) in set(file_ids)
        }
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
                for row in selected.values()
            )
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        self.start_scope_exclusion(list(selected), reason)

    def start_scope_exclusion(self, file_ids: list[int], reason: str) -> None:
        self._start_scope_update(file_ids, reason=reason, force_complete=False)

    def start_scope_repair(self, reason: str) -> None:
        self._start_scope_update([], reason=reason, force_complete=True)

    def _start_scope_update(
        self,
        file_ids: list[int],
        *,
        reason: str,
        force_complete: bool,
    ) -> None:
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请勿重复操作")
            return
        if self.scan_thread is not None:
            self.failed_page.set_status("索引任务仍在运行，请先等待当前任务结束")
            return
        self.pending_search = False
        if self.search_thread is not None:
            self.cancel_search()
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
        readiness = self.db.index_readiness()
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

    def restore_excluded_files(self, file_ids: list[int]) -> None:
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
        try:
            changed = self.db.restore_files_to_index(
                file_ids,
                reason="用户恢复纳入并重试",
                operation_source="ui",
            )
        except Exception as exc:
            self.failed_page.set_status(f"恢复失败：{exc}")
            return
        self.refresh_all()
        self.failed_page.scope_tabs.setCurrentIndex(0)
        self.failed_page.set_status(f"已恢复 {changed} 个文件，正在重新尝试")
        self.start_scan()

    def request_force_complete(self) -> None:
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请稍候")
            return
        if self.scan_thread is not None:
            self.failed_page.set_status("索引任务仍在运行，请先等待当前任务结束")
            return
        readiness = self.db.index_readiness()
        if bool(readiness.get("ready")):
            self.refresh_all()
            self.failed_page.set_status("当前索引已经就绪，无需修复")
            return
        blockers = self.db.failed_files(limit=2000)
        if not blockers:
            reasons = ", ".join(
                str(value) for value in readiness.get("not_ready_reasons", [])
            ) or "索引状态尚未收敛"
            dialog = QMessageBox(self)
            dialog.setWindowTitle("修复并开放搜索")
            dialog.setIcon(QMessageBox.Icon.Warning)
            dialog.setText("文件已经全部处理，但搜索索引状态仍未完成。")
            dialog.setInformativeText(
                "程序将后台清理残留任务、重建全文索引并重新开放搜索；源文件不会被修改。"
            )
            dialog.setDetailedText(f"未就绪原因：{reasons}")
            dialog.setStandardButtons(
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
            )
            dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
            if dialog.exec() == QMessageBox.StandardButton.Yes:
                self.start_scope_repair("用户确认修复零阻断但未就绪的索引状态")
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("强力完成本次索引")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"先对剩余 {len(blockers)} 个阻断项执行最后一次恢复重试。"
        )
        dialog.setInformativeText(
            "如果仍无法完成，系统会再次列出文件并请你确认排除；不会删除源文件。"
        )
        dialog.setStandardButtons(
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel
        )
        dialog.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if dialog.exec() != QMessageBox.StandardButton.Yes:
            return
        self.force_complete_after_retry = True
        self.failed_page.set_status("正在执行最后一次恢复重试")
        self.start_scan()

    def confirm_force_complete_remaining(self) -> None:
        if self.exclusion_thread is not None:
            self.failed_page.set_status("搜索索引正在后台更新，请稍候")
            return
        blockers = self.db.failed_files(limit=2000)
        readiness = self.db.index_readiness()
        if bool(readiness["ready"]):
            self.refresh_all()
            self.failed_page.set_status("最后一次恢复重试已成功，索引现已完成")
            return
        if not blockers:
            self.start_scope_repair("最后一次恢复后修复残留索引状态")
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("确认排除并开放搜索")
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setText(
            f"最后一次恢复后仍有 {len(blockers)} 个文件无法完成。是否排除这些文件并开放搜索？"
        )
        dialog.setInformativeText(
            "被排除文件的文件名、路径和正文不会进入搜索结果；可在“已人工排除”中恢复重试。"
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
            self.failed_page.set_status("已取消强力完成，索引仍保持未完成状态")
            return
        self.start_scope_repair("用户确认多次恢复后排除剩余阻断项并开放搜索")

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
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            return
        if QMessageBox.question(self, "删除搜索范围", "只删除索引和配置，不会删除原文件。确认继续？") != QMessageBox.StandardButton.Yes:
            return
        self.db.remove_root(root_id)
        self.refresh_all()

    def toggle_root(self, root_id: int, enabled: bool) -> None:
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            return
        self.db.set_root_enabled(root_id, enabled)
        self.refresh_all()

    def start_scan(self, *, performance_mode: bool = False) -> None:
        if self.exclusion_thread is not None:
            self.index_page.task_label.setText("搜索索引正在后台更新，请稍候")
            self.top_bar.set_index_status("正在更新搜索索引...", is_running=True)
            return
        if self.scan_thread is not None:
            self.switch_page("index")
            return
        self.pending_monitor_scan = False
        if self.search_thread is not None:
            self.pending_search = False
            self.cancel_search()
        self.search_page.set_index_ready(
            False,
            {
                **self.db.index_readiness(),
                "active_runs": 1,
                "ready": False,
            },
        )
        self.hide_preview()
        self.scan_thread = QThread()
        self.scan_worker = ScanWorker(
            self.db.db_path,
            self.settings,
            performance_mode=performance_mode,
        )
        self.scan_worker.moveToThread(self.scan_thread)
        self.scan_thread.started.connect(self.scan_worker.run)
        self.scan_worker.progress.connect(self.on_scan_progress)
        self.scan_worker.finished.connect(self.on_scan_finished)
        self.scan_worker.failed.connect(self.on_scan_failed)
        self.scan_worker.finished.connect(self.scan_thread.quit)
        self.scan_worker.failed.connect(self.scan_thread.quit)
        self.scan_thread.finished.connect(self.cleanup_scan_thread)
        self.index_page.set_performance_mode(performance_mode)
        self.top_bar.set_index_status(
            "正在准备性能模式..." if performance_mode else "正在索引...",
            is_running=True,
        )
        self.index_page.set_task_running(True)
        self.failed_page.set_index_running(True)
        self.scan_thread.start()

    def confirm_performance_scan(self) -> None:
        answer = QMessageBox.question(
            self,
            "启用性能模式",
            performance_mode_notice() + "\n\n确认开始性能模式索引？",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.start_scan(performance_mode=True)

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
            self.scan_worker.cancel(force=force)

    def on_scan_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        if payload.get("stage") == "performance_profile":
            profile = payload.get("performance_profile")
            if isinstance(profile, dict):
                self.index_page.set_performance_profile(profile)
                self.top_bar.set_index_status("性能模式正在索引...", is_running=True)
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
            self.top_bar.set_index_status(
                phase_label or self.index_page.pause_state_text(),
                is_running=True,
            )
            return
        indexed = int(payload.get("indexed") or 0)
        scanned = int(payload.get("scanned") or 0)
        total = int(payload.get("total_files") or scanned)
        completed = int(payload.get("completed_files") or indexed)
        failed = int(payload.get("failed") or 0)
        current = str(payload.get("current_file") or "")
        stage = str(payload.get("stage") or "indexing")
        phase_label = str(payload.get("phase_label") or "正在索引")
        eta_seconds = int(payload.get("eta_seconds") or 0)
        eta_ready = bool(payload.get("eta_ready"))
        eta_text = format_remaining_single(eta_seconds, eta_ready)
        active_elapsed = int(payload.get("active_elapsed_seconds") or 0)
        active_queue = str(
            payload.get("queue") or payload.get("active_queue") or ""
        )
        active_count = int(payload.get("active_file_count") or 0)
        active_phase = str(payload.get("active_phase") or "")
        active_completed = int(payload.get("active_completed_units") or 0)
        active_total = int(payload.get("active_total_units") or 0)
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
        text = f"{phase_label} {completed:,} / {max(total, completed):,}"
        self.top_bar.set_index_status(text, is_running=True)
        self.index_page.set_scan_progress(
            completed,
            total,
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
        self.index_page.set_task_running(False)
        if self.closing:
            return
        self.refresh_all()
        summary_text, summary_tooltip = format_index_run_summary(payload)
        if summary_text:
            self.index_page.show_run_summary(summary_text, summary_tooltip)
        if self.force_complete_after_retry:
            self.force_complete_after_retry = False
            self.pending_force_complete_confirmation = True

    def on_scan_failed(self, message: str) -> None:
        self.index_page.set_task_running(False)
        if self.closing:
            return
        self.top_bar.set_index_status("索引任务失败", is_error=True)
        QMessageBox.critical(self, "索引任务失败", message)
        if self.force_complete_after_retry:
            self.force_complete_after_retry = False
            self.pending_force_complete_confirmation = True

    def cleanup_scan_thread(self) -> None:
        self.scan_thread = None
        self.scan_worker = None
        self.index_page.set_task_running(False)
        self.failed_page.set_index_running(False)
        self.refresh_failed_page()
        self.update_index_status()
        if self.pending_force_complete_confirmation and not self.closing:
            self.pending_force_complete_confirmation = False
            QTimer.singleShot(0, self.confirm_force_complete_remaining)
        self._finish_close_if_idle()

    def request_search(self) -> None:
        if self.exclusion_thread is not None:
            self.search_page.set_status("索引正在更新，请稍候")
            return
        if not self.search_page.index_ready():
            self.search_page.show_index_not_ready_state()
            return
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
        if not self.search_page.text() or not self.search_page.index_ready():
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
            include_ocr_fuzzy=self.search_page.include_ocr_fuzzy(),
            ocr_min_confidence=self.settings.ocr_min_confidence,
            case_sensitive=self.search_page.case_sensitive(),
            page_size=min(max(self.settings.page_size, 10), 100),
            max_results=self.settings.max_results,
            page=self.page,
        )
        self.search_thread = QThread()
        self.search_worker = SearchWorker(self.db.db_path, query)
        self.search_worker.moveToThread(self.search_thread)
        self.search_thread.started.connect(self.search_worker.run)
        self.search_worker.finished.connect(self.on_search_finished)
        self.search_worker.partial.connect(self.on_search_partial)
        self.search_worker.progress.connect(self.on_search_progress)
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
        self.total_confirmed = page.available_results
        self.search_page.set_results(page)
        if self.settings.save_search_history and self.search_worker is not None:
            self.db.add_search_history(self.search_worker.query.text)
            self.search_page.set_history(self.db.search_history())
        if page.total_confirmed <= 0:
            self.hide_preview()

    def on_search_partial(self, page: object) -> None:
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

    def on_search_failed(self, message: str) -> None:
        self.search_page.clear_timing()
        self.search_page.set_status(f"搜索失败：{message}")
        self.search_page.show_interrupted_state("搜索未完成")

    def cleanup_search_thread(self) -> None:
        self.search_page.set_running(False)
        self.search_thread = None
        self.search_worker = None
        if self.pending_search and not self.closing:
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

    def reindex_file(self, path: str) -> None:
        if self.exclusion_thread is not None:
            self.search_page.set_status("索引正在更新，请稍候")
            return
        self.db.invalidate_file(path)
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
        self.failed_page.set_status(f"已导出：{target}")

    def save_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.settings_service.save(settings)
        self.search_page.apply_settings(settings)
        self.refresh_file_monitor()
        self.search_page.set_status("设置已保存")

    def closeEvent(self, event: QCloseEvent) -> None:
        self.file_monitor.stop()
        if (
            self.scan_thread is None
            and self.search_thread is None
            and self.exclusion_thread is None
        ):
            event.accept()
            return
        if self.closing:
            event.ignore()
            return
        self.closing = True
        self.pending_search = False
        self.pending_monitor_scan = False
        self.cancel_scan(force=True)
        self.cancel_search()
        self.cancel_scope_exclusion()
        self.hide()
        self.force_close_timer.start()
        event.ignore()

    def _finish_close_if_idle(self) -> None:
        if (
            self.closing
            and self.scan_thread is None
            and self.search_thread is None
            and self.exclusion_thread is None
        ):
            self._force_exit()

    def _force_exit(self) -> None:
        if self.closing:
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
    clear_history_requested = Signal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.setObjectName("Page")
        self._stats: dict[str, int] = {}
        self._has_roots = False
        self._index_ready = False
        self._readiness: dict[str, object] = {}
        self.debounce = QTimer(self)
        self.debounce.setSingleShot(True)
        self.debounce.timeout.connect(self.search_requested.emit)
        self.auto_search_enabled = False
        self._history: list[str] = []
        self.partial_results_visible = False
        self._estimator = SearchTimeEstimator()
        self._active_estimate_context: SearchEstimateContext | None = None
        self._build()
        self.apply_settings(settings)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        self.search_box = SearchBox()
        self.search_box.input.textChanged.connect(self._on_text_changed)
        self.search_box.search_requested.connect(self.search_requested.emit)
        self.search_box.stop_requested.connect(self.stop_requested.emit)
        self.search_box.history_requested.connect(self._show_history)
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
        for label in ("全部", "PDF", "Word", "Excel", "PowerPoint", "图片", "文本/日志", "压缩包"):
            self.file_type_combo.addItem("全部格式" if label == "全部" else label, label)
        self.scope_combo = chip_combo()
        self.scope_combo.addItem("全部范围", None)
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
        status_row.addWidget(self.timing_label)
        layout.addLayout(status_row)

        self.search_progress = QProgressBar()
        self.search_progress.setTextVisible(False)
        self.search_progress.setFixedHeight(4)
        self.search_progress.setVisible(False)
        layout.addWidget(self.search_progress)

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
        self.search_box.update_clear_button()
        if not text.strip():
            self.debounce.stop()
            self.show_idle_state()
            return
        if self.auto_search_enabled:
            self.debounce.start()

    def _filter_changed(self) -> None:
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

    def set_index_ready(
        self,
        ready: bool,
        readiness: dict[str, object] | None = None,
    ) -> None:
        self._index_ready = bool(ready)
        self._readiness = dict(readiness or {})
        for widget in (
            self.search_box,
            self.mode_combo,
            self.file_type_combo,
            self.scope_combo,
            self.more_button,
            self.advanced,
        ):
            widget.setEnabled(self._index_ready)
        if not self._index_ready:
            self.debounce.stop()
            self.show_index_not_ready_state()
        elif self.content_stack.currentWidget() is self.empty_state:
            self.show_idle_state()

    def index_ready(self) -> bool:
        return self._index_ready

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

    def include_ocr_fuzzy(self) -> bool:
        return self.advanced.ocr_fuzzy.isChecked()

    def case_sensitive(self) -> bool:
        return self.advanced.case_sensitive.isChecked()

    def set_running(self, running: bool) -> None:
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
            detail = f"{checked:,}/{total:,} · 已找到 {confirmed:,} 个文件"
        else:
            self.search_progress.setRange(0, 0)
            detail = f"候选 {checked:,} · 已找到 {confirmed:,} 个文件" if checked else ""
        if elapsed_ms >= 2_000:
            slow_reason = str(payload.get("slow_reason") or "")
            suffix = f" · {detail}" if detail else ""
            if slow_reason:
                suffix += f" · {slow_reason}"
            suffix += f" · 已用时 {format_elapsed(elapsed_ms)}"
            self.set_status(phase + suffix)
        else:
            self.set_status(phase + (f" · {detail}" if detail else ""))

    def set_partial_results(self, page: object) -> None:
        if page is None or int(page.total_confirmed) <= 0:
            return
        self.partial_results_visible = True
        self.page_label.setText(f"第 {page.page} 页")
        self.content_stack.setCurrentWidget(self.result_view)
        self.result_view.set_results(page.results, self.text())
        self.pager.setVisible(False)
        self.set_status(
            f"已先找到 {page.total_confirmed} 个文件名/路径结果，正在继续搜索正文..."
        )

    def set_results(self, page: object) -> None:
        self.partial_results_visible = False
        if self._active_estimate_context is not None:
            self._estimator.observe(self._active_estimate_context, int(page.elapsed_ms))
            self._active_estimate_context = None
        self.page_label.setText(f"第 {page.page} 页")
        if page.truncated:
            count_text = f"找到 {page.total_confirmed} 条结果，仅显示前 {page.available_results} 条"
        else:
            count_text = f"找到 {page.total_confirmed} 条结果"
        self.set_status(f"{count_text} · 候选 {page.total_candidates}")
        self.timing_label.setText(f"实际用时 {format_elapsed(int(page.elapsed_ms))}")
        self.timing_label.setVisible(True)
        if page.total_confirmed <= 0:
            self.show_no_results()
            return
        self.content_stack.setCurrentWidget(self.result_view)
        self.result_view.set_results(page.results, self.text())
        self.pager.setVisible(page.available_results > page.page_size)

    def apply_settings(self, settings: AppSettings) -> None:
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
            complete = self._stats.get("complete_files", 0)
            eligible = self._stats.get("eligible_files", 0)
            video = self._stats.get("video_excluded", 0)
            excluded = self._stats.get("manual_excluded_files", 0)
            video_text = f"，另排除视频 {video:,} 个" if video else ""
            excluded_text = f"，已人工排除 {excluded:,} 个" if excluded else ""
            self.empty_state.set_content(
                "输入关键词开始搜索",
                "支持搜索 PDF、Word、Excel、PowerPoint、文本、日志和图片中的文字\n"
                f"当前范围完成 {complete:,}/{eligible:,}{excluded_text}{video_text}",
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
        complete = int(self._readiness.get("complete_files") or 0)
        eligible = int(self._readiness.get("eligible_files") or 0)
        blocking = int(self._readiness.get("blocking_files") or 0)
        active = int(self._readiness.get("active_runs") or 0)
        unready_roots = int(self._readiness.get("unready_roots") or 0)
        excluded = int(self._readiness.get("manual_excluded_files") or 0)
        excluded_text = f"，已人工排除 {excluded:,} 个" if excluded else ""
        if active:
            subtitle = (
                f"正在建立当前范围索引：已完成 {complete:,}/{eligible:,}{excluded_text}。"
                "范围内文件达到允许终态后会自动开放搜索。"
            )
        elif blocking:
            subtitle = (
                f"当前范围已完成 {complete:,}/{eligible:,}{excluded_text}，仍有 {blocking:,} 个阻断项。"
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
                subtitle = "文件已经全部处理，但" + "、".join(details) + "。请在“未成功索引”中执行强力完成。"
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
    search_requested = Signal()
    stop_requested = Signal()
    history_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("SearchBox")
        icon = QLabel("⌕")
        icon.setObjectName("SearchIcon")
        self.input = QLineEdit()
        self.input.setObjectName("SearchInput")
        self.input.setPlaceholderText("搜索文件名、正文、表格、幻灯片和图片文字")
        self.input.returnPressed.connect(self.search_requested.emit)
        self.clear_button = QPushButton()
        self.clear_button.setObjectName("SearchActionButton")
        self.clear_button.setText("清空")
        self.clear_button.setFixedSize(48, 30)
        self.clear_button.setToolTip("清空搜索内容")
        self.clear_button.clicked.connect(self.input.clear)
        self.history_button = QPushButton()
        self.history_button.setObjectName("SearchActionButton")
        self.history_button.setText("历史")
        self.history_button.setFixedSize(48, 30)
        self.history_button.setToolTip("搜索历史")
        self.history_button.setEnabled(False)
        self.history_button.setVisible(False)
        self.history_button.clicked.connect(self.history_requested.emit)
        self.stop_button = QPushButton("停止")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        self.stop_button.setVisible(False)
        self.action_separator = QFrame()
        self.action_separator.setObjectName("SearchActionSeparator")
        self.action_separator.setFixedSize(1, 24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 10, 0)
        layout.setSpacing(8)
        layout.addWidget(icon)
        layout.addWidget(self.input, 1)
        layout.addWidget(self.action_separator)
        layout.addWidget(self.history_button)
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
    performance_scan_requested = Signal()
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
        self.start_button = QPushButton("更新全部")
        self.performance_button = QPushButton("性能模式更新")
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
        self.task_file.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.task_file.setMinimumWidth(80)
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
        self.task_label.setText(
            "性能模式已应用 · "
            f"普通 {normal} / Office {office} / PDF {pdf} / ZIP {zip_workers} / OCR {ocr} "
            f"· 内存 {memory} MB · {disk}"
        )
        self.task_label.setToolTip(str(profile))

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
            self.start_button.setText("更新全部")
            self.performance_button.setText("性能模式更新")
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
        self.scan_requested.emit()

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
        current_name = Path(current).name if current else ""
        current_display = compact_text(current_name, 44)
        self.task_file.setText(current_display)
        self.task_file.setToolTip(current)
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
        self.task_units.setText(units_text)
        mode_label = "性能模式" if getattr(self, "performance_mode", False) else "普通模式"
        video_suffix = f"·排除视频 {excluded_video}" if excluded_video else ""
        self.task_label.setText(
            f"{mode_label}·总体 {completed:,}/{max(total, completed):,}"
            f"·失败 {failed}{video_suffix}"
        )
        self.task_eta.setText(eta_text)
        if indeterminate:
            self.task_progress.setRange(0, 0)
        else:
            self.task_progress.setRange(0, max(total, completed, 1))
            self.task_progress.setValue(completed)


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
        top.addWidget(update_button)
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
    retry_requested = Signal()
    refresh_requested = Signal()
    export_requested = Signal(list)
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
        self.index_running = False
        self.exclusion_running = False
        self.readiness: dict[str, object] = {"ready": True, "repairable": False}
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

        filters = QHBoxLayout()
        self.status_filter = chip_combo()
        self.extension_filter = chip_combo()
        self.select_all = QCheckBox("全选当前筛选")
        self.retry_button = QPushButton("重新尝试")
        self.exclude_button = QPushButton("从当前索引范围排除")
        self.restore_button = QPushButton("恢复纳入并重试")
        self.force_complete_button = QPushButton("强力完成本次索引")
        self.export_button = QPushButton("导出明细")
        self.open_log_button = QPushButton("打开日志目录")
        self.exclusion_progress = QProgressBar()
        self.exclusion_progress.setTextVisible(False)
        self.exclusion_progress.setFixedWidth(150)
        self.exclusion_progress.setVisible(False)
        self.cancel_exclusion_button = QPushButton("取消索引更新")
        self.cancel_exclusion_button.setVisible(False)
        self.retry_button.setObjectName("PrimaryButton")
        self.force_complete_button.setObjectName("PrimaryButton")
        self.retry_button.clicked.connect(self.retry_requested.emit)
        self.exclude_button.clicked.connect(self._emit_exclude)
        self.restore_button.clicked.connect(self._emit_restore)
        self.force_complete_button.clicked.connect(
            self.force_complete_requested.emit
        )
        self.cancel_exclusion_button.clicked.connect(
            self.cancel_exclusion_requested.emit
        )
        self.export_button.clicked.connect(lambda: self.export_requested.emit(self.visible_rows()))
        self.open_log_button.clicked.connect(self.open_log_dir)
        filters.addWidget(self.subtitle)
        filters.addSpacing(12)
        filters.addWidget(self.status_filter)
        filters.addWidget(self.extension_filter)
        filters.addWidget(self.select_all)
        filters.addStretch(1)
        layout.addLayout(filters)

        actions = QHBoxLayout()
        self.status = QLabel("")
        self.status.setObjectName("InlineStatus")
        actions.addWidget(self.status, 1)
        actions.addWidget(self.exclusion_progress)
        actions.addWidget(self.cancel_exclusion_button)
        actions.addWidget(self.retry_button)
        actions.addWidget(self.exclude_button)
        actions.addWidget(self.restore_button)
        actions.addWidget(self.force_complete_button)
        actions.addWidget(self.export_button)
        actions.addWidget(self.open_log_button)
        layout.addLayout(actions)

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
        self.scope_tabs.currentChanged.connect(self._scope_changed)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        self.extension_filter.currentIndexChanged.connect(self.apply_filters)
        self.select_all.toggled.connect(self._select_visible)

    def set_rows(
        self,
        blocking: list[object],
        excluded: list[object] | None = None,
        metadata: list[object] | None = None,
    ) -> None:
        self.rows_by_scope = {
            "blocking": [self._normalize_row(row, "blocking") for row in blocking],
            "excluded": [self._normalize_row(row, "excluded") for row in (excluded or [])],
            "metadata": [self._normalize_row(row, "metadata") for row in (metadata or [])],
        }
        self.scope_tabs.setTabText(0, f"阻断项 {len(self.rows_by_scope['blocking'])}")
        self.scope_tabs.setTabText(1, f"已人工排除 {len(self.rows_by_scope['excluded'])}")
        self.scope_tabs.setTabText(2, f"仅元数据完成 {len(self.rows_by_scope['metadata'])}")
        self._scope_changed(self.scope_tabs.currentIndex())

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
        self.subtitle.setText(
            f"{len(self.rows)} 个阻断项"
            if scope == "blocking"
            else f"{len(self.rows)} 个已人工排除文件"
            if scope == "excluded"
            else f"{len(self.rows)} 个仅元数据完成文件"
        )
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
        self.select_all.setChecked(False)
        self.retry_button.setVisible(scope == "blocking")
        self.exclude_button.setVisible(scope == "blocking")
        self.restore_button.setVisible(scope == "excluded")
        self.force_complete_button.setVisible(
            True
        )
        self._update_action_states()
        self.apply_filters()

    def set_index_running(self, running: bool) -> None:
        self.index_running = bool(running)
        self._update_action_states()

    def set_readiness(self, readiness: dict[str, object]) -> None:
        self.readiness = dict(readiness)
        self._update_force_complete_state()

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
        busy = self.index_running or self.exclusion_running
        scope = self.scope_tabs.currentIndex()
        self.retry_button.setEnabled(scope == 0 and not busy)
        self.exclude_button.setEnabled(scope == 0 and not busy)
        self.restore_button.setEnabled(scope == 1 and not busy)
        self.select_all.setEnabled(not self.exclusion_running)
        self._update_force_complete_state()

    def _update_force_complete_state(self) -> None:
        blockers = len(self.rows_by_scope["blocking"])
        ready = bool(self.readiness.get("ready", blockers == 0))
        repairable = bool(self.readiness.get("repairable", blockers > 0))
        reasons = [
            str(value) for value in self.readiness.get("not_ready_reasons", [])
        ]
        enabled = (
            not ready
            and repairable
            and not self.index_running
            and not self.exclusion_running
        )
        self.force_complete_button.setEnabled(enabled)
        if self.exclusion_running:
            tooltip = "搜索索引正在后台更新，完成后可使用强力完成"
        elif self.index_running:
            tooltip = "索引任务正在运行，结束后可使用强力完成"
        elif blockers:
            tooltip = f"仍有 {blockers} 个阻断项，可执行最终重试并排除后开放搜索"
        elif enabled:
            labels = {
                "unfinished_tasks": "残留解析任务",
                "content_fts_dirty": "全文索引待发布",
                "full_batch_incomplete": "完整批次未收敛",
                "unready_root": "搜索范围未就绪",
                "unpublished_candidate": "候选索引未发布",
            }
            detail = "、".join(labels.get(reason, reason) for reason in reasons)
            tooltip = f"文件已处理完成，可修复：{detail or '索引状态未收敛'}"
        elif ready:
            tooltip = "当前索引已经就绪"
        else:
            tooltip = "当前状态暂不可修复，请查看未就绪原因"
        self.force_complete_button.setToolTip(tooltip)

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

    def apply_filters(self) -> None:
        rows = self.visible_rows()
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
                checkbox.setCheckState(Qt.CheckState.Unchecked)
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

    def selected_file_ids(self) -> list[int]:
        selected = []
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 0)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                selected.append(int(item.data(Qt.ItemDataRole.UserRole)))
        return selected

    def _select_visible(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for row_index in range(self.table.rowCount()):
            item = self.table.item(row_index, 0)
            if item is not None and item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(state)

    def _emit_exclude(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids:
            self.set_status("请先选择需要排除的阻断项")
            return
        self.exclude_requested.emit(file_ids)

    def _emit_restore(self) -> None:
        file_ids = self.selected_file_ids()
        if not file_ids:
            self.set_status("请先选择需要恢复纳入的文件")
            return
        self.restore_requested.emit(file_ids)

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
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        host = QWidget()
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(12)

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
        host_layout.addStretch(1)
        scroll.setWidget(host)
        layout.addWidget(scroll, 1)

        save = QPushButton("保存设置")
        save.setObjectName("PrimaryButton")
        save.clicked.connect(self.apply)
        layout.addWidget(save, alignment=Qt.AlignmentFlag.AlignRight)

        self.auto_search.toggled.connect(self.search_delay.setEnabled)
        self.search_delay.setEnabled(self.auto_search.isChecked())
        self.enable_ocr.toggled.connect(self._update_ocr_controls)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self._update_ocr_controls(self.enable_ocr.isChecked())

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
    label_bits = [f"{label_prefix}完成"]
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
    layout.setColumnStretch(0, 1)
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
