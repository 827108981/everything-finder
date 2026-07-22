from __future__ import annotations

import html
import re
from datetime import datetime

from PySide6.QtCore import QSize, Signal, Qt
from PySide6.QtGui import QAction, QKeyEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QVBoxLayout,
)

from local_full_text_search.models.search_result import SearchResult


class ResultView(QListWidget):
    open_requested = Signal(str)
    open_folder_requested = Signal(str)
    reindex_requested = Signal(str)
    selected_result_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ResultList")
        self.setSpacing(10)
        self.setUniformItemSizes(False)
        self.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._results: list[SearchResult] = []
        self._cards: list[ResultCard] = []
        self.itemDoubleClicked.connect(self._open_current)
        self.currentRowChanged.connect(self._emit_selected)
        self.currentRowChanged.connect(self._refresh_selection_style)
        self.customContextMenuRequested.connect(self._show_context_menu)

    def set_results(self, results: list[SearchResult], query_text: str = "") -> None:
        self._results = results
        self._cards = []
        self.clear()
        for result in results:
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, 118))
            self.addItem(item)
            card = ResultCard(result, query_text)
            self._cards.append(card)
            self.setItemWidget(item, card)
        if results:
            self.setCurrentRow(0)
        else:
            self._emit_selected(-1)
        self._refresh_selection_style(self.currentRow())

    def current_result(self) -> SearchResult | None:
        row = self.currentRow()
        if 0 <= row < len(self._results):
            return self._results[row]
        return None

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            result = self.current_result()
            if result:
                self.open_requested.emit(result.file_path)
                return
        super().keyPressEvent(event)

    def _emit_selected(self, row: int) -> None:
        if 0 <= row < len(self._results):
            self.selected_result_changed.emit(self._results[row])
        else:
            self.selected_result_changed.emit(None)

    def _refresh_selection_style(self, row: int) -> None:
        for index, card in enumerate(self._cards):
            card.set_selected(index == row)

    def _open_current(self, _item: QListWidgetItem) -> None:
        result = self.current_result()
        if result:
            self.open_requested.emit(result.file_path)

    def _show_context_menu(self, pos: object) -> None:
        item = self.itemAt(pos)
        if item is None:
            return
        self.setCurrentItem(item)
        result = self.current_result()
        if result is None:
            return
        menu = QMenu(self)
        open_action = QAction("打开文件", self)
        folder_action = QAction("打开所在文件夹", self)
        copy_path_action = QAction("复制路径", self)
        copy_context_action = QAction("复制命中内容", self)
        reindex_action = QAction("重新索引该文件", self)
        open_action.triggered.connect(lambda: self.open_requested.emit(result.file_path))
        folder_action.triggered.connect(lambda: self.open_folder_requested.emit(result.file_path))
        copy_path_action.triggered.connect(lambda: QApplication.clipboard().setText(result.file_path))
        copy_context_action.triggered.connect(lambda: QApplication.clipboard().setText(result.context))
        reindex_action.triggered.connect(lambda: self.reindex_requested.emit(result.file_path))
        for action in (open_action, folder_action, copy_path_action, copy_context_action, reindex_action):
            menu.addAction(action)
        menu.exec(self.viewport().mapToGlobal(pos))


class ResultCard(QFrame):
    def __init__(self, result: SearchResult, query_text: str) -> None:
        super().__init__()
        self.setObjectName("ResultCard")
        self.setProperty("selected", False)
        modified = datetime.fromtimestamp(result.modified_time).strftime("%Y-%m-%d %H:%M")

        type_icon = QLabel(icon_for_extension(result.extension, result.source_type))
        type_icon.setObjectName("TypeIcon")
        type_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title = QLabel(result.filename)
        title.setObjectName("ResultTitle")
        title.setToolTip(result.file_path)
        title.setWordWrap(True)

        hit = QLabel(f"{result.hit_count} 处命中")
        hit.setObjectName("HitBadge")

        location = QLabel(f"{format_type(result)} · {result.location_text}")
        location.setObjectName("ResultMeta")

        context = QLabel(highlight_context(result.context or "文件名/路径命中", query_text))
        context.setObjectName("ResultContext")
        context.setTextFormat(Qt.TextFormat.RichText)
        context.setWordWrap(True)

        path = QLabel(f"{result.file_path}    {modified}")
        path.setObjectName("ResultPath")
        path.setToolTip(result.file_path)

        badge_layout = QHBoxLayout()
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(6)
        if result.source_type == "ocr":
            tag = QLabel(ocr_label(result))
            tag.setObjectName("OcrBadge")
            badge_layout.addWidget(tag)
        if result.parse_status and result.parse_status not in {"success", ""}:
            status = QLabel(result.parse_status)
            status.setObjectName("StatusBadge")
            badge_layout.addWidget(status)
        badge_layout.addStretch(1)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(10)
        top.addWidget(type_icon)
        top.addWidget(title, 1)
        top.addWidget(hit)

        body = QVBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(5)
        body.addLayout(top)
        body.addWidget(location)
        body.addWidget(context)
        body.addWidget(path)
        body.addLayout(badge_layout)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)
        self.accent = QFrame()
        self.accent.setObjectName("ResultAccent")
        self.accent.setProperty("selected", False)
        self.accent.setFixedWidth(3)
        layout.addWidget(self.accent)
        layout.addLayout(body, 1)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.accent.setProperty("selected", selected)
        for widget in (self, self.accent):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


def highlight_context(context: str, query_text: str) -> str:
    escaped = html.escape(context)
    terms = [part for part in re.split(r"\s+", query_text.strip()) if part]
    if not terms:
        return escaped
    pattern = re.compile("|".join(re.escape(html.escape(term)) for term in terms if term), re.IGNORECASE)
    return pattern.sub(lambda match: f'<span style="background:#FEF3C7;color:#182230;">{match.group(0)}</span>', escaped)


def ocr_label(result: SearchResult) -> str:
    if result.ocr_confidence is None:
        return "OCR"
    return f"OCR {result.ocr_confidence * 100:.0f}%"


def icon_for_extension(extension: str, source_type: str) -> str:
    if source_type == "ocr":
        return "OCR"
    mapping = {
        ".pdf": "PDF",
        ".docx": "DOC",
        ".doc": "DOC",
        ".xlsx": "XLS",
        ".xlsm": "XLS",
        ".xls": "XLS",
        ".pptx": "PPT",
        ".ppt": "PPT",
        ".zip": "ZIP",
        ".txt": "TXT",
        ".log": "LOG",
    }
    return mapping.get(extension.lower(), "FILE")


def format_type(result: SearchResult) -> str:
    extension = result.extension.lower()
    if result.source_type == "ocr":
        return "图片 OCR" if extension in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"} else "OCR"
    if extension in {".xlsx", ".xlsm", ".xls"}:
        return "Excel"
    if extension == ".pdf":
        return "PDF"
    if extension in {".docx", ".doc"}:
        return "Word"
    if extension in {".pptx", ".ppt"}:
        return "PowerPoint"
    if extension in {".txt", ".log"}:
        return "文本/日志"
    return extension.upper().lstrip(".") or "文件"
