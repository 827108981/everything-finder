from __future__ import annotations

import html
import math
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

from local_full_text_search.models.search_result import SearchHit, SearchResult


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
            card = ResultCard(result, query_text)
            item = QListWidgetItem()
            item.setSizeHint(QSize(100, card.preferred_row_height(self.viewport().width())))
            self.addItem(item)
            self._cards.append(card)
            self.setItemWidget(item, card)
        if results:
            self.setCurrentRow(0)
        else:
            self._emit_selected(-1)
        self._refresh_selection_style(self.currentRow())

    def resizeEvent(self, event: object) -> None:
        super().resizeEvent(event)
        self._refresh_item_heights()

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

    def _refresh_item_heights(self) -> None:
        width = self.viewport().width()
        for index, card in enumerate(self._cards):
            item = self.item(index)
            if item is not None:
                item.setSizeHint(QSize(100, card.preferred_row_height(width)))

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
        self.result = result
        self.query_text = query_text
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
        hit.setAlignment(Qt.AlignmentFlag.AlignCenter)

        location = QLabel(f"{format_type(result)} · {result.location_text}")
        location.setObjectName("ResultMeta")
        location.setWordWrap(True)

        context = QLabel(render_context_html(result, query_text))
        context.setObjectName("ResultContext")
        context.setTextFormat(Qt.TextFormat.RichText)
        context.setWordWrap(True)

        path = QLabel(f"{wrap_path(result.file_path)}\n{modified}")
        path.setObjectName("ResultPath")
        path.setToolTip(result.file_path)
        path.setWordWrap(True)

        badge_layout = QHBoxLayout()
        badge_layout.setContentsMargins(0, 0, 0, 0)
        badge_layout.setSpacing(6)
        if any(hit.source_type == "ocr" for hit in result_hits(result)):
            tag = QLabel(ocr_label(result))
            tag.setObjectName("OcrBadge")
            badge_layout.addWidget(tag)
        if result.has_fuzzy_match:
            fuzzy = QLabel("疑似匹配")
            fuzzy.setObjectName("StatusBadge")
            badge_layout.addWidget(fuzzy)
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

    def preferred_row_height(self, width: int) -> int:
        usable_width = max(width - 150, 260)
        chars_per_line = max(28, usable_width // 7)
        title_lines = min(wrapped_line_count(self.result.filename, chars_per_line), 3)
        location_lines = min(wrapped_line_count(self.result.location_text, chars_per_line), 2)
        path_lines = min(wrapped_line_count(self.result.file_path, max(24, chars_per_line - 8)) + 1, 4)
        context_lines = 0
        for hit in visible_hits(self.result):
            context_lines += 1
            context_lines += min(wrapped_line_count(hit.context or "文件名/路径命中", chars_per_line), 3)
        if hidden_hit_count(self.result) > 0:
            context_lines += 1
        height = 46 + title_lines * 22 + location_lines * 18 + context_lines * 20 + path_lines * 17
        if self.result.parse_status and self.result.parse_status not in {"success", ""}:
            height += 24
        return max(150, min(height, 360))

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.accent.setProperty("selected", selected)
        for widget in (self, self.accent):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


def render_context_html(result: SearchResult, query_text: str) -> str:
    parts: list[str] = []
    for hit in visible_hits(result):
        label = html.escape(hit_label(hit))
        context = highlight_context(hit.context or "文件名/路径命中", query_text)
        parts.append(
            f'<span style="color:#667085;font-size:12px;">{label}</span><br>{context}'
        )
    hidden = hidden_hit_count(result)
    if hidden > 0:
        parts.append(f'<span style="color:#667085;font-size:12px;">另有 {hidden} 段命中，可在右侧预览查看</span>')
    return "<br><br>".join(parts) if parts else highlight_context(result.context or "文件名/路径命中", query_text)


def highlight_context(context: str, query_text: str) -> str:
    escaped = html.escape(context)
    terms = [part for part in re.split(r"\s+", query_text.strip()) if part]
    if not terms:
        return escaped
    pattern = re.compile("|".join(re.escape(html.escape(term)) for term in terms if term), re.IGNORECASE)
    return pattern.sub(lambda match: f'<span style="background:#FEF3C7;color:#182230;">{match.group(0)}</span>', escaped)


def result_hits(result: SearchResult) -> list[SearchHit]:
    if result.matches:
        return result.matches
    return [
        SearchHit(
            block_id=result.block_id,
            location_text=result.location_text,
            context=result.context,
            hit_count=result.hit_count,
            source_type=result.source_type,
            ocr_confidence=result.ocr_confidence,
            is_fuzzy=result.has_fuzzy_match,
        )
    ]


def visible_hits(result: SearchResult) -> list[SearchHit]:
    return result_hits(result)[:3]


def hidden_hit_count(result: SearchResult) -> int:
    return max(0, len(result_hits(result)) - len(visible_hits(result)))


def hit_label(hit: SearchHit) -> str:
    location = hit.location_text or "命中"
    if hit.source_type == "metadata":
        return "文件名/路径"
    if hit.source_type == "ocr":
        return f"OCR · {location}"
    return f"正文 · {location}"


def wrapped_line_count(text: str, chars_per_line: int) -> int:
    if not text:
        return 1
    total = 0
    for line in text.splitlines() or [text]:
        total += max(1, math.ceil(len(line) / max(1, chars_per_line)))
    return total


def wrap_path(path: str) -> str:
    return path.replace("\\", "\\\u200b").replace("/", "/\u200b")


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
