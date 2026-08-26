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
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from local_full_text_search.core.normalizer import contains_cjk
from local_full_text_search.models.search_result import SearchHit, SearchResult


class ResultView(QListWidget):
    open_requested = Signal(str)
    open_folder_requested = Signal(str)
    open_location_requested = Signal(str, object)
    reindex_requested = Signal(str)
    selected_result_changed = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ResultList")
        self.setSpacing(10)
        self.setUniformItemSizes(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
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
            card.open_location_requested.connect(self.open_location_requested.emit)
            card.height_changed.connect(
                lambda current_card=card, current_item=item: current_item.setSizeHint(
                    QSize(100, current_card.preferred_row_height(self.viewport().width()))
                )
            )
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
    height_changed = Signal()
    open_location_requested = Signal(str, object)

    def __init__(self, result: SearchResult, query_text: str) -> None:
        super().__init__()
        self.result = result
        self.query_text = query_text
        self._expanded = False
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

        self.location_panel = QFrame()
        self.location_panel.setObjectName("HitLocationPanel")
        self.location_layout = QVBoxLayout(self.location_panel)
        self.location_layout.setContentsMargins(0, 0, 0, 0)
        self.location_layout.setSpacing(2)

        self.expand_button = QPushButton()
        self.expand_button.setObjectName("LinkButton")
        self.expand_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_button.clicked.connect(self._toggle_expanded)
        self.expand_button.setVisible(hidden_hit_count(result) > 0)
        self._update_expand_button()

        self.path_label = QLabel(
            f"{wrap_path(result.file_path)}\n{format_size(result.size_bytes)} · 修改于 {modified}"
        )
        self.path_label.setObjectName("ResultPath")
        self.path_label.setToolTip(result.file_path)
        self.path_label.setWordWrap(True)

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
        body.addWidget(self.location_panel)
        body.addWidget(self.expand_button, alignment=Qt.AlignmentFlag.AlignLeft)
        body.addWidget(self.path_label)
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
        self._refresh_location_buttons()

    def preferred_row_height(self, width: int) -> int:
        card_width = max(280, width - 8)
        self.setFixedWidth(card_width)
        card_layout = self.layout()
        if card_layout is None:
            return 180
        card_layout.invalidate()
        calculated = card_layout.heightForWidth(card_width)
        hinted = card_layout.sizeHint().height()
        return max(145, calculated, hinted) + 4

    def _toggle_expanded(self) -> None:
        self._expanded = not self._expanded
        self._refresh_location_buttons()
        self._update_expand_button()
        self.height_changed.emit()

    def _refresh_location_buttons(self) -> None:
        while self.location_layout.count():
            item = self.location_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        hits = visible_hits(self.result, expanded=self._expanded)
        for hit in hits:
            button = QPushButton(hit_label(hit))
            button.setObjectName("HitLocationButton")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setToolTip("点击打开文件并尝试定位到此处")
            button.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            button.clicked.connect(
                lambda _checked=False, current_hit=hit: self.open_location_requested.emit(
                    self.result.file_path, current_hit
                )
            )
            self.location_layout.addWidget(button, alignment=Qt.AlignmentFlag.AlignLeft)
        if hidden_hit_count(self.result, expanded=self._expanded) > 0:
            remaining = QLabel(
                f"另有 {hidden_hit_count(self.result, expanded=self._expanded)} 段命中，展开查看"
            )
            remaining.setObjectName("ResultMeta")
            self.location_layout.addWidget(remaining, alignment=Qt.AlignmentFlag.AlignLeft)

    def _update_expand_button(self) -> None:
        total = len(result_hits(self.result))
        if total <= 3:
            self.expand_button.setVisible(False)
            return
        self.expand_button.setText(
            "收起命中位置" if self._expanded else f"展开全部 {total} 处命中位置"
        )

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.accent.setProperty("selected", selected)
        for widget in (self, self.accent):
            widget.style().unpolish(widget)
            widget.style().polish(widget)


def render_context_html(
    result: SearchResult,
    query_text: str,
    *,
    expanded: bool = False,
) -> str:
    parts: list[str] = []
    for hit in visible_hits(result, expanded=expanded):
        label = html.escape(hit_label(hit))
        context = highlight_context(hit.context or "文件名/路径命中", query_text)
        parts.append(
            f'<span style="color:#667085;font-size:12px;">{label}</span><br>{context}'
        )
    hidden = hidden_hit_count(result, expanded=expanded)
    if hidden > 0:
        parts.append(f'<span style="color:#667085;font-size:12px;">另有 {hidden} 段命中</span>')
    return "<br><br>".join(parts) if parts else highlight_context(result.context or "文件名/路径命中", query_text)


def highlight_context(context: str, query_text: str) -> str:
    terms = [part for part in re.split(r"\s+", query_text.strip()) if part]
    if not terms:
        return html.escape(context).replace("\n", "<br>")
    alternatives: list[str] = []
    for term in terms:
        if contains_cjk(term):
            alternatives.append(r"\s*".join(re.escape(character) for character in term))
        else:
            alternatives.append(re.escape(term))
    pattern = re.compile("|".join(alternatives), re.IGNORECASE)
    rendered: list[str] = []
    offset = 0
    for match in pattern.finditer(context):
        rendered.append(html.escape(context[offset : match.start()]))
        rendered.append(
            '<span style="background:#FDE68A;color:#182230;font-weight:600;">'
            f"{html.escape(match.group(0))}</span>"
        )
        offset = match.end()
    rendered.append(html.escape(context[offset:]))
    return "".join(rendered).replace("\n", "<br>")


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
            page_number=result.page_number,
            slide_number=result.slide_number,
            sheet_name=result.sheet_name,
            cell_start=result.cell_start,
            cell_end=result.cell_end,
            line_start=result.line_start,
            line_end=result.line_end,
            ocr_confidence=result.ocr_confidence,
            is_fuzzy=result.has_fuzzy_match,
        )
    ]


def visible_hits(result: SearchResult, *, expanded: bool = False) -> list[SearchHit]:
    hits = result_hits(result)
    return hits if expanded else hits[:3]


def hidden_hit_count(result: SearchResult, *, expanded: bool = False) -> int:
    return max(0, len(result_hits(result)) - len(visible_hits(result, expanded=expanded)))


def hit_label(hit: SearchHit) -> str:
    location = hit.location_text or "命中"
    if hit.source_type == "metadata":
        return "文件名/路径"
    if hit.source_type == "ocr":
        return f"OCR · {location}"
    return f"正文 · {location}"


def wrap_path(path: str) -> str:
    return path.replace("\\", "\\\u200b").replace("/", "/\u200b")


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


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
