from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.rag.models import RagAnswer, RagQuery, SourceRef


class RagPage(QWidget):
    ask_requested = Signal(object)
    index_requested = Signal()
    cancel_query_requested = Signal()
    cancel_index_requested = Signal()
    open_location_requested = Signal(str, object)
    settings_requested = Signal()

    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.settings = settings
        self.setObjectName("Page")
        self._build()
        self.apply_settings(settings)

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)

        heading = QLabel("智能问答")
        heading.setObjectName("SectionTitle")
        layout.addWidget(heading)

        controls = QHBoxLayout()
        self.root_combo = QComboBox()
        self.root_combo.setObjectName("FilterControl")
        self.root_combo.addItem("全部目录", None)
        controls.addWidget(self.root_combo)
        self.pdf_check = QCheckBox("PDF")
        self.docx_check = QCheckBox("DOCX")
        self.txt_check = QCheckBox("TXT")
        for widget in (self.pdf_check, self.docx_check, self.txt_check):
            controls.addWidget(widget)
        self.include_ocr = QCheckBox("包含 OCR")
        controls.addWidget(self.include_ocr)
        controls.addWidget(QLabel("OCR 最低置信度"))
        self.ocr_confidence = QDoubleSpinBox()
        self.ocr_confidence.setRange(0.0, 1.0)
        self.ocr_confidence.setDecimals(2)
        self.ocr_confidence.setSingleStep(0.05)
        controls.addWidget(self.ocr_confidence)
        controls.addStretch(1)
        self.index_button = QPushButton("构建/更新智能索引")
        controls.addWidget(self.index_button)
        layout.addLayout(controls)

        query_row = QHBoxLayout()
        self.question = QLineEdit()
        self.question.setPlaceholderText("用自然语言提问，例如：索引中断后怎样恢复？")
        self.question.returnPressed.connect(self.submit_question)
        query_row.addWidget(self.question, 1)
        self.ask_button = QPushButton("提问")
        self.ask_button.setObjectName("PrimaryButton")
        self.ask_button.clicked.connect(self.submit_question)
        query_row.addWidget(self.ask_button)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.cancel_query_requested.emit)
        query_row.addWidget(self.stop_button)
        self.cancel_index_button = QPushButton("取消构建")
        self.cancel_index_button.setEnabled(False)
        self.cancel_index_button.clicked.connect(self.cancel_index_requested.emit)
        query_row.addWidget(self.cancel_index_button)
        self.settings_button = QPushButton("模型设置")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        query_row.addWidget(self.settings_button)
        layout.addLayout(query_row)

        self.status = QLabel("请先配置本地 Embedding 模型与本地聊天模型。")
        self.status.setObjectName("InlineStatus")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.answer = QPlainTextEdit()
        self.answer.setReadOnly(True)
        self.answer.setPlaceholderText("答案会显示在这里，并附上可打开的来源。")
        self.answer.setMinimumHeight(140)
        layout.addWidget(self.answer, 1)

        sources_heading = QLabel("引用与证据")
        sources_heading.setObjectName("SectionTitle")
        layout.addWidget(sources_heading)
        self.sources_scroll = QScrollArea()
        self.sources_scroll.setWidgetResizable(True)
        self.sources_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sources_host = QWidget()
        self.sources_layout = QVBoxLayout(self.sources_host)
        self.sources_layout.setContentsMargins(0, 0, 0, 0)
        self.sources_layout.setSpacing(8)
        self.sources_layout.addStretch(1)
        self.sources_scroll.setWidget(self.sources_host)
        layout.addWidget(self.sources_scroll, 2)

        self.index_button.clicked.connect(self.index_requested.emit)
        self.set_task_state("idle")

    def set_roots(self, roots: list[object]) -> None:
        selected = self.root_combo.currentData()
        self.root_combo.blockSignals(True)
        self.root_combo.clear()
        self.root_combo.addItem("全部目录", None)
        for row in roots:
            root_id = _row_value(row, "id")
            path = str(_row_value(row, "path") or "")
            enabled = bool(_row_value(row, "enabled", True))
            if root_id is None or not enabled:
                continue
            self.root_combo.addItem(path, int(root_id))
        index = self.root_combo.findData(selected)
        self.root_combo.setCurrentIndex(index if index >= 0 else 0)
        self.root_combo.blockSignals(False)

    def apply_settings(self, settings: AppSettings) -> None:
        self.settings = settings
        self.include_ocr.setChecked(bool(settings.rag_include_ocr))
        self.ocr_confidence.setValue(float(settings.ocr_min_confidence))
        self.pdf_check.setChecked(True)
        self.docx_check.setChecked(True)
        self.txt_check.setChecked(True)
        self.index_button.setEnabled(bool(settings.rag_enabled))
        self.settings_button.setEnabled(True)
        if not settings.rag_enabled:
            self.status.setText("智能问答已关闭。请在设置中启用并配置本地模型。")

    def make_query(self) -> RagQuery:
        extensions = tuple(
            extension
            for checkbox, extension in (
                (self.pdf_check, ".pdf"),
                (self.docx_check, ".docx"),
                (self.txt_check, ".txt"),
            )
            if checkbox.isChecked()
        )
        root = self.root_combo.currentData()
        return RagQuery(
            question=self.question.text(),
            root_ids=(int(root),) if root is not None else (),
            extensions=extensions,
            include_ocr=self.include_ocr.isChecked(),
            min_ocr_confidence=float(self.ocr_confidence.value()),
            top_k=int(self.settings.rag_top_k),
        )

    def submit_question(self) -> None:
        try:
            query = self.make_query()
        except ValueError as exc:
            self.status.setText(str(exc))
            return
        self.answer.clear()
        self._clear_sources()
        self.ask_requested.emit(query)

    def set_task_state(self, state: str, message: str = "") -> None:
        query_running = state == "query_running"
        index_running = state == "index_running"
        self.ask_button.setEnabled(not query_running and not index_running and self.settings.rag_enabled)
        self.question.setEnabled(not query_running and not index_running)
        self.stop_button.setEnabled(query_running)
        self.index_button.setEnabled(not index_running and not query_running and self.settings.rag_enabled)
        self.cancel_index_button.setEnabled(index_running)
        self.progress.setVisible(query_running or index_running)
        if query_running or index_running:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
        if message:
            self.status.setText(message)

    def set_progress(self, payload: object) -> None:
        if isinstance(payload, dict):
            message = str(payload.get("message") or payload.get("phase_label") or payload.get("stage") or "")
            if message:
                self.status.setText(message)
            return
        message = str(getattr(payload, "message", "") or "")
        if message:
            self.status.setText(message)

    def set_answer(self, answer: RagAnswer) -> None:
        self.answer.setPlainText(answer.answer_text)
        self._clear_sources()
        for citation in answer.citations:
            self._add_source_card(citation.label, citation.source, citation.excerpt)
        if answer.warnings:
            self.status.setText("；".join(answer.warnings))
        elif answer.status == "answered":
            self.status.setText(f"回答完成 · generation {answer.source_generation} · trace {answer.trace_id[:8]}")
        elif answer.status == "insufficient":
            self.status.setText("证据不足，未调用模型生成结论。")
        elif answer.status == "index_stale":
            self.status.setText("智能索引与当前搜索版本不一致，请先更新智能索引。")
        elif answer.status == "validation_failed":
            self.status.setText("模型回答未通过来源检查，下面保留检索证据供核对。")
        else:
            self.status.setText("本地模型暂不可用，请检查设置。")

    def set_roots_status(self, message: str) -> None:
        self.status.setText(message)

    def _add_source_card(self, label: str, source: SourceRef, excerpt: str) -> None:
        card = QFrame()
        card.setObjectName("SettingsCard")
        layout = QVBoxLayout(card)
        header = QHBoxLayout()
        title = QLabel(f"[{label}] {source.filename} · {source.location_text or '正文'}")
        title.setWordWrap(True)
        title.setObjectName("ScopeTitle")
        header.addWidget(title, 1)
        open_button = QPushButton("打开位置")
        open_button.clicked.connect(
            lambda _checked=False, path=source.file_path, ref=source: self.open_location_requested.emit(path, ref)
        )
        header.addWidget(open_button)
        layout.addLayout(header)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(excerpt)
        text.setMaximumHeight(120)
        layout.addWidget(text)
        self.sources_layout.insertWidget(max(0, self.sources_layout.count() - 1), card)

    def _clear_sources(self) -> None:
        while self.sources_layout.count() > 1:
            item = self.sources_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


def _row_value(row: object, name: str, default: object = None) -> object:
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
