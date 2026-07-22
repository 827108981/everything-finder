from __future__ import annotations

from datetime import datetime

from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QPixmap, QTextOption
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from local_full_text_search.config.constants import IMAGE_EXTENSIONS
from local_full_text_search.models.search_result import SearchResult


class PreviewPanel(QFrame):
    open_file_requested = Signal(str)
    open_folder_requested = Signal(str)
    close_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("PreviewPanel")
        self.setMinimumWidth(360)
        self.setMaximumWidth(560)
        self.current_path: str | None = None

        heading = QLabel("文件预览")
        heading.setObjectName("PreviewHeading")
        close_button = QPushButton("×")
        close_button.setObjectName("IconButton")
        close_button.setFixedSize(32, 32)
        close_button.setToolTip("关闭预览")
        close_button.clicked.connect(self.close_requested.emit)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.addWidget(heading)
        title_row.addStretch(1)
        title_row.addWidget(close_button)

        self.title = QLabel("选择一个搜索结果")
        self.title.setObjectName("PreviewTitle")
        self.title.setWordWrap(True)
        self.meta = QLabel("单击结果后显示命中位置和上下文")
        self.meta.setObjectName("MutedText")
        self.meta.setWordWrap(True)

        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setObjectName("PreviewImage")
        self.image.setVisible(False)

        self.context = QTextEdit()
        self.context.setObjectName("PreviewContext")
        self.context.setReadOnly(True)
        self.context.setMinimumHeight(180)
        self.context.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.context.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.context.document().setDefaultTextOption(option)

        self.open_button = QPushButton("打开文件")
        self.open_button.setObjectName("PrimaryButton")
        self.open_folder_button = QPushButton("打开所在文件夹")
        self.copy_path_button = QPushButton("复制路径")

        button_layout = QGridLayout()
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)
        button_layout.addWidget(self.open_button, 0, 0)
        button_layout.addWidget(self.open_folder_button, 0, 1)
        button_layout.addWidget(self.copy_path_button, 1, 0, 1, 2)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(18, 16, 18, 16)
        inner_layout.setSpacing(12)
        inner_layout.addLayout(title_row)
        inner_layout.addWidget(self.title)
        inner_layout.addWidget(self.meta)
        inner_layout.addWidget(self.image)
        location_label = QLabel("命中上下文")
        location_label.setObjectName("SectionTitle")
        inner_layout.addWidget(location_label)
        inner_layout.addWidget(self.context, 1)
        inner_layout.addStretch(1)
        inner_layout.addLayout(button_layout)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(inner)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll)

        self.open_button.clicked.connect(self._open_file)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.copy_path_button.clicked.connect(self._copy_path)

    def show_result(self, result: SearchResult | None) -> None:
        if result is None:
            self.current_path = None
            self.title.setText("选择一个搜索结果")
            self.meta.setText("单击结果后显示命中位置和上下文")
            self.context.clear()
            self.image.clear()
            self.image.setVisible(False)
            return
        self.current_path = result.file_path
        modified = datetime.fromtimestamp(result.modified_time).strftime("%Y-%m-%d %H:%M")
        self.title.setText(result.filename)
        type_text = result.extension.upper().lstrip(".") or "文件"
        confidence = f" · OCR 置信度 {result.ocr_confidence * 100:.0f}%" if result.ocr_confidence is not None else ""
        self.meta.setText(
            f"{type_text} · {format_size(result.size_bytes)} · {modified}{confidence}\n{result.location_text}\n{wrap_path(result.file_path)}"
        )
        self.context.setPlainText(result.context or "文件名/路径命中")
        self._load_image_preview(result.file_path, result.extension)

    def _load_image_preview(self, file_path: str, extension: str) -> None:
        if extension.lower() not in IMAGE_EXTENSIONS:
            self.image.clear()
            self.image.setVisible(False)
            return
        pixmap = QPixmap(file_path)
        if pixmap.isNull():
            self.image.setVisible(False)
            return
        self.image.setPixmap(pixmap.scaledToWidth(360, Qt.TransformationMode.SmoothTransformation))
        self.image.setVisible(True)

    def _open_file(self) -> None:
        if self.current_path:
            self.open_file_requested.emit(self.current_path)

    def _open_folder(self) -> None:
        if self.current_path:
            self.open_folder_requested.emit(self.current_path)

    def _copy_path(self) -> None:
        if self.current_path:
            QApplication.clipboard().setText(self.current_path)


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def wrap_path(path: str) -> str:
    return path.replace("\\", "\\\u200b").replace("/", "/\u200b")
