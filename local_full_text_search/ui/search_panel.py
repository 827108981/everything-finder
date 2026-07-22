from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLineEdit, QPushButton, QWidget


class SearchPanel(QWidget):
    search_requested = Signal()
    stop_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.input = QLineEdit()
        self.input.setPlaceholderText("输入精确关键词、短语或多个关键词")
        self.mode = QComboBox()
        self.mode.addItem("精确包含", "exact")
        self.mode.addItem("全部关键词", "all")
        self.mode.addItem("任一关键词", "any")
        self.mode.addItem("完整短语", "phrase")
        self.mode.addItem("仅文件名", "filename")
        self.search_button = QPushButton("搜索")
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)

        layout = QHBoxLayout(self)
        layout.addWidget(self.input, 1)
        layout.addWidget(self.mode)
        layout.addWidget(self.search_button)
        layout.addWidget(self.stop_button)

        self.search_button.clicked.connect(self.search_requested.emit)
        self.input.returnPressed.connect(self.search_requested.emit)
        self.stop_button.clicked.connect(self.stop_requested.emit)

    def text(self) -> str:
        return self.input.text().strip()

    def mode_value(self) -> str:
        return str(self.mode.currentData())

    def set_running(self, running: bool) -> None:
        self.search_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
