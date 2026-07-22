from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QProgressBar, QVBoxLayout


class IndexDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("索引进度")
        self.label = QLabel("准备索引")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        layout = QVBoxLayout(self)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

    def update_text(self, text: str) -> None:
        self.label.setText(text)
