from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QWidget

from local_full_text_search.config.constants import FILE_TYPE_GROUPS


class FilterPanel(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.file_type = QComboBox()
        for label in FILE_TYPE_GROUPS:
            self.file_type.addItem(label, label)
        self.case_sensitive = QCheckBox("大小写敏感")
        self.include_ocr = QCheckBox("包含 OCR")
        self.include_ocr.setChecked(True)
        self.search_content = QCheckBox("正文")
        self.search_content.setChecked(True)
        self.search_filename = QCheckBox("文件名")
        self.search_filename.setChecked(True)
        self.search_path = QCheckBox("路径")
        self.search_path.setChecked(True)

        layout = QHBoxLayout(self)
        layout.addWidget(QLabel("类型"))
        layout.addWidget(self.file_type)
        layout.addWidget(self.search_filename)
        layout.addWidget(self.search_path)
        layout.addWidget(self.search_content)
        layout.addWidget(self.include_ocr)
        layout.addWidget(self.case_sensitive)
        layout.addStretch(1)

    def extensions(self) -> list[str]:
        label = str(self.file_type.currentData())
        extensions = FILE_TYPE_GROUPS.get(label, set())
        return sorted(extensions)
