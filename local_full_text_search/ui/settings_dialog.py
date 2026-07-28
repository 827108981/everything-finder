from __future__ import annotations

from PySide6.QtWidgets import QCheckBox, QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QVBoxLayout

from local_full_text_search.config.defaults import AppSettings


class SettingsDialog(QDialog):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        self.setWindowTitle("设置")
        self.settings = settings
        self.page_size = QSpinBox()
        self.page_size.setRange(10, 1000)
        self.page_size.setValue(settings.page_size)
        self.enable_ocr = QCheckBox("启用 OCR")
        self.enable_ocr.setChecked(settings.enable_ocr)
        self.ocr_images = QCheckBox("索引图片 OCR")
        self.ocr_images.setChecked(settings.ocr_images)
        self.monitor_changes = QCheckBox("检测文件变化并提示更新")
        self.monitor_changes.setChecked(settings.monitor_file_changes)

        form = QFormLayout()
        form.addRow("每页结果数", self.page_size)
        form.addRow(self.enable_ocr)
        form.addRow(self.ocr_images)
        form.addRow(self.monitor_changes)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def apply_to_settings(self) -> AppSettings:
        self.settings.page_size = int(self.page_size.value())
        self.settings.enable_ocr = self.enable_ocr.isChecked()
        self.settings.ocr_images = self.ocr_images.isChecked()
        self.settings.monitor_file_changes = self.monitor_changes.isChecked()
        return self.settings
