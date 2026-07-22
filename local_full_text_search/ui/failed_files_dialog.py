from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QMessageBox,
    QPushButton,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from local_full_text_search.config.constants import LOG_DIR
from local_full_text_search.core.database import DatabaseManager


class FailedFilesDialog(QDialog):
    def __init__(self, db: DatabaseManager) -> None:
        super().__init__()
        self.setWindowTitle("失败文件")
        self.resize(900, 500)
        self.rows = db.failed_files()
        self.status_filter = QComboBox()
        self.extension_filter = QComboBox()
        self.status_filter.addItem("全部状态", "")
        self.extension_filter.addItem("全部格式", "")
        for value in sorted({str(row["parse_status"]) for row in self.rows}):
            self.status_filter.addItem(value, value)
        for value in sorted({str(row["extension"] or "") for row in self.rows}):
            self.extension_filter.addItem(value or "<无扩展名>", value)
        self.status_filter.currentIndexChanged.connect(self.apply_filters)
        self.extension_filter.currentIndexChanged.connect(self.apply_filters)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(["路径", "扩展名", "状态", "错误码", "原因", "解析器", "时间"])
        self.apply_filters()
        self.export_summary_button = QPushButton("导出汇总")
        self.export_summary_button.clicked.connect(self.export_summary_csv)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("状态"))
        filter_layout.addWidget(self.status_filter)
        filter_layout.addWidget(QLabel("格式"))
        filter_layout.addWidget(self.extension_filter)
        filter_layout.addStretch(1)

        self.export_button = QPushButton("导出失败清单")
        self.open_log_button = QPushButton("打开日志目录")
        self.close_button = QPushButton("关闭")
        self.export_button.clicked.connect(self.export_csv)
        self.open_log_button.clicked.connect(self.open_log_dir)
        self.close_button.clicked.connect(self.accept)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.export_button)
        button_layout.addWidget(self.export_summary_button)
        button_layout.addWidget(self.open_log_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(filter_layout)
        layout.addWidget(self.table)
        layout.addLayout(button_layout)

    def visible_rows(self) -> list[object]:
        status = str(self.status_filter.currentData() or "")
        extension = str(self.extension_filter.currentData() or "")
        rows = []
        for row in self.rows:
            if status and row["parse_status"] != status:
                continue
            if extension and (row["extension"] or "") != extension:
                continue
            rows.append(row)
        return rows

    def apply_filters(self) -> None:
        rows = self.visible_rows()
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = [
                str(row["path"]),
                str(row["extension"] or ""),
                str(row["parse_status"]),
                str(row["parse_error_code"] or ""),
                str(row["parse_error_message"] or ""),
                str(row["parser_name"] or ""),
                str(row["indexed_at"] or ""),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                self.table.setItem(row_index, col, item)

    def export_csv(self) -> None:
        rows = self.visible_rows()
        if not rows:
            QMessageBox.information(self, "导出失败清单", "当前没有失败文件。")
            return
        default_path = Path.home() / "Desktop" / "失败文件清单.csv"
        target, _ = QFileDialog.getSaveFileName(
            self,
            "导出失败清单",
            str(default_path),
            "CSV 文件 (*.csv)",
        )
        if not target:
            return
        try:
            with Path(target).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["路径", "扩展名", "状态", "错误码", "原因", "解析器", "时间"])
                for row in rows:
                    writer.writerow(
                        [
                            str(row["path"]),
                            str(row["extension"] or ""),
                            str(row["parse_status"]),
                            str(row["parse_error_code"] or ""),
                            str(row["parse_error_message"] or ""),
                            str(row["parser_name"] or ""),
                            str(row["indexed_at"] or ""),
                        ]
                    )
            QMessageBox.information(self, "导出失败清单", f"已导出：{target}")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def export_summary_csv(self) -> None:
        rows = self.visible_rows()
        if not rows:
            QMessageBox.information(self, "导出汇总", "当前没有可汇总记录。")
            return
        default_path = Path.home() / "Desktop" / "失败文件汇总.csv"
        target, _ = QFileDialog.getSaveFileName(self, "导出汇总", str(default_path), "CSV 文件 (*.csv)")
        if not target:
            return
        summary: dict[tuple[str, str, str], int] = {}
        for row in rows:
            key = (str(row["extension"] or ""), str(row["parse_status"]), str(row["parse_error_code"] or ""))
            summary[key] = summary.get(key, 0) + 1
        try:
            with Path(target).open("w", newline="", encoding="utf-8-sig") as handle:
                writer = csv.writer(handle)
                writer.writerow(["扩展名", "状态", "错误码", "数量"])
                for (extension, status, code), count in sorted(summary.items(), key=lambda item: item[1], reverse=True):
                    writer.writerow([extension, status, code, count])
            QMessageBox.information(self, "导出汇总", f"已导出：{target}")
        except OSError as exc:
            QMessageBox.warning(self, "导出失败", str(exc))

    def open_log_dir(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(LOG_DIR))  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", str(LOG_DIR)])
        except OSError as exc:
            QMessageBox.warning(self, "打开日志目录失败", str(exc))
