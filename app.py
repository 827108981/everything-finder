from __future__ import annotations

import sys
import tempfile
from importlib import resources
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.config.constants import APP_DISPLAY_NAME
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.logging_service import configure_logging
from local_full_text_search.services.settings_service import SettingsService


def run_self_test() -> int:
    """Non-interactive smoke test used after packaging."""

    try:
        from PySide6.QtWidgets import QApplication
        from local_full_text_search.ui.main_window import MainWindow

        settings_service = SettingsService()
        settings = settings_service.load()
        db = DatabaseManager()
        db.initialize()
        app = QApplication.instance() or QApplication(["LocalFullTextSearch", "--self-test"])
        apply_light_theme(app)
        window = MainWindow(db, settings, settings_service)
        window.close()
        Path("self_test_result.txt").write_text("SELF_TEST_OK\n", encoding="utf-8")
        print("SELF_TEST_OK")
        return 0
    except Exception as exc:
        Path("self_test_result.txt").write_text(f"SELF_TEST_FAILED: {exc}\n", encoding="utf-8")
        print(f"SELF_TEST_FAILED: {exc}")
        return 1


def run_core_validation() -> int:
    """Build a temporary multi-format index and verify exact search hits."""

    result_path = Path("core_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_validation_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            _create_validation_files(root)

            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            engine = SearchEngine(db)

            checks = {
                "TXT_VALIDATION_HIT": "txt",
                "PDF_VALIDATION_HIT": "pdf",
                "DOCX_VALIDATION_HIT": "docx",
                "XLSX_VALIDATION_HIT": "xlsx",
                "PPTX_VALIDATION_HIT": "pptx",
                "OCR TEST 123": "ocr_image",
            }
            failures: list[str] = []
            for term, label in checks.items():
                page = engine.search(SearchQuery(text=term, mode="exact"))
                if page.total_confirmed < 1:
                    failures.append(f"{label}:{term}")
            if failures:
                raise RuntimeError("未命中: " + ", ".join(failures))
            message = (
                "CORE_VALIDATION_OK\n"
                f"scanned={summary.scanned}; indexed={summary.indexed}; "
                f"skipped={summary.skipped}; failed={summary.failed}; unsupported={summary.unsupported}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"CORE_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"CORE_VALIDATION_FAILED: {exc}")
        return 1


def _create_validation_files(root: Path) -> None:
    (root / "sample.txt").write_text("TXT_VALIDATION_HIT\nBS-2800M2", encoding="utf-8")

    from docx import Document
    doc = Document()
    doc.add_paragraph("DOCX_VALIDATION_HIT")
    doc.save(root / "sample.docx")

    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet["A1"] = "XLSX_VALIDATION_HIT"
    workbook.save(root / "sample.xlsx")

    from pptx import Presentation
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    text_box = slide.shapes.add_textbox(914400, 914400, 3657600, 914400)
    text_box.text = "PPTX_VALIDATION_HIT"
    presentation.save(root / "sample.pptx")

    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF_VALIDATION_HIT")
    pdf.save(root / "sample.pdf")
    pdf.close()

    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 86) if font_path.exists() else ImageFont.load_default()
    draw.text((48, 80), "OCR TEST 123", fill="black", font=font)
    image.save(root / "sample_ocr.png")


def apply_light_theme(app: object) -> None:
    try:
        qss = resources.files("local_full_text_search.ui.styles").joinpath("light.qss").read_text(encoding="utf-8")
        app.setStyleSheet(qss)
    except Exception:
        # A missing stylesheet should never prevent the search tool from opening.
        pass


def main() -> int:
    configure_logging()
    if "--self-test" in sys.argv:
        return run_self_test()
    if "--validate-core" in sys.argv:
        return run_core_validation()
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from local_full_text_search.ui.main_window import MainWindow
    except ImportError as exc:
        print("缺少 PySide6，无法启动图形界面。请先运行: python -m pip install -r requirements.txt")
        print(exc)
        return 2

    settings_service = SettingsService()
    settings = settings_service.load()
    db = DatabaseManager()
    db.initialize()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    apply_light_theme(app)
    try:
        window = MainWindow(db, settings, settings_service)
        window.show()
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
