from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.docx_parser import DocxParser
from local_full_text_search.parsers.pptx_parser import PptxParser
from local_full_text_search.parsers.xlsx_parser import XlsxParser


class LegacyOfficeParser(BaseParser):
    """Convert legacy binary Office documents before parsing.

    The old .doc/.xls/.ppt formats are not reliable to parse directly in pure
    Python. We first try Microsoft Office COM because it preserves content best
    on Windows, then fall back to LibreOffice headless if available. Missing
    converters are reported as converter_missing rather than failed.
    """

    name = "legacy_office"

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".doc", ".xls", ".ppt"}

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        cancel_token.throw_if_cancelled()
        converted = self._convert(file_path)
        if converted is None:
            self.set_status("converter_missing", "CONVERTER_MISSING", "未找到 Microsoft Office 或 LibreOffice 转换器")
            return
        parser = self._parser_for_converted(converted)
        for block in parser.parse(converted, cancel_token):
            block.file_path = str(file_path)
            block.location_text = f"转换自 {file_path.suffix.lower()} > {block.location_text}"
            yield block
        if parser.last_status != "success":
            self.set_status(parser.last_status, parser.last_error_code, parser.last_error_message)

    def _convert(self, file_path: Path) -> Path | None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="legacy_office_", dir=TEMP_DIR))
        target = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
        if self._convert_with_office(file_path, target):
            return target
        if self._convert_with_libreoffice(file_path, temp_dir):
            candidate = temp_dir / f"{file_path.stem}{self._target_suffix(file_path)}"
            if candidate.exists():
                return candidate
        return None

    def _convert_with_office(self, source: Path, target: Path) -> bool:
        if source.suffix.lower() == ".doc":
            return _convert_word(source, target)
        if source.suffix.lower() == ".xls":
            return _convert_excel(source, target)
        if source.suffix.lower() == ".ppt":
            return _convert_powerpoint(source, target)
        return False

    def _convert_with_libreoffice(self, source: Path, output_dir: Path) -> bool:
        soffice = _find_soffice()
        if soffice is None:
            return False
        result = subprocess.run(
            [
                str(soffice),
                "--headless",
                "--convert-to",
                self._libreoffice_target(source),
                "--outdir",
                str(output_dir),
                str(source),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return result.returncode == 0

    def _target_suffix(self, source: Path) -> str:
        return {".doc": ".docx", ".xls": ".xlsx", ".ppt": ".pptx"}[source.suffix.lower()]

    def _libreoffice_target(self, source: Path) -> str:
        return {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}[source.suffix.lower()]

    def _parser_for_converted(self, converted: Path) -> BaseParser:
        suffix = converted.suffix.lower()
        if suffix == ".docx":
            return DocxParser()
        if suffix == ".xlsx":
            return XlsxParser()
        if suffix == ".pptx":
            return PptxParser()
        raise ValueError(f"未知转换格式: {converted}")


def _convert_word(source: Path, target: Path) -> bool:
    try:
        import win32com.client

        app = win32com.client.DispatchEx("Word.Application")
        app.Visible = False
        doc = app.Documents.Open(str(source), ReadOnly=True)
        doc.SaveAs2(str(target), FileFormat=16)
        doc.Close(False)
        app.Quit()
        return target.exists()
    except Exception:
        return False


def _convert_excel(source: Path, target: Path) -> bool:
    try:
        import win32com.client

        app = win32com.client.DispatchEx("Excel.Application")
        app.DisplayAlerts = False
        workbook = app.Workbooks.Open(str(source), ReadOnly=True)
        workbook.SaveAs(str(target), FileFormat=51)
        workbook.Close(False)
        app.Quit()
        return target.exists()
    except Exception:
        return False


def _convert_powerpoint(source: Path, target: Path) -> bool:
    try:
        import win32com.client

        app = win32com.client.DispatchEx("PowerPoint.Application")
        presentation = app.Presentations.Open(str(source), WithWindow=False)
        presentation.SaveAs(str(target), 24)
        presentation.Close()
        app.Quit()
        return target.exists()
    except Exception:
        return False


def _find_soffice() -> Path | None:
    found = shutil.which("soffice")
    if found:
        return Path(found)
    candidates = [
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
    ]
    return next((path for path in candidates if path.exists()), None)
