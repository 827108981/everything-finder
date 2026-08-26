from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class IndexedLocation:
    location_text: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    sheet_name: str | None = None
    cell_start: str | None = None
    cell_end: str | None = None
    line_start: int | None = None
    line_end: int | None = None


def open_file(path: str | Path) -> None:
    target = _physical_target(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    if os.name == "nt":
        os.startfile(str(target))  # type: ignore[attr-defined]
    else:
        subprocess.Popen(["xdg-open", str(target)])


def open_parent_folder(path: str | Path) -> None:
    target = _physical_target(path)
    parent = target.parent if target.suffix else target
    if not parent.exists():
        raise FileNotFoundError(str(parent))
    if os.name == "nt":
        subprocess.Popen(["explorer", "/select,", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(parent)])


def open_file_at_location(path: str | Path, location: str | object = "") -> str:
    """Open a file and request the indexed location when the viewer supports it."""
    target = _physical_target(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    indexed_location = _coerce_location(location)
    extension = target.suffix.lower()
    if os.name == "nt":
        if extension == ".pdf":
            page = indexed_location.page_number or _extract_page_number(
                indexed_location.location_text
            )
            if page:
                result = _open_pdf_page(target, page)
                if result:
                    return result
        elif extension in {".doc", ".docx"}:
            result = _open_word_location(target, indexed_location)
            if result:
                return result
        elif extension in {".xls", ".xlsx", ".xlsm"}:
            result = _open_excel_location(target, indexed_location)
            if result:
                return result
        elif extension in {".ppt", ".pptx"}:
            result = _open_powerpoint_location(target, indexed_location)
            if result:
                return result
        elif extension in {".txt", ".log", ".csv", ".md", ".xml", ".json"}:
            line = indexed_location.line_start or _extract_line_number(
                indexed_location.location_text
            )
            if line:
                result = _open_text_line(target, line)
                if result:
                    return result
    open_file(target)
    if indexed_location.location_text.strip():
        return "已打开文件，但当前格式或本机阅读器不支持指定位置定位"
    return "已打开文件"


def _coerce_location(value: str | object) -> IndexedLocation:
    if isinstance(value, IndexedLocation):
        return value
    if isinstance(value, str):
        return IndexedLocation(location_text=value)
    getter = value.get if isinstance(value, dict) else lambda name, default=None: getattr(value, name, default)
    return IndexedLocation(
        location_text=str(getter("location_text", "") or ""),
        page_number=_positive_int(getter("page_number")),
        slide_number=_positive_int(getter("slide_number")),
        sheet_name=_optional_text(getter("sheet_name")),
        cell_start=_optional_text(getter("cell_start")),
        cell_end=_optional_text(getter("cell_end")),
        line_start=_positive_int(getter("line_start")),
        line_end=_positive_int(getter("line_end")),
    )


def _positive_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _extract_page_number(location_text: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*页", location_text or "")
    if not match:
        return None
    page = int(match.group(1))
    return page if page > 0 else None


def _extract_slide_number(location_text: str) -> int | None:
    match = re.search(r"第\s*(\d+)\s*张幻灯片", location_text or "")
    if not match:
        return None
    slide = int(match.group(1))
    return slide if slide > 0 else None


def _extract_line_number(location_text: str) -> int | None:
    match = re.search(r"第\s*(\d+)(?:\s*-\s*\d+)?\s*行", location_text or "")
    if not match:
        return None
    line = int(match.group(1))
    return line if line > 0 else None


def _extract_excel_location(location_text: str) -> tuple[str, int] | None:
    match = re.search(
        r"(?:Sheet|工作表)[：:]\s*(.+?)[；;]\s*第\s*(\d+)\s*行",
        location_text or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1).strip(), int(match.group(2))


def _open_pdf_page(target: Path, page: int) -> str | None:
    sumatra = _find_executable(
        ("SumatraPDF.exe",),
        (
            ("LOCALAPPDATA", "SumatraPDF/SumatraPDF.exe"),
            ("ProgramFiles", "SumatraPDF/SumatraPDF.exe"),
            ("ProgramFiles(x86)", "SumatraPDF/SumatraPDF.exe"),
        ),
    )
    if sumatra:
        subprocess.Popen([sumatra, "-page", str(page), str(target)])
        return f"已请求 SumatraPDF 定位到第 {page} 页"
    acrobat = _find_executable(
        ("AcroRd32.exe", "Acrobat.exe"),
        (
            ("ProgramFiles", "Adobe/Acrobat DC/Acrobat/Acrobat.exe"),
            ("ProgramFiles", "Adobe/Acrobat Reader DC/Reader/AcroRd32.exe"),
            ("ProgramFiles(x86)", "Adobe/Acrobat Reader DC/Reader/AcroRd32.exe"),
        ),
    )
    if acrobat:
        subprocess.Popen([acrobat, "/A", f"page={page}", str(target)])
        return f"已请求 Acrobat 定位到第 {page} 页"
    foxit = _find_executable(
        ("FoxitPDFReader.exe", "FoxitReader.exe"),
        (
            ("ProgramFiles", "Foxit Software/Foxit PDF Reader/FoxitPDFReader.exe"),
            ("ProgramFiles(x86)", "Foxit Software/Foxit PDF Reader/FoxitPDFReader.exe"),
        ),
    )
    if foxit:
        subprocess.Popen([foxit, "/A", f"page={page}", str(target)])
        return f"已请求 Foxit 定位到第 {page} 页"
    browser = _find_executable(
        ("msedge.exe", "chrome.exe"),
        (
            ("ProgramFiles(x86)", "Microsoft/Edge/Application/msedge.exe"),
            ("ProgramFiles", "Microsoft/Edge/Application/msedge.exe"),
            ("ProgramFiles", "Google/Chrome/Application/chrome.exe"),
            ("ProgramFiles(x86)", "Google/Chrome/Application/chrome.exe"),
            ("LOCALAPPDATA", "Google/Chrome/Application/chrome.exe"),
        ),
    )
    if browser:
        uri = target.resolve().as_uri() + f"#page={page}"
        subprocess.Popen([browser, "--new-window", uri])
        return f"已请求浏览器定位到 PDF 第 {page} 页"
    return None


def _open_word_location(target: Path, location: str | object) -> str | None:
    indexed_location = _coerce_location(location)
    location_text = indexed_location.location_text
    paragraph = re.search(r"正文第\s*(\d+)\s*段", location_text or "")
    table_row = re.search(r"表格\s*(\d+)[，,]\s*第\s*(\d+)\s*行", location_text or "")
    page = indexed_location.page_number or _extract_page_number(location_text)
    if not paragraph and not table_row and not page:
        return None
    opened = False
    try:
        pythoncom, client = _office_com()
        app = client.DispatchEx("Word.Application")
        app.Visible = True
        document = app.Documents.Open(str(target), ReadOnly=True, AddToRecentFiles=False)
        opened = True
        if paragraph:
            index = int(paragraph.group(1))
            target_range = document.Paragraphs(index).Range
            target_range.Select()
            app.ActiveWindow.ScrollIntoView(target_range, True)
            if int(app.Selection.Start) != int(target_range.Start):
                raise RuntimeError("Word 段落定位校验未通过")
            return f"已定位到 Word 正文第 {index} 段"
        if table_row:
            table_index = int(table_row.group(1))
            row_index = int(table_row.group(2))
            target_range = document.Tables(table_index).Rows(row_index).Range
            target_range.Select()
            app.ActiveWindow.ScrollIntoView(target_range, True)
            if int(app.Selection.Start) != int(target_range.Start):
                raise RuntimeError("Word 表格定位校验未通过")
            return f"已定位到 Word 表格 {table_index} 第 {row_index} 行"
        assert page is not None
        target_range = document.GoTo(What=1, Which=1, Count=page)
        target_range.Select()
        app.ActiveWindow.ScrollIntoView(target_range, True)
        actual_page = int(app.Selection.Information(3))
        if actual_page != page:
            raise RuntimeError(f"Word 实际定位到第 {actual_page} 页")
        return f"已定位到 Word 第 {page} 页"
    except Exception as exc:
        return f"Word 已打开，但未能确认指定位置：{exc}" if opened else None
    finally:
        if "pythoncom" in locals():
            pythoncom.CoUninitialize()


def _open_excel_location(target: Path, location: str | object) -> str | None:
    indexed_location = _coerce_location(location)
    sheet_name = indexed_location.sheet_name
    row = _cell_row(indexed_location.cell_start) or indexed_location.line_start
    if not sheet_name or not row:
        extracted = _extract_excel_location(indexed_location.location_text)
        if extracted is not None:
            sheet_name, row = extracted
    if not sheet_name or not row:
        return None
    opened = False
    try:
        pythoncom, client = _office_com()
        app = client.DispatchEx("Excel.Application")
        app.Visible = True
        workbook = app.Workbooks.Open(str(target), ReadOnly=True)
        opened = True
        sheet = workbook.Worksheets(sheet_name)
        sheet.Activate()
        app.Goto(sheet.Cells(row, 1), True)
        if str(app.ActiveSheet.Name) != sheet_name or int(app.ActiveCell.Row) != row:
            raise RuntimeError("Excel 工作表或行定位校验未通过")
        return f"已定位到 Excel 工作表“{sheet_name}”第 {row} 行"
    except Exception as exc:
        return f"Excel 已打开，但未能确认指定位置：{exc}" if opened else None
    finally:
        if "pythoncom" in locals():
            pythoncom.CoUninitialize()


def _open_powerpoint_location(target: Path, location: str | object) -> str | None:
    indexed_location = _coerce_location(location)
    slide = indexed_location.slide_number or _extract_slide_number(
        indexed_location.location_text
    )
    if slide is None:
        return None
    pythoncom_module = None
    errors: list[str] = []
    try:
        pythoncom_module, client = _office_com()
        for prog_id in ("PowerPoint.Application", "KWPP.Application"):
            for method_name in ("GetActiveObject", "Dispatch", "DispatchEx"):
                try:
                    method = getattr(client, method_name)
                    app = method(prog_id)
                    app.Visible = True
                    presentation = _find_open_presentation(app, target)
                    if presentation is None:
                        presentation = _open_presentation(app, target)
                    window = _presentation_window(presentation, app)
                    window.View.GotoSlide(slide)
                    actual_slide = int(window.View.Slide.SlideIndex)
                    if actual_slide != slide:
                        raise RuntimeError(f"实际定位到第 {actual_slide} 张幻灯片")
                    return f"已定位到 PowerPoint 第 {slide} 张幻灯片"
                except Exception as exc:
                    errors.append(
                        _automation_error_detail(
                            prog_id=prog_id,
                            method_name=method_name,
                            stage="open_and_goto_slide",
                            target_number=slide,
                            exc=exc,
                        )
                    )
    except Exception as exc:
        errors.append(
            _automation_error_detail(
                prog_id="PowerPoint/WPS",
                method_name="initialize",
                stage="initialize_com",
                target_number=slide,
                exc=exc,
            )
        )
    finally:
        if pythoncom_module is not None:
            pythoncom_module.CoUninitialize()
    logger.warning(
        "PowerPoint location failed: path=%s slide=%s attempts=%s",
        target,
        slide,
        " | ".join(errors),
    )
    open_file(target)
    return (
        f"已打开文件，但 PowerPoint 自动化接口不可用，"
        f"未能定位到第 {slide} 张幻灯片"
    )


def _cell_row(cell_reference: str | None) -> int | None:
    if not cell_reference:
        return None
    match = re.search(r"(\d+)$", cell_reference.strip())
    return _positive_int(match.group(1)) if match else None


def _find_open_presentation(app: object, target: Path) -> object | None:
    presentations = getattr(app, "Presentations")
    target_resolved = target.resolve()
    for index in range(1, int(presentations.Count) + 1):
        presentation = presentations(index)
        try:
            if Path(str(presentation.FullName)).resolve() == target_resolved:
                return presentation
        except (OSError, RuntimeError, ValueError):
            continue
    return None


def _open_presentation(app: object, target: Path) -> object:
    presentations = getattr(app, "Presentations")
    try:
        return presentations.Open(str(target), ReadOnly=True, WithWindow=True)
    except TypeError:
        return presentations.Open(str(target), True, False, True)


def _presentation_window(presentation: object, app: object) -> object:
    windows = getattr(presentation, "Windows")
    if int(windows.Count) > 0:
        return windows(1)
    active_window = getattr(app, "ActiveWindow", None)
    if active_window is None:
        raise RuntimeError("演示文稿没有可用窗口")
    return active_window


def _automation_error_detail(
    *,
    prog_id: str,
    method_name: str,
    stage: str,
    target_number: int,
    exc: Exception,
) -> str:
    hresult = getattr(exc, "hresult", None)
    if not isinstance(hresult, int) and exc.args and isinstance(exc.args[0], int):
        hresult = int(exc.args[0])
    hresult_text = f"0x{hresult & 0xFFFFFFFF:08X}" if isinstance(hresult, int) else "unknown"
    return (
        f"prog_id={prog_id} method={method_name} stage={stage} "
        f"target={target_number} hresult={hresult_text} error={exc}"
    )


def _open_text_line(target: Path, line: int) -> str | None:
    editor = _find_executable(
        ("code.cmd", "code.exe"),
        (("LOCALAPPDATA", "Programs/Microsoft VS Code/Code.exe"),),
    )
    if editor:
        subprocess.Popen([editor, "-g", f"{target}:{line}:1"])
        return f"已请求编辑器定位到第 {line} 行"
    notepad_plus = _find_executable(
        ("notepad++.exe",),
        (
            ("ProgramFiles", "Notepad++/notepad++.exe"),
            ("ProgramFiles(x86)", "Notepad++/notepad++.exe"),
        ),
    )
    if notepad_plus:
        subprocess.Popen([notepad_plus, f"-n{line}", str(target)])
        return f"已请求 Notepad++ 定位到第 {line} 行"
    return None


def _office_com() -> tuple[object, object]:
    import pythoncom
    import win32com.client

    pythoncom.CoInitialize()
    return pythoncom, win32com.client


def _find_executable(
    names: tuple[str, ...],
    known_locations: tuple[tuple[str, str], ...],
) -> str | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return found
    for environment_name, relative_path in known_locations:
        root = os.environ.get(environment_name)
        if not root:
            continue
        candidate = Path(root) / Path(relative_path)
        if candidate.is_file():
            return str(candidate)
    return None


def _physical_target(path: str | Path) -> Path:
    path_text = str(path)
    if " > " in path_text:
        path_text = path_text.split(" > ", 1)[0]
    return Path(path_text)
