from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

from lxml import etree

from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.ooxml.common import (
    clear_element,
    iterparse_end,
    safe_part_target,
    xml_root,
)

S_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SHEET = f"{{{S_NS}}}sheet"
ROW = f"{{{S_NS}}}row"
CELL = f"{{{S_NS}}}c"
VALUE = f"{{{S_NS}}}v"
FORMULA = f"{{{S_NS}}}f"
TEXT = f"{{{S_NS}}}t"
SHARED_ITEM = f"{{{S_NS}}}si"
COMMENT = f"{{{S_NS}}}comment"


class XlsxStreamParser(BaseParser):
    name = "xlsx_stream"

    def __init__(
        self,
        fallback: BaseParser | None = None,
        *,
        defer_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.fallback = fallback
        self.defer_normalization = defer_normalization

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in {".xlsx", ".xlsm"}

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        try:
            yield from self._parse_stream(file_path, cancel_token)
        except CancelledError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, ValueError, IndexError):
            if self.fallback is None:
                raise
            self.fallback.reset_status()
            yield from self.fallback.parse(file_path, cancel_token)
            if self.fallback.last_status != "success":
                self.set_status(
                    self.fallback.last_status,
                    self.fallback.last_error_code,
                    self.fallback.last_error_message,
                )

    def _parse_stream(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        with zipfile.ZipFile(file_path) as archive:
            available = set(archive.namelist())
            if "xl/workbook.xml" not in available or "xl/_rels/workbook.xml.rels" not in available:
                raise ValueError("XLSX is missing workbook parts")
            shared_strings = _shared_strings(archive, cancel_token) if "xl/sharedStrings.xml" in available else []
            sheets = _sheet_parts(archive)
            block_index = 0
            for sheet_name, sheet_part in sheets:
                if sheet_part not in available:
                    continue
                comments = _comments_for_sheet(archive, sheet_part, available, cancel_token)
                for row in iterparse_end(archive, sheet_part, ROW, cancel_token):
                    row_number = int(row.get("r") or 0)
                    parts: list[str] = []
                    first_cell: str | None = None
                    last_cell: str | None = None
                    for cell in row.iter(CELL):
                        coordinate = str(cell.get("r") or "")
                        value = _cell_text(cell, shared_strings)
                        if not coordinate or value is None:
                            continue
                        comment = comments.get(coordinate)
                        if comment:
                            value = f"{value}; 批注: {comment}"
                        first_cell = first_cell or coordinate
                        last_cell = coordinate
                        parts.append(f"{coordinate}={value}")
                    if parts:
                        yield self.make_block(
                            file_path,
                            block_index,
                            "xlsx_row",
                            f"Sheet: {sheet_name}; 第 {row_number} 行",
                            " | ".join(parts),
                            sheet_name=sheet_name,
                            cell_start=first_cell,
                            cell_end=last_cell,
                            extra={"row_start": row_number, "row_end": row_number},
                        )
                        block_index += 1
                    clear_element(row)


def _shared_strings(archive: zipfile.ZipFile, cancel_token: CancelToken) -> list[str]:
    strings: list[str] = []
    for item in iterparse_end(archive, "xl/sharedStrings.xml", SHARED_ITEM, cancel_token):
        strings.append("".join(node.text or "" for node in item.iter(TEXT)))
        clear_element(item)
    return strings


def _sheet_parts(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    rels_root = xml_root(archive, "xl/_rels/workbook.xml.rels")
    targets: dict[str, str] = {}
    for relation in rels_root.iter(f"{{{REL_NS}}}Relationship"):
        relation_id = str(relation.get("Id") or "")
        if str(relation.get("Type") or "").endswith("/worksheet"):
            target = safe_part_target("xl/workbook.xml", str(relation.get("Target") or ""))
            if target:
                targets[relation_id] = target
    workbook_root = xml_root(archive, "xl/workbook.xml")
    result: list[tuple[str, str]] = []
    for sheet in workbook_root.iter(SHEET):
        name = str(sheet.get("name") or "Sheet")
        relation_id = str(sheet.get(f"{{{R_NS}}}id") or "")
        target = targets.get(relation_id)
        if target:
            result.append((name, target))
    return result


def _comments_for_sheet(
    archive: zipfile.ZipFile,
    sheet_part: str,
    available: set[str],
    cancel_token: CancelToken,
) -> dict[str, str]:
    sheet_name = Path(sheet_part).name
    rels_part = f"xl/worksheets/_rels/{sheet_name}.rels"
    if rels_part not in available:
        return {}
    root = xml_root(archive, rels_part)
    comments_part: str | None = None
    for relation in root.iter(f"{{{REL_NS}}}Relationship"):
        if str(relation.get("Type") or "").endswith("/comments"):
            comments_part = safe_part_target(sheet_part, str(relation.get("Target") or ""))
            break
    if not comments_part or comments_part not in available:
        return {}
    result: dict[str, str] = {}
    for comment in iterparse_end(archive, comments_part, COMMENT, cancel_token):
        ref = str(comment.get("ref") or "")
        text = "".join(node.text or "" for node in comment.iter(TEXT)).strip()
        if ref and text:
            result[ref] = text
        clear_element(comment)
    return result


def _cell_text(cell: etree._Element, shared_strings: list[str]) -> str | None:
    cell_type = str(cell.get("t") or "")
    formula = cell.find(FORMULA)
    if formula is not None and formula.text:
        return "=" + formula.text
    if cell_type == "inlineStr":
        text = "".join(node.text or "" for node in cell.iter(TEXT))
        return text if text else None
    value = cell.find(VALUE)
    if value is None or value.text is None:
        return None
    raw = value.text
    if cell_type == "s":
        index = int(raw)
        return shared_strings[index]
    if cell_type == "b":
        return "TRUE" if raw == "1" else "FALSE"
    return raw
