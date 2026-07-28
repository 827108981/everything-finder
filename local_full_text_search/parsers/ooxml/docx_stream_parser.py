from __future__ import annotations

import zipfile
from collections.abc import Iterable
from pathlib import Path

from lxml import etree

from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser
from local_full_text_search.parsers.ooxml.common import clear_element, iterparse_end, xml_parts

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W_P = f"{{{W_NS}}}p"
W_TR = f"{{{W_NS}}}tr"
W_TC = f"{{{W_NS}}}tc"
W_T = f"{{{W_NS}}}t"
W_TAB = f"{{{W_NS}}}tab"
W_BR = f"{{{W_NS}}}br"
W_CR = f"{{{W_NS}}}cr"


class DocxStreamParser(BaseParser):
    name = "docx_stream"

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
        return file_path.suffix.lower() == ".docx"

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        try:
            yield from self._parse_stream(file_path, cancel_token)
        except CancelledError:
            raise
        except (OSError, zipfile.BadZipFile, KeyError, etree.XMLSyntaxError, ValueError):
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
            if "word/document.xml" not in archive.namelist():
                raise ValueError("DOCX is missing word/document.xml")
            block_index = 0
            paragraph_no = 0
            table_row_no = 0
            for element in iterparse_end(
                archive,
                "word/document.xml",
                (W_P, W_TR),
                cancel_token,
            ):
                if element.tag == W_P:
                    if _has_ancestor(element, W_TR):
                        continue
                    paragraph_no += 1
                    text = _element_text(element).strip()
                    if text:
                        yield self.make_block(
                            file_path,
                            block_index,
                            "docx_paragraph",
                            f"正文第 {paragraph_no} 段",
                            text,
                            extra={"paragraph_start": paragraph_no, "paragraph_end": paragraph_no},
                        )
                        block_index += 1
                    clear_element(element)
                else:
                    table_row_no += 1
                    cells = []
                    for cell in element.iter(W_TC):
                        cell_text = _element_text(cell).strip()
                        if cell_text:
                            cells.append(cell_text)
                    if cells:
                        yield self.make_block(
                            file_path,
                            block_index,
                            "docx_table_row",
                            f"表格第 {table_row_no} 行",
                            " | ".join(cells),
                            extra={"row_start": table_row_no, "row_end": table_row_no},
                        )
                        block_index += 1
                    clear_element(element)

            seen_parts: set[str] = set()
            header_footer_parts = xml_parts(archive, r"word/(?:header|footer)\d+\.xml")
            for part_name in header_footer_parts:
                if part_name in seen_parts:
                    continue
                seen_parts.add(part_name)
                texts: list[str] = []
                for paragraph in iterparse_end(archive, part_name, W_P, cancel_token):
                    text = _element_text(paragraph).strip()
                    if text:
                        texts.append(text)
                    clear_element(paragraph)
                if texts:
                    part_label = "页眉" if "/header" in part_name else "页脚"
                    yield self.make_block(
                        file_path,
                        block_index,
                        "docx_header_footer",
                        f"{part_label} {Path(part_name).stem}",
                        "\n".join(texts),
                        extra={"part": part_name},
                    )
                    block_index += 1


def _has_ancestor(element: etree._Element, tag: str) -> bool:
    parent = element.getparent()
    while parent is not None:
        if parent.tag == tag:
            return True
        parent = parent.getparent()
    return False


def _element_text(element: etree._Element) -> str:
    pieces: list[str] = []
    for node in element.iter():
        if node.tag == W_T and node.text:
            pieces.append(node.text)
        elif node.tag == W_TAB:
            pieces.append("\t")
        elif node.tag in {W_BR, W_CR}:
            pieces.append("\n")
    return "".join(pieces)
