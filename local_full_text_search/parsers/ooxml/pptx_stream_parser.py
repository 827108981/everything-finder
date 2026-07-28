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
    xml_parts,
    xml_root,
)

A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
A_P = f"{{{A_NS}}}p"
A_T = f"{{{A_NS}}}t"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


class PptxStreamParser(BaseParser):
    name = "pptx_stream"

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
        return file_path.suffix.lower() == ".pptx"

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
            slides = xml_parts(archive, r"ppt/slides/slide\d+\.xml")
            if not slides:
                raise ValueError("PPTX contains no slide parts")
            available = set(archive.namelist())
            block_index = 0
            for slide_number, slide_part in enumerate(slides, start=1):
                texts = _paragraph_texts(archive, slide_part, cancel_token)
                if texts:
                    yield self.make_block(
                        file_path,
                        block_index,
                        "pptx_slide",
                        f"第 {slide_number} 张幻灯片",
                        "\n".join(texts),
                        slide_number=slide_number,
                    )
                    block_index += 1
                notes_part = _notes_part_for_slide(archive, slide_part, available)
                if notes_part:
                    notes = _paragraph_texts(archive, notes_part, cancel_token)
                    if notes:
                        yield self.make_block(
                            file_path,
                            block_index,
                            "pptx_notes",
                            f"第 {slide_number} 张幻灯片，备注",
                            "\n".join(notes),
                            slide_number=slide_number,
                            extra={"notes_part": notes_part},
                        )
                        block_index += 1


def _paragraph_texts(
    archive: zipfile.ZipFile,
    part_name: str,
    cancel_token: CancelToken,
) -> list[str]:
    result: list[str] = []
    for paragraph in iterparse_end(archive, part_name, A_P, cancel_token):
        text = "".join(node.text or "" for node in paragraph.iter(A_T)).strip()
        if text:
            result.append(text)
        clear_element(paragraph)
    return result


def _notes_part_for_slide(
    archive: zipfile.ZipFile,
    slide_part: str,
    available: set[str],
) -> str | None:
    slide_name = Path(slide_part).name
    rels_part = f"ppt/slides/_rels/{slide_name}.rels"
    if rels_part not in available:
        return None
    root = xml_root(archive, rels_part)
    for relation in root.iter(f"{{{REL_NS}}}Relationship"):
        if str(relation.get("Type") or "").endswith("/notesSlide"):
            target = safe_part_target(slide_part, str(relation.get("Target") or ""))
            if target in available:
                return target
    return None
