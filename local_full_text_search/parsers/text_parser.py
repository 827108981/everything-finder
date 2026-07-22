from __future__ import annotations

import codecs
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import TEXT_EXTENSIONS
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.base_parser import BaseParser


class TextParser(BaseParser):
    name = "text"

    def __init__(self, lines_per_block: int = 500) -> None:
        self.lines_per_block = lines_per_block

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in TEXT_EXTENSIONS

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        encoding = detect_text_encoding(file_path)
        block_lines: list[str] = []
        start_line = 1
        block_index = 0
        with file_path.open("r", encoding=encoding, errors="replace", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                block_lines.append(line.rstrip("\n"))
                if len(block_lines) >= self.lines_per_block:
                    raw = "\n".join(block_lines)
                    yield self.make_block(
                        file_path,
                        block_index,
                        "text",
                        f"第 {start_line}-{line_number} 行",
                        raw,
                        line_start=start_line,
                        line_end=line_number,
                    )
                    block_index += 1
                    start_line = line_number + 1
                    block_lines = []
        if block_lines:
            end_line = start_line + len(block_lines) - 1
            yield self.make_block(
                file_path,
                block_index,
                "text",
                f"第 {start_line}-{end_line} 行",
                "\n".join(block_lines),
                line_start=start_line,
                line_end=end_line,
            )


def detect_text_encoding(file_path: Path) -> str:
    with file_path.open("rb") as handle:
        sample = handle.read(65536)
    if sample.startswith(codecs.BOM_UTF8):
        return "utf-8-sig"
    if sample.startswith(codecs.BOM_UTF16_LE):
        return "utf-16-le"
    if sample.startswith(codecs.BOM_UTF16_BE):
        return "utf-16-be"
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "utf-16"):
        try:
            sample.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes
    except ImportError as exc:
        return "utf-8"
    detected = from_bytes(sample).best()
    if detected and detected.encoding:
        return detected.encoding
    return "utf-8"
