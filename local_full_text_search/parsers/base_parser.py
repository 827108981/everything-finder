from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable

from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock


class BaseParser(ABC):
    name = "base"

    def __init__(self) -> None:
        self.last_status = "success"
        self.last_error_code: str | None = None
        self.last_error_message: str | None = None

    def reset_status(self) -> None:
        self.last_status = "success"
        self.last_error_code = None
        self.last_error_message = None

    def set_status(self, status: str, error_code: str | None = None, message: str | None = None) -> None:
        self.last_status = status
        self.last_error_code = error_code
        self.last_error_message = message

    @abstractmethod
    def supports(self, file_path: Path) -> bool:
        raise NotImplementedError

    @abstractmethod
    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        raise NotImplementedError

    def make_block(
        self,
        file_path: Path,
        block_index: int,
        block_type: str,
        location_text: str,
        raw_text: str,
        *,
        page_number: int | None = None,
        slide_number: int | None = None,
        sheet_name: str | None = None,
        cell_start: str | None = None,
        cell_end: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        source_type: str = "native_text",
        ocr_confidence: float | None = None,
        extra: dict[str, object] | None = None,
    ) -> ContentBlock:
        return ContentBlock(
            file_path=str(file_path),
            block_index=block_index,
            block_type=block_type,
            location_text=location_text,
            raw_text=raw_text,
            normalized_text=normalize_text(raw_text),
            page_number=page_number,
            slide_number=slide_number,
            sheet_name=sheet_name,
            cell_start=cell_start,
            cell_end=cell_end,
            line_start=line_start,
            line_end=line_end,
            source_type=source_type,
            ocr_confidence=ocr_confidence,
            extra=extra or {},
        )
