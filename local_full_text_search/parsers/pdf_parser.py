from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.errors import ParserDependencyError, PasswordProtectedError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.ocr.ocr_engine import OcrEngine
from local_full_text_search.parsers.base_parser import BaseParser


class PdfParser(BaseParser):
    """PDF parser with native text first and optional scanned-page OCR.

    OCR is deliberately page-level and best-effort: a failed OCR pass should not
    discard native PDF text that was already extracted from other pages.
    """

    name = "pdf"

    def __init__(self, enable_scanned_ocr: bool = False, ocr_language: str = "ch") -> None:
        super().__init__()
        self.enable_scanned_ocr = enable_scanned_ocr
        self.ocr = OcrEngine(ocr_language) if enable_scanned_ocr else None

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == ".pdf"

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        try:
            import fitz
        except ImportError as exc:
            raise ParserDependencyError("未安装 PyMuPDF，无法解析 PDF") from exc
        doc = fitz.open(file_path)
        try:
            if doc.needs_pass:
                raise PasswordProtectedError("PDF 已加密，需要密码")
            for index in range(doc.page_count):
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                page = doc.load_page(index)
                text = page.get_text("text") or ""
                has_images = bool(page.get_images(full=True))
                scanned_like = len(text.strip()) < 20 and has_images
                yield self.make_block(
                    file_path,
                    index,
                    "pdf_page",
                    f"第 {index + 1} 页",
                    text,
                    page_number=index + 1,
                    source_type="native_text",
                    extra={"has_images": has_images, "is_scanned_like": scanned_like},
                )
                if scanned_like and self.enable_scanned_ocr and self.ocr is not None:
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    try:
                        image_path = render_pdf_page_for_ocr(page, file_path, index)
                        result = self.ocr.recognize(image_path)
                    except Exception as exc:
                        self.set_status("partial_success", "PDF_OCR_FAILED", f"第 {index + 1} 页 OCR 失败：{exc}")
                        continue
                    yield self.make_block(
                        file_path,
                        index + doc.page_count,
                        "pdf_page_ocr",
                        f"第 {index + 1} 页 OCR",
                        result.text,
                        page_number=index + 1,
                        source_type="ocr",
                        ocr_confidence=result.confidence,
                        extra=result.extra,
                    )
        finally:
            doc.close()


def render_pdf_page_for_ocr(page: object, file_path: Path, page_index: int) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"{file_path}:{file_path.stat().st_mtime_ns}:{page_index}".encode("utf-8")).hexdigest()[:16]
    target = TEMP_DIR / f"pdf_ocr_{digest}_{page_index + 1}.png"
    if target.exists():
        return target
    pixmap = page.get_pixmap(dpi=200)
    pixmap.save(str(target))
    return target
