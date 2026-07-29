from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.errors import ParserDependencyError, PasswordProtectedError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.ocr.ocr_engine import OcrEngine
from local_full_text_search.parsers.base_parser import BaseParser


class PdfParser(BaseParser):
    """Extract native PDF text and optionally OCR scanned pages."""

    name = "pdf"
    supports_resume = True

    def __init__(
        self,
        enable_scanned_ocr: bool = False,
        ocr_language: str = "ch",
        *,
        parallel_min_bytes: int = 64 * 1024 * 1024,
        parallel_min_pages: int = 500,
        parallel_workers: int = 4,
        ocr_engine: OcrEngine | None = None,
        ocr_cpu_threads: int = 2,
    ) -> None:
        super().__init__()
        self.enable_scanned_ocr = enable_scanned_ocr
        self.ocr = (
            ocr_engine or OcrEngine(ocr_language, ocr_cpu_threads)
            if enable_scanned_ocr
            else None
        )
        self.parallel_min_bytes = max(0, int(parallel_min_bytes))
        self.parallel_min_pages = max(1, int(parallel_min_pages))
        self.parallel_workers = max(1, int(parallel_workers))

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
            use_parallel = (
                not self.enable_scanned_ocr
                and self.parallel_workers > 1
                and (
                    file_path.stat().st_size >= self.parallel_min_bytes
                    or doc.page_count >= self.parallel_min_pages
                )
            )
            if use_parallel:
                page_count = doc.page_count
                doc.close()
                doc = None
                yield from self._parse_native_parallel(
                    file_path,
                    page_count,
                    cancel_token,
                    start_page=min(self.resume_cursor, page_count),
                )
                return
            page_count = doc.page_count
            for index in range(min(self.resume_cursor, page_count), page_count):
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                page = doc.load_page(index)
                text = page.get_text("text") or ""
                # Native-only indexing never enumerates page images.
                has_images = bool(page.get_images(full=True)) if self.enable_scanned_ocr else False
                scanned_like = self.enable_scanned_ocr and len(text.strip()) < 20 and has_images
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
                if scanned_like and self.ocr is not None:
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    try:
                        image_path = render_pdf_page_for_ocr(page, file_path, index)
                        result = self.ocr.recognize(image_path)
                    except Exception as exc:
                        self.set_status(
                            "partial_success",
                            "PDF_OCR_FAILED",
                            f"第 {index + 1} 页 OCR 失败：{exc}",
                        )
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
                self.report_progress(
                    "pdf_page",
                    completed=index + 1,
                    total=page_count,
                    unit_type="page",
                    cursor=index + 1,
                    detail=f"第 {index + 1} 页",
                )
        finally:
            if doc is not None:
                doc.close()

    def _parse_native_parallel(
        self,
        file_path: Path,
        page_count: int,
        cancel_token: CancelToken,
        *,
        start_page: int = 0,
    ) -> Iterable[ContentBlock]:
        workers = min(self.parallel_workers, page_count, max(1, os.cpu_count() or 1))
        remaining_pages = max(0, page_count - start_page)
        if remaining_pages <= 0:
            return
        chunk_size = max(1, min(64, (remaining_pages + workers * 4 - 1) // (workers * 4)))
        ranges = [
            (start, min(page_count, start + chunk_size))
            for start in range(start_page, page_count, chunk_size)
        ]
        results: list[tuple[int, str]] = []
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="lfts-pdf-page")
        futures = [
            executor.submit(_extract_pdf_page_range, file_path, start, end)
            for start, end in ranges
        ]
        try:
            for completed_chunks, future in enumerate(as_completed(futures), start=1):
                cancel_token.wait_if_paused()
                cancel_token.throw_if_cancelled()
                results.extend(future.result())
                self.report_progress(
                    "pdf_extract_chunk",
                    completed=completed_chunks,
                    total=len(futures),
                    unit_type="chunk",
                    cursor=start_page,
                    detail=f"已完成 {completed_chunks}/{len(futures)} 个分页批次",
                )
        finally:
            executor.shutdown(wait=not cancel_token.cancelled, cancel_futures=True)
        for page_index, text in sorted(results):
            cancel_token.throw_if_cancelled()
            yield self.make_block(
                file_path,
                page_index,
                "pdf_page",
                f"第 {page_index + 1} 页",
                text,
                page_number=page_index + 1,
                source_type="native_text",
                extra={"has_images": False, "is_scanned_like": False, "parallel": True},
            )
            self.report_progress(
                "pdf_page",
                completed=page_index + 1,
                total=page_count,
                unit_type="page",
                cursor=page_index + 1,
                detail=f"第 {page_index + 1} 页",
            )


def render_pdf_page_for_ocr(page: object, file_path: Path, page_index: int) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        f"{file_path}:{file_path.stat().st_mtime_ns}:{page_index}".encode("utf-8")
    ).hexdigest()[:16]
    target = TEMP_DIR / f"pdf_ocr_{digest}_{page_index + 1}.png"
    if target.exists():
        return target
    pixmap = page.get_pixmap(dpi=200)
    pixmap.save(str(target))
    return target


def _extract_pdf_page_range(file_path: Path, start: int, end: int) -> list[tuple[int, str]]:
    import fitz

    document = fitz.open(file_path)
    try:
        return [
            (page_index, document.load_page(page_index).get_text("text") or "")
            for page_index in range(start, end)
        ]
    finally:
        document.close()
