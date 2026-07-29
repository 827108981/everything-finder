from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import IMAGE_EXTENSIONS
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.ocr.image_preprocess import image_dimensions
from local_full_text_search.ocr.ocr_cache import OcrCache, ocr_models_fingerprint
from local_full_text_search.ocr.ocr_engine import (
    ADAPTIVE_OCR_VERSION,
    DEFAULT_OCR_TILE_OVERLAP,
    DEFAULT_OCR_TILE_SIDE,
    OcrEngine,
)
from local_full_text_search.parsers.base_parser import BaseParser

logger = logging.getLogger(__name__)


class ImageParser(BaseParser):
    """Image OCR parser.

    Team builds are expected to ship PaddleOCR, but the standard build may not.
    Missing OCR is therefore treated as metadata-only/ocr_disabled instead of a
    hard parse failure, so image files remain searchable by name and path.
    """

    name = "image_ocr"

    def __init__(
        self,
        language: str = "ch",
        enabled: bool = True,
        min_pixels: int = 12_000,
        max_side: int = 960,
        ocr_engine: OcrEngine | None = None,
        ocr_cpu_threads: int = 2,
    ) -> None:
        super().__init__()
        self.language = language
        self.enabled = enabled
        self.min_pixels = min_pixels
        self.max_side = max_side
        self.ocr = ocr_engine or OcrEngine(
            language=language,
            cpu_threads=ocr_cpu_threads,
            det_limit_side_len=max_side,
        )
        self.cache = OcrCache()

    def supports(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in IMAGE_EXTENSIONS

    def parse(self, file_path: Path, cancel_token: CancelToken) -> Iterable[ContentBlock]:
        if not self.enabled:
            self.set_status("ocr_disabled", "OCR_DISABLED", "图片 OCR 未启用，仅索引文件名和路径")
            return
        dimensions = image_dimensions(file_path)
        if dimensions is not None:
            width, height = dimensions
            if width * height < self.min_pixels:
                self.report_progress(
                    "image_too_small",
                    completed=1,
                    total=1,
                    unit_type="image",
                    cursor=1,
                    detail="图片尺寸低于 OCR 阈值，按完整空文本处理",
                )
                return
        cancel_token.wait_if_paused()
        cancel_token.throw_if_cancelled()
        try:
            namespace = (
                f"image-ocr-adaptive-{ADAPTIVE_OCR_VERSION}:{self.language}:"
                f"detect={self.max_side}:tile={DEFAULT_OCR_TILE_SIDE}:"
                f"overlap={DEFAULT_OCR_TILE_OVERLAP}:{ocr_models_fingerprint()}"
            )
            key = self.cache.key_for_file(file_path, namespace=namespace)
            result = self.cache.load(key)
            if result is None:
                result = self.ocr.recognize_adaptive(
                    file_path,
                    tile_side=DEFAULT_OCR_TILE_SIDE,
                    tile_overlap=DEFAULT_OCR_TILE_OVERLAP,
                    progress_callback=lambda phase, completed, total, detail: self.report_progress(
                        f"ocr_{phase}",
                        completed=completed,
                        total=total,
                        unit_type="region" if "recognize" in phase else "tile",
                        cursor=0,
                        detail=detail,
                    ),
                    cancel_check=lambda: _check_cancel(cancel_token),
                )
                self.cache.save(key, result)
        except Exception as exc:
            logger.exception("OCR failed for %s", file_path)
            self.set_status("failed_retryable", "OCR_UNAVAILABLE", f"OCR 不可用：{exc}")
            return
        yield self.make_block(
            file_path,
            0,
            "image_ocr",
            "图片 OCR",
            result.text,
            source_type="ocr",
            ocr_confidence=result.confidence,
            extra=result.extra,
        )
        self.report_progress(
            "image_ocr",
            completed=1,
            total=1,
            unit_type="image",
            cursor=1,
            detail=file_path.name,
        )


def _check_cancel(cancel_token: CancelToken) -> None:
    cancel_token.wait_if_paused()
    cancel_token.throw_if_cancelled()
