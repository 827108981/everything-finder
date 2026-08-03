from __future__ import annotations

import logging
import hashlib
from pathlib import Path
from typing import Iterable

from local_full_text_search.config.constants import IMAGE_EXTENSIONS
from local_full_text_search.core.errors import (
    CancelledError,
    PauseRequestedError,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.ocr.image_preprocess import image_dimensions
from local_full_text_search.ocr.ocr_cache import (
    OcrCache,
    OcrExactInput,
    ocr_models_fingerprint,
)
from local_full_text_search.ocr.ocr_engine import (
    ADAPTIVE_OCR_VERSION,
    DEFAULT_OCR_TILE_OVERLAP,
    DEFAULT_OCR_TILE_SIDE,
    OcrEngine,
    OcrResult,
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
    supports_resume = True

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
        progress_cursor = max(0, int(self.resume_cursor))

        def report_ocr_progress(
            phase: str,
            completed: int,
            total: int,
            detail: str,
        ) -> None:
            nonlocal progress_cursor
            progress_cursor += 1
            self.report_progress(
                f"ocr_{phase}",
                completed=completed,
                total=total,
                unit_type=(
                    "region" if "recognize" in phase else "tile"
                ),
                cursor=progress_cursor,
                detail=detail,
            )

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
            width, height = dimensions or (0, 0)
            channels, orientation = _image_channels_orientation(file_path)
            content_sha256 = (
                self.runtime_content_digest.removeprefix("sha256:")
                if self.runtime_content_digest
                else _sha256_file(file_path)
            )
            model_fingerprint = ocr_models_fingerprint()
            key = self.cache.key_for_exact_input(
                OcrExactInput(
                    content_sha256=content_sha256,
                    width=width,
                    height=height,
                    channels=channels,
                    orientation=orientation,
                    crop=None,
                    dpi=0,
                    preprocess_version="image-original-v1",
                    strategy_version=ADAPTIVE_OCR_VERSION,
                    detection_model_fingerprint=model_fingerprint,
                    recognition_model_fingerprint=model_fingerprint,
                    language=self.language,
                    options={
                        "detection_side": self.max_side,
                        "tile_side": DEFAULT_OCR_TILE_SIDE,
                        "tile_overlap": DEFAULT_OCR_TILE_OVERLAP,
                        "deskew": False,
                        "distortion": False,
                    },
                )
            )
            with self.cache.reference(key):
                result, cache_status = self.cache.load_with_status(key)
            checkpoint_path = self.cache.cache_dir / "checkpoints" / f"{key}.json"
            if result is None:
                result = self.ocr.recognize_adaptive(
                    file_path,
                    tile_side=DEFAULT_OCR_TILE_SIDE,
                    tile_overlap=DEFAULT_OCR_TILE_OVERLAP,
                    progress_callback=report_ocr_progress,
                    cancel_check=lambda: _check_cancel(cancel_token),
                    checkpoint_path=checkpoint_path,
                )
                result.extra["ocr_exact_cache_misses"] = 1
                if cache_status != "miss":
                    result.extra["ocr_exact_cache_invalidation_reason"] = cache_status
                cached_extra = dict(result.extra)
                _remove_runtime_metrics(cached_extra)
                self.cache.save(
                    key,
                    OcrResult(
                        result.text,
                        result.confidence,
                        cached_extra,
                    ),
                )
            else:
                checkpoint_path.unlink(missing_ok=True)
                _remove_runtime_metrics(result.extra)
                result.extra["ocr_exact_cache_hits"] = (
                    int(result.extra.get("ocr_exact_cache_hits") or 0) + 1
                )
        except (CancelledError, PauseRequestedError):
            raise
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
            cursor=progress_cursor + 1,
            detail=file_path.name,
        )


def _check_cancel(cancel_token: CancelToken) -> None:
    cancel_token.wait_if_paused()
    cancel_token.throw_if_cancelled()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_channels_orientation(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            channels = len(image.getbands())
            orientation = int(image.getexif().get(274, 1) or 1)
            return channels, orientation
    except Exception:
        return 0, 1


def _remove_runtime_metrics(extra: dict[str, object]) -> None:
    for key in (
        "detect_requests",
        "detect_inference_calls",
        "detect_batch_count",
        "detect_average_batch_size",
        "detect_pixels",
        "recognize_requests",
        "recognize_inference_calls",
        "recognize_batch_count",
        "recognize_average_batch_size",
        "recognize_pixels",
        "microbatch_wait_ms_p50",
        "microbatch_wait_ms_p95",
        "microbatch_wait_ms_max",
        "oversize_single_count",
        "cancelled_before_batch_count",
    ):
        extra.pop(key, None)
