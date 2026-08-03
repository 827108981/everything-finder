from __future__ import annotations

import hashlib
import io
import os
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.errors import (
    CancelledError,
    ParserDependencyError,
    PasswordProtectedError,
    PauseRequestedError,
)
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
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

PDF_DYNAMIC_OCR_VERSION = "1.1"

_OCR_RUNTIME_METRIC_KEYS = (
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
)


def _remove_runtime_metrics(extra: dict[str, object]) -> None:
    for key in _OCR_RUNTIME_METRIC_KEYS:
        extra.pop(key, None)


def _close_pdf_document(document: object) -> None:
    if bool(getattr(document, "is_closed", False)):
        return
    close = getattr(document, "close", None)
    if callable(close):
        close()


def _map_embedded_boxes_to_page(
    boxes: list[object],
    *,
    image_rect: object,
    width: int,
    height: int,
) -> list[object]:
    mapped: list[object] = []
    scale_x = float(image_rect.width) / max(1, int(width))
    scale_y = float(image_rect.height) / max(1, int(height))
    for box in boxes:
        if not isinstance(box, list):
            mapped.append(box)
            continue
        points: list[list[float]] = []
        for point in box:
            if not isinstance(point, list) or len(point) < 2:
                continue
            points.append(
                [
                    float(image_rect.x0)
                    + float(point[0]) * scale_x,
                    float(image_rect.y0)
                    + float(point[1]) * scale_y,
                ]
            )
        mapped.append(points)
    return mapped


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
        self.cache = OcrCache()
        self._scheduled_documents: OrderedDict[
            str,
            tuple[tuple[int, int, str], object],
        ] = OrderedDict()
        self._scheduled_document_cache_limit = 64

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
            if self.enable_scanned_ocr:
                page_count = doc.page_count
                use_parallel_scan = (
                    self.parallel_workers > 1
                    and (
                        file_path.stat().st_size >= self.parallel_min_bytes
                        or page_count >= self.parallel_min_pages
                    )
                )
                if use_parallel_scan:
                    doc.close()
                    doc = None
                    page_scans = _scan_pdf_pages_parallel(
                        file_path,
                        page_count,
                        self.parallel_workers,
                        min(self.resume_cursor, page_count),
                        cancel_token,
                        self.report_progress,
                    )
                else:
                    page_scans = []
                    for index in range(min(self.resume_cursor, page_count), page_count):
                        cancel_token.wait_if_paused()
                        cancel_token.throw_if_cancelled()
                        page = doc.load_page(index)
                        page_scans.append(
                            (
                                index,
                                page.get_text("text") or "",
                                bool(page.get_images(full=True)),
                            )
                        )
                ocr_candidates: list[int] = []
                for index, text, has_images in page_scans:
                    scanned_like = _is_ocr_candidate(text, has_images)
                    yield self.make_block(
                        file_path,
                        index,
                        "pdf_page",
                        f"第 {index + 1} 页",
                        text,
                        page_number=index + 1,
                        source_type="native_text",
                        extra={
                            "has_images": has_images,
                            "is_scanned_like": scanned_like,
                            "parallel": use_parallel_scan,
                        },
                    )
                    if scanned_like:
                        ocr_candidates.append(index)
                    self.report_progress(
                        "pdf_native_page",
                        completed=index + 1,
                        total=page_count,
                        unit_type="page",
                        cursor=index + 1,
                        detail=f"第 {index + 1} 页原生文字",
                    )
                if ocr_candidates and doc is None:
                    doc = fitz.open(file_path)
                for completed, index in enumerate(ocr_candidates, start=1):
                    cancel_token.wait_if_paused()
                    cancel_token.throw_if_cancelled()
                    page = doc.load_page(index)
                    try:
                        result = self._ocr_pdf_page(
                            page,
                            file_path,
                            index,
                            cancel_token,
                        )
                    except (CancelledError, PauseRequestedError):
                        raise
                    except Exception as exc:
                        self.set_status(
                            "partial_success",
                            "PDF_OCR_FAILED",
                            f"第 {index + 1} 页 OCR 失败：{exc}",
                        )
                        continue
                    yield self.make_block(
                        file_path,
                        index + page_count,
                        "pdf_page_ocr",
                        f"第 {index + 1} 页 OCR",
                        result.text,
                        page_number=index + 1,
                        source_type="ocr",
                        ocr_confidence=result.confidence,
                        extra=result.extra,
                    )
                    self.report_progress(
                        "pdf_ocr_page",
                        completed=completed,
                        total=len(ocr_candidates),
                        unit_type="page",
                        cursor=index + 1,
                        detail=f"第 {index + 1} 页 OCR 完成",
                    )
                return
            use_parallel = (
                self.parallel_workers > 1
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
                yield self.make_block(
                    file_path,
                    index,
                    "pdf_page",
                    f"第 {index + 1} 页",
                    text,
                    page_number=index + 1,
                    source_type="native_text",
                    extra={"has_images": False, "is_scanned_like": False},
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

    def parse_scheduled_page(
        self,
        file_path: Path,
        page_number: int,
        task_type: str,
        cancel_token: CancelToken,
    ) -> Iterable[ContentBlock]:
        """Execute exactly one durable page task."""

        try:
            import fitz
        except ImportError as exc:
            raise ParserDependencyError("未安装 PyMuPDF，无法解析 PDF") from exc
        reuse_document = task_type == "pdf_native_page"
        document = (
            self._open_scheduled_document(file_path, fitz)
            if reuse_document
            else fitz.open(file_path)
        )
        try:
            if document.needs_pass:
                raise PasswordProtectedError("PDF 已加密，需要密码")
            page_index = int(page_number) - 1
            if page_index < 0 or page_index >= document.page_count:
                raise IndexError(f"PDF page is out of range: {page_number}")
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            page = document.load_page(page_index)
            if task_type == "pdf_native_page":
                text = page.get_text("text") or ""
                has_images = bool(page.get_images(full=True))
                yield self.make_block(
                    file_path,
                    page_index,
                    "pdf_page",
                    f"第 {page_number} 页",
                    text,
                    page_number=page_number,
                    source_type="native_text",
                    extra={
                        "has_images": has_images,
                        "is_scanned_like": _is_ocr_candidate(text, has_images),
                        "page_task_graph": True,
                    },
                )
                self.report_progress(
                    "pdf_native_page",
                    completed=1,
                    total=1,
                    unit_type="page",
                    cursor=page_number,
                    detail=f"第 {page_number} 页原生文字",
                )
                return
            if task_type != "pdf_ocr_page":
                raise ValueError(f"Unknown PDF page task type: {task_type}")
            result = self._ocr_pdf_page(
                page,
                file_path,
                page_index,
                cancel_token,
            )
            yield self.make_block(
                file_path,
                page_index,
                "pdf_page_ocr",
                f"第 {page_number} 页 OCR",
                result.text,
                page_number=page_number,
                source_type="ocr",
                ocr_confidence=result.confidence,
                extra={**result.extra, "page_task_graph": True},
            )
            self.report_progress(
                "pdf_ocr_page",
                completed=1,
                total=1,
                unit_type="page",
                cursor=page_number,
                detail=f"第 {page_number} 页 OCR 完成",
            )
        finally:
            if not reuse_document:
                _close_pdf_document(document)

    def _open_scheduled_document(
        self,
        file_path: Path,
        fitz_module: object,
    ) -> object:
        path = Path(file_path)
        stat = path.stat()
        key = str(path.resolve())
        identity = (
            int(stat.st_size),
            int(stat.st_mtime_ns),
            str(self.runtime_content_digest or ""),
        )
        cached = self._scheduled_documents.get(key)
        if cached is not None:
            cached_identity, document = cached
            if cached_identity == identity and not bool(
                getattr(document, "is_closed", False)
            ):
                self._scheduled_documents.move_to_end(key)
                return document
            self._scheduled_documents.pop(key, None)
            _close_pdf_document(document)

        document = fitz_module.open(path)
        if document.needs_pass:
            _close_pdf_document(document)
            raise PasswordProtectedError("PDF is password protected")
        self._scheduled_documents[key] = (identity, document)
        while len(self._scheduled_documents) > self._scheduled_document_cache_limit:
            _old_key, (_old_identity, old_document) = (
                self._scheduled_documents.popitem(last=False)
            )
            _close_pdf_document(old_document)
        return document

    def close(self) -> None:
        documents = getattr(self, "_scheduled_documents", None)
        if not documents:
            return
        for _identity, document in list(documents.values()):
            _close_pdf_document(document)
        documents.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ocr_pdf_page(
        self,
        page: object,
        file_path: Path,
        page_index: int,
        cancel_token: CancelToken,
    ) -> object:
        if self.ocr is None:
            raise RuntimeError("OCR engine is unavailable")
        namespace = (
            f"pdf-page-ocr:{ADAPTIVE_OCR_VERSION}:{PDF_DYNAMIC_OCR_VERSION}:"
            f"page={page_index}:"
            "preview=150:preview-side=2400:region=200:upgrade=300:"
            f"tile={DEFAULT_OCR_TILE_SIDE}:overlap={DEFAULT_OCR_TILE_OVERLAP}:"
            f"{ocr_models_fingerprint()}"
        )
        key = ""
        if self.runtime_content_digest:
            key = self.cache.key_for_digest(
                f"{self.runtime_content_digest}:page={page_index}",
                namespace=namespace,
            )
        else:
            key = self.cache.key_for_digest(
                f"{file_path}:{file_path.stat().st_mtime_ns}:page={page_index}",
                namespace=namespace,
            )
        with self.cache.reference(key):
            cached = self.cache.load(key)
        if cached is not None:
            _remove_runtime_metrics(cached.extra)
            return cached
        result = self._ocr_full_page_embedded_image(
            page,
            file_path,
            page_index,
            cancel_token,
        )
        if result is not None:
            pass
        elif hasattr(self.ocr, "detect_file_regions") and hasattr(self.ocr, "recognize_crops"):
            result = self._ocr_pdf_page_dynamic(
                page,
                file_path,
                page_index,
                cancel_token,
            )
        else:
            result = self._ocr_pdf_page_legacy(
                page,
                file_path,
                page_index,
                cancel_token,
            )
        runtime_snapshot = getattr(
            self.ocr,
            "runtime_metrics_snapshot",
            None,
        )
        runtime_metrics = (
            dict(runtime_snapshot())
            if callable(runtime_snapshot)
            else {}
        )
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
        result.extra.update(runtime_metrics)
        return result

    def _ocr_full_page_embedded_image(
        self,
        page: object,
        file_path: Path,
        page_index: int,
        cancel_token: CancelToken,
    ) -> OcrResult | None:
        """Reuse exact OCR for an unmasked raster covering the whole page."""

        if self.ocr is None:
            return None
        try:
            images = list(page.get_images(full=True))
        except (CancelledError, PauseRequestedError):
            raise
        except Exception:
            return None
        distinct = {
            int(item[0]): item
            for item in images
            if item and int(item[0]) > 0
        }
        if len(distinct) != 1:
            return None
        xref, image_info = next(iter(distinct.items()))
        if len(image_info) > 1 and int(image_info[1] or 0) != 0:
            return None
        try:
            rectangles = list(page.get_image_rects(xref))
            if len(rectangles) != 1:
                return None
            image_rect = rectangles[0]
            page_rect = page.rect
            overlap = image_rect & page_rect
            page_area = max(
                1.0,
                float(page_rect.width) * float(page_rect.height),
            )
            overlap_area = max(
                0.0,
                float(overlap.width) * float(overlap.height),
            )
            if overlap_area / page_area < 0.95:
                return None
            extracted = page.parent.extract_image(xref)
            image_bytes = bytes(extracted.get("image") or b"")
            if not image_bytes:
                return None
            with io.BytesIO(image_bytes) as stream:
                from PIL import Image

                with Image.open(stream) as embedded:
                    width, height = embedded.size
                    channels = len(embedded.getbands())
                    orientation = int(
                        embedded.getexif().get(274, 1) or 1
                    )
            content_sha256 = hashlib.sha256(
                image_bytes
            ).hexdigest()
            model_fingerprint = ocr_models_fingerprint()
            exact_input = OcrExactInput(
                content_sha256=content_sha256,
                width=int(width),
                height=int(height),
                channels=int(channels),
                orientation=orientation,
                crop=None,
                dpi=0,
                preprocess_version="image-original-v1",
                strategy_version=ADAPTIVE_OCR_VERSION,
                detection_model_fingerprint=model_fingerprint,
                recognition_model_fingerprint=model_fingerprint,
                language=str(
                    getattr(self.ocr, "language", "ch")
                ),
                options={
                    "detection_side": int(
                        getattr(
                            self.ocr,
                            "det_limit_side_len",
                            960,
                        )
                    ),
                    "tile_side": DEFAULT_OCR_TILE_SIDE,
                    "tile_overlap": DEFAULT_OCR_TILE_OVERLAP,
                    "deskew": False,
                    "distortion": False,
                },
            )
            exact_key = self.cache.key_for_exact_input(exact_input)
            with self.cache.reference(exact_key):
                result, cache_status = self.cache.load_with_status(
                    exact_key
                )
            cache_hit = result is not None
            if result is None:
                extension = str(extracted.get("ext") or "img")
                image_path = TEMP_DIR / (
                    f"pdf-embedded-{os.getpid()}-{time.time_ns()}."
                    f"{extension}"
                )
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(image_bytes)
                checkpoint_path = (
                    self.cache.cache_dir
                    / "checkpoints"
                    / f"{exact_key}.json"
                )
                try:
                    result = self.ocr.recognize_adaptive(
                        image_path,
                        tile_side=DEFAULT_OCR_TILE_SIDE,
                        tile_overlap=DEFAULT_OCR_TILE_OVERLAP,
                        progress_callback=lambda phase, completed, total, detail: self.report_progress(
                            f"pdf_embedded_{phase}",
                            completed=completed,
                            total=total,
                            unit_type=(
                                "region"
                                if "recognize" in phase
                                else "tile"
                            ),
                            cursor=page_index + 1,
                            detail=(
                                f"第 {page_index + 1} 页内嵌原图 · "
                                f"{detail}"
                            ),
                        ),
                        cancel_check=lambda: _check_cancel(
                            cancel_token
                        ),
                        checkpoint_path=checkpoint_path,
                    )
                    cached_extra = dict(result.extra)
                    _remove_runtime_metrics(cached_extra)
                    self.cache.save(
                        exact_key,
                        OcrResult(
                            result.text,
                            result.confidence,
                            cached_extra,
                        ),
                    )
                finally:
                    image_path.unlink(missing_ok=True)
            else:
                _remove_runtime_metrics(result.extra)
            wrapped_extra = dict(result.extra)
            boxes = result.extra.get("boxes")
            if isinstance(boxes, list):
                wrapped_extra["boxes"] = _map_embedded_boxes_to_page(
                    boxes,
                    image_rect=image_rect,
                    width=width,
                    height=height,
                )
            wrapped_extra.update(
                {
                    "embedded_image_source": "full_page_exact",
                    "embedded_image_xref": xref,
                    "embedded_image_sha256": content_sha256,
                    "embedded_image_size": [width, height],
                    "ocr_embedded_image_cache_hits": (
                        1 if cache_hit else 0
                    ),
                    "ocr_exact_cache_hits": 1 if cache_hit else 0,
                    "ocr_exact_cache_misses": 0 if cache_hit else 1,
                    "ocr_exact_cache_status": cache_status,
                }
            )
            return OcrResult(
                result.text,
                result.confidence,
                wrapped_extra,
            )
        except (CancelledError, PauseRequestedError):
            raise
        except Exception:
            # Extraction is an optimization. Rendering the complete page
            # remains the recall-preserving fallback.
            return None

    def _ocr_pdf_page_legacy(
        self,
        page: object,
        file_path: Path,
        page_index: int,
        cancel_token: CancelToken,
    ) -> OcrResult:
        if self.ocr is None:
            raise RuntimeError("OCR engine is unavailable")
        render_started = time.perf_counter()
        image_path = render_pdf_page_for_ocr(
            page,
            file_path,
            page_index,
            dpi=200,
        )
        render_ms = int((time.perf_counter() - render_started) * 1000)
        try:
            try:
                from PIL import Image

                with Image.open(image_path) as rendered:
                    full_pixels = int(rendered.width) * int(rendered.height)
            except Exception:
                full_pixels = 0
            recognize_started = time.perf_counter()
            result = self.ocr.recognize_adaptive(
                image_path,
                tile_side=DEFAULT_OCR_TILE_SIDE,
                tile_overlap=DEFAULT_OCR_TILE_OVERLAP,
                progress_callback=lambda phase, completed, total, detail: self.report_progress(
                    f"pdf_ocr_{phase}",
                    completed=completed,
                    total=total,
                    unit_type="region" if "recognize" in phase else "tile",
                    cursor=page_index + 1,
                    detail=f"第 {page_index + 1} 页 · {detail}",
                ),
                cancel_check=lambda: _check_cancel(cancel_token),
            )
            result.extra.update(
                {
                    "pdf_full_fallback_pixels": full_pixels,
                    "pdf_full_fallback_render_ms": render_ms,
                    "pdf_full_fallback_recognize_ms": int(
                        (time.perf_counter() - recognize_started) * 1000
                    ),
                }
            )
            return result
        finally:
            image_path.unlink(missing_ok=True)

    def _ocr_pdf_page_dynamic(
        self,
        page: object,
        file_path: Path,
        page_index: int,
        cancel_token: CancelToken,
    ) -> OcrResult:
        if self.ocr is None:
            raise RuntimeError("OCR engine is unavailable")
        self._ocr_exact_cache_hits = 0
        self._ocr_exact_cache_misses = 0
        self._ocr_exact_cache_invalidations: dict[str, int] = {}
        self._pdf_stage_metrics: dict[str, int] = {}
        preview_render_started = time.perf_counter()
        preview_path = render_pdf_page_for_ocr(
            page,
            file_path,
            page_index,
            dpi=150,
            max_side=2400,
        )
        self._pdf_stage_metrics["pdf_preview_render_ms"] = int(
            (time.perf_counter() - preview_render_started) * 1000
        )
        try:
            _check_cancel(cancel_token)
            self.report_progress(
                "pdf_ocr_preview",
                completed=0,
                total=1,
                unit_type="page",
                cursor=page_index + 1,
                detail=f"第 {page_index + 1} 页 · 低成本文字区域发现",
            )
            preview_detect_started = time.perf_counter()
            polygons, preview_size = self.ocr.detect_file_regions(preview_path)
            self._pdf_stage_metrics["pdf_preview_detect_ms"] = int(
                (time.perf_counter() - preview_detect_started) * 1000
            )
            self._pdf_stage_metrics["pdf_preview_pixels"] = (
                int(preview_size[0]) * int(preview_size[1])
            )
            self.report_progress(
                "pdf_ocr_preview",
                completed=1,
                total=1,
                unit_type="page",
                cursor=page_index + 1,
                detail=f"第 {page_index + 1} 页 · 发现 {len(polygons)} 个区域",
            )
            preview_fallback_reason = _pdf_preview_fallback_reason(
                polygons,
                preview_size,
            )
            if preview_fallback_reason:
                result = self._ocr_pdf_page_legacy(
                    page,
                    file_path,
                    page_index,
                    cancel_token,
                )
                result.extra.update(
                    {
                        "pdf_dynamic_dpi": True,
                        "pdf_preview_dpi": 150,
                        "pdf_full_page_fallback": True,
                        "pdf_region_dpi": 200,
                        "pdf_upgraded_regions": 0,
                        "pdf_full_page_fallback_reason": preview_fallback_reason,
                        **self._exact_cache_metrics(),
                        **self._pdf_stage_metrics,
                    }
                )
                return result

            base_results = self._recognize_pdf_regions_bounded(
                page,
                polygons,
                preview_size,
                dpi=200,
                phase="pdf_region_200dpi",
                page_index=page_index,
                cancel_token=cancel_token,
            )
            upgrade_indexes = [
                index
                for index, (text, confidence) in enumerate(base_results)
                if (
                    not text.strip()
                    or confidence is None
                    or confidence < 0.75
                    or _polygon_height(polygons[index]) < 18.0
                )
            ]
            final_results = list(base_results)
            if upgrade_indexes:
                _check_cancel(cancel_token)
                upgrade_polygons = [polygons[index] for index in upgrade_indexes]
                upgraded_results = self._recognize_pdf_regions_bounded(
                    page,
                    upgrade_polygons,
                    preview_size,
                    dpi=300,
                    phase="pdf_region_300dpi",
                    page_index=page_index,
                    cancel_token=cancel_token,
                )
                for source_index, upgraded in zip(
                    upgrade_indexes,
                    upgraded_results,
                    strict=False,
                ):
                    original = final_results[source_index]
                    if upgraded[0].strip() and (
                        not original[0].strip()
                        or (upgraded[1] or 0.0) > (original[1] or 0.0)
                    ):
                        final_results[source_index] = upgraded

            non_empty = [
                (text, confidence, polygon)
                for (text, confidence), polygon in zip(
                    final_results,
                    polygons,
                    strict=False,
                )
                if text.strip()
            ]
            confidences = [
                confidence
                for _, confidence, _ in non_empty
                if confidence is not None
            ]
            character_count = sum(len(text.strip()) for text, _, _ in non_empty)
            average = sum(confidences) / len(confidences) if confidences else None
            fallback_reason = _pdf_region_fallback_reason(
                polygons,
                preview_size,
                confidences,
                character_count,
            )
            if fallback_reason:
                result = self._ocr_pdf_page_legacy(
                    page,
                    file_path,
                    page_index,
                    cancel_token,
                )
                result.extra.update(
                    {
                        "pdf_dynamic_dpi": True,
                        "pdf_preview_dpi": 150,
                        "pdf_full_page_fallback": True,
                        "pdf_region_dpi": 200,
                        "pdf_upgrade_dpi": 300,
                        "pdf_upgraded_regions": len(upgrade_indexes),
                        "pdf_full_page_fallback_reason": fallback_reason,
                        **self._exact_cache_metrics(),
                        **self._pdf_stage_metrics,
                    }
                )
                return result
            return OcrResult(
                "\n".join(text for text, _, _ in non_empty),
                average,
                {
                    "boxes": [polygon for _, _, polygon in non_empty],
                    "line_confidences": confidences,
                    "language": self.ocr.language,
                    "engine": "PaddleOCR",
                    "strategy": "pdf-dynamic-region-ocr",
                    "strategy_version": ADAPTIVE_OCR_VERSION,
                    "pdf_dynamic_dpi": True,
                    "pdf_preview_dpi": 150,
                    "pdf_preview_max_side": 2400,
                    "pdf_region_dpi": 200,
                    "pdf_upgrade_dpi": 300,
                    "pdf_detected_regions": len(polygons),
                    "pdf_upgraded_regions": len(upgrade_indexes),
                    "pdf_full_page_fallback": False,
                    "ocr_model_load_count": int(
                        getattr(self.ocr, "model_load_count", 0)
                    ),
                    "ocr_model_load_ms": int(
                        getattr(self.ocr, "model_load_ms", 0)
                    ),
                    "ocr_model_state": str(
                        getattr(self.ocr, "model_state", "unloaded")
                    ),
                    **self._exact_cache_metrics(),
                    **self._pdf_stage_metrics,
                },
            )
        finally:
            preview_path.unlink(missing_ok=True)

    def _recognize_pdf_regions_bounded(
        self,
        page: object,
        polygons: list[list[list[float]]],
        preview_size: tuple[int, int],
        *,
        dpi: int,
        phase: str,
        page_index: int,
        cancel_token: CancelToken,
    ) -> list[tuple[str, float | None]]:
        if self.ocr is None:
            raise RuntimeError("OCR engine is unavailable")
        region_batch_size = 64
        results: list[tuple[str, float | None]] = []
        total = len(polygons)
        model_fingerprint = ocr_models_fingerprint()
        render_metric = f"{phase}_render_ms"
        recognize_metric = f"{phase}_recognize_ms"
        pixel_metric = f"{phase}_pixels"
        self._pdf_stage_metrics.setdefault(render_metric, 0)
        self._pdf_stage_metrics.setdefault(recognize_metric, 0)
        self._pdf_stage_metrics.setdefault(pixel_metric, 0)
        for start in range(0, total, region_batch_size):
            _check_cancel(cancel_token)
            polygon_batch = polygons[start : start + region_batch_size]
            render_started = time.perf_counter()
            crops = render_pdf_regions(
                page,
                polygon_batch,
                preview_size,
                dpi=dpi,
            )
            self._pdf_stage_metrics[render_metric] += int(
                (time.perf_counter() - render_started) * 1000
            )
            self._pdf_stage_metrics[pixel_metric] += sum(
                int(getattr(crop, "shape", (0, 0))[0])
                * int(getattr(crop, "shape", (0, 0))[1])
                for crop in crops
                if len(getattr(crop, "shape", ())) >= 2
            )
            batch_results: list[tuple[str, float | None] | None] = [
                None for _ in crops
            ]
            missing_crops: list[object] = []
            missing_slots: list[tuple[int, str]] = []
            for slot, (crop_image, polygon) in enumerate(
                zip(crops, polygon_batch, strict=False)
            ):
                shape = tuple(int(value) for value in getattr(crop_image, "shape", ()))
                height = shape[0] if len(shape) >= 1 else 0
                width = shape[1] if len(shape) >= 2 else 0
                channels = shape[2] if len(shape) >= 3 else 1
                content_sha256 = _pixel_sha256(crop_image)
                exact_input = OcrExactInput(
                    content_sha256=content_sha256,
                    width=width,
                    height=height,
                    channels=channels,
                    orientation=0,
                    crop=_pdf_region_source_crop(page, polygon, preview_size),
                    dpi=dpi,
                    preprocess_version="pdf-region-render-v1",
                    strategy_version=(
                        f"{ADAPTIVE_OCR_VERSION}:{PDF_DYNAMIC_OCR_VERSION}"
                    ),
                    detection_model_fingerprint=model_fingerprint,
                    recognition_model_fingerprint=model_fingerprint,
                    language=str(self.ocr.language),
                    options={
                        "alpha": False,
                        "deskew": False,
                        "distortion": False,
                    },
                )
                cache_key = self.cache.key_for_exact_input(exact_input)
                with self.cache.reference(cache_key):
                    cached, cache_status = self.cache.load_with_status(
                        cache_key
                    )
                if cached is not None:
                    batch_results[slot] = (cached.text, cached.confidence)
                    self._ocr_exact_cache_hits += 1
                    continue
                self._ocr_exact_cache_misses += 1
                if cache_status not in {"miss", "hit"}:
                    self._ocr_exact_cache_invalidations[cache_status] = (
                        self._ocr_exact_cache_invalidations.get(cache_status, 0) + 1
                    )
                missing_crops.append(crop_image)
                missing_slots.append((slot, cache_key))
            if missing_crops:
                recognize_started = time.perf_counter()
                recognized = self.ocr.recognize_crops(
                    missing_crops,
                    phase=phase,
                    progress_callback=lambda inner_phase, completed, inner_total, detail: self.report_progress(
                        f"pdf_ocr_{inner_phase}",
                        completed=min(total, start + completed),
                        total=total,
                        unit_type="region",
                        cursor=page_index + 1,
                        detail=f"第 {page_index + 1} 页 · {detail}",
                    ),
                    cancel_check=lambda: _check_cancel(cancel_token),
                )
                self._pdf_stage_metrics[recognize_metric] += int(
                    (time.perf_counter() - recognize_started) * 1000
                )
                for (slot, cache_key), recognized_item in zip(
                    missing_slots,
                    recognized,
                    strict=False,
                ):
                    text, confidence = recognized_item
                    batch_results[slot] = (text, confidence)
                    self.cache.save(
                        cache_key,
                        OcrResult(
                            text,
                            confidence,
                            {
                                "cache_level": "pdf_region_crop",
                                "dpi": dpi,
                            },
                        ),
                    )
            results.extend(
                item if item is not None else ("", None)
                for item in batch_results
            )
            self.report_progress(
                f"pdf_ocr_{phase}",
                completed=min(total, start + len(polygon_batch)),
                total=total,
                unit_type="region",
                cursor=page_index + 1,
                detail=(
                    f"第 {page_index + 1} 页 · {dpi} DPI 区域 "
                    f"{min(total, start + len(polygon_batch))}/{total}"
                ),
            )
        return results

    def _exact_cache_metrics(self) -> dict[str, object]:
        return {
            "ocr_exact_cache_hits": int(
                getattr(self, "_ocr_exact_cache_hits", 0)
            ),
            "ocr_exact_cache_misses": int(
                getattr(self, "_ocr_exact_cache_misses", 0)
            ),
            "ocr_exact_cache_invalidations": dict(
                getattr(self, "_ocr_exact_cache_invalidations", {})
            ),
        }

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


def render_pdf_page_for_ocr(
    page: object,
    file_path: Path,
    page_index: int,
    *,
    dpi: int = 200,
    max_side: int | None = None,
) -> Path:
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(
        (
            f"{file_path}:{file_path.stat().st_mtime_ns}:{page_index}:"
            f"dpi={dpi}:max={max_side or 0}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    target = TEMP_DIR / f"pdf_ocr_{digest}_{page_index + 1}_{dpi}.png"
    if target.exists():
        return target
    page_rect = page.rect
    scale = max(0.1, float(dpi) / 72.0)
    if max_side:
        scale = min(
            scale,
            max(0.1, float(max_side) / max(1.0, page_rect.width, page_rect.height)),
        )
    pixmap = page.get_pixmap(matrix=__import__("fitz").Matrix(scale, scale), alpha=False)
    pixmap.save(str(target))
    return target


def render_pdf_regions(
    page: object,
    polygons: list[list[list[float]]],
    preview_size: tuple[int, int],
    *,
    dpi: int,
) -> list[object]:
    import numpy as np
    import fitz

    preview_width, preview_height = preview_size
    page_rect = page.rect
    crops: list[object] = []
    for polygon in polygons:
        left = min(point[0] for point in polygon)
        top = min(point[1] for point in polygon)
        right = max(point[0] for point in polygon)
        bottom = max(point[1] for point in polygon)
        margin_x = max(2.0, (right - left) * 0.08)
        margin_y = max(2.0, (bottom - top) * 0.16)
        clip = fitz.Rect(
            page_rect.x0 + max(0.0, left - margin_x) / max(1, preview_width) * page_rect.width,
            page_rect.y0 + max(0.0, top - margin_y) / max(1, preview_height) * page_rect.height,
            page_rect.x0 + min(float(preview_width), right + margin_x) / max(1, preview_width) * page_rect.width,
            page_rect.y0 + min(float(preview_height), bottom + margin_y) / max(1, preview_height) * page_rect.height,
        )
        clip &= page_rect
        pixmap = page.get_pixmap(dpi=dpi, clip=clip, alpha=False)
        channels = max(1, int(pixmap.n))
        array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            channels,
        )
        if channels == 1:
            array = np.repeat(array, 3, axis=2)
        else:
            array = array[:, :, :3][:, :, ::-1]
        crops.append(array.copy())
    return crops


def _pixel_sha256(image: object) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(int(value)) for value in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _pdf_region_source_crop(
    page: object,
    polygon: list[list[float]],
    preview_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return the expanded crop in stable original-page milli-point units."""

    preview_width, preview_height = preview_size
    page_rect = page.rect
    left = min(point[0] for point in polygon)
    top = min(point[1] for point in polygon)
    right = max(point[0] for point in polygon)
    bottom = max(point[1] for point in polygon)
    margin_x = max(2.0, (right - left) * 0.08)
    margin_y = max(2.0, (bottom - top) * 0.16)
    x0 = page_rect.x0 + max(0.0, left - margin_x) / max(1, preview_width) * page_rect.width
    y0 = page_rect.y0 + max(0.0, top - margin_y) / max(1, preview_height) * page_rect.height
    x1 = page_rect.x0 + min(float(preview_width), right + margin_x) / max(1, preview_width) * page_rect.width
    y1 = page_rect.y0 + min(float(preview_height), bottom + margin_y) / max(1, preview_height) * page_rect.height
    return (
        int(round(x0 * 1000)),
        int(round(y0 * 1000)),
        int(round(max(0.0, x1 - x0) * 1000)),
        int(round(max(0.0, y1 - y0) * 1000)),
    )


def _polygon_height(polygon: list[list[float]]) -> float:
    return max(point[1] for point in polygon) - min(point[1] for point in polygon)


def _pdf_preview_fallback_reason(
    polygons: list[list[list[float]]],
    preview_size: tuple[int, int],
) -> str:
    if not polygons:
        return "no_regions"
    if max(preview_size) >= 2000 and len(polygons) < 8:
        # Skip the region-recognition pass entirely.  On a large diagram,
        # a sparse preview result cannot be trusted and would be discarded by
        # the quality gate after spending time on 200/300-DPI crops.
        return "sparse_regions_on_large_page"
    return ""


def _pdf_region_fallback_reason(
    polygons: list[list[list[float]]],
    preview_size: tuple[int, int],
    confidences: list[float],
    character_count: int,
) -> str:
    if not polygons or character_count <= 0:
        return "no_region_text"
    if confidences:
        average = sum(confidences) / len(confidences)
        low_ratio = sum(score < 0.65 for score in confidences) / len(confidences)
        if average < 0.60:
            return "low_average_confidence"
        if low_ratio > 0.30:
            return "too_many_low_confidence_regions"
    if character_count < max(4, len(polygons) * 2):
        return "insufficient_characters"
    preview_reason = _pdf_preview_fallback_reason(polygons, preview_size)
    if preview_reason:
        return preview_reason
    return ""


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


def _scan_pdf_pages_parallel(
    file_path: Path,
    page_count: int,
    workers: int,
    start_page: int,
    cancel_token: CancelToken,
    progress_callback: Callable[..., None],
) -> list[tuple[int, str, bool]]:
    remaining_pages = max(0, page_count - start_page)
    if remaining_pages <= 0:
        return []
    worker_count = min(max(1, workers), remaining_pages, max(1, os.cpu_count() or 1))
    chunk_size = max(
        1,
        min(64, (remaining_pages + worker_count * 4 - 1) // (worker_count * 4)),
    )
    ranges = [
        (start, min(page_count, start + chunk_size))
        for start in range(start_page, page_count, chunk_size)
    ]
    results: list[tuple[int, str, bool]] = []
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="lfts-pdf-classify",
    )
    futures = [
        executor.submit(_scan_pdf_page_range, file_path, start, end)
        for start, end in ranges
    ]
    try:
        for completed_chunks, future in enumerate(as_completed(futures), start=1):
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            results.extend(future.result())
            progress_callback(
                "pdf_native_scan_chunk",
                completed=completed_chunks,
                total=len(futures),
                unit_type="chunk",
                cursor=start_page,
                detail=f"已完成 {completed_chunks}/{len(futures)} 个原生页扫描批次",
            )
    finally:
        executor.shutdown(wait=not cancel_token.cancelled, cancel_futures=True)
    return sorted(results)


def _scan_pdf_page_range(
    file_path: Path,
    start: int,
    end: int,
) -> list[tuple[int, str, bool]]:
    import fitz

    document = fitz.open(file_path)
    try:
        result: list[tuple[int, str, bool]] = []
        for page_index in range(start, end):
            page = document.load_page(page_index)
            result.append(
                (
                    page_index,
                    page.get_text("text") or "",
                    bool(page.get_images(full=True)),
                )
            )
        return result
    finally:
        document.close()


def _is_ocr_candidate(text: str, has_images: bool) -> bool:
    if not has_images:
        return False
    stripped = "".join(character for character in text if not character.isspace())
    if not stripped:
        return True
    replacement_ratio = stripped.count("\ufffd") / max(1, len(stripped))
    meaningful = sum(
        1
        for character in stripped
        if character.isalnum() or "\u3400" <= character <= "\u9fff"
    )
    return meaningful < 20 or replacement_ratio >= 0.2


def _check_cancel(cancel_token: CancelToken) -> None:
    cancel_token.wait_if_paused()
    cancel_token.throw_if_cancelled()
