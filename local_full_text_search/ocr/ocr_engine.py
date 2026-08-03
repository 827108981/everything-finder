from __future__ import annotations

import os
import hashlib
import json
import math
import re
import shutil
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from local_full_text_search.config.constants import APP_NAME, CACHE_DIR, OCR_MODELS_DIR
from local_full_text_search.core.errors import ParserDependencyError
from local_full_text_search.ocr.model_manifest import model_manifest_fingerprint
from local_full_text_search.ocr.microbatch import RecognitionMicroBatchCoordinator
from local_full_text_search.ocr.adaptive_region_planner import (
    AdaptiveRegionPlanner,
)


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float | None
    extra: dict[str, object]


@dataclass(slots=True)
class _OcrLine:
    text: str
    confidence: float | None
    box: list[list[float]]


ADAPTIVE_OCR_VERSION = "3.0"
DEFAULT_OCR_TILE_SIDE = 1280
DEFAULT_OCR_TILE_OVERLAP = 160


class OcrEngine:
    """Lazy PaddleOCR wrapper. The heavy dependency is imported only when OCR runs."""

    def __init__(
        self,
        language: str = "ch",
        cpu_threads: int = 2,
        det_limit_side_len: int = 960,
        enable_mkldnn: bool = False,
        microbatch_max_requests: int = 64,
        microbatch_max_pixels: int = 8_000_000,
        microbatch_memory_bytes: int = 256 * 1024 * 1024,
        microbatch_wait_ms: int = 30,
    ) -> None:
        self.language = language
        self.cpu_threads = max(1, int(cpu_threads))
        self.det_limit_side_len = max(640, int(det_limit_side_len))
        self.enable_mkldnn = bool(enable_mkldnn)
        self.microbatch_max_requests = max(
            1,
            int(microbatch_max_requests),
        )
        self.microbatch_max_pixels = max(1, int(microbatch_max_pixels))
        self.microbatch_memory_bytes = max(
            1,
            int(microbatch_memory_bytes),
        )
        self.microbatch_wait_ms = max(0, int(microbatch_wait_ms))
        self._engine = None
        self._detector = None
        self._recognizer = None
        self._recognition_coordinator: RecognitionMicroBatchCoordinator | None = None
        self._model_lock = threading.Lock()
        self._detection_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._detection_metrics: dict[str, int] = {
            "detect_requests": 0,
            "detect_inference_calls": 0,
            "detect_batch_count": 0,
            "detect_pixels": 0,
        }
        self.model_load_count = 0
        self.model_load_ms = 0

    @property
    def model_state(self) -> str:
        if self._engine is not None or (
            self._detector is not None
            and self._recognizer is not None
        ):
            return "ready"
        if self._detector is not None or self._recognizer is not None:
            return "partial"
        return "unloaded"

    def recognize(self, image_path: Path) -> OcrResult:
        engine = self._get_engine()
        result = engine.predict(str(image_path)) if hasattr(engine, "predict") else engine.ocr(str(image_path), cls=True)
        texts: list[str] = []
        confidences: list[float] = []
        boxes: list[object] = []
        for page in result or []:
            if isinstance(page, dict) and "rec_texts" in page:
                rec_texts = page.get("rec_texts") or []
                rec_scores = page.get("rec_scores") or []
                rec_boxes = page.get("rec_boxes")
                if rec_boxes is None:
                    rec_boxes = page.get("rec_polys")
                if rec_boxes is None:
                    rec_boxes = []
                for index, text in enumerate(rec_texts):
                    text = str(text)
                    if text.strip():
                        texts.append(text)
                    if index < len(rec_scores):
                        confidences.append(float(rec_scores[index]))
                    if index < len(rec_boxes):
                        boxes.append(to_jsonable(rec_boxes[index]))
                continue
            for item in page or []:
                if not item or len(item) < 2:
                    continue
                box, payload = item[0], item[1]
                if isinstance(payload, (list, tuple)) and payload:
                    text = str(payload[0])
                    confidence = float(payload[1]) if len(payload) > 1 and payload[1] is not None else None
                    if text.strip():
                        texts.append(text)
                    if confidence is not None:
                        confidences.append(confidence)
                    boxes.append(to_jsonable(box))
        average = sum(confidences) / len(confidences) if confidences else None
        return OcrResult(
            "\n".join(texts),
            average,
            {"boxes": boxes, "language": self.language, "engine": "PaddleOCR", "models_dir": str(OCR_MODELS_DIR)},
        )

    def detect_file_regions(
        self,
        image_path: Path,
    ) -> tuple[list[list[list[float]]], tuple[int, int]]:
        """Detect regions in a bounded preview while preserving pixel coordinates."""

        image = _load_image_for_ocr(image_path)
        height, width = image.shape[:2]
        return self._detect(image), (width, height)

    def recognize_crops(
        self,
        crops: list[object],
        *,
        phase: str = "recognize_crop_microbatch",
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> list[tuple[str, float | None]]:
        """Recognize pre-segmented original-resolution crops in shared microbatches."""

        if not crops:
            return []
        report = progress_callback or (lambda _phase, _completed, _total, _detail: None)
        check = cancel_check or (lambda: None)
        unique_crops: list[object] = []
        unique_indexes: dict[tuple[tuple[int, ...], str, str], int] = {}
        source_to_unique: list[int] = []
        for crop in crops:
            normalized = _normalize_recognition_crop(crop)
            key = _crop_pixel_key(normalized)
            unique_index = unique_indexes.get(key)
            if unique_index is None:
                unique_index = len(unique_crops)
                unique_indexes[key] = unique_index
                unique_crops.append(normalized)
            source_to_unique.append(unique_index)
        pixel_counts = [
            _crop_pixel_count(crop)
            for crop in unique_crops
        ]
        unique_results: list[tuple[str, float | None]] = [
            ("", None)
            for _ in unique_crops
        ]
        report(
            phase,
            0,
            1,
            f"微批识别 {len(unique_crops)} 个区域（原始 {len(crops)} 个）",
        )
        check()
        results = self._get_recognition_coordinator().recognize(
            unique_crops,
            pixel_counts=pixel_counts,
            compatibility_key=(
                self.language,
                model_manifest_fingerprint(OCR_MODELS_DIR),
                "normalized-crop-v1",
            ),
            cancel_check=check,
        )
        for offset, result in enumerate(results):
            text = str(result.get("rec_text") or "").strip()
            score_value = result.get("rec_score")
            confidence = (
                float(score_value)
                if score_value is not None
                else None
            )
            unique_results[offset] = (text, confidence)
        report(
            phase,
            1,
            1,
            f"已识别 {len(unique_crops)}/{len(unique_crops)} 个区域",
        )
        return [unique_results[index] for index in source_to_unique]

    def recognize_adaptive(
        self,
        image_path: Path,
        *,
        tile_side: int = DEFAULT_OCR_TILE_SIDE,
        tile_overlap: int = DEFAULT_OCR_TILE_OVERLAP,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
        checkpoint_path: Path | None = None,
    ) -> OcrResult:
        """Detect at 960px, recognize original crops, and tile low-quality images."""

        image = _load_image_for_ocr(image_path)
        height, width = image.shape[:2]
        tile_side = max(self.det_limit_side_len, int(tile_side))
        tile_overlap = max(64, min(tile_side // 3, int(tile_overlap)))
        check = cancel_check or (lambda: None)
        report = progress_callback or (lambda _phase, _completed, _total, _detail: None)
        source_sha256 = _sha256_path(image_path)
        adaptive_model_fingerprint = (
            f"{self.language}:{self.det_limit_side_len}:"
            f"{model_manifest_fingerprint(OCR_MODELS_DIR)}"
        )
        restored_payload = _load_adaptive_checkpoint(
            checkpoint_path,
            source_sha256=source_sha256,
            width=width,
            height=height,
            strategy_version=ADAPTIVE_OCR_VERSION,
            model_fingerprint=adaptive_model_fingerprint,
        )
        recognition_stats: dict[str, int] = {}
        if restored_payload is not None:
            first_polys = [
                [
                    [float(point[0]), float(point[1])]
                    for point in polygon
                ]
                for polygon in restored_payload.get("first_polys", [])
            ]
            first_lines = [
                _line_from_checkpoint(item)
                for item in restored_payload.get("first_lines", [])
            ]
            text_likely = bool(restored_payload.get("text_likely"))
            fallback_used = bool(restored_payload.get("fallback_used"))
            preview_detect_calls = 0
        else:
            check()
            report("detect", 0, 1, "960 像素首轮文字检测")
            first_polys = self._detect(image)
            preview_detect_calls = 1
            report("detect", 1, 1, f"发现 {len(first_polys)} 个文字区域")
            first_lines = self._recognize_regions(
                image,
                first_polys,
                phase="recognize_original_regions",
                report=report,
                check=check,
                stats=recognition_stats,
            )

            text_likely = _looks_textual(image)
            fallback_used = (
                max(width, height) > self.det_limit_side_len
                and _needs_tile_fallback(
                    first_lines,
                    first_polys,
                    text_likely=text_likely,
                    scale_ratio=max(width, height) / self.det_limit_side_len,
                )
            )
        all_lines = list(first_lines)
        resolved_boxes = [
            line.box
            for line in first_lines
            if (line.confidence or 0.0) >= 0.82 and len(_dedupe_text(line.text)) >= 2
        ]
        tiles_planned = 0
        tiles_pruned = 0
        tile_regions_detected = 0
        tile_regions_skipped_resolved = 0
        tile_count = 0
        adaptive_regions_created = 0
        adaptive_regions_split = 0
        adaptive_regions_pruned_blank = 0
        adaptive_regions_resolved = 0
        adaptive_regions_remaining = 0
        adaptive_regions_remaining_peak = 0
        coverage_ratio = 1.0
        checkpoint_regions_reused = 0
        checkpoint_recognition_batches_reused = 0
        fallback_region_pixels = 0
        if fallback_used:
            anchor_boxes = [
                _polygon_rectangle(polygon)
                for polygon in first_polys
            ]
            if restored_payload is not None and isinstance(
                restored_payload.get("planner"),
                dict,
            ):
                planner = AdaptiveRegionPlanner.from_checkpoint(
                    restored_payload["planner"]
                )
                tile_polygons = [
                    [
                        [float(point[0]), float(point[1])]
                        for point in polygon
                    ]
                    for polygon in restored_payload.get("tile_polygons", [])
                ]
                checkpoint_regions_reused = (
                    planner.resolved_count + planner.split_count
                )
            else:
                planner = AdaptiveRegionPlanner(
                    source_sha256=source_sha256,
                    width=width,
                    height=height,
                    target_side=tile_side,
                    model_fingerprint=adaptive_model_fingerprint,
                    strategy_version=ADAPTIVE_OCR_VERSION,
                    anchors=anchor_boxes,
                )
                tile_polygons = []
            adaptive_regions_remaining_peak = max(
                adaptive_regions_remaining_peak,
                planner.pending_count,
            )

            recovered_tile_lines: dict[str, list[_OcrLine]] = {}
            for item in planner.confirmed_lines:
                batch_id = str(item.get("batch_id") or "")
                box = item.get("box")
                if not batch_id or not isinstance(box, list):
                    continue
                recovered_tile_lines.setdefault(
                    batch_id,
                    [],
                ).append(
                    _OcrLine(
                        text=str(item.get("text") or ""),
                        confidence=(
                            float(item["confidence"])
                            if item.get("confidence") is not None
                            else None
                        ),
                        box=[
                            [float(point[0]), float(point[1])]
                            for point in box
                        ],
                    )
                )

            def persist_adaptive_checkpoint() -> None:
                _write_adaptive_checkpoint(
                    checkpoint_path,
                    {
                        "checkpoint_schema": 1,
                        "source_sha256": source_sha256,
                        "width": width,
                        "height": height,
                        "strategy_version": ADAPTIVE_OCR_VERSION,
                        "model_fingerprint": adaptive_model_fingerprint,
                        "first_polys": first_polys,
                        "first_lines": [
                            _line_to_checkpoint(line)
                            for line in first_lines
                        ],
                        "text_likely": text_likely,
                        "fallback_used": True,
                        "planner": planner.checkpoint(),
                        "tile_polygons": tile_polygons,
                    },
                )

            persist_adaptive_checkpoint()
            report(
                "tile_detect",
                0,
                1,
                "首轮质量不足，开始未解决区域递归规划",
            )
            while planner.pending_count:
                adaptive_regions_remaining_peak = max(
                    adaptive_regions_remaining_peak,
                    planner.pending_count,
                )
                check()
                region = planner.pop_next()
                if region is None:
                    break
                left, top, right, bottom = (
                    region.left,
                    region.top,
                    region.right,
                    region.bottom,
                )
                tile = image[top:bottom, left:right]
                if region.heat <= 0.0 and _tile_is_confidently_blank(tile):
                    planner.resolve(region.region_id, "blank")
                    adaptive_regions_pruned_blank += 1
                    persist_adaptive_checkpoint()
                    report(
                        "adaptive_region",
                        planner.resolved_count,
                        planner.created_count,
                        f"严格空白区域已确认，剩余 {planner.pending_count}",
                    )
                    continue
                if region.width > tile_side or region.height > tile_side:
                    planner.split(region.region_id)
                    adaptive_regions_remaining_peak = max(
                        adaptive_regions_remaining_peak,
                        planner.pending_count,
                    )
                    persist_adaptive_checkpoint()
                    report(
                        "adaptive_split",
                        planner.split_count,
                        planner.created_count,
                        f"按文字热度细分区域，待检查 {planner.pending_count}",
                    )
                    continue
                detected = self._detect(tile)
                fallback_region_pixels += max(0, region.width * region.height)
                tile_count += 1
                tile_regions_detected += len(detected)
                for polygon in detected:
                    global_polygon = [
                        [point[0] + left, point[1] + top]
                        for point in polygon
                    ]
                    if _polygon_covered_by_any(global_polygon, resolved_boxes):
                        tile_regions_skipped_resolved += 1
                        continue
                    tile_polygons.append(global_polygon)
                planner.record_detection_batch(
                    f"detect:{region.region_id}"
                )
                planner.resolve(region.region_id, "inspected")
                persist_adaptive_checkpoint()
                report(
                    "adaptive_region",
                    planner.resolved_count,
                    planner.created_count,
                    f"已确认原图区域，剩余 {planner.pending_count}",
                )
            adaptive_regions_created = planner.created_count
            adaptive_regions_split = planner.split_count
            adaptive_regions_resolved = planner.resolved_count
            adaptive_regions_remaining = planner.pending_count
            coverage_ratio = planner.coverage_ratio
            tiles_planned = adaptive_regions_created
            tiles_pruned = adaptive_regions_pruned_blank
            tile_polygons = _dedupe_polygons(tile_polygons)

            def confirm_recognition_batch(
                batch_id: str,
                batch_lines: list[_OcrLine],
            ) -> None:
                planner.record_recognition_batch(batch_id)
                for line in batch_lines:
                    top = int(
                        min(point[1] for point in line.box)
                        if line.box
                        else 0
                    )
                    left = int(
                        min(point[0] for point in line.box)
                        if line.box
                        else 0
                    )
                    line_identity = hashlib.sha256(
                        json.dumps(
                            {
                                "batch_id": batch_id,
                                "text": line.text,
                                "box": line.box,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()
                    planner.confirm_line(
                        line_id=line_identity,
                        text=line.text,
                        confidence=line.confidence,
                        ordering_key=(
                            0,
                            top,
                            left,
                            line_identity,
                        ),
                        batch_id=batch_id,
                        box=line.box,
                    )
                persist_adaptive_checkpoint()

            tile_lines = self._recognize_regions(
                image,
                tile_polygons,
                phase="tile_recognize_microbatch",
                report=report,
                check=check,
                stats=recognition_stats,
                completed_batch_ids=(
                    planner.completed_recognition_batches
                ),
                recovered_lines_by_batch=recovered_tile_lines,
                on_batch_confirmed=confirm_recognition_batch,
            )
            checkpoint_recognition_batches_reused = int(
                recognition_stats.get(
                    "checkpoint_recognition_batches_reused",
                    0,
                )
            )
            all_lines.extend(tile_lines)

        merged_lines = _merge_duplicate_lines(all_lines)
        confidences = [
            line.confidence
            for line in merged_lines
            if line.confidence is not None
        ]
        average = sum(confidences) / len(confidences) if confidences else None
        text = "\n".join(line.text for line in merged_lines if line.text.strip())
        self._crash_validation_worker_after_models_loaded(
            checkpoint_path
        )
        report("complete", 1, 1, f"识别完成，共 {len(merged_lines)} 行")
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
        return OcrResult(
            text,
            average,
            {
                "boxes": [to_jsonable(line.box) for line in merged_lines],
                "line_confidences": confidences,
                "language": self.language,
                "engine": "PaddleOCR",
                "models_dir": str(OCR_MODELS_DIR),
                "strategy": "adaptive-original-regions",
                "strategy_version": ADAPTIVE_OCR_VERSION,
                "detection_side": self.det_limit_side_len,
                "original_size": [width, height],
                "first_pass_regions": len(first_polys),
                "preview_detect_calls": preview_detect_calls,
                "preview_detect_pixels": _bounded_preview_pixels(
                    width,
                    height,
                    self.det_limit_side_len,
                ),
                "original_region_pixels": int(
                    recognition_stats.get("original_region_pixels", 0)
                ),
                "fallback_region_pixels": (
                    fallback_region_pixels
                    + int(
                        recognition_stats.get(
                            "fallback_recognition_pixels",
                            0,
                        )
                    )
                ),
                "fallback_used": fallback_used,
                "tile_side": tile_side,
                "tile_overlap": tile_overlap,
                "tiles_processed": tile_count,
                "tiles_planned": tiles_planned,
                "tiles_pruned": tiles_pruned,
                "tile_regions_detected": tile_regions_detected,
                "tile_regions_recognized": len(tile_polygons) if fallback_used else 0,
                "tile_regions_skipped_resolved": tile_regions_skipped_resolved,
                "adaptive_regions_created": adaptive_regions_created,
                "adaptive_regions_split": adaptive_regions_split,
                "adaptive_regions_pruned_blank": adaptive_regions_pruned_blank,
                "adaptive_regions_resolved": adaptive_regions_resolved,
                "adaptive_regions_remaining": adaptive_regions_remaining,
                "adaptive_regions_remaining_peak": (
                    adaptive_regions_remaining_peak
                ),
                "coverage_ratio": coverage_ratio,
                "checkpoint_regions_reused": checkpoint_regions_reused,
                "checkpoint_recognition_batches_reused": (
                    checkpoint_recognition_batches_reused
                ),
                "crop_dedup_hits": int(recognition_stats.get("crop_dedup_hits", 0)),
                "recognizer_batches": int(recognition_stats.get("recognizer_batches", 0)),
                "text_likely": text_likely,
                "ocr_model_load_count": self.model_load_count,
                "ocr_model_load_ms": self.model_load_ms,
                "ocr_model_state": self.model_state,
                **self.runtime_metrics_snapshot(),
            },
        )

    def _crash_validation_worker_after_models_loaded(
        self,
        checkpoint_path: Path | None,
    ) -> None:
        marker_text = os.environ.get(
            "LFTS_VALIDATION_OCR_CRASH_MARKER",
            "",
        )
        if not marker_text or self.model_state != "ready":
            return
        marker_path = Path(marker_text)
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                marker_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
        except FileExistsError:
            return
        payload = json.dumps(
            {
                "worker_pid": os.getpid(),
                "model_state": self.model_state,
                "model_load_count": int(self.model_load_count),
                "reason": "validation_injected_after_models_ready",
            },
            ensure_ascii=True,
            sort_keys=True,
        ).encode("utf-8")
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        if checkpoint_path is not None:
            checkpoint_path.unlink(missing_ok=True)
        os._exit(86)

    def _detect(self, image: object) -> list[list[list[float]]]:
        detector = self._get_detector()
        shape = getattr(image, "shape", ())
        pixels = (
            int(shape[0]) * int(shape[1])
            if len(shape) >= 2
            else 0
        )
        with self._detection_lock:
            with self._metrics_lock:
                self._detection_metrics["detect_requests"] += 1
                self._detection_metrics["detect_inference_calls"] += 1
                self._detection_metrics["detect_batch_count"] += 1
                self._detection_metrics["detect_pixels"] += pixels
            results = detector.predict(image)
        if not results:
            return []
        polygons = results[0].get("dt_polys")
        if polygons is None:
            return []
        return _sort_polygons(
            [
                [[float(point[0]), float(point[1])] for point in polygon]
                for polygon in polygons
                if len(polygon) >= 4
            ]
        )

    def _recognize_regions(
        self,
        image: object,
        polygons: list[list[list[float]]],
        *,
        phase: str,
        report: Callable[[str, int, int, str], None],
        check: Callable[[], None],
        stats: dict[str, int] | None = None,
        completed_batch_ids: frozenset[str] | set[str] | None = None,
        recovered_lines_by_batch: dict[str, list[_OcrLine]] | None = None,
        on_batch_confirmed: (
            Callable[[str, list[_OcrLine]], None] | None
        ) = None,
    ) -> list[_OcrLine]:
        if not polygons:
            return []
        crops: list[object] = []
        polygon_groups: list[list[list[list[float]]]] = []
        crop_indexes: dict[tuple[tuple[int, ...], str, str], int] = {}
        for polygon in polygons:
            crop = _crop_text_region(image, polygon)
            if crop is None:
                continue
            crop_key = _crop_pixel_key(crop)
            existing_index = crop_indexes.get(crop_key)
            if existing_index is not None:
                polygon_groups[existing_index].append(polygon)
                if stats is not None:
                    stats["crop_dedup_hits"] = stats.get("crop_dedup_hits", 0) + 1
                continue
            crop_indexes[crop_key] = len(crops)
            crops.append(crop)
            polygon_groups.append([polygon])
        if not crops:
            return []
        batch_indexes: list[list[int]] = []
        current: list[int] = []
        current_pixels = 0
        current_memory = 0
        for index, crop in enumerate(crops):
            pixels = _crop_pixel_count(crop)
            memory = pixels * 4
            if current and (
                len(current) >= self.microbatch_max_requests
                or current_pixels + pixels > self.microbatch_max_pixels
                or (
                    current_memory + memory
                    > self.microbatch_memory_bytes
                )
            ):
                batch_indexes.append(current)
                current = []
                current_pixels = 0
                current_memory = 0
            current.append(index)
            current_pixels += pixels
            current_memory += memory
        if current:
            batch_indexes.append(current)
        completed_ids = set(completed_batch_ids or ())
        recovered = recovered_lines_by_batch or {}
        lines: list[_OcrLine] = []
        batch_count = len(batch_indexes)
        report(
            phase,
            0,
            batch_count,
            f"识别 {len(crops)} 个原图文字区域",
        )
        for completed, indexes in enumerate(batch_indexes, start=1):
            batch_crops = [crops[index] for index in indexes]
            batch_polygon_groups = [
                polygon_groups[index] for index in indexes
            ]
            batch_id = _recognition_checkpoint_batch_id(
                phase,
                batch_crops,
                batch_polygon_groups,
            )
            if batch_id in completed_ids:
                lines.extend(recovered.get(batch_id, ()))
                if stats is not None:
                    stats[
                        "checkpoint_recognition_batches_reused"
                    ] = (
                        stats.get(
                            "checkpoint_recognition_batches_reused",
                            0,
                        )
                        + 1
                    )
                report(
                    phase,
                    completed,
                    batch_count,
                    f"从检查点复用识别批 {completed}/{batch_count}",
                )
                continue
            if stats is not None:
                pixel_count = sum(
                    _crop_pixel_count(crop)
                    for crop in batch_crops
                )
                metric = (
                    "fallback_recognition_pixels"
                    if phase.startswith("tile_")
                    else "original_region_pixels"
                )
                stats[metric] = (
                    stats.get(metric, 0) + pixel_count
                )
                stats["recognizer_batches"] = (
                    stats.get("recognizer_batches", 0) + 1
                )
            recognized = self.recognize_crops(
                batch_crops,
                phase=phase,
                progress_callback=None,
                cancel_check=check,
            )
            batch_lines: list[_OcrLine] = []
            for result, polygon_group in zip(
                recognized,
                batch_polygon_groups,
                strict=False,
            ):
                text, confidence = result
                if text:
                    batch_lines.extend(
                        _OcrLine(text, confidence, polygon)
                        for polygon in polygon_group
                    )
            if on_batch_confirmed is not None:
                on_batch_confirmed(batch_id, batch_lines)
            lines.extend(batch_lines)
            report(
                phase,
                completed,
                batch_count,
                f"已识别批次 {completed}/{batch_count}",
            )
        return lines

    def _get_detector(self) -> object:
        if self._detector is not None:
            return self._detector
        with self._model_lock:
            if self._detector is not None:
                return self._detector
            started = time.perf_counter()
            self._configure_runtime()
            try:
                from paddleocr import TextDetection
            except ImportError as exc:
                raise ParserDependencyError("当前 PaddleOCR 不支持独立文字检测") from exc
            runtime_models_dir = prepare_runtime_models_dir()
            model_dir = runtime_models_dir / "PP-OCRv4_mobile_det"
            _require_model_files(model_dir)
            self._detector = TextDetection(
                model_name="PP-OCRv4_mobile_det",
                model_dir=str(model_dir),
                limit_side_len=self.det_limit_side_len,
                limit_type="max",
                enable_mkldnn=self.enable_mkldnn,
                cpu_threads=self.cpu_threads,
            )
            self.model_load_count += 1
            self.model_load_ms += int((time.perf_counter() - started) * 1000)
        return self._detector

    def _get_recognizer(self) -> object:
        if self._recognizer is not None:
            return self._recognizer
        with self._model_lock:
            if self._recognizer is not None:
                return self._recognizer
            started = time.perf_counter()
            self._configure_runtime()
            try:
                from paddleocr import TextRecognition
            except ImportError as exc:
                raise ParserDependencyError("当前 PaddleOCR 不支持独立文字识别") from exc
            runtime_models_dir = prepare_runtime_models_dir()
            model_dir = runtime_models_dir / "PP-OCRv4_mobile_rec"
            _require_model_files(model_dir)
            self._recognizer = TextRecognition(
                model_name="PP-OCRv4_mobile_rec",
                model_dir=str(model_dir),
                enable_mkldnn=self.enable_mkldnn,
                cpu_threads=self.cpu_threads,
            )
            self.model_load_count += 1
            self.model_load_ms += int((time.perf_counter() - started) * 1000)
        return self._recognizer

    def _get_recognition_coordinator(
        self,
    ) -> RecognitionMicroBatchCoordinator:
        if self._recognition_coordinator is not None:
            return self._recognition_coordinator
        with self._model_lock:
            if self._recognition_coordinator is None:
                recognizer = self._get_recognizer_unlocked()
                self._recognition_coordinator = (
                    RecognitionMicroBatchCoordinator(
                        lambda crops: recognizer.predict(crops),
                        max_requests=self.microbatch_max_requests,
                        max_pixels=self.microbatch_max_pixels,
                        max_memory_bytes=self.microbatch_memory_bytes,
                        max_wait_ms=self.microbatch_wait_ms,
                    )
                )
        return self._recognition_coordinator

    def _get_recognizer_unlocked(self) -> object:
        if self._recognizer is None:
            started = time.perf_counter()
            self._configure_runtime()
            try:
                from paddleocr import TextRecognition
            except ImportError as exc:
                raise ParserDependencyError(
                    "当前 PaddleOCR 不支持独立文字识别"
                ) from exc
            runtime_models_dir = prepare_runtime_models_dir()
            model_dir = runtime_models_dir / "PP-OCRv4_mobile_rec"
            _require_model_files(model_dir)
            self._recognizer = TextRecognition(
                model_name="PP-OCRv4_mobile_rec",
                model_dir=str(model_dir),
                enable_mkldnn=self.enable_mkldnn,
                cpu_threads=self.cpu_threads,
            )
            self.model_load_count += 1
            self.model_load_ms += int(
                (time.perf_counter() - started) * 1000
            )
        return self._recognizer

    def runtime_metrics_snapshot(self) -> dict[str, object]:
        with self._metrics_lock:
            metrics: dict[str, object] = dict(self._detection_metrics)
        # PaddleOCR's standalone TextDetection API accepts one image per
        # predict call in the frozen runtime. Requests share one serialized
        # queue/lock, but that must not be reported as multi-image inference.
        metrics["detection_inference_batch_supported"] = False
        metrics[
            "detection_batch_technical_note"
        ] = "paddle_text_detection_single_image_api"
        coordinator = self._recognition_coordinator
        if coordinator is not None:
            metrics.update(dict(coordinator.metrics))
        else:
            metrics.update(
                {
                    "recognize_requests": 0,
                    "recognize_inference_calls": 0,
                    "recognize_batch_count": 0,
                    "recognize_average_batch_size": 0.0,
                    "recognize_pixels": 0,
                    "microbatch_wait_ms_p50": 0.0,
                    "microbatch_wait_ms_p95": 0.0,
                    "microbatch_wait_ms_max": 0.0,
                    "oversize_single_count": 0,
                    "cancelled_before_batch_count": 0,
                }
            )
        metrics["detect_average_batch_size"] = (
            1.0 if int(metrics.get("detect_batch_count") or 0) else 0.0
        )
        return metrics

    def _configure_runtime(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
        os.environ.setdefault("FLAGS_minloglevel", "2")
        os.environ.setdefault("GLOG_minloglevel", "2")

    def _get_engine(self) -> object:
        if self._engine is None:
            started = time.perf_counter()
            # Team builds ship local OCR models; skip PaddleX host checks so
            # first-run OCR does not depend on external connectivity.
            self._configure_runtime()
            warnings.filterwarnings(
                "ignore",
                message="`lang` and `ocr_version` will be ignored.*",
                category=UserWarning,
            )
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:
                raise ParserDependencyError("未安装 PaddleOCR，无法执行图片 OCR") from exc
            kwargs = {
                "lang": self.language,
                "ocr_version": "PP-OCRv4",
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "enable_mkldnn": self.enable_mkldnn,
                "text_det_limit_side_len": self.det_limit_side_len,
                "cpu_threads": self.cpu_threads,
            }
            runtime_models_dir = prepare_runtime_models_dir()
            if runtime_models_dir.exists():
                kwargs.update(
                    {
                        "text_detection_model_name": "PP-OCRv4_mobile_det",
                        "text_detection_model_dir": str(runtime_models_dir / "PP-OCRv4_mobile_det"),
                        "text_recognition_model_name": "PP-OCRv4_mobile_rec",
                        "text_recognition_model_dir": str(runtime_models_dir / "PP-OCRv4_mobile_rec"),
                    }
                )
            try:
                self._engine = PaddleOCR(**kwargs)
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=False, lang=self.language)
            self.model_load_count += 1
            self.model_load_ms += int((time.perf_counter() - started) * 1000)
        return self._engine


OCR_MODEL_NAMES = (
    "PP-OCRv4_mobile_det",
    "PP-OCRv4_mobile_rec",
)
OCR_MODEL_FILES = ("inference.json", "inference.pdiparams", "inference.yml")


def prepare_runtime_models_dir(
    source_dir: Path = OCR_MODELS_DIR,
    cache_roots: Sequence[Path] | None = None,
) -> Path:
    """Return a Paddle-compatible model path, staging only for non-ASCII paths.

    Paddle's Windows C++ inference layer cannot reliably open model files when
    their path contains non-ASCII characters. Frozen apps are commonly unpacked
    below a Chinese directory name, so copy the three local models once into an
    ASCII cache and reuse the fingerprinted directory on later runs.
    """

    if not source_dir.exists() or str(source_dir).isascii():
        return source_dir
    model_key = _models_key(source_dir)
    roots = list(cache_roots) if cache_roots is not None else _default_ascii_cache_roots()
    errors: list[str] = []
    for cache_root in roots:
        if not str(cache_root).isascii():
            continue
        destination = cache_root / model_key
        try:
            cache_root.mkdir(parents=True, exist_ok=True)
            if _models_complete(destination):
                return destination
            temporary = cache_root / f".{model_key}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            try:
                temporary.mkdir()
                for model_name in OCR_MODEL_NAMES:
                    shutil.copytree(
                        source_dir / model_name,
                        temporary / model_name,
                        ignore=shutil.ignore_patterns(".cache"),
                    )
                (temporary / ".ready").write_text(model_key, encoding="ascii")
                try:
                    temporary.replace(destination)
                except FileExistsError:
                    shutil.rmtree(temporary, ignore_errors=True)
                if _models_complete(destination):
                    return destination
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
        except OSError as exc:
            errors.append(f"{cache_root}: {exc}")
    detail = "; ".join(errors) or "no writable ASCII cache directory"
    raise RuntimeError(f"Unable to stage OCR models to an ASCII path: {detail}")


def _default_ascii_cache_roots() -> list[Path]:
    roots = [CACHE_DIR / "runtime_ocr_models"]
    program_data = os.environ.get("PROGRAMDATA")
    if program_data:
        roots.append(Path(program_data) / APP_NAME / "runtime_ocr_models")
    roots.append(Path(tempfile.gettempdir()) / APP_NAME / "runtime_ocr_models")
    return roots


def _models_key(source_dir: Path) -> str:
    fingerprint = model_manifest_fingerprint(source_dir)
    digest = hashlib.sha256(fingerprint.encode("ascii", errors="backslashreplace")).hexdigest()
    return f"models-{digest[:16]}"


def _models_complete(path: Path) -> bool:
    if not (path / ".ready").is_file():
        return False
    return all(
        (path / model_name / file_name).is_file()
        for model_name in OCR_MODEL_NAMES
        for file_name in OCR_MODEL_FILES
    )


def to_jsonable(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def _require_model_files(model_dir: Path) -> None:
    missing = [
        file_name
        for file_name in OCR_MODEL_FILES
        if not (model_dir / file_name).is_file()
    ]
    if missing:
        raise ParserDependencyError(
            f"OCR 模型不完整：{model_dir.name} 缺少 {', '.join(missing)}"
        )


def _load_image_for_ocr(image_path: Path) -> object:
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(image_path) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            array = np.asarray(rgb).copy()
        return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    except Exception as exc:
        raise RuntimeError(f"无法读取图片：{exc}") from exc


def _crop_text_region(
    image: object,
    polygon: list[list[float]],
) -> object | None:
    import cv2
    import numpy as np

    points = np.asarray(polygon[:4], dtype=np.float32)
    center = points.mean(axis=0)
    points = center + (points - center) * 1.08
    height, width = image.shape[:2]
    points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
    points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
    crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3]),
        )
    )
    crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2]),
        )
    )
    if crop_width < 2 or crop_height < 2:
        return None
    destination = np.asarray(
        [
            [0, 0],
            [crop_width - 1, 0],
            [crop_width - 1, crop_height - 1],
            [0, crop_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    crop = cv2.warpPerspective(
        image,
        transform,
        (crop_width, crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if crop_height / max(1, crop_width) >= 1.5:
        crop = np.rot90(crop)
    return crop


def _sort_polygons(
    polygons: list[list[list[float]]],
) -> list[list[list[float]]]:
    return sorted(
        polygons,
        key=lambda polygon: (
            round(sum(point[1] for point in polygon) / len(polygon) / 12),
            min(point[0] for point in polygon),
        ),
    )


def _tile_plan(
    width: int,
    height: int,
    tile_side: int,
    overlap: int,
) -> list[tuple[int, int, int, int]]:
    x_positions = _tile_positions(width, tile_side, overlap)
    y_positions = _tile_positions(height, tile_side, overlap)
    return [
        (left, top, min(width, left + tile_side), min(height, top + tile_side))
        for top in y_positions
        for left in x_positions
    ]


def _adaptive_tile_plan(
    image: object,
    tile_side: int,
    overlap: int,
    *,
    anchors: list[list[list[float]]] | None = None,
) -> list[tuple[int, int, int, int]]:
    """Prune only confidently blank tiles and retain neighbours of detections.

    The rule is deliberately recall-first: any tile with non-trivial visual
    structure is retained.  A tile is skipped only when its low-resolution
    sample is essentially uniform and it is not next to a first-pass region.
    """

    height, width = image.shape[:2]
    tiles = _tile_plan(width, height, tile_side, overlap)
    if len(tiles) <= 1:
        return tiles
    anchor_boxes = [_axis_box(polygon) for polygon in anchors or []]
    anchored: set[int] = {
        index
        for index, tile in enumerate(tiles)
        if any(_rectangles_intersect(tile, anchor) for anchor in anchor_boxes)
    }
    protected = set(anchored)
    for anchored_index in anchored:
        anchored_tile = tiles[anchored_index]
        anchored_center = (
            (anchored_tile[0] + anchored_tile[2]) / 2,
            (anchored_tile[1] + anchored_tile[3]) / 2,
        )
        for index, tile in enumerate(tiles):
            center = ((tile[0] + tile[2]) / 2, (tile[1] + tile[3]) / 2)
            if (
                abs(center[0] - anchored_center[0]) <= tile_side + overlap
                and abs(center[1] - anchored_center[1]) <= tile_side + overlap
            ):
                protected.add(index)
    selected = [
        tile
        for index, tile in enumerate(tiles)
        if index in protected
        or not _tile_is_confidently_blank(
            image[tile[1] : tile[3], tile[0] : tile[2]]
        )
    ]
    # A low-quality first pass already established that a fallback is needed.
    # If the conservative classifier cannot select anything, inspect all tiles
    # instead of risking a recall regression.
    return selected or tiles


def _tile_is_confidently_blank(tile: object) -> bool:
    import cv2
    import numpy as np

    height, width = tile.shape[:2]
    scale = min(1.0, 384 / max(1, width, height))
    if scale < 1.0:
        sample = cv2.resize(
            tile,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = tile
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    standard_deviation = float(np.std(gray))
    tonal_range = int(np.max(gray)) - int(np.min(gray))
    if standard_deviation <= 2.0 and tonal_range <= 12:
        return True
    if standard_deviation >= 8.0:
        return False
    edges = cv2.Canny(gray, 45, 150)
    edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
    return edge_density < 0.0003


def _tile_positions(length: int, tile_side: int, overlap: int) -> list[int]:
    if length <= tile_side:
        return [0]
    step = max(1, tile_side - overlap)
    positions = list(range(0, max(1, length - tile_side + 1), step))
    last = length - tile_side
    if not positions or positions[-1] != last:
        positions.append(last)
    return positions


def _needs_tile_fallback(
    lines: list[_OcrLine],
    polygons: list[list[list[float]]],
    *,
    text_likely: bool,
    scale_ratio: float,
) -> bool:
    if not lines:
        return text_likely
    confidences = [
        line.confidence
        for line in lines
        if line.confidence is not None
    ]
    average = sum(confidences) / len(confidences) if confidences else 0.0
    low_ratio = (
        sum(score < 0.65 for score in confidences) / len(confidences)
        if confidences
        else 1.0
    )
    character_count = sum(len(re.sub(r"\s+", "", line.text)) for line in lines)
    if average < 0.78 or low_ratio > 0.30:
        return True
    if character_count < max(8, len(polygons) * 2):
        return True
    return text_likely and scale_ratio >= 2.0 and character_count < 40


def _looks_textual(image: object) -> bool:
    import cv2
    import numpy as np

    height, width = image.shape[:2]
    scale = min(1.0, 960 / max(width, height))
    if scale < 1.0:
        sample = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        sample = image
    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 180)
    edge_density = float(np.count_nonzero(edges)) / max(1, edges.size)
    if edge_density < 0.006 or edge_density > 0.40:
        return False
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    component_count = 0
    for index in range(1, labels):
        item_width = int(stats[index, cv2.CC_STAT_WIDTH])
        item_height = int(stats[index, cv2.CC_STAT_HEIGHT])
        area = int(stats[index, cv2.CC_STAT_AREA])
        if (
            2 <= item_width <= 180
            and 3 <= item_height <= 100
            and 6 <= area <= 4000
            and 0.08 <= item_width / max(1, item_height) <= 12
        ):
            component_count += 1
    return component_count >= 12


def _crop_pixel_key(crop: object) -> tuple[tuple[int, ...], str, str]:
    import numpy as np

    contiguous = np.ascontiguousarray(crop)
    digest = hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest()
    return tuple(int(value) for value in contiguous.shape), str(contiguous.dtype), digest


def _crop_pixel_count(crop: object) -> int:
    shape = tuple(int(value) for value in getattr(crop, "shape", ()))
    if len(shape) < 2:
        return 1
    return max(1, shape[0] * shape[1])


def _recognition_checkpoint_batch_id(
    phase: str,
    crops: list[object],
    polygon_groups: list[list[list[list[float]]]],
) -> str:
    identity = {
        "phase": str(phase),
        "crops": [
            {
                "shape": list(key[0]),
                "dtype": key[1],
                "pixel_digest": key[2],
                "polygons": polygon_group,
            }
            for key, polygon_group in zip(
                (_crop_pixel_key(crop) for crop in crops),
                polygon_groups,
                strict=True,
            )
        ],
    }
    return hashlib.sha256(
        json.dumps(
            identity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _normalize_recognition_crop(crop: object) -> object:
    import numpy as np

    height, width = crop.shape[:2]
    if height / max(1, width) >= 1.5:
        return np.rot90(crop)
    return crop


def _polygon_covered_by_any(
    polygon: list[list[float]],
    resolved_boxes: list[list[list[float]]],
) -> bool:
    candidate = _axis_box(polygon)
    candidate_area = _box_area(candidate)
    if candidate_area <= 0:
        return False
    for resolved in resolved_boxes:
        resolved_axis = _axis_box(resolved)
        resolved_area = _box_area(resolved_axis)
        if resolved_area <= 0:
            continue
        area_ratio = candidate_area / resolved_area
        if (
            0.80 <= area_ratio <= 1.25
            and _box_iou(candidate, resolved_axis) >= 0.85
        ):
            return True
    return False


def _polygon_rectangle(
    polygon: list[list[float]],
) -> tuple[int, int, int, int]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return (
        int(min(xs)),
        int(min(ys)),
        int(math.ceil(max(xs))),
        int(math.ceil(max(ys))),
    )


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bounded_preview_pixels(width: int, height: int, limit_side: int) -> int:
    longest = max(1, int(width), int(height))
    scale = min(1.0, max(1, int(limit_side)) / longest)
    return max(1, int(round(width * scale))) * max(
        1,
        int(round(height * scale)),
    )


def _line_to_checkpoint(line: _OcrLine) -> dict[str, object]:
    return {
        "text": line.text,
        "confidence": line.confidence,
        "box": to_jsonable(line.box),
    }


def _line_from_checkpoint(payload: object) -> _OcrLine:
    item = dict(payload) if isinstance(payload, dict) else {}
    return _OcrLine(
        text=str(item.get("text") or ""),
        confidence=(
            float(item["confidence"])
            if item.get("confidence") is not None
            else None
        ),
        box=[
            [float(point[0]), float(point[1])]
            for point in item.get("box", [])
        ],
    )


def _load_adaptive_checkpoint(
    checkpoint_path: Path | None,
    *,
    source_sha256: str,
    width: int,
    height: int,
    strategy_version: str,
    model_fingerprint: str,
) -> dict[str, Any] | None:
    if checkpoint_path is None or not checkpoint_path.is_file():
        return None
    try:
        payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            int(payload.get("checkpoint_schema", 0)) != 1
            or str(payload.get("source_sha256") or "") != source_sha256
            or int(payload.get("width") or 0) != int(width)
            or int(payload.get("height") or 0) != int(height)
            or str(payload.get("strategy_version") or "") != strategy_version
            or str(payload.get("model_fingerprint") or "") != model_fingerprint
            or not isinstance(payload.get("planner"), dict)
        ):
            raise ValueError("Adaptive OCR checkpoint identity mismatch")
        return payload
    except Exception:
        checkpoint_path.unlink(missing_ok=True)
        return None


def _write_adaptive_checkpoint(
    checkpoint_path: Path | None,
    payload: dict[str, object],
) -> None:
    if checkpoint_path is None:
        return
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_name(
        f".{checkpoint_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, checkpoint_path)
    finally:
        temporary.unlink(missing_ok=True)


def _dedupe_polygons(
    polygons: list[list[list[float]]],
) -> list[list[list[float]]]:
    unique: list[list[list[float]]] = []
    for polygon in _sort_polygons(polygons):
        candidate = _axis_box(polygon)
        candidate_area = _box_area(candidate)
        duplicate = False
        for existing in unique:
            existing_box = _axis_box(existing)
            existing_area = _box_area(existing_box)
            area_ratio = candidate_area / max(1.0, existing_area)
            if (
                0.85 <= area_ratio <= 1.18
                and _box_iou(candidate, existing_box) >= 0.92
            ):
                duplicate = True
                break
        if not duplicate:
            unique.append(polygon)
    return unique


def _rectangles_intersect(
    first: tuple[float, float, float, float] | tuple[int, int, int, int],
    second: tuple[float, float, float, float],
) -> bool:
    return (
        min(float(first[2]), second[2]) > max(float(first[0]), second[0])
        and min(float(first[3]), second[3]) > max(float(first[1]), second[1])
    )


def _merge_duplicate_lines(lines: list[_OcrLine]) -> list[_OcrLine]:
    merged: list[_OcrLine] = []
    for line in sorted(lines, key=_line_sort_key):
        duplicate_index = next(
            (
                index
                for index, existing in enumerate(merged)
                if _lines_overlap(existing, line)
            ),
            None,
        )
        if duplicate_index is None:
            merged.append(line)
            continue
        existing = merged[duplicate_index]
        if (line.confidence or 0.0) > (existing.confidence or 0.0):
            merged[duplicate_index] = line
    return sorted(merged, key=_line_sort_key)


def _line_sort_key(line: _OcrLine) -> tuple[int, float]:
    top = min(point[1] for point in line.box)
    left = min(point[0] for point in line.box)
    line_height = max(point[1] for point in line.box) - top
    return round(top / max(8.0, line_height * 0.6)), left


def _lines_overlap(first: _OcrLine, second: _OcrLine) -> bool:
    first_text = _dedupe_text(first.text)
    second_text = _dedupe_text(second.text)
    if not first_text or not second_text:
        return False
    same_text = (
        first_text == second_text
        or (min(len(first_text), len(second_text)) >= 4 and (
            first_text in second_text or second_text in first_text
        ))
    )
    if not same_text:
        return False
    return _box_iou(_axis_box(first.box), _axis_box(second.box)) >= 0.30


def _dedupe_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value, flags=re.UNICODE).lower()


def _axis_box(points: list[list[float]]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def _box_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = _box_intersection(first, second)
    if intersection <= 0:
        return 0.0
    return intersection / max(1.0, _box_area(first) + _box_area(second) - intersection)


def _box_area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _box_intersection(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)
