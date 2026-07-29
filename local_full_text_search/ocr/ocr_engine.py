from __future__ import annotations

import os
import hashlib
import math
import re
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from local_full_text_search.config.constants import APP_NAME, CACHE_DIR, OCR_MODELS_DIR
from local_full_text_search.core.errors import ParserDependencyError


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


ADAPTIVE_OCR_VERSION = "1"
DEFAULT_OCR_TILE_SIDE = 1280
DEFAULT_OCR_TILE_OVERLAP = 160


class OcrEngine:
    """Lazy PaddleOCR wrapper. The heavy dependency is imported only when OCR runs."""

    def __init__(
        self,
        language: str = "ch",
        cpu_threads: int = 2,
        det_limit_side_len: int = 960,
    ) -> None:
        self.language = language
        self.cpu_threads = max(1, int(cpu_threads))
        self.det_limit_side_len = max(640, int(det_limit_side_len))
        self._engine = None
        self._detector = None
        self._recognizer = None

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

    def recognize_adaptive(
        self,
        image_path: Path,
        *,
        tile_side: int = DEFAULT_OCR_TILE_SIDE,
        tile_overlap: int = DEFAULT_OCR_TILE_OVERLAP,
        progress_callback: Callable[[str, int, int, str], None] | None = None,
        cancel_check: Callable[[], None] | None = None,
    ) -> OcrResult:
        """Detect at 960px, recognize original crops, and tile low-quality images."""

        image = _load_image_for_ocr(image_path)
        height, width = image.shape[:2]
        tile_side = max(self.det_limit_side_len, int(tile_side))
        tile_overlap = max(64, min(tile_side // 3, int(tile_overlap)))
        check = cancel_check or (lambda: None)
        report = progress_callback or (lambda _phase, _completed, _total, _detail: None)

        check()
        report("detect", 0, 1, "960 像素首轮文字检测")
        first_polys = self._detect(image)
        report("detect", 1, 1, f"发现 {len(first_polys)} 个文字区域")
        first_lines = self._recognize_regions(
            image,
            first_polys,
            phase="recognize_original_regions",
            report=report,
            check=check,
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
        tile_count = 0
        if fallback_used:
            tiles = _tile_plan(width, height, tile_side, tile_overlap)
            tile_count = len(tiles)
            report("tile", 0, tile_count, "首轮质量不足，开始原图分块复核")
            for tile_index, (left, top, right, bottom) in enumerate(tiles, start=1):
                check()
                tile = image[top:bottom, left:right]
                tile_polys = self._detect(tile)
                tile_lines = self._recognize_regions(
                    tile,
                    tile_polys,
                    phase=f"tile_recognize_{tile_index}",
                    report=report,
                    check=check,
                )
                for line in tile_lines:
                    line.box = [
                        [point[0] + left, point[1] + top]
                        for point in line.box
                    ]
                all_lines.extend(tile_lines)
                report(
                    "tile",
                    tile_index,
                    tile_count,
                    f"已复核原图分块 {tile_index}/{tile_count}",
                )

        merged_lines = _merge_duplicate_lines(all_lines)
        confidences = [
            line.confidence
            for line in merged_lines
            if line.confidence is not None
        ]
        average = sum(confidences) / len(confidences) if confidences else None
        text = "\n".join(line.text for line in merged_lines if line.text.strip())
        report("complete", 1, 1, f"识别完成，共 {len(merged_lines)} 行")
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
                "fallback_used": fallback_used,
                "tile_side": tile_side,
                "tile_overlap": tile_overlap,
                "tiles_processed": tile_count,
                "text_likely": text_likely,
            },
        )

    def _detect(self, image: object) -> list[list[list[float]]]:
        detector = self._get_detector()
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
    ) -> list[_OcrLine]:
        if not polygons:
            return []
        crops: list[object] = []
        valid_polygons: list[list[list[float]]] = []
        for polygon in polygons:
            crop = _crop_text_region(image, polygon)
            if crop is None:
                continue
            crops.append(crop)
            valid_polygons.append(polygon)
        if not crops:
            return []
        recognizer = self._get_recognizer()
        batch_size = 16
        batch_count = math.ceil(len(crops) / batch_size)
        lines: list[_OcrLine] = []
        report(phase, 0, batch_count, f"识别 {len(crops)} 个原图文字区域")
        for batch_index, start in enumerate(range(0, len(crops), batch_size), start=1):
            check()
            results = recognizer.predict(crops[start : start + batch_size])
            batch_polygons = valid_polygons[start : start + batch_size]
            for result, polygon in zip(results, batch_polygons, strict=False):
                text = str(result.get("rec_text") or "").strip()
                score_value = result.get("rec_score")
                confidence = float(score_value) if score_value is not None else None
                if text:
                    lines.append(_OcrLine(text, confidence, polygon))
            report(
                phase,
                batch_index,
                batch_count,
                f"已识别文字区域 {min(start + batch_size, len(crops))}/{len(crops)}",
            )
        return lines

    def _get_detector(self) -> object:
        if self._detector is None:
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
                enable_mkldnn=False,
                cpu_threads=self.cpu_threads,
            )
        return self._detector

    def _get_recognizer(self) -> object:
        if self._recognizer is None:
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
                enable_mkldnn=False,
                cpu_threads=self.cpu_threads,
            )
        return self._recognizer

    def _configure_runtime(self) -> None:
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
        os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
        os.environ.setdefault("FLAGS_minloglevel", "2")
        os.environ.setdefault("GLOG_minloglevel", "2")

    def _get_engine(self) -> object:
        if self._engine is None:
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
                "enable_mkldnn": False,
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
    digest = hashlib.sha256()
    for model_name in OCR_MODEL_NAMES:
        for file_name in OCR_MODEL_FILES:
            path = source_dir / model_name / file_name
            stat = path.stat()
            digest.update(f"{model_name}/{file_name}:{stat.st_size}".encode("ascii"))
            if file_name == "inference.yml":
                digest.update(path.read_bytes())
    return f"models-{digest.hexdigest()[:16]}"


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
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    return intersection / max(1.0, first_area + second_area - intersection)
