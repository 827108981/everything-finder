from __future__ import annotations

import os
import hashlib
import shutil
import tempfile
import uuid
import warnings
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

    def _get_engine(self) -> object:
        if self._engine is None:
            # Team builds ship local OCR models; skip PaddleX host checks so
            # first-run OCR does not depend on external connectivity.
            os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
            os.environ.setdefault("PADDLE_PDX_CPU_NUM_THREADS", str(self.cpu_threads))
            os.environ.setdefault("OMP_NUM_THREADS", str(self.cpu_threads))
            os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")
            os.environ.setdefault("FLAGS_minloglevel", "2")
            os.environ.setdefault("GLOG_minloglevel", "2")
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
