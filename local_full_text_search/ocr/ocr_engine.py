from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path

from local_full_text_search.config.constants import OCR_MODELS_DIR
from local_full_text_search.core.errors import ParserDependencyError


@dataclass(slots=True)
class OcrResult:
    text: str
    confidence: float | None
    extra: dict[str, object]


class OcrEngine:
    """Lazy PaddleOCR wrapper. The heavy dependency is imported only when OCR runs."""

    def __init__(self, language: str = "ch") -> None:
        self.language = language
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
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": True,
            }
            if OCR_MODELS_DIR.exists():
                kwargs.update(
                    {
                        "textline_orientation_model_dir": str(OCR_MODELS_DIR / "PP-LCNet_x1_0_textline_ori"),
                        "text_detection_model_dir": str(OCR_MODELS_DIR / "PP-OCRv6_medium_det"),
                        "text_recognition_model_dir": str(OCR_MODELS_DIR / "PP-OCRv6_medium_rec"),
                    }
                )
            try:
                self._engine = PaddleOCR(**kwargs)
            except TypeError:
                self._engine = PaddleOCR(use_angle_cls=True, lang=self.language)
        return self._engine


def to_jsonable(value: object) -> object:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
