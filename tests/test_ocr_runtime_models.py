from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from local_full_text_search.ocr.image_preprocess import preprocess_image
from local_full_text_search.ocr.ocr_engine import OcrEngine, prepare_runtime_models_dir


class OcrRuntimeModelTests(unittest.TestCase):
    def test_small_non_ascii_image_is_staged_to_an_ascii_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source_dir = base / "图片"
            source_dir.mkdir()
            source = source_dir / "示例.png"
            Image.new("RGB", (120, 80), "white").save(source)
            ascii_temp = base / "ascii_temp"

            with patch(
                "local_full_text_search.ocr.image_preprocess.TEMP_DIR",
                ascii_temp,
            ):
                staged = preprocess_image(source, max_side=2400)

            self.assertNotEqual(staged, source)
            self.assertTrue(staged.is_file())
            self.assertTrue(str(staged).isascii())
            with Image.open(staged) as image:
                self.assertEqual(image.size, (120, 80))

    def test_ocr_engine_uses_bounded_cpu_inference_without_orientation(self) -> None:
        captured: dict[str, object] = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCR = FakePaddleOCR
        with tempfile.TemporaryDirectory() as tmp:
            missing_models = Path(tmp) / "missing"
            with patch.dict("sys.modules", {"paddleocr": fake_module}), patch(
                "local_full_text_search.ocr.ocr_engine.prepare_runtime_models_dir",
                return_value=missing_models,
            ):
                OcrEngine(cpu_threads=2, det_limit_side_len=1280)._get_engine()

        self.assertFalse(captured["use_textline_orientation"])
        self.assertFalse(captured["enable_mkldnn"])
        self.assertEqual(captured["ocr_version"], "PP-OCRv4")
        self.assertEqual(captured["text_det_limit_side_len"], 1280)
        self.assertEqual(captured["cpu_threads"], 2)

    def test_ocr_engine_binds_mobile_model_names_to_local_directories(self) -> None:
        captured: dict[str, object] = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        fake_module = types.ModuleType("paddleocr")
        fake_module.PaddleOCR = FakePaddleOCR
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            models.mkdir()
            with patch.dict("sys.modules", {"paddleocr": fake_module}), patch(
                "local_full_text_search.ocr.ocr_engine.prepare_runtime_models_dir",
                return_value=models,
            ):
                OcrEngine()._get_engine()

        self.assertEqual(captured["text_detection_model_name"], "PP-OCRv4_mobile_det")
        self.assertEqual(captured["text_recognition_model_name"], "PP-OCRv4_mobile_rec")
        self.assertEqual(Path(str(captured["text_detection_model_dir"])).name, "PP-OCRv4_mobile_det")
        self.assertEqual(Path(str(captured["text_recognition_model_dir"])).name, "PP-OCRv4_mobile_rec")

    def test_non_ascii_model_path_is_staged_once_without_download_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "模型"
            cache_root = base / "ascii_cache"
            for model_name in (
                "PP-OCRv4_mobile_det",
                "PP-OCRv4_mobile_rec",
            ):
                model_dir = source / model_name
                model_dir.mkdir(parents=True)
                (model_dir / "inference.json").write_text("{}", encoding="ascii")
                (model_dir / "inference.pdiparams").write_bytes(b"params")
                (model_dir / "inference.yml").write_text("model: test", encoding="ascii")
                hidden_cache = model_dir / ".cache"
                hidden_cache.mkdir()
                (hidden_cache / "unused").write_text("unused", encoding="ascii")

            first = prepare_runtime_models_dir(source, [cache_root])
            second = prepare_runtime_models_dir(source, [cache_root])

            self.assertEqual(first, second)
            self.assertTrue(str(first).isascii())
            self.assertTrue((first / ".ready").is_file())
            self.assertFalse((first / "PP-OCRv4_mobile_det" / ".cache").exists())

    def test_ascii_model_path_is_used_without_copying(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "models"
            source.mkdir()
            self.assertEqual(prepare_runtime_models_dir(source, []), source)

    def test_adaptive_ocr_recognizes_original_region_without_tiling_high_quality_result(self) -> None:
        import numpy as np

        class FakeDetector:
            def predict(self, image: object) -> list[dict[str, object]]:
                height, width = image.shape[:2]
                return [
                    {
                        "dt_polys": np.asarray(
                            [[[100, 100], [min(width - 1, 900), 100], [min(width - 1, 900), 240], [100, 240]]],
                            dtype=np.float32,
                        )
                    }
                ]

        crop_shapes: list[tuple[int, int]] = []

        class FakeRecognizer:
            def predict(self, crops: list[object]) -> list[dict[str, object]]:
                crop_shapes.extend((crop.shape[1], crop.shape[0]) for crop in crops)
                return [
                    {
                        "rec_text": "原图文字区域识别结果足够长，不需要触发低质量分块复核。",
                        "rec_score": 0.96,
                    }
                    for _ in crops
                ]

        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "large.png"
            Image.new("RGB", (2400, 1200), "white").save(source)
            engine = OcrEngine(det_limit_side_len=960)
            engine._detector = FakeDetector()
            engine._recognizer = FakeRecognizer()

            result = engine.recognize_adaptive(source)

        self.assertFalse(result.extra["fallback_used"])
        self.assertGreater(crop_shapes[0][0], 700)
        self.assertEqual(result.extra["detection_side"], 960)

    def test_adaptive_ocr_tiles_low_confidence_large_image_and_reports_progress(self) -> None:
        import numpy as np

        class FakeDetector:
            def predict(self, image: object) -> list[dict[str, object]]:
                height, width = image.shape[:2]
                return [
                    {
                        "dt_polys": np.asarray(
                            [[[20, 20], [min(width - 1, 420), 20], [min(width - 1, 420), 100], [20, 100]]],
                            dtype=np.float32,
                        )
                    }
                ]

        class FakeRecognizer:
            def predict(self, crops: list[object]) -> list[dict[str, object]]:
                return [
                    {"rec_text": "低置信度文字", "rec_score": 0.42}
                    for _ in crops
                ]

        progress: list[tuple[str, int, int]] = []
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "large.png"
            Image.new("RGB", (2400, 1600), "white").save(source)
            engine = OcrEngine(det_limit_side_len=960)
            engine._detector = FakeDetector()
            engine._recognizer = FakeRecognizer()

            result = engine.recognize_adaptive(
                source,
                progress_callback=lambda phase, completed, total, _detail: progress.append(
                    (phase, completed, total)
                ),
            )

        self.assertTrue(result.extra["fallback_used"])
        self.assertGreater(result.extra["tiles_processed"], 1)
        self.assertTrue(any(phase == "tile" and completed > 0 for phase, completed, _ in progress))


if __name__ == "__main__":
    unittest.main()
