from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from local_full_text_search.ocr.image_preprocess import preprocess_image
from local_full_text_search.ocr.ocr_engine import (
    OcrEngine,
    _adaptive_tile_plan,
    _dedupe_polygons,
    _polygon_covered_by_any,
    _tile_plan,
    prepare_runtime_models_dir,
)


class OcrRuntimeModelTests(unittest.TestCase):
    def test_detection_metrics_do_not_mislabel_serial_queue_as_inference_batch(
        self,
    ) -> None:
        metrics = OcrEngine().runtime_metrics_snapshot()

        self.assertFalse(
            metrics["detection_inference_batch_supported"]
        )
        self.assertEqual(
            metrics["detection_batch_technical_note"],
            "paddle_text_detection_single_image_api",
        )

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

        recognition_batch_sizes: list[int] = []

        class FakeRecognizer:
            def predict(self, crops: list[object]) -> list[dict[str, object]]:
                recognition_batch_sizes.append(len(crops))
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
        self.assertGreaterEqual(result.extra["tiles_processed"], 1)
        self.assertGreater(result.extra["adaptive_regions_split"], 0)
        self.assertGreater(
            result.extra["adaptive_regions_remaining_peak"],
            0,
        )
        self.assertEqual(result.extra["adaptive_regions_remaining"], 0)
        self.assertEqual(result.extra["coverage_ratio"], 1.0)
        self.assertTrue(
            any(
                phase == "adaptive_region" and completed > 0
                for phase, completed, _ in progress
            )
        )
        self.assertGreaterEqual(result.extra["crop_dedup_hits"], 0)
        self.assertEqual(len(recognition_batch_sizes), 2)
        self.assertEqual(result.extra["preview_detect_calls"], 1)
        self.assertGreater(result.extra["preview_detect_pixels"], 0)
        self.assertGreater(result.extra["original_region_pixels"], 0)
        self.assertGreater(result.extra["fallback_region_pixels"], 0)

    def test_adaptive_ocr_recovers_unresolved_region_graph_from_checkpoint(self) -> None:
        import numpy as np

        class FakeDetector:
            def predict(self, image: object) -> list[dict[str, object]]:
                height, width = image.shape[:2]
                return [
                    {
                        "dt_polys": np.asarray(
                            [
                                [
                                    [20, 20],
                                    [min(width - 1, 420), 20],
                                    [min(width - 1, 420), 100],
                                    [20, 100],
                                ]
                            ],
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

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "checkpoint-large.png"
            checkpoint = base / "adaptive-checkpoint.json"
            Image.new("RGB", (2400, 1600), "white").save(source)
            first_engine = OcrEngine(det_limit_side_len=960)
            first_engine._detector = FakeDetector()
            first_engine._recognizer = FakeRecognizer()

            def interrupt_after_saved_split(
                phase: str,
                _completed: int,
                _total: int,
                _detail: str,
            ) -> None:
                if phase == "adaptive_split":
                    raise RuntimeError("simulated worker crash")

            with self.assertRaisesRegex(RuntimeError, "simulated worker crash"):
                first_engine.recognize_adaptive(
                    source,
                    checkpoint_path=checkpoint,
                    progress_callback=interrupt_after_saved_split,
                )
            self.assertTrue(checkpoint.is_file())

            resumed_engine = OcrEngine(det_limit_side_len=960)
            resumed_engine._detector = FakeDetector()
            resumed_engine._recognizer = FakeRecognizer()
            result = resumed_engine.recognize_adaptive(
                source,
                checkpoint_path=checkpoint,
            )

        self.assertGreater(result.extra["checkpoint_regions_reused"], 0)
        self.assertFalse(checkpoint.exists())
        self.assertEqual(result.extra["adaptive_regions_remaining"], 0)

    def test_adaptive_ocr_resumes_after_confirmed_recognition_batch(
        self,
    ) -> None:
        import numpy as np

        class FakeDetector:
            def predict(self, image: object) -> list[dict[str, object]]:
                height, width = image.shape[:2]
                return [
                    {
                        "dt_polys": np.asarray(
                            [
                                [
                                    [20, 20],
                                    [min(width - 1, 360), 20],
                                    [min(width - 1, 360), 100],
                                    [20, 100],
                                ]
                            ],
                            dtype=np.float32,
                        )
                    }
                ]

        first_calls: list[int] = []

        class CrashingRecognizer:
            def predict(
                self,
                crops: list[object],
            ) -> list[dict[str, object]]:
                first_calls.append(len(crops))
                if len(first_calls) == 3:
                    raise RuntimeError("crash after first tile batch")
                return [
                    {"rec_text": "低置信度批次文字", "rec_score": 0.42}
                    for _ in crops
                ]

        resumed_calls: list[int] = []

        class ResumedRecognizer:
            def predict(
                self,
                crops: list[object],
            ) -> list[dict[str, object]]:
                resumed_calls.append(len(crops))
                return [
                    {"rec_text": "低置信度批次文字", "rec_score": 0.42}
                    for _ in crops
                ]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "recognition-checkpoint.png"
            checkpoint = base / "adaptive-checkpoint.json"
            image = np.random.default_rng(20260730).integers(
                0,
                256,
                size=(1_200, 2_200, 3),
                dtype=np.uint8,
            )
            Image.fromarray(image).save(source)
            first_engine = OcrEngine(
                det_limit_side_len=960,
                microbatch_max_requests=2,
                microbatch_wait_ms=0,
            )
            first_engine._detector = FakeDetector()
            first_engine._recognizer = CrashingRecognizer()

            with self.assertRaisesRegex(
                RuntimeError,
                "crash after first tile batch",
            ):
                first_engine.recognize_adaptive(
                    source,
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(checkpoint.is_file())

            resumed_engine = OcrEngine(
                det_limit_side_len=960,
                microbatch_max_requests=2,
                microbatch_wait_ms=0,
            )
            resumed_engine._detector = FakeDetector()
            resumed_engine._recognizer = ResumedRecognizer()
            result = resumed_engine.recognize_adaptive(
                source,
                checkpoint_path=checkpoint,
            )

        self.assertGreaterEqual(
            result.extra["checkpoint_recognition_batches_reused"],
            1,
        )
        self.assertLess(
            sum(resumed_calls),
            result.extra["tile_regions_recognized"] + 1,
        )

    def test_adaptive_tile_plan_prunes_only_blank_tiles_away_from_detection(self) -> None:
        import numpy as np

        image = np.full((2400, 3600, 3), 255, dtype=np.uint8)
        anchors = [[[20.0, 20.0], [420.0, 20.0], [420.0, 100.0], [20.0, 100.0]]]

        planned = _tile_plan(3600, 2400, 1280, 160)
        selected = _adaptive_tile_plan(
            image,
            1280,
            160,
            anchors=anchors,
        )

        self.assertGreaterEqual(len(selected), 2)
        self.assertLess(len(selected), len(planned))

    def test_geometric_prefilter_keeps_nested_distinct_text_regions(self) -> None:
        large = [[0.0, 0.0], [200.0, 0.0], [200.0, 80.0], [0.0, 80.0]]
        nested = [[10.0, 10.0], [90.0, 10.0], [90.0, 40.0], [10.0, 40.0]]
        near_duplicate = [[2.0, 1.0], [202.0, 1.0], [202.0, 81.0], [2.0, 81.0]]

        self.assertFalse(_polygon_covered_by_any(nested, [large]))
        self.assertTrue(_polygon_covered_by_any(near_duplicate, [large]))
        self.assertEqual(len(_dedupe_polygons([large, nested])), 2)
        self.assertEqual(len(_dedupe_polygons([large, near_duplicate])), 1)


if __name__ == "__main__":
    unittest.main()
