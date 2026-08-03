from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.ocr.ocr_cache import (
    OcrCache,
    OcrExactInput,
    ocr_models_fingerprint,
)
from local_full_text_search.ocr.ocr_engine import OcrResult


class OcrCacheTests(unittest.TestCase):
    def test_exact_key_reuses_identical_pixels_across_sources(self) -> None:
        cache_input = OcrExactInput(
            content_sha256="a" * 64,
            width=640,
            height=480,
            channels=3,
            orientation=0,
            crop=(10, 20, 300, 180),
            dpi=200,
            preprocess_version="prep-2",
            strategy_version="strategy-3",
            detection_model_fingerprint="det-a",
            recognition_model_fingerprint="rec-a",
            language="ch",
            options={"deskew": True, "distortion": False},
        )
        with tempfile.TemporaryDirectory() as tmp:
            cache = OcrCache(Path(tmp))

            directory_key = cache.key_for_exact_input(
                cache_input,
                source_hint=r"E:\directory\image.png",
            )
            zip_key = cache.key_for_exact_input(
                cache_input,
                source_hint=r"E:\archive.zip > image.png",
            )

        self.assertEqual(directory_key, zip_key)

    def test_exact_key_invalidates_every_semantic_configuration_dimension(self) -> None:
        base = OcrExactInput(
            content_sha256="b" * 64,
            width=640,
            height=480,
            channels=3,
            orientation=0,
            crop=(10, 20, 300, 180),
            dpi=200,
            preprocess_version="prep-2",
            strategy_version="strategy-3",
            detection_model_fingerprint="det-a",
            recognition_model_fingerprint="rec-a",
            language="ch",
            options={"deskew": True},
        )
        variants = [
            base.with_changes(width=641),
            base.with_changes(height=481),
            base.with_changes(channels=4),
            base.with_changes(orientation=90),
            base.with_changes(crop=(11, 20, 300, 180)),
            base.with_changes(dpi=300),
            base.with_changes(preprocess_version="prep-3"),
            base.with_changes(strategy_version="strategy-4"),
            base.with_changes(detection_model_fingerprint="det-b"),
            base.with_changes(recognition_model_fingerprint="rec-b"),
            base.with_changes(language="en"),
            base.with_changes(options={"deskew": False}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            cache = OcrCache(Path(tmp))
            original = cache.key_for_exact_input(base)
            keys = {cache.key_for_exact_input(value) for value in variants}

        self.assertEqual(len(keys), len(variants))
        self.assertNotIn(original, keys)

    def test_corrupt_or_tampered_cache_is_rejected_and_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = OcrCache(cache_dir)
            key = "exact-cache-key"
            cache.save(key, OcrResult("expected", 0.91, {"boxes": []}))
            cache_path = cache_dir / f"{key}.json"
            payload = cache_path.read_text(encoding="utf-8")
            cache_path.write_text(payload.replace("expected", "tampered"), encoding="utf-8")

            result, status = cache.load_with_status(key)

            self.assertIsNone(result)
            self.assertEqual(status, "checksum_mismatch")
            self.assertFalse(cache_path.exists())

    def test_atomic_save_leaves_no_temporary_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = OcrCache(cache_dir)
            cache.save("atomic", OcrResult("正文", 0.88, {"source": "test"}))

            self.assertIsNotNone(cache.load("atomic"))
            self.assertEqual(list(cache_dir.glob("*.tmp")), [])

    def test_active_cache_reference_blocks_pruning_until_released(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache = OcrCache(cache_dir)
            cache.save("leased", OcrResult("正文", 0.88, {}))

            with cache.reference("leased"):
                first = cache.prune(max_entries=0)
                self.assertEqual(first["removed_count"], 0)
                self.assertEqual(first["active_reference_skips"], 1)
                self.assertIsNotNone(cache.load("leased"))

            second = cache.prune(max_entries=0)
            self.assertEqual(second["removed_keys"], ["leased"])
            self.assertIsNone(cache.load("leased"))

    def test_active_reference_path_stays_below_legacy_windows_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            while len(str(cache_dir / ".active_references")) < 180:
                cache_dir /= "nested-cache-segment"
            cache = OcrCache(cache_dir)

            with cache.reference("k" * 128):
                references = list(cache.reference_dir.glob("*.json"))
                self.assertEqual(len(references), 1)
                self.assertLess(len(str(references[0])), 260)

    def test_namespace_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "image.bin"
            image.write_bytes(b"same image bytes")
            cache = OcrCache(base / "cache")

            first = cache.key_for_file(image, namespace="model-a:ch:2400")
            second = cache.key_for_file(image, namespace="model-b:ch:2400")

            self.assertNotEqual(first, second)

    def test_precomputed_digest_avoids_reopening_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "image.bin"
            image.write_bytes(b"same image bytes")
            cache = OcrCache(base / "cache")

            with patch.object(Path, "open", side_effect=AssertionError("source reopened")):
                key = cache.key_for_digest("sha256:abc", namespace="model-a")

            self.assertTrue(key)

    def test_model_fingerprint_uses_manifest_without_reading_model_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp)
            model_file = models / "PP-OCRv4_mobile_det" / "inference.pdiparams"
            model_file.parent.mkdir()
            model_file.write_bytes(b"weight-bytes")
            (models / "manifest.json").write_text(
                """
                {
                  "manifest_version": 1,
                  "combined_digest": "manifest-digest",
                  "files": [
                    {
                      "path": "PP-OCRv4_mobile_det/inference.pdiparams",
                      "size": 12,
                      "sha256": "unused-at-runtime"
                    }
                  ]
                }
                """,
                encoding="utf-8",
            )
            ocr_models_fingerprint.cache_clear()
            with patch(
                "local_full_text_search.ocr.ocr_cache.OCR_MODELS_DIR",
                models,
            ), patch.object(Path, "read_bytes", side_effect=AssertionError("weight read")):
                fingerprint = ocr_models_fingerprint()
            ocr_models_fingerprint.cache_clear()

            self.assertEqual(fingerprint, "manifest-digest")


if __name__ == "__main__":
    unittest.main()
