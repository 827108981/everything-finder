from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.ocr.ocr_cache import OcrCache


class OcrCacheTests(unittest.TestCase):
    def test_namespace_changes_cache_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image = base / "image.bin"
            image.write_bytes(b"same image bytes")
            cache = OcrCache(base / "cache")

            first = cache.key_for_file(image, namespace="model-a:ch:2400")
            second = cache.key_for_file(image, namespace="model-b:ch:2400")

            self.assertNotEqual(first, second)


if __name__ == "__main__":
    unittest.main()
