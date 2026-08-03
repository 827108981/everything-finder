from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from local_full_text_search.core.errors import PauseRequestedError
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.ocr.ocr_cache import OcrCache
from local_full_text_search.ocr.ocr_engine import OcrResult
from local_full_text_search.parsers.image_parser import ImageParser
from local_full_text_search.parsers.pdf_parser import (
    PdfParser,
    _pdf_region_fallback_reason,
    _pdf_preview_fallback_reason,
)


class PdfParserTests(unittest.TestCase):
    def test_scheduled_pages_reuse_document_until_source_identity_changes(
        self,
    ) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scheduled.pdf"
            document = fitz.open()
            document.new_page().insert_text((72, 72), "PAGE_ONE")
            document.new_page().insert_text((72, 72), "PAGE_TWO")
            document.save(path)
            document.close()
            parser = PdfParser(enable_scanned_ocr=False)
            original_open = fitz.open
            try:
                with patch("fitz.open", wraps=original_open) as open_pdf:
                    parser.configure_runtime(content_digest="sha256:before")
                    first = list(
                        parser.parse_scheduled_page(
                            path,
                            1,
                            "pdf_native_page",
                            CancelToken(),
                        )
                    )
                    second = list(
                        parser.parse_scheduled_page(
                            path,
                            2,
                            "pdf_native_page",
                            CancelToken(),
                        )
                    )
                    self.assertEqual(open_pdf.call_count, 1)
                    self.assertIn("PAGE_ONE", first[0].raw_text)
                    self.assertIn("PAGE_TWO", second[0].raw_text)

                    parser.configure_runtime(content_digest="sha256:after")
                    list(
                        parser.parse_scheduled_page(
                            path,
                            1,
                            "pdf_native_page",
                            CancelToken(),
                        )
                    )
                    self.assertEqual(open_pdf.call_count, 2)
            finally:
                close = getattr(parser, "close", None)
                if callable(close):
                    close()

    def test_u0_03r_pdf_embedded_ocr_does_not_retry_after_pause(
        self,
    ) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        class PausingOcr:
            language = "ch"
            det_limit_side_len = 960

            def __init__(self) -> None:
                self.calls = 0

            def recognize_adaptive(
                self,
                _image_path: Path,
                **_kwargs: object,
            ) -> OcrResult:
                self.calls += 1
                raise PauseRequestedError("pause")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "scan.png"
            Image.new("RGB", (800, 600), "white").save(image_path)
            pdf_path = base / "scan.pdf"
            document = fitz.open()
            page = document.new_page(width=800, height=600)
            page.insert_image(page.rect, filename=str(image_path))
            document.save(pdf_path)
            document.close()
            ocr = PausingOcr()
            parser = PdfParser(
                enable_scanned_ocr=True,
                ocr_engine=ocr,
            )
            parser.cache = OcrCache(base / "cache")

            with self.assertRaises(PauseRequestedError):
                list(
                    parser.parse_scheduled_page(
                        pdf_path,
                        1,
                        "pdf_ocr_page",
                        CancelToken(),
                    )
                )

        self.assertEqual(ocr.calls, 1)

    def test_pdf_full_page_embedded_image_reuses_directory_image_ocr(
        self,
    ) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        class FakeOcr:
            language = "ch"
            det_limit_side_len = 960

            def __init__(self) -> None:
                self.calls: list[Path] = []

            def recognize_adaptive(
                self,
                image_path: Path,
                **_kwargs: object,
            ) -> OcrResult:
                self.calls.append(image_path)
                return OcrResult(
                    "PDF_EMBEDDED_EXACT_REUSE",
                    0.96,
                    {
                        "boxes": [
                            [
                                [10.0, 10.0],
                                [200.0, 10.0],
                                [200.0, 50.0],
                                [10.0, 50.0],
                            ]
                        ]
                    },
                )

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "same.jpg"
            Image.new("RGB", (640, 480), "white").save(
                image_path,
                quality=95,
            )
            pdf_path = base / "embedded.pdf"
            document = fitz.open()
            page = document.new_page(width=640, height=480)
            page.insert_image(page.rect, filename=str(image_path))
            document.save(pdf_path)
            document.close()
            cache = OcrCache(base / "ocr-cache")
            fake = FakeOcr()
            image_parser = ImageParser(
                enabled=True,
                min_pixels=0,
                ocr_engine=fake,
            )
            image_parser.cache = cache
            image_blocks = list(
                image_parser.parse(image_path, CancelToken())
            )
            self.assertEqual(len(image_blocks), 1)
            self.assertEqual(len(fake.calls), 1)

            pdf_parser = PdfParser(
                enable_scanned_ocr=True,
                ocr_engine=fake,
            )
            pdf_parser.cache = cache
            pdf_blocks = list(
                pdf_parser.parse_scheduled_page(
                    pdf_path,
                    1,
                    "pdf_ocr_page",
                    CancelToken(),
                )
            )

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(
            pdf_blocks[0].raw_text,
            "PDF_EMBEDDED_EXACT_REUSE",
        )
        self.assertEqual(pdf_blocks[0].page_number, 1)
        self.assertEqual(
            pdf_blocks[0].extra["ocr_embedded_image_cache_hits"],
            1,
        )
        self.assertEqual(
            pdf_blocks[0].extra["embedded_image_source"],
            "full_page_exact",
        )

    def test_native_pdf_text_is_extracted_with_page_number(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "PDF_NATIVE_HIT")
            document.save(path)
            document.close()

            blocks = list(PdfParser(enable_scanned_ocr=False).parse(path, CancelToken()))

            self.assertEqual(len(blocks), 1)
            self.assertEqual(blocks[0].page_number, 1)
            self.assertIn("PDF_NATIVE_HIT", blocks[0].raw_text)

    def test_native_pdf_can_extract_page_ranges_in_parallel(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "parallel.pdf"
            document = fitz.open()
            for index in range(6):
                page = document.new_page()
                page.insert_text((72, 72), f"PDF_PARALLEL_PAGE_{index + 1}")
            document.save(path)
            document.close()

            blocks = list(
                PdfParser(
                    enable_scanned_ocr=False,
                    parallel_min_bytes=0,
                    parallel_min_pages=1,
                    parallel_workers=2,
                ).parse(path, CancelToken())
            )

            self.assertEqual([block.page_number for block in blocks], list(range(1, 7)))
            self.assertTrue(all(block.extra.get("parallel") for block in blocks))

    def test_mixed_pdf_extracts_all_native_pages_before_ocr_candidates(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        class FakeOcr:
            def __init__(self) -> None:
                self.calls: list[Path] = []

            def recognize_adaptive(self, image_path: Path, **_kwargs: object) -> OcrResult:
                self.calls.append(image_path)
                return OcrResult("PDF_SCANNED_PAGE_OCR", 0.95, {})

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "scan.png"
            Image.new("RGB", (400, 240), "white").save(image_path)
            path = base / "mixed.pdf"
            document = fitz.open()
            native = document.new_page()
            native.insert_text((72, 72), "PDF_NATIVE_TEXT_PAGE_WITH_ENOUGH_CHARACTERS")
            scanned = document.new_page()
            scanned.insert_image(scanned.rect, filename=str(image_path))
            document.save(path)
            document.close()
            fake_ocr = FakeOcr()
            parser = PdfParser(
                enable_scanned_ocr=True,
                parallel_min_bytes=0,
                parallel_min_pages=1,
                parallel_workers=2,
                ocr_engine=fake_ocr,
            )
            parser.cache = OcrCache(base / "ocr-cache")
            events: list[str] = []
            parser.configure_runtime(
                content_digest="sha256:pdf",
                progress_callback=lambda payload: events.append(str(payload["phase"])),
            )

            blocks = list(parser.parse(path, CancelToken()))

            self.assertEqual(len(fake_ocr.calls), 1)
            self.assertEqual([block.page_number for block in blocks], [1, 2, 2])
            self.assertEqual([block.source_type for block in blocks], ["native_text", "native_text", "ocr"])
            first_ocr = next(index for index, phase in enumerate(events) if phase.startswith("pdf_ocr_"))
            last_native = max(index for index, phase in enumerate(events) if phase == "pdf_native_page")
            self.assertGreater(first_ocr, last_native)

    def test_dynamic_pdf_ocr_upgrades_only_low_quality_regions_to_300_dpi(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        class FakeDynamicOcr:
            language = "ch"

            def __init__(self) -> None:
                self.phases: list[str] = []

            def detect_file_regions(
                self,
                image_path: Path,
            ) -> tuple[list[list[list[float]]], tuple[int, int]]:
                with Image.open(image_path) as preview:
                    width, height = preview.size
                return (
                    [
                        [
                            [20.0, 20.0],
                            [min(width - 1.0, 500.0), 20.0],
                            [min(width - 1.0, 500.0), 120.0],
                            [20.0, 120.0],
                        ]
                    ],
                    (width, height),
                )

            def recognize_crops(
                self,
                crops: list[object],
                *,
                phase: str,
                **_kwargs: object,
            ) -> list[tuple[str, float | None]]:
                self.phases.append(phase)
                if phase == "pdf_region_300dpi":
                    return [("PDF_DYNAMIC_UPGRADED_TEXT", 0.96) for _ in crops]
                return [("PDF_DYNAMIC_LOW_TEXT", 0.55) for _ in crops]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "scan.png"
            Image.new("RGB", (1200, 800), "white").save(image_path)
            path = base / "dynamic.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=400)
            page.insert_image(page.rect, filename=str(image_path))
            document.save(path)
            document.close()
            fake_ocr = FakeDynamicOcr()
            parser = PdfParser(
                enable_scanned_ocr=True,
                ocr_engine=fake_ocr,
            )
            parser.cache = OcrCache(base / "ocr-cache")
            parser.configure_runtime(content_digest="sha256:dynamic")

            blocks = list(parser.parse(path, CancelToken()))

            ocr_block = next(block for block in blocks if block.source_type == "ocr")
            self.assertIn("PDF_DYNAMIC_UPGRADED_TEXT", ocr_block.raw_text)
            self.assertEqual(
                fake_ocr.phases,
                ["pdf_region_200dpi", "pdf_region_300dpi"],
            )
            self.assertTrue(ocr_block.extra["pdf_dynamic_dpi"])
            self.assertEqual(ocr_block.extra["pdf_upgraded_regions"], 1)
            self.assertFalse(ocr_block.extra["pdf_full_page_fallback"])
            self.assertGreater(ocr_block.extra["pdf_preview_pixels"], 0)
            self.assertGreater(ocr_block.extra["pdf_region_200dpi_pixels"], 0)
            self.assertGreater(ocr_block.extra["pdf_region_300dpi_pixels"], 0)
            for metric in (
                "pdf_preview_render_ms",
                "pdf_preview_detect_ms",
                "pdf_region_200dpi_render_ms",
                "pdf_region_200dpi_recognize_ms",
                "pdf_region_300dpi_render_ms",
                "pdf_region_300dpi_recognize_ms",
            ):
                self.assertIsInstance(ocr_block.extra[metric], int)

    def test_identical_pdf_region_pixels_are_reused_across_different_pdfs(self) -> None:
        try:
            import fitz
        except ImportError:
            self.skipTest("PyMuPDF not installed")

        class CountingDynamicOcr:
            language = "ch"

            def __init__(self) -> None:
                self.recognized_crop_count = 0

            def detect_file_regions(
                self,
                image_path: Path,
            ) -> tuple[list[list[list[float]]], tuple[int, int]]:
                with Image.open(image_path) as preview:
                    width, height = preview.size
                return (
                    [
                        [
                            [40.0, 40.0],
                            [min(width - 1.0, 700.0), 40.0],
                            [min(width - 1.0, 700.0), 180.0],
                            [40.0, 180.0],
                        ]
                    ],
                    (width, height),
                )

            def recognize_crops(
                self,
                crops: list[object],
                **_kwargs: object,
            ) -> list[tuple[str, float | None]]:
                self.recognized_crop_count += len(crops)
                return [("CROSS_PDF_EXACT_REGION", 0.96) for _ in crops]

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            image_path = base / "same-scan.png"
            Image.new("RGB", (1200, 800), "white").save(image_path)
            paths = [base / "first.pdf", base / "second.pdf"]
            for index, path in enumerate(paths):
                document = fitz.open()
                document.set_metadata({"title": f"different-container-{index}"})
                page = document.new_page(width=600, height=400)
                page.insert_image(page.rect, filename=str(image_path))
                document.save(path)
                document.close()

            engine = CountingDynamicOcr()
            parser = PdfParser(enable_scanned_ocr=True, ocr_engine=engine)
            parser.cache = OcrCache(base / "ocr-cache")
            parser.configure_runtime(content_digest="sha256:first-container")
            first = list(parser.parse(paths[0], CancelToken()))
            first_count = engine.recognized_crop_count

            parser.configure_runtime(content_digest="sha256:second-container")
            second = list(parser.parse(paths[1], CancelToken()))

            self.assertGreater(first_count, 0)
            self.assertEqual(engine.recognized_crop_count, first_count)
            self.assertEqual(
                [block.raw_text for block in first if block.source_type == "ocr"],
                [block.raw_text for block in second if block.source_type == "ocr"],
            )
            second_ocr = next(block for block in second if block.source_type == "ocr")
            self.assertGreater(second_ocr.extra["ocr_exact_cache_hits"], 0)

    def test_large_pdf_preview_with_sparse_regions_requires_full_page_fallback(self) -> None:
        polygons = [
            [[10.0, 10.0], [400.0, 10.0], [400.0, 80.0], [10.0, 80.0]],
            [[20.0, 200.0], [500.0, 200.0], [500.0, 260.0], [20.0, 260.0]],
            [[30.0, 400.0], [600.0, 400.0], [600.0, 470.0], [30.0, 470.0]],
        ]

        self.assertEqual(
            _pdf_region_fallback_reason(
                polygons,
                (2400, 1600),
                [0.92, 0.90, 0.88],
                80,
            ),
            "sparse_regions_on_large_page",
        )
        self.assertEqual(
            _pdf_preview_fallback_reason(
                polygons,
                (2400, 1600),
            ),
            "sparse_regions_on_large_page",
        )
        self.assertEqual(
            _pdf_region_fallback_reason(
                polygons * 3,
                (2400, 1600),
                [0.92] * 9,
                80,
            ),
            "",
        )


if __name__ == "__main__":
    unittest.main()
