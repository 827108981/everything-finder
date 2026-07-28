from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.pdf_parser import PdfParser


class PdfParserTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
