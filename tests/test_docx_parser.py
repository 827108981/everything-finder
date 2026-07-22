from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.docx_parser import DocxParser


class DocxParserTests(unittest.TestCase):
    def test_docx_dependency_or_parse(self) -> None:
        try:
            from docx import Document
        except ImportError:
            self.skipTest("python-docx not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.docx"
            doc = Document()
            doc.add_paragraph("DOCX 正文命中")
            doc.save(path)
            blocks = list(DocxParser().parse(path, CancelToken()))
            self.assertTrue(any("DOCX 正文命中" in block.raw_text for block in blocks))


if __name__ == "__main__":
    unittest.main()
