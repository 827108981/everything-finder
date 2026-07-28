from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from docx import Document
from openpyxl import Workbook
from openpyxl.comments import Comment
from pptx import Presentation

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.ooxml.docx_stream_parser import DocxStreamParser
from local_full_text_search.parsers.ooxml.pptx_stream_parser import PptxStreamParser
from local_full_text_search.parsers.ooxml.xlsx_stream_parser import XlsxStreamParser
from local_full_text_search.parsers.parser_registry import ParserRegistry


class StreamingOoxmlTests(unittest.TestCase):
    def test_docx_streams_body_table_header_and_footer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.docx"
            document = Document()
            document.add_paragraph("DOCX_STREAM_BODY")
            table = document.add_table(rows=1, cols=2)
            table.cell(0, 0).text = "DOCX_TABLE_A"
            table.cell(0, 1).text = "DOCX_TABLE_B"
            document.sections[0].header.paragraphs[0].text = "DOCX_HEADER"
            document.sections[0].footer.paragraphs[0].text = "DOCX_FOOTER"
            document.save(path)

            blocks = list(DocxStreamParser().parse(path, CancelToken()))
            text = "\n".join(block.raw_text for block in blocks)

            self.assertIn("DOCX_STREAM_BODY", text)
            self.assertIn("DOCX_TABLE_A", text)
            self.assertIn("DOCX_TABLE_B", text)
            self.assertIn("DOCX_HEADER", text)
            self.assertIn("DOCX_FOOTER", text)

    def test_pptx_streams_slide_and_existing_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.pptx"
            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[6])
            box = slide.shapes.add_textbox(914400, 914400, 3657600, 914400)
            box.text = "PPTX_STREAM_SLIDE"
            slide.notes_slide.notes_text_frame.text = "PPTX_STREAM_NOTE"
            presentation.save(path)

            blocks = list(PptxStreamParser().parse(path, CancelToken()))
            text = "\n".join(block.raw_text for block in blocks)

            self.assertIn("PPTX_STREAM_SLIDE", text)
            self.assertIn("PPTX_STREAM_NOTE", text)
            self.assertTrue(all(block.slide_number == 1 for block in blocks))

    def test_xlsx_streams_sparse_shared_formula_and_comment_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Data"
            sheet["D5"] = "XLSX_STREAM_VALUE"
            sheet["D5"].comment = Comment("XLSX_STREAM_COMMENT", "tester")
            sheet["E5"] = "=1+2"
            workbook.save(path)

            blocks = list(XlsxStreamParser().parse(path, CancelToken()))
            text = "\n".join(block.raw_text for block in blocks)

            self.assertIn("D5=XLSX_STREAM_VALUE", text)
            self.assertIn("XLSX_STREAM_COMMENT", text)
            self.assertIn("E5==1+2", text)
            self.assertEqual(blocks[0].sheet_name, "Data")

    def test_registry_selects_streaming_parsers_by_default(self) -> None:
        registry = ParserRegistry(AppSettings(fast_ooxml_enabled=True))

        self.assertIsInstance(registry.parser_for(Path("a.docx")), DocxStreamParser)
        self.assertIsInstance(registry.parser_for(Path("a.pptx")), PptxStreamParser)
        self.assertIsInstance(registry.parser_for(Path("a.xlsx")), XlsxStreamParser)


if __name__ == "__main__":
    unittest.main()
