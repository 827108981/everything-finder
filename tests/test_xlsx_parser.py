from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.xlsx_parser import XlsxParser


class XlsxParserTests(unittest.TestCase):
    def test_xlsx_dependency_or_parse(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet["A1"] = "Excel 命中"
            workbook.save(path)
            blocks = list(XlsxParser().parse(path, CancelToken()))
            self.assertTrue(any("Excel 命中" in block.raw_text for block in blocks))

    def test_sparse_xlsx_empty_cells_do_not_fail(self) -> None:
        try:
            from openpyxl import Workbook
        except ImportError:
            self.skipTest("openpyxl not installed")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sparse.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet["D5"] = "SPARSE_XLSX_HIT"
            workbook.save(path)
            blocks = list(XlsxParser().parse(path, CancelToken()))
            self.assertTrue(any("D5=SPARSE_XLSX_HIT" in block.raw_text for block in blocks))


if __name__ == "__main__":
    unittest.main()
