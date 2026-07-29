from __future__ import annotations

import unittest

from local_full_text_search.core.normalizer import normalize_text, parse_terms
from local_full_text_search.ui.result_view import highlight_context


class NormalizerTests(unittest.TestCase):
    def test_nfkc_case_and_spaces(self) -> None:
        self.assertEqual(normalize_text(" ＢＳ-２８００Ｍ２ \n Test "), "bs-2800m2 test")

    def test_ignore_spaces_and_hyphens(self) -> None:
        value = normalize_text("BS-2800 M2", ignore_spaces=True, ignore_hyphens=True)
        self.assertEqual(value, "bs2800m2")

    def test_parse_terms(self) -> None:
        self.assertEqual(parse_terms("生化 校准 吸光度", "all"), ["生化", "校准", "吸光度"])
        self.assertEqual(parse_terms('"反应杯携带污染率"', "phrase"), ["反应杯携带污染率"])

    def test_highlight_context_preserves_lines_and_marks_all_terms(self) -> None:
        rendered = highlight_context("拔掉3 个\n传感器", "拔掉 3 个传感器")
        self.assertEqual(rendered.count("background:#FDE68A"), 3)
        self.assertIn("<br>", rendered)


if __name__ == "__main__":
    unittest.main()
