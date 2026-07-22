from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.text_parser import TextParser, detect_text_encoding


class TextParserTests(unittest.TestCase):
    def test_utf8_text_chunks_keep_line_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "日志.log"
            path.write_text("\n".join(f"line {i}" for i in range(1, 4)), encoding="utf-8")
            parser = TextParser(lines_per_block=2)
            blocks = list(parser.parse(path, CancelToken()))
            self.assertEqual(len(blocks), 2)
            self.assertEqual(blocks[0].location_text, "第 1-2 行")
            self.assertEqual(blocks[1].line_start, 3)

    def test_gb18030_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gbk.txt"
            path.write_bytes("中文内容".encode("gb18030"))
            self.assertEqual(detect_text_encoding(path), "gb18030")


if __name__ == "__main__":
    unittest.main()
