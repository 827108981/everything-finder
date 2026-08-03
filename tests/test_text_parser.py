from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.index_manager import ParseJob, parse_file_with_registry
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.parsers.parser_registry import ParserRegistry
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

    def test_bomless_utf16le_xml_uses_byte_pattern_and_parses_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "field-export.xml"
            xml = (
                '<?xml version="1.0" encoding="UTF-16"?>\n'
                "<root><field>现场传感器记录</field></root>"
            )
            path.write_bytes(xml.encode("utf-16-le"))

            encoding = detect_text_encoding(path)
            blocks = list(TextParser().parse(path, CancelToken()))

            self.assertEqual(encoding, "utf-16-le")
            self.assertIn("现场传感器记录", blocks[0].raw_text)

    def test_parse_job_consumes_prepared_spool_instead_of_changed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "source.txt"
            source.write_text("CHANGED_SOURCE", encoding="utf-8")
            spool = base / "prepared.txt"
            spool.write_text("PREPARED_SPOOL_TEXT", encoding="utf-8")
            settings = AppSettings(enable_ocr=False)
            job = ParseJob(
                file_id=1,
                file_path=source,
                source_spool_path=spool,
                content_key="sha256:test",
                parser_name="text",
                parser_version="2",
            )

            outcome = parse_file_with_registry(
                job,
                ParserRegistry(settings),
                CancelToken(),
                settings,
            )

            self.assertEqual(outcome.file_path, source)
            self.assertIn("PREPARED_SPOOL_TEXT", outcome.blocks[0].raw_text)
            self.assertNotIn("CHANGED_SOURCE", outcome.blocks[0].raw_text)


if __name__ == "__main__":
    unittest.main()
