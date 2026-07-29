from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.parsers.legacy_office_parser import OfficeConversionSession


class LegacyOfficeParserTests(unittest.TestCase):
    def test_invalid_legacy_document_blocks_complete_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "old.doc").write_bytes(b"legacy placeholder")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.failed, 1)
            diagnostics = db.failed_files()
            self.assertEqual(diagnostics[0]["parse_status"], "failed")
            self.assertEqual(diagnostics[0]["parse_error_code"], "LEGACY_INVALID_FORMAT")
            self.assertFalse(db.index_readiness()["ready"])

    def test_wps_is_used_when_microsoft_word_is_unavailable(self) -> None:
        class FakeDocument:
            def SaveAs2(self, target: str, FileFormat: int) -> None:
                self.target = target
                self.file_format = FileFormat
                Path(target).write_bytes(b"converted")

            def Close(self, _save: bool) -> None:
                return

        class FakeDocuments:
            def Open(self, _source: str, ReadOnly: bool) -> FakeDocument:
                self.read_only = ReadOnly
                return FakeDocument()

        class FakeApp:
            Documents = FakeDocuments()

            def Quit(self) -> None:
                return

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source = base / "old.doc"
            target = base / "old.docx"
            source.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
            session = OfficeConversionSession()
            attempts: list[str] = []

            def application(_kind: str, prog_id: str) -> object:
                attempts.append(prog_id)
                if prog_id == "Word.Application":
                    raise RuntimeError("class not registered")
                return FakeApp()

            with patch.object(session, "_application", side_effect=application):
                result = session.convert(source, target)

            self.assertEqual(attempts, ["Word.Application", "KWPS.Application"])
            self.assertEqual(result.path, target)
            self.assertTrue(target.is_file())


if __name__ == "__main__":
    unittest.main()
