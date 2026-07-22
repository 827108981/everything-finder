from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager


class LegacyOfficeParserTests(unittest.TestCase):
    def test_converter_missing_is_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "old.doc").write_bytes(b"legacy placeholder")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            self.assertEqual(summary.failed, 0)
            diagnostics = db.failed_files()
            if diagnostics:
                self.assertIn(diagnostics[0]["parse_status"], {"converter_missing", "partial_success"})


if __name__ == "__main__":
    unittest.main()
