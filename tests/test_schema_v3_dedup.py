from __future__ import annotations

import shutil
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import IndexNotReadyError
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery


class SchemaV3DedupTests(unittest.TestCase):
    def test_schema_v4_discards_legacy_short_tokens_without_forcing_reparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "legacy-tokens.txt"
            source.write_text("SCHEMA_V4_MIGRATION_HIT", encoding="utf-8")
            db_path = base / "index.db"
            db = DatabaseManager(db_path)
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            token_count = 50_000
            with db.connect() as con:
                block_id = int(con.execute("SELECT id FROM content_blocks LIMIT 1").fetchone()[0])
                con.execute("DROP INDEX idx_short_tokens_block")
                con.executemany(
                    "INSERT INTO short_tokens(token, block_id) VALUES (?, ?)",
                    ((f"legacy-{index}", block_id) for index in range(token_count)),
                )
                con.execute("PRAGMA user_version = 3")

            started = time.perf_counter()
            migrated = DatabaseManager(db_path)
            migrated.initialize()
            elapsed = time.perf_counter() - started

            backup = base / "index.schema-v3.backup.db"
            self.assertTrue(backup.is_file())
            backup_con = sqlite3.connect(backup)
            try:
                self.assertEqual(
                    backup_con.execute("SELECT COUNT(*) FROM short_tokens").fetchone()[0],
                    token_count,
                )
            finally:
                backup_con.close()
            with migrated.connect() as con:
                file_row = con.execute(
                    "SELECT parse_status, parser_version FROM files WHERE path = ?",
                    (str(source),),
                ).fetchone()
                indexes = {
                    str(row["name"]) for row in con.execute("PRAGMA index_list(short_tokens)")
                }
                plan = " ".join(
                    str(row["detail"])
                    for row in con.execute(
                        "EXPLAIN QUERY PLAN DELETE FROM short_tokens WHERE block_id = ?",
                        (block_id,),
                    )
                )
                self.assertEqual(con.execute("PRAGMA user_version").fetchone()[0], 5)
                self.assertEqual(con.execute("SELECT COUNT(*) FROM short_tokens").fetchone()[0], 0)
            self.assertEqual(file_row["parse_status"], "success")
            self.assertTrue(file_row["parser_version"])
            self.assertIn("idx_short_tokens_block", indexes)
            self.assertIn("idx_short_tokens_block", plan)
            self.assertFalse(migrated.requires_full_rebuild())
            self.assertLess(elapsed, 5.0)

    def test_legacy_database_is_backed_up_and_rebuilt_without_duplicate_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "legacy.txt"
            source.write_text("LEGACY_SCHEMA_REBUILD_HIT", encoding="utf-8")
            db_path = base / "index.db"
            db = DatabaseManager(db_path)
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            with db.connect() as con:
                block = con.execute(
                    "SELECT id, file_id FROM content_blocks ORDER BY id LIMIT 1"
                ).fetchone()
                con.execute("UPDATE content_blocks SET document_id = NULL WHERE id = ?", (block["id"],))
                con.execute("UPDATE files SET document_id = NULL WHERE id = ?", (block["file_id"],))
                con.execute("DELETE FROM documents")
                con.execute("DELETE FROM index_state WHERE key = 'schema_v3_rebuild_required'")
                con.execute("PRAGMA user_version = 0")

            migrated = DatabaseManager(db_path)
            migrated.initialize()
            backup = base / "index.schema-v0.backup.db"
            self.assertTrue(backup.is_file())
            con = sqlite3.connect(backup)
            try:
                self.assertEqual(con.execute("PRAGMA integrity_check").fetchone()[0], "ok")
            finally:
                con.close()
            self.assertTrue(migrated.requires_full_rebuild())

            summary = IndexManager(migrated, AppSettings(enable_ocr=False)).index_root(root_id)
            self.assertEqual(summary.indexed, 1)
            self.assertFalse(migrated.requires_full_rebuild())
            self.assertEqual(migrated.stats()["blocks"], 1)
            self.assertEqual(
                SearchEngine(migrated).search(SearchQuery(text="LEGACY_SCHEMA_REBUILD_HIT")).total_confirmed,
                1,
            )

    def test_duplicate_content_is_parsed_once_and_expanded_to_each_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            first = root / "a.txt"
            second = root / "b.txt"
            first.write_text("SCHEMA_V3_SHARED_HIT", encoding="utf-8")
            shutil.copy2(first, second)

            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            manager = IndexManager(db, AppSettings(enable_ocr=False))
            summary = manager.index_root(root_id)

            self.assertEqual(summary.indexed, 2)
            self.assertEqual(db.stats()["files"], 2)
            self.assertEqual(db.stats()["documents"], 1)
            self.assertEqual(db.stats()["blocks"], 1)
            page = SearchEngine(db).search(
                SearchQuery(
                    text="SCHEMA_V3_SHARED_HIT",
                    search_filename=False,
                    search_path=False,
                )
            )
            self.assertEqual(page.total_confirmed, 2)
            self.assertEqual({result.filename for result in page.results}, {"a.txt", "b.txt"})

            first.unlink()
            manager.index_root(root_id)
            remaining = SearchEngine(db).search(SearchQuery(text="SCHEMA_V3_SHARED_HIT"))
            self.assertEqual(remaining.total_confirmed, 1)
            self.assertEqual(db.stats()["documents"], 1)

            second.unlink()
            manager.index_root(root_id)
            self.assertEqual(db.stats()["documents"], 0)
            self.assertEqual(
                SearchEngine(db).search(SearchQuery(text="SCHEMA_V3_SHARED_HIT")).total_confirmed,
                0,
            )
            report = db.integrity_report()
            self.assertEqual(report["integrity"], ["ok"])
            self.assertEqual(report["foreign_key_errors"], [])

    def test_run_metrics_tasks_and_deferred_fts_are_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "one.txt").write_text("METRICS_RUN_HIT", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)

            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            with db.connect() as con:
                run = con.execute("SELECT * FROM index_runs ORDER BY started_at DESC LIMIT 1").fetchone()
                task = con.execute("SELECT * FROM parse_tasks ORDER BY id DESC LIMIT 1").fetchone()
                dirty = con.execute(
                    "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
                ).fetchone()
                metric_count = con.execute("SELECT COUNT(*) FROM index_file_metrics").fetchone()[0]
            self.assertEqual(run["status"], "complete")
            self.assertEqual(run["discovered_files"], 1)
            self.assertGreaterEqual(run["total_ms"], 0)
            self.assertEqual(task["status"], "complete")
            self.assertEqual(dirty["value"], "0")
            self.assertEqual(metric_count, 1)

    def test_incomplete_full_batch_rebuilds_dirty_fts_without_reparsing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "one.txt").write_text("RESUMED_FTS_ONLY_HIT", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(enable_ocr=False)
            IndexManager(db, settings).index_root(root_id)
            with db.connect() as con:
                con.execute("DELETE FROM content_fts")
                con.execute(
                    "INSERT INTO index_state(key, value) VALUES ('content_fts_dirty', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
                con.execute(
                    "INSERT INTO index_state(key, value) VALUES ('full_batch_incomplete', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )

            reopened = DatabaseManager(base / "index.db")
            reopened.initialize()
            with self.assertRaises(IndexNotReadyError):
                SearchEngine(reopened).search(SearchQuery(text="RESUMED_FTS_ONLY_HIT"))
            summary = IndexManager(reopened, settings).index_root(root_id)
            self.assertEqual(summary.indexed, 0)
            self.assertEqual(summary.skipped, 1)
            self.assertFalse(reopened.has_incomplete_full_batch())
            self.assertEqual(
                SearchEngine(reopened).search(SearchQuery(text="RESUMED_FTS_ONLY_HIT")).total_confirmed,
                1,
            )

    def test_deferred_fts_update_does_not_delete_old_rows_per_document(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "one.txt"
            source.write_text("OLD_DEFERRED_FTS_HIT", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(enable_ocr=False)
            IndexManager(db, settings).index_root(root_id)

            with db.connect() as con:
                file_row = con.execute(
                    "SELECT id, content_key, parser_name, parser_version FROM files WHERE path = ?",
                    (str(source),),
                ).fetchone()
                old_fts_rows = int(con.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0])
            assert file_row is not None

            replacement = ContentBlock(
                file_path=str(source),
                block_index=0,
                block_type="text",
                location_text="第 1 行",
                raw_text="NEW_DEFERRED_FTS_HIT",
                normalized_text=normalize_text("NEW_DEFERRED_FTS_HIT"),
            )
            db.begin_deferred_fts()
            db.replace_document_blocks_many(
                [
                    {
                        "file_id": int(file_row["id"]),
                        "file_ids": [int(file_row["id"])],
                        "filename": source.name,
                        "path": str(source),
                        "blocks": [replacement],
                        "parser_name": str(file_row["parser_name"]),
                        "parser_version": str(file_row["parser_version"]),
                        "status": "success",
                        "error_code": None,
                        "error_message": None,
                        "content_key": str(file_row["content_key"]),
                        "task_id": None,
                    }
                ],
                update_fts=False,
            )

            with db.connect() as con:
                self.assertEqual(con.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0], old_fts_rows)
                self.assertEqual(
                    con.execute("SELECT raw_text FROM content_blocks").fetchone()[0],
                    "NEW_DEFERRED_FTS_HIT",
                )
                self.assertEqual(
                    con.execute(
                        "SELECT value FROM index_state WHERE key = 'content_fts_dirty'"
                    ).fetchone()[0],
                    "1",
                )

            db.rebuild_content_fts()
            db.mark_full_batch_complete()
            engine = SearchEngine(db)
            self.assertEqual(engine.search(SearchQuery(text="NEW_DEFERRED_FTS_HIT")).total_confirmed, 1)
            self.assertEqual(engine.search(SearchQuery(text="OLD_DEFERRED_FTS_HIT")).total_confirmed, 0)


if __name__ == "__main__":
    unittest.main()
