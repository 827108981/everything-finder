from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import IndexNotReadyError
from local_full_text_search.core.errors import CancelledError
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.ui.result_view import highlight_context


class SearchEngineTests(unittest.TestCase):
    def test_content_query_uses_separate_indexed_source_joins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            database = DatabaseManager(Path(tmp) / "index.db")
            database.initialize()
            engine = SearchEngine(database)
            query = SearchQuery(
                text="真空",
                mode="exact",
                search_filename=False,
                search_path=False,
                search_content=True,
            )

            sql, params, strategy = engine._content_query(query, ["真空"])

            self.assertEqual(strategy, "like")
            self.assertIn("JOIN files f ON f.document_id = cb.document_id", sql)
            self.assertIn("JOIN files f ON f.id = cb.file_id", sql)
            self.assertNotIn("OR (cb.document_id IS NULL", sql)
            self.assertEqual(params, ["%真空%", 0.6, 0.6])

    def test_search_result_preserves_structured_slide_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "slides.pptx"
            source.write_bytes(b"pptx")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            file_id, _ = db.upsert_file_metadata(root_id, source)
            db.replace_file_blocks(
                file_id,
                source.name,
                str(source),
                [
                    ContentBlock(
                        file_path=str(source),
                        block_index=0,
                        block_type="pptx_slide",
                        location_text="第 46 张幻灯片",
                        raw_text="STRUCTURED_SLIDE_HIT",
                        normalized_text="structured_slide_hit",
                        slide_number=46,
                    )
                ],
                parser_name="pptx",
            )
            db.update_root_scan_time(root_id, "ready")

            page = SearchEngine(db).search(
                SearchQuery(
                    text="STRUCTURED_SLIDE_HIT",
                    mode="exact",
                    search_filename=False,
                    search_path=False,
                    search_content=True,
                )
            )

            self.assertEqual(page.total_confirmed, 1)
            self.assertEqual(page.results[0].slide_number, 46)
            self.assertEqual(page.results[0].matches[0].slide_number, 46)
            self.assertEqual(page.results[0].matches[0].location_text, "第 46 张幻灯片")

    def test_chinese_filename_uses_fts_match_without_content_scan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "设备操作说明书.txt").write_text("普通正文", encoding="utf-8")
            (root / "unrelated.txt").write_text("没有名称命中", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)
            engine = SearchEngine(db)
            query = SearchQuery(
                text="设备操作",
                mode="exact",
                search_filename=True,
                search_path=False,
                search_content=False,
            )

            normalized_terms = ["设备操作"]
            sql, _params, strategy = engine._metadata_query(
                query,
                normalized_terms,
                True,
                False,
            )
            progress: list[dict[str, object]] = []
            page = engine.search(query, progress_callback=progress.append)

            self.assertEqual(strategy, "fts_match")
            self.assertIn("files_fts MATCH", sql)
            self.assertEqual(page.total_candidates, 1)
            self.assertEqual(page.total_confirmed, 1)
            self.assertEqual(page.results[0].filename, "设备操作说明书.txt")
            self.assertIn("searching_metadata", {str(item["stage"]) for item in progress})
            self.assertNotIn("scanning", {str(item["stage"]) for item in progress})

    def test_search_reports_partial_filename_results_before_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "alpha-report.txt").write_text(
                "alpha appears in content",
                encoding="utf-8",
            )
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)
            progress: list[dict[str, object]] = []
            partial: list[object] = []

            final = SearchEngine(db).search(
                SearchQuery(text="alpha", mode="exact"),
                progress_callback=progress.append,
                partial_callback=partial.append,
            )

            self.assertEqual(len(partial), 1)
            self.assertTrue(partial[0].partial)
            self.assertEqual(partial[0].total_confirmed, 1)
            self.assertFalse(final.partial)
            self.assertGreaterEqual(len(final.results[0].matches), 2)
            stages = [str(item["stage"]) for item in progress]
            self.assertLess(stages.index("searching_metadata"), stages.index("searching_content"))

    def test_short_term_progress_discloses_compatibility_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "ab-note.txt").write_text("ab in content", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)
            progress: list[dict[str, object]] = []

            page = SearchEngine(db).search(
                SearchQuery(text="ab", mode="exact"),
                progress_callback=progress.append,
            )

            self.assertEqual(page.total_confirmed, 1)
            self.assertIn(
                "关键词较短，正在使用兼容搜索",
                {str(item.get("slow_reason") or "") for item in progress},
            )

    def test_search_is_blocked_before_first_complete_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            db.add_root(base / "files")

            with self.assertRaises(IndexNotReadyError):
                SearchEngine(db).search(SearchQuery(text="anything"))

    def test_committed_partial_projection_is_searchable_before_scope_complete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            completed = root / "completed.txt"
            pending = root / "pending.txt"
            completed.write_text("PARTIAL_INDEX_HIT", encoding="utf-8")
            pending.write_text("PENDING_MUST_STAY_HIDDEN", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            db.set_partial_search_projection_active(True)
            completed_id, _ = db.upsert_file_metadata(root_id, completed)
            db.upsert_file_metadata(root_id, pending)
            db.replace_file_blocks(
                completed_id,
                completed.name,
                str(completed),
                [
                    ContentBlock(
                        file_path=str(completed),
                        block_index=0,
                        block_type="text",
                        location_text="line 1",
                        raw_text="PARTIAL_INDEX_HIT",
                        normalized_text="partial_index_hit",
                    )
                ],
                parser_name="text",
            )

            readiness = db.index_readiness()
            self.assertFalse(readiness["ready"])
            self.assertTrue(readiness["search_available"])
            self.assertTrue(readiness["search_is_partial"])
            self.assertGreater(int(readiness["search_projection_generation"]), 0)

            page = SearchEngine(db).search(
                SearchQuery(
                    text="PARTIAL_INDEX_HIT",
                    mode="exact",
                    search_filename=False,
                    search_path=False,
                    search_content=True,
                )
            )
            self.assertEqual(page.total_confirmed, 1)

            hidden = SearchEngine(db).search(
                SearchQuery(
                    text="pending",
                    mode="exact",
                    search_filename=True,
                    search_path=False,
                    search_content=False,
                )
            )
            self.assertEqual(hidden.total_confirmed, 0)

    def test_extension_filter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "hit.txt").write_text("needle", encoding="utf-8")
            (root / "hit.md").write_text("needle", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)
            page = SearchEngine(db).search(SearchQuery(text="needle", mode="exact", extensions=[".md"]))
            self.assertEqual(page.total_confirmed, 1)
            self.assertTrue(page.results[0].filename.endswith(".md"))

    def test_filename_and_content_hits_are_grouped_by_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "alpha-report.txt").write_text("alpha appears again in the body", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)

            page = SearchEngine(db).search(SearchQuery(text="alpha", mode="exact"))

            self.assertEqual(page.total_confirmed, 1)
            result = page.results[0]
            self.assertEqual(result.filename, "alpha-report.txt")
            self.assertGreaterEqual(result.hit_count, 2)
            self.assertGreaterEqual(len(result.matches), 2)
            self.assertIn("文件名/路径", result.location_text)
            self.assertIn("body", result.context)

    def test_regex_and_ignore_spaces_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "regex.txt").write_text("device abc123", encoding="utf-8")
            (root / "spaced.txt").write_text("X Y Z", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings()).index_root(root_id)
            engine = SearchEngine(db)

            regex = engine.search(
                SearchQuery(text=r"abc\d+", mode="regex", search_filename=False, search_path=False)
            )
            compact = engine.search(
                SearchQuery(
                    text="XYZ",
                    mode="exact",
                    ignore_spaces=True,
                    search_filename=False,
                    search_path=False,
                )
            )

            self.assertEqual(regex.total_confirmed, 1)
            self.assertEqual(compact.total_confirmed, 1)

    def test_scan_search_can_be_cancelled_from_progress(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(base / "virtual")
            count = 300
            now = datetime.now(timezone.utc).isoformat()
            with db.connect() as con:
                con.executemany(
                    """
                    INSERT INTO files(
                        root_id, path, filename, extension, size_bytes, modified_time,
                        quick_fingerprint, parse_status, parser_version, last_seen_at
                    ) VALUES (?, ?, ?, '.txt', 20, 0, ?, 'success', '2', ?)
                    """,
                    [
                        (
                            root_id,
                            str(base / f"scan-{index:04}.txt"),
                            f"scan-{index:04}.txt",
                            f"20:{index}",
                            now,
                        )
                        for index in range(count)
                    ],
                )
                con.executemany(
                    """
                    INSERT INTO content_blocks(
                        id, file_id, block_index, block_type, location_text,
                        raw_text, normalized_text, source_type, created_at
                    ) VALUES (?, ?, 0, 'text', '正文', ?, ?, 'native_text', ?)
                    """,
                    [
                        (
                            index,
                            index,
                            f"cancellable content {index}",
                            f"cancellable content {index}",
                            now,
                        )
                        for index in range(1, count + 1)
                    ],
                )
                con.executemany(
                    """
                    INSERT INTO content_fts(
                        rowid, block_id, file_id, filename, path,
                        location_text, normalized_text
                    ) VALUES (?, ?, ?, ?, ?, '正文', ?)
                    """,
                    [
                        (
                            index,
                            index,
                            index,
                            f"scan-{index - 1:04}.txt",
                            str(base / f"scan-{index - 1:04}.txt"),
                            f"cancellable content {index}",
                        )
                        for index in range(1, count + 1)
                    ],
                )
            db.update_root_scan_time(root_id, "ready")
            token = CancelToken()

            def cancel_during_scan(payload: dict[str, object]) -> None:
                if int(payload.get("checked_candidates") or 0) >= 256:
                    token.cancel()

            with self.assertRaises(CancelledError):
                SearchEngine(db).search(
                    SearchQuery(
                        text="content",
                        mode="regex",
                        search_filename=False,
                        search_path=False,
                    ),
                    token,
                    progress_callback=cancel_during_scan,
                )

    def test_more_than_5000_candidates_are_counted_before_display_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(base / "virtual")
            count = 5001
            now = datetime.now(timezone.utc).isoformat()
            with db.connect() as con:
                files = [
                    (
                        root_id,
                        str(base / f"f{index:04}.txt"),
                        f"f{index:04}.txt",
                        ".txt",
                        6,
                        0.0,
                        f"6:{index}",
                        "success",
                        "2",
                        now,
                    )
                    for index in range(count)
                ]
                con.executemany(
                    """
                    INSERT INTO files(
                        root_id, path, filename, extension, size_bytes, modified_time,
                        quick_fingerprint, parse_status, parser_version, last_seen_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    files,
                )
                blocks = [
                    (index, index, 0, "text", "正文", "needle", "needle", "native_text", now)
                    for index in range(1, count + 1)
                ]
                con.executemany(
                    """
                    INSERT INTO content_blocks(
                        id, file_id, block_index, block_type, location_text,
                        raw_text, normalized_text, source_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    blocks,
                )
                fts_rows = [
                    (
                        index,
                        index,
                        index,
                        f"f{index - 1:04}.txt",
                        str(base / f"f{index - 1:04}.txt"),
                        "正文",
                        "needle",
                    )
                    for index in range(1, count + 1)
                ]
                con.executemany(
                    """
                    INSERT INTO content_fts(
                        rowid, block_id, file_id, filename, path, location_text, normalized_text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    fts_rows,
                )
            db.update_root_scan_time(root_id, "ready")

            page = SearchEngine(db).search(
                SearchQuery(
                    text="needle",
                    mode="exact",
                    search_filename=False,
                    search_path=False,
                    page_size=100,
                    max_results=1000,
                )
            )

            self.assertEqual(page.total_confirmed, count)
            self.assertEqual(page.total_candidates, count)
            self.assertEqual(page.available_results, 1000)
            self.assertTrue(page.truncated)

    def test_low_confidence_ocr_is_optional(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            for name, confidence in (("high.png", 0.95), ("low.png", 0.40)):
                path = root / name
                path.write_bytes(b"placeholder")
                file_id, _ = db.upsert_file_metadata(root_id, path)
                block = ContentBlock(
                    file_path=str(path),
                    block_index=0,
                    block_type="image_ocr",
                    location_text="图片 OCR",
                    raw_text="OCR_CONFIDENCE_HIT",
                    normalized_text="ocr_confidence_hit",
                    source_type="ocr",
                    ocr_confidence=confidence,
                )
                db.replace_file_blocks(file_id, name, str(path), [block], parser_name="test")
            db.update_root_scan_time(root_id, "ready")

            engine = SearchEngine(db)
            strict = engine.search(
                SearchQuery(
                    text="OCR_CONFIDENCE_HIT",
                    search_filename=False,
                    search_path=False,
                    ocr_min_confidence=0.60,
                    include_ocr_fuzzy=False,
                )
            )
            fuzzy = engine.search(
                SearchQuery(
                    text="OCR_CONFIDENCE_HIT",
                    search_filename=False,
                    search_path=False,
                    ocr_min_confidence=0.60,
                    include_ocr_fuzzy=True,
                )
            )

            self.assertEqual(strict.total_confirmed, 1)
            self.assertEqual(fuzzy.total_confirmed, 2)
            self.assertTrue(any(result.has_fuzzy_match for result in fuzzy.results))

    def test_chinese_query_automatically_ignores_spacing_differences(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            (root / "sensor.txt").write_text("操作时请拔掉3 个传感器。", encoding="utf-8")
            db = DatabaseManager(base / "index.db")
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            page = SearchEngine(db).search(
                SearchQuery(
                    text="拔掉 3 个传感器",
                    mode="exact",
                    search_filename=False,
                    search_path=False,
                )
            )

            self.assertEqual(page.total_confirmed, 1)
            self.assertIn("拔掉3 个传感器", page.results[0].context)
            self.assertIn(
                "background:#FDE68A",
                highlight_context(page.results[0].context, "拔掉 3 个传感器"),
            )


if __name__ == "__main__":
    unittest.main()
