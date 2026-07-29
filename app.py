from __future__ import annotations

import multiprocessing
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from importlib import resources
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.config.constants import APP_DISPLAY_NAME
from local_full_text_search.config.constants import TEMP_DIR
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.search_query import SearchQuery
from local_full_text_search.services.logging_service import configure_logging
from local_full_text_search.services.settings_service import SettingsService


def run_self_test() -> int:
    """Non-interactive smoke test used after packaging."""

    try:
        from PySide6.QtWidgets import QApplication
        from local_full_text_search.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory(prefix="lfts_self_test_") as tmp:
            base = Path(tmp)
            settings_service = SettingsService(base / "settings.json")
            settings = AppSettings(enable_ocr=False, monitor_file_changes=False)
            settings_service.save(settings)
            db = DatabaseManager(base / "self_test.db")
            db.initialize()
            app = QApplication.instance() or QApplication(["LocalFullTextSearch", "--self-test"])
            apply_light_theme(app)
            window = MainWindow(db, settings, settings_service)
            window.close()
        Path("self_test_result.txt").write_text("SELF_TEST_OK\n", encoding="utf-8")
        print("SELF_TEST_OK")
        return 0
    except Exception as exc:
        Path("self_test_result.txt").write_text(f"SELF_TEST_FAILED: {exc}\n", encoding="utf-8")
        print(f"SELF_TEST_FAILED: {exc}")
        return 1


def run_ui_validation() -> int:
    """Render the desktop UI offscreen and reject blank or undersized output."""

    result_path = Path("ui_validation_result.txt")
    image_path = Path("ui_validation.png")
    index_image_path = Path("index_ui_validation.png")
    try:
        from PySide6.QtGui import QImage
        from PySide6.QtWidgets import QApplication
        from local_full_text_search.ui.main_window import MainWindow

        with tempfile.TemporaryDirectory(prefix="lfts_ui_validation_") as tmp:
            base = Path(tmp)
            settings_service = SettingsService(base / "settings.json")
            settings = AppSettings(enable_ocr=False, monitor_file_changes=False)
            settings_service.save(settings)
            db = DatabaseManager(base / "ui.db")
            db.initialize()
            sample_root = base / "files"
            sample_root.mkdir()
            sample_file = sample_root / (
                "CL8000i-301140100~301140400-第一抓杯手手指组件拆卸方法测试文档.txt"
            )
            sample_file.write_text(
                "第 6 页\n操作前请拔掉3 个传感器，并标记插头。\n"
                "完整路径、日期和右侧命中内容都必须可见。",
                encoding="utf-8",
            )
            root_id = db.add_root(sample_root)
            IndexManager(db, settings).index_root(root_id)
            app = QApplication.instance() or QApplication(["LocalFullTextSearch", "--validate-ui"])
            apply_light_theme(app)
            window = MainWindow(db, settings, settings_service)
            window.resize(1280, 800)
            window.show()
            window.search_page.search_box.input.setText("拔掉 3 个传感器")
            page = SearchEngine(db).search(
                SearchQuery(text="拔掉 3 个传感器", mode="exact")
            )
            window.search_page.set_results(page)
            window.show_preview(page.results[0])
            app.processEvents()
            pixmap = window.grab()
            image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGB32)
            if image.width() < 1000 or image.height() < 650:
                raise RuntimeError(f"Unexpected UI size: {image.width()}x{image.height()}")
            colors = {
                image.pixelColor(x, y).rgba()
                for x in range(0, image.width(), 32)
                for y in range(0, image.height(), 32)
            }
            if not pixmap.save(str(image_path), "PNG"):
                raise RuntimeError("Unable to save UI screenshot")
            if len(colors) < 5:
                raise RuntimeError(f"UI render appears blank: {len(colors)} sampled colors")
            window.switch_page("index")
            window.on_scan_progress(
                {
                    "stage": "indexing",
                    "phase_label": "正在解析并写入索引",
                    "indexed": 104,
                    "scanned": 337,
                    "completed_files": 104,
                    "total_files": 337,
                    "failed": 1,
                    "current_file": str(sample_root / "large-document.docx"),
                    "eta_lower_seconds": 420,
                    "eta_upper_seconds": 660,
                    "active_phase": "docx_paragraph",
                    "active_completed_units": 128,
                    "active_total_units": 640,
                    "no_progress_seconds": 7,
                    "retry_count": 1,
                    "excluded_video": 9,
                }
            )
            app.processEvents()
            index_pixmap = window.grab()
            if not index_pixmap.save(str(index_image_path), "PNG"):
                raise RuntimeError("Unable to save index UI screenshot")
            window.close()
            app.processEvents()
            message = (
                "UI_VALIDATION_OK\n"
                f"size={image.width()}x{image.height()}; sampled_colors={len(colors)}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"UI_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"UI_VALIDATION_FAILED: {exc}")
        return 1


def run_core_validation() -> int:
    """Build a temporary multi-format index and verify exact search hits."""

    result_path = Path("core_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_validation_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            _create_validation_files(root)

            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings()).index_root(root_id)
            engine = SearchEngine(db)

            checks = {
                "TXT_VALIDATION_HIT": "txt",
                "PDF_VALIDATION_HIT": "pdf",
                "DOCX_VALIDATION_HIT": "docx",
                "XLSX_VALIDATION_HIT": "xlsx",
                "PPTX_VALIDATION_HIT": "pptx",
                "OCR TEST 123": "ocr_image",
            }
            failures: list[str] = []
            for term, label in checks.items():
                page = engine.search(
                    SearchQuery(
                        text=term,
                        mode="exact",
                        include_ocr_fuzzy=label == "ocr_image",
                    )
                )
                if page.total_confirmed < 1:
                    failures.append(f"{label}:{term}")
            if failures:
                diagnostic_text = "; ".join(
                    f"{row['path']} [{row['parse_error_code']}]: {row['parse_error_message']}"
                    for row in db.failed_files()
                )
                detail = f"; diagnostics={diagnostic_text}" if diagnostic_text else ""
                raise RuntimeError("未命中: " + ", ".join(failures) + detail)
            message = (
                "CORE_VALIDATION_OK\n"
                f"scanned={summary.scanned}; indexed={summary.indexed}; "
                f"skipped={summary.skipped}; failed={summary.failed}; unsupported={summary.unsupported}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"CORE_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"CORE_VALIDATION_FAILED: {exc}")
        return 1


def run_process_pool_validation() -> int:
    """Force DOCX/XLSX through the spawned process parser in a frozen build."""

    result_path = Path("process_pool_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_process_validation_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            _create_process_validation_files(root)

            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            queues: list[str] = []
            worker_pids: list[int] = []
            result_bytes: list[int] = []
            descriptor_bytes: list[int] = []

            def capture_progress(payload: dict[str, object]) -> None:
                queue_name = payload.get("completed_queue")
                if payload.get("stage") == "indexing" and isinstance(queue_name, str):
                    queues.append(queue_name)
                    if isinstance(payload.get("worker_pid"), int):
                        worker_pids.append(int(payload["worker_pid"]))
                    if isinstance(payload.get("process_result_bytes"), int):
                        result_bytes.append(int(payload["process_result_bytes"]))
                    if isinstance(payload.get("process_descriptor_bytes"), int):
                        descriptor_bytes.append(int(payload["process_descriptor_bytes"]))

            settings = AppSettings(
                enable_ocr=False,
                process_parser_workers=1,
                process_pending_tasks=1,
                process_max_tasks_per_child=1,
                large_office_process_min_bytes=0,
            )
            summary = IndexManager(db, settings).index_root(root_id, progress_callback=capture_progress)
            engine = SearchEngine(db)
            terms = ("PROCESS_DOCX_VALIDATION_HIT", "PROCESS_XLSX_VALIDATION_HIT")
            failures = [
                term
                for term in terms
                if engine.search(SearchQuery(text=term, mode="exact")).total_confirmed < 1
            ]
            if failures:
                raise RuntimeError("Missing process-pool hits: " + ", ".join(failures))
            if queues.count("office_process") != 2:
                raise RuntimeError(f"Unexpected parse queues: {queues}")
            if len(set(worker_pids)) != 2:
                raise RuntimeError(f"Process recycling was not observed: {worker_pids}")
            if not descriptor_bytes or max(descriptor_bytes) >= 4096:
                raise RuntimeError(f"Process descriptor transfer is unexpectedly large: {descriptor_bytes}")
            message = (
                "PROCESS_POOL_VALIDATION_OK\n"
                f"scanned={summary.scanned}; indexed={summary.indexed}; failed={summary.failed}; "
                f"office_process={queues.count('office_process')}; recycled_workers={len(set(worker_pids))}; "
                f"spooled_bytes={sum(result_bytes)}; ipc_descriptor_bytes={sum(descriptor_bytes)}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"PROCESS_POOL_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"PROCESS_POOL_VALIDATION_FAILED: {exc}")
        return 1


def run_schema_v3_validation() -> int:
    """Verify deduplication, search expansion, metrics, and SQLite integrity."""

    result_path = Path("schema_v3_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_schema_v3_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            first = root / "first.txt"
            second = root / "second.txt"
            first.write_text("SCHEMA_V3_VALIDATION_HIT", encoding="utf-8")
            shutil.copy2(first, second)
            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            summary = IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)
            page = SearchEngine(db).search(
                SearchQuery(
                    text="SCHEMA_V3_VALIDATION_HIT",
                    search_filename=False,
                    search_path=False,
                )
            )
            stats = db.stats()
            integrity = db.integrity_report()
            if summary.indexed != 2 or stats["documents"] != 1 or stats["blocks"] != 1:
                raise RuntimeError(f"Unexpected dedup stats: summary={summary}; stats={stats}")
            if page.total_confirmed != 2:
                raise RuntimeError(f"Duplicate paths were not expanded: {page.total_confirmed}")
            if integrity["integrity"] != ["ok"] or integrity["foreign_key_errors"]:
                raise RuntimeError(f"SQLite integrity failed: {integrity}")
            with db.connect() as con:
                run = con.execute(
                    "SELECT status, discovered_files FROM index_runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
                task_status = con.execute(
                    "SELECT status FROM parse_tasks ORDER BY id DESC LIMIT 1"
                ).fetchone()
            if run is None or run["status"] != "complete" or int(run["discovered_files"]) != 2:
                raise RuntimeError("Run metrics were not finalized")
            if task_status is None or task_status["status"] != "complete":
                raise RuntimeError("Parse task state was not finalized")
            message = (
                "SCHEMA_V3_VALIDATION_OK\n"
                f"files={stats['files']}; documents={stats['documents']}; blocks={stats['blocks']}; "
                f"expanded_hits={page.total_confirmed}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"SCHEMA_V3_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"SCHEMA_V3_VALIDATION_FAILED: {exc}")
        return 1


def run_schema_v4_validation() -> int:
    """Migrate a token-heavy schema-v3 database and verify a real update."""

    result_path = Path("schema_v4_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_schema_v4_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            source = root / "legacy-tokens.txt"
            source.write_text("SCHEMA_V4_OLD_HIT", encoding="utf-8")
            db_path = base / "validation.db"
            db = DatabaseManager(db_path)
            db.initialize()
            root_id = db.add_root(root)
            IndexManager(db, AppSettings(enable_ocr=False)).index_root(root_id)

            legacy_token_count = 250_000
            with db.connect() as con:
                block_id = int(con.execute("SELECT id FROM content_blocks LIMIT 1").fetchone()[0])
                con.execute("DROP INDEX idx_short_tokens_block")
                con.executemany(
                    "INSERT INTO short_tokens(token, block_id) VALUES (?, ?)",
                    ((f"legacy-{index}", block_id) for index in range(legacy_token_count)),
                )
                con.execute("PRAGMA user_version = 3")

            migration_started = time.perf_counter()
            migrated = DatabaseManager(db_path)
            migrated.initialize()
            migration_ms = int((time.perf_counter() - migration_started) * 1000)
            with migrated.connect() as con:
                token_rows = int(con.execute("SELECT COUNT(*) FROM short_tokens").fetchone()[0])
                version = int(con.execute("PRAGMA user_version").fetchone()[0])
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
            if token_rows != 0 or version != 4:
                raise RuntimeError(f"Unexpected migration state: tokens={token_rows}; version={version}")
            if "idx_short_tokens_block" not in indexes or "idx_short_tokens_block" not in plan:
                raise RuntimeError(f"Missing block lookup index: indexes={indexes}; plan={plan}")
            if migrated.requires_full_rebuild():
                raise RuntimeError("Schema-v4 migration incorrectly forced a content reparse")

            source.write_text("SCHEMA_V4_NEW_HIT", encoding="utf-8")
            update_started = time.perf_counter()
            summary = IndexManager(migrated, AppSettings(enable_ocr=False)).index_root(root_id)
            update_ms = int((time.perf_counter() - update_started) * 1000)
            hits = SearchEngine(migrated).search(
                SearchQuery(text="SCHEMA_V4_NEW_HIT", mode="exact")
            ).total_confirmed
            report = migrated.integrity_report()
            backup = base / "validation.schema-v3.backup.db"
            if not backup.is_file():
                raise RuntimeError("Schema-v3 backup was not created")
            if summary.indexed != 1 or summary.failed != 0 or hits != 1:
                raise RuntimeError(
                    f"Post-migration update failed: summary={summary}; hits={hits}"
                )
            if report["integrity"] != ["ok"] or report["foreign_key_errors"]:
                raise RuntimeError(f"SQLite integrity failed: {report}")
            message = (
                "SCHEMA_V4_VALIDATION_OK\n"
                f"legacy_tokens={legacy_token_count}; migration_ms={migration_ms}; "
                f"update_ms={update_ms}; indexed={summary.indexed}; hits={hits}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"SCHEMA_V4_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"SCHEMA_V4_VALIDATION_FAILED: {exc}")
        return 1


def run_field_reindex_validation(db_path: Path, settings_path: Path) -> int:
    """Run a real configured update against an explicitly supplied database copy."""

    result_path = Path("field_reindex_validation_result.txt")
    try:
        settings = SettingsService(settings_path).load()
        db = DatabaseManager(db_path)
        migration_started = time.perf_counter()
        db.initialize()
        migration_ms = int((time.perf_counter() - migration_started) * 1000)
        latest_progress: dict[str, object] = {}
        last_reported = 0.0

        def capture_progress(payload: dict[str, object]) -> None:
            nonlocal latest_progress, last_reported
            latest_progress = dict(payload)
            now = time.monotonic()
            if now - last_reported >= 10.0 or payload.get("stage") in {
                "finished",
                "cancelled",
            }:
                print(
                    "FIELD_PROGRESS "
                    + json.dumps(
                        {
                            "stage": payload.get("stage"),
                            "completed": payload.get("completed_files"),
                            "total": payload.get("total_files"),
                            "failed": payload.get("failed"),
                            "queue": payload.get("queue"),
                            "current_file": payload.get("current_file"),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                last_reported = now

        update_started = time.perf_counter()
        summary = IndexManager(db, settings).index_enabled_roots(
            progress_callback=capture_progress
        )
        update_ms = int((time.perf_counter() - update_started) * 1000)
        report = db.integrity_report()
        with db.connect() as con:
            version = int(con.execute("PRAGMA user_version").fetchone()[0])
            short_tokens = int(con.execute("SELECT COUNT(*) FROM short_tokens").fetchone()[0])
            blocks = int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0])
            fts_rows = int(con.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0])
            latest_run = con.execute(
                "SELECT id, status, discovered_files, write_ms, fts_ms, total_ms "
                "FROM index_runs ORDER BY started_at DESC LIMIT 1"
            ).fetchone()
            incomplete_states = {
                str(row["key"]): str(row["value"])
                for row in con.execute(
                    "SELECT key, value FROM index_state WHERE key IN ("
                    "'content_fts_dirty', 'full_batch_incomplete', 'schema_v3_rebuild_required')"
                )
            }
            task_counts = {
                str(row["status"]): int(row["n"])
                for row in con.execute(
                    "SELECT status, COUNT(*) AS n FROM parse_tasks "
                    "WHERE run_id = ? GROUP BY status",
                    (str(latest_run["id"]),),
                )
            } if latest_run is not None else {}
        if summary.cancelled or summary.failed:
            raise RuntimeError(f"Real update reported failures: {summary}")
        if version != 4 or short_tokens != 0:
            raise RuntimeError(f"Migration incomplete: version={version}; short_tokens={short_tokens}")
        if latest_run is None or str(latest_run["status"]) != "complete":
            raise RuntimeError(f"Latest run did not complete: {dict(latest_run or {})}")
        if fts_rows != blocks:
            raise RuntimeError(f"FTS mismatch: fts={fts_rows}; blocks={blocks}")
        if any(value != "0" for value in incomplete_states.values()):
            raise RuntimeError(f"Incomplete index state: {incomplete_states}")
        if report["integrity"] != ["ok"] or report["foreign_key_errors"]:
            raise RuntimeError(f"SQLite integrity failed: {report}")
        message = (
            "FIELD_REINDEX_VALIDATION_OK\n"
            f"migration_ms={migration_ms}; update_ms={update_ms}; scanned={summary.scanned}; "
            f"indexed={summary.indexed}; skipped={summary.skipped}; failed={summary.failed}; "
            f"partial_success={summary.partial_success}; blocks={blocks}; fts_rows={fts_rows}; "
            f"write_ms={int(latest_run['write_ms'])}; fts_ms={int(latest_run['fts_ms'])}; "
            f"tasks={json.dumps(task_counts, ensure_ascii=False, sort_keys=True)}; "
            f"last_stage={latest_progress.get('stage', '')}\n"
        )
        result_path.write_text(message, encoding="utf-8")
        print(message, end="")
        return 0
    except Exception as exc:
        result_path.write_text(f"FIELD_REINDEX_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"FIELD_REINDEX_VALIDATION_FAILED: {exc}")
        return 1


def run_database_lock_validation() -> int:
    """Verify busy-writer tolerance and migrated FTS rebuild performance."""

    result_path = Path("database_lock_validation_result.txt")
    try:
        with tempfile.TemporaryDirectory(prefix="lfts_database_lock_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            file_count = 64
            for index in range(file_count):
                (root / f"document-{index:03d}.txt").write_text(
                    f"DATABASE_LOCK_VALIDATION_HIT_{index:03d}",
                    encoding="utf-8",
                )

            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(
                enable_ocr=False,
                parser_workers=4,
                normal_pending_tasks=4,
                index_write_batch_size=4,
                db_write_batch_blocks=4,
                db_write_max_delay_ms=10,
                defer_fts_during_full_scan=True,
                enable_parse_cache=False,
            )
            first = IndexManager(db, settings).index_root(root_id)
            if first.indexed != file_count:
                raise RuntimeError(f"Unable to prepare baseline index: {first}")

            with db.connect() as con:
                probe_file_id = int(
                    con.execute("SELECT id FROM files ORDER BY id LIMIT 1").fetchone()[0]
                )
            probe_task_id = db.create_parse_task(
                probe_file_id,
                "database-lock-probe",
                "normal",
            )
            locker = sqlite3.connect(db.db_path, timeout=0.1)
            try:
                locker.execute("BEGIN IMMEDIATE")
                lock_skip_started = time.perf_counter()
                running_updated = db.try_mark_tasks_running(
                    [probe_task_id],
                    timeout_seconds=0.0,
                )
                spool_updated = db.try_mark_task_spooled(
                    probe_task_id,
                    base / "lock-probe.spool",
                    "lock-probe-checksum",
                    timeout_seconds=0.0,
                )
                lock_skip_elapsed = time.perf_counter() - lock_skip_started
            finally:
                locker.rollback()
                locker.close()
            if running_updated or spool_updated or lock_skip_elapsed >= 0.2:
                raise RuntimeError(
                    "Busy diagnostic writes did not skip promptly: "
                    f"running={running_updated}; spool={spool_updated}; "
                    f"elapsed={lock_skip_elapsed:.3f}s"
                )
            db.mark_task_failed(probe_task_id, "LOCK_PROBE_COMPLETE", "validation probe")

            stale_rows = 40_000
            with db.connect() as con:
                con.executemany(
                    """
                    INSERT INTO content_fts(
                        rowid, block_id, file_id, filename, path, location_text, normalized_text
                    ) VALUES (?, ?, 0, 'stale', 'stale', '', 'MIGRATED_STALE_FTS')
                    """,
                    ((1_000_000 + index, 1_000_000 + index) for index in range(stale_rows)),
                )
                con.execute(
                    "INSERT INTO index_state(key, value) VALUES ('content_fts_dirty', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
                con.execute(
                    "INSERT INTO index_state(key, value) VALUES ('full_batch_incomplete', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
                con.execute(
                    "INSERT INTO index_state(key, value) VALUES ('schema_v3_rebuild_required', '1') "
                    "ON CONFLICT(key) DO UPDATE SET value='1'"
                )
                con.execute(
                    "UPDATE files SET parser_version = NULL, parse_status = 'pending' "
                    "WHERE is_deleted = 0"
                )

            started = time.perf_counter()
            summary = IndexManager(db, settings).index_root(root_id)
            elapsed = time.perf_counter() - started
            report = db.integrity_report()
            with db.connect() as con:
                fts_rows = int(con.execute("SELECT COUNT(*) FROM content_fts").fetchone()[0])
                block_rows = int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0])
                latest_run = con.execute(
                    "SELECT status, write_ms FROM index_runs ORDER BY started_at DESC LIMIT 1"
                ).fetchone()
                states = {
                    str(row["key"]): str(row["value"])
                    for row in con.execute(
                        "SELECT key, value FROM index_state WHERE key IN ("
                        "'content_fts_dirty', 'full_batch_incomplete', 'schema_v3_rebuild_required')"
                    )
                }
            hits = SearchEngine(db).search(
                SearchQuery(text="DATABASE_LOCK_VALIDATION_HIT_063", mode="exact")
            ).total_confirmed
            if summary.indexed != file_count or summary.failed != 0:
                raise RuntimeError(f"Unexpected rebuild summary: {summary}")
            if elapsed >= 20:
                raise RuntimeError(f"Migrated FTS rebuild remained too slow: {elapsed:.3f}s")
            if latest_run is None or latest_run["status"] != "complete":
                raise RuntimeError(f"Index run did not complete: {dict(latest_run or {})}")
            if fts_rows != block_rows or hits != 1:
                raise RuntimeError(
                    f"FTS rebuild mismatch: fts={fts_rows}; blocks={block_rows}; hits={hits}"
                )
            state_keys = (
                "content_fts_dirty",
                "full_batch_incomplete",
                "schema_v3_rebuild_required",
            )
            if any(states.get(key) != "0" for key in state_keys):
                raise RuntimeError(f"Rebuild flags were not cleared: {states}")
            if report["integrity"] != ["ok"] or report["foreign_key_errors"]:
                raise RuntimeError(f"SQLite integrity failed: {report}")
            message = (
                "DATABASE_LOCK_VALIDATION_OK\n"
                f"files={file_count}; stale_fts_rows={stale_rows}; elapsed_seconds={elapsed:.3f}; "
                f"lock_skip_ms={int(lock_skip_elapsed * 1000)}; "
                f"write_ms={int(latest_run['write_ms'])}; fts_rows={fts_rows}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"DATABASE_LOCK_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"DATABASE_LOCK_VALIDATION_FAILED: {exc}")
        return 1


def run_shutdown_validation() -> int:
    """Force-cancel an active spawned process index and enforce a bounded exit."""

    result_path = Path("shutdown_validation_result.txt")
    try:
        from docx import Document

        with tempfile.TemporaryDirectory(prefix="lfts_shutdown_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            for file_index in range(8):
                document = Document()
                for paragraph in range(800):
                    document.add_paragraph(
                        f"SHUTDOWN_VALIDATION_{file_index}_{paragraph} " + "payload " * 12
                    )
                document.save(root / f"document-{file_index}.docx")
            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(
                enable_ocr=False,
                large_office_process_min_bytes=0,
                process_parser_workers=2,
                process_pending_tasks=2,
            )
            manager = IndexManager(db, settings)
            token = CancelToken()
            planned = threading.Event()
            failure: list[BaseException] = []

            def progress(payload: dict[str, object]) -> None:
                if payload.get("stage") in {"planning", "indexing"}:
                    planned.set()

            def run_index() -> None:
                try:
                    manager.index_root(root_id, token, progress)
                except BaseException as exc:
                    failure.append(exc)

            thread = threading.Thread(target=run_index, name="shutdown-validation")
            thread.start()
            planned.wait(timeout=10)
            time.sleep(0.25)
            started = time.perf_counter()
            token.cancel(force=True)
            manager.force_terminate_processes()
            thread.join(timeout=4)
            elapsed = time.perf_counter() - started
            if thread.is_alive():
                raise RuntimeError("Index thread did not stop within 4 seconds")
            if failure:
                raise RuntimeError(f"Index cancellation raised: {failure[0]}")
            message = f"SHUTDOWN_VALIDATION_OK\nstop_seconds={elapsed:.3f}\n"
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(f"SHUTDOWN_VALIDATION_FAILED: {exc}\n", encoding="utf-8")
        print(f"SHUTDOWN_VALIDATION_FAILED: {exc}")
        return 1


def run_checkpoint_timeout_validation() -> int:
    """Force an OCR no-progress timeout and verify incomplete text is not published."""

    result_path = Path("checkpoint_timeout_validation_result.txt")
    try:
        import fitz
        from PIL import Image, ImageDraw, ImageFont

        with tempfile.TemporaryDirectory(prefix="lfts_checkpoint_timeout_") as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            image_path = base / "scanned-page.png"
            image = Image.new("RGB", (1600, 1200), "white")
            draw = ImageDraw.Draw(image)
            font_path = Path("C:/Windows/Fonts/arial.ttf")
            font = (
                ImageFont.truetype(str(font_path), 72)
                if font_path.exists()
                else ImageFont.load_default()
            )
            for line in range(12):
                draw.text(
                    (60, 50 + line * 90),
                    f"OCR TIMEOUT PAGE CONTENT {line:02d}",
                    fill="black",
                    font=font,
                )
            image.save(image_path)

            pdf_path = root / "checkpoint.pdf"
            document = fitz.open()
            native_page = document.new_page()
            native_page.insert_text((72, 72), "NATIVE_CHECKPOINT_SURVIVES_TIMEOUT")
            for _ in range(30):
                scanned_page = document.new_page(width=800, height=600)
                scanned_page.insert_image(scanned_page.rect, filename=str(image_path))
            document.save(pdf_path)
            document.close()

            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(
                enable_ocr=True,
                ocr_images=True,
                ocr_scanned_pdf=True,
                ocr_cpu_threads=1,
                ocr_no_progress_timeout_seconds=1,
                no_progress_max_retries=0,
                process_parser_workers=1,
                process_pending_tasks=1,
            )
            started = time.perf_counter()
            summary = IndexManager(db, settings).index_root(root_id)
            elapsed = time.perf_counter() - started
            search_blocked = False
            try:
                SearchEngine(db).search(
                    SearchQuery(text="NATIVE_CHECKPOINT_SURVIVES_TIMEOUT", mode="exact")
                )
            except Exception as exc:
                search_blocked = exc.__class__.__name__ == "IndexNotReadyError"
            with db.connect() as con:
                row = con.execute(
                    "SELECT parse_status, parse_error_code FROM files WHERE path = ?",
                    (str(pdf_path),),
                ).fetchone()
                block_count = int(con.execute("SELECT COUNT(*) FROM content_blocks").fetchone()[0])
            if row is None:
                raise RuntimeError("Checkpoint PDF was not recorded")
            if row["parse_status"] != "failed_retryable" or row["parse_error_code"] != "PARSE_NO_PROGRESS":
                raise RuntimeError(f"Unexpected timeout status: {dict(row)}")
            if summary.failed != 1 or not search_blocked or block_count != 0:
                raise RuntimeError(
                    f"Incomplete content publication was not blocked: summary={summary}; "
                    f"search_blocked={search_blocked}; blocks={block_count}"
                )
            message = (
                "CHECKPOINT_TIMEOUT_VALIDATION_OK\n"
                f"elapsed_seconds={elapsed:.3f}; failed={summary.failed}; "
                f"search_blocked={search_blocked}; published_blocks={block_count}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(
            f"CHECKPOINT_TIMEOUT_VALIDATION_FAILED: {exc}\n",
            encoding="utf-8",
        )
        print(f"CHECKPOINT_TIMEOUT_VALIDATION_FAILED: {exc}")
        return 1


def run_legacy_shutdown_validation(source: Path) -> int:
    """Start a real Office conversion, force-cancel it, and verify PID cleanup."""

    result_path = Path("legacy_shutdown_validation_result.txt")
    try:
        source = source.resolve()
        if not source.is_file() or source.suffix.lower() not in {".doc", ".xls", ".ppt"}:
            raise ValueError("需要现有的 .doc/.xls/.ppt 文件")
        runtime_parent = Path.cwd() if Path.cwd().anchor == source.anchor else None
        with tempfile.TemporaryDirectory(
            prefix="lfts_legacy_shutdown_",
            dir=runtime_parent,
        ) as tmp:
            base = Path(tmp)
            root = base / "files"
            root.mkdir()
            linked = root / source.name
            try:
                os.link(source, linked)
            except OSError:
                shutil.copy2(source, linked)
            db = DatabaseManager(base / "validation.db")
            db.initialize()
            root_id = db.add_root(root)
            settings = AppSettings(
                enable_ocr=False,
                process_parser_workers=1,
                process_pending_tasks=1,
                legacy_no_progress_timeout_seconds=600,
                legacy_conversion_cache=False,
            )
            manager = IndexManager(db, settings)
            token = CancelToken()
            failure: list[BaseException] = []

            def run_index() -> None:
                try:
                    manager.index_root(root_id, token)
                except BaseException as exc:
                    failure.append(exc)

            thread = threading.Thread(target=run_index, name="legacy-shutdown-validation")
            started = time.perf_counter()
            started_wall = time.time()
            thread.start()
            record: dict[str, object] | None = None
            deadline = time.monotonic() + 45
            while time.monotonic() < deadline:
                records = [
                    path
                    for path in (TEMP_DIR / "process_results").glob("*/office_processes/*.json")
                    if path.stat().st_mtime >= started_wall
                ]
                if records:
                    record = json.loads(records[-1].read_text(encoding="ascii"))
                    break
                if not thread.is_alive():
                    break
                time.sleep(0.1)
            if record is None:
                raise RuntimeError("未观察到应用登记的 Office 自动化进程")
            office_pid = int(record["pid"])
            cancel_started = time.perf_counter()
            token.cancel(force=True)
            manager.force_terminate_processes()
            thread.join(timeout=4)
            stop_seconds = time.perf_counter() - cancel_started
            if thread.is_alive():
                raise RuntimeError("老版 Office 索引线程未在 4 秒内停止")
            if failure:
                raise RuntimeError(f"取消过程抛出异常: {failure[0]}")
            try:
                import psutil

                if psutil.pid_exists(office_pid):
                    process = psutil.Process(office_pid)
                    expected = float(record.get("create_time") or 0.0)
                    if not expected or abs(process.create_time() - expected) <= 1.0:
                        raise RuntimeError(f"Office 自动化进程仍然存在: PID {office_pid}")
            except ImportError:
                pass
            message = (
                "LEGACY_SHUTDOWN_VALIDATION_OK\n"
                f"office_pid={office_pid}; stop_seconds={stop_seconds:.3f}; "
                f"elapsed_seconds={time.perf_counter() - started:.3f}\n"
            )
            result_path.write_text(message, encoding="utf-8")
            print(message, end="")
            return 0
    except Exception as exc:
        result_path.write_text(
            f"LEGACY_SHUTDOWN_VALIDATION_FAILED: {exc}\n",
            encoding="utf-8",
        )
        print(f"LEGACY_SHUTDOWN_VALIDATION_FAILED: {exc}")
        return 1


def _create_process_validation_files(root: Path) -> None:
    from docx import Document
    from openpyxl import Workbook

    document = Document()
    for index in range(100):
        document.add_paragraph(f"PROCESS_DOCX_VALIDATION_HIT paragraph {index}")
    document.save(root / "process_sample.docx")

    workbook = Workbook(write_only=True)
    sheet = workbook.create_sheet("ProcessValidation")
    for index in range(1000):
        sheet.append([index, "PROCESS_XLSX_VALIDATION_HIT", f"row-{index}"])
    workbook.save(root / "process_sample.xlsx")


def _create_validation_files(root: Path) -> None:
    (root / "sample.txt").write_text("TXT_VALIDATION_HIT\nBS-2800M2", encoding="utf-8")

    from docx import Document
    doc = Document()
    doc.add_paragraph("DOCX_VALIDATION_HIT")
    doc.save(root / "sample.docx")

    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet["A1"] = "XLSX_VALIDATION_HIT"
    workbook.save(root / "sample.xlsx")

    from pptx import Presentation
    presentation = Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[6])
    text_box = slide.shapes.add_textbox(914400, 914400, 3657600, 914400)
    text_box.text = "PPTX_VALIDATION_HIT"
    presentation.save(root / "sample.pptx")

    import fitz
    pdf = fitz.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "PDF_VALIDATION_HIT")
    pdf.save(root / "sample.pdf")
    pdf.close()

    from PIL import Image, ImageDraw, ImageFont
    image = Image.new("RGB", (900, 260), "white")
    draw = ImageDraw.Draw(image)
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 86) if font_path.exists() else ImageFont.load_default()
    draw.text((48, 80), "OCR TEST 123", fill="black", font=font)
    image.save(root / "sample_ocr.png")


def apply_light_theme(app: object) -> None:
    try:
        qss = resources.files("local_full_text_search.ui.styles").joinpath("light.qss").read_text(encoding="utf-8")
        app.setStyleSheet(qss)
    except Exception:
        # A missing stylesheet should never prevent the search tool from opening.
        pass


def main() -> int:
    configure_logging()
    if "--self-test" in sys.argv:
        return run_self_test()
    if "--validate-core" in sys.argv:
        return run_core_validation()
    if "--validate-process-pool" in sys.argv:
        return run_process_pool_validation()
    if "--validate-schema-v3" in sys.argv:
        return run_schema_v3_validation()
    if "--validate-schema-v4" in sys.argv:
        return run_schema_v4_validation()
    if "--validate-field-reindex" in sys.argv:
        index = sys.argv.index("--validate-field-reindex")
        if index + 2 >= len(sys.argv):
            print("--validate-field-reindex 需要数据库副本和设置文件路径")
            return 2
        return run_field_reindex_validation(
            Path(sys.argv[index + 1]),
            Path(sys.argv[index + 2]),
        )
    if "--validate-database-lock" in sys.argv:
        return run_database_lock_validation()
    if "--validate-shutdown" in sys.argv:
        return run_shutdown_validation()
    if "--validate-checkpoint-timeout" in sys.argv:
        return run_checkpoint_timeout_validation()
    if "--validate-legacy-shutdown" in sys.argv:
        index = sys.argv.index("--validate-legacy-shutdown")
        if index + 1 >= len(sys.argv):
            print("--validate-legacy-shutdown 需要一个 .doc/.xls/.ppt 路径")
            return 2
        return run_legacy_shutdown_validation(Path(sys.argv[index + 1]))
    if "--validate-ui" in sys.argv:
        return run_ui_validation()
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox
        from local_full_text_search.ui.main_window import MainWindow
    except ImportError as exc:
        print("缺少 PySide6，无法启动图形界面。请先运行: python -m pip install -r requirements.txt")
        print(exc)
        return 2

    settings_service = SettingsService()
    settings = settings_service.load()
    db = DatabaseManager()
    db.initialize()
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    apply_light_theme(app)
    try:
        window = MainWindow(db, settings, settings_service)
        window.show()
        return app.exec()
    except Exception as exc:
        QMessageBox.critical(None, "启动失败", str(exc))
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
