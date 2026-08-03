from __future__ import annotations

import argparse
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


def _run_row(db: DatabaseManager, run_id: str) -> dict[str, object]:
    with db.connect() as con:
        row = con.execute("SELECT * FROM index_runs WHERE id = ?", (run_id,)).fetchone()
    if row is None:
        return {}
    result = dict(row)
    summary_json = result.get("summary_json")
    if summary_json:
        result["summary"] = json.loads(str(summary_json))
    result.pop("summary_json", None)
    return result


def _latest_run_id(db: DatabaseManager) -> str:
    with db.connect() as con:
        row = con.execute("SELECT id FROM index_runs ORDER BY started_at DESC LIMIT 1").fetchone()
    return str(row["id"]) if row else ""


def _extension_metrics(db: DatabaseManager, run_id: str) -> list[dict[str, object]]:
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT
                COALESCE(extension, '') AS extension,
                COUNT(*) AS parsed_files,
                SUM(size_bytes) AS size_bytes,
                SUM(parse_ms) AS parse_ms,
                SUM(block_count) AS blocks,
                SUM(text_chars) AS text_chars,
                SUM(spool_bytes) AS spool_bytes
            FROM index_file_metrics
            WHERE run_id = ?
            GROUP BY extension
            ORDER BY parse_ms DESC
            """,
            (run_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _status_metrics(db: DatabaseManager) -> list[dict[str, object]]:
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT extension, parse_status, COUNT(*) AS files, SUM(size_bytes) AS size_bytes
            FROM files
            WHERE is_deleted = 0
            GROUP BY extension, parse_status
            ORDER BY extension, parse_status
            """
        ).fetchall()
    return [dict(row) for row in rows]


def _slowest_files(
    db: DatabaseManager,
    run_id: str,
    limit: int = 20,
) -> list[dict[str, object]]:
    with db.connect() as con:
        rows = con.execute(
            """
            SELECT f.path, m.extension, m.size_bytes, m.queue_name, m.queue_wait_ms,
                   m.parse_ms, m.block_count, m.text_chars, m.spool_bytes, m.worker_pid
            FROM index_file_metrics m
            JOIN files f ON f.id = m.file_id
            WHERE m.run_id = ?
            ORDER BY m.parse_ms DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
    return [dict(row) for row in rows]


def _search_checks(db: DatabaseManager, queries: list[str]) -> dict[str, int]:
    engine = SearchEngine(db)
    return {
        query: engine.search(SearchQuery(text=query, mode="exact")).total_confirmed
        for query in queries
    }


def _sqlite_checks(db_path: Path, db: DatabaseManager) -> dict[str, object]:
    report = db.integrity_report()
    con = sqlite3.connect(db_path)
    try:
        report["quick_check"] = [str(row[0]) for row in con.execute("PRAGMA quick_check")]
        report["user_version"] = int(con.execute("PRAGMA user_version").fetchone()[0])
    finally:
        con.close()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Index a real directory into an isolated database and emit a validation report."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query", action="append", default=[])
    parser.add_argument("--no-ocr", action="store_true")
    parser.add_argument("--no-image-ocr", action="store_true")
    parser.add_argument("--no-scanned-pdf-ocr", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    database = args.database.resolve()
    output = args.output.resolve()
    if not root.is_dir():
        parser.error(f"Root directory does not exist: {root}")
    if database.exists() and not args.resume:
        parser.error("Database already exists; pass --resume or choose a new path")
    database.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    settings = AppSettings(
        enable_ocr=not args.no_ocr,
        ocr_images=not args.no_ocr and not args.no_image_ocr,
        ocr_scanned_pdf=not args.no_ocr and not args.no_scanned_pdf_ocr,
        monitor_file_changes=False,
        auto_scan_on_start=False,
        fast_ooxml_enabled=True,
        enable_parse_cache=True,
        defer_fts_during_full_scan=True,
    )
    db = DatabaseManager(database)
    db.initialize()
    root_id = db.add_root(root)
    manager = IndexManager(db, settings)
    last_progress = 0.0

    def progress(payload: dict[str, object]) -> None:
        nonlocal last_progress
        now = time.monotonic()
        stage = str(payload.get("stage") or "")
        if now - last_progress < 5 and stage not in {"finished", "cancelled", "fts"}:
            return
        last_progress = now
        completed = int(payload.get("completed_files") or payload.get("indexed") or 0)
        total = int(payload.get("total_files") or payload.get("scanned") or 0)
        current = Path(str(payload.get("current_file") or "")).name
        print(
            json.dumps(
                {
                    "stage": stage,
                    "completed": completed,
                    "total": total,
                    "failed": int(payload.get("failed") or 0),
                    "queue": payload.get("queue"),
                    "current": current,
                    "eta_seconds": payload.get("eta_seconds"),
                    "eta_ready": payload.get("eta_ready"),
                    "eta_confidence": payload.get("eta_confidence"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    started = time.perf_counter()
    summary = manager.index_root(root_id, progress_callback=progress)
    full_elapsed = time.perf_counter() - started
    full_run_id = _latest_run_id(db)
    full_run = _run_row(db, full_run_id)
    extension_metrics = _extension_metrics(db, full_run_id)
    slowest_files = _slowest_files(db, full_run_id)

    incremental_started = time.perf_counter()
    incremental_summary = manager.index_root(root_id)
    incremental_elapsed = time.perf_counter() - incremental_started
    incremental_run_id = _latest_run_id(db)
    queries = list(dict.fromkeys(args.query))
    report = {
        "root": str(root),
        "database": str(database),
        "database_bytes": database.stat().st_size,
        "settings": settings.to_dict(),
        "full_index_elapsed_seconds": round(full_elapsed, 3),
        "full_index_summary": asdict(summary),
        "full_index_run": full_run,
        "incremental_elapsed_seconds": round(incremental_elapsed, 3),
        "incremental_summary": asdict(incremental_summary),
        "incremental_run": _run_row(db, incremental_run_id),
        "database_stats": db.stats(),
        "extension_metrics": extension_metrics,
        "status_metrics": _status_metrics(db),
        "slowest_files": slowest_files,
        "failed_files": [dict(row) for row in db.failed_files(limit=1000)],
        "search_checks": _search_checks(db, queries),
        "sqlite_checks": _sqlite_checks(database, db),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["sqlite_checks"]["integrity"] == ["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
