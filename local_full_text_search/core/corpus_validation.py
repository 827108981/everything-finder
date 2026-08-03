from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import psutil

from local_full_text_search.config.constants import APP_DATA_DIR, DB_PATH
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import IndexManager
from local_full_text_search.core.hardware_profile import (
    build_performance_profile,
    detect_hardware_profile_for_roots,
    settings_for_profile,
)
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


DEFAULT_GOLDEN_QUERIES = (
    "拔掉 3 个传感器",
    "Dispersion",
)


def compare_index_databases(
    baseline_path: Path,
    candidate_path: Path,
    queries_path: Path,
) -> dict[str, Any]:
    baseline_path = Path(baseline_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    queries = _load_queries(Path(queries_path))
    baseline = _database_snapshot(baseline_path, queries)
    candidate = _database_snapshot(candidate_path, queries)
    file_inventory_equal = (
        baseline["file_inventory"] == candidate["file_inventory"]
    )
    content_digest_equal = (
        baseline["content_digest"] == candidate["content_digest"]
    )
    query_results_equal = (
        baseline["query_results"] == candidate["query_results"]
    )
    status_equal = baseline["status_counts"] == candidate["status_counts"]
    passed = bool(
        file_inventory_equal
        and content_digest_equal
        and query_results_equal
        and status_equal
        and baseline["integrity_ok"]
        and candidate["integrity_ok"]
    )
    return {
        "passed": passed,
        "baseline": baseline,
        "candidate": candidate,
        "queries": queries,
        "file_inventory_equal": file_inventory_equal,
        "content_digest_equal": content_digest_equal,
        "query_results_equal": query_results_equal,
        "status_counts_equal": status_equal,
        "differences": {
            "files": _mapping_differences(
                baseline["file_inventory"],
                candidate["file_inventory"],
            ),
            "queries": _mapping_differences(
                baseline["query_results"],
                candidate["query_results"],
            ),
        },
    }


def run_cold_index_benchmark(
    root: Path,
    output_path: Path,
    *,
    state_dir: Path,
    queries: tuple[str, ...] = DEFAULT_GOLDEN_QUERIES,
    performance_mode: bool = False,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output_path = Path(output_path).resolve()
    state_dir = Path(state_dir).resolve()
    if not root.is_dir():
        raise ValueError(f"测试目录不存在: {root}")
    if output_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖冷索引证据: {output_path}")
    state_dir.mkdir(parents=True, exist_ok=True)
    if APP_DATA_DIR.resolve() != state_dir:
        raise RuntimeError(
            "冷索引进程未隔离应用数据目录: "
            f"expected={state_dir}; actual={APP_DATA_DIR.resolve()}"
        )
    if DB_PATH.exists():
        raise FileExistsError(f"冷索引数据库已经存在: {DB_PATH}")
    for cold_area in ("cache", "temp"):
        area = state_dir / cold_area
        if area.exists() and any(area.rglob("*")):
            raise FileExistsError(
                f"冷索引缓存目录不是空目录: {area}"
            )

    before_manifest = _source_manifest(root)
    baseline_children = _child_pids()
    settings = AppSettings(
        monitor_file_changes=False,
        auto_scan_on_start=False,
        fast_ooxml_enabled=True,
        enable_parse_cache=True,
        defer_fts_during_full_scan=True,
    )
    hardware, root_disk_classes = detect_hardware_profile_for_roots([root])
    effective_profile: dict[str, object] = {
        "mode": "normal",
        "parser_workers": settings.parser_workers,
        "process_parser_workers": settings.process_parser_workers,
        "pdf_parser_workers": settings.pdf_parser_workers,
        "ocr_workers": settings.ocr_workers,
        "ocr_cpu_threads": settings.ocr_cpu_threads,
        "slow_file_workers": settings.slow_file_workers,
        "memory_budget_mb": settings.index_memory_budget_mb,
        "cpu_token_budget": settings.index_cpu_token_budget,
        "disk_class": hardware.disk_class,
    }
    if performance_mode:
        profile = build_performance_profile(hardware)
        effective_profile = profile.to_dict()
        settings = settings_for_profile(settings, profile)
    database = DatabaseManager(DB_PATH)
    database.initialize()
    root_id = database.add_root(root)
    progress_events: list[dict[str, Any]] = []
    last_progress = 0.0

    def progress(payload: dict[str, object]) -> None:
        nonlocal last_progress
        now = time.monotonic()
        diagnostic_state = str(payload.get("diagnostic_state") or "")
        if now - last_progress < 10.0 and not diagnostic_state:
            return
        last_progress = now
        event = {
            "elapsed_seconds": round(now - started, 3),
            "stage": str(payload.get("stage") or ""),
            "completed": int(
                payload.get("completed_files")
                or payload.get("indexed")
                or 0
            ),
            "total": int(
                payload.get("total_files")
                or payload.get("scanned")
                or 0
            ),
            "failed": int(payload.get("failed") or 0),
            "queue": str(payload.get("queue") or ""),
            "current_file": str(payload.get("current_file") or ""),
            "eta_seconds": int(payload.get("eta_seconds") or 0),
            "eta_ready": bool(payload.get("eta_ready")),
            "diagnostic_state": diagnostic_state,
        }
        progress_events.append(event)
        try:
            print(json.dumps(event, ensure_ascii=False), flush=True)
        except OSError:
            # Frozen windowed builds do not expose a writable stdout handle.
            pass

    started = time.monotonic()
    summary = IndexManager(
        database,
        settings,
        run_context={
            "execution_mode": "performance" if performance_mode else "normal",
            "validation_kind": "cold_real_corpus",
            "hardware": hardware.to_dict(),
            "root_disk_classes": root_disk_classes,
            "effective_profile": effective_profile,
        },
    ).index_root(root_id, progress_callback=progress)
    elapsed = time.monotonic() - started
    after_manifest = _source_manifest(root)
    _wait_for_children_to_exit(baseline_children, timeout_seconds=5.0)
    residual_pids = sorted(_child_pids() - baseline_children)
    readiness = database.index_readiness()
    integrity = database.integrity_report()
    snapshot = _database_snapshot(DB_PATH, list(queries))
    index_runs = database.recent_index_runs_since("")
    index_run = index_runs[-1] if index_runs else {}
    index_run_summary = index_run.get("summary")
    metrics = (
        index_run_summary.get("metrics", {})
        if isinstance(index_run_summary, dict)
        else {}
    )
    report: dict[str, Any] = {
        "passed": bool(
            summary.failed == 0
            and bool(readiness["ready"])
            and before_manifest == after_manifest
            and not residual_pids
            and snapshot["integrity_ok"]
        ),
        "root": str(root),
        "output": str(output_path),
        "state_dir": str(state_dir),
        "database": str(DB_PATH),
        "app_data_isolated": APP_DATA_DIR.resolve() == state_dir,
        "cold_cache_precondition": True,
        "performance_mode": performance_mode,
        "hardware": hardware.to_dict(),
        "root_disk_classes": root_disk_classes,
        "effective_profile": effective_profile,
        "elapsed_seconds": round(elapsed, 3),
        "summary": asdict(summary),
        "index_run": index_run,
        "metrics": metrics,
        "readiness": readiness,
        "settings": settings.to_dict(),
        "source_manifest_before": before_manifest,
        "source_manifest_after": after_manifest,
        "source_unchanged": before_manifest == after_manifest,
        "database_snapshot": snapshot,
        "failed_files": [
            dict(row) for row in database.failed_files(limit=10_000)
        ],
        "integrity": integrity,
        "progress_events": progress_events,
        "residual_child_pids": residual_pids,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return report


def _database_snapshot(
    database_path: Path,
    queries: list[str],
) -> dict[str, Any]:
    if not database_path.is_file():
        raise FileNotFoundError(database_path)
    connection = sqlite3.connect(
        f"file:{database_path.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    connection.row_factory = sqlite3.Row
    try:
        integrity = [
            str(row[0])
            for row in connection.execute("PRAGMA integrity_check")
        ]
        foreign_keys = [
            tuple(row)
            for row in connection.execute("PRAGMA foreign_key_check")
        ]
        rows = connection.execute(
            """
            SELECT f.path, f.extension, f.size_bytes, f.parse_status,
                   COALESCE(f.parse_error_code, '') AS parse_error_code,
                   f.document_id
            FROM files AS f
            WHERE f.is_deleted = 0
            ORDER BY f.path
            """
        ).fetchall()
        file_inventory: dict[str, dict[str, Any]] = {}
        for row in rows:
            document_id = row["document_id"]
            blocks = []
            if document_id is not None:
                blocks = [
                    {
                        "block_index": int(block["block_index"]),
                        "source_type": str(block["source_type"]),
                        "page_number": block["page_number"],
                        "slide_number": block["slide_number"],
                        "sheet_name": str(block["sheet_name"] or ""),
                        "location_text": str(block["location_text"] or ""),
                        "raw_text": str(block["raw_text"]),
                        "normalized_text": str(block["normalized_text"]),
                        "ocr_confidence": block["ocr_confidence"],
                    }
                    for block in connection.execute(
                        """
                        SELECT block_index, source_type, page_number,
                               slide_number, sheet_name, location_text,
                               raw_text, normalized_text, ocr_confidence
                        FROM content_blocks
                        WHERE document_id = ?
                        ORDER BY block_index
                        """,
                        (int(document_id),),
                    )
                ]
            block_digest = hashlib.sha256(
                _canonical_json(blocks).encode("utf-8")
            ).hexdigest()
            file_inventory[str(row["path"])] = {
                "extension": str(row["extension"] or ""),
                "size_bytes": int(row["size_bytes"] or 0),
                "parse_status": str(row["parse_status"]),
                "parse_error_code": str(row["parse_error_code"] or ""),
                "block_count": len(blocks),
                "block_digest": block_digest,
            }
        status_counts = {
            f"{row['extension'] or ''}|{row['parse_status']}": int(row["n"])
            for row in connection.execute(
                """
                SELECT extension, parse_status, COUNT(*) AS n
                FROM files
                WHERE is_deleted = 0
                GROUP BY extension, parse_status
                ORDER BY extension, parse_status
                """
            )
        }
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
    finally:
        connection.close()

    database = DatabaseManager(database_path)
    engine = SearchEngine(database)
    query_results: dict[str, list[dict[str, Any]]] = {}
    query_errors: dict[str, str] = {}
    for query in queries:
        try:
            page = engine.search(
                SearchQuery(
                    text=query,
                    mode="exact",
                    page_size=100,
                    max_results=100_000,
                )
            )
            query_results[query] = sorted(
                (
                    {
                        "file_path": result.file_path,
                        "location_text": result.location_text,
                        "source_type": result.source_type,
                        "hit_count": result.hit_count,
                    }
                    for result in page.results
                ),
                key=lambda item: (
                    item["file_path"],
                    item["location_text"],
                    item["source_type"],
                ),
            )
        except Exception as exc:
            query_results[query] = []
            query_errors[query] = str(exc)
    return {
        "database": str(database_path),
        "database_bytes": database_path.stat().st_size,
        "user_version": user_version,
        "integrity": integrity,
        "foreign_key_errors": foreign_keys,
        "integrity_ok": integrity == ["ok"] and not foreign_keys,
        "file_inventory": file_inventory,
        "file_count": len(file_inventory),
        "status_counts": status_counts,
        "content_digest": hashlib.sha256(
            _canonical_json(file_inventory).encode("utf-8")
        ).hexdigest(),
        "query_results": query_results,
        "query_errors": query_errors,
    }


def _load_queries(path: Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("queries")
    if not isinstance(payload, list):
        raise ValueError("查询文件必须是字符串数组或包含 queries 数组的对象")
    queries = [str(item).strip() for item in payload if str(item).strip()]
    if not queries:
        raise ValueError("查询文件不能为空")
    return list(dict.fromkeys(queries))


def _mapping_differences(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_keys = set(baseline)
    candidate_keys = set(candidate)
    changed = sorted(
        key
        for key in baseline_keys & candidate_keys
        if baseline[key] != candidate[key]
    )
    return {
        "missing_from_candidate": sorted(baseline_keys - candidate_keys),
        "new_in_candidate": sorted(candidate_keys - baseline_keys),
        "changed": changed,
    }


def _source_manifest(root: Path) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda item: str(item.relative_to(root)).lower(),
    ):
        stat = path.stat()
        total_bytes += int(stat.st_size)
        files.append(
            {
                "path": str(path.relative_to(root)),
                "size_bytes": int(stat.st_size),
                "modified_time_ns": int(stat.st_mtime_ns),
            }
        )
    return {
        "file_count": len(files),
        "total_bytes": total_bytes,
        "digest": hashlib.sha256(
            _canonical_json(files).encode("utf-8")
        ).hexdigest(),
    }


def _child_pids() -> set[int]:
    try:
        return {
            int(child.pid)
            for child in psutil.Process().children(recursive=True)
            if child.is_running()
        }
    except psutil.Error:
        return set()


def _wait_for_children_to_exit(
    baseline: set[int],
    *,
    timeout_seconds: float,
) -> None:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while time.monotonic() < deadline:
        if not (_child_pids() - baseline):
            return
        time.sleep(0.05)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
