from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.index_manager import (
    IndexManager,
    IndexSummary,
    ParseOutcome,
    parser_identity_for_path,
    parse_file_with_registry,
)
from local_full_text_search.core.pdf_task_graph import PdfTaskGraphRepository
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.index_metrics import IndexRunMetrics
from local_full_text_search.parsers.parser_registry import ParserRegistry


def validate(paths: list[Path], output_dir: Path) -> dict[str, object]:
    resolved = [path.resolve(strict=True) for path in paths]
    if len(resolved) != 2 or any(path.suffix.lower() != ".pdf" for path in resolved):
        raise ValueError("验证要求恰好两个 PDF 原始路径")
    output_dir.mkdir(parents=True, exist_ok=True)
    database_path = output_dir / "validation.db"
    if database_path.exists():
        raise FileExistsError(database_path)
    source_before = [_source_identity(path) for path in resolved]
    settings = AppSettings(
        enable_ocr=False,
        ocr_scanned_pdf=False,
        monitor_file_changes=False,
        auto_scan_on_start=False,
    )
    database = DatabaseManager(database_path)
    database.initialize()
    root_id = database.add_root(resolved[0].parent)
    run_id = "original-path-injected-failure"
    manager = IndexManager(database, settings)
    first_jobs = manager._prepare_jobs(
        root_id,
        resolved,
        run_id,
        IndexSummary(scanned=2),
        IndexRunMetrics(run_id=run_id),
        CancelToken(),
    )
    if not first_jobs:
        raise RuntimeError("原路径首次计划没有生成 PDF 页任务")
    document_task_id = int(first_jobs[0].pdf_document_task_id or 0)
    graph = PdfTaskGraphRepository(database)
    failed_task_ids = [int(job.task_id or 0) for job in first_jobs]
    for task_id in failed_task_ids:
        database.mark_task_failed(task_id, "FILE_IN_USE", "injected transient lock")
    manager.force_terminate_processes()

    retry_run_id = "original-path-retry"
    retried = IndexManager(database, settings)
    retry_jobs = retried._prepare_jobs(
        root_id,
        resolved,
        retry_run_id,
        IndexSummary(scanned=2),
        IndexRunMetrics(run_id=retry_run_id),
        CancelToken(),
    )
    retry_task_ids = [int(job.task_id or 0) for job in retry_jobs]
    registry = ParserRegistry(settings)
    merged: tuple[ParseOutcome, object] | None = None
    for index, job in enumerate(retry_jobs):
        job.pdf_confirmation_batch_end = index == len(retry_jobs) - 1
        database.mark_task_running(int(job.task_id or 0))
        outcome = parse_file_with_registry(
            job,
            registry,
            CancelToken(),
            settings,
        )
        merged = retried._record_pdf_page_outcome(job, outcome) or merged
    if merged is None:
        raise RuntimeError("PDF 合并没有返回文档结果")
    merged_outcome, parent_job = merged
    parser_name, parser_version = parser_identity_for_path(resolved[0], settings)
    with database.connect() as connection:
        task_rows = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, status, error_code, run_id
                FROM parse_tasks
                WHERE parent_task_id = ?
                  AND task_type IN ('pdf_native_page', 'pdf_ocr_page')
                ORDER BY id
                """,
                (document_task_id,),
            )
        ]
    retried.force_terminate_processes()
    source_after = [_source_identity(path) for path in resolved]
    source_unchanged = all(
        _stable_source_identity(before) == _stable_source_identity(after)
        for before, after in zip(source_before, source_after, strict=True)
    )
    result = {
        "passed": bool(
            source_unchanged
            and retry_task_ids == failed_task_ids
            and len(retry_jobs) == 13
            and len(merged_outcome.blocks) == 13
            and len(parent_job.alias_file_ids) == 1
            and all(row["status"] == "complete" for row in task_rows)
            and all(row["error_code"] is None for row in task_rows)
        ),
        "source_paths": [str(path) for path in resolved],
        "source_unchanged": source_unchanged,
        "source_identity": source_after,
        "same_content": source_after[0]["sha256"] == source_after[1]["sha256"],
        "parser": {"name": parser_name, "version": parser_version},
        "document_task_id": document_task_id,
        "failed_task_ids": failed_task_ids,
        "retry_task_ids": retry_task_ids,
        "reused_task_ids": retry_task_ids == failed_task_ids,
        "retried_pages": len(retry_jobs),
        "merged_blocks": len(merged_outcome.blocks),
        "alias_file_ids": list(parent_job.alias_file_ids),
        "task_rows": task_rows,
        "database": str(database_path),
    }
    report = output_dir / "result.json"
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _source_identity(path: Path) -> dict[str, object]:
    started = time.perf_counter()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
        "hash_seconds": round(time.perf_counter() - started, 3),
    }


def _stable_source_identity(identity: dict[str, object]) -> tuple[object, ...]:
    return tuple(
        identity[key] for key in ("path", "size", "mtime_ns", "sha256")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs=2, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = validate(args.paths, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
