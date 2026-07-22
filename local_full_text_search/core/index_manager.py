from __future__ import annotations

import logging
import threading
import traceback
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from local_full_text_search.config.constants import ARCHIVE_EXTENSIONS, IMAGE_EXTENSIONS, LEGACY_OFFICE_EXTENSIONS
from local_full_text_search.config.defaults import AppSettings
from local_full_text_search.core.database import DatabaseManager
from local_full_text_search.core.errors import (
    CancelledError,
    ParserDependencyError,
    PasswordProtectedError,
    UnsupportedFormatError,
)
from local_full_text_search.core.normalizer import normalize_text
from local_full_text_search.core.scanner import iter_files
from local_full_text_search.core.task_manager import CancelToken
from local_full_text_search.models.content_block import ContentBlock
from local_full_text_search.parsers.parser_registry import ParserRegistry

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, object]], None]


@dataclass(slots=True)
class IndexSummary:
    scanned: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    unsupported: int = 0
    metadata_only: int = 0
    partial_success: int = 0
    deleted: int = 0
    cancelled: bool = False


@dataclass(slots=True)
class ParseJob:
    file_id: int
    file_path: Path


@dataclass(slots=True)
class ParseOutcome:
    file_id: int
    file_path: Path
    blocks: list[ContentBlock]
    parser_name: str
    status: str
    error_code: str | None = None
    error_message: str | None = None


_worker_state = threading.local()


class IndexManager:
    def __init__(self, db: DatabaseManager, settings: AppSettings) -> None:
        self.db = db
        self.settings = settings

    def index_enabled_roots(
        self,
        cancel_token: CancelToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexSummary:
        token = cancel_token or CancelToken()
        summary = IndexSummary()
        for root in self.db.list_roots(enabled_only=True):
            token.wait_if_paused()
            token.throw_if_cancelled()
            root_path = Path(str(root["path"]))
            if not root_path.exists():
                logger.warning("Root does not exist: %s", root_path)
                continue
            root_summary = self.index_root(int(root["id"]), token, progress_callback)
            merge_summary(summary, root_summary)
        return summary

    def index_root(
        self,
        root_id: int,
        cancel_token: CancelToken | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexSummary:
        token = cancel_token or CancelToken()
        root = next((row for row in self.db.list_roots() if int(row["id"]) == root_id), None)
        if root is None:
            raise ValueError(f"根目录不存在: {root_id}")

        root_path = Path(str(root["path"]))
        include_subfolders = bool(root["include_subfolders"])
        summary = IndexSummary()
        previous_paths = self.db.active_paths_for_root(root_id)
        seen_paths: set[str] = set()
        futures: set[Future[ParseOutcome]] = set()
        write_buffer: list[ParseOutcome] = []

        normal_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.settings.parser_workers or 1)),
            thread_name_prefix="lfts-parser",
        )
        ocr_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.settings.ocr_workers or 1)),
            thread_name_prefix="lfts-ocr",
        )
        slow_executor = ThreadPoolExecutor(
            max_workers=max(1, int(self.settings.slow_file_workers or 1)),
            thread_name_prefix="lfts-slow",
        )

        try:
            for file_path in iter_files(
                root_path,
                include_subfolders=include_subfolders,
                settings=self.settings,
                cancel_token=token,
            ):
                token.wait_if_paused()
                token.throw_if_cancelled()
                summary.scanned += 1
                seen_paths.add(str(file_path))
                self._emit(progress_callback, "scanning", summary, current_file=str(file_path), pending=len(futures))
                try:
                    file_id, changed = self.db.upsert_file_metadata(root_id, file_path)
                    if not changed:
                        summary.skipped += 1
                        drain_completed_futures(futures, write_buffer, summary, self, token, progress_callback)
                        continue
                    self.db.mark_processing(file_id)
                    executor = executor_for(file_path, self.settings, normal_executor, ocr_executor, slow_executor)
                    futures.add(executor.submit(parse_file_worker, ParseJob(file_id, file_path), self.settings, token))
                    while len(futures) >= max(1, int(self.settings.max_pending_parse_tasks or 1)):
                        drain_completed_futures(futures, write_buffer, summary, self, token, progress_callback, block=True)
                    drain_completed_futures(futures, write_buffer, summary, self, token, progress_callback)
                except CancelledError:
                    summary.cancelled = True
                    raise
                except Exception as exc:
                    logger.error("Failed to queue %s\n%s", file_path, traceback.format_exc())
                    outcome = self._failure_outcome_for_path(root_id, file_path, exc)
                    write_buffer.append(outcome)
                    record_parse_outcome(summary, outcome.status)
                    self._flush_outcomes(write_buffer)
                    self._emit(progress_callback, "indexing", summary, current_file=str(file_path), pending=len(futures))

            while futures:
                drain_completed_futures(futures, write_buffer, summary, self, token, progress_callback, block=True)
            self._flush_outcomes(write_buffer)

            missing_paths = previous_paths - seen_paths
            summary.deleted += self.db.mark_deleted_paths(missing_paths)
            self.db.update_root_scan_time(root_id, "ready")
            self._emit(progress_callback, "finished", summary, current_file="", pending=0)
        except CancelledError:
            summary.cancelled = True
            for future in futures:
                future.cancel()
            self._flush_outcomes(write_buffer)
            self.db.update_root_scan_time(root_id, "cancelled")
            self._emit(progress_callback, "cancelled", summary, current_file="", pending=len(futures))
        finally:
            for executor in (normal_executor, ocr_executor, slow_executor):
                executor.shutdown(wait=True, cancel_futures=True)
        return summary

    def _flush_outcomes(self, write_buffer: list[ParseOutcome]) -> None:
        if not write_buffer:
            return
        batch = [
            (
                outcome.file_id,
                outcome.file_path.name,
                str(outcome.file_path),
                outcome.blocks,
                outcome.parser_name,
                outcome.status,
                outcome.error_code,
                outcome.error_message,
            )
            for outcome in write_buffer
        ]
        self.db.replace_file_blocks_many(batch)
        write_buffer.clear()

    def _failure_outcome_for_path(self, root_id: int, file_path: Path, exc: Exception) -> ParseOutcome:
        try:
            file_id, _ = self.db.upsert_file_metadata(root_id, file_path)
        except Exception:
            raise exc
        retryable = isinstance(exc, ParserDependencyError)
        status = "failed_retryable" if retryable else "failed"
        return ParseOutcome(
            file_id=file_id,
            file_path=file_path,
            blocks=[metadata_block(file_path)],
            parser_name=exc.__class__.__name__,
            status=status,
            error_code=error_code_for_exception(exc),
            error_message=user_message_for_exception(exc),
        )

    def _emit(
        self,
        progress_callback: ProgressCallback | None,
        stage: str,
        summary: IndexSummary,
        **extra: object,
    ) -> None:
        if progress_callback is None:
            return
        payload = {
            "stage": stage,
            "scanned": summary.scanned,
            "indexed": summary.indexed,
            "skipped": summary.skipped,
            "failed": summary.failed,
            "unsupported": summary.unsupported,
            "metadata_only": summary.metadata_only,
            "partial_success": summary.partial_success,
            "deleted": summary.deleted,
            "cancelled": summary.cancelled,
        }
        payload.update(extra)
        progress_callback(payload)


def drain_completed_futures(
    futures: set[Future[ParseOutcome]],
    write_buffer: list[ParseOutcome],
    summary: IndexSummary,
    manager: IndexManager,
    token: CancelToken,
    progress_callback: ProgressCallback | None,
    *,
    block: bool = False,
) -> None:
    if not futures:
        return
    if block:
        done, pending = wait(futures, return_when=FIRST_COMPLETED)
        futures.clear()
        futures.update(pending)
    else:
        done = {future for future in futures if future.done()}
        futures.difference_update(done)
    for future in done:
        token.wait_if_paused()
        token.throw_if_cancelled()
        outcome = future.result()
        write_buffer.append(outcome)
        record_parse_outcome(summary, outcome.status)
        if len(write_buffer) >= max(1, int(manager.settings.index_write_batch_size or 1)):
            manager._flush_outcomes(write_buffer)
        manager._emit(
            progress_callback,
            "indexing",
            summary,
            current_file=str(outcome.file_path),
            pending=len(futures),
        )


def parse_file_worker(job: ParseJob, settings: AppSettings, cancel_token: CancelToken) -> ParseOutcome:
    try:
        cancel_token.wait_if_paused()
        cancel_token.throw_if_cancelled()
        registry = worker_registry(settings)
        parser = registry.parser_for(job.file_path)
        parser.reset_status()
        blocks: list[ContentBlock] = [metadata_block(job.file_path)]
        for block in parser.parse(job.file_path, cancel_token):
            cancel_token.wait_if_paused()
            cancel_token.throw_if_cancelled()
            if block.raw_text.strip():
                block.block_index = len(blocks)
                blocks.append(block)
        status = parser.last_status or "success"
        return ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=blocks,
            parser_name=parser.name,
            status=status,
            error_code=(parser.last_error_code if status != "success" else None),
            error_message=(parser.last_error_message if status != "success" else None),
        )
    except UnsupportedFormatError as exc:
        return ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=[metadata_block(job.file_path)],
            parser_name="metadata",
            status="unsupported",
            error_code="UNSUPPORTED_FORMAT",
            error_message=str(exc),
        )
    except CancelledError:
        raise
    except Exception as exc:
        logger.error("Failed to parse %s\n%s", job.file_path, traceback.format_exc())
        retryable = isinstance(exc, ParserDependencyError)
        return ParseOutcome(
            file_id=job.file_id,
            file_path=job.file_path,
            blocks=[metadata_block(job.file_path)],
            parser_name=exc.__class__.__name__,
            status="failed_retryable" if retryable else "failed",
            error_code=error_code_for_exception(exc),
            error_message=user_message_for_exception(exc),
        )


def worker_registry(settings: AppSettings) -> ParserRegistry:
    settings_id = id(settings)
    if getattr(_worker_state, "settings_id", None) != settings_id:
        _worker_state.settings_id = settings_id
        _worker_state.registry = ParserRegistry(settings)
    return _worker_state.registry


def executor_for(
    file_path: Path,
    settings: AppSettings,
    normal_executor: ThreadPoolExecutor,
    ocr_executor: ThreadPoolExecutor,
    slow_executor: ThreadPoolExecutor,
) -> ThreadPoolExecutor:
    suffix = file_path.suffix.lower()
    if settings.enable_ocr and settings.ocr_images and suffix in IMAGE_EXTENSIONS:
        return ocr_executor
    if suffix in ARCHIVE_EXTENSIONS or suffix in LEGACY_OFFICE_EXTENSIONS:
        return slow_executor
    return normal_executor


def metadata_block(file_path: Path) -> ContentBlock:
    raw = f"{file_path.name}\n{file_path}"
    return ContentBlock(
        file_path=str(file_path),
        block_index=0,
        block_type="metadata",
        location_text="文件名/路径",
        raw_text=raw,
        normalized_text=normalize_text(raw),
        source_type="metadata",
    )


def merge_summary(target: IndexSummary, source: IndexSummary) -> None:
    target.scanned += source.scanned
    target.indexed += source.indexed
    target.skipped += source.skipped
    target.failed += source.failed
    target.unsupported += source.unsupported
    target.metadata_only += source.metadata_only
    target.partial_success += source.partial_success
    target.deleted += source.deleted
    target.cancelled = target.cancelled or source.cancelled


def record_parse_outcome(summary: IndexSummary, status: str) -> None:
    if status in {"failed", "failed_retryable", "password_protected"}:
        summary.failed += 1
        return
    if status == "unsupported":
        summary.unsupported += 1
        return
    if status == "skipped":
        summary.skipped += 1
        return
    summary.indexed += 1
    record_index_status(summary, status)


def record_index_status(summary: IndexSummary, status: str) -> None:
    """Reflect parser diagnostic statuses in the scan summary."""

    if status in {"metadata_only", "ocr_disabled", "converter_missing"}:
        summary.metadata_only += 1
    elif status == "partial_success":
        summary.partial_success += 1


def error_code_for_exception(exc: Exception) -> str:
    if isinstance(exc, ParserDependencyError):
        return "PARSER_DEPENDENCY_MISSING"
    if isinstance(exc, PasswordProtectedError):
        return "PASSWORD_PROTECTED"
    if isinstance(exc, PermissionError):
        return "PERMISSION_DENIED"
    if isinstance(exc, FileNotFoundError):
        return "FILE_NOT_FOUND"
    if isinstance(exc, OSError):
        return "FILE_IN_USE"
    return "PARSER_ERROR"


def user_message_for_exception(exc: Exception) -> str:
    if isinstance(exc, ParserDependencyError):
        return str(exc)
    if isinstance(exc, PasswordProtectedError):
        return "文件已加密，需要密码"
    if isinstance(exc, PermissionError):
        return "没有权限读取该文件"
    if isinstance(exc, FileNotFoundError):
        return "文件不存在或已被移动"
    return str(exc) or exc.__class__.__name__
