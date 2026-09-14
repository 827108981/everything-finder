"""Atomic activity pointer and physical generation isolation."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from local_full_text_search.services.logging_service import log_event


logger = logging.getLogger(__name__)


MANIFEST_FORMAT_VERSION = 1


def _table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
            (str(name),),
        ).fetchone()
        is not None
    )


def _table_columns(connection: sqlite3.Connection, name: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f"PRAGMA table_info({name})").fetchall()
    }


def _is_compact_file(path: Path) -> bool:
    try:
        con = sqlite3.connect(f"file:{Path(path).resolve().as_posix()}?mode=ro", uri=True)
        try:
            return con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='view' AND name='search_blocks'"
            ).fetchone() is not None
        finally:
            con.close()
    except sqlite3.Error:
        return False


@dataclass(frozen=True, slots=True)
class PublishedGeneration:
    generation: int
    path: Path
    published_at: str
    file_count: int = 0
    document_count: int = 0
    block_count: int = 0
    partial: bool = False


@dataclass(frozen=True, slots=True)
class CandidateGeneration:
    generation: int
    path: Path
    baseline: PublishedGeneration | None
    scan_id: str = ""
    owner_nonce: str = ""


class GenerationLease:
    def __init__(self, registry: "GenerationLeaseRegistry", generation: int) -> None:
        self._registry = registry
        self.generation = int(generation)
        self._released = False

    def release(self) -> None:
        if not self._released:
            self._released = True
            self._registry.release(self.generation)

    def __enter__(self) -> "GenerationLease":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class GenerationLeaseRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counts: dict[int, int] = {}

    def acquire(self, generation: int) -> GenerationLease:
        with self._lock:
            key = int(generation)
            self._counts[key] = self._counts.get(key, 0) + 1
        return GenerationLease(self, int(generation))

    def release(self, generation: int) -> None:
        with self._lock:
            key = int(generation)
            count = self._counts.get(key, 0) - 1
            if count <= 0:
                self._counts.pop(key, None)
            else:
                self._counts[key] = count

    def active(self, generation: int) -> int:
        with self._lock:
            return int(self._counts.get(int(generation), 0))


# Search workers and the index worker create separate store objects, but they
# still share one Python process.  Keeping the registry on each object made a
# cleanup pass unable to see leases held by another worker and could remove a
# retired generation while a search was still reading it.
_SHARED_LEASES_LOCK = threading.Lock()
_SHARED_LEASES: dict[tuple[Path, str], GenerationLeaseRegistry] = {}


class IndexGenerationStore:
    """Keep the search reader on one immutable file while another is built."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path).resolve()
        self.root = self.database_path.parent
        self.manifest_path = self.root / f"{self.database_path.stem}.manifest.json"
        self.manifest_backup_path = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".bak"
        )
        self.failure_report_path = self.root / f"{self.database_path.stem}.last_failure.json"
        self.build_lock_path = self.root / f"{self.database_path.stem}.build.lock"
        self.retired_root = self.root / "retired"
        lease_key = (self.root, self.database_path.stem)
        with _SHARED_LEASES_LOCK:
            self._leases = _SHARED_LEASES.setdefault(
                lease_key,
                GenerationLeaseRegistry(),
            )

    @property
    def lease_registry(self) -> GenerationLeaseRegistry:
        return self._leases

    def active(self) -> PublishedGeneration | None:
        payload = self._read_manifest()
        if payload is None:
            return None
        try:
            generation = int(payload["active_generation"])
            path = (self.root / str(payload["active_file"])).resolve()
            if path.parent != self.root or not path.is_file() or path.stat().st_size <= 0:
                return None
            self._validate_read_only_database(path, generation)
            return PublishedGeneration(
                generation=generation,
                path=path,
                published_at=str(payload.get("published_at") or ""),
                file_count=int(payload.get("file_count") or 0),
                document_count=int(payload.get("document_count") or 0),
                block_count=int(payload.get("block_count") or 0),
                partial=bool(payload.get("partial", False)),
            )
        except (OSError, KeyError, TypeError, ValueError, sqlite3.Error):
            return None

    def building_candidate(self) -> CandidateGeneration | None:
        """Return the candidate currently protected by the build lock.

        The candidate is intentionally not considered searchable.  This
        read-only locator is used by the management page only, so it can show
        live parse failures while the last published generation remains the
        search snapshot.
        """

        if not self.build_lock_path.is_file():
            return None
        owner_pid = self._build_lock_pid()
        if owner_pid is None or not self._pid_is_alive(owner_pid):
            return None
        candidates: list[tuple[int, Path]] = []
        prefix = f"{self.database_path.stem}.gen-"
        suffix = ".building.db"
        for path in self.root.glob(f"{self.database_path.stem}.gen-*.building.db"):
            name = path.name
            if not name.startswith(prefix) or not name.endswith(suffix):
                continue
            raw_generation = name[len(prefix) : -len(suffix)]
            try:
                generation = int(raw_generation)
            except (TypeError, ValueError):
                continue
            try:
                if path.is_file() and path.stat().st_size > 0:
                    candidates.append((generation, path.resolve()))
            except OSError:
                # The scan worker may publish/abort the candidate between the
                # glob and stat calls. A disappearing candidate is simply not
                # a live snapshot source.
                continue
        if not candidates:
            return None
        generation, path = max(candidates, key=lambda item: item[0])
        return CandidateGeneration(
            generation,
            path,
            self.active(),
            self._build_lock_scan_id(),
            self._build_lock_owner_nonce(),
        )

    @contextmanager
    def acquire_active(self) -> Iterator[tuple[PublishedGeneration, GenerationLease]]:
        generation = self.active()
        if generation is None:
            raise FileNotFoundError("尚无完整发布的搜索数据库")
        lease = self._leases.acquire(generation.generation)
        try:
            yield generation, lease
        finally:
            lease.release()

    def write_failure_report(self, payload: dict[str, object]) -> None:
        """Persist the last candidate failure after its DB is removed."""

        temporary = self.failure_report_path.with_suffix(
            self.failure_report_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        os.replace(temporary, self.failure_report_path)

    def _read_failure_report(self) -> dict[str, object] | None:
        try:
            payload = json.loads(
                self.failure_report_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    def read_failure_report(self) -> dict[str, object] | None:
        """Read the durable report for the most recent candidate attempt."""

        payload = self._read_failure_report()
        return dict(payload) if payload is not None else None

    def mark_failure_report_items_excluded(
        self,
        file_ids: list[int] | tuple[int, ...] | set[int],
        *,
        reason: str,
        operation_source: str = "ui",
    ) -> dict[str, object]:
        """Move report-only blockers to the durable manual-exclusion list.

        A blocker from a discarded candidate has no row in the control DB by
        design.  Keeping its decision in the failure report is therefore the
        only way to make the UI action durable without re-introducing the
        discarded candidate's operational tables into the control database.
        """

        selected = {int(value) for value in file_ids}
        if not selected:
            return {
                "changed": 0,
                "remaining_blocking_files": 0,
                "remaining_blockers": [],
            }
        reason_text = str(reason).strip() or "用户确认当前文件暂时无法解析"
        payload = self._read_failure_report()
        if payload is None:
            raise FileNotFoundError("当前没有可处理的索引失败报告")
        blocking = [
            dict(row)
            for row in (payload.get("blocking_files") or [])
            if isinstance(row, dict)
        ]
        excluded = [
            dict(row)
            for row in (payload.get("excluded_files") or [])
            if isinstance(row, dict)
        ]
        now = datetime.now(timezone.utc).isoformat()
        changed_rows: list[dict[str, object]] = []
        remaining: list[dict[str, object]] = []
        for row in blocking:
            row_id = row.get("id")
            if row_id is not None and int(row_id) in selected:
                row["scope_state"] = "manual_excluded"
                row["exclusion_reason"] = reason_text
                row["exclusion_operation_source"] = (
                    str(operation_source).strip() or "ui"
                )
                row["excluded_at"] = now
                changed_rows.append(row)
            else:
                remaining.append(row)
        if not changed_rows:
            raise ValueError("所选报告项已变化或已经处理")

        excluded_ids = {
            int(row["id"])
            for row in excluded
            if row.get("id") is not None
        }
        excluded.extend(
            row for row in changed_rows
            if row.get("id") is None or int(row["id"]) not in excluded_ids
        )
        payload["blocking_files"] = remaining
        payload["excluded_files"] = excluded
        resolved_root_ids = {
            int(row["root_id"])
            for row in changed_rows
            if row.get("root_id") is not None
        }
        remaining_root_ids = {
            int(row["root_id"])
            for row in remaining
            if row.get("root_id") is not None
        }
        if resolved_root_ids:
            payload["unready_roots"] = [
                dict(root)
                for root in (payload.get("unready_roots") or [])
                if not isinstance(root, dict)
                or root.get("id") is None
                or int(root["id"]) not in resolved_root_ids
                or int(root["id"]) in remaining_root_ids
            ]
        payload["report_resolved_at"] = now if not remaining else payload.get(
            "report_resolved_at"
        )
        payload["report_resolution"] = (
            "manual_exclusion" if not remaining else "partially_manual_excluded"
        )

        readiness = payload.get("readiness")
        if isinstance(readiness, dict):
            readiness = dict(readiness)
            readiness["blocking_files"] = max(
                0,
                int(readiness.get("blocking_files") or 0) - len(changed_rows),
            )
            readiness["eligible_files"] = max(
                0,
                int(readiness.get("eligible_files") or 0) - len(changed_rows),
            )
            readiness["manual_excluded_files"] = int(
                readiness.get("manual_excluded_files") or 0
            ) + len(changed_rows)
            status_counts = dict(readiness.get("blocking_status_counts") or {})
            for row in changed_rows:
                status = str(row.get("parse_status") or "")
                if status in status_counts:
                    status_counts[status] = max(
                        0, int(status_counts.get(status) or 0) - 1
                    )
                    if status_counts[status] == 0:
                        status_counts.pop(status, None)
            readiness["blocking_status_counts"] = status_counts
            readiness["blocking_file_samples"] = remaining
            payload["readiness"] = readiness
        self.write_failure_report(payload)

        # Once every candidate blocker has been explicitly excluded, the
        # active compact snapshot covers the complete *search scope*.  Clear
        # only the manifest's partial marker; never modify the immutable DB.
        if not remaining and not (payload.get("unready_roots") or []):
            self._set_active_partial(False, payload.get("candidate_generation"))
        return {
            "changed": len(changed_rows),
            "changed_rows": changed_rows,
            "remaining_blocking_files": len(remaining),
            "remaining_blockers": remaining,
            "report_path": str(self.failure_report_path),
        }

    def restore_failure_report_items(
        self,
        file_ids: list[int] | tuple[int, ...] | set[int],
    ) -> dict[str, object]:
        """Restore report-only exclusions to the blocking list."""

        selected = {int(value) for value in file_ids}
        payload = self._read_failure_report()
        if payload is None:
            raise FileNotFoundError("当前没有可处理的索引失败报告")
        excluded = [
            dict(row)
            for row in (payload.get("excluded_files") or [])
            if isinstance(row, dict)
        ]
        blocking = [
            dict(row)
            for row in (payload.get("blocking_files") or [])
            if isinstance(row, dict)
        ]
        restored: list[dict[str, object]] = []
        remaining: list[dict[str, object]] = []
        for row in excluded:
            row_id = row.get("id")
            if row_id is not None and int(row_id) in selected:
                row.pop("scope_state", None)
                row.pop("exclusion_reason", None)
                row.pop("exclusion_operation_source", None)
                row.pop("excluded_at", None)
                restored.append(row)
            else:
                remaining.append(row)
        if not restored:
            raise ValueError("所选报告排除项已变化或已经恢复")
        blocking.extend(restored)
        payload["excluded_files"] = remaining
        payload["blocking_files"] = blocking
        payload.pop("report_resolved_at", None)
        payload["report_resolution"] = "restored_to_blocking"
        unready_roots = [
            dict(root)
            for root in (payload.get("unready_roots") or [])
            if isinstance(root, dict)
        ]
        known_root_ids = {
            int(root["id"])
            for root in unready_roots
            if root.get("id") is not None
        }
        now = datetime.now(timezone.utc).isoformat()
        for row in restored:
            root_id = row.get("root_id")
            if root_id is None or int(root_id) in known_root_ids:
                continue
            unready_roots.append(
                {
                    "id": int(root_id),
                    "path": row.get("root_path") or "",
                    "enabled": 1,
                    "status": "incomplete",
                    "last_scan_at": now,
                    "updated_at": now,
                }
            )
            known_root_ids.add(int(root_id))
        payload["unready_roots"] = unready_roots
        readiness = payload.get("readiness")
        if isinstance(readiness, dict):
            readiness = dict(readiness)
            readiness["blocking_files"] = int(readiness.get("blocking_files") or 0) + len(restored)
            readiness["eligible_files"] = int(
                readiness.get("eligible_files") or 0
            ) + len(restored)
            readiness["manual_excluded_files"] = max(
                0,
                int(readiness.get("manual_excluded_files") or 0) - len(restored),
            )
            status_counts = dict(readiness.get("blocking_status_counts") or {})
            for row in restored:
                status = str(row.get("parse_status") or "")
                status_counts[status] = int(status_counts.get(status) or 0) + 1
            readiness["blocking_status_counts"] = status_counts
            readiness["blocking_file_samples"] = blocking
            payload["readiness"] = readiness
        self.write_failure_report(payload)
        if restored:
            self._set_active_partial(True, payload.get("candidate_generation"))
        return {
            "changed": len(restored),
            "restored_rows": restored,
            "remaining_excluded_files": len(remaining),
            "report_path": str(self.failure_report_path),
        }

    def _set_active_partial(
        self,
        partial: bool,
        expected_generation: object = None,
    ) -> None:
        """Atomically update only the active manifest's partial flag."""

        payload = self._read_manifest()
        if payload is None:
            return
        if (
            expected_generation is not None
            and int(payload.get("active_generation") or 0)
            != int(expected_generation)
        ):
            return
        if bool(payload.get("partial", False)) == bool(partial):
            return
        payload["partial"] = bool(partial)
        temporary = self.manifest_path.with_suffix(
            self.manifest_path.suffix + ".tmp"
        )
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, self.manifest_path)
        self.manifest_backup_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear_failure_report(self) -> None:
        self.failure_report_path.unlink(missing_ok=True)

    def _log_candidate_event(
        self,
        event: str,
        *,
        scan_id: str = "",
        **fields: object,
    ) -> None:
        try:
            log_event(
                event,
                component="index_generation_store",
                scan_id=scan_id or None,
                database_path=str(self.database_path),
                **fields,
            )
        except BaseException:
            logger.debug("Unable to write candidate lifecycle event", exc_info=True)

    def begin_candidate(
        self,
        *,
        force: bool = False,
        scan_id: str = "",
    ) -> CandidateGeneration:
        self.root.mkdir(parents=True, exist_ok=True)
        baseline = self.active()
        baseline_compact = bool(baseline and _is_compact_file(baseline.path))
        self._log_candidate_event(
            "candidate.begin.started",
            scan_id=scan_id,
            baseline_generation=baseline.generation if baseline else 0,
            baseline_path=str(baseline.path) if baseline else "",
            baseline_compact=baseline_compact,
            baseline_bytes=(baseline.path.stat().st_size if baseline else 0),
            force=bool(force),
        )
        if self.build_lock_path.exists() and not force:
            owner_pid = self._build_lock_pid()
            if owner_pid is not None and self._pid_is_alive(owner_pid):
                self._log_candidate_event(
                    "candidate.begin.blocked",
                    scan_id=scan_id,
                    owner_pid=owner_pid,
                    reason="live_build_lock",
                )
                raise RuntimeError("已有索引候选正在构建")
            stale_lock_payload = self._read_build_lock_payload() or {}
            worker_pid = self._int_field(stale_lock_payload.get("worker_pid"))
            if worker_pid and self._pid_is_alive(worker_pid):
                self._log_candidate_event(
                    "candidate.recovery.worker_terminate_requested",
                    scan_id=scan_id,
                    owner_pid=owner_pid,
                    worker_pid=worker_pid,
                    reason="owner_process_gone_worker_still_alive",
                )
                still_alive = self._terminate_pid(worker_pid)
                if still_alive:
                    self._log_candidate_event(
                        "candidate.recovery.worker_terminate_failed",
                        scan_id=scan_id,
                        owner_pid=owner_pid,
                        worker_pid=worker_pid,
                        reason="worker_still_alive",
                    )
                    raise RuntimeError(
                        "上一次候选扩展子进程仍在运行，未删除候选库；请稍后重试"
                    )
            # A dead owner cannot publish its candidate. Recover only the
            # exact build artifacts belonging to this store, then start a
            # fresh candidate with the last published generation as baseline.
            for stale in self.root.glob(f"{self.database_path.stem}.gen-*.building.db*"):
                try:
                    self._log_candidate_event(
                        "candidate.recovery.artifact_removed",
                        scan_id=scan_id,
                        owner_pid=owner_pid,
                        path=str(stale),
                        size_bytes=stale.stat().st_size if stale.is_file() else 0,
                    )
                    stale.unlink()
                except OSError:
                    pass
            self.build_lock_path.unlink(missing_ok=True)
            self._log_candidate_event(
                "candidate.recovery.stale_lock_cleared",
                scan_id=scan_id,
                owner_pid=owner_pid,
                lock_path=str(self.build_lock_path),
            )
        owner_nonce = uuid.uuid4().hex
        owner_pid = os.getpid()
        lock_payload = json.dumps(
            {
                "pid": owner_pid,
                "owner_nonce": owner_nonce,
                "created_at": time.time(),
                "baseline_generation": baseline.generation if baseline else 0,
                "scan_id": str(scan_id or ""),
                "parent_pid": owner_pid,
            }
        )
        try:
            with self.build_lock_path.open("x", encoding="utf-8") as lock:
                lock.write(lock_payload)
        except FileExistsError as exc:
            raise RuntimeError("已有索引候选正在构建") from exc
        generation = (baseline.generation + 1) if baseline else 1
        candidate = self.root / (
            f"{self.database_path.stem}.gen-{generation:06d}.building.db"
        )
        candidate.unlink(missing_ok=True)
        for sidecar in (
            Path(str(candidate) + "-wal"),
            Path(str(candidate) + "-shm"),
            Path(str(candidate) + "-journal"),
        ):
            sidecar.unlink(missing_ok=True)
        try:
            if baseline is not None:
                if baseline_compact:
                    from local_full_text_search.core.compact_snapshot_builder import (
                        expand_compact_snapshot_isolated,
                    )

                    attempt_id = f"candidate-expand-{os.getpid()}-{time.time_ns()}"

                    def report_progress(payload: dict[str, object]) -> None:
                        progress_payload = dict(payload)
                        progress_payload.pop("scan_id", None)
                        progress_payload.pop("attempt_id", None)
                        if progress_payload.get("phase") == "worker_start":
                            worker_pid = progress_payload.get("worker_pid")
                            if worker_pid:
                                self._update_build_lock_worker(
                                    owner_nonce=owner_nonce,
                                    worker_pid=int(worker_pid),
                                )
                        self._log_candidate_event(
                            "candidate.expand.progress",
                            scan_id=scan_id,
                            attempt_id=attempt_id,
                            candidate_generation=generation,
                            candidate_path=str(candidate),
                            **progress_payload,
                        )

                    timeout_seconds = float(
                        os.environ.get("LFTS_CANDIDATE_EXPAND_TIMEOUT_SECONDS", "900")
                    )
                    stall_seconds = float(
                        os.environ.get("LFTS_CANDIDATE_EXPAND_STALL_SECONDS", "180")
                    )
                    self._log_candidate_event(
                        "candidate.expand.worker_started",
                        scan_id=scan_id,
                        attempt_id=attempt_id,
                        candidate_generation=generation,
                        source_path=str(baseline.path),
                        target_path=str(candidate),
                        owner_pid=owner_pid,
                        owner_nonce=owner_nonce,
                        parent_pid=owner_pid,
                        timeout_seconds=timeout_seconds,
                        stall_seconds=stall_seconds,
                    )
                    expand_compact_snapshot_isolated(
                        baseline.path,
                        candidate,
                        scan_id=scan_id,
                        attempt_id=attempt_id,
                        progress_callback=report_progress,
                        timeout_seconds=timeout_seconds,
                        stall_seconds=stall_seconds,
                        parent_pid=owner_pid,
                    )
                    self._log_candidate_event(
                        "candidate.expand.worker_completed",
                        scan_id=scan_id,
                        attempt_id=attempt_id,
                        candidate_generation=generation,
                        target_path=str(candidate),
                        owner_pid=owner_pid,
                        target_bytes=candidate.stat().st_size if candidate.exists() else 0,
                    )
                else:
                    shutil.copy2(baseline.path, candidate)
                    self._log_candidate_event(
                        "candidate.copy.completed",
                        scan_id=scan_id,
                        candidate_generation=generation,
                        source_path=str(baseline.path),
                        target_path=str(candidate),
                        target_bytes=candidate.stat().st_size if candidate.exists() else 0,
                    )
            self._log_candidate_event(
                "candidate.begin.completed",
                scan_id=scan_id,
                candidate_generation=generation,
                candidate_path=str(candidate),
                candidate_bytes=candidate.stat().st_size if candidate.exists() else 0,
            )
        except BaseException as exc:
            self._log_candidate_event(
                "candidate.begin.failed",
                scan_id=scan_id,
                candidate_generation=generation,
                candidate_path=str(candidate),
                error_type=type(exc).__name__,
                error_message=str(exc)[:2000],
            )
            self.abort_candidate(
                CandidateGeneration(
                    generation,
                    candidate,
                    baseline,
                    str(scan_id or ""),
                    owner_nonce,
                )
            )
            raise
        return CandidateGeneration(
            generation,
            candidate,
            baseline,
            str(scan_id or ""),
            owner_nonce,
        )

    def _build_lock_pid(self) -> int | None:
        payload = self._read_build_lock_payload()
        try:
            return int(payload.get("pid")) if payload else None
        except (TypeError, ValueError):
            return None

    def _build_lock_scan_id(self) -> str:
        payload = self._read_build_lock_payload()
        return str(payload.get("scan_id") or "") if payload else ""

    def _build_lock_owner_nonce(self) -> str:
        payload = self._read_build_lock_payload()
        return str(payload.get("owner_nonce") or "") if payload else ""

    def _read_build_lock_payload(self) -> dict[str, object] | None:
        try:
            payload = json.loads(self.build_lock_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _int_field(value: object) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _update_build_lock_worker(self, *, owner_nonce: str, worker_pid: int) -> None:
        """Atomically attach the isolated expansion PID to the owner lock."""

        payload = self._read_build_lock_payload()
        if not payload or str(payload.get("owner_nonce") or "") != owner_nonce:
            return
        payload["worker_pid"] = int(worker_pid)
        try:
            import psutil

            payload["worker_created_at"] = psutil.Process(int(worker_pid)).create_time()
        except Exception:
            payload["worker_created_at"] = None
        temporary = self.build_lock_path.with_suffix(self.build_lock_path.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(temporary, self.build_lock_path)
        except OSError:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _terminate_pid(pid: int) -> bool:
        """Terminate an orphaned worker and report whether it remains alive."""

        try:
            import psutil
        except Exception:
            # Do not delete a candidate when the worker state is uncertain.
            return True
        try:
            process = psutil.Process(int(pid))
            process.terminate()
            try:
                process.wait(timeout=5)
            except psutil.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=5)
                except psutil.TimeoutExpired:
                    pass
            return bool(process.is_running())
        except psutil.NoSuchProcess:
            return False
        except Exception:
            # Do not delete a candidate when the worker state is uncertain.
            return True

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if int(pid) <= 0:
            return False
        try:
            import psutil

            return bool(psutil.pid_exists(int(pid)))
        except Exception:
            # A missing process library should not make a live lock appear
            # stale.  The lock owner will be checked again before publish.
            return True
            return False

    def publish(
        self,
        candidate: CandidateGeneration,
        *,
        file_count: int,
        document_count: int,
        block_count: int,
        published_at: str,
        partial: bool = False,
    ) -> PublishedGeneration:
        if candidate.owner_nonce:
            lock_payload = self._read_build_lock_payload()
            lock_pid = self._int_field(lock_payload.get("pid")) if lock_payload else None
            worker_pid = (
                self._int_field(lock_payload.get("worker_pid"))
                if lock_payload
                else None
            )
            if (
                not lock_payload
                or str(lock_payload.get("owner_nonce") or "") != candidate.owner_nonce
                or lock_pid != os.getpid()
                or str(lock_payload.get("scan_id") or "") != str(candidate.scan_id or "")
            ):
                self._log_candidate_event(
                    "candidate.publish.blocked",
                    scan_id=candidate.scan_id,
                    candidate_generation=candidate.generation,
                    candidate_path=str(candidate.path),
                    reason="candidate_owner_mismatch",
                    lock_pid=lock_pid,
                    lock_scan_id=lock_payload.get("scan_id") if lock_payload else None,
                )
                raise RuntimeError("候选库所有权已失效，禁止发布残留候选")
            if worker_pid and self._pid_is_alive(worker_pid):
                self._log_candidate_event(
                    "candidate.publish.blocked",
                    scan_id=candidate.scan_id,
                    candidate_generation=candidate.generation,
                    candidate_path=str(candidate.path),
                    reason="expansion_worker_still_alive",
                    worker_pid=worker_pid,
                )
                raise RuntimeError("候选扩展子进程仍在运行，禁止发布")
        path = Path(candidate.path).resolve()
        if path.parent != self.root or not path.is_file():
            raise FileNotFoundError("候选数据库不存在")
        source_path = path
        compact_path = self.root / (
            f"{self.database_path.stem}.gen-{candidate.generation:06d}.compact.db"
        )
        try:
            if _is_compact_file(path):
                self._seal_candidate_database(
                    path,
                    generation=candidate.generation,
                    file_count=file_count,
                    document_count=document_count,
                    block_count=block_count,
                    published_at=published_at,
                )
            else:
                from local_full_text_search.core.compact_snapshot_builder import (
                    build_compact_snapshot,
                )

                counts = build_compact_snapshot(
                    path,
                    compact_path,
                    generation=candidate.generation,
                )
                file_count = int(counts["file_count"])
                document_count = int(counts["document_count"])
                block_count = int(counts["block_count"])
                path = compact_path
            self._validate_read_only_database(
                path,
                candidate.generation,
                require_published=True,
                full_check=True,
            )
        except Exception:
            compact_path.unlink(missing_ok=True)
            raise
        final_path = self.root / f"{self.database_path.stem}.gen-{candidate.generation:06d}.db"
        os.replace(path, final_path)
        if source_path != final_path:
            source_path.unlink(missing_ok=True)
        payload = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "active_generation": candidate.generation,
            "active_file": final_path.name,
            "previous_file": candidate.baseline.path.name if candidate.baseline else "",
            "published_at": published_at,
            "file_count": int(file_count),
            "document_count": int(document_count),
            "block_count": int(block_count),
            "partial": bool(partial),
        }
        temporary = self.manifest_path.with_suffix(self.manifest_path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.manifest_path)
        self.manifest_backup_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.build_lock_path.unlink(missing_ok=True)
        return PublishedGeneration(
            candidate.generation,
            final_path,
            published_at,
            int(file_count),
            int(document_count),
            int(block_count),
            bool(partial),
        )

    @staticmethod
    def _seal_candidate_database(
        path: Path,
        *,
        generation: int,
        file_count: int,
        document_count: int,
        block_count: int,
        published_at: str,
    ) -> None:
        """Seal a candidate after all normal writer connections are closed.

        The current indexer still uses its operational schema.  The marker is
        deliberately tiny and lets the generation store provide the same
        atomic-publish contract while the compact projection is introduced.
        """

        con = sqlite3.connect(path, timeout=2.0)
        try:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    generation INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL DEFAULT 1,
                    published_at TEXT NOT NULL,
                    root_count INTEGER NOT NULL DEFAULT 0,
                    file_count INTEGER NOT NULL,
                    document_count INTEGER NOT NULL,
                    block_count INTEGER NOT NULL,
                    parser_version TEXT NOT NULL DEFAULT '',
                    state INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            root_count = 0
            if _table_exists(con, "roots"):
                root_columns = _table_columns(con, "roots")
                root_sql = (
                    "SELECT COUNT(*) FROM roots WHERE enabled = 1"
                    if "enabled" in root_columns
                    else "SELECT COUNT(*) FROM roots"
                )
                root_count = int(con.execute(root_sql).fetchone()[0])
            con.execute(
                """
                INSERT INTO snapshot_meta(
                    id, generation, schema_version, published_at,
                    root_count, file_count, document_count, block_count,
                    parser_version, state
                ) VALUES (1, ?, 1, ?, ?, ?, ?, ?, '', 1)
                ON CONFLICT(id) DO UPDATE SET
                    generation = excluded.generation,
                    published_at = excluded.published_at,
                    root_count = excluded.root_count,
                    file_count = excluded.file_count,
                    document_count = excluded.document_count,
                    block_count = excluded.block_count,
                    state = 1
                """,
                (
                    int(generation),
                    str(published_at),
                    root_count,
                    int(file_count),
                    int(document_count),
                    int(block_count),
                ),
            )
            con.commit()
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.execute("PRAGMA journal_mode=DELETE")
            con.commit()
        finally:
            con.close()

    def abort_candidate(self, candidate: CandidateGeneration) -> None:
        lock_payload = self._read_build_lock_payload()
        if (
            candidate.owner_nonce
            and lock_payload
            and str(lock_payload.get("owner_nonce") or "") == candidate.owner_nonce
        ):
            worker_pid = self._int_field(lock_payload.get("worker_pid"))
            if worker_pid and self._pid_is_alive(worker_pid):
                self._log_candidate_event(
                    "candidate.abort.worker_terminate_requested",
                    scan_id=candidate.scan_id,
                    candidate_generation=candidate.generation,
                    worker_pid=worker_pid,
                )
                if self._terminate_pid(worker_pid):
                    self._log_candidate_event(
                        "candidate.abort.worker_terminate_failed",
                        scan_id=candidate.scan_id,
                        candidate_generation=candidate.generation,
                        worker_pid=worker_pid,
                    )
        for path in (
            candidate.path,
            candidate.path.with_name(candidate.path.name + ".expanded"),
            Path(str(candidate.path) + "-wal"),
            Path(str(candidate.path) + "-shm"),
            Path(str(candidate.path) + "-journal"),
        ):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        self.build_lock_path.unlink(missing_ok=True)

    def cleanup_retired(self, *, keep_generations: int = 1) -> list[str]:
        active = self.active()
        active_generation = active.generation if active else None
        removed: list[str] = []
        for path in self.root.glob(f"{self.database_path.stem}.gen-*.db"):
            try:
                generation = int(path.stem.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                continue
            if generation == active_generation or generation >= (active_generation or 0) - keep_generations:
                continue
            if self._leases.active(generation):
                continue
            try:
                path.unlink()
                removed.append(str(path))
            except OSError:
                continue
        return removed

    def _read_manifest(self) -> dict[str, object] | None:
        for path in (self.manifest_path, self.manifest_backup_path):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload.get("format_version") or 0) != MANIFEST_FORMAT_VERSION:
                    continue
                return payload
            except (OSError, ValueError, TypeError):
                continue
        return None

    @staticmethod
    def _validate_read_only_database(
        path: Path,
        generation: int,
        *,
        require_published: bool = False,
        full_check: bool = False,
    ) -> None:
        con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, timeout=2)
        try:
            row = con.execute(
                "SELECT generation, state FROM snapshot_meta WHERE id = 1"
            ).fetchone()
            if row is None or int(row[0]) != int(generation):
                raise sqlite3.DatabaseError("快照 generation 不一致")
            if require_published and int(row[1]) != 1:
                raise sqlite3.DatabaseError("候选快照尚未封存")
            if full_check:
                check = con.execute("PRAGMA quick_check").fetchone()
                if check is None or str(check[0]).lower() != "ok":
                    raise sqlite3.DatabaseError("快照 quick_check 失败")
        finally:
            con.close()
