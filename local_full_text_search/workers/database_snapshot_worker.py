from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from local_full_text_search.core.database import DatabaseManager


logger = logging.getLogger(__name__)


def collect_database_snapshot(
    db: DatabaseManager,
    *,
    page_size: int,
    page_offsets: dict[str, int],
) -> dict[str, object]:
    """Collect the expensive management-page state away from the UI thread."""

    timings: dict[str, int] = {}

    def measured(name: str, operation: object) -> object:
        started = time.perf_counter()
        value = operation()  # type: ignore[operator]
        timings[name] = int((time.perf_counter() - started) * 1000)
        return value

    roots = [
        dict(row)
        for row in measured("roots", db.list_roots)  # type: ignore[arg-type]
    ]
    readiness = dict(
        measured("readiness", db.index_readiness)  # type: ignore[arg-type]
    )

    def root_stats() -> dict[int, dict[str, int]]:
        data: dict[int, dict[str, int]] = {}
        with db.connect() as connection:
            rows = connection.execute(
                """
                SELECT root_id, extension, COUNT(*) AS n
                FROM files
                WHERE is_deleted = 0
                GROUP BY root_id, extension
                """
            ).fetchall()
        for row in rows:
            root_id = int(row["root_id"])
            extension = str(row["extension"] or "")
            data.setdefault(root_id, {})[extension] = int(row["n"])
        return data

    root_stats_by_id = measured("root_stats", root_stats)
    failed_counts = dict(
        measured("failed_counts", db.failed_file_counts)  # type: ignore[arg-type]
    )
    limit = max(1, int(page_size))
    failed_rows = [
        dict(row)
        for row in measured(
            "failed_rows",
            lambda: db.failed_files(
                limit=limit,
                offset=max(0, int(page_offsets.get("blocking", 0))),
            ),
        )  # type: ignore[arg-type]
    ]
    excluded_rows = [
        dict(row)
        for row in measured(
            "excluded_rows",
            lambda: db.excluded_files(
                limit=limit,
                offset=max(0, int(page_offsets.get("excluded", 0))),
            ),
        )  # type: ignore[arg-type]
    ]
    metadata_rows = [
        dict(row)
        for row in measured(
            "metadata_rows",
            lambda: db.metadata_only_files(
                limit=limit,
                offset=max(0, int(page_offsets.get("metadata", 0))),
            ),
        )  # type: ignore[arg-type]
    ]
    history = list(
        measured("history", db.search_history)  # type: ignore[arg-type]
    )
    return {
        "roots": roots,
        # index_readiness already contains the aggregate stats. Avoid the
        # duplicate full-table scan previously performed by db.stats().
        "stats": readiness,
        "readiness": readiness,
        "root_stats_by_id": root_stats_by_id,
        "failed_counts": failed_counts,
        "failed_rows": failed_rows,
        "excluded_rows": excluded_rows,
        "metadata_rows": metadata_rows,
        "history": history,
        "timings_ms": timings,
        "total_ms": sum(timings.values()),
    }


class DatabaseSnapshotWorker(QObject):
    quick_ready = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        db_path: Path,
        *,
        page_size: int,
        page_offsets: dict[str, int],
    ) -> None:
        super().__init__()
        self.db = DatabaseManager(Path(db_path))
        self.page_size = max(1, int(page_size))
        self.page_offsets = dict(page_offsets)
        self._cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            quick_started = time.perf_counter()
            roots = [dict(row) for row in self.db.list_roots()]
            quick = self.db.quick_index_readiness()
            self.quick_ready.emit(
                {
                    "roots": roots,
                    "readiness": quick,
                    "elapsed_ms": int(
                        (time.perf_counter() - quick_started) * 1000
                    ),
                }
            )
            if self._cancelled:
                self.cancelled.emit()
                return
            snapshot = collect_database_snapshot(
                self.db,
                page_size=self.page_size,
                page_offsets=self.page_offsets,
            )
            if self._cancelled:
                self.cancelled.emit()
                return
            logger.info(
                "Background database snapshot completed: total_ms=%s timings=%s",
                snapshot.get("total_ms"),
                snapshot.get("timings_ms"),
            )
            self.finished.emit(snapshot)
        except sqlite3.OperationalError as exc:
            if self._cancelled and "interrupt" in str(exc).lower():
                self.cancelled.emit()
                return
            logger.exception("Background database snapshot failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)
        except Exception as exc:
            logger.exception("Background database snapshot failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def cancel(self) -> None:
        self._cancelled = True
        self.db.interrupt_active_connections()


class IndexReadinessWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self.db = DatabaseManager(Path(db_path))
        self.cancelled = False

    @Slot()
    def run(self) -> None:
        try:
            started = time.perf_counter()
            readiness = self.db.index_readiness()
            if self.cancelled:
                return
            self.finished.emit(
                {
                    "readiness": readiness,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                }
            )
        except sqlite3.OperationalError as exc:
            if self.cancelled and "interrupt" in str(exc).lower():
                return
            logger.debug("Background readiness refresh failed: %s", exc)
            self.failed.emit(str(exc) or exc.__class__.__name__)
        except Exception as exc:
            logger.exception("Background readiness refresh failed")
            self.failed.emit(str(exc) or exc.__class__.__name__)

    def cancel(self) -> None:
        self.cancelled = True
        self.db.interrupt_active_connections()
