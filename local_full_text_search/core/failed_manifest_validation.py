from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from local_full_text_search.core.database import (
    MANUALLY_EXCLUDABLE_STATUSES,
    DatabaseManager,
)


REQUIRED_COLUMNS = {
    "路径",
    "扩展名",
    "状态",
    "错误码",
    "原因",
    "解析器",
    "时间",
}


def replay_failed_manifest(
    manifest_path: Path,
    base: Path,
) -> dict[str, object]:
    manifest_path = manifest_path.resolve()
    base.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing_columns = sorted(REQUIRED_COLUMNS - columns)
        if missing_columns:
            raise ValueError(
                "失败清单缺少列：" + ", ".join(missing_columns)
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("失败清单为空")

    replay_root = base / "replay-scope"
    replay_root.mkdir()
    database = DatabaseManager(base / "replay.db")
    database.initialize()
    root_id = database.add_root(replay_root)
    file_ids: list[int] = []
    status_by_file: dict[int, str] = {}
    original_paths = [Path(str(row.get("路径") or "")) for row in rows]
    for index, row in enumerate(rows):
        extension = str(row.get("扩展名") or "").strip()
        if extension and not extension.startswith("."):
            extension = "." + extension
        source = replay_root / f"manifest-{index:04d}{extension}"
        source.write_bytes(f"manifest-row-{index}".encode("ascii"))
        file_id, _ = database.upsert_file_metadata(root_id, source)
        status = str(row.get("状态") or "").strip()
        database.set_file_error_status(
            file_id,
            status,
            str(row.get("错误码") or "").strip(),
            str(row.get("原因") or "").strip(),
            parser_name=str(row.get("解析器") or "").strip() or None,
        )
        file_ids.append(file_id)
        status_by_file[file_id] = status
    database.update_root_scan_time(root_id, "ready")

    before = database.index_readiness()
    blocking_rows = database.failed_files(limit=len(rows) + 1)
    blocking_ids = [int(row["id"]) for row in blocking_rows]
    unexpected_statuses = sorted(
        {
            status_by_file[file_id]
            for file_id in blocking_ids
            if status_by_file[file_id]
            not in MANUALLY_EXCLUDABLE_STATUSES
        }
    )
    if unexpected_statuses:
        raise ValueError(
            "清单包含不可人工排除的阻断状态："
            + ", ".join(unexpected_statuses)
        )
    database.exclude_files_from_index(
        blocking_ids,
        reason="现场失败清单语义重放",
        operation_source="validation_tool",
    )
    after = database.index_readiness()
    with database.connect() as connection:
        status_after = {
            int(row["id"]): str(row["parse_status"])
            for row in connection.execute(
                "SELECT id, parse_status FROM files ORDER BY id"
            )
        }
    statuses = Counter(str(row.get("状态") or "") for row in rows)
    error_codes = Counter(
        str(row.get("错误码") or "") for row in rows
    )
    source_available = sum(path.is_file() for path in original_paths)
    preserved = all(
        status_after.get(file_id) == status
        for file_id, status in status_by_file.items()
    )
    passed = bool(
        int(before["metadata_only_complete_files"])
        == statuses.get("metadata_only", 0)
        and int(before["blocking_files"]) == len(blocking_ids)
        and int(after["manual_excluded_files"]) == len(blocking_ids)
        and int(after["blocking_files"]) == 0
        and bool(after["ready"])
        and preserved
    )
    return {
        "passed": passed,
        "manifest": str(manifest_path),
        "input_rows": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "error_code_counts": dict(sorted(error_codes.items())),
        "nonblocking_metadata_only": int(
            before["metadata_only_complete_files"]
        ),
        "blocking_before_exclusion": int(before["blocking_files"]),
        "manual_excluded_after": int(after["manual_excluded_files"]),
        "blocking_after_exclusion": int(after["blocking_files"]),
        "eligible_after": int(after["eligible_files"]),
        "complete_after": int(after["complete_files"]),
        "parse_statuses_preserved": preserved,
        "source_files_available": source_available,
        "source_files_unavailable": len(rows) - source_available,
        "source_binary_verification_complete": source_available == len(rows),
        "source_binary_verification_note": (
            "所有清单原文件均可访问"
            if source_available == len(rows)
            else "清单状态语义已重放，但不可访问的原文件尚未完成文件头、CRC、OLE 和 Office 打开复核"
        ),
    }
