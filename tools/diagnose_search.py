"""Print one search/index diagnostic timeline from date-partitioned JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_full_text_search.config.constants import LOG_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="还原一次搜索或索引的结构化事件时间线")
    parser.add_argument("--date", default=None, help="日志日期，例如 2026-09-01")
    parser.add_argument("--request-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--query-hash", default=None)
    parser.add_argument("--event", default=None)
    args = parser.parse_args()
    directories = [LOG_DIR / args.date] if args.date else sorted(LOG_DIR.glob("????-??-??"))
    events: list[dict[str, object]] = []
    for directory in directories:
        path = directory / "events.jsonl"
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.request_id and event.get("request_id") != args.request_id:
                continue
            if args.run_id and event.get("run_id") != args.run_id:
                continue
            if args.query_hash and event.get("query_hash") != args.query_hash:
                continue
            if args.event and event.get("event") != args.event:
                continue
            events.append(event)
    events.sort(key=lambda event: str(event.get("timestamp") or ""))
    if not events:
        print("未找到匹配事件", file=sys.stderr)
        return 1
    for event in events:
        timestamp = str(event.get("timestamp") or "")
        level = str(event.get("level") or "INFO")
        name = str(event.get("event") or "")
        details = []
        for key in (
            "stage", "state", "scan_id", "run_id", "run_ids", "root_id",
            "root_path", "generation", "candidate_generation", "database_path",
            "candidate_path", "pending_files", "pending_before", "pending_after",
            "synced_blocks", "attempt", "blocking_files", "unready_roots",
            "readiness", "elapsed_ms", "checked_candidates",
            "total_candidates", "confirmed_files", "error_type", "error_message",
        ):
            if key in event:
                details.append(f"{key}={event[key]}")
        print(f"{timestamp} {level:<7} {name} " + " ".join(details))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
