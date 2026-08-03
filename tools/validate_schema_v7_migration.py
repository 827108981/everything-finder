from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_full_text_search.core.database import (
    SCHEMA_VERSION,
    DatabaseManager,
)
from local_full_text_search.core.search_engine import SearchEngine
from local_full_text_search.models.search_query import SearchQuery


GOLDEN_QUERIES = (
    "拔掉 3 个传感器",
    "Dispersion",
)
GOLDEN_MINIMUM_HITS = {
    "拔掉 3 个传感器": 1,
    "Dispersion": 1,
}


def _backup_database(source: Path, destination: Path) -> None:
    source_connection = sqlite3.connect(
        f"file:{source.as_posix()}?mode=ro",
        uri=True,
        timeout=60,
    )
    destination_connection = sqlite3.connect(
        destination,
        timeout=60,
    )
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()


def _snapshot(database: DatabaseManager) -> dict[str, object]:
    digest = hashlib.sha256()
    with database.connect() as connection:
        counts = {
            table: int(
                connection.execute(
                    f"SELECT COUNT(*) FROM {table}"
                ).fetchone()[0]
            )
            for table in (
                "roots",
                "files",
                "documents",
                "content_blocks",
                "content_fts",
                "search_history",
                "parse_tasks",
                "parse_task_attempts",
            )
        }
        for row in connection.execute(
            """
            SELECT file_id, block_index, block_type, location_text,
                   page_number, raw_text, normalized_text, source_type,
                   extra_json
            FROM content_blocks
            ORDER BY file_id, block_index, id
            """
        ):
            digest.update(
                json.dumps(
                    list(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            )
            digest.update(b"\n")
        user_version = int(
            connection.execute("PRAGMA user_version").fetchone()[0]
        )
    search = SearchEngine(database)
    hits = {
        query: search.search(
            SearchQuery(text=query, mode="exact")
        ).total_confirmed
        for query in GOLDEN_QUERIES
    }
    return {
        "user_version": user_version,
        "counts": counts,
        "content_digest": digest.hexdigest(),
        "golden_hits": hits,
    }


def validate(source: Path, output_dir: Path) -> dict[str, object]:
    source = source.resolve()
    output_dir = output_dir.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    output_dir.mkdir(parents=True, exist_ok=True)
    copy_path = output_dir / "user_database_copy.db"
    if copy_path.exists():
        raise FileExistsError(
            f"validation copy already exists: {copy_path}"
        )
    _backup_database(source, copy_path)
    copied = DatabaseManager(copy_path)
    before = _snapshot(copied)
    started = time.perf_counter()
    copied.initialize()
    migration_seconds = time.perf_counter() - started
    after = _snapshot(copied)
    integrity = copied.integrity_report()
    preserved = {
        "counts": before["counts"] == after["counts"],
        "content_digest": (
            before["content_digest"] == after["content_digest"]
        ),
        "golden_hits": (
            before["golden_hits"] == after["golden_hits"]
        ),
        "golden_queries_nonempty": all(
            int(after["golden_hits"].get(query, 0)) >= minimum
            for query, minimum in GOLDEN_MINIMUM_HITS.items()
        ),
    }
    passed = bool(
        int(after["user_version"]) == SCHEMA_VERSION
        and all(preserved.values())
        and integrity["integrity"] == ["ok"]
        and not integrity["foreign_key_errors"]
    )
    return {
        "passed": passed,
        "source": str(source),
        "copy": str(copy_path),
        "migration_seconds": round(migration_seconds, 3),
        "before": before,
        "after": after,
        "preserved": preserved,
        "integrity": integrity,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    try:
        report = validate(args.source, args.output_dir)
    except Exception as exc:
        report = {"passed": False, "error": str(exc)}
    output = args.output_dir / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
